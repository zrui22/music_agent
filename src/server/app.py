"""声乐评测 Agent Web 服务（FastAPI）

启动：
    PYTHONPATH=. .venv/bin/uvicorn src.server.app:app --host 0.0.0.0 --port 9527

API：
    GET  /api/health     服务状态（模型/设备/曲库规模）
    GET  /api/library    参考曲库列表，?q= 过滤（歌名补全）
    POST /api/diagnose   multipart: file=音频, song?=歌名, auto_match?=bool -> 完整评测 JSON
    POST /api/llm/relay  可选的 LLM 无状态中转（仅当浏览器直连被 CORS 拦截时使用）
    POST /api/tts        语音合成（edge-tts 在线声线，免费无 Key；失败时前端自动回退浏览器 TTS）
    GET  /               前端单页（web/index.html）
LLM 评语/对话默认由前端持有用户自填的 API Key 直连服务商，本服务不存储任何 LLM 凭据；
/api/llm/relay 仅为绕过浏览器 CORS 的无状态转发——Key 随请求来、随响应走，不落盘、不记日志。
交互式 API 文档：/docs（Swagger UI）
"""
import asyncio
import collections
import os
import re
import tempfile
import threading
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "web"

app = FastAPI(title="Singing Coach Agent", version="1.1.0",
              description="智能声乐评测 Agent：歌声门控 / 哼唱匹配 / 六维评分（LLM 由前端直连，不经后端）")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])
# 前端静态资源（3D 形象模型、three.js 本地 vendor）
for sub in ("vendor", "assets"):
    (WEB / sub).mkdir(parents=True, exist_ok=True)
    app.mount(f"/{sub}", StaticFiles(directory=WEB / sub), name=sub)

_scorer = None            # 评分模型单例（懒加载，启动不阻塞）
_scorer_lock = threading.Lock()
_diag_sem = asyncio.Semaphore(2)   # GPU 推理并发上限


def get_scorer():
    global _scorer
    if _scorer is None:
        with _scorer_lock:
            if _scorer is None:
                from src.agent.scorer import Scorer
                _scorer = Scorer()
    return _scorer


# ---------- 状态 ----------

@app.get("/api/health")
def health():
    from src.pitchref.netease import REF_DIR
    n_lib = sum(1 for d in REF_DIR.iterdir()
                if (d / "ref_f0.npz").exists()) if REF_DIR.exists() else 0
    import torch
    return {"status": "ok", "version": app.version,
            "model_loaded": _scorer is not None,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "library_size": n_lib,
            "time": time.strftime("%Y-%m-%d %H:%M:%S")}


@app.get("/api/library")
def library(q: str = "", limit: int = 20):
    from src.agent import reference
    items = reference.library()
    if q:
        ql = q.lower()
        items = [it for it in items
                 if ql in (it.get("song") or "").lower()
                 or ql in (it.get("artists") or "").lower()]
    return {"total": len(items),
            "items": [{"song": it["song"], "artists": it["artists"],
                       "ref_id": it["ref_id"]} for it in items[:limit]]}


# ---------- 评测 ----------

@app.post("/api/diagnose")
async def api_diagnose(file: UploadFile = File(...), song: str = Form(""),
                       auto_match: bool = Form(True), windows: int = Form(5)):
    from src.agent.diagnose import diagnose
    suffix = Path(file.filename or "audio").suffix or ".bin"
    tmp = Path(tempfile.mkdtemp(prefix="api_"))
    src = tmp / f"input{suffix}"
    src.write_bytes(await file.read())
    async with _diag_sem:
        loop = asyncio.get_running_loop()
        try:
            diag = await loop.run_in_executor(
                None, lambda: diagnose(str(src), song=song or None,
                                       auto_match=auto_match,
                                       n_windows=windows, scorer=get_scorer()))
        except Exception as e:
            return JSONResponse({"error": f"{type(e).__name__}: {e}"}, 400)
    return diag


# ---------- LLM 无状态中转（仅用于绕过浏览器 CORS；Key 随请求转发，不存储、不记日志） ----------

@app.post("/api/llm/relay")
async def llm_relay(req: Request):
    import httpx
    try:
        body = await req.json()
    except Exception:
        return JSONResponse({"error": "请求体不是 JSON"}, 400)
    proto = str(body.get("proto") or "")
    base = str(body.get("base") or "").rstrip("/")
    key = str(body.get("key") or "")
    model = str(body.get("model") or "")
    messages = body.get("messages") or []
    system = str(body.get("system") or "")
    if proto not in ("openai", "anthropic"):
        return JSONResponse({"error": "proto 必须是 openai / anthropic"}, 400)
    if not base or not key:
        return JSONResponse({"error": "缺少 base / key"}, 400)
    if not isinstance(messages, list) or len(messages) > 40:
        return JSONResponse({"error": "messages 不合法"}, 400)
    # SSRF 防护：只允许 https://，或本机回环 http://（本地 vLLM/Ollama 场景）
    low = base.lower()
    if not (low.startswith("https://") or low.startswith("http://127.")
            or low.startswith("http://localhost")):
        return JSONResponse({"error": "base 仅允许 https://（或本机 http://127.x / localhost）"}, 400)
    if proto == "openai":
        url = base + "/chat/completions"
        headers = {"Authorization": "Bearer " + key}
        payload = {"model": model or "gpt-4o-mini", "temperature": 0.7, "max_tokens": 2048,
                   "messages": ([{"role": "system", "content": system}] if system else []) + messages}
    else:
        url = base + "/v1/messages"
        headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
        payload = {"model": model or "claude-sonnet-4-5", "max_tokens": 2048,
                   "system": system, "messages": messages}
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=15.0)) as cli:
            r = await cli.post(url, json=payload, headers=headers)
    except Exception as e:
        return JSONResponse({"error": f"中转请求失败: {type(e).__name__}: {e}"}, 502)
    try:
        d = r.json()
    except Exception:
        return JSONResponse({"error": f"上游响应非 JSON（HTTP {r.status_code}）"}, 502)
    if r.status_code >= 400:
        msg = ""
        if isinstance(d, dict):
            err = d.get("error") or {}
            msg = str(err.get("message") or d.get("message") or "")[:200]
        return JSONResponse({"error": f"LLM HTTP {r.status_code}：{msg}"}, 502)
    if proto == "openai":
        text = ((d.get("choices") or [{}])[0].get("message") or {}).get("content", "")
    else:
        text = "".join(b.get("text", "") for b in d.get("content", [])
                       if isinstance(b, dict) and b.get("type") == "text")
    return {"text": text}


# ---------- TTS（默认本地 CosyVoice2 克隆声线 sidecar；edge-tts 在线兜底；再失败前端回退浏览器 TTS） ----------

_TTS_CACHE = collections.OrderedDict()   # (voice, pitch, rate, text) -> (mime, bytes)，上限 32 条
_TTS_VOICE = re.compile(r"^zh-CN-[A-Za-z]+Neural$")
_TTS_SIDECAR = os.environ.get("TTS_SIDECAR_URL", "http://127.0.0.1:9555")


async def _sidecar_tts(text: str):
    """调用本地 CosyVoice2 sidecar；不可用/超时返回 None。"""
    try:
        import httpx
        async with httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=3.0)) as cli:
            r = await cli.post(f"{_TTS_SIDECAR}/synthesize", json={"text": text})
        if r.status_code == 200 and r.content:
            return bytes(r.content)
    except Exception:
        pass
    return None


@app.post("/api/tts")
async def api_tts(req: Request):
    try:
        body = await req.json()
    except Exception:
        return JSONResponse({"error": "请求体不是 JSON"}, 400)
    text = str(body.get("text") or "").strip()[:280]
    if not text:
        return JSONResponse({"error": "text 为空"}, 400)
    voice = str(body.get("voice") or "")
    if voice == "paimon":
        key = ("paimon", "", "", text)
        if key in _TTS_CACHE:
            _TTS_CACHE.move_to_end(key)
            mime, data = _TTS_CACHE[key]
            return Response(content=data, media_type=mime)
        data = await _sidecar_tts(text)
        if data:
            _TTS_CACHE[key] = ("audio/wav", data)
            if len(_TTS_CACHE) > 32:
                _TTS_CACHE.popitem(last=False)
            return Response(content=data, media_type="audio/wav")
        voice = "zh-CN-XiaoyiNeural"   # sidecar 不在线 → 落到在线兜底
    if not _TTS_VOICE.fullmatch(voice):
        voice = "zh-CN-XiaoyiNeural"
    pitch = str(body.get("pitch") or "")
    if not re.fullmatch(r"[+-]\d{1,3}Hz", pitch):
        pitch = "+15Hz"
    rate = str(body.get("rate") or "")
    if not re.fullmatch(r"[+-]\d{1,2}%", rate):
        rate = "+4%"
    key = (voice, pitch, rate, text)
    if key in _TTS_CACHE:
        _TTS_CACHE.move_to_end(key)
        mime, data = _TTS_CACHE[key]
        return Response(content=data, media_type=mime)
    try:
        import edge_tts
        comm = edge_tts.Communicate(text, voice=voice, rate=rate, pitch=pitch)
        buf = bytearray()
        async for chunk in comm.stream():
            if chunk.get("type") == "audio" and chunk.get("data"):
                buf.extend(chunk["data"])
    except Exception as e:
        return JSONResponse({"error": f"TTS 失败: {type(e).__name__}: {e}"}, 502)
    if not buf:
        return JSONResponse({"error": "TTS 无音频输出"}, 502)
    data = bytes(buf)
    _TTS_CACHE[key] = ("audio/mpeg", data)
    if len(_TTS_CACHE) > 32:
        _TTS_CACHE.popitem(last=False)
    return Response(content=data, media_type="audio/mpeg")


# ---------- 前端 ----------

@app.get("/", include_in_schema=False)
def index():
    return FileResponse(WEB / "index.html")
