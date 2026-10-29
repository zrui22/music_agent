"""声乐评测 Agent Web 服务（FastAPI）

启动：
    PYTHONPATH=. .venv/bin/uvicorn src.server.app:app --host 0.0.0.0 --port 9527

API：
    GET  /api/health     服务状态（模型/设备/曲库规模）
    GET  /api/library    参考曲库列表，?q= 过滤（本地曲库）
    GET  /api/library/online  在线搜索候选曲目（?q=），in_lib 标记是否已收录
    POST /api/library/add     收录在线候选到本地曲库（下载音频+歌词+F0，慢操作）
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
import shutil
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
    weights = (ROOT / "models" / "singing_scorer_v1.pt").exists() \
        or (ROOT / "runs" / "r3" / "best.pt").exists()
    stats = (ROOT / "models" / "feat_stats.json").exists()
    return {"status": "ok", "version": app.version,
            "model_loaded": _scorer is not None,
            "model_weights": weights and stats,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "library_size": n_lib,
            "time": time.strftime("%Y-%m-%d %H:%M:%S")}


@app.get("/api/library")
def library(q: str = "", limit: int = 20):
    from src.agent import reference
    items = reference.library(load_f0=False)   # 列表只需歌名/歌手，免 F0 加载
    if q:
        ql = q.lower()
        items = [it for it in items
                 if ql in (it.get("song") or "").lower()
                 or ql in (it.get("artists") or "").lower()]
    return {"total": len(items),
            "items": [{"song": it["song"], "artists": it["artists"],
                       "ref_id": it["ref_id"]} for it in items[:limit]]}


# ---------- 在线曲库搜索 / 收录 ----------

_add_lock = threading.Lock()
_adding: set = set()   # 正在收录中的网易云 song_id，防并发重复下载


@app.get("/api/library/online")
def library_online(q: str = "", limit: int = 8):
    """在线搜索候选曲目，in_lib 标记是否已收录本地曲库。"""
    q = q.strip()
    if not q:
        return {"items": []}
    from src.pitchref.netease import search_song, REF_DIR
    try:
        cands = search_song(q, limit=limit)
    except Exception as e:
        return JSONResponse({"error": f"在线搜索失败: {type(e).__name__}: {e}"}, 502)
    return {"items": [{"id": c["id"], "song": c["name"], "artists": c["artists"],
                       "duration_s": round(c.get("duration_s", 0)),
                       "in_lib": (REF_DIR / str(c["id"]) / "ref_f0.npz").exists()}
                      for c in cands]}


@app.post("/api/library/add")
async def library_add(req: Request):
    """收录在线候选到本地曲库：下载音频+歌词并提取 F0（慢操作，约 10-60s）。"""
    from src.agent.reference import fetch_by_id
    try:
        item = await req.json()
    except Exception:
        return JSONResponse({"error": "请求体不是 JSON"}, 400)
    try:
        song_id = int(item.get("id") or 0)
    except (TypeError, ValueError):
        song_id = 0
    if not song_id:
        return JSONResponse({"error": "缺少歌曲 id"}, 400)
    with _add_lock:
        if song_id in _adding:
            return JSONResponse({"error": "该曲目正在收录中，请稍候"}, 409)
        _adding.add(song_id)
    try:
        ref = await asyncio.to_thread(
            fetch_by_id, song_id, item.get("query") or item.get("name", ""),
            item.get("name", ""), item.get("artists", ""),
            float(item.get("duration_s") or 0))
    except Exception as e:
        return JSONResponse({"error": f"收录失败: {type(e).__name__}: {e}"}, 500)
    finally:
        with _add_lock:
            _adding.discard(song_id)
    if ref is None:
        return JSONResponse({"error": "音频下载失败（VIP/无版权歌曲暂不支持）"}, 502)
    return {"ok": True, "ref_id": ref["ref_id"], "song": ref["song"],
            "artists": ref["artists"]}


# ---------- 评测 ----------

MAX_UPLOAD_BYTES = 50 * 1024 * 1024   # 50MB


@app.post("/api/diagnose")
async def api_diagnose(file: UploadFile = File(...), song: str = Form(""),
                       auto_match: bool = Form(True), windows: int = Form(5)):
    from src.agent.diagnose import diagnose
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        return JSONResponse({"error": "文件超过 50MB 上限，请裁剪后再上传"}, 413)
    suffix = Path(file.filename or "audio").suffix or ".bin"
    tmp = Path(tempfile.mkdtemp(prefix="api_"))
    try:
        src = tmp / f"input{suffix}"
        src.write_bytes(data)
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
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


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
        payload = {"model": model or "gpt-4o-mini", "temperature": 0.7, "max_tokens": 4096,
                   "messages": ([{"role": "system", "content": system}] if system else []) + messages}
    else:
        url = base + "/v1/messages"
        headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
        payload = {"model": model or "claude-sonnet-4-5", "max_tokens": 4096,
                   "system": system, "messages": messages}
    # 默认关闭思考模式（qwen3 等混合模型默认开思考，慢且容易烧光 max_tokens 返回空）。
    # 标准 Anthropic/OpenAI 默认本就不开思考，故只对已知默认开思考的阿里云百炼/MaaS 注入；
    # 若上游 400 报 thinking 参数不认识，自动去掉该参数重试一次。
    is_bailian = ("aliyuncs.com" in low) or ("dashscope" in low)
    if is_bailian:
        if proto == "openai":
            payload["enable_thinking"] = False
        else:
            payload["thinking"] = {"type": "disabled"}
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=15.0)) as cli:
            r = await cli.post(url, json=payload, headers=headers)
            if r.status_code == 400 and is_bailian and "thinking" in r.text.lower():
                payload.pop("enable_thinking", None)
                payload.pop("thinking", None)
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
        ch = (d.get("choices") or [{}])[0]
        text = (ch.get("message") or {}).get("content", "")
        finish = str(ch.get("finish_reason") or "")
    else:
        text = "".join(b.get("text", "") for b in d.get("content", [])
                       if isinstance(b, dict) and b.get("type") == "text")
        finish = str(d.get("stop_reason") or "")
    # 后台直接判定空内容为错误（不再让前端收到空 text 渲染空气泡）
    if not str(text).strip():
        tip = ("输出达到长度上限被截断" if finish in ("max_tokens", "length")
               else "模型名可能不正确或无权限")
        return JSONResponse(
            {"error": f"上游响应空内容（HTTP {r.status_code}，stop={finish or '?'}）：{tip}"}, 502)
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
