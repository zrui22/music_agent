"""声乐评测 Agent Web 服务（FastAPI）

启动：
    PYTHONPATH=. .venv/bin/uvicorn src.server.app:app --host 0.0.0.0 --port 8900

API：
    GET  /api/health     服务状态（模型/设备/LLM/曲库规模）
    GET  /api/library    参考曲库列表，?q= 过滤（歌名补全）
    POST /api/diagnose   multipart: file=音频, song?=歌名, auto_match?=bool -> 完整评测 JSON
    POST /api/chat       {messages:[{role,content}], song?, last_diag?} -> {reply}
    POST /api/report     {diag} -> LLM 结构化评语报告
    GET  /               前端单页（web/index.html）
交互式 API 文档：/docs（Swagger UI）
"""
import asyncio
import json
import tempfile
import threading
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "web"

PERSONA = """你是一位经验丰富、鼓励为主的声乐教练「小嗓」。用户是业余唱歌爱好者。
1. 回答声乐技巧、练声方法、选歌建议等问题，具体可执行，不空谈；
2. 用户发来 [评测结果] JSON 时，翻译成通俗中文点评：先一句话总评，再逐维说问题
   （低于 3.5 的重点说），最后给 2-3 条本周练习建议；warnings 要如实转达；
3. 语气亲切专业，像真人教练，用比喻，不暴露 JSON 字段名。分数均为 0-5 分制。"""

app = FastAPI(title="Singing Coach Agent", version="1.0.0",
              description="智能声乐评测 Agent：哼唱匹配 / 六维评分 / LLM 教练评语")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

_scorer = None            # 评分模型单例（懒加载，启动不阻塞）
_scorer_lock = threading.Lock()
_diag_sem = asyncio.Semaphore(2)   # GPU 推理并发上限
_llm_ok = None


def get_scorer():
    global _scorer
    if _scorer is None:
        with _scorer_lock:
            if _scorer is None:
                from src.agent.scorer import Scorer
                _scorer = Scorer()
    return _scorer


def llm_available() -> bool:
    global _llm_ok
    if _llm_ok is None:
        from src.agent import llm_client
        _llm_ok = bool(llm_client._load_codex().get("CODEX_API_KEY")
                       or llm_client._load_anthropic().get("ANTHROPIC_API_KEY"))
    return _llm_ok


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
            "llm": llm_available(), "library_size": n_lib,
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


# ---------- 对话 / 评语 ----------

class ChatReq(BaseModel):
    messages: list
    song: str | None = None
    last_diag: dict | None = None


@app.post("/api/chat")
def api_chat(req: ChatReq):
    if not llm_available():
        return JSONResponse({"error": "未配置 LLM 后端"}, 503)
    from src.agent import llm_client
    system = PERSONA
    ctx = []
    if req.song:
        ctx.append(f"用户当前练习曲目：《{req.song}》")
    if req.last_diag:
        d = dict(req.last_diag)
        d.pop("segments", None)  # 控制上下文长度
        ctx.append("最近一次评测结果摘要：" + json.dumps(d, ensure_ascii=False)[:2000])
    if ctx:
        system += "\n\n[当前上下文]\n" + "\n".join(ctx)
    try:
        reply = llm_client.chat(req.messages[-12:], system=system)
    except Exception as e:
        return JSONResponse({"error": str(e)}, 502)
    return {"reply": reply}


class ReportReq(BaseModel):
    diag: dict
    model: str | None = None


@app.post("/api/report")
def api_report(req: ReportReq):
    if not llm_available():
        return JSONResponse({"error": "未配置 LLM 后端"}, 503)
    from src.agent.report import build_report
    try:
        return {"report": build_report(req.diag, model=req.model)}
    except Exception as e:
        return JSONResponse({"error": str(e)}, 502)


# ---------- 前端 ----------

@app.get("/", include_in_schema=False)
def index():
    return FileResponse(WEB / "index.html")
