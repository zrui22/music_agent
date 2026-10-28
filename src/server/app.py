"""声乐评测 Agent Web 服务（FastAPI）

启动：
    PYTHONPATH=. .venv/bin/uvicorn src.server.app:app --host 0.0.0.0 --port 8900

API：
    GET  /api/health     服务状态（模型/设备/曲库规模）
    GET  /api/library    参考曲库列表，?q= 过滤（歌名补全）
    POST /api/diagnose   multipart: file=音频, song?=歌名, auto_match?=bool -> 完整评测 JSON
    GET  /               前端单页（web/index.html）
LLM 评语/对话不在后端：前端持有用户自填的 API Key，浏览器直连 LLM 服务商，
本服务不接触、不转发、不存储任何 LLM 凭据。
交互式 API 文档：/docs（Swagger UI）
"""
import asyncio
import tempfile
import threading
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "web"

app = FastAPI(title="Singing Coach Agent", version="1.0.0",
              description="智能声乐评测 Agent：哼唱匹配 / 六维评分（LLM 由前端直连，不经后端）")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

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


# ---------- 前端 ----------

@app.get("/", include_in_schema=False)
def index():
    return FileResponse(WEB / "index.html")
