"""LLM 客户端：OpenAI / Anthropic 双协议，环境变量或项目根 .env 配置。

配置方式（任选其一，环境变量优先）：
  export OPENAI_BASE_URL=... OPENAI_API_KEY=... OPENAI_MODEL=...       # OpenAI 协议
  export ANTHROPIC_BASE_URL=... ANTHROPIC_API_KEY=... ANTHROPIC_MODEL=...  # Anthropic 协议
或在项目根目录 .env 文件中写同名变量（KEY=VALUE，每行一个）。

注意：Web 前端的「陪教」对话默认由浏览器直连 LLM（或经 /api/llm/relay 中转），
本模块只服务于命令行模式 `python -m src.agent.chat`。
"""
import json
import os
import re
import urllib.request
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


def _env_file() -> dict:
    cfg = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                cfg[k.strip()] = v.strip().strip('"').strip("'")
    return {**cfg, **{k: v for k, v in os.environ.items() if v}}


def _load_openai() -> dict:
    env = _env_file()
    return {"OPENAI_BASE_URL": env.get("OPENAI_BASE_URL"),
            "OPENAI_API_KEY": env.get("OPENAI_API_KEY"),
            "OPENAI_MODEL": env.get("OPENAI_MODEL")}


def _load_anthropic() -> dict:
    env = _env_file()
    return {"ANTHROPIC_BASE_URL": env.get("ANTHROPIC_BASE_URL"),
            "ANTHROPIC_API_KEY": env.get("ANTHROPIC_API_KEY"),
            "ANTHROPIC_MODEL": env.get("ANTHROPIC_MODEL", "claude-sonnet-4-5")}


def _repair_truncated_json(text: str) -> str:
    if text.count('"') % 2 == 1:
        text += '"'
    return text + "]" * max(text.count("[") - text.count("]"), 0) \
                + "}" * max(text.count("{") - text.count("}"), 0)


def extract_json(text: str):
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    body = (m.group(1) if m else text).strip()
    for candidate in (body, _repair_truncated_json(body)):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    idx = min([i for i in (body.find("{"), body.find("[")) if i >= 0], default=-1)
    if idx >= 0:
        try:
            return json.loads(_repair_truncated_json(body[idx:]))
        except json.JSONDecodeError:
            pass
    # JSONL 风格：逐行 JSON 对象 -> 收拢为数组
    rows = []
    for line in body.splitlines():
        line = line.strip().rstrip(",")
        line = re.sub(r"^\[\d+\]\s*", "", line)  # 剥掉模型回显的输入序号 "[0] "
        if not line.startswith(("{", "[")):
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if rows:
        return rows
    raise ValueError(f"无法解析JSON: {text[:200]}")


def _call_openai(cfg, system, user, model, max_tokens, temperature, timeout):
    payload = {"model": model or cfg.get("OPENAI_MODEL", "gpt-4o-mini"),
               "messages": [{"role": "system", "content": system},
                            {"role": "user", "content": user}],
               "temperature": temperature, "max_tokens": max_tokens}
    req = urllib.request.Request(
        f"{cfg['OPENAI_BASE_URL'].rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {cfg['OPENAI_API_KEY']}",
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode())
    return data["choices"][0]["message"]["content"]


def _call_anthropic(cfg, system, user, model, max_tokens, temperature, timeout):
    payload = {"model": model or cfg["ANTHROPIC_MODEL"], "max_tokens": max_tokens,
               "system": system,
               "messages": [{"role": "user", "content": user}]}
    req = urllib.request.Request(
        f"{cfg['ANTHROPIC_BASE_URL'].rstrip('/')}/v1/messages",
        data=json.dumps(payload).encode(),
        headers={"x-api-key": cfg["ANTHROPIC_API_KEY"],
                 "anthropic-version": "2023-06-01",
                 "content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode())
    parts = [b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"]
    return "".join(parts)


_PREFERRED_BACKEND = []  # 首个可用后端名缓存，避免每次先试失效后端


def chat_json(system: str, user: str, model: str = None, max_tokens: int = 4096,
              temperature: float = 0.1, timeout: int = 180):
    """调 LLM 并解析 JSON。openai 后端失败时自动回落 anthropic 后端。"""
    openai, anth = _load_openai(), _load_anthropic()
    backends = [("openai", openai, _call_openai), ("anthropic", anth, _call_anthropic)]
    if _PREFERRED_BACKEND:
        backends.sort(key=lambda b: b[0] != _PREFERRED_BACKEND[0])
    errors = []
    for backend, cfg, caller in backends:
        key = cfg.get("OPENAI_API_KEY") if backend == "openai" else cfg.get("ANTHROPIC_API_KEY")
        if not key:
            continue
        try:
            content = caller(cfg, system, user, model, max_tokens, temperature, timeout)
            result = extract_json(content)
            if not _PREFERRED_BACKEND:
                _PREFERRED_BACKEND.append(backend)
            return result
        except Exception as e:
            errors.append(f"{backend}: {type(e).__name__} {str(e)[:100]}")
            continue
    raise RuntimeError("所有LLM后端失败 | " + " | ".join(errors))


def chat(messages: list, system: str = "", model: str = None, max_tokens: int = 2048,
         temperature: float = 0.7, timeout: int = 120) -> str:
    """纯文本多轮对话。messages=[{"role","content"},...]，返回文本。无后端抛 RuntimeError。"""
    errors = []
    for caller, cfg in (
        (_chat_openai, _load_openai()),
        (_chat_anthropic, _load_anthropic()),
    ):
        try:
            return caller(cfg, messages, system, model, max_tokens, temperature, timeout)
        except Exception as e:
            errors.append(f"{type(e).__name__} {str(e)[:100]}")
            continue
    raise RuntimeError("所有LLM后端失败 | " + " | ".join(errors))


def _chat_openai(cfg, messages, system, model, max_tokens, temperature, timeout):
    if not cfg.get("OPENAI_API_KEY"):
        raise RuntimeError("no openai backend")
    msgs = ([{"role": "system", "content": system}] if system else []) + list(messages)
    payload = {"model": model or cfg.get("OPENAI_MODEL", "gpt-4o-mini"),
               "messages": msgs, "temperature": temperature, "max_tokens": max_tokens}
    req = urllib.request.Request(
        f"{cfg['OPENAI_BASE_URL'].rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {cfg['OPENAI_API_KEY']}",
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode())
    return data["choices"][0]["message"]["content"]


def _chat_anthropic(cfg, messages, system, model, max_tokens, temperature, timeout):
    if not cfg.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("no anthropic backend")
    payload = {"model": model or cfg["ANTHROPIC_MODEL"], "max_tokens": max_tokens,
               "system": system or "你是声乐教练。",
               "messages": list(messages)}
    req = urllib.request.Request(
        f"{cfg['ANTHROPIC_BASE_URL'].rstrip('/')}/v1/messages",
        data=json.dumps(payload).encode(),
        headers={"x-api-key": cfg["ANTHROPIC_API_KEY"],
                 "anthropic-version": "2023-06-01",
                 "content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode())
    parts = [b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"]
    return "".join(parts)
