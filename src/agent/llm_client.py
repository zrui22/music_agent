"""LLM 客户端：双后端自动回落

后端优先级：
  1. openai 协议（CODEX_BASE_URL/CODEX_API_KEY/CODEX_MODEL，或 ~/.kimi-code/secrets/codex-backend.env）
  2. anthropic 协议（ANTHROPIC_BASE_URL/ANTHROPIC_API_KEY/ANTHROPIC_MODEL，
     默认回落到 ~/.kimi-code/config.toml 的 maas-anthropic provider）
"""
import json
import os
import re
import urllib.request

CODEX_ENV_FILE = os.path.expanduser("~/.kimi-code/secrets/codex-backend.env")
KIMI_CFG = os.path.expanduser("~/.kimi-code/config.toml")


def _load_codex() -> dict:
    cfg = {}
    if os.path.exists(CODEX_ENV_FILE):
        for line in open(CODEX_ENV_FILE, encoding="utf-8"):
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                cfg[k.strip()] = v.strip().strip('"').strip("'")
    for k in ("CODEX_BASE_URL", "CODEX_API_KEY", "CODEX_MODEL"):
        if os.environ.get(k):
            cfg[k] = os.environ[k]
    return cfg


def _load_anthropic() -> dict:
    cfg = {"ANTHROPIC_BASE_URL": os.environ.get("ANTHROPIC_BASE_URL"),
           "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY"),
           "ANTHROPIC_MODEL": os.environ.get("ANTHROPIC_MODEL", "kimi-k3")}
    if not (cfg["ANTHROPIC_BASE_URL"] and cfg["ANTHROPIC_API_KEY"]):
        # 从 kimi-code config.toml 提取 maas-anthropic provider（轻量正则解析，避免引 toml 依赖）
        try:
            text = open(KIMI_CFG, encoding="utf-8").read()
            m = re.search(r'\[providers\.maas-anthropic\](.*?)(?:\n\[|\Z)', text, re.S)
            block = m.group(1)
            cfg["ANTHROPIC_BASE_URL"] = re.search(r'base_url\s*=\s*"([^"]+)"', block).group(1)
            cfg["ANTHROPIC_API_KEY"] = re.search(r'api_key\s*=\s*"([^"]+)"', block).group(1)
            mm = re.search(r'\[models\."([^"]+)"\]\s*\n\s*provider\s*=\s*"maas-anthropic"', text)
            if mm and not os.environ.get("ANTHROPIC_MODEL"):
                cfg["ANTHROPIC_MODEL"] = mm.group(1)
        except Exception:
            pass
    return cfg


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
    payload = {"model": model or cfg.get("CODEX_MODEL", "gpt-5.5"),
               "messages": [{"role": "system", "content": system},
                            {"role": "user", "content": user}],
               "temperature": temperature, "max_tokens": max_tokens}
    req = urllib.request.Request(
        f"{cfg['CODEX_BASE_URL'].rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {cfg['CODEX_API_KEY']}",
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode())
    return data["choices"][0]["message"]["content"]


def _call_anthropic(cfg, system, user, model, max_tokens, temperature, timeout):
    payload = {"model": model or cfg["ANTHROPIC_MODEL"], "max_tokens": max_tokens,
               "system": system,
               "thinking": {"type": "disabled"},  # 分类任务关思考链，提速数倍
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
    """调 LLM 并解析 JSON。openai 后端 401/失败时自动回落 anthropic 后端。"""
    codex = _load_codex()
    anth = _load_anthropic()
    backends = [("openai", codex, _call_openai), ("anthropic", anth, _call_anthropic)]
    if _PREFERRED_BACKEND:
        backends.sort(key=lambda b: b[0] != _PREFERRED_BACKEND[0])
    errors = []
    for backend, cfg, caller in backends:
        key = cfg.get("CODEX_API_KEY") if backend == "openai" else cfg.get("ANTHROPIC_API_KEY")
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
