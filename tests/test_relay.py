"""/api/llm/relay 无状态中转测试：本地假上游（http.server 线程），覆盖
正常透传（openai/anthropic）、上游空内容 -> 502、上游错误透传、SSRF 拦截。"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi.testclient import TestClient

from src.server.app import app


class _Upstream(BaseHTTPRequestHandler):
    """按路径返回固定的模拟上游响应。"""

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(n)
        if self.path.startswith("/bad"):      # 上游报错（须先于正常分支判断）
            body = {"error": {"message": "invalid api key"}}
            code = 401
        elif self.path.startswith("/empty"):  # 空内容（base 带 /empty 前缀）
            body = {"content": [], "stop_reason": "max_tokens"}
            code = 200
        elif self.path.endswith("/chat/completions"):
            body = {"choices": [{"message": {"content": "pong"},
                                 "finish_reason": "stop"}]}
            code = 200
        else:  # /v1/messages 正常
            body = {"content": [{"type": "text", "text": "pong"}],
                    "stop_reason": "end_turn"}
            code = 200
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


@pytest.fixture(scope="module")
def upstream():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def _payload(base, proto="openai", path_suffix=""):
    return {"proto": proto, "base": base + path_suffix, "key": "sk-test",
            "model": "m", "messages": [{"role": "user", "content": "ping"}]}


def test_relay_openai_ok(client, upstream):
    r = client.post("/api/llm/relay", json=_payload(upstream))
    assert r.status_code == 200 and r.json()["text"] == "pong"


def test_relay_anthropic_ok(client, upstream):
    r = client.post("/api/llm/relay", json=_payload(upstream, "anthropic"))
    assert r.status_code == 200 and r.json()["text"] == "pong"


def test_relay_empty_content_502(client, upstream):
    r = client.post("/api/llm/relay",
                    json=_payload(upstream, "anthropic", "/empty"))
    assert r.status_code == 502
    assert "空内容" in r.json()["error"]


def test_relay_upstream_error_passthrough(client, upstream):
    r = client.post("/api/llm/relay", json=_payload(upstream, path_suffix="/bad"))
    assert r.status_code == 502
    assert "401" in r.json()["error"] and "invalid api key" in r.json()["error"]


def test_relay_ssrf_blocked(client):
    r = client.post("/api/llm/relay",
                    json=_payload("http://192.168.1.1:8080"))
    assert r.status_code == 400 and "仅允许" in r.json()["error"]


def test_relay_bad_proto(client, upstream):
    r = client.post("/api/llm/relay", json=_payload(upstream, "gemini"))
    assert r.status_code == 400


def test_relay_missing_key(client, upstream):
    p = _payload(upstream)
    p["key"] = ""
    r = client.post("/api/llm/relay", json=p)
    assert r.status_code == 400
