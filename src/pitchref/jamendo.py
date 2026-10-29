"""Jamendo CC 授权音乐源（可选）：官方 API，全合规。

需免费 client_id（https://devportal.jamendo.com/ 注册即得），设置环境变量
JAMENDO_CLIENT_ID 后自动启用；未设置时搜索返回空、收录返回失败。
"""
import json
import os
import urllib.parse
import urllib.request
from pathlib import Path

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}

CLIENT_ID = os.environ.get("JAMENDO_CLIENT_ID", "")


def _get(url: str, timeout: int = 15) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def available() -> bool:
    return bool(CLIENT_ID)


def search_song(name: str, limit: int = 5) -> list:
    """返回 [{key, name, artists, duration_s, source='jamendo'}]。未配置 client_id 时返回 []。"""
    if not CLIENT_ID:
        return []
    q = urllib.parse.quote(name)
    url = (f"https://api.jamendo.com/v3.0/tracks/?client_id={CLIENT_ID}"
           f"&format=json&search={q}&limit={limit}&audioformat=mp32")
    data = json.loads(_get(url))
    out = []
    for t in data.get("results", []):
        if not t.get("audiodownload_allowed"):
            continue
        out.append({
            "key": str(t["id"]),
            "name": t.get("name", ""),
            "artists": t.get("artist_name", ""),
            "duration_s": int(t.get("duration") or 0),
            "source": "jamendo",
        })
    return out


def download_audio(key: str, dest: Path) -> bool:
    """key = track id。先取 audiodownload 直链再下载。失败返回 False。"""
    if not CLIENT_ID:
        return False
    url = (f"https://api.jamendo.com/v3.0/tracks/?client_id={CLIENT_ID}"
           f"&format=json&id={key}&audioformat=mp32")
    try:
        res = json.loads(_get(url)).get("results", [])
        dl = res[0].get("audiodownload", "") if res else ""
        if not dl:
            return False
        blob = _get(dl, timeout=60)
    except Exception:
        return False
    if len(blob) < 100_000:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(blob)
    return True
