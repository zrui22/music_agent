"""酷狗音乐免费源：搜索 + 播放地址解析 + 音频下载。

接口无需登录，无风控，覆盖大量版权歌曲的现场/翻唱/DJ 版本。
"""
import json
import urllib.parse
import urllib.request
from pathlib import Path

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}


def _get(url: str, timeout: int = 15) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def search_song(name: str, limit: int = 5) -> list:
    """返回 [{hash, name, artists, duration_s, source='kugou'}]，按相关度序。"""
    q = urllib.parse.quote(name)
    url = f"https://songsearch.kugou.com/song_search_v2?keyword={q}&pagesize={limit}&page=1"
    data = json.loads(_get(url))
    out = []
    for s in data.get("data", {}).get("lists", []):
        fh = s.get("FileHash", "")
        if not fh:
            continue
        out.append({
            "hash": fh,
            "name": s.get("SongName", "").replace("<em class=\"keyword\">", "").replace("</em>", ""),
            "artists": s.get("SingerName", "").replace("<em class=\"keyword\">", "").replace("</em>", ""),
            "duration_s": int(s.get("Duration", 0)),
            "source": "kugou",
        })
    return out


def get_play_url(file_hash: str) -> str:
    """hash -> 可下载的音频直链（mp3/m4a）。"""
    url = f"https://m.kugou.com/app/i/getSongInfo.php?cmd=playInfo&hash={file_hash}"
    data = json.loads(_get(url))
    if data.get("errcode") != 0:
        return ""
    return data.get("url", "")


def download_audio(file_hash: str, dest: Path) -> bool:
    """下载酷狗音频到 dest。失败返回 False。"""
    url = get_play_url(file_hash)
    if not url:
        return False
    try:
        blob = _get(url, timeout=60)
    except Exception:
        return False
    if len(blob) < 100_000:  # <100KB 基本是错误页/无效
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(blob)
    return True
