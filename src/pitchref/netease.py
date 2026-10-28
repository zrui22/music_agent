"""网易云 API：原曲搜索 / LRC 歌词 / 参考音频下载（带磁盘缓存）

实测可用（2026-10-27）：搜索、歌词、outer-url 音频下载均免登录。
"""
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REF_DIR = ROOT / "data" / "reference"

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
      "Referer": "https://music.163.com/"}


def _get(url: str, timeout: int = 15) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def search_song(name: str, limit: int = 5) -> list:
    """返回 [{id, name, artists, duration_s}]，按相关度序。"""
    q = urllib.parse.quote(name)
    data = json.loads(_get(f"https://music.163.com/api/search/get?s={q}&type=1&limit={limit}"))
    out = []
    for s in data.get("result", {}).get("songs", []):
        out.append({
            "id": s["id"],
            "name": s["name"],
            "artists": "/".join(a.get("name", "") for a in s.get("artists", [])),
            "duration_s": s.get("duration", 0) / 1000,
        })
    return out


def pick_best(cands: list, song: str, cover_dur_s: float = 0) -> dict | None:
    """歌名完全相等优先；时长合理性（cover<60s 时只要求原曲>60s）。"""
    exact = [c for c in cands if c["name"].strip() == song.strip()]
    pool = exact or cands
    if not pool:
        return None
    if cover_dur_s >= 60:
        dur_ok = [c for c in pool if 0.3 * cover_dur_s < c["duration_s"] < cover_dur_s * 3 + 60]
        pool = dur_ok or pool
    pool = [c for c in pool if c["duration_s"] > 60] or pool
    return pool[0] if pool else None


def fetch_lyric(song_id: int) -> list:
    """返回 [(t_s, text)] 按时间序；无歌词返回 []。"""
    data = json.loads(_get(f"https://music.163.com/api/song/lyric?id={song_id}&lv=1"))
    lrc = (data.get("lrc") or {}).get("lyric", "")
    out = []
    for line in lrc.splitlines():
        m = re.match(r"\[(\d+):(\d+(?:\.\d+)?)\]\s*(.*)", line)
        if not m:
            continue
        t = int(m.group(1)) * 60 + float(m.group(2))
        text = m.group(3).strip()
        if text and not re.match(r"^(作词|作曲|编曲|制作人|混音|母带|监制)\s*[:：]", text):
            out.append((t, text))
    return sorted(out)


def download_audio(song_id: int, dest: Path) -> bool:
    """outer url mp3 -> dest；VIP/无版权会拿到小体积错误页，用大小校验。"""
    try:
        blob = _get(f"https://music.163.com/song/media/outer/url?id={song_id}.mp3", timeout=30)
    except Exception:
        return False
    if len(blob) < 200_000:  # <200KB 基本是错误页/试听截断
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(blob)
    return True


def get_reference(song: str, cover_dur_s: float = 0, interval: float = 1.0) -> dict | None:
    """一条龙：搜索->按候选序尝试下载->歌词，带缓存。

    参考不必是原唱本人版本：旋律忠实的录音室版本即可用于音准参考，
    因此按相关度顺序尝试候选直到下载成功（绕过单版本 VIP 限制）。
    """
    REF_DIR.mkdir(parents=True, exist_ok=True)
    for d in REF_DIR.iterdir():  # 缓存命中：歌名匹配
        mp = d / "meta.json"
        if mp.exists():
            meta = json.loads(mp.read_text(encoding="utf-8"))
            if meta.get("query") == song:
                hit = _load_cached(d)
                if hit:
                    return hit
    cands = search_song(song)
    time.sleep(interval)
    if not cands:
        return None
    # 同名候选优先，其余按相关度续上
    exact = [c for c in cands if c["name"].strip() == song.strip()]
    others = [c for c in cands if c not in exact]
    best0 = pick_best(exact, song, cover_dur_s)
    ordered = ([best0] if best0 else []) + [c for c in exact if c != best0] + others
    for best in [o for o in ordered if o]:
        if cover_dur_s >= 60 and not (0.3 * cover_dur_s < best["duration_s"] < cover_dur_s * 3 + 60):
            continue
        if best["duration_s"] <= 60:
            continue
        d = REF_DIR / str(best["id"])
        d.mkdir(parents=True, exist_ok=True)
        mp3 = d / "ref.mp3"
        if not mp3.exists() and not download_audio(best["id"], mp3):
            continue
        lrc = fetch_lyric(best["id"])
        time.sleep(interval)
        meta = {"query": song, **best, "n_lrc": len(lrc)}
        (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        (d / "ref.lrc.json").write_text(json.dumps(lrc, ensure_ascii=False), encoding="utf-8")
        return {"dir": d, "mp3": mp3, "lrc": lrc, **best}
    return None


def _load_cached(d: Path) -> dict | None:
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    mp3 = d / "ref.mp3"
    if not mp3.exists():
        return None
    lrc = json.loads((d / "ref.lrc.json").read_text(encoding="utf-8")) if (d / "ref.lrc.json").exists() else []
    return {"dir": d, "mp3": mp3, "lrc": lrc, **meta}
