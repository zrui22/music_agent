"""5sing（中国原创音乐基地）翻唱/原创源：搜索 + 播放地址解析 + 音频下载。

搜索走网页端内嵌 JSON（双重转义），播放地址走 mobileapi，均无需登录。
typeEname：fc=翻唱 yc=原创 bz=伴奏（无干声，过滤）。素人翻唱丰富，
与用户演唱场景接近，是本项目的重点补充源。key 编码为 "<stype>:<songId>"。
"""
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}


def _get(url: str, timeout: int = 15) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _js_unescape(s: str) -> str:
    """JS 单引号字符串字面量反转义一层：\\" -> "，\\\\ -> \\（\\u 保留给 json.loads）。"""
    out, i = [], 0
    while i < len(s):
        if s[i] == "\\" and i + 1 < len(s):
            c = s[i + 1]
            if c in ('\\', '"', "'", "/"):
                out.append(c)
                i += 2
            elif c == "n":
                out.append("\n")
                i += 2
            elif c == "t":
                out.append("\t")
                i += 2
            else:               # \uXXXX 等：保留反斜杠交给 json.loads
                out.append("\\")
                i += 1
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


_EM = re.compile(r"</?em[^>]*>")


def search_song(name: str, limit: int = 5) -> list:
    """返回 [{key, name, artists, duration_s, source='5sing'}]，按页内相关度序。
    搜索页数据在 var dataList = '[...]'（JS 转义的 JSON 数组），伴奏(bz)过滤。"""
    q = urllib.parse.quote(name)
    html = _get(f"http://search.5sing.kugou.com/?keyword={q}").decode("utf-8", "ignore")
    m = re.search(r"var dataList\s*=\s*'", html)
    if not m:
        return []
    start, j = m.end(), m.end()
    while True:   # 找未被转义的结束单引号
        j = html.find("'", j)
        if j < 0:
            return []
        k, n = j - 1, 0
        while k >= 0 and html[k] == "\\":
            k -= 1
            n += 1
        if n % 2 == 0:
            break
        j += 1
    try:
        items = json.loads(_js_unescape(html[start:j]))
    except Exception:
        return []
    out = []
    for s in items:
        stype = s.get("typeEname", "")
        if stype not in ("fc", "yc") or not s.get("songId"):
            continue
        nm = _EM.sub("", s.get("songName", "")).strip()
        if not nm:
            continue
        out.append({
            "key": f"{stype}:{s['songId']}",
            "name": nm,
            "artists": (s.get("singer") or s.get("nickName") or "").strip(),
            "duration_s": 0,   # 搜索页无时长字段
            "source": "5sing",
        })
        if len(out) >= limit:
            break
    return out


def get_play_url(key: str) -> str:
    """key ("fc:15510927") -> 可下载音频直链。优先 hq，sq/lq 兜底。"""
    stype, _, sid = key.partition(":")
    url = (f"http://mobileapi.5sing.kugou.com/song/getsongurl"
           f"?songid={sid}&songtype={stype or 'fc'}")
    d = json.loads(_get(url)).get("data") or {}
    for k in ("hqurl", "squrl", "lqurl", "hqurl_backup", "squrl_backup", "lqurl_backup"):
        if d.get(k):
            return d[k]
    return ""


def download_audio(key: str, dest: Path) -> bool:
    """下载 5sing 音频到 dest。失败返回 False。"""
    url = get_play_url(key)
    if not url:
        return False
    try:
        blob = _get(url, timeout=60)
    except Exception:
        return False
    if len(blob) < 100_000:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(blob)
    return True
