"""歌名识别：从媒体元信息提取标准歌名

K歌: info.title 即歌名，剥版本后缀
B站: 标题《歌名》正则 -> info.keyword 兜底
"""
import json
import re
from pathlib import Path

VERSION_SUFFIX = re.compile(r"[（(][^()（）]*(?:版|Live|live|Cover|cover|翻唱|剪辑|remix|Remix)[^()（）]*[)）]\s*$", re.I)
BRACKET_SONG = re.compile(r"《([^》]{1,30})》")


def from_kge_title(title: str) -> str:
    t = (title or "").strip()
    prev = None
    while prev != t:  # 多层后缀如 "xx(对唱版)(Live)"
        prev = t
        t = VERSION_SUFFIX.sub("", t).strip()
    return t


def from_info(info_path: Path) -> str:
    """从 info.json 提取歌名；失败返回空串。"""
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    title = info.get("title", "") or ""
    if info.get("platform", "") == "kge":
        return from_kge_title(title)
    m = BRACKET_SONG.search(title)
    if m:
        return m.group(1).strip()
    return (info.get("keyword") or "").strip()
