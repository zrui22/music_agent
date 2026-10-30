"""参考曲库解析：歌名 -> 网易云原曲（音频+LRC+F0 曲线），磁盘缓存即曲库

曲库策略：不预建大数据库。data/reference/ 下每首歌一个目录（训练期已积累
数百首），Agent 按需现搜现建，缓存随使用自然增长。
"""
import json
import time
from pathlib import Path

import numpy as np

from src.pitchref.netease import get_reference, REF_DIR
from src.pitchref.f0 import extract_f0
from src.utils.audio import to_wav


def ensure_ref_wav(ref_dir: Path) -> Path | None:
    """ref.mp3 -> ref.wav（缓存）。"""
    wav = ref_dir / "ref.wav"
    if wav.exists():
        return wav
    mp3 = ref_dir / "ref.mp3"
    if mp3.exists() and to_wav(mp3, wav):
        return wav
    return None


def ref_f0(ref_dir: Path):
    """原曲 F0 曲线（npz 缓存）。返回 (times, midi, voiced) 或 None。"""
    npz = ref_dir / "ref_f0.npz"
    if npz.exists():
        try:
            d = np.load(npz)
            return d["times"], d["midi"], d["voiced"]
        except Exception:  # 并发写半成品/损坏：删掉重建
            npz.unlink(missing_ok=True)
    wav = ensure_ref_wav(ref_dir)
    if wav is None:
        return None
    try:
        t, m, v = extract_f0(str(wav))
    except Exception:
        return None
    tmp = ref_dir / "ref_f0.tmp.npz"  # savez 只认 .npz 结尾不追加扩展名
    np.savez_compressed(tmp, times=t, midi=m, voiced=v)
    tmp.replace(npz)  # 原子替换，匹配侧不会读到半文件
    return t, m, v


def resolve_by_name(song: str, cover_dur_s: float = 0, interval: float = 1.0) -> dict | None:
    """歌名 -> 参考（搜索/下载/缓存），并确保 ref.wav + F0 曲线就绪。"""
    ref = get_reference(song, cover_dur_s=cover_dur_s, interval=interval)
    if not ref:
        return None
    return _assemble(ref["dir"])


def _assemble(ref_dir: Path) -> dict | None:
    """ref 目录 -> {dir, song, artists, ref_id, lrc, f0}，F0 不可用时 None。"""
    f0 = ref_f0(ref_dir)
    if f0 is None:
        return None
    lrc = []
    lp = ref_dir / "ref.lrc.json"
    if lp.exists():
        try:
            lrc = json.loads(lp.read_text(encoding="utf-8"))
        except Exception:
            pass
    meta = {}
    mp = ref_dir / "meta.json"
    if mp.exists():
        try:
            meta = json.loads(mp.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"dir": ref_dir, "song": meta.get("name"), "artists": meta.get("artists"),
            "ref_id": meta.get("id"), "lrc": lrc, "f0": f0}


def fetch_by_id(song_id: int, query: str, name: str, artists: str,
                duration_s: float = 0) -> dict | None:
    """按 id 落库一个参考版本（下载+歌词+F0），已缓存直接装配。
    下载源 fallback：网易云 outer（免费）→ 酷狗（免费，覆盖现场/DJ/翻唱版）。"""
    from src.pitchref.netease import download_audio as netease_dl, fetch_lyric
    from src.pitchref.kugou import search_song as kg_search, download_audio as kugou_dl
    d = REF_DIR / str(song_id)
    d.mkdir(parents=True, exist_ok=True)
    mp3 = d / "ref.mp3"
    if not mp3.exists():
        if not netease_dl(song_id, mp3):
            # 网易云 VIP/无版权 → 酷狗 fallback（搜歌名+歌手，选时长最接近的）
            kg = kg_search(f"{name} {artists}".strip() or query, limit=5)
            kg = sorted(kg, key=lambda x: abs(x["duration_s"] - duration_s)) if duration_s else kg
            ok = any(kugou_dl(x["hash"], mp3) for x in kg[:2])
            if not ok:
                return None
    mp = d / "meta.json"
    if not mp.exists():
        mp.write_text(json.dumps({"query": query, "id": song_id, "name": name,
                                  "artists": artists, "duration_s": duration_s},
                                 ensure_ascii=False), encoding="utf-8")
    lp = d / "ref.lrc.json"
    if not lp.exists():
        try:
            lp.write_text(json.dumps(fetch_lyric(song_id), ensure_ascii=False),
                          encoding="utf-8")
        except Exception:
            pass
    return _assemble(d)


_SOURCE_PREFIX = {"kugou": "kg", "5sing": "sg", "jamendo": "jm"}


def source_dir_key(source: str, key: str) -> str:
    """在线候选 -> 本地目录名（netease 用纯数字 id，其余 <前缀>_<key>）。"""
    if source == "netease":
        return str(key)
    prefix = _SOURCE_PREFIX.get(source, "x")
    return f"{prefix}_{key.replace(':', '_')}"


def fetch_by_key(source: str, key: str, query: str, name: str, artists: str,
                 duration_s: float = 0) -> dict | None:
    """按源落库一个参考版本（非网易源统一入口）。无标准 LRC（歌词留空占位，
    评测靠 F0 对齐不依赖歌词；网易同曲歌词可由 fetch_by_id 路径补）。"""
    if source == "kugou":
        from src.pitchref.kugou import download_audio as dl
    elif source == "5sing":
        from src.pitchref.sing5 import download_audio as dl
    elif source == "jamendo":
        from src.pitchref.jamendo import download_audio as dl
    else:
        return None
    d = REF_DIR / source_dir_key(source, key)
    d.mkdir(parents=True, exist_ok=True)
    mp3 = d / "ref.mp3"
    if not mp3.exists():
        if not dl(key, mp3):
            return None
    mp = d / "meta.json"
    if not mp.exists():
        mp.write_text(json.dumps({"query": query, "id": d.name,
                                  "name": name, "artists": artists,
                                  "duration_s": duration_s, "source": source},
                                 ensure_ascii=False), encoding="utf-8")
    lp = d / "ref.lrc.json"
    if not lp.exists():
        lp.write_text("[]", encoding="utf-8")   # 占位，避免每次重试
    return _assemble(d)


def _align_cost(cover_f0, ref_f0_tuple, lrc=None) -> tuple:
    """cover 对某参考版本的对齐代价与有效帧数；失败返回 (inf, 0)。"""
    from src.pitchref.align import align_best
    c_t, c_m, c_v = cover_f0
    r_t, r_m, r_v = ref_f0_tuple
    try:
        view, ks = align_best(c_t, c_m, c_v, r_t, r_m, r_v, lrc=lrc)
    except Exception:
        return float("inf"), 0, 0
    if view is None or len(view.ct) < 100:
        return float("inf"), 0, 0
    n_tot = sum(n for _, n in view.seg_costs)
    cost = sum(c * n for c, n in view.seg_costs) / max(n_tot, 1)
    return float(cost), len(view.ct), int(ks)


def resolve_best_version(song: str, cover_f0, cover_dur_s: float = 0,
                         max_versions: int = 3, interval: float = 1.0) -> dict | None:
    """同名多版本时逐一对齐，选代价最低的版本（实测决定性问题：同名不同歌/编曲）。

    cover_f0 = (times, midi, voiced)。返回的 dict 多带 tried 列表记录各版本代价。
    本地曲库（含酷狗 kg_ 目录）同名参考优先对齐，能免下载则免。
    """
    tried, best = [], None
    seen_dirs = set()

    def _try(ref, tag):
        nonlocal best
        if ref is None or str(ref["dir"]) in seen_dirs:
            return
        seen_dirs.add(str(ref["dir"]))
        cost, n_frames, ks = _align_cost(cover_f0, ref["f0"], ref.get("lrc"))
        tried.append({"ref_id": tag, "artists": ref.get("artists"),
                      "cost": round(cost, 3) if np.isfinite(cost) else None,
                      "n_frames": n_frames})
        if n_frames >= 100 and (best is None or cost < best[0]):
            best = (cost, ks, ref)

    # 本地曲库同名参考优先（含 kg_ 目录），免下载
    for it in library(load_f0=False):
        if (it.get("song") or "").strip() == song.strip():
            _try(_assemble(it["dir"]), it.get("ref_id"))

    # 本地无可对齐版本（或对齐代价过高疑似错版本）→ 联网搜网易候选
    if best is None or best[0] > 0.75:
        from src.pitchref.netease import search_song
        try:
            cands = search_song(song)
        except Exception:
            cands = []
        exact = [c for c in cands if c["name"].strip() == song.strip()]
        # 同名精确匹配优先，其次相关结果；已在库里的版本排最前（免下载）
        def _in_lib(c):
            return (REF_DIR / str(c["id"]) / "ref_f0.npz").exists()
        pool = sorted(exact, key=lambda c: -int(_in_lib(c))) or cands
        for c in pool[:max_versions]:
            ref = _assemble(REF_DIR / str(c["id"]))
            if ref is None:
                ref = fetch_by_id(c["id"], song, c["name"], c["artists"],
                                  c.get("duration_s", 0))
                time.sleep(interval)
            _try(ref, c["id"])
        if best is None:  # 全部对齐失败：退回首个可用版本（旧行为兜底）
            for c in pool[:max_versions]:
                ref = _assemble(REF_DIR / str(c["id"]))
                if ref:
                    ref["tried"] = tried
                    ref["version_confidence"] = 0.2
                    return ref
            return None
    if best is None:
        return None
    cost, ks, ref = best
    ref["tried"] = tried
    ref["version_confidence"] = round(max(0.0, min(1.0, (0.9 - cost) / 0.6)), 3)
    return ref


def library(require_cached_f0: bool = True, load_f0: bool = True) -> list:
    """枚举缓存曲库，供哼唱匹配。

    require_cached_f0=True：只收已有 ref_f0.npz 的参考（匹配侧要求快；
    全库 F0 预建用 scripts/build_ref_f0_cache.py 离线跑）。
    load_f0=False：只读 meta.json、不加载 npz（歌名补全等列表场景，
    避免每次请求对全库做数百 MB IO），返回项的 f0 字段为 None。
    """
    out = []
    if not REF_DIR.exists():
        return out
    for d in sorted(REF_DIR.iterdir()):
        mp = d / "meta.json"
        if not mp.exists():
            continue
        if require_cached_f0 and not (d / "ref_f0.npz").exists():
            continue
        try:
            meta = json.loads(mp.read_text(encoding="utf-8"))
        except Exception:
            continue
        if load_f0:
            f0 = ref_f0(d)
            if f0 is None or f0[2].mean() < 0.15:
                continue
        else:
            f0 = None
        out.append({"dir": d, "song": meta.get("name"), "artists": meta.get("artists"),
                    "ref_id": meta.get("id"), "query": meta.get("query"), "f0": f0})
    return out
