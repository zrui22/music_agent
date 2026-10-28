"""诊断主管线：用户唱段 -> 匹配 -> 对齐 -> 客观真值 + 网络主观分 -> 结构化 JSON

用法:
  PYTHONPATH=. .venv/bin/python -m src.agent.diagnose AUDIO [--song 歌名]
      [--no-match] [--windows 5] [--ckpt runs/r3/best.pt] [-o out.json]

匹配优先级：--song 指定 > 哼唱库检索 > 免参考（仅网络分+发声特征）。
"""
import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np

from src.pitchref.f0 import extract_f0
from src.pitchref.align import (estimate_key_shift, align_multi, clip_pitch_score,
                                clip_rhythm_score, global_mapping_slope)
from src.features.prosody import prosody_features
from src.features.timbre import timbre_features
from src.utils.audio import to_wav
from src.agent import reference, matcher
from src.agent.scorer import Scorer, SR

ROOT = Path(__file__).resolve().parents[2]


def _lyric_at(lrc: list, t_s: float) -> str:
    """ref 时间 -> 所在歌词行。"""
    if not lrc:
        return ""
    cur = ""
    for t, text in lrc:
        if t <= t_s:
            cur = text
        else:
            break
    return cur


def _segment_evidence(view, lrc: list, min_dur: float = 4.0) -> list:
    """逐唱段证据：时间区间 / 音分误差 / 对应原曲歌词。"""
    segs = []
    for i in range(len(view.bounds) - 1):
        lo, hi = view.bounds[i], view.bounds[i + 1]
        t0, t1 = float(view.ct[lo]), float(view.ct[hi - 1])
        if t1 - t0 < min_dur:
            continue
        s = clip_pitch_score(t0, t1, view)
        if not s:
            continue
        mid_ref = float(np.median(view.mapping[lo:hi][view.mapping[lo:hi] >= 0])) \
            * view.step if (view.mapping[lo:hi] >= 0).any() else None
        segs.append({"start_s": round(t0, 1), "end_s": round(t1, 1),
                     "median_abs_cents": s["median_abs_cents"],
                     "in_tune_ratio": s["in_tune_ratio"],
                     "ref_time_s": round(mid_ref, 1) if mid_ref else None,
                     "lyric": _lyric_at(lrc, mid_ref) if mid_ref else ""})
    return segs


def diagnose(audio_path: str, song: str = None, auto_match: bool = True,
             n_windows: int = 5, scorer: Scorer = None,
             work_dir: str = None) -> dict:
    t_start = time.time()
    audio_path = Path(audio_path)
    tmp = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="agent_"))
    tmp.mkdir(parents=True, exist_ok=True)
    wav24 = tmp / f"{audio_path.stem}_24k.wav"
    if not to_wav(audio_path, wav24, sr=SR):
        raise RuntimeError(f"音频转换失败: {audio_path}")

    # F0 + 发声特征（同一份 F0 缓存复用）
    c_t, c_m, c_v = extract_f0(str(wav24))
    dur = float(c_t[-1]) if len(c_t) else 0.0
    f0_cache = {"times": c_t, "midi": c_m, "voiced": c_v}
    pro = prosody_features(str(wav24), 0.0, dur, f0_cache=f0_cache)
    tbr = timbre_features(str(wav24), 0.0, dur)
    if tbr.get("ok") is False:
        tbr = {}

    # --- 匹配 ---
    match = {"method": "none", "confidence": 0.0}
    ref = None
    if song:
        r = reference.resolve_best_version(song, (c_t, c_m, c_v), cover_dur_s=dur)
        if r:
            ref, match = r, {"method": "name", "song": r["song"],
                             "artists": r["artists"], "ref_id": r["ref_id"],
                             "confidence": r.get("version_confidence", 1.0),
                             "versions_tried": r.get("tried", [])}
    if ref is None and auto_match:
        lib = reference.library()
        cands = matcher.humming_match(c_t, c_m, c_v, lib, mode="union")
        slim = [{k: c[k] for k in ("song", "artists", "hit", "confidence")}
                for c in cands]
        if cands and cands[0]["confidence"] >= 0.5:
            best = cands[0]
            r = reference._assemble(best["dir"])
            if r is not None:
                ref = r
                match = {"method": "humming", "song": best["song"],
                         "artists": best["artists"], "ref_id": best["ref_id"],
                         "confidence": best["confidence"],
                         "candidates": slim}
        elif cands:
            # 低置信不硬猜：候选带给上游 Agent 层请用户确认
            match = {"method": "humming_unconfirmed",
                     "confidence": cands[0]["confidence"],
                     "candidates": slim}

    # --- 客观分（有参考时） ---
    objective, pref, segments = None, {}, []
    if ref is not None:
        r_t, r_m, r_v = ref["f0"]
        ks = estimate_key_shift(c_m, c_v, r_m, r_v)
        view = align_multi(c_t, c_m - ks, c_v, r_t, r_m, r_v)
        if view is not None and len(view.ct) >= 100:
            g_slope = global_mapping_slope(view)
            ps = clip_pitch_score(0.0, dur, view)
            rs = clip_rhythm_score(0.0, dur, view, global_slope=g_slope)
            n_tot = sum(n for _, n in view.seg_costs)
            cost = sum(c * n for c, n in view.seg_costs) / max(n_tot, 1)
            match.update({"key_shift": int(ks), "dtw_cost": round(float(cost), 3),
                          "ref_song": ref["song"], "ref_artists": ref["artists"]})
            if ps:
                pref.update({k: ps[k] for k in ("median_abs_cents", "in_tune_ratio",
                                                "octave_off_ratio")})
            if rs:
                pref.update({k: rs[k] for k in ("tempo_ratio", "timing_jitter_ms")})
            objective = {"pitch": ps, "rhythm": rs}
            segments = _segment_evidence(view, ref.get("lrc") or [])

    # --- 网络主观分 ---
    scorer = scorer or Scorer()
    import soundfile as sf
    wav, sr = sf.read(str(wav24), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    net = scorer.score(wav, {"prosody": pro, "timbre": tbr, "pref": pref or None},
                       n_windows=n_windows)

    result = {
        "audio": str(audio_path), "duration_s": round(dur, 1),
        "voiced_ratio": round(float(c_v.mean()), 3),
        "match": match,
        "objective": objective,          # 算法真值（有参考时），最终呈现以此为准
        "subjective": {k: net[k] for k in ("stability", "timbre",
                                           "expression", "overall")},
        "network_objective_fallback": {  # 免参考兜底/交叉校验，非常规呈现
            "pitch": net["pitch"], "rhythm": net["rhythm"]},
        "segments": segments,
        "warnings": _warnings(match, objective, c_v, dur),
        "elapsed_s": round(time.time() - t_start, 1),
    }
    return result


def _warnings(match, objective, c_v, diag_dur: float) -> list:
    w = []
    if match["method"] == "none":
        w.append("未匹配到参考歌曲：pitch/rhythm 为网络免参考估计，置信度低；"
                 "建议提供歌名（--song）获得算法真值")
    elif match["method"] == "humming_unconfirmed":
        tops = "、".join(f"《{c['song']}》" for c in match.get("candidates", [])[:3])
        w.append(f"哼唱匹配不确定（置信度 {match.get('confidence')}），"
                 f"候选：{tops}——请用户确认或报歌名；本次 pitch/rhythm 为免参考估计")
    elif match["method"] == "humming" and match.get("confidence", 0) < 0.7:
        w.append(f"哼唱匹配置信度 {match['confidence']} 不高，请核对 match.song 是否正确")
    if objective and objective.get("pitch"):
        n, dur = objective["pitch"]["n_frames"], diag_dur
        if n < min(300, 0.4 * dur * 20):  # 20 帧/秒(0.05s步长)，要求 40% 覆盖
            w.append(f"有效对齐帧覆盖偏低({n}帧/{dur:.0f}s)，客观音准分代表性有限")
    if c_v.mean() < 0.15:
        w.append("发声占比过低：可能是纯伴奏/说话，或人声分离缺失导致 F0 提取失败")
    return w


def main():
    ap = argparse.ArgumentParser(description="智能声乐诊断 Agent")
    ap.add_argument("audio")
    ap.add_argument("--song", default=None, help="歌名（跳过哼唱匹配，最可靠）")
    ap.add_argument("--no-match", action="store_true", help="不做任何匹配，免参考模式")
    ap.add_argument("--windows", type=int, default=5)
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()
    result = diagnose(args.audio, song=args.song,
                      auto_match=not args.no_match, n_windows=args.windows)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"写出 {args.out}")
    print(text)


if __name__ == "__main__":
    main()
