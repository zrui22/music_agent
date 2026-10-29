"""逐乐句声乐诊断（B1）：对齐视图 + 歌词行 -> 每句客观指标与故障标签

设计原则：诊断做"差值"不做"绝对值"。
  - 音准/节奏：与原曲对齐帧直接比较（复用 clip_pitch_score / clip_rhythm_score）
  - 颤音/长音稳定度：与同句原唱 ref 网格比较（原唱有颤音而你平直 -> 才是问题）
  - 起音/气息音量：绝对声学阈值（RMS 包络），不与混音后的原曲比响度

乐句切分：lrc 行优先（cover 帧 -> mapping -> ref 时间 -> 所属歌词行，
重复副歌天然按 cover 时间连续性分成不同实例）；无歌词时退回气口段。

全部纯算法、无训练，CPU 毫秒级。
"""
from __future__ import annotations

import bisect

import numpy as np

from src.pitchref.align import clip_pitch_score, clip_rhythm_score
from src.features.prosody import _read_wav_mono, _rms_envelope

# ---- 阈值标定（2026-10 按 K歌/翻唱轨实测分布设定，均为 50ms 网格口径） ----
PITCH_MID_CENTS = 35.0      # 音分中位误差 >= -> 音准偏差(中)
PITCH_HIGH_CENTS = 60.0     #                    >= -> 明显偏差(高)
OCTAVE_RATIO = 0.25         # 八度错误帧占比
TEMPO_DRAG, TEMPO_RUSH = 1.08, 0.92
JITTER_MS = 150.0
VIB_EXTENT_WILD = 200.0     # 颤音幅度(音分 p95-p5) 失控
VIB_RATE_FAST = 8.0         # 颤音频率过快
VIB_REF_MIN_EXTENT = 30.0   # 原唱有颤音的认定下限
WOBBLE_CENTS = 45.0         # 长音去趋势后音高晃动（扣除颤音贡献后）
RMS_STD_DB = 4.5            # 长音音量波动
ONSET_RISE_MS = 200.0       # 起音 -20dB->-3dB 上升时间
TAIL_FALL_CENTS = 60.0      # 尾音下滑
SEVERITY_PENALTY = {"high": 20, "mid": 10, "low": 5}


def _line_index(lrc_times: list, t_ref: float) -> int:
    """ref 时间 -> 歌词行 idx（最后一个 start <= t 的行）。"""
    return bisect.bisect_right(lrc_times, t_ref) - 1


def _phrase_instances(view, lrc: list, min_dur: float = 1.2) -> list:
    """按歌词行把对齐帧切成乐句实例：[{line_idx, i0, i1}]（view 数组下标）。

    逐帧归属：帧的 mapping -> ref 时间 -> 行号；行号变化或跨段界即切。
    短于 min_dur 的碎片并入前一实例（间隙<0.5s），否则丢弃。
    """
    instances = []
    if lrc:
        lrc_times = [t for t, _ in lrc]
        cur = None  # [line_idx, i0, i1)
        for gi in range(len(view.ct)):
            rj = view.mapping[gi]
            li = _line_index(lrc_times, rj * view.step) if rj >= 0 else -1
            if cur is not None and li == cur[0] and gi == cur[2]:
                cur[2] = gi + 1  # 连续延伸
            else:
                if cur is not None:
                    instances.append(cur)
                cur = [li, gi, gi + 1] if li >= 0 else None
        if cur is not None:
            instances.append(cur)
    else:
        # 无歌词：退回气口段（align 分段边界）
        for k in range(len(view.bounds) - 1):
            instances.append([None, int(view.bounds[k]), int(view.bounds[k + 1])])

    # 碎片合并/丢弃 + 时长过滤
    out = []
    for li, i0, i1 in instances:
        t0, t1 = float(view.ct[i0]), float(view.ct[i1 - 1])
        if t1 - t0 < min_dur:
            if out and li is not None and t0 - out[-1]["end_s"] < 0.5:
                out[-1]["i1"] = i1
                out[-1]["end_s"] = t1
            continue
        out.append({"line_idx": li, "i0": i0, "i1": i1,
                    "start_s": t0, "end_s": t1})
    return out[:60]


def _longest_voiced_run(view, i0: int, i1: int, min_s: float = 1.0):
    """乐句内最长连续发声段 (a,b) 下标；短于 min_s 返回 None。"""
    min_n = max(int(min_s / view.step), 4)
    best, cur = None, None
    for i in range(i0, i1):
        if view.cv[i]:
            cur = i if cur is None else cur
        else:
            if cur is not None and i - cur >= min_n and \
                    (best is None or i - cur > best[1] - best[0]):
                best = (cur, i)
            cur = None
    if cur is not None and i1 - cur >= min_n and \
            (best is None or i1 - cur > best[1] - best[0]):
        best = (cur, i1)
    return best


def _vibrato_grid(midi: np.ndarray, voiced: np.ndarray, step: float,
                  lo_hz=4.0, hi_hz=7.5):
    """50ms 网格上的颤音检测（prosody._vibrato 的网格适配版）。

    返回 (rate_hz, extent_cents) 或 None。50ms 采样 Nyquist=10Hz，4-7.5Hz 可检。
    """
    m = midi[voiced]
    if len(m) < 12:  # <0.6s 不判
        return None
    k = 5  # 0.5s 滑动中值去趋势
    trend = np.array([np.median(m[max(0, i - k):i + k + 1]) for i in range(len(m))])
    resid = (m - trend) * 100  # cents
    if resid.std() < 8:
        return None
    zc = np.where(np.diff(np.sign(resid - resid.mean())) != 0)[0]
    if len(zc) < 4:
        return None
    dur = len(m) * step
    rate = len(zc) / 2 / max(dur, 1e-6)
    extent = float(np.percentile(resid, 95) - np.percentile(resid, 5))
    if lo_hz <= rate <= hi_hz and 20 <= extent <= 250:
        return float(rate), extent
    return None


def _detrended_std_cents(midi: np.ndarray) -> float:
    """去趋势（0.5s 中值滤波）后的音高残差 std（音分）。"""
    if len(midi) < 8:
        return 0.0
    k = 5
    trend = np.array([np.median(midi[max(0, i - k):i + k + 1]) for i in range(len(midi))])
    return float(np.std((midi - trend) * 100))


def _onset_rise_ms(rms_t: np.ndarray, rms_db: np.ndarray, t0: float) -> float | None:
    """起音上升时间：句首 0.5s 内 RMS 从 (峰-20dB) 爬到 (峰-3dB) 的毫秒数。"""
    m = (rms_t >= t0 - 0.02) & (rms_t <= t0 + 0.5)
    if m.sum() < 10:
        return None
    tt, rr = rms_t[m], rms_db[m]
    peak = float(rr.max())
    if peak < -50:
        return None
    lo, hi = peak - 20.0, peak - 3.0
    above_lo = np.where(rr >= lo)[0]
    above_hi = np.where(rr >= hi)[0]
    if not len(above_lo) or not len(above_hi):
        return None
    return float((tt[above_hi[0]] - tt[above_lo[0]]) * 1000)


def _signed_pitch(view, i0: int, i1: int):
    """乐句内逐帧有符号音分误差 -> (median_signed_folded, median_abs, errs)。

    signed 折叠到 [-600,+600) 音分（一个八度内）：八度级偏差由 octave_off_ratio
    单独捕获，signed 只表达"在正确八度内偏高还是偏低"，避免 -2000 音分这类
    来自八度跳变的无意义方向值混进报告。
    """
    errs = []
    for gi in range(i0, i1):
        rj = view.mapping[gi]
        if rj < 0 or not view.cv[gi] or not view.rv[rj]:
            continue
        errs.append(float(view.cm[gi] - view.rm[rj]) * 100)
    if len(errs) < 8:
        return None
    errs = np.array(errs)
    folded = (errs + 600.0) % 1200.0 - 600.0
    return float(np.median(folded)), float(np.median(np.abs(errs))), errs


def _rules(m: dict) -> list:
    """指标 -> 故障标签（type 与 B2 专家词表对齐）。"""
    issues = []

    def add(t, sev, detail):
        issues.append({"type": t, "severity": sev, "detail": detail})

    p = m.get("pitch")
    if p:
        med, signed = p["median_abs_cents"], m.get("signed_cents") or 0.0
        direction = "偏高" if signed > 15 else ("偏低" if signed < -15 else "")
        if med >= PITCH_HIGH_CENTS:
            add("pitch_off", "high",
                f"音准明显{direction}偏差：中位误差 {med:.0f} 音分"
                f"（职业级 <20），准音符占比 {p['in_tune_ratio'] * 100:.0f}%")
        elif med >= PITCH_MID_CENTS:
            add("pitch_off", "mid",
                f"音准{direction}偏差：中位误差 {med:.0f} 音分，"
                f"准音符占比 {p['in_tune_ratio'] * 100:.0f}%")
        if p["octave_off_ratio"] >= OCTAVE_RATIO:
            add("octave_error", "high",
                f"疑似整句高/低八度演唱（八度偏离帧占 {p['octave_off_ratio'] * 100:.0f}%）")

    r = m.get("rhythm")
    if r:
        tr = r["tempo_ratio"]
        if tr >= TEMPO_DRAG:
            add("rhythm_drag", "mid", f"本句拖拍：局部速度为原曲的 {tr:.2f} 倍")
        elif tr <= TEMPO_RUSH:
            add("rhythm_rush", "mid", f"本句赶拍：局部速度为原曲的 {tr:.2f} 倍")
        if r["timing_jitter_ms"] >= JITTER_MS:
            add("rhythm_unsteady", "mid",
                f"节奏不稳：卡拍抖动 {r['timing_jitter_ms']:.0f}ms（参考 <80ms）")

    v_c, v_r = m.get("vibrato_cover"), m.get("vibrato_ref")
    if m.get("long_note_s", 0) >= 1.5:
        if v_r and not v_c:
            add("flat_long_note", "low",
                f"长音偏平直：原唱此处有颤音（{v_r[0]:.1f}Hz/±{v_r[1] / 2:.0f}音分），"
                "你的长音缺少自然的颤音修饰")
        elif v_c and (v_c[1] >= VIB_EXTENT_WILD or v_c[0] >= VIB_RATE_FAST):
            add("wild_vibrato", "mid",
                f"颤音失控：{v_c[0]:.1f}Hz、幅度 ±{v_c[1] / 2:.0f} 音分，"
                "幅度过大听感摇晃")
    if m.get("wobble_cents", 0) >= WOBBLE_CENTS:
        add("unsteady_pitch", "mid",
            f"长音音高晃动 ±{m['wobble_cents']:.0f} 音分：气息支撑不稳，"
            "音高在目标音上下游移")
    if m.get("rms_std_db", 0) >= RMS_STD_DB:
        add("unsteady_volume", "mid",
            f"长音音量忽大忽小（波动 {m['rms_std_db']:.1f}dB）：气息输出不匀")
    if m.get("onset_rise_ms") is not None and m["onset_rise_ms"] >= ONSET_RISE_MS:
        add("soft_onset", "low",
            f"起音偏软偏慢（{m['onset_rise_ms']:.0f}ms 才唱到位）："
            "句首音头含糊，建议声带闭合在先、气息随后")
    if m.get("tail_fall_cents", 0) >= TAIL_FALL_CENTS:
        add("tail_fall", "low",
            f"尾音下滑 {m['tail_fall_cents']:.0f} 音分：句尾气息泄掉导致掉音")
    return issues


def _phrase_score(m: dict, issues: list) -> int:
    """0-100 乐句分：音准主项 + 节奏/故障罚项。"""
    score = 100.0
    p = m.get("pitch")
    if p:
        score -= min(45.0, p["median_abs_cents"] * 0.6)
        score -= p["octave_off_ratio"] * 30
    r = m.get("rhythm")
    if r:
        score -= min(15.0, r["timing_jitter_ms"] / 20.0)
        if 0.5 <= r["tempo_ratio"] <= 2.0:
            score -= min(12.0, abs(r["tempo_ratio"] - 1) * 60)
    score -= min(40.0, sum(SEVERITY_PENALTY[i["severity"]] for i in issues))
    return int(round(float(np.clip(score, 0, 100))))


def phrase_diagnostics(view, lrc: list, wav_path: str,
                       global_slope: float = None) -> list:
    """逐乐句诊断主入口。

    view: AlignedView；lrc: [(ref_t, text)]；wav_path: 24k 单声道 cover wav。
    返回 [{idx,start_s,end_s,lyric,ref_time_s,pitch,rhythm,vibrato_*,
           onset_rise_ms,issues,score,focus}]，按 cover 时间排序；
    focus 标出得分最低的至多 3 句（前端/LLM 重点关注）。
    """
    x, sr = _read_wav_mono(wav_path)
    rms_t, rms_db = _rms_envelope(x, sr)
    lrc = sorted(lrc or [])
    phrases = []
    for inst in _phrase_instances(view, lrc):
        i0, i1 = inst["i0"], inst["i1"]
        t0, t1 = inst["start_s"], inst["end_s"]
        m: dict = {}

        sp = _signed_pitch(view, i0, i1)
        if sp is None:
            continue
        m["signed_cents"] = round(sp[0], 1)
        p = clip_pitch_score(t0 - 0.01, t1 + 0.01, view)
        if p:
            m["pitch"] = p
        r = clip_rhythm_score(t0, t1, view, global_slope=global_slope)
        if r:
            m["rhythm"] = r

        # 长音段：颤音（cover vs ref）+ 音高晃动 + 音量波动
        run = _longest_voiced_run(view, i0, i1)
        if run:
            a, b = run
            m["long_note_s"] = round((b - a) * view.step, 2)
            vc = _vibrato_grid(view.cm[a:b], view.cv[a:b], view.step)
            rj = view.mapping[a:b]
            rj = rj[rj >= 0]
            vr = None
            if len(rj) >= 12:
                rlo, rhi = int(rj.min()), int(rj.max()) + 1
                vr = _vibrato_grid(view.rm[rlo:rhi], view.rv[rlo:rhi], view.step)
            if vc:
                m["vibrato_cover"] = [round(vc[0], 2), round(vc[1], 1)]
            if vr:
                m["vibrato_ref"] = [round(vr[0], 2), round(vr[1], 1)]
            mid_seg = view.cm[a:b][view.cv[a:b]]
            std = _detrended_std_cents(mid_seg)
            if vc:  # 扣除健康颤音对 std 的贡献（正弦近似 std≈extent/2.8）
                std = float(np.sqrt(max(std ** 2 - (vc[1] / 2.8) ** 2, 0.0)))
            m["wobble_cents"] = round(std, 1)
            seg_rms = rms_db[(rms_t >= t0) & (rms_t <= t1)]
            if len(seg_rms) > 10 and float(seg_rms.mean()) > -60:
                m["rms_std_db"] = round(float(seg_rms.std()), 2)
            # 尾音下滑：末 0.3s 中位 vs 全段中位
            n_tail = max(int(0.3 / view.step), 3)
            if len(mid_seg) >= n_tail * 2:
                tail = float(np.median(mid_seg[-n_tail:]))
                body = float(np.median(mid_seg))
                m["tail_fall_cents"] = round(max(0.0, (body - tail) * 100), 1)

        rise = _onset_rise_ms(rms_t, rms_db, t0)
        if rise is not None:
            m["onset_rise_ms"] = round(rise, 0)

        issues = _rules(m)
        ref_t = float(np.median(view.mapping[i0:i1][view.mapping[i0:i1] >= 0])) \
            * view.step if (view.mapping[i0:i1] >= 0).any() else None
        lyric = lrc[inst["line_idx"]][1] if inst["line_idx"] is not None \
            and 0 <= inst["line_idx"] < len(lrc) else ""
        phrases.append({
            "idx": len(phrases),
            "start_s": round(t0, 1), "end_s": round(t1, 1),
            "lyric": lyric,
            "ref_time_s": round(ref_t, 1) if ref_t else None,
            "pitch": m.get("pitch"), "rhythm": m.get("rhythm"),
            "signed_cents": m.get("signed_cents"),
            "vibrato_cover": m.get("vibrato_cover"),
            "vibrato_ref": m.get("vibrato_ref"),
            "wobble_cents": m.get("wobble_cents"),
            "rms_std_db": m.get("rms_std_db"),
            "onset_rise_ms": m.get("onset_rise_ms"),
            "tail_fall_cents": m.get("tail_fall_cents"),
            "issues": issues,
            "score": _phrase_score(m, issues),
        })

    # focus：最低分至多 3 句（分数 >=85 的句不标）
    worst = sorted((p for p in phrases if p["score"] < 85),
                   key=lambda p: p["score"])[:3]
    for p in worst:
        p["focus"] = True
    return phrases
