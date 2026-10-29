"""逐乐句声乐诊断（B1 深度版）：对齐视图 + 歌词行 -> 每句客观指标与故障标签

设计原则：诊断做"差值"不做"绝对值"。
  - 音准/节奏：与原曲对齐帧直接比较（复用 clip_pitch_score / clip_rhythm_score）
  - 逐音核对：原唱 F0 切单音序列，逐音算 cover 偏差 -> "8 个音唱准 5 个"
  - 颤音/长音稳定度：与同句原唱 ref 网格比较（原唱有颤音而你平直 -> 才是问题）
  - 起音/气息音量/高音压力：绝对声学阈值（RMS 包络），不与混音后的原曲比响度

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
ENTRY_LATE_S = 0.35         # 进句比原唱晚
ONSET_ERR_CENTS = 60.0      # 句首前 6 帧中位偏差：起音没唱准
SETTLED_ERR_CENTS = 35.0    # 随后 12 帧收敛到此值内 -> 属于"滑入找音"
WRONG_NOTE_CENTS = 100.0    # 单音偏差超过 -> 唱错音
NOTE_HIT_CENTS = 60.0       # 单音偏差在此之内 -> 唱准
HIGH_ZONE_SHARE = 0.3       # 句内高音区帧占比达到 -> 做高音压力分析
SHOUT_DB = 4.0              # 高音区均响度高出中音区 -> 喊叫
FLIP_OCTAVE_RATIO = 0.3     # 高音区八度跳变帧占比 -> 换声断裂/转假
TAIL_SHORT_S = 0.6          # 句尾提前收声
FLAT_DYN_DB = 2.5           # 句内三段响度极差低于 -> 线条平
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


def _frame_errors(view, i0: int, i1: int):
    """乐句内逐帧音分误差 -> (帧下标 gis, 折叠误差 folded, 原始误差 raw)。

    folded 折叠到 [-600,+600) 音分（一个八度内）：八度级偏差由 octave 类指标
    单独捕获，folded 只表达"在正确八度内偏高还是偏低"；raw 保留真实半音差。
    """
    gis, folded, raw = [], [], []
    for gi in range(i0, i1):
        rj = view.mapping[gi]
        if rj < 0 or not view.cv[gi] or not view.rv[rj]:
            continue
        e = float(view.cm[gi] - view.rm[rj]) * 100
        gis.append(gi)
        raw.append(e)
        folded.append((e + 600.0) % 1200.0 - 600.0)
    if len(gis) < 8:
        return None
    return np.array(gis), np.array(folded), np.array(raw)


def _ref_notes(view, rlo: int, rhi: int, min_dur_s: float = 0.15,
               cut_semis: float = 0.7) -> list:
    """把 ref 网格 [rlo,rhi) 切成单音序列 [(n0, n1, midi_median)]。

    规则：发声帧连续、与当前音运行中位数的偏差 >cut_semis 半音即切新音；
    短于 min_dur_s 的碎片并入相邻音。这是"逐音核对"的基准谱面。
    """
    seg_m, seg_v = view.rm[rlo:rhi], view.rv[rlo:rhi]
    notes, cur, cur_med = [], [], None

    def flush():
        nonlocal cur
        if cur and cur_med is not None:
            notes.append([cur[0] + rlo, cur[-1] + rlo + 1, cur_med])
        cur = []

    for i in range(len(seg_m)):
        if not seg_v[i]:
            flush()
            cur_med = None
            continue
        if cur_med is None or abs(seg_m[i] - cur_med) > cut_semis:
            flush()
            cur_med = float(seg_m[i])
        cur.append(i)
        cur_med = float(np.median([seg_m[j] for j in cur]))
    flush()
    # 碎片并入前一个音
    merged = []
    min_n = max(int(min_dur_s / view.step), 2)
    for n0, n1, med in notes:
        if merged and (n1 - n0) < min_n:
            merged[-1][1] = n1
        else:
            merged.append([n0, n1, med])
    return merged


def _note_hits_fast(view, gis: np.ndarray, folded: np.ndarray, notes: list) -> dict | None:
    """逐音核对（向量化）：mapping 落在音符 ref 区间内的 cover 帧 -> 折叠误差中位数。"""
    if not notes or len(gis) < 4:
        return None
    rj = view.mapping[gis]
    total, hit, worst = 0, 0, []
    for n0, n1, med in notes:
        sel = (rj >= n0) & (rj < n1)
        if sel.sum() < 2:
            continue
        e = float(np.median(folded[sel]))
        total += 1
        if abs(e) <= NOTE_HIT_CENTS:
            hit += 1
        elif abs(e) >= WRONG_NOTE_CENTS:
            worst.append({"t": round(n0 * view.step, 1), "cents": round(e, 0),
                          "direction": "偏高" if e > 0 else "偏低"})
    if total < 3:
        return None
    worst.sort(key=lambda w: -abs(w["cents"]))
    return {"total": total, "hit": hit, "hit_ratio": round(hit / total, 2),
            "worst": worst[:2]}


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

    nh = m.get("notes")
    if nh and nh["worst"]:
        worst = nh["worst"][0]
        sev = "high" if (len(nh["worst"]) >= 2 and
                         abs(nh["worst"][1]["cents"]) >= WRONG_NOTE_CENTS
                         or abs(worst["cents"]) >= 250) else "mid"
        detail = (f"唱错音：本句 {nh['total']} 个音唱准 {nh['hit']} 个；"
                  f"约 {worst['t']}s 处有个音{worst['direction']} "
                  f"{abs(worst['cents']):.0f} 音分（100 音分=1 个半音）")
        add("wrong_note", sev, detail)

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

    if m.get("entry_late_s") is not None:
        e = m["entry_late_s"]
        if e >= 0.8:
            add("rhythm_late_entry", "mid",
                f"进句比原唱晚 {e:.1f}s：起句时机明显拖了，伴奏过门时要提前准备")
        elif e >= ENTRY_LATE_S:
            add("rhythm_late_entry", "low",
                f"进句比原唱晚 {e:.1f}s：起句略迟疑")

    onset_err, settled_err = m.get("onset_err_cents"), m.get("settled_err_cents")
    if onset_err is not None and settled_err is not None:
        if abs(onset_err) >= ONSET_ERR_CENTS and abs(settled_err) < SETTLED_ERR_CENTS:
            add("onset_search", "low",
                f"音头没唱准：起音偏{'高' if onset_err > 0 else '低'} "
                f"{abs(onset_err):.0f} 音分后滑回目标音——起音要先在脑中听到音高再出声")

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

    if m.get("high_shout_db"):
        add("high_shout", "mid",
            f"高音区音量比句内其他音大 {m['high_shout_db']:.1f}dB："
            "在靠喊的顶高音，喉咙偏紧；试着减半音量、保持位置唱高音")
    if m.get("falsetto_flip"):
        add("falsetto_flip", "mid",
            "高音区出现成片八度跳变：疑似换声点断裂或躲进假声，"
            "换声区（男约 E4-F4 / 女约 A4-B4）要做混声过渡而不是切换")

    if m.get("onset_rise_ms") is not None and m["onset_rise_ms"] >= ONSET_RISE_MS:
        add("soft_onset", "low",
            f"起音偏软偏慢（{m['onset_rise_ms']:.0f}ms 才唱到位）："
            "句首音头含糊，建议声带闭合在先、气息随后")
    if m.get("tail_fall_cents", 0) >= TAIL_FALL_CENTS:
        add("tail_fall", "low",
            f"尾音下滑 {m['tail_fall_cents']:.0f} 音分：句尾气息泄掉导致掉音")
    if m.get("tail_short_s") is not None and m["tail_short_s"] >= TAIL_SHORT_S:
        add("tail_short", "low",
            f"尾音提前约 {m['tail_short_s']:.1f}s 收掉：句尾时值没唱满，"
            "把最后一个音的气送到拍子结束")
    if m.get("flat_dynamics"):
        add("flat_dynamics", "low",
            "整句音量一条直线（三段起伏 <2.5dB）：缺少渐强渐弱的线条设计，"
            "试着把句尾推向高潮或收回")
    return issues


def _phrase_score(m: dict, issues: list) -> int:
    """0-100 乐句分：音准主项 + 逐音命中率 + 节奏/故障罚项。"""
    score = 100.0
    p = m.get("pitch")
    if p:
        score -= min(45.0, p["median_abs_cents"] * 0.6)
        score -= p["octave_off_ratio"] * 30
    nh = m.get("notes")
    if nh:
        score -= (1 - nh["hit_ratio"]) * 10  # 唱错音比例轻罚（重罚已在 wrong_note）
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
    返回 [{idx,start_s,end_s,lyric,ref_time_s,pitch,rhythm,notes,entry_late_s,
           vibrato_*,onset_*,tail_*,issues,score,focus?}]，按 cover 时间排序；
    focus 标出得分最低的至多 3 句（前端/LLM 重点关注）。
    """
    x, sr = _read_wav_mono(wav_path)
    rms_t, rms_db = _rms_envelope(x, sr)
    lrc = sorted(lrc or [])
    # 全曲音域 p80：高音区阈值（按 cover 自身音域，免声部假定）
    all_m = view.cm[view.cv]
    p80 = float(np.percentile(all_m, 80)) if len(all_m) > 50 else None

    phrases = []
    for inst in _phrase_instances(view, lrc):
        i0, i1 = inst["i0"], inst["i1"]
        t0, t1 = inst["start_s"], inst["end_s"]
        li = inst["line_idx"]
        m: dict = {}

        fe = _frame_errors(view, i0, i1)
        if fe is None:
            continue
        gis, folded, raw = fe
        m["signed_cents"] = round(float(np.median(folded)), 1)
        p = clip_pitch_score(t0 - 0.01, t1 + 0.01, view)
        if p:
            m["pitch"] = p
        r = clip_rhythm_score(t0, t1, view, global_slope=global_slope)
        if r:
            m["rhythm"] = r

        # 逐音核对：ref 单音序列 × cover 折叠误差
        rj_all = view.mapping[gis]
        rlo, rhi = int(rj_all.min()), int(rj_all.max()) + 1
        notes = _ref_notes(view, rlo, rhi)
        nh = _note_hits_fast(view, gis, folded, notes)
        if nh:
            m["notes"] = nh

        # 句首：进句时机 + 音头找音高
        if li is not None:
            first_rj = [view.mapping[g] for g in gis[:8] if view.mapping[g] >= 0]
            if len(first_rj) >= 2:
                m["entry_late_s"] = round(
                    float(np.median(first_rj[:5])) * view.step - lrc[li][0], 2)
        if len(folded) >= 18:
            m["onset_err_cents"] = round(float(np.median(folded[:6])), 0)
            m["settled_err_cents"] = round(float(np.median(folded[6:18])), 0)

        # 高音区压力：音量突增（喊）/ 八度跳变聚集（换声断裂或躲假声）
        if p80 is not None:
            hi = view.cm[gis] > p80
            if hi.mean() >= HIGH_ZONE_SHARE and hi.sum() >= 8:
                r_at = np.interp(view.ct[gis], rms_t, rms_db)
                lo_mask = ~hi
                if lo_mask.sum() >= 8:
                    gap = float(r_at[hi].mean() - r_at[lo_mask].mean())
                    if gap >= SHOUT_DB:
                        m["high_shout_db"] = round(gap, 1)
                flip = ((np.abs(raw[hi]) >= 1000) & (np.abs(raw[hi]) <= 1400)).mean()
                if flip >= FLIP_OCTAVE_RATIO:
                    m["falsetto_flip"] = True

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
                vr = _vibrato_grid(view.rm[int(rj.min()):int(rj.max()) + 1],
                                   view.rv[int(rj.min()):int(rj.max()) + 1],
                                   view.step)
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

        # 尾音时值：最后一个映射帧距歌词行末尾的缺口
        if li is not None and li + 1 < len(lrc):
            last_ref = float(rj_all.max()) * view.step
            gap_end = lrc[li + 1][0] - last_ref
            if gap_end >= TAIL_SHORT_S:
                m["tail_short_s"] = round(gap_end, 1)

        # 动态弧线：句内前中后三段响度极差
        if t1 - t0 >= 3.0:
            seg_rms = rms_db[(rms_t >= t0) & (rms_t <= t1)]
            if len(seg_rms) >= 30:
                thirds = np.array_split(seg_rms, 3)
                arc = float(max(float(t_.mean()) for t_ in thirds)
                            - min(float(t_.mean()) for t_ in thirds))
                if arc < FLAT_DYN_DB:
                    m["flat_dynamics"] = True

        rise = _onset_rise_ms(rms_t, rms_db, t0)
        if rise is not None:
            m["onset_rise_ms"] = round(rise, 0)

        issues = _rules(m)
        ref_t = float(np.median(rj_all)) * view.step
        lyric = lrc[li][1] if li is not None and 0 <= li < len(lrc) else ""
        phrases.append({
            "idx": len(phrases),
            "start_s": round(t0, 1), "end_s": round(t1, 1),
            "lyric": lyric,
            "ref_time_s": round(ref_t, 1),
            "pitch": m.get("pitch"), "rhythm": m.get("rhythm"),
            "signed_cents": m.get("signed_cents"),
            "notes": m.get("notes"),
            "entry_late_s": m.get("entry_late_s"),
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


def phrase_summary(phrases: list) -> dict:
    """乐句级聚合：反复出现的系统性问题（>=2 句同型）+ 均分，供报告/前端。"""
    if not phrases:
        return {"total": 0, "avg_score": None, "issue_counts": {}, "systemic": []}
    from collections import Counter
    cnt = Counter(i["type"] for p in phrases for i in p["issues"])
    return {
        "total": len(phrases),
        "avg_score": round(sum(p["score"] for p in phrases) / len(phrases), 1),
        "issue_counts": dict(cnt.most_common()),
        "systemic": [t for t, n in cnt.most_common(3) if n >= 2],
    }
