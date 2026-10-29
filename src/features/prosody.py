"""气息/技巧专家特征：从 F0 曲线 + RMS 包络提取可计算的唱功代理信号

设计动机：音准（pitch）只是可客观验证的一个锚点；气息稳定性、颤音质量、
动态控制等维度没有现成标签，但声学上有可靠代理量。这些特征既可直接作为
模型输入（辅路特征），也可作为弱监督锚点参与训练。

全部指标均为 clip 级，输出 dict。仅依赖 numpy（谱通量 onset 为自实现，
不引 librosa，保持推理端轻依赖）。

维度映射（与 DATA_DESIGN 四维对齐）：
  stability(气息) <- voiced_seg_dur_med / breath_pause_rate / long_note_f0_std / rms_cv_long
  技巧(technique) <- vibrato_coverage / vibrato_extent / run_rate / dynamic_range_db
  律动(groove)   <- onset_rate / ioi_cv / ioi_entropy / tempo_cv
  动态弧线        <- dyn_arc_iqr_db / dyn_arc_trend_db_s / dyn_arc_curvature
"""
import wave
from pathlib import Path

import numpy as np

from ..pitchref.f0 import extract_f0


def _read_wav_mono(path: str) -> tuple:
    with wave.open(str(path)) as w:
        sr = w.getframerate()
        n = w.getnframes()
        ch = w.getnchannels()
        raw = w.readframes(n)
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, sr


def _rms_envelope(x: np.ndarray, sr: int, hop_s=0.01, win_s=0.02):
    hop, win = int(sr * hop_s), int(sr * win_s)
    n = max((len(x) - win) // hop + 1, 0)
    if n <= 0:
        return np.array([]), np.array([])
    frames = np.lib.stride_tricks.as_strided(
        x, shape=(n, win), strides=(x.strides[0] * hop, x.strides[0]))
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    times = (np.arange(n) * hop + win / 2) / sr
    return times, 20 * np.log10(rms + 1e-8)  # dB


def _voiced_segments(times, voiced, min_gap=0.06):
    """连续发声段 [(t0,t1)]，允许 <min_gap 的断裂合并。"""
    segs, start, last = [], None, None
    for t, v in zip(times, voiced):
        if v:
            if start is None:
                start = t
            last = t
        else:
            if start is not None and (last - start) > 0.1:
                segs.append((start, last))
            start = None
    if start is not None and (last - start) > 0.1:
        segs.append((start, last))
    # 合并间隙过小的段
    merged = []
    for s in segs:
        if merged and s[0] - merged[-1][1] < min_gap:
            merged[-1] = (merged[-1][0], s[1])
        else:
            merged.append(s)
    return merged


def _vibrato(times, midi, voiced, t0, t1, lo_hz=4.0, hi_hz=7.5):
    """段内颤音检测：去趋势后残差的 4-7.5Hz 调制。返回 (rate_hz, extent_cents) 或 None。"""
    m = (times >= t0) & (times <= t1) & voiced
    if m.sum() < 40:  # <0.4s 不判
        return None
    tt, mm = times[m], midi[m]
    # 去趋势：0.5s 滑动中值
    k = 25
    trend = np.array([np.median(mm[max(0, i - k):i + k + 1]) for i in range(len(mm))])
    resid = (mm - trend) * 100  # cents
    if resid.std() < 8:  # 太平 = 无颤音
        return None
    # 过零率估频率
    zc = np.where(np.diff(np.sign(resid - resid.mean())) != 0)[0]
    if len(zc) < 4:
        return None
    dur = tt[-1] - tt[0]
    rate = len(zc) / 2 / max(dur, 1e-6)
    extent = float(np.percentile(resid, 95) - np.percentile(resid, 5))
    if lo_hz <= rate <= hi_hz and 20 <= extent <= 250:
        return float(rate), extent
    return None


def _spectral_flux_onsets(x: np.ndarray, sr: int, hop=256, win=1024) -> np.ndarray:
    """谱通量 onset 检测（numpy 自实现，干声人声上足够稳）。
    novelty = log 谱正向差分求和；自适应阈值 = 局部中值 + 1.5*MAD；最小间隔 60ms。"""
    n = (len(x) - win) // hop + 1
    if n < 8:
        return np.array([])
    frames = np.lib.stride_tricks.as_strided(
        x, shape=(n, win), strides=(x.strides[0] * hop, x.strides[0]))
    sp = np.log1p(1000.0 * np.abs(np.fft.rfft(frames * np.hanning(win), axis=1)))
    flux = np.maximum(sp[1:] - sp[:-1], 0.0).sum(axis=1)
    if len(flux) < 5:
        return np.array([])
    # 自适应阈值（1s 局部窗）
    k = max(int(sr / hop), 1)
    thr = np.array([np.median(flux[max(0, i - k):i + k + 1]) +
                    1.5 * np.median(np.abs(flux[max(0, i - k):i + k + 1] -
                                    np.median(flux[max(0, i - k):i + k + 1])))
                    for i in range(len(flux))])
    min_gap = max(int(0.06 * sr / hop), 1)
    onsets, last = [], -min_gap
    for i in range(1, len(flux) - 1):
        if flux[i] > thr[i] and flux[i] >= flux[i - 1] and flux[i] >= flux[i + 1]:
            if i - last >= min_gap:
                onsets.append(i)
                last = i
    return np.array(onsets) * hop / sr  # 秒


def _tempo_cv(flux_times: np.ndarray, onsets: np.ndarray) -> float | None:
    """局部速度稳定性：onset 间隔序列分段估 IOI 中位数，取段间变异系数。
    ≥3 个 4s 窗且每窗 ≥4 个 onset 才给出，否则 None（短段不适用）。"""
    if len(onsets) < 12:
        return None
    t_end = flux_times
    win = 4.0
    tempos = []
    t0 = 0.0
    while t0 + win <= t_end + 1e-6:
        o = onsets[(onsets >= t0) & (onsets < t0 + win)]
        if len(o) >= 4:
            ioi = np.diff(o)
            ioi = ioi[(ioi >= 0.06) & (ioi <= 2.0)]
            if len(ioi) >= 3:
                tempos.append(float(np.median(ioi)))
        t0 += win / 2
    if len(tempos) < 3:
        return None
    tempos = np.array(tempos)
    med = np.median(tempos)
    if med <= 0:
        return None
    return round(float(np.std(tempos) / med), 3)


def groove_dynarc_features(x: np.ndarray, sr: int, t0: float, t1: float,
                           rr: np.ndarray = None, rt: np.ndarray = None) -> dict:
    """律动(groove) + 动态弧线特征。仅依赖波形，可独立批量回填。

    groove：onset 率 / IOI 变异系数 / IOI 直方图熵 / 局部速度稳定性
    动态弧线：0.01s RMS(dB) 包络的 IQR / 线性趋势 / 二次曲率（乐句级强弱塑造）
    """
    i0, i1 = int(t0 * sr), int(t1 * sr)
    seg = x[i0:i1]
    dur = (i1 - i0) / sr
    if len(seg) < sr * 1.0 or dur <= 0:
        return {}
    onsets = _spectral_flux_onsets(seg, sr)
    out = {}
    out["onset_rate"] = round(len(onsets) / dur, 2)
    ioi = np.diff(onsets)
    ioi = ioi[(ioi >= 0.06) & (ioi <= 2.0)]
    if len(ioi) >= 3:
        out["ioi_cv"] = round(float(np.std(ioi) / max(np.mean(ioi), 1e-6)), 3)
        hist, _ = np.histogram(ioi, bins=24, range=(0.06, 2.0))
        p = hist / max(hist.sum(), 1)
        p = p[p > 0]
        out["ioi_entropy"] = round(float(-(p * np.log(p)).sum() / np.log(24)), 3)
    else:
        out["ioi_cv"] = None
        out["ioi_entropy"] = None
    out["tempo_cv"] = _tempo_cv(t1 - t0, onsets + t0)

    # 动态弧线（RMS 包络已由调用方给出时复用，否则自算）
    if rr is None or rt is None:
        rt, rr = _rms_envelope(x, sr)
    m = (rt >= t0) & (rt < t1)
    r = rr[m]
    if len(r) >= 20:
        out["dyn_arc_iqr_db"] = round(float(np.percentile(r, 75) - np.percentile(r, 25)), 2)
        tt = rt[m] - t0
        tn = tt / max(tt[-1], 1e-6) * 2 - 1  # 归一化到 [-1,1]
        out["dyn_arc_trend_db_s"] = round(float(np.polyfit(tt, r, 1)[0]), 3)
        out["dyn_arc_curvature"] = round(float(np.polyfit(tn, r, 2)[0]), 3)
    else:
        out["dyn_arc_iqr_db"] = None
        out["dyn_arc_trend_db_s"] = None
        out["dyn_arc_curvature"] = None
    return out


def prosody_features(wav_path, clip_start=None, clip_end=None,
                     f0_cache: dict = None) -> dict:
    """提取 clip（或全曲）的气息/技巧特征。

    f0_cache: 传入 {"times","midi","voiced"} 避免同媒体多 clip 重复跑 F0。
    """
    if f0_cache is None:
        times, midi, voiced = extract_f0(str(wav_path))
    else:
        times, midi, voiced = f0_cache["times"], f0_cache["midi"], f0_cache["voiced"]
    x, sr = _read_wav_mono(str(wav_path))
    rt, rms = _rms_envelope(x, sr)

    t0 = clip_start if clip_start is not None else 0.0
    t1 = clip_end if clip_end is not None else (times[-1] if len(times) else 0.0)
    tm = (times >= t0) & (times < t1)
    rm = (rt >= t0) & (rt < t1)
    if tm.sum() < 50 or rm.sum() < 50:
        return {"ok": False}

    tt, mm, vv = times[tm], midi[tm], voiced[tm]
    rr = rms[rm]
    segs = _voiced_segments(tt, vv)

    # ---- 气息 ----
    seg_durs = [b - a for a, b in segs]
    voiced_ratio = float(vv.mean())
    gaps = [segs[i + 1][0] - segs[i][1] for i in range(len(segs) - 1)]
    breath_gaps = [g for g in gaps if 0.15 <= g <= 2.5]
    dur_min = max((t1 - t0) / 60, 1e-6)
    long_segs = [(a, b) for a, b in segs if b - a >= 1.0]
    f0_stds, rms_cvs = [], []
    vib_rates, vib_extents = [], []
    for a, b in long_segs:
        m2 = (times >= a) & (times <= b) & voiced
        if m2.sum() < 30:
            continue
        seg_midi = midi[m2]
        k = 25  # 去颤音趋势后残余抖动感
        trend = np.array([np.median(seg_midi[max(0, i - k):i + k + 1]) for i in range(len(seg_midi))])
        f0_stds.append(float(np.std((seg_midi - trend) * 100)))
        seg_rms = rr[(rt[rm] >= a) & (rt[rm] <= b)]
        if len(seg_rms) > 10 and seg_rms.mean() > -60:
            rms_cvs.append(float(seg_rms.std()))
        v = _vibrato(times, midi, voiced, a, b)
        if v:
            vib_rates.append(v[0])
            vib_extents.append(v[1])

    # ---- 技巧 ----
    mv = mm[vv]
    if len(mv) >= 5:
        # 5帧(50ms)中值滤波压掉 F0 毛刺，保留 ≥8音/s 的真实跑动
        med = np.array([np.median(mv[max(0, i - 2):i + 3]) for i in range(len(mv))])
        dm = np.diff(med)  # 相邻发声帧半音差
    else:
        dm = np.diff(mv)
    run_rate = float((np.abs(dm) >= 0.5).sum() / dur_min) if len(dm) else 0.0  # 快速转音/跑动
    dyn = float(np.percentile(rr, 95) - np.percentile(rr, 5)) if len(rr) > 10 else 0.0

    # 律动 + 动态弧线（Step3：语义类故障"翻译成"半声学特征）
    gd = groove_dynarc_features(x, sr, t0, t1, rr=rms, rt=rt)

    return {
        "ok": True,
        # 气息
        "voiced_ratio": round(voiced_ratio, 3),
        "voiced_seg_dur_med": round(float(np.median(seg_durs)), 2) if seg_durs else 0.0,
        "breath_pause_per_min": round(len(breath_gaps) / dur_min, 1),
        "long_note_f0_std_cents": round(float(np.median(f0_stds)), 1) if f0_stds else None,
        "long_note_rms_std_db": round(float(np.median(rms_cvs)), 2) if rms_cvs else None,
        # 技巧
        "n_long_notes": len(long_segs),
        "vibrato_coverage": round(len(vib_rates) / len(long_segs), 2) if long_segs else 0.0,
        "vibrato_rate_hz": round(float(np.median(vib_rates)), 2) if vib_rates else None,
        "vibrato_extent_cents": round(float(np.median(vib_extents)), 1) if vib_extents else None,
        "run_per_min": round(run_rate, 1),
        "dynamic_range_db": round(dyn, 1),
        # 律动 / 动态弧线（Step3）
        **{k: gd.get(k) for k in ("onset_rate", "ioi_cv", "ioi_entropy", "tempo_cv",
                                  "dyn_arc_iqr_db", "dyn_arc_trend_db_s",
                                  "dyn_arc_curvature")},
    }
