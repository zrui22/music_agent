"""音色/发声质量专家特征（timbre 轨）

维度映射（与 DATA_DESIGN 对齐）：
  timbre(音色)   <- hnr_db_mean / spectral_centroid_hz / range_semitones
  stability(气息) <- jitter/shimmer 的异常升高也反映发声不稳定（与 prosody 互补）

指标说明（均为 clip 级，0-5 分映射留给模型学习，此处只给原始量）：
  hnr_db_*        谐波噪声比（Harmonicity cc），高=声音纯净圆润；混响/嘶哑会降低
  jitter/shimmer  周期扰动（局部），过高=发声不稳或音频嘈杂；Praat 局部算法
  spectral_centroid_hz  频谱质心，音色亮度的代理（亮/暗）
  f0_p5/p50/p95   音高分布；range_semitones 音域跨度（半音）

注意：jitter/shimmer/HNR 对混响与底噪敏感，live 音频上绝对值不可跨样本直接比较，
需结合 conf（voiced_ratio）使用；模型应学"同歌曲上下文内的相对表现"。
"""
import wave

import numpy as np
import parselmouth


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


def _spectral_centroid(x: np.ndarray, sr: int, win=2048, hop=1024) -> float:
    if len(x) < win:
        return float("nan")
    n = (len(x) - win) // hop + 1
    frames = np.lib.stride_tricks.as_strided(
        x, shape=(n, win), strides=(x.strides[0] * hop, x.strides[0]))
    sp = np.abs(np.fft.rfft(frames * np.hanning(win), axis=1)) + 1e-10
    freqs = np.linspace(0, sr / 2, sp.shape[1])
    cents = (sp * freqs).sum(axis=1) / sp.sum(axis=1)
    return float(np.mean(cents))


def timbre_features(wav_path: str, start_s: float, end_s: float,
                    f0_cache=None) -> dict:
    """clip 级音色/发声质量特征。f0_cache 保留接口一致性（本轨自己提 F0 做百分位）。"""
    try:
        x, sr = _read_wav_mono(wav_path)
        i0, i1 = int(start_s * sr), int(end_s * sr)
        seg = x[i0:i1]
        if len(seg) < sr * 1.0:
            return {"ok": False, "reason": "too_short"}
        snd = parselmouth.Sound(seg, sampling_frequency=sr)

        # F0 分布（音域）
        pit = parselmouth.praat.call(snd, "To Pitch", 0.0, 75, 600)
        f0 = pit.selected_array["frequency"]
        f0v = f0[f0 > 0]
        voiced_ratio = float(len(f0v) / max(len(f0), 1))
        if len(f0v) < 20:
            return {"ok": False, "reason": "voiced_frames_lt_20",
                    "voiced_ratio": round(voiced_ratio, 3)}

        # HNR（仅统计发声帧：Harmonicity 对无声帧给 -200）
        harm = parselmouth.praat.call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hvals = harm.values[0] if harm.values.ndim == 2 else harm.values.flatten()
        hvals = hvals[hvals > -50]
        hnr_mean = float(np.mean(hvals)) if len(hvals) else float("nan")
        hnr_std = float(np.std(hvals)) if len(hvals) else float("nan")

        # jitter / shimmer（周期扰动；发声点太少时 Praat 返回 nan）
        pp = parselmouth.praat.call([snd, pit], "To PointProcess (cc)")
        jitter = parselmouth.praat.call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shimmer = parselmouth.praat.call([snd, pp], "Get shimmer (local)",
                                         0, 0, 0.0001, 0.02, 1.3, 1.6)

        p5, p50, p95 = (float(v) for v in np.percentile(f0v, [5, 50, 95]))
        return {
            "ok": True,
            "voiced_ratio": round(voiced_ratio, 3),
            "hnr_db_mean": round(hnr_mean, 2),
            "hnr_db_std": round(hnr_std, 2),
            "jitter_local": None if np.isnan(jitter) else round(float(jitter), 5),
            "shimmer_local": None if np.isnan(shimmer) else round(float(shimmer), 4),
            "spectral_centroid_hz": round(_spectral_centroid(seg, sr), 1),
            "f0_p5": round(p5, 1), "f0_p50": round(p50, 1), "f0_p95": round(p95, 1),
            "range_semitones": round(float(12 * np.log2(p95 / max(p5, 1e-3))), 1),
        }
    except Exception as e:
        return {"ok": False, "reason": f"{type(e).__name__}:{str(e)[:60]}"}
