"""评分维度与专家特征组定义（推理/训练共用，无任何数据依赖）

  groups: 5 路专家特征组
    prosody 12d / timbre 10d / pref 5d（客观音准节奏统计）
    dm 68d（w2v 64d + 弹幕统计 4d）/ fame 2d
  heads: 6 头 {pitch,rhythm,stability,timbre,expression,overall} -> 0-5 分
"""

SR = 24000
WIN_S = 10.0
WIN = int(SR * WIN_S)

PROSODY_KEYS = ["breath_pause_per_min", "dynamic_range_db", "long_note_f0_std_cents",
                "long_note_rms_std_db", "n_long_notes", "run_per_min",
                "vibrato_coverage", "vibrato_extent_cents", "vibrato_rate_hz",
                "voiced_ratio", "voiced_seg_dur_med"]
TIMBRE_KEYS = ["f0_p5", "f0_p50", "f0_p95", "hnr_db_mean", "hnr_db_std",
               "jitter_local", "range_semitones", "shimmer_local",
               "spectral_centroid_hz", "voiced_ratio"]
PREF_KEYS = ["median_abs_cents", "in_tune_ratio", "octave_off_ratio",
             "tempo_ratio", "timing_jitter_ms"]
DM_STAT_KEYS = ["dm_density", "emo_ratio", "flat_ratio", "excl_per_dm"]
FAME_KEYS = ["log_view", "like_rate"]
GROUPS = ["prosody", "timbre", "pref", "dm", "fame"]
HEADS = ["pitch", "rhythm", "stability", "timbre", "expression", "overall"]
LOG1P_KEYS = {"median_abs_cents", "timing_jitter_ms", "breath_pause_per_min",
              "run_per_min", "n_long_notes", "dm_density"}
