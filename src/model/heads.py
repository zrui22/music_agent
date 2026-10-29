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
# 32 类演唱故障（B3 归因头，顺序固定=模型输出维度顺序）：
# 前 18 类声学可规则验证，后 14 类为专家语义诊断
FAULT_TYPES = [
    "pitch_off", "octave_error", "wrong_note", "rhythm_drag", "rhythm_rush",
    "rhythm_unsteady", "rhythm_late_entry", "onset_search", "flat_long_note",
    "wild_vibrato", "unsteady_pitch", "unsteady_volume", "high_shout",
    "falsetto_flip", "soft_onset", "tail_fall", "tail_short", "flat_dynamics",
    "breath_leak", "squeeze", "larynx_position", "mix_voice", "chest_weak",
    "head_weak", "resonance", "nasal", "diction", "breath_manage",
    "emotion_flat", "over_sing", "range_stretch", "groove",
]
FAULT_IDX = {t: i for i, t in enumerate(FAULT_TYPES)}
# 故障中文名（诊断报告/前端展示用）与声学可规则验证子集
FAULT_ZH = {
    "pitch_off": "音准偏差", "octave_error": "八度问题", "wrong_note": "唱错音",
    "rhythm_drag": "拖拍", "rhythm_rush": "赶拍", "rhythm_unsteady": "节奏不稳",
    "rhythm_late_entry": "进句偏晚", "onset_search": "音头找音",
    "flat_long_note": "长音平直", "wild_vibrato": "颤音失控",
    "unsteady_pitch": "音高晃动", "unsteady_volume": "音量不稳",
    "high_shout": "高音喊叫", "falsetto_flip": "换声断裂/躲假声",
    "soft_onset": "起音偏软", "tail_fall": "尾音下滑", "tail_short": "尾音时值不足",
    "flat_dynamics": "线条偏平", "breath_leak": "漏气/气声过泄",
    "squeeze": "挤卡/喉咙紧", "larynx_position": "喉位问题",
    "mix_voice": "混声/换声区问题", "chest_weak": "胸声不足",
    "head_weak": "头声/假声虚弱", "resonance": "共鸣位置问题", "nasal": "鼻音过重",
    "diction": "咬字问题", "breath_manage": "换气/气口问题",
    "emotion_flat": "情感平/层次单一", "over_sing": "过度演唱/技巧堆砌",
    "range_stretch": "超音域硬唱", "groove": "律动/语感问题",
}
ACOUSTIC_FAULTS = frozenset(FAULT_TYPES[:18])
LOG1P_KEYS = {"median_abs_cents", "timing_jitter_ms", "breath_pause_per_min",
              "run_per_min", "n_long_notes", "dm_density"}
