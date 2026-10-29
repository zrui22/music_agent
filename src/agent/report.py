"""诊断 JSON -> LLM 导师评语报告

客观维度的"证据"全部来自算法真值（逐段音分误差、节奏偏差、对应歌词），
LLM 只负责把结构化数据转译成有教学逻辑的评语，不凭空打分。
"""
import json

from src.agent.llm_client import chat_json

SYSTEM = """你是一位专业但亲切的声乐教练。给你一份 AI 系统对学员唱段的结构化诊断 JSON，包含：
- match: 匹配到的原曲信息（method=name 用户指定 / humming 哼唱识别 / humming_unconfirmed 哼唱候选待用户确认 / none 未匹配）
- objective.pitch: 算法对齐原曲得出的音准真值（median_abs_cents 中位音分误差，越小越好，职业级<20；in_tune_ratio 准音符比例；octave_off_ratio 八度错误占比；pitch_ref 0-5 换算分）
- objective.rhythm: 节奏真值（tempo_ratio 局部速度比，>1.05 拖拍 <0.95 赶拍；timing_jitter_ms 节奏抖动毫秒；rhythm_ref 0-5 换算分）
- subjective: 神经网络主观维度分（stability 气息稳定 / timbre 音色 / expression 表现力 / overall 综合，0-5）
- segments: 逐唱段证据（时间区间、音分误差、对应原曲歌词）
- phrases: 逐乐句诊断（每句含 score 0-100、pitch/rhythm 客观值、notes 逐音核对 {total,hit,hit_ratio,worst}（如 8 个音唱准 5 个）、vibrato_cover/vibrato_ref 颤音对比、wobble_cents 长音晃动、onset_rise_ms 起音毫秒、tail_fall_cents 尾音下滑、issues 故障标签列表含 type/severity/detail；focus=true 是最需优先练的句子）
- phrase_summary: 全曲聚合（avg_score 均分、issue_counts 各故障出现次数、systemic 出现≥2句的反复问题——这是学员的系统性短板，必须在 practice_plan 里总结为本周主攻方向）
  issues.type 词表：pitch_off 音准偏差 / octave_error 八度错误 / wrong_note 唱错音(单音偏差>100音分) / rhythm_drag 拖拍 / rhythm_rush 赶拍 / rhythm_unsteady 节奏不稳 / rhythm_late_entry 进句偏晚 / onset_search 音头没唱准后滑入 / flat_long_note 长音平直缺颤音 / wild_vibrato 颤音失控 / unsteady_pitch 长音音高晃动(气息支撑不足) / unsteady_volume 音量忽大忽小 / high_shout 高音靠喊 / falsetto_flip 换声断裂/躲假声 / soft_onset 起音偏软 / tail_fall 尾音下滑 / tail_short 尾音时值不足 / flat_dynamics 线条平缺少起伏
- warnings: 系统告警

要求：
1. 只基于数据说话，客观维度以 objective 与 phrases 为准，不要引用 network_objective_fallback 除非 objective 缺失
2. 指出具体问题时引用时间段和歌词（如"0:23-0:31『xxx』这句音分误差 45"），优先讲 focus=true 的句子
3. 每个问题给出一个可执行的练习建议，练习要对应故障类型（如 tail_fall→句尾保持气息支撑，soft_onset→先闭声带再给气）
4. 语气鼓励为主，但问题要明确指出，不糊弄
5. warnings 非空时，在报告开头说明数据可信度限制
6. phrase_summary.systemic 非空时（同一故障反复出现≥2句），practice_plan 第一条必须是针对该系统性问题的「本周主攻方向」，并解释它如何影响了多个乐句

输出 JSON：
{
  "summary": "一段话总评（含最强项和最需改进项）",
  "scores_comment": {"pitch": "...", "rhythm": "...", "stability": "...", "timbre": "...", "expression": "..."},
  "issues": [{"time": "0:23-0:31", "lyric": "...", "problem": "...", "drill": "..."}],
  "practice_plan": ["本周练习建议1", "建议2", "建议3"]
}"""


def build_report(diag: dict, model: str = None) -> dict:
    """diagnose() 结果 -> 导师报告（结构化 JSON）。"""
    # 压缩输入，只保留 LLM 需要理解的字段；phrases 按分数升序取最差 12 句（重点证据）
    phrases = sorted(diag.get("phrases") or [], key=lambda p: p.get("score", 100))
    slim_phrases = [{
        "start_s": p["start_s"], "end_s": p["end_s"], "lyric": p.get("lyric", ""),
        "score": p.get("score"), "focus": bool(p.get("focus")),
        "signed_cents": p.get("signed_cents"),
        "notes": p.get("notes"),
        "issues": p.get("issues") or [],
    } for p in phrases[:12]]
    compact = {
        "duration_s": diag["duration_s"],
        "match": diag["match"],
        "objective": diag["objective"],
        "subjective": diag["subjective"],
        "segments": diag["segments"][:12],
        "phrases": slim_phrases,
        "phrase_summary": diag.get("phrase_summary"),
        "phrase_stats": {"total": len(diag.get("phrases") or []),
                         "shown_worst": len(slim_phrases)},
        "warnings": diag["warnings"],
    }
    if compact["match"].get("candidates"):
        compact["match"] = {k: v for k, v in compact["match"].items()
                            if k != "candidates"}
    return chat_json(SYSTEM, json.dumps(compact, ensure_ascii=False),
                     model=model, max_tokens=4096)
