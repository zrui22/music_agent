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
- warnings: 系统告警

要求：
1. 只基于数据说话，客观维度以 objective 为准，不要引用 network_objective_fallback 除非 objective 缺失
2. 指出具体问题时引用时间段和歌词（如"0:23-0:31『xxx』这句音分误差 45"）
3. 每个问题给出一个可执行的练习建议
4. 语气鼓励为主，但问题要明确指出，不糊弄
5. warnings 非空时，在报告开头说明数据可信度限制

输出 JSON：
{
  "summary": "一段话总评（含最强项和最需改进项）",
  "scores_comment": {"pitch": "...", "rhythm": "...", "stability": "...", "timbre": "...", "expression": "..."},
  "issues": [{"time": "0:23-0:31", "lyric": "...", "problem": "...", "drill": "..."}],
  "practice_plan": ["本周练习建议1", "建议2", "建议3"]
}"""


def build_report(diag: dict, model: str = None) -> dict:
    """diagnose() 结果 -> 导师报告（结构化 JSON）。"""
    # 压缩输入，只保留 LLM 需要理解的字段
    compact = {
        "duration_s": diag["duration_s"],
        "match": diag["match"],
        "objective": diag["objective"],
        "subjective": diag["subjective"],
        "segments": diag["segments"][:12],
        "warnings": diag["warnings"],
    }
    if compact["match"].get("candidates"):
        compact["match"] = {k: v for k, v in compact["match"].items()
                            if k != "candidates"}
    return chat_json(SYSTEM, json.dumps(compact, ensure_ascii=False),
                     model=model, max_tokens=4096)
