# Music Agent — 智能声乐评估 Agent（评分核心）

输入一段演唱音频，自动完成：**曲目匹配 → 客观声学分析 → 六维唱功评分 → LLM 导师评语**。

本仓库只包含评分/诊断/匹配相关的运行时代码。数据采集、清洗、弱监督标注与模型训练代码未包含在本仓库中；模型权重与参考曲库需单独获取（见下文）。

## 功能

- **曲目匹配**：支持直接报歌名，或对任意演唱/哼唱片段做自动匹配——三路并集召回（软互相关 / 自由起止子序列 DTW / 音程序列模糊匹配）+ numba 加速的移调×伸缩精排；匹配置信度不足时返回 top-5 候选由用户确认，不硬猜。
- **客观声学分析**：F0 轨迹提取与参考曲对齐，输出音准偏差、节奏偏差、音域、气息等可计算指标（客观指标由算法直出，不走神经网络）。
- **六维主观评分**：音准 pitch / 节奏 rhythm / 气息稳定 stability / 音色 timbre / 表现力 expression / 综合 overall（0–5 分），网络为 MERT 预训练编码器 + 多路专家特征（prosody 韵律 / timbre 音色 / pref 客观参考统计 / dm 弹幕统计 / fame 热度）融合的多任务结构。
- **导师评语**：将客观指标与主观分输入 LLM，生成具备声乐教学逻辑的诊断报告（OpenAI 兼容或 Anthropic 兼容接口，自动回落）。

## 安装

```bash
pip install -r requirements.txt
```

## 使用

```bash
# 报歌名（最可靠，跳过自动匹配）
PYTHONPATH=. python src/agent/diagnose.py your_singing.wav --song 歌名

# 自动匹配（哼唱/任意演唱片段）
PYTHONPATH=. python src/agent/diagnose.py your_singing.wav

# 免参考模式（不做匹配，仅免参考估计 + 主观评分）
PYTHONPATH=. python src/agent/diagnose.py your_singing.wav --no-match

# 结果写文件
PYTHONPATH=. python src/agent/diagnose.py your_singing.wav -o result.json
```

LLM 评语（可选）通过环境变量配置，二选一（自动回落）：

```bash
# OpenAI 兼容
export CODEX_BASE_URL=... CODEX_API_KEY=... CODEX_MODEL=...
# Anthropic 兼容
export ANTHROPIC_BASE_URL=... ANTHROPIC_API_KEY=... ANTHROPIC_MODEL=...
```

未配置时跳过评语生成，评分与客观指标不受影响。

## 模型权重与参考曲库

仓库不含以下资产，运行前需自行准备：

- `models/MERT-v1-95M/`：MERT 预训练编码器（HuggingFace: `m-a-p/MERT-v1-95M`）
- `models/best.pt`：评分网络权重
- `models/feat_stats.json`：特征 z-score 归一化统计
- `data/reference/`：参考曲库（歌曲元信息 + 歌词 + 参考 F0）

## 输出示例（节选）

```json
{
  "match": {"song": "...", "confidence": 0.61},
  "objective": {"median_abs_cents": 38.2, "in_tune_ratio": 0.71, "...": "..."},
  "subjective": {"pitch": 3.4, "rhythm": 3.8, "stability": 3.1,
                 "timbre": 3.6, "expression": 3.2, "overall": 3.4},
  "report": "……（LLM 导师诊断与练习建议）"
}
```

## 代码结构

```
src/
├── agent/       # Agent 编排：diagnose(主入口) / matcher(曲目匹配) / scorer(评分) / report(评语) / reference(曲库)
├── model/       # 评分网络：singing_net(结构) / heads(维度与特征组定义)
├── features/    # 专家特征：prosody(韵律气息) / timbre(音色质噪)
├── pitchref/    # 音高参考：f0(提取) / align(移调与对齐) / netease(参考曲获取) / song_id
└── utils/       # 音频转码
```
