<div align="center">

# 🎤 Music Agent — 智能声乐评估 Agent

**唱一段，AI 告诉你：音准差多少音分、哪句拖拍、气息稳不稳、该怎么练。**

[功能](#-功能) • [安装](#-安装) • [快速开始](#-快速开始) • [输出说明](#-输出说明) • [模型资产](#-模型资产) • [FAQ](#-faq)

</div>

---

> **开源范围说明**：本仓库只包含**评分/诊断/匹配的运行时代码**（即"真正关系评分的部分"）。
> 数据采集、清洗、弱监督标注与模型训练代码未包含在本仓库中；模型权重与参考曲库需单独获取。

## ✨ 功能

你给一段演唱（整曲、片段、甚至哼唱），Agent 自动完成四件事：

| 步骤 | 说明 |
|---|---|
| **1. 曲目匹配** | 报歌名最可靠；也支持哼唱/片段自动识别——三路并集召回（软互相关 / 自由起止子序列 DTW / 音程序列模糊匹配）+ numba 加速的移调×伸缩精排。置信度不足时返回 top-5 候选请你确认，**不硬猜** |
| **2. 客观分析** | F0 轨迹提取并与原曲参考对齐：音分误差、准音符比例、八度错误、节奏速度比、节奏抖动……**客观指标全部由算法直出，不走神经网络** |
| **3. 主观评分** | 音准 / 节奏 / 气息稳定 / 音色 / 表现力 / 综合，六维 0–5 分。MERT 预训练编码器 + 多路专家特征（韵律、音色质噪、客观统计）融合的多任务网络 |
| **4. 导师评语** | 把结构化诊断喂给 LLM，生成"哪句有问题 + 怎么练"的教学报告（OpenAI / Anthropic 兼容接口，可选） |

## 📦 安装

```bash
git clone https://github.com/zrui22/music_agent.git
cd music_agent
pip install -r requirements.txt
```

要求 Python ≥ 3.10；有 NVIDIA GPU 时自动用 CUDA，否则走 CPU（慢一些）。

## 🚀 快速开始

```bash
# 方式一：报歌名（最可靠，跳过自动匹配）
PYTHONPATH=. python src/agent/diagnose.py 我唱的.wav --song 月亮代表我的心

# 方式二：自动匹配（随便唱/哼一段）
PYTHONPATH=. python src/agent/diagnose.py 我唱的.wav

# 方式三：免参考模式（不比对原曲，只给网络评分）
PYTHONPATH=. python src/agent/diagnose.py 我唱的.wav --no-match

# 结果写文件
PYTHONPATH=. python src/agent/diagnose.py 我唱的.wav -o result.json
```

## 📊 输出说明

`diagnose` 输出结构化 JSON：

```jsonc
{
  "duration_s": 25.0,
  "voiced_ratio": 0.62,              // 发声占比，过低说明可能没在唱
  "match": {
    "method": "name | humming | humming_unconfirmed | none",
    "song": "相思河", "confidence": 0.61,
    "key_shift": 2,                   // 你唱的调 vs 原调（半音数）
    "candidates": [...]               // 仅不确定时给出 top-5
  },
  "objective": {                      // 算法真值（有参考时），以此为准
    "pitch":  {"median_abs_cents": 38.2,   // 中位音分误差，职业级 <20
               "in_tune_ratio": 0.71, "octave_off_ratio": 0.02, "pitch_ref": 3.5},
    "rhythm": {"tempo_ratio": 1.03,        // >1.05 拖拍 / <0.95 赶拍
               "timing_jitter_ms": 45.0, "rhythm_ref": 3.8}
  },
  "subjective": {                     // 神经网络主观分（0-5）
    "stability": 3.1, "timbre": 3.6, "expression": 3.2, "overall": 3.4
  },
  "network_objective_fallback": {"pitch": 3.2, "rhythm": 3.5},  // 免参考兜底，交叉校验用
  "segments": [{"start": 23.0, "end": 31.0, "abs_cents": 45, "lyric": "……"}],
  "warnings": ["……"]                  // 数据可信度提示，先看这个再看分
}
```

### LLM 导师评语（可选）

```python
from src.agent.report import build_report
report = build_report(diag_result)   # dict -> 总评/逐维点评/问题清单/练习计划
```

需配置任一 LLM 后端（未配置时 `build_report` 会抛错，请自行 `try/except`）：

```bash
# OpenAI 兼容接口
export CODEX_BASE_URL=... CODEX_API_KEY=... CODEX_MODEL=...
# 或 Anthropic 兼容
export ANTHROPIC_BASE_URL=... ANTHROPIC_API_KEY=... ANTHROPIC_MODEL=...
```

## 🗂 模型资产（仓库不包含，需单独准备）

```
models/
├── MERT-v1-95M/      # MERT 预训练编码器，HuggingFace: m-a-p/MERT-v1-95M
└── feat_stats.json   # 特征 z-score 归一化统计
runs/r3/best.pt       # 评分网络权重
data/reference/<id>/  # 参考曲库，每首一个目录：
                      #   meta.json(歌曲元信息) + ref.lrc.json(逐字歌词)
                      #   ref.mp3 / ref.wav / ref_f0.npz 首次使用时自动缓存
```

## 🧩 代码结构

```
src/
├── agent/       # Agent 编排：diagnose(主入口) matcher(曲目匹配) scorer(评分)
│                #             report(LLM评语) reference(曲库) llm_client
├── model/       # singing_net(MERT+多路专家特征多任务网络) heads(六维与特征组定义)
├── features/    # 专家特征：prosody(韵律/气息) timbre(音色/质噪)
├── pitchref/    # f0(音高提取) align(移调与对齐) netease(参考曲获取) song_id
└── utils/       # 音频转码
```

## ❓ FAQ

**Q: 为什么仓库里没有权重和数据？**
本仓库只开源评分相关代码。权重、归一化统计与参考曲库属于数据资产，需单独获取或自行构建。

**Q: 哼唱识别准吗？**
离线评测（200 个留出混音片段，845 首参考库）top1 36% / top5 45%。真实哼唱是无伴奏干净输入，实际会更好；混音重的中文现场片段识别率偏低。置信度 <0.5 时系统会返回候选请你确认，不会瞎猜。

**Q: 客观分和主观分有什么区别？**
音准/节奏的**客观分**是与原曲参考直接对齐算出来的（算法真值）；气息/音色/表现力等无法精确计算的维度由神经网络给**主观分**。两者冲突时以客观分为准。

**Q: 曲库里没有的歌怎么办？**
`--no-match` 免参考模式仍可给出主观评分；音准/节奏退化为网络估计（置信度低，warnings 会提示）。

## ⚠️ 限制

- 主观评分模型在弱监督网络数据上训练，分数反映大众/专家审美分布，仅供参考
- 自动匹配对混音重的现场版片段召回有限；报歌名始终是最可靠路径
- 仅供参考学习，不构成专业声乐指导

## 📄 License

待定（TBD）。权重与数据资产不在本仓库授权范围内。
