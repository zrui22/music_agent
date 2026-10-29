<div align="center">

# 🎤 Music Agent — 智能声乐评估 Agent

**唱一段，AI 告诉你：音准差多少音分、哪句拖拍、气息稳不稳、该怎么练。**

[![CI](https://github.com/zrui22/music_agent/actions/workflows/ci.yml/badge.svg)](https://github.com/zrui22/music_agent/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Release](https://img.shields.io/github/v/release/zrui22/music_agent?display_name=tag)](https://github.com/zrui22/music_agent/releases)
[![Stars](https://img.shields.io/github/stars/zrui22/music_agent?style=social)](https://github.com/zrui22/music_agent)

[功能](#-功能) • [安装](#-安装) • [快速开始](#-快速开始) • [Docker](#-docker) • [输出说明](#-输出说明) • [模型与训练数据](#-模型与训练数据) • [模型资产](#-模型资产) • [测试](#-测试) • [FAQ](#-faq)

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

## 🐳 Docker

无需配环境，一条命令起完整服务（镜像默认 CPU 版 PyTorch，体积小）：

```bash
docker build -t peijiao .
docker run -p 9527:9527 -v $PWD/models:/app/models -v $PWD/data:/app/data peijiao
# 打开 http://localhost:9527
```

`models/` 与 `data/` 通过挂载注入（内容见[模型资产](#-模型资产)）；
国内拉取 MERT 编码器慢时加 `-e HF_ENDPOINT=https://hf-mirror.com`。

## 💬 对话模式（教练「陪教」）

运行后进入对话：直接打字聊声乐问题；`/sing` 对麦克风唱一段自动评测；
`/file` 评测本地音频；LLM 会把分数翻译成教练评语。

```bash
PYTHONPATH=. python src/agent/chat.py
```

```
你 > /song 月亮代表我的心        # 设定当前曲目
你 > /sing                      # 回车开始录音，唱，回车结束 → 自动评测+评语
你 > /file demo.wav             # 评测文件（哼唱也行，自动匹配曲目）
你 > 我高音总挤嗓子怎么办？      # 自由对话
```

依赖 `sounddevice`（已在 requirements）；Linux 无 PortAudio 时 `/sing` 不可用、
`/file` 不受影响（`conda install -c conda-forge portaudio` 可补齐）。
未配置 LLM 后端时进入纯命令模式，只出分数表。

## 🌐 Web 服务（前端 + REST API）

```bash
PYTHONPATH=. uvicorn src.server.app:app --host 0.0.0.0 --port 9527
```

浏览器打开 `http://localhost:9527` 即得完整前端：🎤 麦克风演唱、📁 文件上传评测、
曲目补全、六维雷达图、教练对话。交互式 API 文档在 `/docs`。

| 接口 | 说明 |
|---|---|
| `GET /api/health` | 模型/设备/曲库状态 |
| `GET /api/library?q=` | 参考曲库检索（歌名补全） |
| `GET /api/library/online?q=` | 在线多源曲目检索（多平台并行合并、标注来源、自动判重） |
| `POST /api/library/add` | 收录在线候选到本地曲库（下载音频 + 提取 F0；网易源失败自动 fallback 其他源） |
| `POST /api/diagnose` | multipart 上传音频评测（可带 `song` / `auto_match`），内置歌声门控 |
| `POST /api/llm/relay` | LLM 无状态中转（前端默认经它转发 LLM 请求，Key 不落地） |
| `POST /api/tts` | 语音合成三级回退：本地克隆声线 sidecar（可选）→ edge-tts 在线声线 → 浏览器 TTS |

**多源参考曲库**：曲目搜索/收录支持多个公开音源平台并行检索（含翻唱平台，
素人翻唱版本对评测对齐同样有效）。设置环境变量 `JAMENDO_CLIENT_ID`
（[Jamendo 开发者](https://devportal.jamendo.com/)免费注册）可额外启用
CC 授权音乐源。评测时本地曲库同名版本优先对齐，能免下载则免。

**歌声门控**：`/api/diagnose` 先判断"有没有唱"（响度 / F0 发声占比 / 最长持续音
三个轻量信号），空录音、纯噪声、说话会直接返回 `{"singing": false, "message": …}`
而不会误匹配、误评分。调试时可用 CLI 的 `--no-gate` 跳过。

**LLM 评语前端配置、后端无状态中转**：评测后端零 LLM 依赖、不存储任何 Key。在网页
「⚙ 设置」里自填协议（OpenAI/Anthropic 兼容）、Base URL、API Key、模型名，凭据只存
本浏览器 localStorage，「🔌 测试连接」通过后才算配置完成。请求默认经 `/api/llm/relay`
无状态中转转发（Key 随请求转发、不落地、不记日志；只放行 `https://` 目标或本机回环
http），因此不受浏览器跨域（CORS）限制，阿里云 MaaS 等不放行浏览器直连的端点也能用；
评测后端不可达时（例如网页脱离服务器单独分发），前端自动回退浏览器直连。

前端是纯静态单页（`web/index.html`，JS 库全部本地 vendor、无 CDN 依赖），默认调同源 API；
在「⚙ 设置」里可改成任意后端地址，网页可脱离服务器单独分发。
内置 3D 教练形象（three.js + VRM，模型在 `web/assets/avatar.vrm`，
为 VRM 官方示例模型 Alicia Solid）：评测时会根据分数做出表情反馈，
朗读点评时口型同步（语音三级回退：本地克隆声线 → edge-tts → 浏览器 TTS，
可在侧栏选择与关闭）；录音停止后先试听、确认发送或丢弃重录。

**本地克隆声线（可选）**：想要更有亲和力的专属声线，可自建 CosyVoice2 sidecar
（不随本仓库分发）：本地起任意兼容 `POST {url}/synthesize {"text":…} → audio/wav`
的 TTS 服务，然后用环境变量 `TTS_SIDECAR_URL`（默认 `http://127.0.0.1:9555`）
指给后端即可；sidecar 不在线时自动回退在线声线，无感切换。

## 📊 输出说明

`diagnose` 输出结构化 JSON：

```jsonc
{
  "duration_s": 25.0,
  "singing": true,                   // 歌声门控结果；false 时带 message 直接返回
  "gate": {"reason": "ok", "rms_db": -28.8, "voiced_ratio": 0.59, "longest_voiced_s": 1.9},
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
  "segments": [{"start_s": 23.0, "end_s": 31.0, "median_abs_cents": 45,
                "in_tune_ratio": 0.6, "ref_time_s": 95.2, "lyric": "……"}],
  "warnings": ["……"]                  // 数据可信度提示，先看这个再看分
}
```

### LLM 导师评语（可选）

> Web 前端无需此配置（在网页「⚙ 设置」里填 Key，请求经后端无状态中转转发）。
> 以下为 Python API / CLI `chat.py` 的本机配置方式。

```python
from src.agent.report import build_report
report = build_report(diag_result)   # dict -> 总评/逐维点评/问题清单/练习计划
```

需配置任一 LLM 后端（未配置时 `build_report` 会抛错，请自行 `try/except`）。
环境变量优先，也可写在项目根目录 `.env` 文件中：

```bash
# OpenAI 兼容接口
export OPENAI_BASE_URL=... OPENAI_API_KEY=... OPENAI_MODEL=...
# 或 Anthropic 兼容
export ANTHROPIC_BASE_URL=... ANTHROPIC_API_KEY=... ANTHROPIC_MODEL=...
```

## 🧠 模型与训练数据

**模型**（`singing_scorer_v1.pt`，Release 分发，fp16 186M）：

- 主干：[MERT-v1-95M](https://huggingface.co/m-a-p/MERT-v1-95M) 音乐理解预训练编码器
  （冻结卷积特征提取器与前 6 层，可训练 47.2M 参数）
- 融合：5 路专家特征组（韵律 / 音色质噪 / 参考对齐 / 实时评论流统计 / 名气协变量）
  各经 MLP 投影为 token，音频摘要作 query 做 cross-attention 互相整合；
  缺失组用可学习 null token 兜底（融合与头部 2.8M 参数）
- 输出：6 个独立评分头（音准/节奏/气息稳定/音色/表现力/综合），sigmoid×5 → 0–5 分

**训练数据**（自建弱监督数据集，仅用于研究训练，不随仓库/Release 分发）：

| 标签轨 | 规模 | 监督方式 |
|---|---|---|
| 大众文本评价弱标签 | ~10,000 切片 | LLM 清洗后广播到媒体级（conf 0.4–0.9） |
| 在线 K 歌平台 | ~3,200 切片 | 平台机打分分位映射（低权重）+ 文本标签 |
| 专家点评精标 | ~3,500 演唱段 | LLM 精标（高置信，0.85×pair_conf） |
| 算法真值（全轨覆盖） | 全部样本 | 参考对齐 pitch_ref / 逐字时间戳 rhythm_ref |

共 15,417 个演唱切片（train 14,509 / val 908，按来源 ID 哈希 85/7.5/7.5 划分防泄漏），
三轨按标签置信度分层加权（w=conf）。素材来自公开网络演唱内容及配套公开评价，
为对冲"捧场偏差"定向补充了普通人演唱与翻车现场负样本（低分段占比 11%）。

**验证集表现**（Spearman 等级相关；文本弱监督主观维度的信息上限约 0.4–0.5）：

| 音准 | 节奏 | 气息稳定 | 音色 | 综合 | 表现力 |
|---|---|---|---|---|---|
| 0.76 | 0.73 | 0.60 | 0.50 | 0.47 | 0.42 |

因此产品形态上：**音准/节奏以算法客观真值直出为准，网络负责气息/音色/表现力等
主观维度**——整体诊断精度高于任何单一网络输出。

## 🗂 模型资产（仓库不包含，从 Release 下载）

评分运行需要三件资产（共约 300M），托管在
[Release v1.0.0](https://github.com/zrui22/music_agent/releases/tag/v1.0.0)：

| 文件 | 大小 | md5 | 说明 |
|---|---|---|---|
| `singing_scorer_v1.pt` | 186M | `0b4a29d38c6fa34eecfd8953b234fdf3` | 评分网络权重（fp16），Scorer 默认加载 |
| `feat_stats.json` | 4K | `1ff0e372bddaef22a6732074c25b9578` | 专家特征 z-score 归一化统计（必需） |
| `reference_lib_v1.tar.gz` | 110M | `da47e8fdce94e5ce493ef9bf74efa490` | 参考曲库（meta + 逐字歌词 + F0 曲线） |

```bash
mkdir -p models data
REL=https://github.com/zrui22/music_agent/releases/download/v1.0.0

# 1) 评分权重 + 归一化统计（wget -c 支持断点续传）
wget -c -O models/singing_scorer_v1.pt $REL/singing_scorer_v1.pt
wget -c -O models/feat_stats.json      $REL/feat_stats.json

# 2) 参考曲库：下载 → 校验 → 解压到 data/（生成 data/reference/<id>/）
wget -c -O /tmp/reference_lib_v1.tar.gz $REL/reference_lib_v1.tar.gz
echo "da47e8fdce94e5ce493ef9bf74efa490  /tmp/reference_lib_v1.tar.gz" | md5sum -c -
tar xzf /tmp/reference_lib_v1.tar.gz -C data/

# 3) 可选：校验权重完整性
echo "0b4a29d38c6fa34eecfd8953b234fdf3  models/singing_scorer_v1.pt" | md5sum -c -
echo "1ff0e372bddaef22a6732074c25b9578  models/feat_stats.json"      | md5sum -c -
```

MERT 编码器（361M）不在 Release 里，首次运行自动从 HuggingFace
[`m-a-p/MERT-v1-95M`](https://huggingface.co/m-a-p/MERT-v1-95M) 拉取并缓存到
`models/MERT-v1-95M/`。国内网络拉不动时先设镜像再启动：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

资产布局：

```
models/
├── singing_scorer_v1.pt   # 评分网络权重（Scorer 默认加载）
└── feat_stats.json        # 特征 z-score 归一化统计
data/reference/<id>/       # 参考曲库，每首一个目录：
                           #   meta.json(歌曲元信息) + ref.lrc.json(逐字歌词) + ref_f0.npz(F0曲线)
                           #   推理只需这三件；新歌曲目首次使用时自动搜索下载
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

## ✅ 测试

核心算法（歌声门控 / 哼唱检索 / 曲库读取 / LLM 中转）有纯算法级测试，
**无需 GPU、无需模型权重**（约 2 秒跑完）：

```bash
pip install -r tests/requirements-test.txt
PYTHONPATH=. python -m pytest tests/ -q
```

CI 在每次 push / PR 自动运行（Python 3.10 / 3.12 矩阵）。

## 🤝 参与贡献

欢迎 Issue 与 PR，请先阅读 [CONTRIBUTING.md](CONTRIBUTING.md)。
两条红线：**不要提交任何音频/歌词/权重文件**，**不要提交任何 API Key**。

## ❓ FAQ

**Q: 为什么仓库里没有权重和数据？**
本仓库只开源评分相关代码。权重、归一化统计与参考曲库在 GitHub Release（v1.0.0）单独分发，见上方"模型资产"；MERT 编码器由 HuggingFace 自动下载。

**Q: 哼唱识别准吗？**
离线评测（留出混音片段测试集）top1 36% / top5 45%。真实哼唱是无伴奏干净输入，实际会更好；混音重的中文现场片段识别率偏低。置信度 <0.5 时系统会返回候选请你确认，不会瞎猜。

**Q: 客观分和主观分有什么区别？**
音准/节奏的**客观分**是与原曲参考直接对齐算出来的（算法真值）；气息/音色/表现力等无法精确计算的维度由神经网络给**主观分**。两者冲突时以客观分为准。

**Q: 曲库里没有的歌怎么办？**
`--no-match` 免参考模式仍可给出主观评分；音准/节奏退化为网络估计（置信度低，warnings 会提示）。

## ⚠️ 限制

- 主观评分模型在弱监督网络数据上训练，分数反映大众/专家审美分布，仅供参考
- 自动匹配对混音重的现场版片段召回有限；报歌名始终是最可靠路径
- 仅供参考学习，不构成专业声乐指导
