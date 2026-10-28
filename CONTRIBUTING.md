# 参与贡献

感谢你对「陪教」的兴趣！这个项目还处于早期，任何反馈和 PR 都欢迎。

## 报 Bug

请用 Issue 模板的 **Bug Report**，并附上：

1. `curl http://localhost:9527/api/health` 的输出（模型/曲库状态一目了然）
2. 复现步骤与完整报错堆栈
3. 如果是评测结果异常：音频时长、是否清唱、目标歌曲名（**不要上传音频文件本身**）

## 提 PR

1. Fork 后从 `main` 切分支，提交信息用中文或英文均可，一句话说清「做了什么」
2. 改动 `src/` 下代码前请先跑通测试：

   ```bash
   pip install -r tests/requirements-test.txt
   PYTHONPATH=. python -m pytest tests/ -q
   ```

3. 纯算法函数（特征提取、匹配、门控）优先写成不依赖 torch 的纯 numpy 实现，
   让 CI 保持轻量

## 红线（务必遵守）

- **不要提交任何音频、歌词、曲库、模型权重文件**——本仓库只含代码，
  数据资产通过 Release 分发，且仅限学术研究用途
- **不要提交任何 API Key / Cookie**，`.secrets/` 已在 .gitignore 中
- 爬虫与数据清洗代码不在本仓库范围内，也不会合入

## 代码结构速览

```
src/
├── server/app.py      # FastAPI 后端（HTTP API 的唯一入口）
├── agent/
│   ├── diagnose.py    # 评测主链路：门控 -> 匹配 -> 对齐 -> 打分
│   ├── matcher.py     # 哼唱检索（子序列 DTW，无模型依赖）
│   ├── scorer.py      # 神经网络打分头（懒加载 torch）
│   └── reference.py   # 参考曲库读写
├── model/heads.py     # 特征组与评分头定义（纯常量，训练/推理共用）
└── features/          # 韵律/音色专家特征（praat-parselmouth）
web/index.html         # 全部前端（单文件，three.js 本地 vendor）
```
