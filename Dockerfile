# 陪教（PeiJiao）— AI 声乐陪教评分服务
# 构建:  docker build -t peijiao .
# 运行:  docker run -p 9527:9527 -v $PWD/models:/app/models -v $PWD/data:/app/data peijiao
# 然后打开 http://localhost:9527
FROM python:3.10-slim

# ffmpeg 由 imageio-ffmpeg 自带 wheel 提供，无需系统安装
WORKDIR /app

# 推理默认走 CPU：用 PyTorch 官方 CPU 源，避免拉取数 GB 的 CUDA 运行时
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY web ./web
COPY README.md .

ENV PYTHONPATH=/app \
    PYTHONNOUSERSITE=1
    # 国内网络可加速 MERT 编码器下载：-e HF_ENDPOINT=https://hf-mirror.com

# 模型资产不入镜像（见 README「模型资产」），运行时挂载：
#   models/singing_scorer_v1.pt  评分网络权重
#   models/feat_stats.json       特征归一化统计
#   data/reference/              参考曲库（可选，缺失时仅不能做原曲对齐打分）
VOLUME ["/app/models", "/app/data"]

EXPOSE 9527
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:9527/api/health',timeout=4)"

CMD ["uvicorn", "src.server.app:app", "--host", "0.0.0.0", "--port", "9527"]
