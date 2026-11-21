# Local Docker box: full GPT-2 + GReaT replica for S6E11 (Deotte/matapanino).
# Mac Docker Desktop has no CUDA; image is CPU (linux/arm64 on M3).
FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=8 \
    MKL_NUM_THREADS=8 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# CPU torch wheels (arm64 + amd64)
RUN pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu \
      torch==2.5.1

RUN pip install --no-cache-dir \
      'transformers==4.51.3' \
      'accelerate==1.15.0' \
      'datasets==5.0.1' \
      'tokenizers==0.21.4' \
      'safetensors==0.8.0' \
      'be-great==0.0.14' \
      'scikit-learn==1.7.2' \
      'pandas==2.3.3' \
      'scipy==1.15.3' \
      'numpy==2.2.6'

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["help"]
