FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_CACHE=/root/.cache/huggingface

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl docker.io docker-compose ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY . /workspace

RUN pip install --upgrade pip \
    && pip install -e .[api,hf,train,multimodal]

CMD ["python", "-m", "c3rnt2", "doctor", "--profile", "rtx4080_16gb_programming_qwen_coder_local", "--mock"]
