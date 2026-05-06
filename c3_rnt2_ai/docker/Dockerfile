FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_CACHE=/root/.cache/huggingface \
    FLUTTER_HOME=/opt/flutter \
    PATH=/opt/flutter/bin:/opt/flutter/bin/cache/dart-sdk/bin:${PATH}

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential pkg-config git curl docker.io docker-compose ffmpeg ca-certificates unzip xz-utils zip libglu1-mesa \
    && git clone --depth 1 -b stable https://github.com/flutter/flutter.git /opt/flutter \
    && flutter config --no-analytics --enable-web \
    && rm -rf /var/lib/apt/lists/*

COPY . /workspace

RUN pip install --upgrade pip \
    && pip install -e .[api,hf,train,llama_cpp,multimodal]

CMD ["python", "-m", "c3rnt2", "doctor", "--profile", "rtx4080_16gb_llama2_7b_q4_local", "--mock"]
