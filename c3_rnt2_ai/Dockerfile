ARG VORTEX_BACKEND_IMAGE=python:3.11-slim-bookworm
FROM ${VORTEX_BACKEND_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    C3RNT2_PROFILE=rtx4080_16gb_llama2_7b_q4_local \
    FLUTTER_HOME=/opt/flutter \
    PATH=/opt/venv/bin:/opt/flutter/bin:$PATH

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        git \
        libglu1-mesa \
        ninja-build \
        nodejs \
        npm \
        python3 \
        python3-pip \
        python3-venv \
        unzip \
        xz-utils \
        zip \
    && rm -rf /var/lib/apt/lists/* \
    && if ! command -v python >/dev/null 2>&1; then ln -s /usr/bin/python3 /usr/local/bin/python; fi \
    && python -m venv /opt/venv

RUN git clone --depth 1 --branch stable https://github.com/flutter/flutter.git /opt/flutter \
    && flutter config --no-analytics \
    && flutter precache --universal

WORKDIR /app

ARG LLAMA_CPP_CMAKE_ARGS=
ENV CMAKE_ARGS=${LLAMA_CPP_CMAKE_ARGS}

COPY pyproject.toml README.md ./
COPY src ./src
COPY config ./config
COPY documentos ./documentos

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 \
    && python -m pip install -e ".[api]"

EXPOSE 8000 8765

HEALTHCHECK --interval=20s --timeout=10s --start-period=180s --retries=12 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/readyz', timeout=5)" || exit 1

CMD ["python", "-m", "c3rnt2", "serve", "--profile", "rtx4080_16gb_llama2_7b_q4_local", "--host", "0.0.0.0", "--port", "8000"]
