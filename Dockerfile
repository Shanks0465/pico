# ── Stage 1: build llama-server with CUDA ────────────────────────────────────
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS llama-builder

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake ninja-build build-essential libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /llama
RUN git clone --depth 1 https://github.com/ggerganov/llama.cpp .

RUN cmake -B build \
        -DGGML_CUDA=ON \
        -DLLAMA_CURL=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -G Ninja \
    && cmake --build build --target llama-server -j$(nproc)

# ── Stage 2: runtime image ────────────────────────────────────────────────────
# nvidia/cuda:runtime includes cuBLAS, which llama-server links against at runtime.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
# Python 3.12 — Ubuntu 22.04 ships 3.10, deadsnakes provides 3.12
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common curl ca-certificates \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv libcurl4 \
    && python3.12 -m ensurepip --upgrade \
    && rm -rf /var/lib/apt/lists/*

# llama-server binary and shared libs from builder stage
COPY --from=llama-builder /llama/build/bin/llama-server /usr/local/bin/llama-server
COPY --from=llama-builder /llama/build/lib/*.so* /usr/local/lib/
RUN ldconfig

# ── Pico gateway ──────────────────────────────────────────────────────────────
WORKDIR /app
COPY requirements.txt .
RUN python3.12 -m pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

# Pico config — mirrors pydantic-settings field names in src/config.py
ENV LLAMA_CPP_BASE_URL=http://127.0.0.1:8080 \
    LLAMA_MODEL_NAME=qwen2.5-coder-7b-instruct-q4_k_m \
    LLAMA_CONTEXT_LENGTH=16384 \
    LLAMA_PARALLEL_SLOTS=4 \
    MODEL_PATH=/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf \
    MODEL_URL="" \
    GPU_LAYERS=999 \
    PICO_PORT=8000

# Model files are expected on a mounted volume; see docker-compose.yml
VOLUME ["/models"]
EXPOSE 8000

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["docker-entrypoint.sh"]
