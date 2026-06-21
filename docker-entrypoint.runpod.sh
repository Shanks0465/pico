#!/usr/bin/env bash
set -euo pipefail

# ── 0. Build llama-server on the pod if not already present ──────────────────
if [ ! -f /usr/local/bin/llama-server ]; then
    echo "llama-server not found — building with CUDA on this pod..."
    git clone --depth 1 https://github.com/ggerganov/llama.cpp /tmp/llama.cpp
    cmake -B /tmp/llama.cpp/build \
        -S /tmp/llama.cpp \
        -DGGML_CUDA=ON \
        -DLLAMA_CURL=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -G Ninja
    cmake --build /tmp/llama.cpp/build --target llama-server -j"$(nproc)"
    cp /tmp/llama.cpp/build/bin/llama-server /usr/local/bin/llama-server
    cp /tmp/llama.cpp/build/lib/*.so* /usr/local/lib/ 2>/dev/null || true
    ldconfig
    rm -rf /tmp/llama.cpp
    echo "llama-server build complete."
fi

# ── 1. Ensure the model is present ───────────────────────────────────────────
if [ ! -f "$MODEL_PATH" ]; then
    if [ -z "${MODEL_URL:-}" ]; then
        echo "ERROR: model not found at $MODEL_PATH and MODEL_URL is not set." >&2
        echo "Mount a directory containing the model file at $(dirname "$MODEL_PATH")," >&2
        echo "or set MODEL_URL to a direct download URL (e.g. from HuggingFace)." >&2
        exit 1
    fi
    echo "Model not found — downloading to $MODEL_PATH ..."
    mkdir -p "$(dirname "$MODEL_PATH")"
    curl -L --progress-bar -o "$MODEL_PATH" "$MODEL_URL"
    echo "Download complete."
fi

# ── 2. Start llama-server in the background ───────────────────────────────────
echo "Starting llama-server (GPU layers: ${GPU_LAYERS:-999}) ..."
FLASH_ATTN_FLAG=""
if [ "${GPU_LAYERS:-999}" != "0" ]; then
    FLASH_ATTN_FLAG="--flash-attn"
fi

llama-server \
    -m "$MODEL_PATH" \
    --host 127.0.0.1 --port 8080 \
    -c "${LLAMA_CONTEXT_LENGTH:-16384}" \
    -ngl "${GPU_LAYERS:-999}" \
    -np "${LLAMA_PARALLEL_SLOTS:-4}" \
    -cb \
    ${FLASH_ATTN_FLAG} \
    --cache-type-k q8_0 \
    --cache-type-v q8_0 \
    --mlock &

LLAMA_PID=$!
trap 'echo "Shutting down llama-server..."; kill "$LLAMA_PID" 2>/dev/null; wait "$LLAMA_PID" 2>/dev/null' EXIT

# ── 3. Wait for llama-server to be ready ─────────────────────────────────────
echo "Waiting for llama-server to be ready..."
RETRIES=90
until curl -sf "http://127.0.0.1:8080/health" > /dev/null 2>&1; do
    if ! kill -0 "$LLAMA_PID" 2>/dev/null; then
        echo "ERROR: llama-server process exited unexpectedly." >&2
        exit 1
    fi
    RETRIES=$((RETRIES - 1))
    if [ "$RETRIES" -le 0 ]; then
        echo "ERROR: llama-server did not become healthy in time." >&2
        exit 1
    fi
    sleep 2
done
echo "llama-server is ready."

# ── 4. Start Pico (foreground) ────────────────────────────────────────────────
echo "Starting Pico on port ${PICO_PORT:-8000} ..."
exec python3.12 -m uvicorn main:app \
    --app-dir /app/src \
    --host 0.0.0.0 \
    --port "${PICO_PORT:-8000}" \
    --workers 1
