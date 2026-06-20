# pico

A lightweight FastAPI gateway that runs a local LLM via [llama.cpp](https://github.com/ggerganov/llama.cpp) and exposes an **OpenAI-compatible API** on your machine. Drop it into any tool that speaks OpenAI (Continue.dev, Cursor, etc.) and point it at your own hardware.

```
┌─────────────────────────────┐
│  IDE / API client           │
│  (OpenAI-compatible)        │
└────────────┬────────────────┘
             │  POST /v1/chat/completions
             ▼
┌─────────────────────────────┐
│  Pico  (FastAPI :8000)      │
│  • request validation       │
│  • streaming proxy          │
└────────────┬────────────────┘
             │  http://127.0.0.1:8080
             ▼
┌─────────────────────────────┐
│  llama-server (:8080)       │
│  (bundled inside Docker)    │
└─────────────────────────────┘
```

## Quickstart

### Linux + NVIDIA GPU

```bash
# 1. Set the path to your local models directory
export MODELS_DIR=/path/to/your/models

# 2. Build and start (first run compiles llama.cpp with CUDA — takes a few minutes)
docker compose up --build
```

Pico will be available at `http://localhost:8000`.

### macOS (CPU-only)

Docker on macOS cannot access Metal/MPS, so inference runs on CPU via OpenBLAS. On Apple Silicon, ARM64 containers run natively with NEON SIMD — throughput is reasonable for a coding assistant.

```bash
export MODELS_DIR=/path/to/your/models

docker compose -f docker-compose.macos.yml up --build
```

> **Tip:** For full Metal speed on macOS, run `llama-server` natively and set `LLAMA_CPP_BASE_URL` to point at it. Pico itself can run in Docker or directly with `uvicorn`.

### Without Docker

If you already have `llama-server` running:

```bash
pip install -r requirements.txt
LLAMA_CPP_BASE_URL=http://127.0.0.1:8080 uvicorn main:app --app-dir src --port 8000
```

## Model

The default model is **Qwen2.5 Coder 7B Instruct (Q4\_K\_M)** — a capable coding model that fits in ~5 GB of VRAM/RAM.

Place the `.gguf` file in your models directory and mount it via `MODELS_DIR`. Alternatively, set `MODEL_URL` to a direct download link (e.g. from HuggingFace) and the entrypoint will download it automatically on first boot.

## API

Pico exposes an OpenAI-compatible REST API:

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Root check |
| `GET` | `/health` | Health status |
| `GET` | `/v1/models` | List loaded models |
| `POST` | `/v1/chat/completions` | Chat completions (streaming supported) |

### Example request

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-coder-7b-instruct-q4_k_m",
    "messages": [{"role": "user", "content": "Write a Python hello world"}],
    "stream": false
  }'
```

### Streaming

Set `"stream": true` in the request body to receive a Server-Sent Events (SSE) stream — identical to the OpenAI streaming format.

## Configuration

All settings are environment variables (also loadable from a `.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_CPP_BASE_URL` | `http://127.0.0.1:8080` | llama-server address |
| `LLAMA_MODEL_NAME` | `qwen2.5-coder-7b-instruct-q4_k_m` | Model name returned by `/v1/models` |
| `LLAMA_CONTEXT_LENGTH` | `16384` | Context window size (tokens) |
| `LLAMA_PARALLEL_SLOTS` | `4` | Concurrent request slots (`-np` in llama-server) |
| `MODEL_PATH` | `/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf` | Path to the `.gguf` model file |
| `MODEL_URL` | _(empty)_ | If set, auto-downloads the model on first boot |
| `GPU_LAYERS` | `999` (all) | Layers offloaded to GPU; `0` = CPU-only |
| `PICO_PORT` | `8000` | Port Pico listens on |

## IDE Integration

### Continue.dev

Copy `examples/continue_config.yaml` into `~/.continue/config.yaml` (or merge it with your existing config). Pico does not enforce authentication — `apiKey` can be any non-empty string.

```yaml
models:
  - name: Qwen2.5 Coder 7B (Pico)
    provider: openai
    model: qwen2.5-coder-7b-instruct-q4_k_m
    apiBase: http://localhost:8000/v1
    apiKey: dummy
    roles: [chat, edit, summarize, autocomplete]
```

Any tool that supports a custom OpenAI base URL works the same way — point it at `http://localhost:8000/v1`.

## Architecture

| Component | Tech |
|-----------|------|
| API gateway | FastAPI + Uvicorn |
| Inference backend | llama.cpp (`llama-server`) |
| GPU support | CUDA 12.4 (Linux), OpenBLAS / CPU (macOS) |
| Containerisation | Docker multi-stage build |
| Schema validation | Pydantic v2 |
| HTTP client | httpx (async) |

The gateway validates and forwards requests to `llama-server`, which handles the actual model inference. Streaming responses are proxied as SSE without buffering.
