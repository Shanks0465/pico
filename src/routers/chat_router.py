from collections.abc import AsyncGenerator

import httpx
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from fastapi.routing import APIRouter

from config import settings
from schemas import ChatCompletionRequest

router = APIRouter(prefix="/v1")

# Each queued request waits up to one slot's worth of generation time.
# With -np 4 and ~60 s per slot the worst-case wait before a slot frees is ~180 s.
_TIMEOUT = httpx.Timeout(connect=5.0, read=settings.llama_parallel_slots * 60.0, write=10.0, pool=5.0)


async def _stream_llama(payload: dict) -> AsyncGenerator[str, None]:
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        async with client.stream(
            "POST",
            f"{settings.llama_cpp_base_url}/v1/chat/completions",
            json=payload,
        ) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                raise HTTPException(status_code=resp.status_code, detail=body.decode())
            async for line in resp.aiter_lines():
                if line:
                    yield f"{line}\n\n"


@router.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    payload = request.model_dump(exclude_none=True)

    if request.stream:
        return StreamingResponse(
            _stream_llama(payload),
            media_type="text/event-stream",
        )

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.post(
            f"{settings.llama_cpp_base_url}/v1/chat/completions",
            json=payload,
        )

    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    return resp.json()
