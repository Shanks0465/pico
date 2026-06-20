import httpx
from fastapi import HTTPException
from fastapi.routing import APIRouter

from config import settings
from schemas import ModelsResponse

router = APIRouter(prefix="/v1")


@router.get("/models", response_model=ModelsResponse)
async def list_models():
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            resp = await client.get(f"{settings.llama_cpp_base_url}/v1/models")
        except httpx.ConnectError:
            raise HTTPException(
                status_code=503,
                detail="llama.cpp server is not reachable at "
                f"{settings.llama_cpp_base_url}",
            )

    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    return resp.json()
