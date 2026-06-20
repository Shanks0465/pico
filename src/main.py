from fastapi import FastAPI
from routers import chat_router, models_router

app = FastAPI(title="Pico", description="Local LLM proxy for llama.cpp")

app.include_router(chat_router.router, tags=["Chat"])
app.include_router(models_router.router, tags=["Models"])


@app.get("/")
async def root():
    return {"message": "The application is up and running!"}


@app.get("/health")
async def health():
    return {"status": "ok"}
