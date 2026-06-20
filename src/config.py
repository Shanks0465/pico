from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    llama_cpp_base_url: str = "http://127.0.0.1:8080"
    llama_model_name: str = "qwen2.5-coder-7b-instruct-q4_k_m"
    # matches -c 16384 passed to llama-server; used to cap max_tokens
    llama_context_length: int = 16384
    # matches -np 4; used to set a sensible per-request timeout (slots * base)
    llama_parallel_slots: int = 4

    class Config:
        env_file = ".env"


settings = Settings()
