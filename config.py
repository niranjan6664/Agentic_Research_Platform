"""Central configuration. All secrets come from environment variables / .env."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Embeddings
    embedding_backend: str = "sentence-transformers"  # or "hashing"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Chunking / retrieval
    chunk_size: int = Field(800, ge=100, le=8000)
    chunk_overlap: int = Field(120, ge=0)
    top_k: int = Field(5, ge=1, le=50)
    min_score: float = Field(0.15, ge=-1.0, le=1.0)

    # Storage / limits
    upload_dir: Path = Path("data/uploads")
    index_dir: Path = Path("data/index")
    max_upload_mb: int = Field(20, ge=1, le=200)

    # LLM + search (consumed in Milestone 2)
    llm_provider: str = "ollama"
    llm_model: str = "llama3.1"
    ollama_base_url: str = "http://localhost:11434"
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    search_provider: str = "none"
    tavily_api_key: SecretStr | None = None

    @model_validator(mode="after")
    def _check(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        if self.embedding_backend not in {"sentence-transformers", "hashing"}:
            raise ValueError("EMBEDDING_BACKEND must be 'sentence-transformers' or 'hashing'")
        return self

    def safe_dict(self) -> dict:
        """Settings safe to show in a UI: secrets reduced to configured/not configured."""
        out = {}
        for k, v in self.model_dump().items():
            out[k] = (v is not None) if isinstance(v, SecretStr) or k.endswith("_api_key") else str(v) if isinstance(v, Path) else v
        return out


@lru_cache
def get_settings() -> Settings:
    return Settings()
