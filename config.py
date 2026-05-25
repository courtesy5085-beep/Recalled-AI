from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True, slots=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "Recalled AI")
    db_path: str = os.getenv("APP_DB_PATH", "recalled.db")
    chroma_path: str = os.getenv("CHROMA_DB_PATH", "chroma_db")

    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_embedding_model: str = os.getenv(
        "OPENAI_EMBEDDING_MODEL",
        "text-embedding-3-small",
    )
    openai_chat_model: str = os.getenv(
        "OPENAI_CHAT_MODEL",
        "gpt-4o-mini",
    )

    local_embedding_model: str = os.getenv(
        "LOCAL_EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )
    reranker_model: str = os.getenv(
        "RERANKER_MODEL",
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
    )

    chunk_size: int = int(os.getenv("TEXT_CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("TEXT_CHUNK_OVERLAP", "50"))

    vector_top_k: int = int(os.getenv("VECTOR_TOP_K", "25"))
    final_top_k: int = int(os.getenv("FINAL_TOP_K", "10"))

    @property
    def use_openai(self) -> bool:
        return bool(self.openai_api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

