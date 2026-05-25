from __future__ import annotations

import logging
from datetime import datetime

from langchain_text_splitters import RecursiveCharacterTextSplitter

from domain.models import MemoryCreateRequest, VectorChunk
from repositories.memory_repository import MemoryRepository
from services.ai_service import EmbeddingProvider, MetadataGenerator
from vectorstore.chroma_store import ChromaVectorStore

logger = logging.getLogger(__name__)


class MemoryService:
    def __init__(
        self,
        memory_repository: MemoryRepository,
        metadata_generator: MetadataGenerator,
        embedding_provider: EmbeddingProvider,
        vector_store: ChromaVectorStore,
        splitter: RecursiveCharacterTextSplitter,
    ) -> None:
        self._memories = memory_repository
        self._metadata = metadata_generator
        self._embeddings = embedding_provider
        self._vector_store = vector_store
        self._splitter = splitter

    def save_memory(
        self,
        user_id: int,
        text: str,
        memory_type: str = "note",
        source_type: str = "text",
    ) -> int:
        normalized_text = text.strip()
        if not normalized_text:
            raise ValueError("Memory content cannot be empty")

        metadata = self._metadata.generate(normalized_text)
        created_at = datetime.utcnow().isoformat()

        chunks = self._splitter.split_text(normalized_text)
        if not chunks:
            chunks = [normalized_text]

        request = MemoryCreateRequest(
            user_id=user_id,
            title=metadata.title,
            content=normalized_text,
            summary=metadata.summary,
            emotion=metadata.emotion,
            tags=metadata.tags,
            memory_type=memory_type.lower(),
            source_type=source_type.lower(),
            created_at=created_at,
        )

        memory_id = self._memories.create_memory_with_chunks(request, chunks)

        try:
            embeddings = self._embeddings.embed_many(chunks)
            vector_chunks = [
                VectorChunk(
                    chunk_id=f"{memory_id}:{idx}",
                    memory_id=memory_id,
                    user_id=user_id,
                    chunk_index=idx,
                    content=chunk,
                    embedding=embedding,
                    metadata={
                        "chunk_id": f"{memory_id}:{idx}",
                        "memory_id": str(memory_id),
                        "user_id": str(user_id),
                        "title": metadata.title,
                        "emotion": metadata.emotion,
                        "tags": ",".join(metadata.tags),
                    },
                )
                for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings))
            ]
            self._vector_store.upsert_chunks(vector_chunks)
            self._memories.update_index_status(memory_id, "indexed")
        except Exception:
            logger.exception("Memory saved but vector indexing failed")
            self._memories.update_index_status(memory_id, "failed")

        return memory_id

