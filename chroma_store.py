from __future__ import annotations

from collections.abc import Sequence

import chromadb
from chromadb.config import Settings as ChromaSettings

from domain.models import RetrievedChunk, VectorChunk


class ChromaVectorStore:
    def __init__(self, path: str, collection_name: str) -> None:
        self._client = chromadb.PersistentClient(
            path=path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(name=collection_name)

    def upsert_chunks(self, chunks: Sequence[VectorChunk]) -> None:
        if not chunks:
            return

        self._collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.content for chunk in chunks],
            embeddings=[chunk.embedding for chunk in chunks],
            metadatas=[chunk.metadata for chunk in chunks],
        )

    def query(
        self,
        query_embedding: list[float],
        user_id: int,
        top_k: int,
    ) -> list[RetrievedChunk]:
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={"user_id": str(user_id)},
            include=["documents", "metadatas", "distances"],
        )

        docs = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        items: list[RetrievedChunk] = []
        for idx, (doc, meta, distance) in enumerate(zip(docs, metadatas, distances)):
            tags = meta.get("tags", "")
            if isinstance(tags, str):
                parsed_tags = [t.strip() for t in tags.split(",") if t.strip()]
            else:
                parsed_tags = []

            items.append(
                RetrievedChunk(
                    chunk_id=meta.get("chunk_id", f"unknown:{idx}"),
                    memory_id=int(meta["memory_id"]),
                    user_id=int(meta["user_id"]),
                    title=str(meta.get("title", "Untitled")),
                    emotion=str(meta.get("emotion", "neutral")),
                    tags=parsed_tags,
                    content=doc,
                    distance=float(distance),
                )
            )
        return items
