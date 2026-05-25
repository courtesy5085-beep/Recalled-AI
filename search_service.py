from __future__ import annotations

import re
from collections.abc import Sequence

from domain.models import SearchResult
from services.ai_service import EmbeddingProvider, Reranker
from vectorstore.chroma_store import ChromaVectorStore


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _normalize(values: Sequence[float]) -> list[float]:
    if not values:
        return []

    min_v = min(values)
    max_v = max(values)

    if max_v == min_v:
        return [1.0 for _ in values]

    return [(v - min_v) / (max_v - min_v) for v in values]


class SearchService:
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        reranker: Reranker,
        vector_store: ChromaVectorStore,
        vector_top_k: int = 25,
        final_top_k: int = 10,
    ) -> None:
        self._embeddings = embedding_provider
        self._reranker = reranker
        self._vector_store = vector_store
        self._vector_top_k = vector_top_k
        self._final_top_k = final_top_k

    def search(self, query: str, user_id: int) -> list[SearchResult]:
        query = query.strip()
        if not query:
            return []

        query_embedding = self._embeddings.embed_many([query])[0]
        candidates = self._vector_store.query(
            query_embedding=query_embedding,
            user_id=user_id,
            top_k=self._vector_top_k,
        )

        if not candidates:
            return []

        documents = [item.content for item in candidates]

        # Vector score: smaller distance is better.
        vector_scores_raw = [1.0 / (1.0 + item.distance) for item in candidates]
        vector_scores = _normalize(vector_scores_raw)

        # Lightweight lexical score.
        query_tokens = set(_tokenize(query))
        lexical_raw = []
        for doc in documents:
            doc_tokens = set(_tokenize(doc))
            overlap = len(query_tokens.intersection(doc_tokens))
            lexical_raw.append(float(overlap))
        lexical_scores = _normalize(lexical_raw)

        # Semantic reranker.
        rerank_raw = self._reranker.score(query, documents)
        rerank_scores = _normalize(rerank_raw)

        ranked_items = []
        for item, vec, lex, rerank in zip(
            candidates,
            vector_scores,
            lexical_scores,
            rerank_scores,
        ):
            combined = (0.20 * vec) + (0.20 * lex) + (0.60 * rerank)
            ranked_items.append((item, combined))

        # Deduplicate by memory to avoid many chunks from one note dominating.
        best_by_memory: dict[int, SearchResult] = {}
        for item, score in sorted(ranked_items, key=lambda pair: pair[1], reverse=True):
            result = SearchResult(
                memory_id=item.memory_id,
                title=item.title,
                excerpt=item.content[:240],
                emotion=item.emotion,
                tags=item.tags,
                score=score,
            )
            if item.memory_id not in best_by_memory:
                best_by_memory[item.memory_id] = result

        return sorted(
            best_by_memory.values(),
            key=lambda result: result.score,
            reverse=True,
        )[: self._final_top_k]
