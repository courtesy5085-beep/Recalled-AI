from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from functools import lru_cache
from typing import Protocol

from openai import OpenAI
from sentence_transformers import CrossEncoder, SentenceTransformer

from domain.models import MemoryMetadata

logger = logging.getLogger(__name__)


class EmbeddingProvider(Protocol):
    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        ...


class Reranker(Protocol):
    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        ...


class MetadataGenerator(Protocol):
    def generate(self, text: str) -> MemoryMetadata:
        ...


@lru_cache(maxsize=2)
def load_embedding_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


@lru_cache(maxsize=2)
def load_reranker_model(model_name: str) -> CrossEncoder:
    return CrossEncoder(model_name)


class LocalEmbeddingProvider:
    def __init__(self, model_name: str) -> None:
        self._model = load_embedding_model(model_name)

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._model.encode(list(texts), convert_to_numpy=False).tolist()


class OpenAIEmbeddingProvider:
    def __init__(self, client: OpenAI, model: str) -> None:
        self._client = client
        self._model = model

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        response = self._client.embeddings.create(
            model=self._model,
            input=list(texts),
        )
        return [item.embedding for item in response.data]


class FallbackEmbeddingProvider:
    def __init__(
        self,
        primary: EmbeddingProvider,
        fallback: EmbeddingProvider,
    ) -> None:
        self._primary = primary
        self._fallback = fallback

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        try:
            return self._primary.embed_many(texts)
        except Exception:
            logger.exception("Primary embedding provider failed, using fallback")
            return self._fallback.embed_many(texts)


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        self._model = load_reranker_model(model_name)

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        if not documents:
            return []
        pairs = [(query, doc) for doc in documents]
        scores = self._model.predict(pairs)
        return [float(score) for score in scores]


class RuleBasedMetadataGenerator:
    def generate(self, text: str) -> MemoryMetadata:
        cleaned = re.sub(r"\s+", " ", text).strip()
        title = cleaned[:60] if cleaned else "Untitled"
        summary = cleaned[:160] if cleaned else ""
        return MemoryMetadata(
            title=title,
            summary=summary,
            emotion="neutral",
            tags=["memory"],
        )


class OpenAIMetadataGenerator:
    def __init__(
        self,
        client: OpenAI,
        model: str,
        fallback: MetadataGenerator,
    ) -> None:
        self._client = client
        self._model = model
        self._fallback = fallback

    def generate(self, text: str) -> MemoryMetadata:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Return compact JSON with keys: "
                            "title, summary, emotion, tags."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"""
Analyze the following memory and return JSON only.

Schema:
{{
  "title": "short title",
  "summary": "brief summary",
  "emotion": "single word emotion",
  "tags": ["tag1", "tag2"]
}}

Memory:
{text}
""",
                    },
                ],
            )
            payload = json.loads(response.choices[0].message.content)
            return self._normalize(payload, text)
        except Exception:
            logger.exception("Metadata generation failed, using fallback")
            return self._fallback.generate(text)

    @staticmethod
    def _normalize(payload: dict, text: str) -> MemoryMetadata:
        title = str(payload.get("title") or text[:60] or "Untitled").strip()[:120]
        summary = str(payload.get("summary") or text[:160] or "").strip()[:300]
        emotion = str(payload.get("emotion") or "neutral").strip().lower()[:30]

        raw_tags = payload.get("tags", [])
        if not isinstance(raw_tags, list):
            raw_tags = ["memory"]

        tags = []
        for tag in raw_tags[:10]:
            normalized = str(tag).strip().lower()
            if normalized:
                tags.append(normalized)

        return MemoryMetadata(
            title=title or "Untitled",
            summary=summary,
            emotion=emotion or "neutral",
            tags=tags or ["memory"],
  )

