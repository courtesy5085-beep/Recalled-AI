from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class UserRecord:
    id: int
    username: str
    password_hash: str
    email: str | None
    created_at: str


@dataclass(frozen=True, slots=True)
class MemoryMetadata:
    title: str
    summary: str
    emotion: str
    tags: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class MemoryCreateRequest:
    user_id: int
    title: str
    content: str
    summary: str
    emotion: str
    tags: list[str]
    memory_type: str
    source_type: str
    created_at: str


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    id: int
    user_id: int
    title: str
    content: str
    summary: str
    emotion: str
    tags: list[str]
    memory_type: str
    source_type: str
    created_at: str
    index_status: str


@dataclass(frozen=True, slots=True)
class VectorChunk:
    chunk_id: str
    memory_id: int
    user_id: int
    chunk_index: int
    content: str
    embedding: list[float]
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    chunk_id: str
    memory_id: int
    user_id: int
    title: str
    emotion: str
    tags: list[str]
    content: str
    distance: float


@dataclass(frozen=True, slots=True)
class SearchResult:
    memory_id: int
    title: str
    excerpt: str
    emotion: str
    tags: list[str]
    score: float

