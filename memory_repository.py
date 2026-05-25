from __future__ import annotations

from collections.abc import Sequence

from core.database import Database
from domain.models import MemoryCreateRequest, MemoryRecord


def _parse_tags(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


class MemoryRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    def create_memory_with_chunks(
        self,
        request: MemoryCreateRequest,
        chunk_texts: Sequence[str],
    ) -> int:
        tags_csv = ",".join(sorted(set(tag.strip().lower() for tag in request.tags if tag.strip())))

        with self._db.connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO memories(
                    user_id, title, content, summary, emotion, tags,
                    memory_type, source_type, created_at, index_status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')
                """,
                (
                    request.user_id,
                    request.title,
                    request.content,
                    request.summary,
                    request.emotion,
                    tags_csv,
                    request.memory_type,
                    request.source_type,
                    request.created_at,
                ),
            )
            memory_id = int(cursor.lastrowid)

            conn.executemany(
                """
                INSERT INTO memory_chunks(
                    id, memory_id, user_id, chunk_index, content, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        f"{memory_id}:{idx}",
                        memory_id,
                        request.user_id,
                        idx,
                        chunk,
                        request.created_at,
                    )
                    for idx, chunk in enumerate(chunk_texts)
                ],
            )

            return memory_id

    def update_index_status(self, memory_id: int, status: str) -> None:
        with self._db.connection() as conn:
            conn.execute(
                """
                UPDATE memories
                SET index_status = ?
                WHERE id = ?
                """,
                (status, memory_id),
            )

    def list_by_user(self, user_id: int) -> list[MemoryRecord]:
        with self._db.connection() as conn:
            rows = conn.execute(
                """
                SELECT id, user_id, title, content, summary, emotion, tags,
                       memory_type, source_type, created_at, index_status
                FROM memories
                WHERE user_id = ?
                ORDER BY created_at DESC
                """,
                (user_id,),
            ).fetchall()

        return [
            MemoryRecord(
                id=row["id"],
                user_id=row["user_id"],
                title=row["title"],
                content=row["content"],
                summary=row["summary"],
                emotion=row["emotion"],
                tags=_parse_tags(row["tags"]),
                memory_type=row["memory_type"],
                source_type=row["source_type"],
                created_at=row["created_at"],
                index_status=row["index_status"],
            )
            for row in rows
        ]
