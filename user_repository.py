from __future__ import annotations

import sqlite3
from datetime import datetime

from core.database import Database
from domain.models import UserRecord


class UserRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    def create(self, username: str, password_hash: str, email: str | None) -> int:
        with self._db.connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO users(username, password_hash, email, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    username.strip(),
                    password_hash,
                    email.strip() if email else None,
                    datetime.utcnow().isoformat(),
                ),
            )
            return int(cursor.lastrowid)

    def get_by_username(self, username: str) -> UserRecord | None:
        with self._db.connection() as conn:
            row = conn.execute(
                """
                SELECT id, username, password_hash, email, created_at
                FROM users
                WHERE username = ?
                """,
                (username.strip(),),
            ).fetchone()

        if not row:
            return None

        return UserRecord(
            id=row["id"],
            username=row["username"],
            password_hash=row["password_hash"],
            email=row["email"],
            created_at=row["created_at"],
        )

    def exists(self, username: str) -> bool:
        return self.get_by_username(username) is not None

