from __future__ import annotations

import sqlite3

from repositories.user_repository import UserRepository
from security.passwords import PasswordHasher


class AuthService:
    def __init__(
        self,
        user_repository: UserRepository,
        password_hasher: PasswordHasher,
    ) -> None:
        self._users = user_repository
        self._passwords = password_hasher

    def register(self, username: str, password: str, email: str | None) -> int:
        username = username.strip()

        if not username:
            raise ValueError("Username is required")
        if not password or len(password) < 8:
            raise ValueError("Password must be at least 8 characters")

        if self._users.exists(username):
            raise ValueError("Username already exists")

        password_hash = self._passwords.hash(password)

        try:
            return self._users.create(username, password_hash, email)
        except sqlite3.IntegrityError as exc:
            raise ValueError("Username already exists") from exc

    def authenticate(self, username: str, password: str) -> int | None:
        user = self._users.get_by_username(username)
        if user is None:
            return None

        if not self._passwords.verify(password, user.password_hash):
            return None

        return user.id

