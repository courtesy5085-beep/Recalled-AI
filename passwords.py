from __future__ import annotations

import hashlib
import hmac
import os


class PasswordHasher:
    _ALGORITHM = "pbkdf2_sha256"
    _ITERATIONS = 390_000
    _SALT_BYTES = 16

    def hash(self, password: str) -> str:
        if not password:
            raise ValueError("Password cannot be empty")

        salt = os.urandom(self._SALT_BYTES)
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            self._ITERATIONS,
        )
        return (
            f"{self._ALGORITHM}"
            f"${self._ITERATIONS}"
            f"${salt.hex()}"
            f"${digest.hex()}"
        )

    def verify(self, password: str, encoded: str) -> bool:
        try:
            algorithm, iterations, salt_hex, expected_hex = encoded.split("$", 3)
            if algorithm != self._ALGORITHM:
                return False

            candidate = hashlib.pbkdf2_hmac(
                "sha256",
                password.encode("utf-8"),
                bytes.fromhex(salt_hex),
                int(iterations),
            )
            return hmac.compare_digest(candidate.hex(), expected_hex)
        except Exception:
            return False

