from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path


DB_PATH = Path(__file__).parent / "app_users.db"


def _connect() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


def init_auth_db() -> None:
    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def signup_user(name: str, email: str, password: str) -> tuple[bool, str]:
    if not name.strip() or not email.strip() or not password:
        return False, "All signup fields are required."

    conn = _connect()
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO users(name, email, password_hash) VALUES(?, ?, ?)",
            (name.strip(), email.strip().lower(), _hash_password(password)),
        )
        conn.commit()
        return True, "Signup successful. Please login."
    except sqlite3.IntegrityError:
        return False, "Email already registered. Please login."
    finally:
        conn.close()


def login_user(email: str, password: str) -> tuple[bool, str]:
    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        "SELECT name, password_hash FROM users WHERE email = ?",
        (email.strip().lower(),),
    )
    row = cur.fetchone()
    conn.close()

    if not row:
        return False, "No account found for this email."

    name, password_hash = row
    if _hash_password(password) != password_hash:
        return False, "Incorrect password."
    return True, name
