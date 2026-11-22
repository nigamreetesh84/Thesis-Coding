"""Minimal role-based login for the bank portal (demo-grade, standard library only).

Roles:
* ``admin``    -- a bank employee; may upload and approve rules for ONE bank only.
* ``customer`` -- may ask for recommendations; nothing about them is stored.

Passwords are stored as salted PBKDF2-SHA256 hashes in ``config/portal_users.json``
(created with the demo accounts below on first use). Login returns an
HMAC-signed bearer token that expires after ``TOKEN_TTL_SECONDS``.

This is deliberately simple: it demonstrates role separation for the thesis
prototype, it is not production authentication (no lockout, no rotation).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from dataclasses import dataclass
from pathlib import Path

PBKDF2_ITERATIONS = 200_000
TOKEN_TTL_SECONDS = 8 * 60 * 60
BANKS = ("SBI", "HDFC", "AXIS")
BANK_DISPLAY = {"SBI": "SBI", "HDFC": "HDFC", "AXIS": "Axis"}

# Demo accounts (documented in PORTAL_AND_FL.md). Change them with
# `python -m src.portal.auth set-password <user> <password>`.
DEMO_USERS: dict[str, tuple[str, str, str | None]] = {
    "admin_sbi": ("sbi@123", "admin", "SBI"),
    "admin_hdfc": ("hdfc@123", "admin", "HDFC"),
    "admin_axis": ("axis@123", "admin", "AXIS"),
    "customer": ("customer@123", "customer", None),
}

DEFAULT_USERS_PATH = Path("config/portal_users.json")


@dataclass(frozen=True)
class User:
    username: str
    role: str  # "admin" | "customer"
    bank: str | None  # bank code for admins, None for customers


def _hash_password(password: str, salt: bytes) -> str:
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return base64.b64encode(digest).decode("ascii")


def _record(password: str, role: str, bank: str | None) -> dict[str, str | None]:
    salt = secrets.token_bytes(16)
    return {"salt": base64.b64encode(salt).decode("ascii"), "hash": _hash_password(password, salt), "role": role, "bank": bank}


class UserStore:
    """Users and their password hashes, kept in a small JSON file."""

    def __init__(self, path: str | Path = DEFAULT_USERS_PATH) -> None:
        self.path = Path(path)
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            users = {name: _record(pw, role, bank) for name, (pw, role, bank) in DEMO_USERS.items()}
            self.path.write_text(json.dumps(users, indent=2), encoding="utf-8")
        self._users: dict[str, dict[str, str | None]] = json.loads(self.path.read_text(encoding="utf-8"))

    def verify(self, username: str, password: str) -> User | None:
        rec = self._users.get(username)
        if rec is None:
            # same work as a real check, so timing does not reveal which usernames exist
            _hash_password(password, b"0" * 16)
            return None
        salt = base64.b64decode(str(rec["salt"]))
        if not hmac.compare_digest(_hash_password(password, salt), str(rec["hash"])):
            return None
        return User(username=username, role=str(rec["role"]), bank=rec.get("bank"))

    def set_password(self, username: str, password: str, role: str | None = None, bank: str | None = None) -> None:
        old = self._users.get(username, {})
        self._users[username] = _record(password, role or str(old.get("role", "customer")), bank or old.get("bank"))
        self.path.write_text(json.dumps(self._users, indent=2), encoding="utf-8")


class TokenSigner:
    """HMAC-signed bearer tokens: base64(payload).signature."""

    def __init__(self, secret: str | None = None) -> None:
        self.secret = (secret or os.environ.get("PORTAL_SECRET") or secrets.token_hex(32)).encode("utf-8")

    def issue(self, user: User, now: float | None = None) -> str:
        payload = {"u": user.username, "r": user.role, "b": user.bank, "exp": int((now or time.time()) + TOKEN_TTL_SECONDS)}
        body = base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode("utf-8")).decode("ascii")
        return f"{body}.{self._sign(body)}"

    def verify(self, token: str, now: float | None = None) -> User | None:
        try:
            body, sig = token.rsplit(".", 1)
        except ValueError:
            return None
        if not hmac.compare_digest(self._sign(body), sig):
            return None
        try:
            payload = json.loads(base64.urlsafe_b64decode(body.encode("ascii")))
        except (ValueError, json.JSONDecodeError):
            return None
        if int(payload.get("exp", 0)) < (now or time.time()):
            return None
        return User(username=str(payload["u"]), role=str(payload["r"]), bank=payload.get("b"))

    def _sign(self, body: str) -> str:
        return hmac.new(self.secret, body.encode("ascii"), hashlib.sha256).hexdigest()


def main() -> None:
    """`python -m src.portal.auth set-password <user> <password> [role] [bank]`"""
    import sys

    if len(sys.argv) >= 4 and sys.argv[1] == "set-password":
        role = sys.argv[4] if len(sys.argv) > 4 else None
        bank = sys.argv[5].upper() if len(sys.argv) > 5 else None
        UserStore().set_password(sys.argv[2], sys.argv[3], role, bank)
        print(f"Password set for {sys.argv[2]}")
    else:
        print(main.__doc__)


if __name__ == "__main__":
    main()
