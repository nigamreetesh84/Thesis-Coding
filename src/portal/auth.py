"""Accounts, roles and sessions for the bank portal (standard library only).

Roles
-----
* ``platform_admin`` -- runs the portal; registers and manages bank users and
  customers. Cannot upload or approve rules (separation of duties).
* ``admin``          -- a bank employee; may upload and approve rules for ONE bank only.
* ``customer``       -- may ask for recommendations; nothing about the request is stored.

Security controls (round 7b, user management)
---------------------------------------------
* Passwords: salted PBKDF2-SHA256 (200,000 iterations); never logged or returned.
* Password policy for every password set through the portal: at least 10
  characters, three of four character classes, must not contain the username,
  must not be a common password.
* Lock-out: 5 wrong passwords lock the account for 15 minutes (a platform
  administrator can unlock it earlier). Unknown usernames take the same time
  to reject, so timing does not reveal which usernames exist.
* Sessions: HMAC-signed bearer tokens that expire after 8 hours and carry a
  per-user *token version*. Disabling a user, resetting their password or
  changing their role or bank bumps the version, so every token they already
  hold stops working immediately (server-side revocation without a session table).
* First login: accounts created or reset by an administrator must choose a new
  password before they can do anything else.
* Accounts are deactivated, never deleted, so every past upload, approval and
  audit entry stays attributable to a known account.
* Every change is written to the hash-chained audit log (who, what, when). Only
  field NAMES are logged for personal details (name, e-mail, employee ID), never
  their values or any password.

Storage is a JSON file (``config/portal_users.json``) written atomically under a
lock. This is a prototype for a thesis, not a production identity provider:
there is no MFA, SSO or password history (see thesis, limitations).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

PBKDF2_ITERATIONS = 200_000
TOKEN_TTL_SECONDS = 8 * 60 * 60
MAX_FAILED_LOGINS = 5
LOCKOUT_SECONDS = 15 * 60
MIN_PASSWORD_LENGTH = 10

BANKS = ("SBI", "HDFC", "AXIS")
BANK_DISPLAY = {"SBI": "SBI", "HDFC": "HDFC", "AXIS": "Axis"}
ROLES = ("platform_admin", "admin", "customer")
ROLE_LABELS = {"platform_admin": "Platform administrator", "admin": "Bank administrator", "customer": "Customer"}
STATUSES = ("active", "disabled")

USERNAME_RE = re.compile(r"^[a-z][a-z0-9_.-]{2,31}$")
EMAIL_RE = re.compile(r"^[^@\s]{1,64}@[^@\s]+\.[A-Za-z]{2,}$")
EMPLOYEE_ID_RE = re.compile(r"^[A-Za-z0-9/_-]{1,32}$")
COMMON_PASSWORDS = {
    "password", "password1", "password123", "passw0rd", "12345678", "123456789", "1234567890", "qwerty123",
    "qwertyuiop", "iloveyou", "admin123", "admin@123", "welcome1", "welcome123", "welcome@123", "letmein123",
    "india@123", "india123", "abc@1234", "abcd@1234", "changeme", "changeme123",
}

# Demo accounts (documented in PORTAL_AND_FL.md). They are created on first start
# without the password policy so the documented demo passwords keep working.
DEMO_USERS: dict[str, tuple[str, str, str | None, str]] = {
    "platform_admin": ("platform@123", "platform_admin", None, "Portal Operations"),
    "admin_sbi": ("sbi@123", "admin", "SBI", "SBI Card Policy Desk"),
    "admin_hdfc": ("hdfc@123", "admin", "HDFC", "HDFC Card Policy Desk"),
    "admin_axis": ("axis@123", "admin", "AXIS", "Axis Card Policy Desk"),
    "customer": ("customer@123", "customer", None, "Demo Customer"),
}

DEMO_EMPLOYEE_IDS = {"admin_sbi": "SBI-DEMO-001", "admin_hdfc": "HDFC-DEMO-001", "admin_axis": "AXIS-DEMO-001"}

DEFAULT_USERS_PATH = Path("config/portal_users.json")
PUBLIC_FIELDS = ("username", "full_name", "email", "employee_id", "role", "bank", "status", "must_change_password",
                 "created_at", "created_by", "updated_at", "updated_by", "last_login", "password_changed_at",
                 "failed_attempts", "locked_until", "disabled_reason")


class AuditSink(Protocol):
    def append(self, event: str, data: dict[str, Any], actor: str) -> dict[str, Any]: ...


@dataclass(frozen=True)
class User:
    username: str
    role: str  # "platform_admin" | "admin" | "customer"
    bank: str | None  # bank code for bank administrators, None otherwise
    token_version: int = 0
    must_change_password: bool = False
    full_name: str = ""


class UserError(ValueError):
    """A request the store refuses (shown to the administrator). ``status`` is the HTTP code."""

    status = 400


class UserNotFound(UserError):
    status = 404


class UserConflict(UserError):
    status = 409


class AuthError(Exception):
    """Login refused; the message is safe to show."""


# ---- helpers ----------------------------------------------------------------------------------
def _now_iso(ts: float | None = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(ts if ts is not None else time.time()))


def _hash_password(password: str, salt: bytes) -> str:
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return base64.b64encode(digest).decode("ascii")


def _secret_fields(password: str) -> dict[str, str]:
    salt = secrets.token_bytes(16)
    return {"salt": base64.b64encode(salt).decode("ascii"), "hash": _hash_password(password, salt)}


def password_problems(password: str, username: str = "") -> list[str]:
    """Reasons a password breaks the policy (empty list = acceptable)."""
    problems: list[str] = []
    if len(password) < MIN_PASSWORD_LENGTH:
        problems.append(f"at least {MIN_PASSWORD_LENGTH} characters")
    if len(password) > 128:
        problems.append("at most 128 characters")
    classes = sum([any(c.islower() for c in password), any(c.isupper() for c in password),
                   any(c.isdigit() for c in password), any(not c.isalnum() for c in password)])
    if classes < 3:
        problems.append("at least three of: lower-case, upper-case, digit, symbol")
    if username and username.lower() in password.lower():
        problems.append("must not contain the username")
    if password.lower() in COMMON_PASSWORDS:
        problems.append("too common")
    return problems


def generate_password(length: int = 14) -> str:
    """A random temporary password that always meets the policy (no look-alike characters)."""
    lower, upper, digits, symbols = "abcdefghjkmnpqrstuvwxyz", "ABCDEFGHJKLMNPQRSTUVWXYZ", "23456789", "@#$%&*!?"
    pool = lower + upper + digits + symbols
    while True:
        chars = [secrets.choice(s) for s in (lower, upper, digits, symbols)]
        chars += [secrets.choice(pool) for _ in range(length - 4)]
        secrets.SystemRandom().shuffle(chars)
        pw = "".join(chars)
        if not password_problems(pw):
            return pw


def _clean(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if any(c in text for c in "<>") or any(ord(c) < 32 for c in text):
        raise UserError("Names and IDs may not contain < > or control characters")
    if len(text) > limit:
        raise UserError(f"Value longer than {limit} characters")
    return text


class UserStore:
    """Portal accounts kept in a JSON file; every change is audited when an audit sink is given."""

    def __init__(self, path: str | Path = DEFAULT_USERS_PATH, audit: AuditSink | None = None) -> None:
        self.path = Path(path)
        self.audit = audit
        self._lock = threading.RLock()
        if self.path.exists():
            self._users: dict[str, dict[str, Any]] = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self._users = {}
        changed = self._migrate()
        if changed or not self.path.exists():
            self._save()

    # ---- persistence ----------------------------------------------------------------------------
    def _migrate(self) -> bool:
        """Seed demo accounts on first start and add new fields to files from earlier rounds."""
        changed = False
        now = _now_iso()
        fresh = not self._users
        has_platform_admin = any(r.get("role") == "platform_admin" for r in self._users.values())
        for name, (pw, role, bank, full_name) in DEMO_USERS.items():
            # a new install gets every demo account; an install from round 7 gets only the platform administrator
            if name not in self._users and (fresh or (role == "platform_admin" and not has_platform_admin)):
                self._users[name] = {**_secret_fields(pw), "role": role, "bank": bank, "full_name": full_name,
                                     "created_at": now, "created_by": "system (demo seed)"}
                changed = True
            if name in self._users and not self._users[name].get("full_name"):
                self._users[name]["full_name"] = full_name
                changed = True
            if name in self._users and name in DEMO_EMPLOYEE_IDS and not self._users[name].get("employee_id"):
                self._users[name]["employee_id"] = DEMO_EMPLOYEE_IDS[name]
                changed = True
        defaults: dict[str, Any] = {"full_name": "", "email": "", "employee_id": "", "status": "active",
                                    "must_change_password": False, "created_at": now, "created_by": "system (migrated)",
                                    "updated_at": None, "updated_by": None, "last_login": None,
                                    "password_changed_at": None, "failed_attempts": 0, "locked_until": None,
                                    "disabled_reason": "", "token_version": 0}
        for rec in self._users.values():
            for key, value in defaults.items():
                if key not in rec:
                    rec[key] = value
                    changed = True
        return changed

    def _save(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(json.dumps(self._users, indent=2, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, self.path)

    def _audit(self, event: str, actor: str, target: str, **data: Any) -> None:
        if self.audit is not None:
            rec = self._users.get(target, {})
            self.audit.append(event, {"scope": "users", "target": target, "role": rec.get("role"),
                                      "bank": rec.get("bank"), **data}, actor)

    def _rec(self, username: str) -> dict[str, Any]:
        rec = self._users.get(username)
        if rec is None:
            raise UserNotFound(f"No user called {username!r}")
        return rec

    @staticmethod
    def _user(username: str, rec: dict[str, Any]) -> User:
        return User(username=username, role=str(rec["role"]), bank=rec.get("bank"),
                    token_version=int(rec.get("token_version", 0)),
                    must_change_password=bool(rec.get("must_change_password")), full_name=str(rec.get("full_name", "")))

    def public(self, username: str, now: float | None = None) -> dict[str, Any]:
        rec = self._rec(username)
        out = {k: rec.get(k) for k in PUBLIC_FIELDS}
        out["username"] = username
        locked_until = rec.get("locked_until")
        out["locked"] = bool(locked_until and float(locked_until) > (now or time.time()))
        out["locked_until"] = _now_iso(float(locked_until)) if out["locked"] else None
        return out

    # ---- login and sessions ------------------------------------------------------------------------
    def authenticate(self, username: str, password: str, now: float | None = None) -> User:
        """Check a login. Raises AuthError with a message that is safe to show."""
        now = now or time.time()
        username = username.strip().lower()
        with self._lock:
            rec = self._users.get(username)
            if rec is None:
                _hash_password(password, b"0" * 16)  # same work as a real check (no username probing)
                raise AuthError("Wrong username or password")
            locked_until = rec.get("locked_until")
            if locked_until and float(locked_until) > now:
                minutes = max(1, round((float(locked_until) - now) / 60))
                raise AuthError(f"Account locked after {MAX_FAILED_LOGINS} wrong passwords. Try again in about "
                                f"{minutes} minute(s) or ask the platform administrator to unlock it.")
            salt = base64.b64decode(str(rec["salt"]))
            if not hmac.compare_digest(_hash_password(password, salt), str(rec["hash"])):
                rec["failed_attempts"] = int(rec.get("failed_attempts", 0)) + 1
                if rec["failed_attempts"] >= MAX_FAILED_LOGINS:
                    rec["locked_until"] = now + LOCKOUT_SECONDS
                    rec["failed_attempts"] = 0
                    self._save()
                    self._audit("account_locked", "system", username, failed_attempts=MAX_FAILED_LOGINS,
                                lock_minutes=LOCKOUT_SECONDS // 60)
                    raise AuthError(f"Account locked for {LOCKOUT_SECONDS // 60} minutes after "
                                    f"{MAX_FAILED_LOGINS} wrong passwords.")
                self._save()
                raise AuthError("Wrong username or password")
            if rec.get("status") != "active":
                raise AuthError("This account is disabled. Contact the platform administrator.")
            rec.update(failed_attempts=0, locked_until=None, last_login=_now_iso(now))
            self._save()
            return self._user(username, rec)

    def verify(self, username: str, password: str) -> User | None:
        """Backward-compatible check (None instead of an exception)."""
        try:
            return self.authenticate(username, password)
        except AuthError:
            return None

    def session_user(self, username: str, token_version: int) -> User | None:
        """The live account behind a token, or None if it was disabled, removed or revoked."""
        rec = self._users.get(username)
        if rec is None or rec.get("status") != "active" or int(rec.get("token_version", 0)) != token_version:
            return None
        return self._user(username, rec)

    # ---- queries ------------------------------------------------------------------------------------
    def list_users(self, role: str | None = None, bank: str | None = None, status: str | None = None,
                   query: str = "") -> list[dict[str, Any]]:
        q = query.strip().lower()
        out = []
        for name in sorted(self._users):
            u = self.public(name)
            if role and u["role"] != role:
                continue
            if bank and u["bank"] != bank:
                continue
            if status == "locked":
                if not u["locked"]:
                    continue
            elif status and u["status"] != status:
                continue
            if q and not any(q in str(u.get(f) or "").lower() for f in ("username", "full_name", "email", "employee_id")):
                continue
            out.append(u)
        return out

    def summary(self) -> dict[str, Any]:
        users = self.list_users()
        per_bank = {b: {"admins": 0, "active_admins": 0} for b in BANKS}
        for u in users:
            if u["role"] == "admin" and u["bank"] in per_bank:
                per_bank[u["bank"]]["admins"] += 1
                per_bank[u["bank"]]["active_admins"] += u["status"] == "active"
        return {"total": len(users), "active": sum(u["status"] == "active" for u in users),
                "disabled": sum(u["status"] == "disabled" for u in users), "locked": sum(u["locked"] for u in users),
                "must_change_password": sum(bool(u["must_change_password"]) for u in users),
                "by_role": {r: sum(u["role"] == r for u in users) for r in ROLES}, "by_bank": per_bank}

    # ---- validation ---------------------------------------------------------------------------------
    def _check_role_bank(self, role: str, bank: str | None) -> str | None:
        if role not in ROLES:
            raise UserError(f"Role must be one of {', '.join(ROLES)}")
        if role == "admin":
            bank = (bank or "").upper()
            if bank not in BANKS:
                raise UserError(f"A bank administrator needs a bank: {', '.join(BANKS)}")
            return bank
        if bank:
            raise UserError("Only bank administrators belong to a bank")
        return None

    def _check_profile(self, username: str, email: str, employee_id: str, bank: str | None) -> None:
        if email and not EMAIL_RE.match(email):
            raise UserError("E-mail address is not valid")
        if employee_id and not EMPLOYEE_ID_RE.match(employee_id):
            raise UserError("Employee ID may use letters, digits, / _ - only (max 32)")
        for other, rec in self._users.items():
            if other == username:
                continue
            if email and str(rec.get("email", "")).lower() == email.lower():
                raise UserConflict("Another account already uses this e-mail address")
            if employee_id and bank and rec.get("bank") == bank and str(rec.get("employee_id", "")).lower() == employee_id.lower():
                raise UserConflict(f"Employee ID {employee_id} is already registered at {bank}")

    def _check_password(self, password: str, username: str) -> None:
        problems = password_problems(password, username)
        if problems:
            raise UserError("Password needs " + "; ".join(problems))

    def _active_platform_admins(self) -> list[str]:
        return [n for n, r in self._users.items() if r["role"] == "platform_admin" and r.get("status") == "active"]

    def _guard_last_platform_admin(self, username: str) -> None:
        if self._users[username]["role"] == "platform_admin" and self._active_platform_admins() == [username]:
            raise UserError("This is the last active platform administrator; create another one first")

    # ---- changes (administrator) ------------------------------------------------------------------
    def create(self, actor: str, username: str, role: str, bank: str | None = None, full_name: str = "",
               email: str = "", employee_id: str = "", password: str | None = None,
               must_change_password: bool = True) -> tuple[dict[str, Any], str | None]:
        """Register an account. Returns (public record, generated temporary password or None)."""
        username = username.strip().lower()
        if not USERNAME_RE.match(username):
            raise UserError("Username: 3-32 characters, start with a letter, then a-z 0-9 . _ -")
        with self._lock:
            if username in self._users:
                raise UserConflict(f"Username {username!r} is already taken")
            bank = self._check_role_bank(role, bank)
            full_name, email, employee_id = _clean(full_name, 80), _clean(email, 120), _clean(employee_id, 32)
            if not full_name:
                raise UserError("Full name is required")
            if role == "admin" and not employee_id:
                raise UserError("A bank administrator needs the bank's employee ID")
            self._check_profile(username, email, employee_id, bank)
            generated = None
            if not password:
                password = generated = generate_password()
            self._check_password(password, username)
            now = _now_iso()
            self._users[username] = {**_secret_fields(password), "role": role, "bank": bank, "full_name": full_name,
                                     "email": email, "employee_id": employee_id, "status": "active",
                                     "must_change_password": bool(must_change_password), "created_at": now,
                                     "created_by": actor, "updated_at": None, "updated_by": None, "last_login": None,
                                     "password_changed_at": now, "failed_attempts": 0, "locked_until": None,
                                     "disabled_reason": "", "token_version": 0}
            self._save()
            self._audit("user_created", actor, username, must_change_password=bool(must_change_password),
                        profile_fields=[f for f, v in (("full_name", full_name), ("email", email),
                                                       ("employee_id", employee_id)) if v])
            return self.public(username), generated

    def update(self, actor: str, username: str, full_name: str | None = None, email: str | None = None,
               employee_id: str | None = None, role: str | None = None, bank: str | None = None) -> dict[str, Any]:
        with self._lock:
            rec = self._rec(username)
            new_role = role or rec["role"]
            if new_role != rec["role"] and actor == username:
                raise UserError("You cannot change your own role")
            if new_role != rec["role"]:
                self._guard_last_platform_admin(username)
            new_bank = self._check_role_bank(new_role, bank if bank is not None else
                                             (rec.get("bank") if new_role == "admin" else None))
            new = {"full_name": _clean(full_name, 80) if full_name is not None else rec["full_name"],
                   "email": _clean(email, 120) if email is not None else rec["email"],
                   "employee_id": _clean(employee_id, 32) if employee_id is not None else rec["employee_id"]}
            if not new["full_name"]:
                raise UserError("Full name is required")
            if new_role == "admin" and not new["employee_id"]:
                raise UserError("A bank administrator needs the bank's employee ID")
            self._check_profile(username, new["email"], new["employee_id"], new_bank)
            changes: dict[str, Any] = {}
            for field in ("full_name", "email", "employee_id"):
                if new[field] != rec[field]:
                    changes[field] = "changed"  # personal data: name of the field only
            if new_role != rec["role"]:
                changes["role"] = [rec["role"], new_role]
            if new_bank != rec.get("bank"):
                changes["bank"] = [rec.get("bank"), new_bank]
            if not changes:
                return self.public(username)
            rec.update(new, role=new_role, bank=new_bank, updated_at=_now_iso(), updated_by=actor)
            if "role" in changes or "bank" in changes:
                rec["token_version"] = int(rec.get("token_version", 0)) + 1  # permissions changed: re-login
            self._save()
            self._audit("user_updated", actor, username, changes=changes,
                        sessions_revoked="role" in changes or "bank" in changes)
            return self.public(username)

    def set_status(self, actor: str, username: str, active: bool, reason: str = "") -> dict[str, Any]:
        with self._lock:
            rec = self._rec(username)
            reason = _clean(reason, 200)
            if not active:
                if actor == username:
                    raise UserError("You cannot disable your own account")
                self._guard_last_platform_admin(username)
                if rec["status"] == "disabled":
                    return self.public(username)
                rec.update(status="disabled", disabled_reason=reason, token_version=int(rec.get("token_version", 0)) + 1)
                event = "user_disabled"
            else:
                if rec["status"] == "active":
                    return self.public(username)
                rec.update(status="active", disabled_reason="", failed_attempts=0, locked_until=None)
                event = "user_enabled"
            rec.update(updated_at=_now_iso(), updated_by=actor)
            self._save()
            self._audit(event, actor, username, reason=reason)
            return self.public(username)

    def reset_password(self, actor: str, username: str, new_password: str | None = None) -> tuple[dict[str, Any], str | None]:
        """Administrator reset: new temporary password, user must change it at next login."""
        with self._lock:
            rec = self._rec(username)
            if actor == username:
                raise UserError("Use 'Change my password' for your own account")
            generated = None
            if not new_password:
                new_password = generated = generate_password()
            self._check_password(new_password, username)
            rec.update(_secret_fields(new_password), must_change_password=True, failed_attempts=0, locked_until=None,
                       password_changed_at=_now_iso(), updated_at=_now_iso(), updated_by=actor,
                       token_version=int(rec.get("token_version", 0)) + 1)
            self._save()
            self._audit("password_reset", actor, username, sessions_revoked=True)
            return self.public(username), generated

    def unlock(self, actor: str, username: str) -> dict[str, Any]:
        with self._lock:
            rec = self._rec(username)
            rec.update(failed_attempts=0, locked_until=None, updated_at=_now_iso(), updated_by=actor)
            self._save()
            self._audit("user_unlocked", actor, username)
            return self.public(username)

    # ---- changes (own account) ----------------------------------------------------------------------
    def change_password(self, username: str, current: str, new: str) -> User:
        with self._lock:
            rec = self._rec(username)
            salt = base64.b64decode(str(rec["salt"]))
            if not hmac.compare_digest(_hash_password(current, salt), str(rec["hash"])):
                raise UserError("Current password is wrong")
            if current == new:
                raise UserError("The new password must be different from the current one")
            self._check_password(new, username)
            rec.update(_secret_fields(new), must_change_password=False, password_changed_at=_now_iso(),
                       token_version=int(rec.get("token_version", 0)) + 1)
            self._save()
            self._audit("password_changed", username, username)
            return self._user(username, rec)

    def set_password(self, username: str, password: str, role: str | None = None, bank: str | None = None) -> None:
        """Operator command-line reset (no policy, no audit actor); kept from round 7."""
        with self._lock:
            old = self._users.get(username, {})
            self._users[username] = {**old, **_secret_fields(password), "role": role or str(old.get("role", "customer")),
                                     "bank": bank or old.get("bank"), "token_version": int(old.get("token_version", 0)) + 1}
            self._migrate()
            self._save()


class TokenSigner:
    """HMAC-signed bearer tokens: base64(payload).signature."""

    def __init__(self, secret: str | None = None) -> None:
        self.secret = (secret or os.environ.get("PORTAL_SECRET") or secrets.token_hex(32)).encode("utf-8")

    def issue(self, user: User, now: float | None = None) -> str:
        payload = {"u": user.username, "r": user.role, "b": user.bank, "v": user.token_version,
                   "exp": int((now or time.time()) + TOKEN_TTL_SECONDS)}
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
        return User(username=str(payload["u"]), role=str(payload["r"]), bank=payload.get("b"),
                    token_version=int(payload.get("v", 0)))

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
