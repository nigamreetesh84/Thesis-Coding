"""FastAPI routes for the bank portal (mounted by src/api.py under /portal)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.domain.entities import UserProfile
from src.domain.query_parsing import ParsedQuery
from src.portal.auth import (
    BANK_DISPLAY,
    BANKS,
    MAX_FAILED_LOGINS,
    MIN_PASSWORD_LENGTH,
    ROLE_LABELS,
    ROLES,
    AuthError,
    TokenSigner,
    User,
    UserError,
    UserStore,
    password_problems,
)
from src.portal.ingest import IngestError
from src.portal.registry import RegistryError, diff_requirements
from src.portal.service import PortalService

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
SAMPLES_DIR = Path("data/portal_samples")

router = APIRouter(prefix="/portal", tags=["bank portal"])
_state: dict[str, Any] = {}


def configure(service: PortalService, users: UserStore, signer: TokenSigner) -> None:
    _state.update(service=service, users=users, signer=signer)


def _service() -> PortalService:
    if "service" not in _state:
        raise HTTPException(503, "Portal not initialised")
    return _state["service"]  # type: ignore[no-any-return]


def _users() -> UserStore:
    if "users" not in _state:
        raise HTTPException(503, "Portal not initialised")
    return _state["users"]  # type: ignore[no-any-return]


def session_user(authorization: str = Header(default="")) -> User:
    """Any logged-in account, even one that still has to change its password."""
    token = authorization.removeprefix("Bearer ").strip()
    claimed = _state["signer"].verify(token) if token else None
    # the token must still match a live account: disabling, password resets and role
    # changes bump the account's token version and so revoke every older token
    user = _users().session_user(claimed.username, claimed.token_version) if claimed else None
    if user is None:
        raise HTTPException(401, "Please log in")
    return user


def current_user(user: User = Depends(session_user)) -> User:
    if user.must_change_password:
        raise HTTPException(403, "Password change required before using the portal")
    return user


def admin_user(user: User = Depends(current_user)) -> User:
    if user.role != "admin" or not user.bank:
        raise HTTPException(403, "Bank administrators only")
    return user


def platform_admin_user(user: User = Depends(current_user)) -> User:
    if user.role != "platform_admin":
        raise HTTPException(403, "Platform administrators only")
    return user


def customer_user(user: User = Depends(current_user)) -> User:
    if user.role != "customer":
        raise HTTPException(403, "Customers only")
    return user


def _bad_request(exc: Exception) -> HTTPException:
    return HTTPException(400, str(exc))


# ---- auth ----------------------------------------------------------------------------
class LoginIn(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=128)


def _session_out(user: User) -> dict[str, Any]:
    return {"token": _state["signer"].issue(user), "username": user.username, "role": user.role, "bank": user.bank,
            "full_name": user.full_name, "must_change_password": user.must_change_password}


@router.post("/login")
def login(body: LoginIn) -> dict[str, Any]:
    try:
        user = _users().authenticate(body.username, body.password)
    except AuthError as exc:
        raise HTTPException(401, str(exc)) from exc
    return _session_out(user)


@router.get("/me")
def me(user: User = Depends(session_user)) -> dict[str, Any]:
    return _users().public(user.username)


class ChangePasswordIn(BaseModel):
    current_password: str = Field(min_length=1, max_length=128)
    new_password: str = Field(min_length=1, max_length=128)


@router.post("/me/password")
def change_my_password(body: ChangePasswordIn, user: User = Depends(session_user)) -> dict[str, Any]:
    try:
        fresh = _users().change_password(user.username, body.current_password, body.new_password)
    except UserError as exc:
        raise HTTPException(exc.status, str(exc)) from exc
    return _session_out(fresh)  # older tokens are revoked; the caller continues with this one


@router.get("/banks")
def banks(user: User = Depends(current_user)) -> list[dict[str, Any]]:
    return _service().banks_overview()


# ---- bank administrator ------------------------------------------------------------------
@router.post("/admin/upload")
async def upload(file: UploadFile = File(...), card_names: str = Form(default=""),
                 user: User = Depends(admin_user)) -> dict[str, Any]:
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File larger than 10 MB")
    names = [c.strip() for c in card_names.split(",") if c.strip()]
    svc = _service()
    try:
        detail = svc.registry.create_draft(str(user.bank), data, file.filename or "upload.txt", user.username,
                                           svc.extractor, names)
    except (IngestError, RegistryError) as exc:
        raise _bad_request(exc) from exc
    return _with_preview(svc, str(user.bank), detail)


def _with_preview(svc: PortalService, bank: str, detail: dict[str, Any]) -> dict[str, Any]:
    active = svc.registry.active_version(bank)
    old = svc.registry.version_detail(bank, active)["approved"] if active else []
    new = detail["approved"] or detail["extracted"]
    return {**detail, "active_version": active, "changes_vs_active": diff_requirements(old or [], new)}


@router.get("/admin/versions")
def versions(user: User = Depends(admin_user)) -> dict[str, Any]:
    svc = _service()
    return {"bank": user.bank, "active_version": svc.registry.active_version(str(user.bank)),
            "versions": svc.registry.versions(str(user.bank))}


@router.get("/admin/versions/{version}")
def version_detail(version: str, user: User = Depends(admin_user)) -> dict[str, Any]:
    svc = _service()
    try:
        return _with_preview(svc, str(user.bank), svc.registry.version_detail(str(user.bank), version))
    except RegistryError as exc:
        raise _bad_request(exc) from exc


class ApproveIn(BaseModel):
    approved: list[dict[str, Any]]
    note: str = ""


@router.post("/admin/versions/{version}/approve")
def approve(version: str, body: ApproveIn, user: User = Depends(admin_user)) -> dict[str, Any]:
    try:
        return _service().registry.approve(str(user.bank), version, body.approved, user.username, body.note)
    except RegistryError as exc:
        raise _bad_request(exc) from exc


class RejectIn(BaseModel):
    note: str = ""


@router.post("/admin/versions/{version}/reject")
def reject(version: str, body: RejectIn, user: User = Depends(admin_user)) -> dict[str, Any]:
    try:
        return _service().registry.reject(str(user.bank), version, user.username, body.note)
    except RegistryError as exc:
        raise _bad_request(exc) from exc


@router.get("/admin/audit")
def audit(limit: int = 30, user: User = Depends(admin_user)) -> dict[str, Any]:
    log = _service().registry.audit

    def visible(e: dict[str, Any]) -> bool:
        d = e["data"]
        if d.get("scope") == "users":  # account events: only those about this bank's staff
            return d.get("bank") == user.bank
        return d.get("bank") in (user.bank, None) or user.bank in (d.get("rule_versions") or {})

    own = [e for e in log.entries() if visible(e)]
    return {"verify": log.verify(), "entries": own[-limit:]}


@router.get("/admin/team")
def bank_team(user: User = Depends(admin_user)) -> dict[str, Any]:
    """Read-only list of this bank's administrators (accounts are managed by the platform administrator)."""
    return {"bank": user.bank, "users": _users().list_users(role="admin", bank=user.bank)}


@router.get("/admin/samples/{name}")
def sample(name: str, user: User = Depends(admin_user)) -> FileResponse:
    path = SAMPLES_DIR / Path(name).name
    if not path.exists():
        raise HTTPException(404, "No such sample")
    return FileResponse(path, filename=path.name)


# ---- customer ---------------------------------------------------------------------------------
class PortalAskIn(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    banks: list[str]
    previous: dict[str, Any] | None = None


def _parsed_from_dict(data: dict[str, Any], banks: list[str]) -> ParsedQuery:
    return ParsedQuery(income=data.get("income"), age=data.get("age"), cibil=data.get("cibil"),
                       employment=data.get("employment"), existing_relationship=data.get("existing_relationship"),
                       banks=tuple(banks), preferences=str(data.get("preferences") or ""),
                       income_text=data.get("income_text"),
                       existing_banks=tuple(data["existing_banks"]) if data.get("existing_banks") else None)


@router.post("/ask")
def portal_ask(body: PortalAskIn, user: User = Depends(customer_user)) -> dict[str, Any]:
    previous = _parsed_from_dict(body.previous, body.banks) if body.previous else None
    try:
        return _service().ask(body.message, previous, body.banks, user.username)
    except RegistryError as exc:
        raise _bad_request(exc) from exc


class PortalProfileIn(BaseModel):
    annual_income: float = Field(ge=0)
    age: int = Field(ge=18, le=100)
    cibil: int = Field(ge=300, le=900)
    employment: Literal["salaried", "self-employed", "business owner"]
    existing_banks: list[str] = Field(default_factory=list)
    banks: list[str]


@router.post("/recommend")
def portal_recommend(body: PortalProfileIn, user: User = Depends(customer_user)) -> dict[str, Any]:
    profile = UserProfile(annual_income=body.annual_income, age=body.age, cibil=body.cibil,
                          employment=body.employment, existing_relationship=bool(body.existing_banks),
                          banks=body.banks)
    try:
        return _service().recommend_form(profile, body.existing_banks, user.username)
    except RegistryError as exc:
        raise _bad_request(exc) from exc


# ---- platform administrator: user management --------------------------------------------------
def _user_error(exc: UserError) -> HTTPException:
    return HTTPException(exc.status, str(exc))


@router.get("/users/meta")
def users_meta(user: User = Depends(platform_admin_user)) -> dict[str, Any]:
    """Choices and rules the user-management screen needs."""
    return {"roles": [{"code": r, "label": ROLE_LABELS[r]} for r in ROLES],
            "banks": [{"code": b, "name": BANK_DISPLAY[b]} for b in BANKS],
            "password_policy": {"min_length": MIN_PASSWORD_LENGTH, "classes": 3,
                                "text": f"At least {MIN_PASSWORD_LENGTH} characters with three of: lower-case, "
                                        "upper-case, digit, symbol; must not contain the username or be a common password."},
            "lockout": {"max_failed": MAX_FAILED_LOGINS}}


@router.get("/users")
def list_users(role: str | None = None, bank: str | None = None, status: str | None = None, q: str = "",
               user: User = Depends(platform_admin_user)) -> dict[str, Any]:
    users = _users()
    return {"summary": users.summary(), "users": users.list_users(role or None, bank or None, status or None, q)}


class UserCreateIn(BaseModel):
    username: str = Field(min_length=3, max_length=32)
    full_name: str = Field(min_length=1, max_length=80)
    role: Literal["platform_admin", "admin", "customer"]
    bank: str | None = None
    email: str = Field(default="", max_length=120)
    employee_id: str = Field(default="", max_length=32)
    password: str | None = Field(default=None, max_length=128)  # empty = generate a temporary password
    must_change_password: bool = True


@router.post("/users", status_code=201)
def create_user(body: UserCreateIn, user: User = Depends(platform_admin_user)) -> dict[str, Any]:
    try:
        record, temp = _users().create(user.username, body.username, body.role, body.bank, body.full_name, body.email,
                                       body.employee_id, body.password or None, body.must_change_password)
    except UserError as exc:
        raise _user_error(exc) from exc
    return {"user": record, "temporary_password": temp}


@router.get("/users/activity")
def users_activity(limit: int = 50, user: User = Depends(platform_admin_user)) -> dict[str, Any]:
    log = _service().registry.audit
    entries = [e for e in log.entries() if e["data"].get("scope") == "users"]
    return {"verify": log.verify(), "entries": entries[-limit:]}


@router.post("/users/check-password")
def check_password(body: dict[str, str], user: User = Depends(session_user)) -> dict[str, Any]:
    problems = password_problems(str(body.get("password", "")), str(body.get("username", "")))
    return {"ok": not problems, "problems": problems}


@router.get("/users/{username}")
def get_user(username: str, user: User = Depends(platform_admin_user)) -> dict[str, Any]:
    try:
        return _users().public(username)
    except UserError as exc:
        raise _user_error(exc) from exc


class UserUpdateIn(BaseModel):
    full_name: str | None = Field(default=None, max_length=80)
    email: str | None = Field(default=None, max_length=120)
    employee_id: str | None = Field(default=None, max_length=32)
    role: Literal["platform_admin", "admin", "customer"] | None = None
    bank: str | None = None


@router.patch("/users/{username}")
def update_user(username: str, body: UserUpdateIn, user: User = Depends(platform_admin_user)) -> dict[str, Any]:
    try:
        return _users().update(user.username, username, body.full_name, body.email, body.employee_id, body.role,
                               body.bank)
    except UserError as exc:
        raise _user_error(exc) from exc


class StatusIn(BaseModel):
    reason: str = Field(default="", max_length=200)


@router.post("/users/{username}/disable")
def disable_user(username: str, body: StatusIn, user: User = Depends(platform_admin_user)) -> dict[str, Any]:
    try:
        return _users().set_status(user.username, username, False, body.reason)
    except UserError as exc:
        raise _user_error(exc) from exc


@router.post("/users/{username}/enable")
def enable_user(username: str, body: StatusIn, user: User = Depends(platform_admin_user)) -> dict[str, Any]:
    try:
        return _users().set_status(user.username, username, True, body.reason)
    except UserError as exc:
        raise _user_error(exc) from exc


class ResetIn(BaseModel):
    password: str | None = Field(default=None, max_length=128)


@router.post("/users/{username}/reset-password")
def reset_password(username: str, body: ResetIn, user: User = Depends(platform_admin_user)) -> dict[str, Any]:
    try:
        record, temp = _users().reset_password(user.username, username, body.password or None)
    except UserError as exc:
        raise _user_error(exc) from exc
    return {"user": record, "temporary_password": temp}


@router.post("/users/{username}/unlock")
def unlock_user(username: str, user: User = Depends(platform_admin_user)) -> dict[str, Any]:
    try:
        return _users().unlock(user.username, username)
    except UserError as exc:
        raise _user_error(exc) from exc
