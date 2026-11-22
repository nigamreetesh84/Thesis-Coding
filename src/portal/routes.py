"""FastAPI routes for the bank portal (mounted by src/api.py under /portal)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.domain.entities import UserProfile
from src.domain.query_parsing import ParsedQuery
from src.portal.auth import TokenSigner, User, UserStore
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


def current_user(authorization: str = Header(default="")) -> User:
    token = authorization.removeprefix("Bearer ").strip()
    user = _state["signer"].verify(token) if token else None
    if user is None:
        raise HTTPException(401, "Please log in")
    return user  # type: ignore[no-any-return]


def admin_user(user: User = Depends(current_user)) -> User:
    if user.role != "admin" or not user.bank:
        raise HTTPException(403, "Bank administrators only")
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


@router.post("/login")
def login(body: LoginIn) -> dict[str, Any]:
    user = _state["users"].verify(body.username, body.password)
    if user is None:
        raise HTTPException(401, "Wrong username or password")
    return {"token": _state["signer"].issue(user), "username": user.username, "role": user.role, "bank": user.bank}


@router.get("/me")
def me(user: User = Depends(current_user)) -> dict[str, Any]:
    return {"username": user.username, "role": user.role, "bank": user.bank}


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
    own = [e for e in log.entries() if e["data"].get("bank") in (user.bank, None)
           or user.bank in (e["data"].get("rule_versions") or {})]
    return {"verify": log.verify(), "entries": own[-limit:]}


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
