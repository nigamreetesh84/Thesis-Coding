"""Round 7b user management: store rules, API permissions, session revocation, audit trail."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from config.settings import settings
from src.api import app
from src.portal.audit import AuditLog
from src.portal.auth import (
    DEMO_USERS,
    LOCKOUT_SECONDS,
    MAX_FAILED_LOGINS,
    AuthError,
    TokenSigner,
    UserConflict,
    UserError,
    UserStore,
    generate_password,
    password_problems,
)

GOOD_PW = "Branch#Desk2026"


@pytest.fixture
def store(tmp_path: Path) -> UserStore:
    return UserStore(tmp_path / "users.json", audit=AuditLog(tmp_path / "audit.jsonl"))


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(settings, "use_stub_llm", True)
    with TestClient(app) as c:
        yield c


def login(client: TestClient, user: str, pw: str) -> dict[str, str]:
    r = client.post("/portal/login", json={"username": user, "password": pw})
    assert r.status_code == 200, r.text
    return {"Authorization": "Bearer " + r.json()["token"]}


def platform(client: TestClient) -> dict[str, str]:
    return login(client, "platform_admin", "platform@123")


# ---- store units --------------------------------------------------------------------------------
def test_demo_accounts_seeded_hashed_and_complete(store: UserStore, tmp_path: Path) -> None:
    raw = json.loads((tmp_path / "users.json").read_text())
    assert set(raw) == set(DEMO_USERS)
    for name, (pw, role, bank, _) in DEMO_USERS.items():
        assert pw not in json.dumps(raw[name])  # never stored in clear
        assert raw[name]["role"] == role and raw[name]["bank"] == bank and raw[name]["status"] == "active"
    assert store.summary()["by_bank"]["AXIS"] == {"admins": 1, "active_admins": 1}


def test_round7_file_is_migrated_and_gets_a_platform_admin(tmp_path: Path) -> None:
    old = {"admin_sbi": {"salt": "AAAAAAAAAAAAAAAAAAAAAA==", "hash": "x", "role": "admin", "bank": "SBI"}}
    path = tmp_path / "users.json"
    path.write_text(json.dumps(old))
    s = UserStore(path)
    assert {"admin_sbi", "platform_admin"} == set(json.loads(path.read_text()))
    assert s.public("admin_sbi")["status"] == "active" and s.public("admin_sbi")["full_name"]


def test_password_policy() -> None:
    assert password_problems(GOOD_PW) == []
    assert password_problems("short1!")  # too short
    assert password_problems("alllowercaseletters")  # one class only
    assert password_problems("Sbi_officer_2026", "sbi_officer")  # contains username
    assert password_problems("Password123")  # common (case-insensitive)
    for _ in range(20):
        assert password_problems(generate_password()) == []


def test_create_validates_role_bank_and_uniqueness(store: UserStore) -> None:
    rec, temp = store.create("platform_admin", "Priya.Axis", "admin", "axis", "Priya Sharma", "priya@axis.example",
                             "AX-1042")
    assert rec["username"] == "priya.axis" and rec["bank"] == "AXIS" and rec["must_change_password"]
    assert temp and password_problems(temp) == []
    with pytest.raises(UserConflict):
        store.create("platform_admin", "priya.axis", "customer", None, "Someone")
    with pytest.raises(UserConflict):  # same employee ID at the same bank
        store.create("platform_admin", "other.axis", "admin", "AXIS", "Other", employee_id="ax-1042")
    with pytest.raises(UserConflict):  # e-mail reused
        store.create("platform_admin", "dup.mail", "customer", None, "Dup", email="PRIYA@axis.example")
    with pytest.raises(UserError):  # bank admin without a bank
        store.create("platform_admin", "nobank", "admin", None, "No Bank", employee_id="1")
    with pytest.raises(UserError):  # bank admin without employee ID
        store.create("platform_admin", "noid", "admin", "SBI", "No Id")
    with pytest.raises(UserError):  # customers do not belong to a bank
        store.create("platform_admin", "cust.bank", "customer", "SBI", "C")
    with pytest.raises(UserError):  # unknown bank
        store.create("platform_admin", "icici.user", "admin", "ICICI", "I", employee_id="1")
    with pytest.raises(UserError):  # weak chosen password
        store.create("platform_admin", "weak.pw", "customer", None, "Weak", password="abc12345")
    with pytest.raises(UserError):  # markup in a name
        store.create("platform_admin", "xss.user", "customer", None, "<script>")
    with pytest.raises(UserError):  # bad username
        store.create("platform_admin", "9lives", "customer", None, "Nine")


def test_lockout_after_failed_logins_and_unlock(store: UserStore) -> None:
    for _ in range(MAX_FAILED_LOGINS - 1):
        with pytest.raises(AuthError, match="Wrong"):
            store.authenticate("admin_sbi", "nope")
    with pytest.raises(AuthError, match="locked"):
        store.authenticate("admin_sbi", "nope")
    with pytest.raises(AuthError, match="locked"):  # even the right password is refused while locked
        store.authenticate("admin_sbi", "sbi@123")
    assert store.public("admin_sbi")["locked"]
    import time
    assert store.authenticate("admin_sbi", "sbi@123", now=time.time() + LOCKOUT_SECONDS + 1).bank == "SBI"
    for _ in range(MAX_FAILED_LOGINS):
        with pytest.raises(AuthError):
            store.authenticate("admin_sbi", "nope")
    store.unlock("platform_admin", "admin_sbi")
    assert store.authenticate("admin_sbi", "sbi@123").username == "admin_sbi"


def test_guards_on_self_and_last_platform_admin(store: UserStore) -> None:
    with pytest.raises(UserError):
        store.set_status("platform_admin", "platform_admin", False)
    with pytest.raises(UserError):
        store.update("platform_admin", "platform_admin", role="customer")
    with pytest.raises(UserError):  # last active platform admin cannot be disabled by anyone
        store.set_status("someone_else", "platform_admin", False)
    store.create("platform_admin", "ops2", "platform_admin", None, "Second Operator", password=GOOD_PW)
    store.set_status("ops2", "platform_admin", False, "left the team")
    assert store.public("platform_admin")["status"] == "disabled"


def test_audit_logs_changes_without_personal_values_or_passwords(store: UserStore, tmp_path: Path) -> None:
    store.create("platform_admin", "rahul.hdfc", "admin", "HDFC", "Rahul Mehta", "rahul@hdfc.example", "HD-77",
                 password=GOOD_PW)
    store.update("platform_admin", "rahul.hdfc", email="rahul.m@hdfc.example", bank="SBI")
    store.reset_password("platform_admin", "rahul.hdfc")
    store.set_status("platform_admin", "rahul.hdfc", False, "moved role")
    log = AuditLog(tmp_path / "audit.jsonl")
    text = (tmp_path / "audit.jsonl").read_text()
    assert [e["event"] for e in log.entries()] == ["user_created", "user_updated", "password_reset", "user_disabled"]
    assert log.entries()[1]["data"]["changes"] == {"email": "changed", "bank": ["HDFC", "SBI"]}
    assert "rahul@hdfc" not in text and "Rahul Mehta" not in text and GOOD_PW not in text
    assert log.verify()["ok"]


def test_token_revoked_by_disable_reset_and_role_change(store: UserStore) -> None:
    signer = TokenSigner("test")
    store.create("platform_admin", "neha.sbi", "admin", "SBI", "Neha", employee_id="S1", password=GOOD_PW,
                 must_change_password=False)
    tok = signer.verify(signer.issue(store.authenticate("neha.sbi", GOOD_PW)))
    assert store.session_user(tok.username, tok.token_version) is not None
    store.update("platform_admin", "neha.sbi", bank="HDFC")
    assert store.session_user(tok.username, tok.token_version) is None  # bank changed -> old token dead
    tok = signer.verify(signer.issue(store.authenticate("neha.sbi", GOOD_PW)))
    store.reset_password("platform_admin", "neha.sbi", "Fresh#Start2026")
    assert store.session_user(tok.username, tok.token_version) is None


# ---- API -----------------------------------------------------------------------------------------------
def test_only_platform_admin_can_manage_users(client: TestClient) -> None:
    for user, pw in (("admin_sbi", "sbi@123"), ("customer", "customer@123")):
        h = login(client, user, pw)
        assert client.get("/portal/users", headers=h).status_code == 403
        assert client.post("/portal/users", headers=h, json={"username": "x.y.z", "full_name": "X",
                                                             "role": "customer"}).status_code == 403
    p = platform(client)
    data = client.get("/portal/users", headers=p).json()
    assert {u["username"] for u in data["users"]} == set(DEMO_USERS)
    assert all("hash" not in u and "salt" not in u for u in data["users"])
    assert client.get("/portal/users?bank=AXIS", headers=p).json()["users"][0]["username"] == "admin_axis"
    # separation of duties: the platform administrator cannot upload or approve rules
    assert client.get("/portal/admin/versions", headers=p).status_code == 403


def test_registered_bank_user_can_log_in_and_upload_for_own_bank_only(client: TestClient) -> None:
    p = platform(client)
    r = client.post("/portal/users", headers=p, json={"username": "kavya.axis", "full_name": "Kavya Rao",
                                                      "role": "admin", "bank": "AXIS", "employee_id": "AX-2001",
                                                      "email": "kavya@axis.example"})
    assert r.status_code == 201, r.text
    temp = r.json()["temporary_password"]
    # first login: must change password before anything else
    first = client.post("/portal/login", json={"username": "kavya.axis", "password": temp}).json()
    assert first["must_change_password"] and first["bank"] == "AXIS"
    h = {"Authorization": "Bearer " + first["token"]}
    assert client.get("/portal/admin/versions", headers=h).status_code == 403
    assert client.post("/portal/me/password", headers=h, json={"current_password": temp,
                                                               "new_password": "weak"}).status_code == 400
    r = client.post("/portal/me/password", headers=h, json={"current_password": temp, "new_password": GOOD_PW})
    assert r.status_code == 200 and not r.json()["must_change_password"]
    assert client.get("/portal/admin/versions", headers=h).status_code == 401  # old token revoked
    h = {"Authorization": "Bearer " + r.json()["token"]}
    versions = client.get("/portal/admin/versions", headers=h).json()
    assert versions["bank"] == "AXIS"
    up = client.post("/portal/admin/upload", headers=h, files={"file": ("axis.md", b"### Axis Bank ACE Credit Card"
                     b" \xe2\x80\x93 Income\nMinimum annual income Rs 6 lakh.\n", "text/markdown")})
    assert up.status_code == 200, up.text
    assert up.json()["meta"]["bank"] == "AXIS" and up.json()["meta"]["uploader"] == "kavya.axis"
    team = client.get("/portal/admin/team", headers=h).json()["users"]
    assert {u["username"] for u in team} == {"admin_axis", "kavya.axis"}


def test_disable_revokes_live_session_and_blocks_login(client: TestClient) -> None:
    p = platform(client)
    hdfc = login(client, "admin_hdfc", "hdfc@123")
    assert client.get("/portal/admin/versions", headers=hdfc).status_code == 200
    assert client.post("/portal/users/admin_hdfc/disable", headers=p, json={"reason": "left the bank"}).status_code == 200
    assert client.get("/portal/admin/versions", headers=hdfc).status_code == 401
    r = client.post("/portal/login", json={"username": "admin_hdfc", "password": "hdfc@123"})
    assert r.status_code == 401 and "disabled" in r.json()["detail"]
    assert client.post("/portal/users/admin_hdfc/enable", headers=p, json={}).status_code == 200
    login(client, "admin_hdfc", "hdfc@123")


def test_lockout_via_api_and_unlock_by_platform_admin(client: TestClient) -> None:
    for _ in range(MAX_FAILED_LOGINS):
        client.post("/portal/login", json={"username": "customer", "password": "bad"})
    r = client.post("/portal/login", json={"username": "customer", "password": "customer@123"})
    assert r.status_code == 401 and "locked" in r.json()["detail"]
    p = platform(client)
    assert client.get("/portal/users?status=locked", headers=p).json()["users"][0]["username"] == "customer"
    assert client.post("/portal/users/customer/unlock", headers=p).status_code == 200
    login(client, "customer", "customer@123")


def test_user_events_in_activity_feed_and_bank_scoped_audit(client: TestClient) -> None:
    p = platform(client)
    client.post("/portal/users", headers=p, json={"username": "amit.sbi", "full_name": "Amit", "role": "admin",
                                                  "bank": "SBI", "employee_id": "SB-9", "password": GOOD_PW})
    client.post("/portal/users", headers=p, json={"username": "new.customer", "full_name": "Cust",
                                                  "role": "customer", "password": GOOD_PW})
    act = client.get("/portal/users/activity", headers=p).json()
    assert act["verify"]["ok"] and [e["data"]["target"] for e in act["entries"]] == ["amit.sbi", "new.customer"]
    sbi = {e["data"].get("target") for e in client.get("/portal/admin/audit", headers=login(client, "admin_sbi",
                                                                                              "sbi@123")).json()["entries"]}
    hdfc = {e["data"].get("target") for e in client.get("/portal/admin/audit", headers=login(client, "admin_hdfc",
                                                                                               "hdfc@123")).json()["entries"]}
    assert "amit.sbi" in sbi and "amit.sbi" not in hdfc and "new.customer" not in sbi | hdfc


def test_reset_password_returns_temporary_password_once(client: TestClient) -> None:
    p = platform(client)
    r = client.post("/portal/users/admin_axis/reset-password", headers=p, json={})
    temp = r.json()["temporary_password"]
    assert r.status_code == 200 and temp and r.json()["user"]["must_change_password"]
    assert client.post("/portal/login", json={"username": "admin_axis", "password": "axis@123"}).status_code == 401
    assert client.post("/portal/login", json={"username": "admin_axis", "password": temp}).json()["must_change_password"]
    assert "temporary_password" not in json.dumps(client.get("/portal/users/admin_axis", headers=p).json())
    assert client.patch("/portal/users/ghost", headers=p, json={"full_name": "G"}).status_code == 404
