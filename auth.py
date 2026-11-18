"""User accounts: sign-up, login, password hashing and the four demo users.

Accounts live in ``data/users.json`` (written atomically). Passwords are stored as salted PBKDF2-HMAC-SHA256
hashes, compared in constant time, and failed logins are throttled per email address.

This is demo-grade authentication for a single local app. For real use put the app behind company SSO and map
directory groups to roles. Set ``HR_DEMO_ACCOUNTS=0`` to stop creating the demo users.
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
from datetime import datetime, timezone

import config
import roles

USERS_PATH = config.DATA_DIR / "users.json"
PBKDF2_ITERATIONS = 200_000
MIN_PASSWORD_LENGTH = 8
MAX_FAILED_LOGINS = 5
LOCKOUT_SECONDS = 300
DEMO_ENABLED = os.getenv("HR_DEMO_ACCOUNTS", "1") != "0"
DEMO_PASSWORD = os.getenv("HR_DEMO_PASSWORD", "Welcome@2026")  # demo accounts only; override in any shared setup

_EMAIL = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+$")
_lock = threading.RLock()
_failures: dict[str, list[float]] = {}  # email -> timestamps of recent failed logins

DEMO_USERS = [
    ("priya.raman@meridianlabs.example", "Priya Raman", "HR Operations Administrator", roles.ADMIN),
    ("daniel.whitfield@meridianlabs.example", "Daniel Whitfield", "Director of Engineering", roles.DIRECTOR),
    ("meera.iyer@meridianlabs.example", "Meera Iyer", "Senior Software Engineer", roles.SENIOR),
    ("arjun.kapoor@meridianlabs.example", "Arjun Kapoor", "Software Engineer", roles.EMPLOYEE),
]


class AuthError(ValueError):
    """A problem the user can fix; the message is safe to show on screen."""


# --------------------------------------------------------------------------- hashing
def _hash(password: str, salt: bytes, iterations: int | None = None) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations or PBKDF2_ITERATIONS)


def _encode(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _new_credentials(password: str) -> dict:
    salt = secrets.token_bytes(16)
    return {"salt": _encode(salt), "hash": _encode(_hash(password, salt)), "iterations": PBKDF2_ITERATIONS}


def _password_matches(user: dict, password: str) -> bool:
    salt = base64.b64decode(user["salt"])
    expected = base64.b64decode(user["hash"])
    return hmac.compare_digest(_hash(password, salt, int(user.get("iterations", PBKDF2_ITERATIONS))), expected)


# --------------------------------------------------------------------------- storage
def _read() -> dict[str, dict]:
    if not USERS_PATH.exists():
        return {}
    try:
        return json.loads(USERS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        config.logger.exception("Could not read %s", USERS_PATH)
        return {}


def _write(users: dict[str, dict]) -> None:
    USERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp = USERS_PATH.with_suffix(".tmp")
    temp.write_text(json.dumps(users, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(USERS_PATH)


def _public(user: dict) -> dict:
    """The fields the app keeps in the session: never the password hash."""
    return {key: user[key] for key in ("email", "name", "title", "role")}


def normalize_email(email: str) -> str:
    return (email or "").strip().lower()


# --------------------------------------------------------------------------- seeding
def ensure_demo_users() -> None:
    """Create any missing demo account. Existing accounts (and changed passwords or roles) are left alone."""
    if not DEMO_ENABLED:
        return
    with _lock:
        users = _read()
        changed = False
        for email, name, title, role in DEMO_USERS:
            if email not in users:
                users[email] = {
                    "email": email,
                    "name": name,
                    "title": title,
                    "role": role,
                    "demo": True,
                    "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    **_new_credentials(DEMO_PASSWORD),
                }
                changed = True
        if changed:
            _write(users)


def demo_accounts() -> list[dict]:
    """The demo users that currently exist, for the login page."""
    if not DEMO_ENABLED:
        return []
    users = _read()
    return [_public(users[email]) for email, *_ in DEMO_USERS if email in users]


# --------------------------------------------------------------------------- validation
def validate_password(password: str) -> None:
    if len(password) < MIN_PASSWORD_LENGTH:
        raise AuthError(f"Use at least {MIN_PASSWORD_LENGTH} characters for the password.")
    if not re.search(r"[A-Za-z]", password) or not re.search(r"\d", password):
        raise AuthError("The password must contain at least one letter and one number.")


# --------------------------------------------------------------------------- public API
def register(name: str, email: str, password: str, confirm: str) -> dict:
    """Create an Employee account (the role of every self sign-up) and return the session user."""
    name = " ".join((name or "").split())
    email = normalize_email(email)
    if len(name) < 2:
        raise AuthError("Enter your full name.")
    if not _EMAIL.match(email):
        raise AuthError("Enter a valid email address.")
    validate_password(password)
    if password != confirm:
        raise AuthError("The two passwords do not match.")
    with _lock:
        users = _read()
        if email in users:
            raise AuthError("An account with this email already exists. Sign in instead.")
        users[email] = {
            "email": email,
            "name": name,
            "title": "Employee",
            "role": roles.DEFAULT_ROLE,
            "demo": False,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **_new_credentials(password),
        }
        _write(users)
        return _public(users[email])


def _locked_for(email: str) -> int:
    """Seconds left on the lockout for ``email`` (0 when it may try)."""
    now = time.time()
    recent = [t for t in _failures.get(email, []) if now - t < LOCKOUT_SECONDS]
    _failures[email] = recent
    if len(recent) >= MAX_FAILED_LOGINS:
        return int(LOCKOUT_SECONDS - (now - recent[0])) + 1
    return 0


def authenticate(email: str, password: str, *, trusted: bool = False) -> dict:
    """Return the session user for valid credentials, otherwise raise ``AuthError``.

    ``trusted`` skips the password check for demo accounts only; it backs the demo-account shortcut buttons.
    """
    email = normalize_email(email)
    with _lock:
        wait = _locked_for(email)
        if wait:
            raise AuthError(f"Too many failed attempts. Try again in {(wait + 59) // 60} minute(s).")
        user = _read().get(email)
        shortcut = trusted and DEMO_ENABLED and user is not None and user.get("demo") is True
        valid = user is not None and (shortcut or _password_matches(user, password or ""))
        if not valid:
            if user is None:
                _hash(password or "", b"0" * 16)  # similar timing whether or not the account exists
            _failures.setdefault(email, []).append(time.time())
            raise AuthError("Incorrect email or password.")
        _failures.pop(email, None)
        return _public(user)


def get_user(email: str) -> dict | None:
    user = _read().get(normalize_email(email))
    return _public(user) if user else None


def list_users() -> list[dict]:
    return sorted((_public(u) for u in _read().values()), key=lambda u: (u["role"], u["name"]))


def set_role(actor_role: str | None, email: str, role: str) -> dict:
    """Change a user's role. Admin only, enforced here as well as in the UI."""
    roles.require(actor_role, roles.MANAGE_USERS)
    if role not in roles.ROLES:
        raise AuthError("Unknown role.")
    email = normalize_email(email)
    with _lock:
        users = _read()
        if email not in users:
            raise AuthError("No such user.")
        if users[email]["role"] == roles.ADMIN and role != roles.ADMIN:
            if sum(1 for u in users.values() if u["role"] == roles.ADMIN) <= 1:
                raise AuthError("At least one administrator must remain.")
        users[email]["role"] = role
        _write(users)
        return _public(users[email])
