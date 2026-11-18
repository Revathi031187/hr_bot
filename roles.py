"""Roles, permissions and the suggested questions shown to each role.

Roles only change what the *interface* offers (sidebar tools, suggested questions). Every restricted pipeline
operation also checks the role on the server side with ``require``, so hiding a button is never the only guard.
"""

from __future__ import annotations

from dataclasses import dataclass

ADMIN = "admin"
DIRECTOR = "director"
SENIOR = "senior"
EMPLOYEE = "employee"
DEFAULT_ROLE = EMPLOYEE  # what self sign-up creates; only an admin can change it

# Permissions an admin has and everyone else lacks.
MANAGE_KB = "manage_kb"  # upload / remove documents, re-ingest
VIEW_SYSTEM = "view_system"  # chunk counts, model status, "how it works"
VIEW_AUDIT = "view_audit"  # audit log
MANAGE_USERS = "manage_users"  # change roles

_ADMIN_ONLY = {MANAGE_KB, VIEW_SYSTEM, VIEW_AUDIT, MANAGE_USERS}


@dataclass(frozen=True)
class Role:
    key: str
    label: str
    tagline: str  # one line shown on the login cards and the welcome panel
    welcome: str
    starters: tuple[tuple[str, str], ...]  # (topic, question)


ROLES: dict[str, Role] = {
    ADMIN: Role(
        ADMIN,
        "Administrator",
        "Manage the knowledge base and monitor the system",
        "You can add or remove policy documents, check index health and review the audit log. "
        "Ask a few questions below to confirm that new documents are answered correctly.",
        (
            ("Check ingestion", "What is the parental leave duration?"),
            ("Check ingestion", "How many vacation days can I carry over?"),
            ("Check ingestion", "What is the Home Office Setup Allowance?"),
            ("Check retrieval", "Explain the 401(k) retirement match"),
            ("Check refusal", "What is the company's policy on cryptocurrency payroll?"),
            ("Check safety", "Can I be fired for taking medical leave?"),
        ),
    ),
    DIRECTOR: Role(
        DIRECTOR,
        "Director",
        "Policies for leading teams, pay and people decisions",
        "These suggestions focus on the people-management side of policy: pay, performance, promotion, "
        "discipline and handling complaints.",
        (
            ("Compensation", "What are the merit increase bands?"),
            ("Bonus", "How are annual bonus eligibility and targets decided?"),
            ("Promotions", "How do the promotion policy and career levels work?"),
            ("Discipline", "What are the steps of the progressive discipline ladder?"),
            ("Complaints", "How must a harassment or discrimination complaint be handled?"),
            ("Equity", "How does the stock option vesting schedule work?"),
        ),
    ),
    SENIOR: Role(
        SENIOR,
        "Senior employee",
        "Policies for experienced staff who guide and mentor others",
        "These suggestions cover career growth, performance reviews and the guidelines you may need to explain "
        "to colleagues.",
        (
            ("Performance", "How does the bi-annual performance review cycle work?"),
            ("Promotion", "What is needed to move to the next career level?"),
            ("Hybrid work", "What are the hybrid work schedule guidelines?"),
            ("Security", "What are the remote work security and workspace requirements?"),
            ("Ethics", "How do I report an ethics or compliance concern?"),
            ("Bonus", "Am I eligible for the annual bonus?"),
        ),
    ),
    EMPLOYEE: Role(
        EMPLOYEE,
        "Employee",
        "Everyday questions about leave, benefits and working arrangements",
        "These suggestions cover the questions employees ask most: time off, benefits, equipment and retirement.",
        (
            ("Parental leave", "What is the parental leave duration?"),
            ("Time off", "How many vacation days can I carry over?"),
            ("Sick leave", "How does sick leave work?"),
            ("Equipment", "Can I expense a home monitor?"),
            ("Retirement", "Explain our 401(k) retirement match"),
            ("Wellbeing", "What is the mental health therapy stipend?"),
        ),
    ),
}


def get_role(key: str | None) -> Role:
    return ROLES.get(key or "", ROLES[DEFAULT_ROLE])


def can(role: str | None, permission: str) -> bool:
    """True when ``role`` holds ``permission``. Unknown roles hold nothing."""
    if role not in ROLES:
        return False
    return role == ADMIN if permission in _ADMIN_ONLY else True


def require(role: str | None, permission: str) -> None:
    """Server-side guard: raise ``PermissionError`` unless ``role`` holds ``permission``."""
    if not can(role, permission):
        raise PermissionError("You do not have permission to do this.")
