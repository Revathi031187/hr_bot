"""End-to-end UI test with Streamlit's AppTest (no browser, no LLM). Run: python tests/test_app_ui.py
Uses a temporary users file and audit log, so real data is untouched."""
import os
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
import auth, roles
import audit, config

tmp = pathlib.Path(tempfile.mkdtemp())
auth.USERS_PATH = tmp / "users.json"
config.AUDIT_PATH = tmp / "audit.jsonl"

import rag_pipeline as _rp
_rp.HRRagPipeline.start_background_load = lambda self: None  # these tests do not need the LLM
from streamlit.testing.v1 import AppTest

at = AppTest.from_file(r"d:\hr-chatbot\app.py", default_timeout=180)


def texts(at):
    return " ".join(m.value for m in at.markdown)


def buttons(at):
    return [b.label for b in at.button]


def starter_topics(at):
    return [b.label.split("\n")[0] for b in at.button if b.key and b.key.startswith("starter_")]


def demo(at, who):
    return [b for b in at.button if b.key == f"demo_{who}@meridianlabs.example"][0]


def check_no_exception(at, where):
    assert not at.exception, f"{where}: {[e.value for e in at.exception]}"


# ---- 1. logged-out: login page only
at.run()
check_no_exception(at, "login page")
assert "Sign in" in texts(at) and len([b for b in at.button if b.key and b.key.startswith("demo_")]) == 4
assert not at.sidebar.button and not any(b.key and b.key.startswith("starter_") for b in at.button)
print("1 login page ok; demo buttons:", [b.label for b in at.button if b.key and b.key.startswith("demo_")])

# ---- 2. wrong password shows an error, stays logged out
at.text_input(key="login_email").set_value("arjun.kapoor@meridianlabs.example")
at.text_input(key="login_password").set_value("wrong-pass1")
[b for b in at.button if b.label == "Sign in"][0].click().run()
check_no_exception(at, "bad login")
assert at.session_state["user"] is None and any("Incorrect" in e.value for e in at.error)
print("2 bad password rejected")

# ---- 3. employee via password
at.text_input(key="login_email").set_value("arjun.kapoor@meridianlabs.example")
at.text_input(key="login_password").set_value(auth.DEMO_PASSWORD)
[b for b in at.button if b.label == "Sign in"][0].click().run()
check_no_exception(at, "employee login")
assert at.session_state["user"]["role"] == "employee"
emp = starter_topics(at)
assert sorted(emp) == sorted(t for t, _ in roles.ROLES["employee"].starters), emp
assert not at.sidebar.radio and not at.get("file_uploader")
assert "Sign out" in buttons(at)
side = texts(at)
assert "Arjun Kapoor" in side and "Employee" in side
assert "Knowledge" not in side and "Audit" not in side
print("3 employee ok:", emp)

# server-side guard: employee cannot call admin operations even directly
import rag_pipeline
pipe = at.session_state  # noqa
try:
    rag_pipeline.roles.require("employee", roles.MANAGE_KB)
    raise SystemExit("guard failed")
except PermissionError:
    print("3b server-side guard ok")

# ---- 4. sign out
[b for b in at.button if b.label == "Sign out"][0].click().run()
check_no_exception(at, "logout")
assert at.session_state["user"] is None and "Sign in" in texts(at)
print("4 sign out ok")

# ---- 5. director and senior get different starters
seen = {"employee": emp}
for role, label in (("director", "Director"), ("senior", "Senior employee")):
    demo(at, {"director": "daniel.whitfield", "senior": "meera.iyer"}[role]).click().run()
    check_no_exception(at, role)
    assert at.session_state["user"]["role"] == role
    seen[role] = starter_topics(at)
    assert sorted(seen[role]) == sorted(t for t, _ in roles.ROLES[role].starters)
    assert not at.sidebar.radio
    [b for b in at.button if b.label == "Sign out"][0].click().run()
print("5 director/senior ok:", {k: v[:3] for k, v in seen.items()})
assert len({tuple(sorted(v)) for v in seen.values()}) == 3

# ---- 6. admin: views
demo(at, "priya.raman").click().run()
check_no_exception(at, "admin login")
assert at.session_state["user"]["role"] == "admin"
assert [r.key for r in at.sidebar.radio] == ["admin_view"], "admin must see the sidebar navigation"
print("6 admin sees switcher; starters:", starter_topics(at))
for view in ("Knowledge base", "System", "Users", "Assistant"):
    at.session_state["admin_view"] = view
    at.run()
    check_no_exception(at, view)
    t = texts(at)
    if view == "Knowledge base":
        assert "Vector chunks" in t and "Coverage by category" in t and at.get("file_uploader")
    if view == "System":
        assert "Component status" in t
    if view == "Users":
        assert "Change a role" in t
    print("   view ok:", view)

# ---- 7. admin changes a role through the UI; takes effect for that user
at.session_state["admin_view"] = "Users"
at.run()
main_selects = [s for s in at.selectbox if s.label in ("User", "Role")]
main_selects[0].set_value("arjun.kapoor@meridianlabs.example")
main_selects[1].set_value("senior")
[b for b in at.button if b.label == "Save role"][0].click().run()
check_no_exception(at, "role change")
assert auth.get_user("arjun.kapoor@meridianlabs.example")["role"] == "senior"
print("7 role change ok")

# ---- 8. sign up creates an Employee, then the user must sign in
[b for b in at.button if b.label == "Sign out"][0].click().run()
[b for b in at.button if b.key == "to_signup"][0].click().run()
at.text_input(key="signup_name").set_value("Nina Shah")
at.text_input(key="signup_email").set_value("nina.shah@meridianlabs.example")
at.text_input(key="signup_password").set_value("Sup3rSecret")
at.text_input(key="signup_confirm").set_value("Sup3rSecret")
[b for b in at.button if b.label == "Create account"][0].click().run()
check_no_exception(at, "signup")
assert at.session_state["user"] is None, "sign-up must not sign the user in"
assert any("Account created" in m.value for m in at.success)
assert at.text_input(key="login_email").value == "nina.shah@meridianlabs.example"
at.text_input(key="login_password").set_value("Sup3rSecret")
[b for b in at.button if b.label == "Sign in"][0].click().run()
check_no_exception(at, "first login")
assert at.session_state["user"]["role"] == "employee" and at.session_state["user"]["name"] == "Nina Shah"
print("8 sign up then sign in ok")

ev = [e["event"] + ":" + e.get("role", "") for e in audit.read_recent(50)][::-1]
print("audit events:", ev)
print("ALL UI TESTS PASSED")
