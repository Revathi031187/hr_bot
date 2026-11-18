"""Accounts, passwords, lockout and role permissions. Run: python tests/test_auth_roles.py (uses a temporary users file)."""
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import auth, roles

tmp = pathlib.Path(tempfile.mkdtemp()) / "users.json"
auth.USERS_PATH = tmp
auth.PBKDF2_ITERATIONS = 1000  # speed only


def raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc as e:
        return str(e)
    raise AssertionError(f"{fn.__name__} did not raise {exc.__name__}")


auth.ensure_demo_users()
assert len(auth.demo_accounts()) == 4
assert [a["role"] for a in auth.demo_accounts()] == ["admin", "director", "senior", "employee"]
auth.ensure_demo_users()  # idempotent
assert len(auth.list_users()) == 4

# password login
u = auth.authenticate("Priya.Raman@meridianlabs.example ", auth.DEMO_PASSWORD)
assert u["role"] == "admin" and "hash" not in u and "salt" not in u
print("wrong pw:", raises(auth.AuthError, auth.authenticate, "priya.raman@meridianlabs.example", "nope"))
print("unknown:", raises(auth.AuthError, auth.authenticate, "ghost@x.com", "nope"))

# trusted shortcut only for demo accounts
assert auth.authenticate("arjun.kapoor@meridianlabs.example", "", trusted=True)["role"] == "employee"
auth.register("Sam Lee", "sam@meridianlabs.example", "Passw0rdX", "Passw0rdX")
raises(auth.AuthError, auth.authenticate, "sam@meridianlabs.example", "", trusted=True)

# sign-up rules
new = auth.register("  Ravi   Kumar ", "Ravi@Meridianlabs.example", "abc12345", "abc12345")
assert new["role"] == "employee" and new["name"] == "Ravi Kumar" and new["email"] == "ravi@meridianlabs.example"
print("dup:", raises(auth.AuthError, auth.register, "Ravi K", "ravi@meridianlabs.example", "abc12345", "abc12345"))
print("short:", raises(auth.AuthError, auth.register, "A B", "a@b.com", "ab1", "ab1"))
print("nodigit:", raises(auth.AuthError, auth.register, "A B", "a@b.com", "abcdefgh", "abcdefgh"))
print("mismatch:", raises(auth.AuthError, auth.register, "A B", "a@b.com", "abcdefg1", "abcdefg2"))
print("email:", raises(auth.AuthError, auth.register, "A B", "not-an-email", "abcdefg1", "abcdefg1"))
assert auth.authenticate("ravi@meridianlabs.example", "abc12345")["name"] == "Ravi Kumar"

# stored passwords are hashed
raw = tmp.read_text()
assert "abc12345" not in raw and auth.DEMO_PASSWORD not in raw

# lockout
for _ in range(auth.MAX_FAILED_LOGINS):
    raises(auth.AuthError, auth.authenticate, "ravi@meridianlabs.example", "bad")
print("lock:", raises(auth.AuthError, auth.authenticate, "ravi@meridianlabs.example", "abc12345"))

# role management: admin only
raises(PermissionError, auth.set_role, "employee", "sam@meridianlabs.example", "admin")
raises(PermissionError, auth.set_role, "director", "sam@meridianlabs.example", "admin")
assert auth.set_role("admin", "sam@meridianlabs.example", "senior")["role"] == "senior"
print("last admin:", raises(auth.AuthError, auth.set_role, "admin", "priya.raman@meridianlabs.example", "employee"))
raises(auth.AuthError, auth.set_role, "admin", "sam@meridianlabs.example", "boss")

# permissions
for r in ("director", "senior", "employee", None, "hacker"):
    assert not roles.can(r, roles.MANAGE_KB) and not roles.can(r, roles.VIEW_AUDIT)
    raises(PermissionError, roles.require, r, roles.MANAGE_KB)
assert roles.can("admin", roles.MANAGE_KB)
starters = {k: [p for _, p in v.starters] for k, v in roles.ROLES.items()}
assert len({tuple(v) for v in starters.values()}) == 4, "starter lists must differ per role"
print("ALL AUTH TESTS PASSED")
