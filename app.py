"""Streamlit front end for the HR policy RAG chatbot."""

from __future__ import annotations

import html
import logging
import time
import uuid

import streamlit as st

import audit
import auth
import config
import grounding
import roles
from ingest import UPLOAD_TYPES
from rag_pipeline import HRRagPipeline, IndexNotFoundError, confidence_label
from styles import CSS, LOGIN_CSS

config.setup_logging()
log = logging.getLogger("hr_chatbot.app")

st.set_page_config(
    page_title=f"{config.COMPANY_NAME} HR Assistant",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

NEW_CATEGORY_HINT = "Or type a new category name"
ADMIN_VIEWS = ["Assistant", "Knowledge base", "System", "Users"]


# --------------------------------------------------------------------------- resources
@st.cache_resource(show_spinner="Loading retrieval indexes...")
def load_pipeline() -> HRRagPipeline:
    return HRRagPipeline()


def init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("pending_prompt", None)
    st.session_state.setdefault("flash", [])
    st.session_state.setdefault("session_id", uuid.uuid4().hex[:8])
    st.session_state.setdefault("user", None)
    st.session_state.setdefault("login_error", "")
    st.session_state.setdefault("admin_view", ADMIN_VIEWS[0])


def queue_prompt(prompt: str) -> None:
    st.session_state.pending_prompt = prompt


def clear_history() -> None:
    st.session_state.messages = []
    st.session_state.pending_prompt = None


def sid() -> str:
    return st.session_state.get("session_id", "")


def actor(user: dict | None = None) -> dict:
    """Who is acting, for the audit log (display name, not the email, which the log would mask)."""
    user = user or st.session_state.get("user") or {}
    return {"user": user.get("name", ""), "role": user.get("role", "")}


def initials(name: str) -> str:
    parts = [p for p in name.replace(".", " ").split() if p]
    return "".join(p[0] for p in parts[:2]).upper() or "?"


def role_badge(role: str) -> str:
    return f'<span class="role-badge {html.escape(role)}">{html.escape(roles.get_role(role).label)}</span>'


# --------------------------------------------------------------------------- sign in / sign up
def start_session(user: dict, event: str) -> None:
    st.session_state.user = user
    st.session_state.messages = []
    st.session_state.pending_prompt = None
    st.session_state.admin_view = ADMIN_VIEWS[0]
    st.session_state.login_error = ""
    audit.record(event, sid(), **actor(user))


def do_login(email: str, password: str = "", trusted: bool = False) -> None:
    """Callback for the sign-in form and the demo shortcuts; failures are shown on the login page."""
    try:
        start_session(auth.authenticate(email, password, trusted=trusted), "login")
    except auth.AuthError as exc:
        st.session_state.login_error = str(exc)


def do_logout() -> None:
    audit.record("logout", sid(), **actor())
    for key in ("user", "messages", "pending_prompt", "flash", "login_error", "admin_view", "session_id", "auth_mode", "login_notice"):
        st.session_state.pop(key, None)
    init_state()


def resolve_user() -> dict | None:
    """The signed-in user, re-read from the store so a role change by an admin applies on the next action."""
    session_user = st.session_state.get("user")
    if not session_user:
        return None
    fresh = auth.get_user(session_user["email"])
    if fresh is None:
        do_logout()
        return None
    st.session_state.user = fresh
    return fresh


def set_auth_mode(mode: str) -> None:
    st.session_state.auth_mode = mode
    st.session_state.login_error = ""


def render_login() -> None:
    st.markdown(LOGIN_CSS, unsafe_allow_html=True)
    auth.ensure_demo_users()
    mode = st.session_state.get("auth_mode", "signin")
    left, right = st.columns(2, gap="small")

    with left, st.container(key="auth_brand"):
        st.markdown(
            f'<div class="auth-logo"><div class="mark">{html.escape(config.COMPANY_NAME[:1])}</div>'
            f'<div class="name">{html.escape(config.COMPANY_NAME)}</div>'
            '<div class="sub">HR Policy Assistant</div></div>',
            unsafe_allow_html=True,
        )

    with right, st.container(key="auth_card"):
        if mode == "signin":
            st.markdown('<div class="auth-title">Sign in</div>', unsafe_allow_html=True)
            if notice := st.session_state.pop("login_notice", ""):
                st.success(notice, icon="✅")
            if st.session_state.login_error:
                st.error(st.session_state.login_error, icon="⚠️")
                st.session_state.login_error = ""
            with st.form("login_form", border=False):
                email = st.text_input(
                    "Work email", placeholder="name@meridianlabs.example", key="login_email",
                    value=st.session_state.pop("prefill_email", ""),
                )
                password = st.text_input("Password", type="password", key="login_password")
                submitted = st.form_submit_button("Sign in", type="primary", width="stretch")
            if submitted:
                do_login(email, password)
                st.rerun()
            st.button("New here? Create an account", key="to_signup", on_click=set_auth_mode, args=("signup",), type="tertiary")

            demo = auth.demo_accounts()
            if demo:
                st.markdown('<div class="demo-head">Demo accounts</div>', unsafe_allow_html=True)
                with st.container(key="demo_grid"):
                    columns = st.columns(2)
                    for index, account in enumerate(demo):
                        with columns[index % 2]:
                            st.button(
                                f"{account['name'].split()[0]} · {roles.get_role(account['role']).label}",
                                key=f"demo_{account['email']}",
                                on_click=do_login,
                                args=(account["email"], "", True),
                                width="stretch",
                            )
        else:
            st.markdown('<div class="auth-title">Create account</div>', unsafe_allow_html=True)
            with st.form("signup_form", border=False):
                name = st.text_input("Full name", key="signup_name")
                new_email = st.text_input("Work email", placeholder="name@meridianlabs.example", key="signup_email")
                new_password = st.text_input(
                    "Password", type="password", help="At least 8 characters, with a letter and a number.", key="signup_password"
                )
                confirm = st.text_input("Confirm password", type="password", key="signup_confirm")
                created = st.form_submit_button("Create account", type="primary", width="stretch")
            if created:
                try:
                    user = auth.register(name, new_email, new_password, confirm)
                except auth.AuthError as exc:
                    st.error(str(exc), icon="⚠️")
                else:
                    audit.record("signup", sid(), **actor(user))
                    st.session_state.login_notice = "Account created. Sign in to continue."
                    st.session_state.prefill_email = user["email"]
                    st.session_state.auth_mode = "signin"
                    st.rerun()
            st.button("Already have an account? Sign in", key="to_signin", on_click=set_auth_mode, args=("signin",), type="tertiary")


def remove_upload(pipeline: HRRagPipeline, document_id: str) -> None:
    try:
        role = (st.session_state.get("user") or {}).get("role")
        pipeline.remove_uploaded_document(document_id, role=role)
        audit.record("document_removed", sid(), document_id=document_id, **actor())
        st.session_state.flash = [("ok", "Document removed from the knowledge base.")]
    except Exception as exc:
        st.session_state.flash = [("error", f"Could not remove the document: {exc}")]


# --------------------------------------------------------------------------- sidebar
def status_row(label: str, state: str, value: str) -> str:
    return (
        f'<div class="status-row"><div><span class="dot {state}"></span>{html.escape(label)}</div>'
        f'<span class="val">{html.escape(value)}</span></div>'
    )


@st.fragment(run_every=3)
def status_panel(pipeline: HRRagPipeline) -> None:
    s = pipeline.status()
    if s["llm_state"] == "ready":
        llm = status_row("Language model", "on", s["llm_mode"])
    elif s["llm_state"] == "error":
        llm = status_row("Language model", "off", "failed to load")
    else:
        llm = status_row("Language model", "busy", "warming up...")
    rows = [
        llm,
        status_row("Embeddings", "on", f"{s['embedding_model'].split('/')[-1]} · {s['embedding_device']}"),
        status_row("Vector store", "on", f"{s['vector_chunks']} chunks"),
        status_row("Keyword index", "on", f"{s['bm25_chunks']} chunks"),
        status_row("Source data", "idle" if s["index_stale"] else "on", "changed, re-ingest" if s["index_stale"] else "up to date"),
    ]
    st.markdown(f'<div class="status-list">{"".join(rows)}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="side-note">{s["documents"]} documents · {s["llm_model"].split("/")[-1]}</div>',
        unsafe_allow_html=True,
    )


@st.fragment(run_every=3)
def model_banner(pipeline: HRRagPipeline) -> None:
    state, detail = pipeline.llm_state, pipeline.llm_detail
    if state == "loading":
        st.markdown(
            '<div class="banner info"><b>The language model is starting up.</b> '
            f"({html.escape(detail or 'preparing')}). You can already ask a question; the answer appears as soon as it is ready.</div>",
            unsafe_allow_html=True,
        )
    elif state == "error":
        st.markdown(
            '<div class="banner error"><b>The language model could not be loaded.</b> '
            f"{html.escape(detail)}<br>Send a message to retry; interrupted downloads resume where they stopped.</div>",
            unsafe_allow_html=True,
        )


def show_flash() -> None:
    for kind, message in st.session_state.flash:
        if kind == "warn":
            st.warning(message, icon="⚠️")
        else:
            (st.success if kind == "ok" else st.error)(message, icon="✅" if kind == "ok" else "⚠️")
    st.session_state.flash = []


def handle_upload(pipeline: HRRagPipeline, files, category: str, role: str | None) -> None:
    results = []
    with st.spinner("Reading, chunking and embedding..."):
        for file in files:
            try:
                info = pipeline.add_uploaded_document(file.name, file.getvalue(), category, role=role)
                results.append(("ok", f"{file.name}: {info['chunks']} chunks added to “{info['category']}”."))
                results.extend(("warn", f"{file.name}: {warning}") for warning in info["warnings"])
                audit.record(
                    "document_added",
                    sid(),
                    **actor(),
                    filename=file.name,
                    document_id=info["document_id"],
                    category=info["category"],
                    chunks=info["chunks"],
                    ocr_confidence=info["ocr_confidence"],
                    warnings=info["warnings"],
                )
            except Exception as exc:
                results.append(("error", f"{file.name}: {exc}"))
                audit.record("document_rejected", sid(), **actor(), filename=file.name, reason=str(exc))
    st.session_state.flash = results
    st.rerun()


def kpi_grid(items: list[tuple[str, str, str]]) -> None:
    cells = "".join(
        f'<div class="kpi"><div class="label">{html.escape(label)}</div><div class="value">{html.escape(value)}</div>'
        f'<div class="hint">{html.escape(hint)}</div></div>'
        for label, value, hint in items
    )
    st.markdown(f'<div class="kpi-grid">{cells}</div>', unsafe_allow_html=True)


def render_knowledge_base(pipeline: HRRagPipeline, user: dict) -> None:
    """Administrator view: index size, per-category breakdown, and document upload/removal."""
    s = pipeline.status()
    uploads = pipeline.list_uploads()
    kpi_grid(
        [
            ("Documents", f"{s['documents']:,}", f"{len(uploads)} uploaded by admins"),
            ("Vector chunks", f"{s['vector_chunks']:,}", "stored in ChromaDB"),
            ("Keyword chunks", f"{s['bm25_chunks']:,}", "in the BM25 index"),
            ("Categories", f"{len(s['categories'])}", "policy groups"),
        ]
    )
    if s["vector_chunks"] != s["bm25_chunks"]:
        st.warning("The vector and keyword indexes differ in size. Run `python ingest.py` to rebuild them.", icon="⚠️")
    if s["index_stale"]:
        st.info("The source documents changed after the last ingestion. Run `python ingest.py`, then restart the app.", icon="ℹ️")

    left, right = st.columns([1.15, 1], gap="large")
    with left, st.container(key="kb_upload"):
        st.markdown('<div class="panel-title">Add documents</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-sub">PDF, DOCX, TXT, Markdown or images (read with OCR). Files are split into chunks and are searchable straight away.</div>',
            unsafe_allow_html=True,
        )
        show_flash()
        options = sorted({*pipeline.categories, config.DEFAULT_UPLOAD_CATEGORY})
        with st.form("upload_form", clear_on_submit=True, border=False):
            files = st.file_uploader("Files", type=UPLOAD_TYPES, accept_multiple_files=True)
            chosen = st.selectbox("Category", options, index=options.index(config.DEFAULT_UPLOAD_CATEGORY))
            new_category = st.text_input("New category", placeholder=NEW_CATEGORY_HINT, label_visibility="collapsed")
            submitted = st.form_submit_button("Add to knowledge base", width="stretch", type="primary")
        if submitted:
            if files:
                handle_upload(pipeline, files, new_category.strip() or chosen, user["role"])
            else:
                st.warning("Choose at least one file first.")

    with right:
        st.markdown('<div class="panel-title">Coverage by category</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-sub">Documents and chunks currently indexed.</div>', unsafe_allow_html=True)
        st.dataframe(
            pipeline.category_stats(),
            hide_index=True,
            width="stretch",
            column_config={
                "Category": st.column_config.TextColumn(width="medium"),
                "Documents": st.column_config.NumberColumn(width="small"),
                "Chunks": st.column_config.NumberColumn(width="small"),
            },
        )

    st.markdown('<div class="panel-title">Uploaded documents</div>', unsafe_allow_html=True)
    if not uploads:
        st.caption("Nothing has been uploaded yet. The built-in policy corpus cannot be removed here.")
    for item in uploads:
        left, right = st.columns([5, 1], vertical_alignment="center")
        left.markdown(
            f'<div class="upload-item">{html.escape(item["title"])}'
            f'<span>{html.escape(item["category"])} · {item["chunks"]} chunks</span></div>',
            unsafe_allow_html=True,
        )
        if item.get("warnings"):
            left.caption("⚠️ " + item["warnings"].replace(" | ", "\n\n⚠️ "))
        right.button(
            "Remove", key=f"rm_{item['document_id']}", on_click=remove_upload, args=(pipeline, item["document_id"]), width="stretch"
        )


def render_system(pipeline: HRRagPipeline) -> None:
    """Administrator view: live component status, an explanation of the pipeline, and the audit log."""
    with st.container(key="admin_panel"):
        st.markdown('<div class="panel-title">Component status</div>', unsafe_allow_html=True)
        status_panel(pipeline)
    st.markdown('<div class="panel-title" style="margin-top:18px">Pipeline reference</div>', unsafe_allow_html=True)
    render_how_it_works(pipeline)
    st.markdown('<div class="panel-title" style="margin-top:18px">Activity</div>', unsafe_allow_html=True)
    render_audit_log()


def render_users(user: dict) -> None:
    """Administrator view: list accounts and change roles."""
    accounts = auth.list_users()
    counts = {key: sum(1 for a in accounts if a["role"] == key) for key in roles.ROLES}
    kpi_grid([(roles.ROLES[key].label, str(counts[key]), "accounts") for key in roles.ROLES])

    with st.container(key="user_admin"):
        show_flash()
        st.markdown('<div class="panel-title">Accounts</div>', unsafe_allow_html=True)
        st.dataframe(
            [
                {"Name": a["name"], "Email": a["email"], "Title": a["title"], "Role": roles.get_role(a["role"]).label}
                for a in accounts
            ],
            hide_index=True,
            width="stretch",
        )
        st.markdown('<div class="panel-title" style="margin-top:14px">Change a role</div>', unsafe_allow_html=True)
        by_email = {a["email"]: a for a in accounts}
        with st.form("role_form", border=False):
            col_user, col_role = st.columns(2)
            target = col_user.selectbox("User", list(by_email), format_func=lambda e: f"{by_email[e]['name']} ({e})")
            new_role = col_role.selectbox("Role", list(roles.ROLES), format_func=lambda r: roles.ROLES[r].label, index=list(roles.ROLES).index(roles.EMPLOYEE))
            saved = st.form_submit_button("Save role", type="primary")
        if saved:
            try:
                updated = auth.set_role(user["role"], target, new_role)
            except (auth.AuthError, PermissionError) as exc:
                st.error(str(exc), icon="⚠️")
            else:
                audit.record("role_changed", sid(), **actor(), target=updated["name"], new_role=updated["role"])
                st.session_state.flash = [("ok", f"{updated['name']} is now {roles.get_role(updated['role']).label}.")]
                st.rerun()


def render_how_it_works(pipeline: HRRagPipeline) -> None:
    s = pipeline.status()
    with st.expander("Models, chunking and search"):
        st.markdown(
            f"""
**Language model** — `{s['llm_model'].split('/')[-1]}` ({s['llm_mode']}). It only writes the answer from the
passages it is given, and is told to refuse when the policies do not cover the question.

**Embedding model** — `{s['embedding_model'].split('/')[-1]}` turns text into vectors so that passages with a
similar meaning can be found even when the words differ. Runs on {s['embedding_device']}.

**Chunking** — each document is split into pieces of about **{config.CHUNK_SIZE_TOKENS} tokens** with
**{config.CHUNK_OVERLAP_TOKENS} tokens of overlap**, so a sentence cut at a boundary still appears whole in a
neighbouring chunk. Each chunk keeps its policy title, category and effective date.

**Vector store (ChromaDB)** — holds the chunk vectors on disk (`{s['vector_chunks']}` chunks now), so
nothing is re-embedded on restart.

**Keyword index (BM25)** — a second index that matches exact words such as “401(k)” or “FMLA”
(`{s['bm25_chunks']}` chunks).

**Hybrid retrieval** — both indexes are searched. The relevance score is
**{config.DENSE_WEIGHT} × semantic + {config.BM25_WEIGHT} × keyword**. The top passages go to the model.

**Passages per answer** — how many passages the model reads. More gives more context but a slower answer.

**Confidence** — High or Medium or Low, from the best passage's score. If nothing is relevant enough the
assistant declines instead of guessing.

**Search scope** — limits retrieval to one policy category.

**Knowledge base** — uploaded PDF, DOCX, TXT, Markdown and image files (text read by OCR) are chunked the same way and added to both
indexes, so you can ask about them straight away.

**Section-aware chunking** — documents are split at their headings first, so a chunk covers one topic. A very
long section is split further and every piece keeps its heading.

**Reranker** — the best candidates are re-read together with your question by a small cross-encoder
(`{config.RERANKER_MODEL.split('/')[-1] if config.RERANKER_ENABLED else 'disabled'}`), which orders them far more accurately than
the first search. The *Reranker* tag in Sources shows its score.

**Grounding check** — after an answer, its numbers and wording are compared with the passages. *Grounded in
sources* means they match. *Check figures* lists numbers that appear in no source, so verify them with HR.

**OCR warnings** — images and scans are read by OCR. If recognition is uncertain, an upload warning tells you to
check names and numbers against the original.

**Injection guard** — instruction-like text hidden in a document or passage ("ignore the rules...") is removed,
passages are fenced off as data for the model, and questions that try to override the assistant are refused.

**Audit log** — each question, the sources used, confidence, grounding result and every upload or removal is
recorded (emails and phone numbers are masked) in `data/audit_log.jsonl`.
"""
        )


def render_audit_log() -> None:
    if not config.AUDIT_ENABLED:
        return
    events = audit.read_recent(30)
    with st.expander(f"Audit log ({len(events)} recent)"):
        if not events:
            st.caption("No activity recorded yet.")
            return
        rows = [
            {
                "Time (UTC)": e["ts"].replace("T", " ").removesuffix("+00:00"),
                "User": e.get("user", ""),
                "Role": roles.get_role(e["role"]).label if e.get("role") else "",
                "Event": e["event"].replace("_", " "),
                "Detail": (e.get("question") or e.get("filename") or e.get("document_id") or "")[:70],
                "Outcome": e.get("outcome") or ("; ".join(e.get("warnings", []))[:50] if e.get("warnings") else ""),
                "Grounding": e.get("grounding", ""),
            }
            for e in events
        ]
        st.dataframe(rows, hide_index=True, width="stretch")
        if config.AUDIT_PATH.exists():
            st.download_button(
                "Download full log (.jsonl)", config.AUDIT_PATH.read_bytes(), file_name="hr_audit_log.jsonl",
                mime="application/json", width="stretch",
            )


def render_profile_menu(user: dict) -> None:
    """Collapsible account menu fixed to the top-right corner of the page."""
    with st.container(key="profile_menu"):
        with st.popover(user["name"], icon=":material/account_circle:"):
            st.markdown(
                f'<div class="profile"><div class="avatar {html.escape(user["role"])}">{html.escape(initials(user["name"]))}</div>'
                f'<div class="who"><div class="name">{html.escape(user["name"])}</div>'
                f'<div class="title">{html.escape(user["title"])}</div>{role_badge(user["role"])}</div></div>',
                unsafe_allow_html=True,
            )
            st.button("Sign out", on_click=do_logout, width="stretch", type="primary", key="sign_out")


def render_sidebar(pipeline: HRRagPipeline, user: dict) -> tuple[str | None, int, str]:
    with st.sidebar:
        st.markdown(
            f'<div class="brand"><div class="brand-mark">{html.escape(config.COMPANY_NAME[:1])}</div>'
            f'<div><div class="brand-name">{html.escape(config.COMPANY_NAME)}</div>'
            '<div class="brand-sub">HR Policy Assistant</div></div></div>',
            unsafe_allow_html=True,
        )

        view = ADMIN_VIEWS[0]
        if roles.can(user["role"], roles.MANAGE_KB):
            st.markdown('<div class="side-title">Navigate</div>', unsafe_allow_html=True)
            view = st.radio("View", ADMIN_VIEWS, key="admin_view", label_visibility="collapsed")

        choice, top_k = "All categories", config.DEFAULT_TOP_K
        if view == ADMIN_VIEWS[0]:
            st.markdown('<div class="side-title">Search settings</div>', unsafe_allow_html=True)
            choice = st.selectbox("Category", ["All categories", *pipeline.categories], help="Limit answers to one policy category.")
            top_k = st.slider(
                "Passages per answer", 2, 8, config.DEFAULT_TOP_K, help="How many policy passages are given to the model. More gives more context but a slower answer."
            )

        st.markdown('<div class="side-title">Conversation</div>', unsafe_allow_html=True)
        st.button("Clear history", on_click=clear_history, width="stretch")
        st.markdown(
            f'<div class="side-note">Need a person? {html.escape(config.HR_FALLBACK_CONTACT)}<br>'
            "Answers summarise policy and are not legal or medical advice.</div>",
            unsafe_allow_html=True,
        )
    return (None if choice == "All categories" else choice), top_k, view


# --------------------------------------------------------------------------- main area
def section_label(source: dict) -> str:
    section = (source.get("section") or "").strip()
    return f" &rsaquo; {html.escape(section[:70])}" if section else ""


def rerank_tag(source: dict) -> str:
    score = source.get("rerank_score")
    return "" if score is None else f'<span class="tag brand">Reranker {score:.2f}</span>'


def plain_dollars(text: str) -> str:
    """Streamlit renders $...$ as math; escape dollar signs so amounts like $500 and $250 stay plain text."""
    return text.replace("$", "\\$")


def escaped_stream(chunks):
    for chunk in chunks:
        yield plain_dollars(chunk)


def grounding_tag(result: dict | None) -> str:
    if not result:
        return ""
    text, level = grounding.describe(result)
    return f'<span class="tag {level}" title="Share of the answer wording found in the sources: {result["coverage"]:.0%}">{html.escape(text)}</span>'


def render_sources(message: dict) -> None:
    sources = message.get("sources") or []
    if not sources:
        return
    label = message.get("confidence", "Low")
    st.markdown(
        '<div class="meta-row">'
        f'<span class="tag {label.lower()}">Confidence: {label}</span>'
        f'<span class="tag neutral">Retrieval {message["retrieval_latency"] * 1000:.0f} ms</span>'
        f'<span class="tag neutral">Generation {message["generation_latency"]:.1f} s</span>'
        f'<span class="tag neutral">{len(sources)} sources</span>'
        f"{grounding_tag(message.get('grounding'))}"
        "</div>",
        unsafe_allow_html=True,
    )
    with st.expander(f"Sources ({len(sources)})"):
        cards = []
        for source in sources:
            text = html.escape(source["content"]).replace("$", "&#36;").replace("\n", "<br>")
            chunk_label = confidence_label(source["hybrid_score"]).lower()
            uploaded = '<span class="tag brand">Uploaded</span>' if source.get("origin") == "upload" else ""
            cards.append(
                '<details class="source-card">'
                f'<summary>{source["rank"]}. {html.escape(source["title"])}{section_label(source)}'
                f'<div class="sub">{html.escape(source["category"])} · {html.escape(source["document_id"])} · '
                f'effective {html.escape(source["effective_date"])} · passage {source["chunk_index"] + 1} of {source["total_chunks"]}</div>'
                f'<div class="bar"><div style="width:{source["hybrid_score"] * 100:.0f}%"></div></div></summary>'
                '<div class="scores">'
                f'{uploaded}'
                f'<span class="tag {chunk_label}">Relevance {source["hybrid_score"]:.2f}</span>'
                f'{rerank_tag(source)}'
                f'<span class="tag neutral">Semantic {source["dense_score"]:.2f} × {config.DENSE_WEIGHT}</span>'
                f'<span class="tag neutral">Keyword {source["keyword_score"]:.2f} × {config.BM25_WEIGHT}</span>'
                f'<span class="tag neutral">{html.escape(source["contact_email"])}</span>'
                "</div>"
                f'<div class="chunk">{text}</div>'
                "</details>"
            )
        st.markdown("".join(cards), unsafe_allow_html=True)


def render_header(category: str | None, view: str = "Assistant") -> None:
    scope = category or "All categories"
    titles = {
        "Assistant": (f"{config.COMPANY_NAME} HR Assistant", "Answers about leave, benefits, remote work, pay and conduct, with the source policy cited."),
        "Knowledge base": ("Knowledge base", "Add policy documents and check what the assistant can search."),
        "System": ("System", "Component health, how the pipeline works, and the activity log."),
        "Users": ("Users and roles", "Review accounts and decide what each person can see."),
    }
    title, subtitle = titles.get(view, titles["Assistant"])
    pill = f'<div class="scope-pill">Scope: {html.escape(scope)}</div>' if view == "Assistant" else ""
    st.markdown(
        '<div class="page-head"><div>'
        f'<div class="eyebrow">{"Policy assistant" if view == "Assistant" else "Administration"}</div>'
        f"<h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p></div>{pill}</div>",
        unsafe_allow_html=True,
    )


def render_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_sources(message)


def render_empty_state(pipeline: HRRagPipeline, user: dict) -> None:
    role = roles.get_role(user["role"])
    first_name = user["name"].split()[0]
    n_categories = len(pipeline.categories)
    st.markdown(
        f'<div class="welcome"><div class="hello"><h3>Hello, {html.escape(first_name)}. How can I help?</h3>{role_badge(role.key)}</div>'
        f"<p>{html.escape(role.welcome)} I search {n_categories} policy categories and any documents your administrator "
        "has added, then answer using only what those documents say.</p></div>"
        '<div class="starter-label">Suggested for you</div>',
        unsafe_allow_html=True,
    )
    with st.container(key="starters"):
        columns = st.columns(3)
        for index, (topic, prompt) in enumerate(role.starters):
            columns[index % 3].button(
                f"{topic}\n\n{prompt}", width="stretch", key=f"starter_{index}", on_click=queue_prompt, args=(prompt,)
            )


# --------------------------------------------------------------------------- chat turn
def answer_question(pipeline: HRRagPipeline, prompt: str, category: str | None, top_k: int, user: dict) -> None:
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            if not pipeline.llm_ready:
                with st.spinner("Waiting for the language model to finish starting up..."):
                    pipeline.ensure_llm()
            with st.spinner("Searching policies..."):
                retrieval = pipeline.retrieve(prompt, category, history, top_k)

            started = time.perf_counter()
            answer = st.write_stream(escaped_stream(pipeline.stream_answer(prompt, retrieval)))
            generation_latency = time.perf_counter() - started
        except Exception as exc:  # surface model/network failures in the UI instead of a stack trace
            log.exception("Answering failed")
            audit.record("error", sid(), **actor(user), question=prompt, error=f"{type(exc).__name__}: {exc}")
            st.error(f"Something went wrong while answering: {type(exc).__name__}: {exc or 'no details, see the terminal log'}")
            st.session_state.messages.pop()
            return

        audit.query_event(
            sid(), prompt, answer.replace("\\$", "$"), retrieval, generation_latency, retrieval.grounding, **actor(user)
        )
        message = {
            "role": "assistant",
            "content": answer,
            "grounding": retrieval.grounding,
            "sources": [] if (retrieval.blocked or retrieval.smalltalk) else retrieval.sources,
            "confidence": "Low" if retrieval.declined else retrieval.confidence,
            "retrieval_latency": retrieval.latency,
            "generation_latency": generation_latency,
        }
        render_sources(message)
    st.session_state.messages.append(message)


def render_assistant(pipeline: HRRagPipeline, user: dict, category: str | None, top_k: int) -> None:
    model_banner(pipeline)

    pending = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

    render_history()
    if not st.session_state.messages and not pending:
        render_empty_state(pipeline, user)

    prompt = st.chat_input("Ask an HR policy question...") or pending
    if prompt:
        answer_question(pipeline, prompt, category, top_k, user)


# --------------------------------------------------------------------------- main
def main() -> None:
    init_state()
    st.markdown(CSS, unsafe_allow_html=True)

    user = resolve_user()
    if user is None:
        render_login()
        return

    try:
        pipeline = load_pipeline()
    except IndexNotFoundError as exc:
        st.error(f"{exc}")
        st.code("python ingest.py", language="bash")
        st.stop()

    pipeline.start_background_load()  # no-op after the first call; keeps the UI responsive
    render_profile_menu(user)
    category, top_k, view = render_sidebar(pipeline, user)

    render_header(category, view)
    if view == "Knowledge base":
        render_knowledge_base(pipeline, user)
    elif view == "System":
        render_system(pipeline)
    elif view == "Users":
        render_users(user)
    else:
        render_assistant(pipeline, user, category, top_k)



main()
