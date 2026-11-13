"""Streamlit front end for the HR policy RAG chatbot."""

from __future__ import annotations

import html
import logging
import time
import uuid

import streamlit as st

import audit
import config
import grounding
from ingest import UPLOAD_TYPES
from rag_pipeline import HRRagPipeline, IndexNotFoundError, confidence_label

config.setup_logging()
log = logging.getLogger("hr_chatbot.app")

st.set_page_config(
    page_title=f"{config.COMPANY_NAME} HR Assistant",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

STARTER_PROMPTS = [
    ("Parental leave", "What is the parental leave duration?"),
    ("Equipment", "Can I expense a home monitor?"),
    ("Retirement", "Explain our 401(k) / retirement match"),
    ("Equity", "How does the stock option vesting schedule work?"),
    ("Time off", "How many vacation days can I carry over?"),
    ("Wellbeing", "What is the mental health therapy stipend?"),
]
NEW_CATEGORY_HINT = "Or type a new category name"

CSS = """
<style>
:root {
  --ink: #0f172a;
  --body: #334155;
  --muted: #64748b;
  --line: #e2e8f0;
  --canvas: #f8fafc;
  --surface: #ffffff;
  --brand: #1d4ed8;
  --brand-soft: #eff6ff;
  --brand-line: #bfdbfe;
  --good: #047857; --good-bg: #ecfdf5; --good-line: #a7f3d0;
  --warn: #b45309; --warn-bg: #fffbeb; --warn-line: #fde68a;
  --bad: #b91c1c;  --bad-bg: #fef2f2;  --bad-line: #fecaca;
}
html, body, .stApp, [class*="css"] {
  font-family: "Inter", "Segoe UI", -apple-system, "Helvetica Neue", Arial, sans-serif;
  color: var(--body);
}
.stApp { background: var(--canvas); }
#MainMenu, footer, [data-testid="stToolbarActions"], [data-testid="stMainMenu"] { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
/* The button that reopens a collapsed sidebar must stay visible. */
[data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapseButton"] { visibility: visible !important; }
[data-testid="stExpandSidebarButton"] { background: var(--surface); border: 1px solid var(--line); border-radius: 8px; }
.block-container { padding-top: 2.6rem; padding-bottom: 6rem; max-width: 920px; }

/* ---- Page header ---- */
.page-head { display: flex; align-items: flex-end; justify-content: space-between; gap: 16px;
  padding-bottom: 14px; margin-bottom: 18px; border-bottom: 1px solid var(--line); }
.eyebrow { font-size: 0.72rem; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase; color: var(--brand); }
.page-head h1 { font-size: 1.65rem; line-height: 1.2; font-weight: 700; color: var(--ink); margin: 2px 0 4px 0; padding: 0; }
.page-head p { margin: 0; color: var(--muted); font-size: 0.93rem; }
.scope-pill { font-size: 0.78rem; font-weight: 600; color: var(--brand); background: var(--brand-soft);
  border: 1px solid var(--brand-line); border-radius: 999px; padding: 5px 12px; white-space: nowrap; }

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] { background: var(--surface); border-right: 1px solid var(--line); }
section[data-testid="stSidebar"] .block-container { padding-top: 1.6rem; }
.brand { display: flex; align-items: center; gap: 10px; margin-bottom: 4px; }
.brand-mark { width: 34px; height: 34px; border-radius: 9px; background: var(--brand); color: #fff;
  display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 0.95rem; }
.brand-name { font-weight: 700; color: var(--ink); font-size: 1rem; line-height: 1.1; }
.brand-sub { color: var(--muted); font-size: 0.78rem; }
.side-title { font-size: 0.7rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase;
  color: var(--muted); margin: 22px 0 8px 0; }
.status-list { border: 1px solid var(--line); border-radius: 10px; background: var(--surface); }
.status-row { display: flex; align-items: center; justify-content: space-between; gap: 10px;
  padding: 9px 12px; font-size: 0.83rem; color: var(--ink); }
.status-row + .status-row { border-top: 1px solid var(--line); }
.status-row .val { color: var(--muted); font-size: 0.76rem; text-align: right; }
.dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 8px; }
.dot.on { background: #10b981; } .dot.idle { background: #f59e0b; } .dot.off { background: #ef4444; }
.dot.busy { background: #3b82f6; animation: pulse 1.2s ease-in-out infinite; }
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.3; } }
.side-note { color: var(--muted); font-size: 0.76rem; margin-top: 6px; }
.upload-item { font-size: 0.82rem; color: var(--ink); font-weight: 600; line-height: 1.25; }
.upload-item span { display: block; color: var(--muted); font-weight: 400; font-size: 0.74rem; }

/* ---- Banners ---- */
.banner { border-radius: 10px; padding: 10px 14px; font-size: 0.86rem; margin-bottom: 14px; border: 1px solid; }
.banner.info { background: var(--brand-soft); border-color: var(--brand-line); color: #1e3a8a; }
.banner.error { background: var(--bad-bg); border-color: var(--bad-line); color: var(--bad); }

/* ---- Empty state ---- */
.welcome { background: var(--surface); border: 1px solid var(--line); border-radius: 14px; padding: 22px 24px; margin-bottom: 14px; }
.welcome h3 { margin: 0 0 6px 0; color: var(--ink); font-size: 1.05rem; font-weight: 700; }
.welcome p { margin: 0; color: var(--muted); font-size: 0.9rem; line-height: 1.55; }
.st-key-starters .stButton > button {
  width: 100%; height: 100%; min-height: 64px; text-align: left; justify-content: flex-start;
  border-radius: 12px; border: 1px solid var(--line); background: var(--surface); color: var(--ink);
  padding: 12px 14px; font-weight: 500; box-shadow: none; transition: border-color .15s, box-shadow .15s;
}
.st-key-starters [data-testid="stHorizontalBlock"] { align-items: stretch; }
.st-key-starters [data-testid="stColumn"] > div, .st-key-starters .stButton { height: 100%; }
.st-key-starters .stButton > button > div { width: 100%; justify-content: flex-start; text-align: left; }
.st-key-starters .stButton > button [data-testid="stMarkdownContainer"] { width: 100%; text-align: left; }
.st-key-starters .stButton > button p { white-space: normal; text-align: left; margin: 0; line-height: 1.4; font-size: 0.88rem; }
.st-key-starters .stButton > button p:first-child { font-size: 0.7rem; font-weight: 700; letter-spacing: 0.08em;
  text-transform: uppercase; color: var(--brand); margin-bottom: 4px; }
.st-key-starters .stButton > button:hover { border-color: var(--brand); box-shadow: 0 2px 8px rgba(29,78,216,.10); color: var(--brand); }
.starter-label { font-size: 0.68rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted); margin: 4px 0 -4px 2px; }

/* ---- Chat ---- */
[data-testid="stChatMessage"] { background: var(--surface); border: 1px solid var(--line); border-radius: 14px;
  padding: 14px 18px; margin-bottom: 10px; box-shadow: 0 1px 2px rgba(15,23,42,.04); }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { background: var(--brand-soft); border-color: var(--brand-line); }
[data-testid="stChatInput"] { border-radius: 14px; }
.meta-row { display: flex; flex-wrap: wrap; gap: 6px; margin: 12px 0 8px 0; }
.tag { display: inline-block; padding: 2px 9px; border-radius: 6px; font-size: 0.72rem; font-weight: 600; border: 1px solid transparent; }
.tag.neutral { background: #f1f5f9; color: var(--body); border-color: var(--line); }
.tag.brand { background: var(--brand-soft); color: var(--brand); border-color: var(--brand-line); }
.tag.high { background: var(--good-bg); color: var(--good); border-color: var(--good-line); }
.tag.medium { background: var(--warn-bg); color: var(--warn); border-color: var(--warn-line); }
.tag.warn { background: var(--warn-bg); color: var(--warn); border-color: var(--warn-line); }
.tag.good { background: var(--good-bg); color: var(--good); border-color: var(--good-line); }
.tag.bad { background: var(--bad-bg); color: var(--bad); border-color: var(--bad-line); }
.tag.low { background: var(--bad-bg); color: var(--bad); border-color: var(--bad-line); }

/* ---- Source cards ---- */
.source-card { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 11px 14px; margin: 8px 0; }
.source-card[open] { border-color: var(--brand-line); }
.source-card summary { cursor: pointer; list-style: none; font-weight: 600; font-size: 0.9rem; color: var(--ink); }
.source-card summary::-webkit-details-marker { display: none; }
.source-card .sub { color: var(--muted); font-weight: 400; font-size: 0.78rem; margin-top: 3px; }
.source-card .scores { display: flex; flex-wrap: wrap; gap: 6px; margin: 12px 0 8px 0; }
.source-card .chunk { background: var(--canvas); border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px;
  font-size: 0.83rem; line-height: 1.55; color: var(--body); }
.bar { height: 4px; background: var(--line); border-radius: 4px; overflow: hidden; margin-top: 9px; }
.bar > div { height: 100%; background: var(--brand); }

/* ---- Buttons ---- */
section[data-testid="stSidebar"] .stButton > button, section[data-testid="stSidebar"] [data-testid="stFormSubmitButton"] > button {
  border-radius: 8px; font-weight: 600; font-size: 0.84rem; }
</style>
"""


# --------------------------------------------------------------------------- resources
@st.cache_resource(show_spinner="Loading retrieval indexes...")
def load_pipeline() -> HRRagPipeline:
    return HRRagPipeline()


def init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("pending_prompt", None)
    st.session_state.setdefault("flash", [])
    st.session_state.setdefault("session_id", uuid.uuid4().hex[:8])


def queue_prompt(prompt: str) -> None:
    st.session_state.pending_prompt = prompt


def clear_history() -> None:
    st.session_state.messages = []
    st.session_state.pending_prompt = None


def remove_upload(pipeline: HRRagPipeline, document_id: str) -> None:
    try:
        pipeline.remove_uploaded_document(document_id)
        audit.record("document_removed", st.session_state.get("session_id", ""), document_id=document_id)
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


def handle_upload(pipeline: HRRagPipeline, files, category: str) -> None:
    results = []
    with st.spinner("Reading, chunking and embedding..."):
        for file in files:
            try:
                info = pipeline.add_uploaded_document(file.name, file.getvalue(), category)
                results.append(("ok", f"{file.name}: {info['chunks']} chunks added to “{info['category']}”."))
                results.extend(("warn", f"{file.name}: {warning}") for warning in info["warnings"])
                audit.record(
                    "document_added",
                    st.session_state.get("session_id", ""),
                    filename=file.name,
                    document_id=info["document_id"],
                    category=info["category"],
                    chunks=info["chunks"],
                    ocr_confidence=info["ocr_confidence"],
                    warnings=info["warnings"],
                )
            except Exception as exc:
                results.append(("error", f"{file.name}: {exc}"))
                audit.record("document_rejected", st.session_state.get("session_id", ""), filename=file.name, reason=str(exc))
    st.session_state.flash = results
    st.rerun()


def render_knowledge_base(pipeline: HRRagPipeline) -> None:
    st.markdown('<div class="side-title">Knowledge base</div>', unsafe_allow_html=True)
    show_flash()

    options = sorted({*pipeline.categories, config.DEFAULT_UPLOAD_CATEGORY})
    with st.form("upload_form", clear_on_submit=True, border=False):
        files = st.file_uploader(
            "Add documents",
            type=UPLOAD_TYPES,
            accept_multiple_files=True,
            help="PDF, DOCX, TXT, Markdown or images (PNG, JPG: text is read with OCR). Files are split into chunks and stored in ChromaDB so you can ask about them.",
        )
        chosen = st.selectbox("Category", options, index=options.index(config.DEFAULT_UPLOAD_CATEGORY))
        new_category = st.text_input("New category", placeholder=NEW_CATEGORY_HINT, label_visibility="collapsed")
        submitted = st.form_submit_button("Add to knowledge base", width="stretch", type="primary")
    if submitted:
        if files:
            handle_upload(pipeline, files, new_category.strip() or chosen)
        else:
            st.warning("Choose at least one file first.")

    uploads = pipeline.list_uploads()
    if uploads:
        with st.expander(f"Uploaded documents ({len(uploads)})"):
            for item in uploads:
                left, right = st.columns([4, 2], vertical_alignment="center")
                left.markdown(
                    f'<div class="upload-item">{html.escape(item["title"])}'
                    f'<span>{html.escape(item["category"])} · {item["chunks"]} chunks</span></div>',
                    unsafe_allow_html=True,
                )
                if item.get("warnings"):
                    left.caption("⚠️ " + item["warnings"].replace(" | ", "\n\n⚠️ "))
                right.button("Remove", key=f"rm_{item['document_id']}", on_click=remove_upload,
                             args=(pipeline, item["document_id"]), width="stretch")


def render_how_it_works(pipeline: HRRagPipeline) -> None:
    s = pipeline.status()
    st.markdown('<div class="side-title">How it works</div>', unsafe_allow_html=True)
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


def render_sidebar(pipeline: HRRagPipeline) -> tuple[str | None, int]:
    with st.sidebar:
        st.markdown(
            f'<div class="brand"><div class="brand-mark">{html.escape(config.COMPANY_NAME[:1])}</div>'
            f'<div><div class="brand-name">{html.escape(config.COMPANY_NAME)}</div>'
            '<div class="brand-sub">HR Policy Assistant</div></div></div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="side-title">System</div>', unsafe_allow_html=True)
        status_panel(pipeline)

        st.markdown('<div class="side-title">Search scope</div>', unsafe_allow_html=True)
        choice = st.selectbox("Category", ["All categories", *pipeline.categories], label_visibility="collapsed")
        top_k = st.slider("Passages per answer", 2, 8, config.DEFAULT_TOP_K, help="How many policy passages are given to the model.")

        render_knowledge_base(pipeline)
        render_how_it_works(pipeline)
        render_audit_log()

        st.markdown('<div class="side-title">Conversation</div>', unsafe_allow_html=True)
        st.button("Clear history", on_click=clear_history, width="stretch")
        st.markdown(
            f'<div class="side-note">Need a person? {html.escape(config.HR_FALLBACK_CONTACT)}<br>'
            "Answers summarise policy and are not legal or medical advice.</div>",
            unsafe_allow_html=True,
        )
    return (None if choice == "All categories" else choice), top_k


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


def render_header(category: str | None) -> None:
    scope = category or "All categories"
    st.markdown(
        '<div class="page-head"><div>'
        '<div class="eyebrow">Policy assistant</div>'
        f"<h1>{html.escape(config.COMPANY_NAME)} HR Assistant</h1>"
        "<p>Answers about leave, benefits, remote work, pay and conduct, with the source policy cited.</p></div>"
        f'<div class="scope-pill">Scope: {html.escape(scope)}</div></div>',
        unsafe_allow_html=True,
    )


def render_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_sources(message)


def render_empty_state(pipeline: HRRagPipeline) -> None:
    n_categories = len(pipeline.categories)
    st.markdown(
        '<div class="welcome"><h3>How can I help?</h3>'
        f"<p>I search {n_categories} policy categories and any documents you upload, then answer using only what those "
        "documents say. Pick a topic below or type your own question.</p></div>"
        '<div class="starter-label">Suggested questions</div>',
        unsafe_allow_html=True,
    )
    with st.container(key="starters"):
        columns = st.columns(3)
        for index, (topic, prompt) in enumerate(STARTER_PROMPTS):
            columns[index % 3].button(
                f"{topic}\n\n{prompt}", width="stretch", key=f"starter_{index}", on_click=queue_prompt, args=(prompt,)
            )


# --------------------------------------------------------------------------- chat turn
def answer_question(pipeline: HRRagPipeline, prompt: str, category: str | None, top_k: int) -> None:
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
            audit.record("error", st.session_state.get("session_id", ""), question=prompt, error=f"{type(exc).__name__}: {exc}")
            st.error(f"Something went wrong while answering: {type(exc).__name__}: {exc or 'no details, see the terminal log'}")
            st.session_state.messages.pop()
            return

        audit.query_event(
            st.session_state.get("session_id", ""), prompt, answer.replace("\\$", "$"), retrieval, generation_latency, retrieval.grounding
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


# --------------------------------------------------------------------------- main
def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    init_state()

    try:
        pipeline = load_pipeline()
    except IndexNotFoundError as exc:
        st.error(f"{exc}")
        st.code("python ingest.py", language="bash")
        st.stop()

    pipeline.start_background_load()  # no-op after the first call; keeps the UI responsive
    category, top_k = render_sidebar(pipeline)

    render_header(category)
    model_banner(pipeline)

    pending = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

    render_history()
    if not st.session_state.messages and not pending:
        render_empty_state(pipeline)

    prompt = st.chat_input("Ask an HR policy question...") or pending
    if prompt:
        answer_question(pipeline, prompt, category, top_k)


main()
