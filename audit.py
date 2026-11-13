"""Append-only audit log (JSON lines) of questions, answers, sources and knowledge-base changes.

Each line is one event. Emails, phone numbers and ID-like numbers are redacted before anything is written, text
is truncated, and the file rotates once it passes ``HR_AUDIT_MAX_BYTES`` (the previous file is kept as ``.1``).
Logging never raises: a failure to write is reported to the application log and the chat carries on.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from datetime import datetime, timezone
from pathlib import Path

import config

log = logging.getLogger("hr_chatbot.audit")
_lock = threading.Lock()

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_PHONE = re.compile(r"(?<!\w)\+?\d[\d\s().-]{8,}\d(?!\w)")
_LONG_ID = re.compile(r"\b\d{9,}\b")
MAX_TEXT = 2000


def redact(text: str) -> str:
    """Mask emails, phone numbers and long ID numbers, keeping the company's own HR contact address."""
    text = _EMAIL.sub(lambda m: m.group(0) if m.group(0) == config.HR_FALLBACK_CONTACT else "[email]", text)
    text = _SSN.sub("[id]", text)
    text = _PHONE.sub("[phone]", text)
    return _LONG_ID.sub("[id]", text)


def _clip(text: str) -> str:
    text = redact(text or "")
    return text if len(text) <= MAX_TEXT else text[:MAX_TEXT] + "…"


def _rotate(path: Path) -> None:
    if path.exists() and path.stat().st_size > config.AUDIT_MAX_BYTES:
        path.replace(path.with_suffix(path.suffix + ".1"))


def record(event: str, session_id: str = "", **fields) -> None:
    """Write one event. ``fields`` must be JSON-serialisable; text fields are redacted and clipped."""
    if not config.AUDIT_ENABLED:
        return
    entry = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "event": event, "session": session_id}
    for key, value in fields.items():
        entry[key] = _clip(value) if isinstance(value, str) else value
    try:
        with _lock:
            config.AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
            _rotate(config.AUDIT_PATH)
            with config.AUDIT_PATH.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        log.exception("Could not write the audit log")


def query_event(session_id: str, question: str, answer: str, retrieval, generation_latency: float, grounding: dict | None) -> None:
    """Log one answered question with the passages that were used."""
    record(
        "question",
        session_id,
        question=question,
        standalone_question=retrieval.standalone_question if retrieval.standalone_question != question else "",
        answer=answer,
        confidence="Low" if retrieval.declined else retrieval.confidence,
        outcome="blocked" if retrieval.blocked else ("smalltalk" if retrieval.smalltalk else ("declined" if retrieval.declined else "answered")),
        grounding=(grounding or {}).get("status", ""),
        unsupported_numbers=(grounding or {}).get("unsupported_numbers", []),
        injection_flags=retrieval.injection_flags,
        sources=[
            {
                "document_id": s["document_id"],
                "title": s["title"],
                "chunk": f"{s['chunk_index'] + 1}/{s['total_chunks']}",
                "relevance": round(s["hybrid_score"], 3),
                "rerank": round(s["rerank_score"], 3) if s.get("rerank_score") is not None else None,
            }
            for s in retrieval.sources
        ],
        retrieval_ms=round(retrieval.latency * 1000),
        generation_s=round(generation_latency, 1),
    )


def read_recent(limit: int = 25) -> list[dict]:
    """The most recent events, newest first."""
    path = config.AUDIT_PATH
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()[-limit:]
    except OSError:
        return []
    events = []
    for line in reversed(lines):
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events
