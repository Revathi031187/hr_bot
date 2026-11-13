"""Answer-grounding check.

After the model answers, compare the answer with the passages it was given. Small models sometimes state a
number or a fact that is not in the text, so this check looks for:

* figures (numbers, amounts, percentages, dates) in the answer that appear nowhere in the passages, and
* the share of the answer's substantive words that occur in the passages.

It is a fast lexical check, not a proof: "verified" means the wording is supported by the passages, not that the
reasoning is right. Nothing here calls the model, so it adds no noticeable latency.
"""

from __future__ import annotations

import re

_NUMBER = re.compile(r"(?<![\w.])\$?(\d[\d,]*(?:\.\d+)?)\s*%?")
_LIST_MARKER = re.compile(r"^\s*(?:\d{1,2}[.)]|[-*•])\s+", re.MULTILINE)
_CITATION = re.compile(r"\(Policy:[^)]*\)", re.IGNORECASE)
_WORD = re.compile(r"[A-Za-z][A-Za-z'-]{4,}")
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3})")

# Words that carry no policy facts (and generic words the model uses to frame an answer).
_FILLER = {
    "about", "above", "across", "after", "again", "against", "also", "although", "always", "among", "answer",
    "because", "before", "being", "below", "between", "both", "cannot", "could", "during", "either", "every",
    "following", "further", "gives", "given", "however", "include", "includes", "including", "likely", "other",
    "others", "please", "policy", "policies", "provided", "provides", "please", "refers", "should", "since",
    "specific", "state", "states", "stated", "their", "there", "these", "those", "through", "under", "until",
    "which", "while", "would", "where", "within", "without", "document", "documents", "context", "section",
    "lasts", "topped", "offers", "offered", "applies", "allows", "allowed", "means", "covers", "covered", "mentioned", "information", "employee", "employees", "company", "generally", "typically", "overall",
}


def _norm_numbers(text: str) -> set[str]:
    text = _THOUSANDS.sub("", text)
    return {m.group(1).rstrip(".,").lstrip("0") or "0" for m in _NUMBER.finditer(text)}


def _prefixes(text: str) -> set[str]:
    return {word.lower()[:5] for word in _WORD.findall(text)}


def check(answer: str, context: str) -> dict:
    """Return ``{"status", "coverage", "unsupported_numbers"}`` for ``answer`` against ``context``.

    status is one of ``verified``, ``partial``, ``unverified`` or ``check_figures``.
    """
    body = _CITATION.sub(" ", answer)
    body = _LIST_MARKER.sub("", body).replace("**", "").replace("__", "")

    context_numbers = _norm_numbers(context)
    unsupported = []
    for number in sorted(_norm_numbers(body), key=lambda n: (len(n), n)):
        if number not in context_numbers:
            unsupported.append(number)

    context_prefixes = _prefixes(context)
    words = [w for w in _WORD.findall(body) if w.lower() not in _FILLER]
    if words:
        supported = sum(1 for w in words if w.lower()[:5] in context_prefixes)
        coverage = supported / len(words)
    else:
        coverage = 1.0

    if unsupported:
        status = "check_figures"
    elif coverage >= 0.7:
        status = "verified"
    elif coverage >= 0.5:
        status = "partial"
    else:
        status = "unverified"
    return {"status": status, "coverage": round(coverage, 2), "unsupported_numbers": unsupported[:6]}


def describe(result: dict) -> tuple[str, str]:
    """(tag text, css level good|warn|bad) for the UI."""
    status = result["status"]
    if status == "verified":
        return "Grounded in sources", "good"
    if status == "partial":
        return "Partly grounded", "warn"
    if status == "check_figures":
        figures = ", ".join(result["unsupported_numbers"])
        return f"Check figures: {figures}", "bad"
    return "Not grounded: verify with HR", "bad"
