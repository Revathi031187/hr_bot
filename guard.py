"""Prompt-injection guard.

Text that reaches the language model from outside (uploaded documents, retrieved passages, the user's own
question) is treated as untrusted data. Three layers work together:

1. ``neutralize`` removes instruction-like sentences ("ignore previous instructions", fake chat-template
   tokens, ...) from document text, both when a file is uploaded and again when passages are retrieved.
2. ``is_attack`` recognises a user question that tries to override the assistant, so it is refused without
   calling the model.
3. The prompt wraps every passage in ``<passage>`` tags and tells the model that their content is data only.

Pattern matching cannot catch every attack; it removes the common ones and the prompt rules cover the rest.
"""

from __future__ import annotations

import re

INJECTION_PATTERNS = [
    r"\b(ignore|disregard|forget|override|bypass)\b[^.\n]{0,40}\b(previous|prior|above|earlier|preceding|all|any|your|the|these|those)\b[^.\n]{0,30}\b(instructions?|rules?|prompts?|guidelines?|directions?|restrictions?|safeguards?)\b",
    r"\b(do not|don't|never)\s+(follow|obey|apply)\b[^.\n]{0,30}\b(instructions?|rules?|guidelines?)\b",
    r"\b(reveal|show|print|repeat|display|leak|output)\b[^.\n]{0,30}\b(system|hidden|initial|original|secret)\s+(prompt|instructions?|message)\b",
    r"\bsystem\s+prompt\b",
    r"\byou are now\b\s+(?:a|an|the|in|no longer|free|dan)\b",
    r"\bpretend\s+(?:to be|you are|that you)\b",
    r"\b(new|updated|revised)\s+instructions?\s*:",
    r"\b(developer|jailbreak|dan|god)\s+mode\b",
    r"\bjailbreak\b",
    r"\bact as (?:if you (?:are|were) )?(?:an? )?(?:unrestricted|unfiltered|jailbroken)\b",
    r"</?\s*(system|assistant|user|instructions?)\s*>",
    r"<\|[^|>\n]{1,24}\|>",
    r"\[/?INST\]",
    r"^\s*#{1,3}\s*(system|instruction)s?\b",
]
_COMPILED = [re.compile(pattern, re.IGNORECASE | re.MULTILINE) for pattern in INJECTION_PATTERNS]
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_INVISIBLE = re.compile(r"[​-‏‪-‮⁠-⁤﻿\x00-\x08\x0b\x0c\x0e-\x1f]")
REMOVED_MARK = "[instruction-like text removed]"

ATTACK_REPLY = (
    "I can only answer questions about company HR policy, and I can't change my rules or share my "
    "instructions. Please ask a question about leave, benefits, remote work, pay or conduct."
)


def is_attack(text: str) -> bool:
    """True when ``text`` contains an instruction-override pattern."""
    cleaned = _INVISIBLE.sub("", text)
    return any(pattern.search(cleaned) for pattern in _COMPILED)


def neutralize(text: str) -> tuple[str, list[str]]:
    """Remove instruction-like sentences from document text.

    Returns the cleaned text and the list of removed snippets (shortened), empty when the text is clean.
    """
    cleaned = _INVISIBLE.sub("", text)
    if not is_attack(cleaned):
        return cleaned, []

    removed: list[str] = []
    lines = []
    for line in cleaned.split("\n"):
        if not is_attack(line):
            lines.append(line)
            continue
        kept = []
        for sentence in _SENTENCE_END.split(line):
            if is_attack(sentence):
                removed.append(sentence.strip()[:120])
                kept.append(REMOVED_MARK)
            else:
                kept.append(sentence)
        lines.append(" ".join(kept))
    return "\n".join(lines), removed
