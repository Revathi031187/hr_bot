"""The model must not copy example 'documents do not mention X' wording into unrelated answers."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import rag_pipeline as rp

Q = "How must a harassment or discrimination complaint be handled?"


def test_prompt_has_no_concrete_example():
    assert "LumiX" not in rp.SYSTEM_PROMPT


def test_stray_sentence_removed():
    text = "1. All complaints are investigated.\n\nThe documents do not mention LumiX.\n"
    out = rp.drop_stray_missing_notes(text, Q)
    assert "LumiX" not in out and "investigated" in out


def test_real_missing_topic_kept():
    q = "What is the policy on stock options and leave?"
    text = "Leave is 20 days. The documents do not mention stock options."
    assert "stock options" in rp.drop_stray_missing_notes(text, q)


def test_streaming_filter_across_fragments():
    f = rp._LineFilter(Q)
    out = "".join(f.feed(x) for x in ["Answer here.\n\nThe docu", "ments do not men", "tion LumiX.", "\n"]) + f.flush()
    assert "LumiX" not in out and "Answer here." in out


if __name__ == "__main__":
    for n, fn in list(globals().items()):
        if n.startswith("test_"):
            fn()
    print("ALL STRAY-NOTE TESTS PASSED")
