"""Compare two transcripts: find where they disagree and estimate error rates."""
import difflib
import html
import re

_WORD = r"\w\u0900-\u097F\u0600-\u06FF"
_STRIP = re.compile(f"[^{_WORD}]+")


def key_of(word, fold=None):
    w = _STRIP.sub("", word.lower())
    return fold(w) if fold else w


def align(a_text, b_text, fold=None):
    """Word-level alignment. Returns segments: {'kind': 'equal'|'diff', 'a': [...], 'b': [...]}."""
    a, b = a_text.split(), b_text.split()
    ka = [key_of(w, fold) for w in a]
    kb = [key_of(w, fold) for w in b]
    matcher = difflib.SequenceMatcher(None, ka, kb, autojunk=False)
    segments = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        segments.append({"kind": "equal" if tag == "equal" else "diff", "a": a[i1:i2], "b": b[j1:j2]})
    return segments


def summary(segments):
    equal = sum(len(s["a"]) for s in segments if s["kind"] == "equal")
    diff_words = sum(max(len(s["a"]), len(s["b"])) for s in segments if s["kind"] == "diff")
    count = sum(1 for s in segments if s["kind"] == "diff")
    total = equal + diff_words
    return {"disagreements": count, "agreement": (equal / total) if total else 1.0}


def highlight_html(segments, side):
    """HTML for one transcript with the disagreeing words highlighted."""
    parts = []
    for s in segments:
        words = s[side] if s["kind"] == "diff" else s["a" if side == "a" else "b"]
        text = html.escape(" ".join(words))
        if not text:
            continue
        if s["kind"] == "diff":
            parts.append(f'<mark style="background: rgba(255,170,0,0.40); color: inherit; padding: 0 2px; border-radius: 3px;">{text}</mark>')
        else:
            parts.append(text)
    return '<div style="line-height: 1.9; font-size: 0.95rem;">' + " ".join(parts) + "</div>"


def merge(segments, choices):
    """Build the final text. choices: {disagreement index: 'A' or 'B'} (default 'A')."""
    out, d = [], 0
    for s in segments:
        if s["kind"] == "equal":
            out += s["a"]
        else:
            out += s["a"] if choices.get(d, "A") == "A" else s["b"]
            d += 1
    return " ".join(out)


def error_rate(reference, hypothesis, fold=None):
    """Approximate word error rate of `hypothesis` against `reference` (None if no reference).
    Uses the same alignment as the comparison, so it can differ slightly from a strict
    minimum-edit WER."""
    ref = [key_of(w, fold) for w in reference.split()]
    hyp = [key_of(w, fold) for w in hypothesis.split()]
    if not ref:
        return None
    matcher = difflib.SequenceMatcher(None, ref, hyp, autojunk=False)
    errors = sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in matcher.get_opcodes() if tag != "equal")
    return errors / len(ref)
