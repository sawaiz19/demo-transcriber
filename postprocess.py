"""Text clean-up applied after transcription / Roman conversion:

1. Custom word list  - your own find -> replace pairs (names, brands, technical terms).
2. Spelling standard - one Roman spelling per common word (kay/ke -> ke, bohat -> bahut ...).

Both are stored as small JSON files next to the app and are editable in the 'Word lists' tab.
Matching is whole-word and ignores capitalisation.
"""
import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
WORDLIST_PATH = APP_DIR / "wordlist.json"
SPELLING_PATH = APP_DIR / "spelling.json"

# Starter list. Only variants that are NOT ordinary English words are included
# (for example "he", "key", "say" or "ham" are deliberately left out).
DEFAULT_SPELLING = {
    "hai": ["hy", "hay", "hae"],
    "nahi": ["nahin", "nhi", "nahee"],
    "theek": ["thik", "theak", "thek", "teek"],
    "bahut": ["bohat", "bohot", "bhut", "bahot", "bohut"],
    "accha": ["acha", "achha", "achchha", "acchha"],
    "kya": ["kia", "kyaa"],
    "kyun": ["kyon", "kiun", "kion", "kyu"],
    "kyunki": ["kyuki", "kyonki", "kiunke"],
    "kuch": ["kuchh", "kch"],
    "liye": ["liay", "lie"],
    "mujhe": ["mujhy", "mujhay", "mujhey"],
    "aap": ["ap"],
    "bhi": ["bhee"],
    "woh": ["wo", "vo", "voh"],
    "kaise": ["kese", "kaisay", "kaisey"],
    "abhi": ["abi"],
    "agar": ["agr"],
    "lekin": ["lakin", "laikin"],
    "saath": ["sath"],
    "zyada": ["zyaada", "ziada", "jyada", "zayada"],
    "karna": ["krna"],
    "karke": ["krke"],
    "raha": ["rha", "rahaa"],
    "rahi": ["rhi"],
    "rahe": ["rhe"],
    "gaya": ["gya"],
    "gayi": ["gyi"],
    "gaye": ["gye"],
    "jaise": ["jese", "jaisay"],
    "isliye": ["islye", "islie"],
}
DEFAULT_SPELLING["liye"].remove("lie")  # "lie" is a real English word

# Characters that count as part of a word (Latin, Devanagari, Urdu script)
_WORD = r"\w\u0900-\u097F\u0600-\u06FF"
_PUNCT = re.compile(f"[^{_WORD}]+")


# ------------------------------ load / save ---------------------------------

def _read_json(path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def load_wordlist():
    """[{'find': ..., 'replace': ...}, ...]"""
    rows = _read_json(WORDLIST_PATH, [])
    return [r for r in rows if isinstance(r, dict) and str(r.get("find", "")).strip()]


def save_wordlist(rows):
    clean = [
        {"find": str(r["find"]).strip(), "replace": str(r.get("replace", ""))}
        for r in rows if str(r.get("find", "")).strip()
    ]
    WORDLIST_PATH.write_text(json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf-8")


def load_spelling():
    """{standard spelling: [variants]}"""
    data = _read_json(SPELLING_PATH, None)
    return data if isinstance(data, dict) else {k: list(v) for k, v in DEFAULT_SPELLING.items()}


def save_spelling(mapping):
    clean = {}
    for standard, variants in mapping.items():
        standard = str(standard).strip()
        if standard:
            clean[standard] = [v.strip() for v in variants if v.strip()]
    SPELLING_PATH.write_text(json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf-8")


def reset_spelling():
    save_spelling(DEFAULT_SPELLING)


def signature(use_spelling, use_wordlist):
    """Changes whenever the lists or the on/off switches change (used for widget keys)."""
    blob = json.dumps([use_spelling, use_wordlist, load_wordlist(), load_spelling()], sort_keys=True)
    return hashlib.md5(blob.encode()).hexdigest()[:10]


# -------------------------------- matching ----------------------------------

@lru_cache(maxsize=64)
def _compile(keys):
    ordered = sorted(keys, key=len, reverse=True)  # longest first, so phrases win over words
    pattern = "|".join(re.escape(k) for k in ordered)
    return re.compile(f"(?<![{_WORD}])(?:{pattern})(?![{_WORD}])", re.IGNORECASE)


def _match_case(original, standard):
    if original.isupper() and len(original) > 1:
        return standard.upper()
    if original[:1].isupper():
        return standard[:1].upper() + standard[1:]
    return standard


def variant_map(spelling):
    return {v.lower(): std for std, variants in spelling.items() for v in variants if v.strip()}


def apply_spelling(text, spelling):
    vmap = variant_map(spelling)
    if not vmap:
        return text
    rx = _compile(tuple(vmap))
    return rx.sub(lambda m: _match_case(m.group(), vmap[m.group().lower()]), text)


def apply_wordlist(text, rows):
    rmap = {str(r["find"]).strip().lower(): str(r.get("replace", "")) for r in rows if str(r.get("find", "")).strip()}
    if not rmap:
        return text
    rx = _compile(tuple(rmap))
    return rx.sub(lambda m: rmap[m.group().lower()], text)


def make_folder(spelling):
    """Returns fold(word) -> a key where spelling variants collapse to one value (for comparing)."""
    vmap = variant_map(spelling)
    return lambda w: vmap.get(w, w)
