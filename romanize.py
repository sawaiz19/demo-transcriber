"""Hindi (Devanagari) -> Roman Hindi with LIPI (abhishekgautamm/lipi).

LIPI is a tiny (0.5M-parameter) CPU model trained on how people actually type Hindi
in English letters. It converts text of any length in well under a second, so there
is no chunking, timeout or truncation to worry about.
Only the Devanagari parts are sent to the model; English words and numbers stay as is.
"""
import re

import streamlit as st

MODEL_ID = "abhishekgautamm/lipi"
BACKENDS = {
    "Lexicon + model (recommended)": "map_lipi",
    "Model only": "lipi",
}

_DEV = "\u0900-\u0963\u0966-\u097f"  # Devanagari, without the danda
_RUN = re.compile(f"[{_DEV}]+(?: +[{_DEV}]+)*")
_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


@st.cache_resource(show_spinner="Loading LIPI (first run downloads the model, ~45 MB)...")
def load_lipi(backend):
    from huggingface_hub import snapshot_download
    from lipi import Transliterator

    model_dir = snapshot_download(MODEL_ID)
    if backend == "map_lipi":
        return Transliterator(
            backend="map_lipi", model_dir=model_dir, map_path=f"{model_dir}/lipi_map.json"
        )
    return Transliterator(model_dir=model_dir)


def romanize_hindi(text, backend="map_lipi"):
    t = load_lipi(backend)
    out = _RUN.sub(lambda m: t.transliterate_text(m.group()), text)
    out = out.replace("।", ".").replace("॥", ".").translate(_DIGITS)
    return re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), out)
