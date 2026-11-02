"""Speaker diarization, enrollment and recognition built on pyannote.audio.

Flow used by the app:
  1. diarize()         - whole-file diarization (global clustering, far more stable than
                         labelling short chunks one by one).
  2. label_speakers()  - one embedding centroid per diarized speaker, matched against the
                         enrolled profiles (one-to-one assignment), the rest become
                         "Anonymous 1", "Anonymous 2", ... in order of first appearance.

Enrollment accepts several voice samples per person; their windowed embeddings are averaged
into one robust profile.

Gated models: accept the terms (free) on Hugging Face for the diarization pipeline you use,
e.g. https://hf.co/pyannote/speaker-diarization-3.1 and https://hf.co/pyannote/segmentation-3.0,
then create a read token at https://hf.co/settings/tokens (sidebar field or HF_TOKEN env var).
"""
import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import streamlit as st
import torch

from audio import SR, audio_to_16k_mono

APP_DIR = Path(__file__).resolve().parent
DB_PATH = APP_DIR / "speakers.sqlite3"

DIARIZATION_MODELS = [
    "pyannote/speaker-diarization-3.1",
    "pyannote/speaker-diarization-community-1",
]
EMBEDDING_MODEL = "pyannote/wespeaker-voxceleb-resnet34-LM"
EMBED_TAG = "pyannote-wespeaker-v1"  # stored with every profile; profiles from other models are ignored

WIN_S, HOP_S = 6.0, 3.0          # sliding window used to embed long audio
MIN_EMBED_S = 0.8                # shorter audio is not embedded
SILENCE_RMS = 0.003


# --------------------------------- database ---------------------------------

def db_connect():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS speakers (
            speaker_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            embedding_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    cols = {r[1] for r in conn.execute("PRAGMA table_info(speakers)")}
    if "model" not in cols:  # migrate old (SpeechBrain ECAPA) databases
        conn.execute("ALTER TABLE speakers ADD COLUMN model TEXT NOT NULL DEFAULT 'ecapa-legacy'")
    if "samples" not in cols:
        conn.execute("ALTER TABLE speakers ADD COLUMN samples INTEGER NOT NULL DEFAULT 1")
    conn.commit()
    return conn


def save_speaker(name, embedding, samples=1):
    speaker_id = str(uuid.uuid4())
    with db_connect() as conn:
        conn.execute(
            "INSERT INTO speakers (speaker_id, name, embedding_json, created_at, model, samples) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                speaker_id,
                name.strip(),
                json.dumps(np.asarray(embedding, dtype=float).tolist()),
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                EMBED_TAG,
                int(samples),
            ),
        )
        conn.commit()
    return speaker_id


def load_embeddings():
    """[(speaker_id, name, embedding)] for profiles made with the current embedding model."""
    with db_connect() as conn:
        rows = conn.execute(
            "SELECT speaker_id, name, embedding_json FROM speakers WHERE model = ?", (EMBED_TAG,)
        ).fetchall()
    return [(sid, name, np.asarray(json.loads(emb), dtype=np.float32)) for sid, name, emb in rows]


def list_speakers():
    with db_connect() as conn:
        return conn.execute(
            "SELECT speaker_id, name, created_at FROM speakers ORDER BY created_at DESC"
        ).fetchall()


def legacy_profile_names():
    """Names of profiles created with the old SpeechBrain model (they must be re-enrolled)."""
    with db_connect() as conn:
        rows = conn.execute("SELECT name FROM speakers WHERE model != ?", (EMBED_TAG,)).fetchall()
    return [r[0] for r in rows]


def delete_speaker(speaker_id):
    with db_connect() as conn:
        conn.execute("DELETE FROM speakers WHERE speaker_id = ?", (speaker_id,))
        conn.commit()


# ---------------------------------- models ----------------------------------

def _device(choice="auto"):
    if choice != "cpu" and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def resolve_token(token=""):
    return (token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or "").strip() or None


@st.cache_resource(show_spinner="Loading pyannote diarization pipeline (first run downloads models)...")
def load_diarization_pipeline(model_id, token, device_str):
    try:
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise RuntimeError("Speaker diarization needs pyannote.audio. Run: pip install pyannote.audio") from exc

    try:
        pipeline = Pipeline.from_pretrained(model_id, token=token)
    except TypeError:  # pyannote.audio 3.x uses use_auth_token
        pipeline = Pipeline.from_pretrained(model_id, use_auth_token=token)
    except Exception as exc:
        raise RuntimeError(_access_help(model_id, exc)) from exc
    if pipeline is None:
        raise RuntimeError(_access_help(model_id, None))
    pipeline.to(torch.device(device_str))
    return pipeline


def _access_help(model_id, exc):
    return (
        f"Could not load '{model_id}'. Accept the model terms on Hugging Face "
        f"(https://hf.co/{model_id} and https://hf.co/pyannote/segmentation-3.0), then paste a "
        f"read token in the sidebar or set HF_TOKEN. {exc or ''}"
    ).strip()


@st.cache_resource(show_spinner="Loading speaker-embedding model...")
def load_embedder(token, device_str):
    from pyannote.audio import Inference, Model

    try:
        model = Model.from_pretrained(EMBEDDING_MODEL, token=token)
    except TypeError:
        model = Model.from_pretrained(EMBEDDING_MODEL, use_auth_token=token)
    if model is None:
        raise RuntimeError(_access_help(EMBEDDING_MODEL, None))
    return Inference(model, window="whole", device=torch.device(device_str))


# -------------------------------- embeddings --------------------------------

def _unit(v):
    v = np.asarray(v, dtype=np.float32).flatten()
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else None


def _embed_array(embedder, samples):
    wav = torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32)).unsqueeze(0)
    emb = embedder({"waveform": wav, "sample_rate": SR})
    return _unit(emb)


def embed_audio(embedder, audio, max_windows=40):
    """Robust embedding of arbitrary-length audio: average of normalized window embeddings
    (silent windows skipped, evenly subsampled when there are many)."""
    win, hop = int(WIN_S * SR), int(HOP_S * SR)
    n = len(audio)
    if n < MIN_EMBED_S * SR:
        return None
    if n <= win:
        starts = [0]
        win = n
    else:
        starts = list(range(0, n - win + 1, hop))
        if starts[-1] + win < n:
            starts.append(n - win)
    if len(starts) > max_windows:
        starts = [starts[i] for i in np.linspace(0, len(starts) - 1, max_windows).astype(int)]
    embs = []
    for s in starts:
        piece = audio[s:s + win]
        if float(np.sqrt(np.mean(piece ** 2))) < SILENCE_RMS:
            continue
        e = _embed_array(embedder, piece)
        if e is not None:
            embs.append(e)
    if not embs:
        return None
    return _unit(np.mean(embs, axis=0))


def make_embedding(audio_files, token="", device="auto"):
    """Enrollment: one or several voice samples -> (centroid embedding, total seconds, sample count)."""
    if not isinstance(audio_files, (list, tuple)):
        audio_files = [audio_files]
    embedder = load_embedder(resolve_token(token), str(_device(device)))
    embs, total = [], 0.0
    for f in audio_files:
        audio, duration = audio_to_16k_mono(f)
        if duration < 2:
            raise ValueError("Each sample needs at least 2 seconds of speech; 8-15 s is better.")
        e = embed_audio(embedder, audio)
        if e is None:
            raise ValueError("One of the samples contained no usable speech.")
        embs.append(e)
        total += duration
    return _unit(np.mean(embs, axis=0)), total, len(embs)


# -------------------------------- diarization -------------------------------

def diarize(audio, model_id=DIARIZATION_MODELS[0], token="", device="auto",
            num_speakers=None, min_speakers=None, max_speakers=None):
    """Whole-recording diarization. Returns [{'start','end','label'}] sorted by time,
    non-overlapping where the pipeline offers an exclusive output."""
    pipeline = load_diarization_pipeline(model_id, resolve_token(token), str(_device(device)))
    wav = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).unsqueeze(0)
    kwargs = {}
    if num_speakers:
        kwargs["num_speakers"] = int(num_speakers)
    else:
        if min_speakers:
            kwargs["min_speakers"] = int(min_speakers)
        if max_speakers:
            kwargs["max_speakers"] = int(max_speakers)
    out = pipeline({"waveform": wav, "sample_rate": SR}, **kwargs)

    annotation = getattr(out, "exclusive_speaker_diarization", None)
    if annotation is None:
        annotation = getattr(out, "speaker_diarization", out)
    turns = [
        {"start": float(seg.start), "end": float(seg.end), "label": str(label)}
        for seg, _, label in annotation.itertracks(yield_label=True)
    ]
    turns.sort(key=lambda t: (t["start"], t["end"]))
    return turns


def merge_turns(turns, max_gap=0.6, min_len=0.3):
    """Join consecutive turns of the same speaker separated by short gaps; drop slivers."""
    merged = []
    for t in turns:
        if merged and merged[-1]["label"] == t["label"] and t["start"] - merged[-1]["end"] <= max_gap:
            merged[-1]["end"] = max(merged[-1]["end"], t["end"])
        else:
            merged.append(dict(t))
    return [t for t in merged if t["end"] - t["start"] >= min_len]


# --------------------------- identifying the speakers -------------------------

def label_speakers(audio, turns, enrolled, threshold=0.5, token="", device="auto"):
    """Map each diarization label to (display name, similarity, is_enrolled).

    Each label gets ONE centroid from its longest turns. Enrolled profiles are assigned
    one-to-one (Hungarian) so two different voices cannot both claim the same person; leftover
    clusters can still join an enrolled profile when the match is clearly strong.
    """
    labels = []
    for t in turns:  # order of first appearance
        if t["label"] not in labels:
            labels.append(t["label"])

    embedder = None
    centroids = {}
    if enrolled:
        embedder = load_embedder(resolve_token(token), str(_device(device)))
        for lab in labels:
            own = sorted((t for t in turns if t["label"] == lab), key=lambda t: t["end"] - t["start"], reverse=True)
            embs = []
            for t in own[:12]:
                piece = audio[int(t["start"] * SR):int(t["end"] * SR)]
                e = embed_audio(embedder, piece, max_windows=6)
                if e is not None:
                    embs.append(e)
            centroids[lab] = _unit(np.mean(embs, axis=0)) if embs else None

    result = {}
    if enrolled and centroids:
        usable = [l for l in labels if centroids.get(l) is not None]
        if usable:
            sim = np.array([[float(np.dot(centroids[l], e)) for _, _, e in enrolled] for l in usable])
            from scipy.optimize import linear_sum_assignment

            rows, cols = linear_sum_assignment(-sim)
            for r, c in zip(rows, cols):
                if sim[r, c] >= threshold:
                    result[usable[r]] = (enrolled[c][1], float(sim[r, c]), True)
            strong = min(0.95, threshold + 0.10)
            for r, lab in enumerate(usable):
                if lab not in result:
                    c = int(np.argmax(sim[r]))
                    if sim[r, c] >= strong:
                        result[lab] = (enrolled[c][1], float(sim[r, c]), True)

    n_anon = 0
    for lab in labels:
        if lab not in result:
            n_anon += 1
            result[lab] = (f"Anonymous {n_anon}", 1.0, False)
    return result