"""Speaker enrollment / recognition (unchanged behaviour, moved out of app.py)."""
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F

from audio import audio_to_16k_mono

APP_DIR = Path(__file__).resolve().parent
DB_PATH = APP_DIR / "speakers.sqlite3"
SPEAKER_MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"


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
    conn.commit()
    return conn


def save_speaker(name, embedding):
    speaker_id = str(uuid.uuid4())
    with db_connect() as conn:
        conn.execute(
            "INSERT INTO speakers (speaker_id, name, embedding_json, created_at) VALUES (?, ?, ?, ?)",
            (
                speaker_id,
                name.strip(),
                json.dumps(embedding.astype(float).tolist()),
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
            ),
        )
        conn.commit()
    return speaker_id


def load_embeddings():
    with db_connect() as conn:
        rows = conn.execute("SELECT speaker_id, name, embedding_json FROM speakers").fetchall()
    return [(sid, name, np.asarray(json.loads(emb), dtype=np.float32)) for sid, name, emb in rows]


def list_speakers():
    with db_connect() as conn:
        return conn.execute(
            "SELECT speaker_id, name, created_at FROM speakers ORDER BY created_at DESC"
        ).fetchall()


def delete_speaker(speaker_id):
    with db_connect() as conn:
        conn.execute("DELETE FROM speakers WHERE speaker_id = ?", (speaker_id,))
        conn.commit()


@st.cache_resource(show_spinner="Loading speaker-recognition model (first run downloads it)...")
def load_speaker_model():
    # Imported lazily so the app starts faster when speaker features are not used.
    from speechbrain.inference.speaker import SpeakerRecognition
    from speechbrain.utils.fetching import LocalStrategy

    return SpeakerRecognition.from_hparams(
        source=SPEAKER_MODEL_SOURCE,
        savedir=str(APP_DIR / "pretrained_models" / "spkrec-ecapa-voxceleb"),
        run_opts={"device": "cpu"},
        local_strategy=LocalStrategy.COPY,  # avoids Windows symlink privileges
    )


def make_embedding(audio_file):
    audio, duration = audio_to_16k_mono(audio_file)
    if duration < 2:
        raise ValueError("Please provide at least 2 seconds of speech; 5–10 seconds is better.")
    model = load_speaker_model()
    signal = torch.from_numpy(audio).unsqueeze(0)
    with torch.no_grad():
        embedding = model.encode_batch(signal).squeeze().cpu().numpy().astype(np.float32)
    norm = np.linalg.norm(embedding)
    if norm:
        embedding = embedding / norm
    return embedding, duration


def cosine_similarity(a, b):
    at = torch.tensor(a, dtype=torch.float32).flatten()
    bt = torch.tensor(b, dtype=torch.float32).flatten()
    return float(F.cosine_similarity(at.unsqueeze(0), bt.unsqueeze(0)).item())
