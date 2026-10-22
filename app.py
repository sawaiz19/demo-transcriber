import io
import json
import os
import sqlite3
import tempfile
import uuid
from datetime import datetime, timezone
from math import gcd

import numpy as np
import soundfile as sf
import streamlit as st
import torch
import torch.nn.functional as F
from scipy.signal import resample_poly
from faster_whisper import WhisperModel
from speechbrain.inference.speaker import SpeakerRecognition

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(APP_DIR, "speakers.sqlite3")
SPEAKER_MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"

st.set_page_config(page_title="Urdu + English Speech & Speaker ID", page_icon="🎙️", layout="wide")


def db_connect():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS speakers (
            speaker_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            embedding_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def save_speaker(name, embedding):
    speaker_id = str(uuid.uuid4())
    with db_connect() as conn:
        conn.execute(
            "INSERT INTO speakers (speaker_id, name, embedding_json, created_at) VALUES (?, ?, ?, ?)",
            (speaker_id, name.strip(), json.dumps(embedding.astype(float).tolist()),
             datetime.now(timezone.utc).isoformat(timespec="seconds")),
        )
        conn.commit()
    return speaker_id


def load_embeddings():
    with db_connect() as conn:
        rows = conn.execute("SELECT speaker_id, name, embedding_json FROM speakers").fetchall()
    return [(sid, name, np.asarray(json.loads(e), dtype=np.float32)) for sid, name, e in rows]


def list_speakers():
    with db_connect() as conn:
        return conn.execute("SELECT speaker_id, name, created_at FROM speakers ORDER BY created_at DESC").fetchall()


def delete_speaker(speaker_id):
    with db_connect() as conn:
        conn.execute("DELETE FROM speakers WHERE speaker_id = ?", (speaker_id,))
        conn.commit()


@st.cache_resource(show_spinner="Loading multilingual speech-to-text model (first run downloads model files)...")
def load_whisper_model(model_size):
    # CPU-friendly defaults; int8 reduces memory use. Model downloads on first use.
    return WhisperModel(model_size, device="cpu", compute_type="int8")


@st.cache_resource(show_spinner="Loading speaker-recognition model (first run can take a few minutes)...")
def load_speaker_model():
    return SpeakerRecognition.from_hparams(
        source=SPEAKER_MODEL_SOURCE,
        savedir=os.path.join(APP_DIR, "pretrained_models", "spkrec-ecapa-voxceleb"),
        run_opts={"device": "cpu"},
    )


def audio_to_16k_mono(audio_file):
    """Decode common uploaded audio formats and return mono 16 kHz samples."""
    raw = audio_file.getvalue() if hasattr(audio_file, "getvalue") else audio_file
    try:
        audio, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
    except Exception:
        # PyAV (installed with faster-whisper) handles formats such as MP3, M4A, and WebM.
        import av
        chunks = []
        with av.open(io.BytesIO(raw)) as container:
            stream = next((s for s in container.streams if s.type == "audio"), None)
            if stream is None:
                raise ValueError("No audio track was found in this file.")
            sr = int(stream.rate or 16000)
            resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
            for frame in container.decode(stream):
                for converted in resampler.resample(frame):
                    chunks.append(converted.to_ndarray().reshape(-1).astype(np.float32))
            for converted in resampler.resample(None):
                chunks.append(converted.to_ndarray().reshape(-1).astype(np.float32))
        if not chunks:
            raise ValueError("Could not decode audio from this file.")
        audio = np.concatenate(chunks)
        sr = 16000
    if sr != 16000:
        g = gcd(int(sr), 16000)
        audio = resample_poly(audio, 16000 // g, sr // g).astype(np.float32)
    if len(audio) == 0:
        raise ValueError("The audio file is empty.")
    return np.asarray(audio, dtype=np.float32), len(audio) / 16000


def make_embedding(audio_file):
    audio, duration = audio_to_16k_mono(audio_file)
    if duration < 2.0:
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


def transcribe_audio(audio_file, language, model_size):
    raw = audio_file.getvalue() if hasattr(audio_file, "getvalue") else audio_file
    # Preserve the uploaded extension so FFmpeg/PyAV can identify the format correctly.
    filename = getattr(audio_file, "name", "audio.wav")
    suffix = os.path.splitext(filename)[1].lower() or ".wav"
    if suffix not in {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".mp4", ".mpeg", ".mpga"}:
        suffix = ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(raw)
        temp_path = tmp.name
    try:
        model = load_whisper_model(model_size)
        language_code = None if language == "Auto-detect" else {"Urdu": "ur", "English": "en"}[language]
        segments, info = model.transcribe(
            temp_path,
            language=language_code,
            beam_size=5,
            vad_filter=True,
        )
        segments = list(segments)
        text = " ".join(s.text.strip() for s in segments).strip()
        return text, info.language, float(info.language_probability), [
            {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
            for s in segments
        ]
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


st.title("🎙️ Urdu + English Speech-to-Text & Speaker Recognition")
st.caption("Local prototype: transcribe spoken Urdu/English and compare a recording with consented speaker profiles.")

with st.sidebar:
    st.header("Model settings")
    model_size = st.selectbox(
        "Speech-to-text model",
        ["tiny", "base", "small", "medium"],
        index=2,
        help="Larger models are usually more accurate but need more RAM and run more slowly. All listed models are multilingual.",
    )
    st.caption("The first run downloads model files. Keep this app local until you add authentication and privacy protections.")

tab_transcribe, tab_enroll, tab_identify, tab_profiles = st.tabs(
    ["Speech to text", "Enroll speaker", "Recognize speaker", "Saved profiles"]
)

with tab_transcribe:
    st.subheader("Transcribe spoken Urdu or English")
    language = st.radio("Spoken language", ["Auto-detect", "Urdu", "English"], horizontal=True)
    audio_source = st.radio("Audio source", ["Record with microphone", "Upload an audio file"], horizontal=True, key="transcribe_source")
    if audio_source == "Record with microphone":
        audio = st.audio_input("Record speech (5–30 seconds works well)", sample_rate=16000, key="transcribe_audio")
    else:
        audio = st.file_uploader(
            "Choose an audio file", type=["wav", "mp3", "m4a", "flac", "ogg", "webm", "mp4", "mpeg", "mpga"],
            key="transcribe_upload", help="Supported formats include WAV, MP3, M4A, FLAC, OGG, and WebM."
        )
    if audio:
        st.audio(audio)
    if st.button("Transcribe audio", type="primary"):
        if not audio:
            st.error("Record speech or upload an audio file first.")
        else:
            try:
                with st.spinner("Transcribing audio..."):
                    text, detected_language, language_probability, segments = transcribe_audio(audio, language, model_size)
                if text:
                    st.success(f"Transcription complete · Detected language: {detected_language} ({language_probability:.0%} model confidence)")
                    st.text_area("Recognized text", value=text, height=180)
                    st.download_button("Download transcript (.txt)", data=text, file_name="transcript.txt", mime="text/plain")
                    with st.expander("Show segment timestamps"):
                        for seg in segments:
                            st.write(f"{seg['start']:.2f}s – {seg['end']:.2f}s: {seg['text']}")
                else:
                    st.warning("No speech was recognized. Try a longer, clearer recording.")
            except Exception as e:
                st.error(f"Transcription failed: {e}")
                st.caption("Check that the audio is a valid WAV recording and that the model finished downloading.")

with tab_enroll:
    st.subheader("Create a speaker profile")
    st.warning("Only enroll people who have explicitly agreed to have their voice processed and stored.")
    speaker_name = st.text_input("Profile name or label", placeholder="e.g., Speaker 1")
    consent = st.checkbox("I have permission to record and store this person's voice profile.")
    enroll_source = st.radio("Voice sample source", ["Record with microphone", "Upload an audio file"], horizontal=True, key="enroll_source")
    if enroll_source == "Record with microphone":
        enroll_audio = st.audio_input("Record 5–10 seconds of natural speech", sample_rate=16000, key="enroll_audio")
    else:
        enroll_audio = st.file_uploader(
            "Upload a voice sample", type=["wav", "mp3", "m4a", "flac", "ogg", "webm", "mp4", "mpeg", "mpga"],
            key="enroll_upload", help="Use a clear recording with at least 2 seconds of speech; 5–10 seconds is better."
        )
    if enroll_audio:
        st.audio(enroll_audio)
    if st.button("Create speaker profile", type="primary"):
        if not consent:
            st.error("Please confirm consent before creating a profile.")
        elif not speaker_name.strip():
            st.error("Enter a profile name.")
        elif not enroll_audio:
            st.error("Record a voice sample first.")
        else:
            try:
                with st.spinner("Analyzing voice; the first run downloads the model..."):
                    emb, duration = make_embedding(enroll_audio)
                    sid = save_speaker(speaker_name, emb)
                st.success(f"Profile created for **{speaker_name.strip()}**.")
                st.code(sid, language=None)
                st.caption(f"Processed {duration:.1f} seconds of audio.")
            except Exception as e:
                st.error(f"Could not create profile: {e}")

with tab_identify:
    st.subheader("Check whether a speaker is known")
    profiles = load_embeddings()
    if not profiles:
        st.info("No profiles yet. Create one in the 'Enroll speaker' tab first.")
    threshold = st.slider(
        "Similarity threshold (demo only; calibrate with your own recordings)",
        min_value=0.10, max_value=0.90, value=0.35, step=0.01,
        help="This is not a probability. Tune it using recordings of same and different speakers.",
    )
    identify_source = st.radio("Comparison audio source", ["Record with microphone", "Upload an audio file"], horizontal=True, key="identify_source")
    if identify_source == "Record with microphone":
        identify_audio = st.audio_input("Record 5–10 seconds of speech to compare", sample_rate=16000, key="identify_audio")
    else:
        identify_audio = st.file_uploader(
            "Upload a recording to compare", type=["wav", "mp3", "m4a", "flac", "ogg", "webm", "mp4", "mpeg", "mpga"],
            key="identify_upload", help="Use a clear recording with at least 2 seconds of speech; 5–10 seconds is better."
        )
    if identify_audio:
        st.audio(identify_audio)
    if st.button("Identify speaker", type="primary"):
        if not identify_audio:
            st.error("Record a voice sample first.")
        elif not profiles:
            st.error("There are no enrolled speaker profiles to compare against.")
        else:
            try:
                with st.spinner("Comparing voice with saved profiles..."):
                    query_emb, duration = make_embedding(identify_audio)
                    scores = sorted(
                        [(cosine_similarity(query_emb, emb), sid, name) for sid, name, emb in load_embeddings()],
                        reverse=True,
                    )
                score, sid, name = scores[0]
                st.metric("Best similarity score", f"{score:.3f}")
                if score >= threshold:
                    st.success(f"Likely returning speaker: **{name}**")
                    st.caption(f"Profile ID: {sid}")
                else:
                    st.warning("No enrolled speaker matched this threshold. Treat the speaker as unknown.")
                st.caption(
                    f"Compared {duration:.1f} seconds. Scores are not probabilities; "
                    "validate performance on Urdu and English recordings from your intended users."
                )
                with st.expander("Show all comparison scores"):
                    for s, profile_id, profile_name in scores:
                        st.write(f"{profile_name}: {s:.3f} · ID {profile_id}")
            except Exception as e:
                st.error(f"Could not identify speaker: {e}")

with tab_profiles:
    st.subheader("Manage saved speaker profiles")
    profiles = list_speakers()
    if not profiles:
        st.info("No saved profiles.")
    else:
        st.caption("Profiles are stored in a local SQLite database on this computer.")
        for sid, name, created_at in profiles:
            with st.container(border=True):
                left, right = st.columns([4, 1])
                with left:
                    st.write(f"**{name}**")
                    st.caption(f"Created: {created_at}")
                    st.code(sid, language=None)
                with right:
                    confirm = st.checkbox("Confirm delete", key=f"confirm_{sid}")
                    if st.button("Delete", key=f"delete_{sid}", disabled=not confirm):
                        delete_speaker(sid)
                        st.success(f"Deleted profile: {name}")
                        st.rerun()

st.divider()
st.caption(
    "Prototype limitations: Whisper is multilingual, but Urdu accuracy varies with accent, recording quality, and code-switching. "
    "The speaker model is trained mainly on VoxCeleb data, so Urdu speaker matching must be validated. "
    "This app compares one speaker per recording; it does not perform multi-speaker diarization or liveness detection."
)
