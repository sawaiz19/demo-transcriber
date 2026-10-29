import hashlib
import time

import streamlit as st

import asr
import romanize
import speakers
from audio import SR, audio_to_16k_mono, split_audio

AUDIO_TYPES = ["wav", "mp3", "m4a", "flac", "ogg", "webm", "mp4", "mpeg", "mpga"]

st.set_page_config(page_title="Roman Hindi Transcriber", page_icon="🎙️", layout="wide")


def md5(data):
    return hashlib.md5(data if isinstance(data, bytes) else data.encode()).hexdigest()


# ------------------------------- sidebar ------------------------------------

with st.sidebar:
    st.header("Transcription")
    model_label = st.selectbox(
        "Speech-recognition model",
        list(asr.ENGINES),
        help="Only one model is kept in memory at a time, so switching frees the previous one.",
    )
    st.caption(asr.ENGINES[model_label]["note"])
    language_choice = st.radio(
        "ASR language hint", list(asr.LANGUAGES), index=0,
        help="Used by Qwen, Tara and Shruti. Ignored by Whisper-tiny Hinglish.",
    )
    st.header("Hardware")
    device_label = st.selectbox(
        "Run on", list(asr.DEVICE_CHOICES),
        help="Auto uses your NVIDIA GPU when PyTorch can see it. Shruti always runs on CPU.",
    )
    device_choice = asr.DEVICE_CHOICES[device_label]
    gpu = asr.gpu_info()
    if gpu:
        st.success(f"GPU ready: {gpu[0]} ({gpu[1]:.1f} GB VRAM)")
        if gpu[1] < 5.5 and any(k in model_label for k in ("Tara", "1.7B")):
            st.warning(
                "This model is tight on a GPU with this little VRAM. If you get an out-of-memory "
                "error, set Precision to float16 or choose a smaller model."
            )
    else:
        if asr.gpu_problem() == "cpu_build":
            st.warning(
                "This PyTorch install is CPU-only, so the NVIDIA GPU can't be used. Install the "
                "CUDA build of PyTorch from pytorch.org (Pip, Windows, CUDA), then restart the app."
            )
        else:
            st.warning(
                "PyTorch can't see an NVIDIA GPU. Update your NVIDIA driver and restart the app. "
                "Running on CPU for now."
            )
    with st.expander("Advanced"):
        dtype_name = st.selectbox(
            "Precision",
            ["auto", "bfloat16", "float16", "float32"],
            help="Applies to Qwen and Tara. Auto picks bfloat16 (or float16 on older GPUs). "
            "float16 uses the same memory and works if bfloat16 gives problems.",
        )

    st.divider()
    st.header("Roman Hindi conversion")
    backend_label = st.radio(
        "LIPI mode",
        list(romanize.BACKENDS),
        help="'Lexicon + model' looks words up in LIPI's curated word list first and uses the "
        "model for anything not found. 'Model only' skips the lookup.",
    )

tab_transcribe, tab_enroll, tab_identify, tab_profiles = st.tabs(
    ["Transcribe audio", "Enroll speaker", "Recognize speaker", "Saved profiles"]
)

# ---------------------------- transcribe tab --------------------------------

with tab_transcribe:
    st.title("🎙️ Speech to Roman Hindi")
    st.caption(
        "Long recordings are split into ~30 s pieces at pauses, transcribed one by one with "
        "progress, and can be resumed if something interrupts them."
    )
    audio = st.file_uploader("Upload audio", type=AUDIO_TYPES, key="transcribe_upload")
    if audio:
        st.audio(audio)
        audio_bytes = audio.getvalue()
        job_key = md5(audio_bytes) + model_label + language_choice
        job = st.session_state.get("job")
        if not job or job["key"] != job_key:
            job = {"key": job_key, "audio": None, "bounds": None, "texts": [], "elapsed": 0.0}
            st.session_state["job"] = job

        done = job["bounds"] is not None and len(job["texts"]) == len(job["bounds"])
        started = bool(job["texts"])
        label = "Transcribe" if not started else ("Re-run" if done else "Resume transcription")

        if st.button(label, type="primary"):
            if done:
                job.update(texts=[], elapsed=0.0)
                st.session_state.pop("roman_llm", None)
            bar, status, live = st.progress(0.0), st.empty(), st.empty()
            t0 = time.perf_counter()
            try:
                if job["bounds"] is None:
                    status.info("Decoding audio...")
                    job["audio"], duration = audio_to_16k_mono(audio_bytes)
                    job["bounds"] = split_audio(job["audio"])
                    job["duration"] = duration
                status.info(f"Loading {model_label} (first run downloads the model)...")
                model = asr.get_engine(model_label, dtype_name, device_choice)
                job["ran_on"] = str(model.device)
                lang = asr.LANGUAGES[language_choice]
                total = len(job["bounds"])
                while len(job["texts"]) < total:
                    i = len(job["texts"])
                    status.info(f"Transcribing part {i + 1} of {total}...")
                    s, e = job["bounds"][i]
                    t_part = time.perf_counter()
                    job["texts"].append(asr.transcribe_chunk(model, job["audio"][s:e], lang))
                    last = time.perf_counter() - t_part
                    bar.progress(len(job["texts"]) / total)
                    live.caption(f"Last part: {last:.1f}s on {model.device} · …" + " ".join(job["texts"])[-500:])
                status.success(f"Finished {total} parts.")
                job["audio"] = None  # free memory once complete
            except Exception as exc:
                st.error(f"Stopped at part {len(job['texts']) + 1}: {exc}")
                st.caption(
                    "Everything transcribed so far is kept below. Click 'Resume transcription' "
                    "to continue. On low RAM, close other apps or pick a smaller model (Whisper-tiny Hinglish or Shruti)."
                )
            finally:
                job["elapsed"] += time.perf_counter() - t0
                bar.empty()
                live.empty()

        if job["texts"]:
            total = len(job["bounds"])
            st.success(
                f"{len(job['texts'])}/{total} parts transcribed · "
                f"{job['elapsed']:.0f}s of processing on {job.get('ran_on', '?')} · "
                f"audio length {job.get('duration', 0):.0f}s"
            )
            raw_joined = " ".join(t for t in job["texts"] if t)
            if asr.ENGINES[model_label]["script"] == "roman":
                st.subheader("Roman Hindi transcript")
                st.caption(f"{model_label.split(' (')[0]} writes Roman text directly, so no conversion step is needed.")
                final = st.text_area(
                    "Review and edit", value=raw_joined, height=260,
                    key=f"roman_direct_{job_key}_{len(job['texts'])}",
                )
                st.download_button(
                    "Download Roman Hindi transcript", final.rstrip() + "\n",
                    file_name="transcript_roman_hindi.txt", mime="text/plain",
                )
            else:
                st.subheader("1. Raw transcript (Hindi / Devanagari)")
                raw_edited = st.text_area(
                    "Edit if needed", value=raw_joined, height=200,
                    key=f"raw_{job_key}_{len(job['texts'])}",
                )
                st.download_button(
                    "Download raw transcript", raw_edited.rstrip() + "\n",
                    file_name="transcript_raw_hindi.txt", mime="text/plain",
                )

                st.subheader("2. Roman Hindi transcript")
                try:
                    roman = romanize.romanize_hindi(raw_edited, romanize.BACKENDS[backend_label])
                except ImportError:
                    roman = None
                    st.error(
                        "LIPI is not installed. Run: "
                        "pip install huggingface_hub git+https://github.com/ABHISHEKgauti25/lipi.git"
                    )
                except Exception as exc:
                    roman = None
                    st.error(f"Roman Hindi conversion failed: {exc}")
                if roman:
                    final = st.text_area(
                        "Review and edit", value=roman, height=220,
                        key=f"roman_{backend_label}_{md5(raw_edited)}",
                    )
                    st.download_button(
                        "Download Roman Hindi transcript", final.rstrip() + "\n",
                        file_name="transcript_roman_hindi.txt", mime="text/plain",
                    )
    st.caption(
        "Always review names, numbers and unclear words. Qwen3-ASR does not officially list Urdu, "
        "so Urdu speech is recognised via the Hindi setting."
    )

# --------------------------- speaker tabs (unchanged) -----------------------

with tab_enroll:
    st.subheader("Create a speaker profile")
    st.warning("Only enroll people who have explicitly agreed to have their voice processed and stored.")
    speaker_name = st.text_input("Profile name or label", placeholder="e.g. Speaker 1")
    consent = st.checkbox("I have permission to record and store this person's voice profile.")
    enroll_audio = st.file_uploader(
        "Upload a 5–10 second clear voice sample", type=AUDIO_TYPES, key="enroll_upload"
    )
    if enroll_audio:
        st.audio(enroll_audio)
    if st.button("Create speaker profile", type="primary"):
        if not consent:
            st.error("Please confirm consent before creating a profile.")
        elif not speaker_name.strip():
            st.error("Enter a profile name.")
        elif not enroll_audio:
            st.error("Upload a voice sample first.")
        else:
            try:
                with st.spinner("Analyzing voice; first run downloads the model..."):
                    embedding, duration = speakers.make_embedding(enroll_audio)
                    speaker_id = speakers.save_speaker(speaker_name, embedding)
                st.success(f"Profile created for **{speaker_name.strip()}**.")
                st.code(speaker_id)
                st.caption(f"Processed {duration:.1f} seconds of audio.")
            except Exception as exc:
                st.error(f"Could not create profile: {exc}")

with tab_identify:
    st.subheader("Check whether a speaker is known")
    if not speakers.list_speakers():
        st.info("No profiles yet. Create one in the 'Enroll speaker' tab first.")
    threshold = st.slider(
        "Similarity threshold (demo only; calibrate with your recordings)",
        min_value=0.10, max_value=0.90, value=0.35, step=0.01,
    )
    identify_audio = st.file_uploader(
        "Upload a 5–10 second recording to compare", type=AUDIO_TYPES, key="identify_upload"
    )
    if identify_audio:
        st.audio(identify_audio)
    if st.button("Identify speaker", type="primary", key="identify_btn"):
        profiles = speakers.load_embeddings()
        if not identify_audio:
            st.error("Upload a voice sample first.")
        elif not profiles:
            st.error("There are no enrolled speaker profiles to compare against.")
        else:
            try:
                with st.spinner("Comparing voice with saved profiles..."):
                    query, duration = speakers.make_embedding(identify_audio)
                    scores = sorted(
                        [(speakers.cosine_similarity(query, emb), sid, name) for sid, name, emb in profiles],
                        reverse=True,
                    )
                score, _, name = scores[0]
                st.metric("Best similarity score", f"{score:.3f}")
                if score >= threshold:
                    st.success(f"Likely returning speaker: **{name}**")
                else:
                    st.warning("No enrolled speaker matched this threshold. Treat the speaker as unknown.")
                st.caption(
                    f"Compared {duration:.1f} seconds. Scores are not probabilities; "
                    "validate on recordings from your intended users."
                )
                with st.expander("Show all comparison scores"):
                    for similarity, sid, profile_name in scores:
                        st.write(f"{profile_name}: {similarity:.3f} · ID {sid}")
            except Exception as exc:
                st.error(f"Could not identify speaker: {exc}")

with tab_profiles:
    st.subheader("Manage saved speaker profiles")
    saved = speakers.list_speakers()
    if not saved:
        st.info("No saved profiles.")
    else:
        st.caption("Profiles are stored in a local SQLite database on this computer.")
        for speaker_id, name, created_at in saved:
            with st.container(border=True):
                left, right = st.columns([4, 1])
                with left:
                    st.write(f"**{name}**")
                    st.caption(f"Created: {created_at}")
                    st.code(speaker_id)
                with right:
                    confirm = st.checkbox("Confirm delete", key=f"confirm_{speaker_id}")
                    if st.button("Delete", key=f"delete_{speaker_id}", disabled=not confirm):
                        speakers.delete_speaker(speaker_id)
                        st.success(f"Deleted profile: {name}")
                        st.rerun()
