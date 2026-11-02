import hashlib
import json
import os
import time
import uuid

import numpy as np
import pandas as pd
import streamlit as st

import asr
import compare
import pipeline
import postprocess
import romanize
import speakers
import summary
from audio import SR, split_audio

AUDIO_TYPES = ["wav", "mp3", "m4a", "flac", "ogg", "webm", "mp4", "mpeg", "mpga", "mov", "mkv"]

st.set_page_config(
    page_title="Urdu & English AI Transcriber",
    page_icon="TMC",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------- custom modern styling -----------------------------
st.markdown(
    """
    <style>
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 34, 45, 0.6) 0%, rgba(20, 24, 33, 0.8) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .status-card {
        background: rgba(14, 165, 233, 0.08);
        border: 1px solid rgba(14, 165, 233, 0.3);
        border-radius: 10px;
        padding: 14px 18px;
        margin: 10px 0;
    }
    .speaker-pill-enrolled {
        display: inline-block;
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        color: white;
        font-weight: 600;
        font-size: 0.82rem;
        padding: 3px 10px;
        border-radius: 9999px;
        margin-right: 8px;
        letter-spacing: 0.02em;
    }
    .speaker-pill-anon {
        display: inline-block;
        background: linear-gradient(135deg, #475569 0%, #64748b 100%);
        color: white;
        font-weight: 600;
        font-size: 0.82rem;
        padding: 3px 10px;
        border-radius: 9999px;
        margin-right: 8px;
        letter-spacing: 0.02em;
    }
    .time-pill {
        display: inline-block;
        background: rgba(255, 255, 255, 0.08);
        color: #94a3b8;
        font-size: 0.78rem;
        font-family: monospace;
        padding: 2px 8px;
        border-radius: 6px;
        margin-right: 8px;
    }
    .dialogue-turn {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 10px;
        transition: all 0.2s ease;
    }
    .dialogue-turn:hover {
        background: rgba(255, 255, 255, 0.05);
        border-color: rgba(255, 255, 255, 0.15);
    }
    .dialogue-text {
        font-size: 1.02rem;
        line-height: 1.55;
        margin-top: 6px;
        color: #e2e8f0;
    }
    .hero-title {
        font-size: 1.85rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def md5(data):
    return hashlib.md5(data if isinstance(data, bytes) else data.encode()).hexdigest()


def plan_segments(audio, turns):
    """Turn diarization turns into ASR chunks (<= 30 s each), keeping each chunk's speaker label."""
    n = len(audio)
    pad = int(0.15 * SR)
    bounds, labels = [], []
    for t in turns:
        s = max(0, int(t["start"] * SR) - pad)
        e = min(n, int(t["end"] * SR) + pad)
        if e - s <= 0:
            continue
        if e - s > 30 * SR:
            for a, b in split_audio(audio[s:e]):
                bounds.append((s + a, s + b))
                labels.append(t["label"])
        else:
            bounds.append((s, e))
            labels.append(t["label"])
    return bounds, labels


# ------------------------------- sidebar ------------------------------------

with st.sidebar:
    st.markdown("ASR Configuration")
    model_label = st.selectbox(
        "Speech-recognition model",
        list(asr.ENGINES),
        index=0,
        help="Recommended: Urdu Whisper Large-v3 Turbo for best Pakistani Urdu and mixed English recognition.",
    )
    st.caption(asr.ENGINES[model_label]["note"])

    language_choice = st.radio(
        "Spoken language hint",
        list(asr.LANGUAGES),
        index=0,
        help="Urdu models prioritize Urdu speech; Qwen/Tara perform best when set to Hindi/Devanagari.",
    )

    st.divider()
    st.markdown("Speaker Diarization (pyannote)")
    auto_speaker = st.checkbox(
        "Automatic speaker identification",
        value=True,
        help="Diarizes the whole recording with pyannote, then names speakers from enrolled profiles "
             "or labels them Anonymous 1, Anonymous 2, ...",
    )
    diar_model = st.selectbox("Diarization pipeline", speakers.DIARIZATION_MODELS, index=0)
    hf_token = st.text_input(
        "Hugging Face token",
        type="password",
        value=os.environ.get("HF_TOKEN", ""),
        help="Needed once to download the gated pyannote models. Accept their terms on huggingface.co first.",
    )
    num_speakers = st.number_input(
        "Number of speakers (0 = auto-detect)", min_value=0, max_value=20, value=0, step=1,
        help="If you know how many people speak, set it: this is the single biggest accuracy boost.",
    )
    min_speakers = max_speakers = 0
    if num_speakers == 0:
        with st.expander("Speaker count range (optional)"):
            min_speakers = st.number_input("Minimum speakers", min_value=0, max_value=20, value=0, step=1)
            max_speakers = st.number_input("Maximum speakers", min_value=0, max_value=20, value=0, step=1)
    speaker_threshold = st.slider(
        "Enrolled speaker match threshold",
        min_value=0.20,
        max_value=0.90,
        value=0.50,
        step=0.01,
        help="Minimum cosine similarity to accept an enrolled profile. Raise if the wrong name appears, "
             "lower if a known person stays 'Anonymous'.",
    )
    saved_speakers = speakers.list_speakers()
    usable_profiles = speakers.load_embeddings()
    if usable_profiles:
        st.success(f" {len(usable_profiles)} enrolled profile(s) ready")
    else:
        st.info("No enrolled profiles. Speakers will be labeled Anonymous 1, Anonymous 2...")
    legacy = speakers.legacy_profile_names()
    if legacy:
        st.warning(f"{len(legacy)} old SpeechBrain profile(s) must be re-enrolled: {', '.join(legacy[:4])}")

    st.divider()
    st.markdown("AI Summary & Key Points")
    enable_summary = st.checkbox(
        "Auto-generate AI summary & key points",
        value=True,
        help="Uses an LLM to generate executive summary, key points, speaker highlights, and action items.",
    )
    ollama_ready = summary.check_ollama_alive()
    llm_providers = ["Ollama (Local)"] if ollama_ready else []
    llm_providers.extend(["Local Transformers (Hugging Face)", "OpenAI / Groq / Cloud API"])
    if not ollama_ready:
        llm_providers.insert(0, "Ollama (Local)")

    llm_provider = st.selectbox("LLM Provider", llm_providers, index=0 if ollama_ready else 1)

    if llm_provider == "Ollama (Local)":
        if ollama_ready:
            st.success("Ollama connected")
            ollama_models = summary.list_ollama_models()
            llm_model = st.selectbox("Ollama Model", ollama_models if ollama_models else ["qwen2.5:7b", "llama3.2:3b", "qwen3:1.7b"], index=0)
        else:
            st.warning("Ollama is not running on localhost:11434.")
            llm_model = st.text_input("Ollama Model Name", value="qwen2.5:7b")
        llm_api_key = ""
    elif llm_provider == "Local Transformers (Hugging Face)":
        llm_model = st.selectbox(
            "Local Model",
            ["Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"],
            index=0,
            help="Runs locally in Python. Requires ~1.5 GB VRAM or runs on CPU.",
        )
        llm_api_key = ""
    else:
        llm_api_key = st.text_input("API Key (Groq / OpenAI)", type="password", help="Enter your Groq (e.g. gsk_...) or OpenAI API key")
        llm_model = st.selectbox(
            "Model Name",
            ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "gpt-4o-mini"],
            index=0,
        )

    st.divider()
    st.markdown("Hardware & Cleanup")
    device_label = st.selectbox("Run on", list(asr.DEVICE_CHOICES))
    device_choice = asr.DEVICE_CHOICES[device_label]
    gpu = asr.gpu_info()
    if gpu:
        st.success(f"GPU: {gpu[0]} ({gpu[1]:.1f} GB VRAM)")
    else:
        st.warning("Running on CPU. CUDA GPU not active.")

    with st.expander("Advanced Options"):
        dtype_name = st.selectbox("Precision", ["auto", "bfloat16", "float16", "float32"])
        backend_label = st.radio("LIPI Mode (Hindi only)", list(romanize.BACKENDS))
        use_denoise = st.checkbox("Extract voice first (Demucs)", value=False)
        use_spelling = st.checkbox("Apply spelling standard", value=True)
        use_wordlist = st.checkbox("Apply custom word list", value=True)


# --------------------------------- tabs ------------------------------------

tab_transcribe, tab_compare, tab_lists, tab_enroll, tab_profiles = st.tabs(
    [
        "Transcribe & Analyze",
        "Compare & Cross-check",
        "Word Lists & Spelling",
        "Enroll Speaker",
        "Saved Profiles",
    ]
)

# ============================================================================
# TAB 1: Transcribe & Analyze
# ============================================================================

with tab_transcribe:
    st.markdown('<div class="hero-title">Urdu & English Transcriber & Meeting Analyst</div>', unsafe_allow_html=True)
    st.caption("Automated speech-to-text with pyannote speaker diarization, natural Roman Urdu transliteration, and AI meeting summaries.")

    media_file = st.file_uploader(
        "Upload audio or video recording",
        type=AUDIO_TYPES,
        key="main_media_upload",
        help="Supports MP3, WAV, M4A, MP4, WEBM, FLAC, and more.",
    )

    if media_file:
        file_bytes = media_file.getvalue()
        file_name = media_file.name
        is_video = any(file_name.lower().endswith(ext) for ext in (".mp4", ".webm", ".mov", ".mkv"))

        col_player, col_meta = st.columns([3, 1])
        with col_player:
            if is_video:
                st.video(media_file)
            else:
                st.audio(media_file)
        with col_meta:
            st.metric("File Name", file_name[:24] + "..." if len(file_name) > 24 else file_name)
            st.caption(f"Size: {len(file_bytes) / (1024*1024):.1f} MB")

        job_key = (
            md5(file_bytes)
            + model_label
            + language_choice
            + str(auto_speaker)
            + diar_model
            + f"{num_speakers}-{min_speakers}-{max_speakers}"
            + str(speaker_threshold)
            + str(use_denoise)
        )
        job = st.session_state.get("job")
        if not job or job.get("key") != job_key:
            job = {
                "key": job_key,
                "audio": None,
                "bounds": None,
                "seg_speakers": None,
                "segments": [],
                "elapsed": 0.0,
                "duration": 0.0,
                "summary": None,
                "ran_on": "",
            }
            st.session_state["job"] = job

        done = job["bounds"] is not None and len(job["segments"]) == len(job["bounds"])
        started = bool(job["segments"])
        btn_label = "Transcribe & Analyze" if not started else ("Re-run Transcription" if done else "Resume Transcription")

        c_btn, c_opt = st.columns([2, 5])
        with c_btn:
            run_clicked = st.button(btn_label, type="primary", use_container_width=True)

        if run_clicked:
            if done:
                # the audio was freed after the last run, so decode and diarize again
                job.update(segments=[], elapsed=0.0, summary=None, bounds=None, seg_speakers=None, audio=None)

            progress_container = st.container()
            with progress_container:
                st.markdown("Processing Pipeline")
                pipeline_status = st.empty()
                progress_bar = st.progress(0.0)
                active_card = st.empty()
                live_text_stream = st.empty()

            t0 = time.perf_counter()
            try:
                # ── Stage 1: decode audio, diarize, name the speakers ───────────
                if job["bounds"] is None:
                    pipeline_status.info("🎵 Decoding audio...")
                    job["audio"], duration = pipeline.prepare_audio(file_bytes, use_denoise, device_choice)
                    job["duration"] = duration

                    if auto_speaker:
                        pipeline_status.info("🔬 Running pyannote speaker diarization (whole recording)...")
                        turns = speakers.merge_turns(
                            speakers.diarize(
                                job["audio"],
                                model_id=diar_model,
                                token=hf_token,
                                device=device_choice,
                                num_speakers=int(num_speakers) or None,
                                min_speakers=int(min_speakers) or None,
                                max_speakers=int(max_speakers) or None,
                            )
                        )
                        if turns:
                            pipeline_status.info("Matching speakers against enrolled profiles...")
                            mapping = speakers.label_speakers(
                                job["audio"], turns, speakers.load_embeddings(),
                                threshold=speaker_threshold, token=hf_token, device=device_choice,
                            )
                            bounds, labels = plan_segments(job["audio"], turns)
                            job["seg_speakers"] = [mapping[l] for l in labels]
                            job["bounds"] = bounds
                        else:
                            job["bounds"] = split_audio(job["audio"])
                            job["seg_speakers"] = [("Anonymous 1", 1.0, False)] * len(job["bounds"])
                    else:
                        job["bounds"] = split_audio(job["audio"])
                        job["seg_speakers"] = [("Anonymous 1", 1.0, False)] * len(job["bounds"])

                # ── Stage 2: Load ASR model ─────────────────────────────────────
                pipeline_status.info(f"Initializing speech model: {model_label.split(' (')[0]}...")
                engine = asr.get_engine(model_label, dtype_name, device_choice)
                job["ran_on"] = str(engine.device)
                model_script = asr.ENGINES[model_label]["script"]
                lang = asr.LANGUAGES[language_choice]

                total_chunks = len(job["bounds"])
                backend = romanize.BACKENDS[backend_label]

                # ── Stage 3: Transcribe each speaker turn ───────────────────────
                while len(job["segments"]) < total_chunks:
                    i = len(job["segments"])
                    s, e = job["bounds"][i]
                    start_s = s / SR
                    end_s = e / SR
                    chunk_samples = job["audio"][s:e]
                    spk_name, spk_score, is_enrolled_spk = job["seg_speakers"][i]

                    pipeline_status.info(
                        f"🎙️ Segment {i + 1} / {total_chunks} "
                        f"[{int(start_s//60):02d}:{int(start_s%60):02d} – "
                        f"{int(end_s//60):02d}:{int(end_s%60):02d}]"
                    )

                    t_chunk0 = time.perf_counter()
                    raw_chunk = asr.transcribe_chunk(engine, chunk_samples, lang)
                    chunk_time = time.perf_counter() - t_chunk0

                    roman_chunk = pipeline.finalize_text(
                        raw_chunk, model_script, backend, use_spelling, use_wordlist
                    )

                    seg_data = {
                        "index": i + 1,
                        "start": start_s,
                        "end": end_s,
                        "speaker": spk_name,
                        "is_enrolled": is_enrolled_spk,
                        "similarity": spk_score,
                        "raw_text": raw_chunk,
                        "text": roman_chunk,
                    }
                    job["segments"].append(seg_data)

                    frac = len(job["segments"]) / total_chunks
                    progress_bar.progress(frac)

                    badge_class = "speaker-pill-enrolled" if is_enrolled_spk else "speaker-pill-anon"
                    badge_label = (
                        f"👤 {spk_name} (Enrolled)" if is_enrolled_spk else f"🎙️ {spk_name}"
                    )
                    time_label = (
                        f"{int(start_s//60):02d}:{int(start_s%60):02d} – "
                        f"{int(end_s//60):02d}:{int(end_s%60):02d}"
                    )

                    active_card.markdown(
                        f"""
                        <div class="status-card">
                            <span class="{badge_class}">{badge_label}</span>
                            <span class="time-pill">{time_label}</span>
                            <span style="color:#64748b; font-size:0.85rem;">⏱ {chunk_time:.1f}s &nbsp;|&nbsp; {i+1}/{total_chunks}</span>
                            <div style="margin-top:6px; font-weight:500; color:#38bdf8;">
                                "{roman_chunk or raw_chunk or '(pause / silence)'}"
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    preview_text = " ".join(seg["text"] for seg in job["segments"] if seg["text"])
                    live_text_stream.caption(f"…{preview_text[-400:]}")

                job["audio"] = None  # free memory once complete
                pipeline_status.success(f"Transcribed all {total_chunks} segments successfully!")

                # ── Stage 4: AI Summary ─────────────────────────────────────────
                if enable_summary and not job.get("summary"):
                    pipeline_status.info("Generating AI executive summary & key points...")
                    try:
                        job["summary"] = summary.generate_summary(
                            job["segments"],
                            provider=llm_provider,
                            model_name=llm_model,
                            api_key=llm_api_key,
                            device=device_choice,
                        )
                        pipeline_status.success("Transcription and AI analysis completed!")
                    except Exception as sum_exc:
                        st.warning(f"ASR succeeded, but AI summary generation encountered an issue: {sum_exc}")

            except Exception as exc:
                st.error(f"Processing paused at part {len(job['segments']) + 1}: {exc}")
                st.info("Everything transcribed so far has been saved. Click 'Resume Transcription' to continue.")
            finally:
                job["elapsed"] += time.perf_counter() - t0
                time.sleep(0.5)
                progress_container.empty()

        # ------------------ Results Display ------------------
        if job["segments"]:
            total_duration = job.get("duration", 0.0)
            elapsed = job.get("elapsed", 0.0)
            rtf = elapsed / max(total_duration, 1.0)
            all_speakers = sorted(list({s["speaker"] for s in job["segments"]}))
            total_words = sum(len(s["text"].split()) for s in job["segments"])

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Audio Duration", f"{int(total_duration//60)}m {int(total_duration%60)}s")
            m2.metric("Processing Time", f"{elapsed:.1f}s ({rtf:.2f}x Realtime)")
            m3.metric("Speakers Detected", f"{len(all_speakers)} ({', '.join(all_speakers[:2])}{'...' if len(all_speakers)>2 else ''})")
            m4.metric("Word Count", f"{total_words:,} words")

            st.write("")
            res_tab_chat, res_tab_text, res_tab_summary, res_tab_stats = st.tabs(
                ["Dialogue View", "Full Transcript & Export", "AI Summary & Key Points", "Speaker Distribution"]
            )

            with res_tab_chat:
                st.caption("Speaker turns — consecutive segments from the same speaker are merged into a single turn.")

                merged_turns = []
                for seg in job["segments"]:
                    txt = seg["text"].strip()
                    if not txt:
                        continue
                    spk = seg["speaker"]
                    if merged_turns and merged_turns[-1]["speaker"] == spk:
                        merged_turns[-1]["text"] += " " + txt
                        merged_turns[-1]["end"] = seg["end"]
                    else:
                        merged_turns.append({
                            "speaker": spk,
                            "is_enrolled": seg["is_enrolled"],
                            "start": seg["start"],
                            "end": seg["end"],
                            "text": txt,
                        })

                with st.container(height=520, border=True):
                    for turn in merged_turns:
                        spk = turn["speaker"]
                        is_enr = turn["is_enrolled"]
                        s_min, s_sec = divmod(int(turn["start"]), 60)
                        e_min, e_sec = divmod(int(turn["end"]), 60)
                        time_tag = f"{s_min:02d}:{s_sec:02d} – {e_min:02d}:{e_sec:02d}"
                        pill_class = "speaker-pill-enrolled" if is_enr else "speaker-pill-anon"
                        badge_text = f"👤 {spk}" if is_enr else f"🎙️ {spk}"

                        st.markdown(
                            f"""
                            <div class="dialogue-turn">
                                <div>
                                    <span class="{pill_class}">{badge_text}</span>
                                    <span class="time-pill">{time_tag}</span>
                                </div>
                                <div class="dialogue-text">{turn['text']}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            with res_tab_text:
                formatted_dialogue = pipeline.format_segments_as_dialogue(job["segments"], include_timestamps=True)
                raw_full = " ".join(s["raw_text"] for s in job["segments"] if s["raw_text"])

                edited_transcript = st.text_area(
                    "Speaker-labeled transcript (editable)",
                    value=formatted_dialogue,
                    height=320,
                    key=f"text_area_{job_key}_{len(job['segments'])}",
                )

                c_d1, c_d2, c_d3 = st.columns(3)
                with c_d1:
                    st.download_button(
                        "Download Transcript (.txt)",
                        edited_transcript.rstrip() + "\n",
                        file_name=f"{file_name.rsplit('.', 1)[0]}_transcript.txt",
                        mime="text/plain",
                        use_container_width=True,
                    )
                with c_d2:
                    json_data = json.dumps(
                        {
                            "file": file_name,
                            "duration_seconds": total_duration,
                            "segments": job["segments"],
                            "summary": job.get("summary", ""),
                        },
                        indent=2,
                        ensure_ascii=False,
                    )
                    st.download_button(
                        "Download Structured (.json)",
                        json_data,
                        file_name=f"{file_name.rsplit('.', 1)[0]}_data.json",
                        mime="application/json",
                        use_container_width=True,
                    )
                with c_d3:
                    st.download_button(
                        "Download Raw ASR Output",
                        raw_full + "\n",
                        file_name=f"{file_name.rsplit('.', 1)[0]}_raw.txt",
                        mime="text/plain",
                        use_container_width=True,
                    )

            with res_tab_summary:
                st.subheader("Meeting Executive Summary & Discussion Points")
                current_summary = job.get("summary")

                if current_summary:
                    st.markdown(current_summary)
                    st.divider()
                    c_sum_down, c_sum_regen = st.columns([1, 1])
                    with c_sum_down:
                        st.download_button(
                            "Download Summary (.md)",
                            current_summary + "\n",
                            file_name=f"{file_name.rsplit('.', 1)[0]}_summary.md",
                            mime="text/markdown",
                        )
                    with c_sum_regen:
                        if st.button("Regenerate Summary with current LLM"):
                            with st.spinner("Analyzing transcript and generating updated summary..."):
                                try:
                                    job["summary"] = summary.generate_summary(
                                        job["segments"],
                                        provider=llm_provider,
                                        model_name=llm_model,
                                        api_key=llm_api_key,
                                        device=device_choice,
                                    )
                                    st.rerun()
                                except Exception as e_sum:
                                    st.error(f"Generation failed: {e_sum}")
                else:
                    st.info("No AI summary generated yet.")
                    if st.button("Generate Summary & Key Points Now", type="primary"):
                        with st.spinner("Analyzing transcript and generating summary..."):
                            try:
                                job["summary"] = summary.generate_summary(
                                    job["segments"],
                                    provider=llm_provider,
                                    model_name=llm_model,
                                    api_key=llm_api_key,
                                    device=device_choice,
                                )
                                st.rerun()
                            except Exception as e_sum:
                                st.error(f"Generation failed: {e_sum}")

            with res_tab_stats:
                st.subheader("Speaker Speaking Time & Turn Distribution")
                speaker_durations = {}
                speaker_turns = {}
                for s in job["segments"]:
                    name = s["speaker"]
                    dur = s["end"] - s["start"]
                    speaker_durations[name] = speaker_durations.get(name, 0.0) + dur
                    speaker_turns[name] = speaker_turns.get(name, 0) + 1

                stats_rows = []
                for spk, dur in speaker_durations.items():
                    pct = (dur / max(total_duration, 1.0)) * 100
                    turns = speaker_turns.get(spk, 0)
                    is_enrolled = any(s["is_enrolled"] for s in job["segments"] if s["speaker"] == spk)
                    stats_rows.append(
                        {
                            "Speaker": spk,
                            "Status": "Enrolled Profile" if is_enrolled else "Anonymous",
                            "Speaking Time (s)": round(dur, 1),
                            "Share of Audio": f"{pct:.1f}%",
                            "Speech Turns": turns,
                        }
                    )
                st.dataframe(pd.DataFrame(stats_rows), use_container_width=True)

# ============================================================================
# TAB 2: Compare & Cross-check
# ============================================================================

with tab_compare:
    st.subheader("Compare & Cross-Check ASR Models")
    st.caption("Run two models or preprocessing configurations side-by-side to highlight discrepancies and measure agreement.")

    cmp_audio = st.file_uploader("Upload audio for comparison", type=AUDIO_TYPES, key="cmp_upload")
    if cmp_audio:
        st.audio(cmp_audio)
    reference = st.text_area(
        "Ground truth transcript (optional)",
        height=100,
        key="cmp_ref",
        help="Optional Roman Urdu ground-truth text to compute word error rate.",
    )
    names = list(asr.ENGINES)
    default_b = min(1, len(names) - 1)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Run A**")
        model_a = st.selectbox("Model", names, index=0, key="cmp_model_a")
        dn_a = st.checkbox("Extract voice first (Demucs)", key="cmp_dn_a")
    with col_b:
        st.markdown("**Run B**")
        model_b = st.selectbox("Model", names, index=default_b, key="cmp_model_b")
        dn_b = st.checkbox("Extract voice first (Demucs)", key="cmp_dn_b")

    if st.button("Run Model Comparison", type="primary"):
        if not cmp_audio:
            st.error("Upload an audio file first.")
        else:
            audio_bytes_cmp, results, cache = cmp_audio.getvalue(), {}, {}
            try:
                for tag, label_x, dn_x in (("A", model_a, dn_a), ("B", model_b, dn_b)):
                    bar = st.progress(0.0, text=f"Run {tag}: starting...")
                    results[tag] = pipeline.transcribe_audio(
                        audio_bytes_cmp,
                        label_x,
                        language_choice,
                        dtype_name,
                        device_choice,
                        dn_x,
                        on_progress=lambda f, m, tag=tag, bar=bar: bar.progress(min(max(f, 0.0), 1.0), text=f"Run {tag}: {m}"),
                        cache=cache,
                    )
                    results[tag].update(model=label_x, denoise=dn_x)
                    bar.empty()
                st.session_state["cmp_results"] = results
                st.session_state["cmp_id"] = uuid.uuid4().hex[:8]
            except Exception as exc:
                st.error(f"Comparison stopped: {exc}")

    results = st.session_state.get("cmp_results")
    if results:
        run_id = st.session_state["cmp_id"]
        backend = romanize.BACKENDS[backend_label]
        texts = {}
        try:
            for tag in ("A", "B"):
                r = results[tag]
                texts[tag] = pipeline.finalize_text(
                    r["raw"], asr.ENGINES[r["model"]]["script"], backend, use_spelling, use_wordlist
                )
        except Exception as exc:
            st.error(f"Roman conversion failed: {exc}")
            texts = {}
        if texts:
            fold = postprocess.make_folder(postprocess.load_spelling())
            segments = compare.align(texts["A"], texts["B"], fold)
            stats = compare.summary(segments)

            def describe(tag):
                r = results[tag]
                dn = " · voice extraction" if r["denoise"] else ""
                return f"{r['model'].split(' (')[0]}{dn}"

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("A Time", f"{results['A']['seconds']:.0f}s", help=describe("A"))
            m2.metric("B Time", f"{results['B']['seconds']:.0f}s", help=describe("B"))
            m3.metric("Agreement", f"{stats['agreement'] * 100:.0f}%")
            m4.metric("Disagreements", stats["disagreements"])

            if reference.strip():
                wa = compare.error_rate(reference, texts["A"], fold)
                wb = compare.error_rate(reference, texts["B"], fold)
                e1, e2 = st.columns(2)
                e1.metric("A Error Rate", f"{wa * 100:.1f}%")
                e2.metric("B Error Rate", f"{wb * 100:.1f}%")

            v1, v2 = st.columns(2)
            for col, tag in ((v1, "A"), (v2, "B")):
                with col:
                    st.markdown(f"**{tag}: {describe(tag)}**")
                    with st.container(height=300, border=True):
                        st.markdown(compare.highlight_html(segments, tag.lower()), unsafe_allow_html=True)

# ============================================================================
# TAB 3: Word Lists & Spelling
# ============================================================================

with tab_lists:
    st.subheader("Custom Vocabulary & Word Replacement")
    st.caption("Define custom find-and-replace pairs for names, brands, technical acronyms, and specialized terms.")
    wl_rows = postprocess.load_wordlist()
    wl_df = pd.DataFrame(
        [{"Find": r["find"], "Replace with": r["replace"]} for r in wl_rows]
        or {"Find": pd.Series(dtype=str), "Replace with": pd.Series(dtype=str)}
    )
    wl_edit = st.data_editor(wl_df, num_rows="dynamic", key="wordlist_editor", use_container_width=True)
    if st.button("Save Custom Word List"):
        postprocess.save_wordlist(
            [{"find": str(r["Find"] or ""), "replace": str(r["Replace with"] or "")} for r in wl_edit.to_dict("records")]
        )
        st.success("Custom word list saved successfully.")

    st.divider()
    st.subheader("Roman Urdu Spelling Standard")
    st.caption("Standardize common Roman Urdu spelling variants into one uniform form.")
    sp_map = postprocess.load_spelling()
    sp_df = pd.DataFrame(
        [{"Standard spelling": k, "Variants (comma-separated)": ", ".join(v)} for k, v in sp_map.items()]
        or {"Standard spelling": pd.Series(dtype=str), "Variants (comma-separated)": pd.Series(dtype=str)}
    )
    sp_edit = st.data_editor(sp_df, num_rows="dynamic", key="spelling_editor", use_container_width=True)
    c_save, c_reset = st.columns(2)
    if c_save.button("Save Spelling Standard"):
        new_map = {}
        for r in sp_edit.to_dict("records"):
            std = str(r["Standard spelling"] or "").strip()
            if std:
                new_map[std] = [v.strip() for v in str(r["Variants (comma-separated)"] or "").split(",") if v.strip()]
        postprocess.save_spelling(new_map)
        st.success("Spelling standard saved.")
    if c_reset.button("Reset to Default Starter List"):
        postprocess.reset_spelling()
        st.rerun()

# ============================================================================
# TAB 4: Enroll Speaker
# ============================================================================

with tab_enroll:
    st.subheader("Create a New Speaker Profile")
    st.caption("Enrolled voices are recognized and labeled with their real names. Anyone not enrolled shows up as Anonymous 1, Anonymous 2, ...")
    st.warning("Ensure you have permission from the individual to store and process their voice sample.")

    speaker_name = st.text_input("Speaker Full Name", placeholder="e.g. Sawaiz Ahmad")
    consent = st.checkbox("I confirm that I have consent to store and identify this person's voice profile.")
    enroll_audio = st.file_uploader(
        "Upload one or more clear voice samples (8–30 s each; several samples from different recordings give the best accuracy)",
        type=AUDIO_TYPES,
        key="enroll_upload",
        accept_multiple_files=True,
    )
    for f in enroll_audio or []:
        st.audio(f)

    if st.button("Create Speaker Profile", type="primary"):
        if not consent:
            st.error("Please confirm consent before creating a profile.")
        elif not speaker_name.strip():
            st.error("Please enter a speaker name.")
        elif not enroll_audio:
            st.error("Please upload at least one voice sample.")
        else:
            try:
                with st.spinner("Computing voice embedding (pyannote)..."):
                    embedding, duration, n_samples = speakers.make_embedding(
                        list(enroll_audio), token=hf_token, device=device_choice
                    )
                    speaker_id = speakers.save_speaker(speaker_name, embedding, samples=n_samples)
                st.success(f"Profile created successfully for **{speaker_name.strip()}**!")
                st.caption(f"Profile ID: `{speaker_id}` · {n_samples} sample(s) · {duration:.1f}s total")
                st.info("This speaker will now be recognized automatically when transcribing recordings.")
            except Exception as exc:
                st.error(f"Could not create speaker profile: {exc}")

# ============================================================================
# TAB 5: Saved Profiles
# ============================================================================

with tab_profiles:
    st.subheader("Enrolled Speaker Profiles")
    saved = speakers.list_speakers()
    if not saved:
        st.info("No enrolled profiles found. You can create one in the 'Enroll Speaker' tab.")
    else:
        st.caption(f"Total enrolled profiles: {len(saved)}. Saved locally in SQLite database.")
        legacy_names = set(speakers.legacy_profile_names())
        for speaker_id, name, created_at in saved:
            with st.container(border=True):
                left, right = st.columns([4, 1])
                with left:
                    st.markdown(f"### 👤 {name}")
                    st.caption(f"Enrolled: {created_at} · ID: `{speaker_id}`")
                    if name in legacy_names:
                        st.caption("May be an old SpeechBrain profile — re-enroll to use it with pyannote.")
                with right:
                    confirm = st.checkbox("Confirm delete", key=f"confirm_{speaker_id}")
                    if st.button("Delete Profile", key=f"delete_{speaker_id}", disabled=not confirm):
                        speakers.delete_speaker(speaker_id)
                        st.success(f"Deleted profile: {name}")
                        st.rerun()