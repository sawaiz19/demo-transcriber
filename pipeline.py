"""Glue: audio preparation, full-file transcription and final text clean-up."""
import time

import asr
import postprocess
import romanize
from audio import audio_to_16k_mono, split_audio


def prepare_audio(audio_bytes, use_denoise, device_choice):
    """Decode to 16 kHz mono, optionally extracting the voice with Demucs first."""
    if use_denoise:
        import denoise

        audio = denoise.extract_vocals(audio_bytes, asr.resolve_device(device_choice))
        return audio, len(audio) / 16000
    return audio_to_16k_mono(audio_bytes)


def transcribe_audio(audio_bytes, model_label, language_label, dtype_name, device_choice,
                     use_denoise, on_progress=None, cache=None):
    """Transcribe a whole recording (used by the Compare tab). Returns a dict."""
    say = on_progress or (lambda fraction, message: None)
    cache = cache if cache is not None else {}
    t0 = time.perf_counter()
    if use_denoise not in cache:  # reuse the prepared audio when both runs use the same setting
        say(0.0, "Extracting voice (Demucs)..." if use_denoise else "Decoding audio...")
        cache[use_denoise] = prepare_audio(audio_bytes, use_denoise, device_choice)
    audio, duration = cache[use_denoise]
    bounds = split_audio(audio)
    say(0.0, f"Loading {model_label}...")
    engine = asr.get_engine(model_label, dtype_name, device_choice)
    language = asr.LANGUAGES[language_label]
    texts = []
    for i, (start, end) in enumerate(bounds):
        say(i / len(bounds), f"Transcribing part {i + 1} of {len(bounds)}...")
        texts.append(asr.transcribe_chunk(engine, audio[start:end], language))
    say(1.0, "Done")
    return {
        "raw": " ".join(t for t in texts if t),
        "seconds": time.perf_counter() - t0,
        "duration": duration,
        "ran_on": str(engine.device),
    }


def finalize_text(raw, script, lipi_backend, use_spelling=True, use_wordlist=True):
    """raw ASR text -> final Roman text: word list, Roman conversion, spelling standard, word list."""
    if not raw:
        return ""
    text = raw
    rows = postprocess.load_wordlist() if use_wordlist else []
    if rows:
        text = postprocess.apply_wordlist(text, rows)  # catches script entries

    if script == "urdu":
        text = romanize.romanize_urdu(text)
    elif script == "devanagari":
        text = romanize.romanize_hindi(text, lipi_backend)
    elif script == "roman":
        # May still contain occasional native script tokens
        text = romanize.transliterate_to_roman(text, script="roman", lipi_backend=lipi_backend)
    else:
        text = romanize.transliterate_to_roman(text, script=script, lipi_backend=lipi_backend)

    if use_spelling:
        text = postprocess.apply_spelling(text, postprocess.load_spelling())
    if rows:
        text = postprocess.apply_wordlist(text, rows)  # catches Roman entries
    return text.strip()


def format_segments_as_dialogue(segments, include_timestamps=True):
    """Format structured segments into a speaker dialogue string.
    Consecutive segments from the same speaker are merged into a single turn.
    """
    # ── Merge consecutive same-speaker segments ──────────────────────────────
    merged = []
    for seg in segments:
        txt = seg.get("text", "").strip()
        if not txt:
            continue
        spk = seg.get("speaker", "Unknown")
        if merged and merged[-1]["speaker"] == spk:
            merged[-1]["text"] += " " + txt
            merged[-1]["end"] = seg.get("end", merged[-1]["end"])
        else:
            merged.append({
                "speaker": spk,
                "text": txt,
                "start": seg.get("start", 0),
                "end": seg.get("end", 0),
            })

    lines = []
    for turn in merged:
        spk = turn["speaker"]
        txt = turn["text"]
        if include_timestamps:
            start_s = int(turn["start"])
            end_s = int(turn["end"])
            sm, ss = divmod(start_s, 60)
            em, es = divmod(end_s, 60)
            tag = f"[{sm:02d}:{ss:02d} – {em:02d}:{es:02d}]"
            lines.append(f"{tag} {spk}: {txt}")
        else:
            lines.append(f"{spk}: {txt}")
    return "\n\n".join(lines)
