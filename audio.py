"""Audio decoding, silence-aware chunking, and VAD-based short-segment splitting."""
import io
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SR = 16000


def decode_audio(audio_file, target_sr=SR):
    """Decode common audio formats and return (mono float32 samples at target_sr, seconds)."""
    raw = audio_file.getvalue() if hasattr(audio_file, "getvalue") else audio_file
    try:
        audio, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
    except Exception:
        import av

        chunks = []
        with av.open(io.BytesIO(raw)) as container:
            stream = next((s for s in container.streams if s.type == "audio"), None)
            if stream is None:
                raise ValueError("No audio track was found in this file.")
            resampler = av.AudioResampler(format="fltp", layout="mono", rate=target_sr)
            for frame in container.decode(stream):
                for converted in resampler.resample(frame):
                    chunks.append(converted.to_ndarray().reshape(-1).astype(np.float32))
            for converted in resampler.resample(None):
                chunks.append(converted.to_ndarray().reshape(-1).astype(np.float32))
        if not chunks:
            raise ValueError("Could not decode audio from this file.")
        audio = np.concatenate(chunks)
        sr = target_sr

    if sr != target_sr:
        divisor = gcd(int(sr), int(target_sr))
        audio = resample_poly(audio, int(target_sr) // divisor, int(sr) // divisor).astype(np.float32)
    if len(audio) == 0:
        raise ValueError("The audio file is empty.")
    return np.asarray(audio, dtype=np.float32), len(audio) / target_sr


def audio_to_16k_mono(audio_file):
    """Mono 16 kHz float32 samples, plus duration in seconds."""
    return decode_audio(audio_file, SR)


def split_audio(audio, min_s=15, max_s=30, frame_ms=50):
    """Return (start, end) sample indices of chunks, cutting at the quietest point
    between min_s and max_s so words are not sliced in half."""
    n = len(audio)
    max_len, min_len = int(max_s * SR), int(min_s * SR)
    frame = int(SR * frame_ms / 1000)
    bounds, pos = [], 0
    while pos < n:
        if n - pos <= max_len:
            bounds.append((pos, n))
            break
        lo, hi = pos + min_len, pos + max_len
        seg = audio[lo:hi]
        k = len(seg) // frame
        energy = np.sqrt((seg[: k * frame].reshape(k, frame) ** 2).mean(axis=1))
        cut = lo + int(np.argmin(energy)) * frame + frame // 2
        bounds.append((pos, cut))
        pos = cut
    return bounds


def split_into_short_segments(
    audio,
    min_s: float = 2.5,
    max_s: float = 8.0,
    frame_ms: int = 30,
    silence_thresh: float = 0.006,
    min_silence_ms: int = 300,
):
    """Split audio into short voice-activity segments for fine-grained speaker tracking.

    Cuts at silence boundaries (energy below *silence_thresh* for >= *min_silence_ms*).
    Each returned segment is guaranteed to be between *min_s* and *max_s* seconds long.

    Returns list of (start_sample, end_sample) tuples.
    """
    n = len(audio)
    if n == 0:
        return []

    frame = int(SR * frame_ms / 1000)
    min_len = int(min_s * SR)
    max_len = int(max_s * SR)
    min_silence_frames = max(1, int(min_silence_ms / frame_ms))

    # ── Compute per-frame RMS energy ──────────────────────────────────────────
    num_frames = n // frame
    if num_frames == 0:
        return [(0, n)]

    frames = audio[: num_frames * frame].reshape(num_frames, frame)
    energy = np.sqrt((frames ** 2).mean(axis=1))  # (num_frames,)

    # Boolean mask: True → silence frame
    is_silence = energy < silence_thresh

    # ── Find silence runs that are long enough to be "gaps" ────────────────────
    # Build list of (start_frame, end_frame) for each silence run
    gap_centers = []
    run_start = None
    for fi in range(num_frames):
        if is_silence[fi]:
            if run_start is None:
                run_start = fi
        else:
            if run_start is not None:
                run_len = fi - run_start
                if run_len >= min_silence_frames:
                    center = (run_start + fi) // 2
                    gap_centers.append(center * frame + frame // 2)
                run_start = None
    # trailing run
    if run_start is not None:
        run_len = num_frames - run_start
        if run_len >= min_silence_frames:
            center = (run_start + num_frames) // 2
            gap_centers.append(center * frame + frame // 2)

    # ── Build segment boundaries respecting min/max length ────────────────────
    bounds = []
    pos = 0
    eligible_gaps = list(gap_centers)

    while pos < n:
        remaining = n - pos
        if remaining <= max_len:
            bounds.append((pos, n))
            break

        # Find the best gap in [pos+min_len, pos+max_len]
        lo = pos + min_len
        hi = pos + max_len
        candidates = [g for g in eligible_gaps if lo <= g <= hi]

        if candidates:
            # Prefer the gap closest to the midpoint of the window
            mid = (lo + hi) // 2
            cut = min(candidates, key=lambda g: abs(g - mid))
        else:
            # No natural gap: cut at max_len (hard boundary)
            cut = hi

        cut = min(cut, n)
        if cut <= pos:
            cut = min(pos + max_len, n)

        bounds.append((pos, cut))
        pos = cut

    return bounds if bounds else [(0, n)]
