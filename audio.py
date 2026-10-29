"""Audio decoding and silence-aware chunking."""
import io
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SR = 16000


def audio_to_16k_mono(audio_file):
    """Decode common audio formats and return (mono 16 kHz float32 samples, seconds)."""
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
            resampler = av.AudioResampler(format="fltp", layout="mono", rate=SR)
            for frame in container.decode(stream):
                for converted in resampler.resample(frame):
                    chunks.append(converted.to_ndarray().reshape(-1).astype(np.float32))
            for converted in resampler.resample(None):
                chunks.append(converted.to_ndarray().reshape(-1).astype(np.float32))
        if not chunks:
            raise ValueError("Could not decode audio from this file.")
        audio = np.concatenate(chunks)
        sr = SR

    if sr != SR:
        divisor = gcd(int(sr), SR)
        audio = resample_poly(audio, SR // divisor, sr // divisor).astype(np.float32)
    if len(audio) == 0:
        raise ValueError("The audio file is empty.")
    return np.asarray(audio, dtype=np.float32), len(audio) / SR


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
