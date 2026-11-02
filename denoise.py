"""Optional voice extraction with Demucs (separates speech from music and background noise).

Needs:  pip install demucs
The model is loaded, used and freed again inside one call, so it never competes with the
speech-recognition model for memory.
"""
import gc
from math import gcd

import numpy as np
import torch
from scipy.signal import resample_poly

from audio import SR, decode_audio

DEMUCS_MODEL = "htdemucs"


def extract_vocals(audio_bytes, device="cpu"):
    """Return the vocals of the recording as mono 16 kHz float32 samples."""
    try:
        from demucs.apply import apply_model
        from demucs.pretrained import get_model
    except ImportError as exc:
        raise RuntimeError(
            "Voice extraction needs the 'demucs' package. Run: pip install demucs"
        ) from exc

    model = get_model(DEMUCS_MODEL)
    model.eval()
    rate = int(model.samplerate)
    try:
        # Decode straight to Demucs' own sample rate (44.1 kHz) from the ORIGINAL file,
        # so nothing is upsampled from a 16 kHz copy.
        mono, _ = decode_audio(audio_bytes, rate)
        wav = torch.from_numpy(mono).float()[None].repeat(2, 1)  # (channels, samples)
        ref = wav.mean(0)
        wav = (wav - ref.mean()) / (ref.std() + 1e-8)  # same normalisation as the Demucs CLI
        with torch.no_grad():
            out = apply_model(model, wav[None], device=device, split=True, overlap=0.25, progress=False)[0]
        out = out * ref.std() + ref.mean()
        vocals = out[model.sources.index("vocals")].mean(0).cpu().numpy().astype(np.float32)
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    divisor = gcd(rate, SR)
    vocals = resample_poly(vocals, SR // divisor, rate // divisor).astype(np.float32)
    peak = float(np.max(np.abs(vocals))) if len(vocals) else 0.0
    if peak < 1e-4:
        raise RuntimeError("Voice extraction produced silence. Try again with it switched off.")
    return (vocals / peak * 0.9).astype(np.float32)
