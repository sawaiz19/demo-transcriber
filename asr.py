"""Speech-recognition engines (GPU-aware).

Every engine has transcribe(chunk, language) -> str, where `chunk` is 16 kHz mono float32,
and a `.device` attribute. Only one engine is kept in memory at a time: choosing a new
one frees the previous one (and its GPU memory) first.
"""
import gc
import os
import tempfile
from contextlib import contextmanager

import numpy as np
import soundfile as sf
import torch

from audio import SR

MAX_NEW_TOKENS = 600  # one <=30 s chunk never needs more
SILENCE_RMS = 0.003  # quieter chunks are skipped (avoids hallucinated text)

LANGUAGES = {
    "Hindi (recommended for Urdu speech)": "Hindi",
    "Auto-detect": None,
    "English": "English",
}

DEVICE_CHOICES = {
    "Auto (GPU if available)": "auto",
    "GPU (NVIDIA CUDA)": "cuda",
    "CPU": "cpu",
}

OOM_MESSAGE = (
    "The GPU ran out of memory (VRAM). Try Precision = float16, pick a smaller model "
    "(Qwen3-ASR 0.6B or Whisper-tiny Hinglish), close other GPU-heavy apps, or set 'Run on' to CPU. "
    "Then click 'Resume transcription'."
)


# ------------------------------ device helpers -------------------------------

def gpu_info():
    """(name, total VRAM in GB) of the first CUDA GPU, or None."""
    if not torch.cuda.is_available():
        return None
    p = torch.cuda.get_device_properties(0)
    return p.name, p.total_memory / 1024 ** 3


def gpu_problem():
    """Why CUDA is unavailable: 'cpu_build', 'no_gpu_or_driver', or None if it works."""
    if torch.cuda.is_available():
        return None
    return "cpu_build" if torch.version.cuda is None else "no_gpu_or_driver"


def resolve_device(choice):
    if choice != "cpu" and torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def resolve_dtype(name, device):
    if name == "auto":
        if device.startswith("cuda"):
            return "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
        return "bfloat16"
    if device == "cpu" and name == "float16":
        return "float32"  # half precision is not supported well on CPU
    return name


@contextmanager
def _wav_file(chunk):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        path = tmp.name
    try:
        sf.write(path, chunk, SR)
        yield path
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


# --------------------------------- engines ----------------------------------

class QwenEngine:
    def __init__(self, repo, dtype_name, device):
        from qwen_asr import Qwen3ASRModel

        self.device = device
        if device == "cpu":
            torch.set_num_threads(os.cpu_count() or 4)
        self.model = Qwen3ASRModel.from_pretrained(
            repo,
            device_map=device,
            dtype=getattr(torch, dtype_name),
            max_inference_batch_size=1,
            max_new_tokens=MAX_NEW_TOKENS,
        )

    def transcribe(self, chunk, language):
        with _wav_file(chunk) as path, torch.inference_mode():
            results = self.model.transcribe(audio=path, language=language)
        return str(results[0].text).strip() if results else ""


class WhisperTinyHinglishEngine:
    """Fine-tuned whisper-tiny that writes Roman Hinglish directly (no language hint)."""

    def __init__(self, repo, device):
        from transformers import pipeline

        self.device = device
        on_gpu = device.startswith("cuda")
        if not on_gpu:
            torch.set_num_threads(os.cpu_count() or 4)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=repo,
            device=0 if on_gpu else -1,
            torch_dtype=torch.float16 if on_gpu else torch.float32,
        )

    def transcribe(self, chunk, language):
        return self.pipe({"raw": chunk, "sampling_rate": SR})["text"].strip()


class TaraEngine:
    """Trelis/tara: Whisper large-v3 fine-tune. Hindi in Devanagari, English in Latin."""

    def __init__(self, repo, dtype_name, device):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.device = device
        if device == "cpu":
            torch.set_num_threads(os.cpu_count() or 4)
        self.processor = WhisperProcessor.from_pretrained(repo)
        self.model = (
            WhisperForConditionalGeneration.from_pretrained(
                repo, torch_dtype=getattr(torch, dtype_name)
            )
            .to(device)
            .eval()
        )
        ids = self.processor.tokenizer.convert_tokens_to_ids
        self.hi, self.en, self.mc = ids("<|hi|>"), ids("<|en|>"), ids("<|mixedcode|>")
        self.transcribe_tok, self.no_ts = ids("<|transcribe|>"), ids("<|notimestamps|>")
        self.sot = ids("<|startoftranscript|>")

    def transcribe(self, chunk, language):
        # Prompt tokens (language, mixed-code mode, task) go straight to the decoder and the
        # text is decoded greedily; this avoids the removed `forced_decoder_ids` argument.
        feats = self.processor(chunk, sampling_rate=SR, return_tensors="pt").input_features
        feats = feats.to(self.device, self.model.dtype)
        if language == "English":
            prompt = [self.sot, self.en, self.transcribe_tok, self.no_ts]
        else:  # Hindi + mixed-code mode: the model card's safe default for Hinglish
            prompt = [self.sot, self.hi, self.mc, self.transcribe_tok, self.no_ts]
        ids = torch.tensor([prompt], device=self.device)
        eos = self.processor.tokenizer.eos_token_id
        tokens, past = [], None
        with torch.inference_mode():
            enc = self.model.model.encoder(feats)  # run the audio encoder once
            for _ in range(444):
                out = self.model(
                    encoder_outputs=enc, decoder_input_ids=ids,
                    past_key_values=past, use_cache=True,
                )
                past = out.past_key_values
                logits = out.logits[:, -1].float()
                logits[:, eos + 1:] = float("-inf")  # no special / language / timestamp tokens
                nxt = int(logits.argmax(-1))
                if nxt == eos:
                    break
                tokens.append(nxt)
                if len(tokens) >= 20 and len(set(tokens[-20:])) == 1:  # stuck repeating
                    break
                ids = torch.tensor([[nxt]], device=self.device)
        return self.processor.tokenizer.decode(tokens, skip_special_tokens=True).strip()


class ShrutiEngine:
    """udgamlabs/Shruti: 29.5 MB offline model on the Cactus 'needle' runtime (CPU only)."""

    def __init__(self):
        self.device = "cpu"
        os.environ.setdefault("NEEDLE_TELEMETRY", "0")
        from huggingface_hub import hf_hub_download

        try:
            from needle import Whistle
        except ImportError as exc:
            raise RuntimeError(
                "Shruti needs the 'cactus-needle' package. Run: pip install cactus-needle"
            ) from exc

        self.model = Whistle(hf_hub_download("udgamlabs/Shruti", "shruti.cact"))

    def transcribe(self, chunk, language):
        kwargs = {"language": "en"} if language == "English" else {}  # else auto-detect
        with _wav_file(chunk) as path:
            result = self.model.transcribe(path, **kwargs)
        text = result.get("text", "") if isinstance(result, dict) else result
        return str(text).strip()


# ---------------------------- registry / loading -----------------------------
# script: "devanagari" -> needs the LIPI step; "roman" -> already Roman Hindi.
# uses_gpu: False means the engine has its own runtime and ignores the 'Run on' setting.

ENGINES = {
    "Qwen3-ASR 1.7B (most accurate)": dict(
        script="devanagari", uses_dtype=True, uses_gpu=True,
        note="Best accuracy of the Qwen pair. Needs roughly 4-5 GB of VRAM on a GPU.",
        factory=lambda dtype, dev: QwenEngine("Qwen/Qwen3-ASR-1.7B", dtype, dev),
    ),
    "Qwen3-ASR 0.6B (faster)": dict(
        script="devanagari", uses_dtype=True, uses_gpu=True,
        note="Roughly 2-3x faster than 1.7B and needs about 2 GB of VRAM, with lower accuracy.",
        factory=lambda dtype, dev: QwenEngine("Qwen/Qwen3-ASR-0.6B", dtype, dev),
    ),
    "Whisper-tiny Hinglish (lightest, writes Roman directly)": dict(
        script="roman", uses_dtype=False, uses_gpu=True,
        note="39M parameters. Outputs Roman text directly, so no conversion step. "
        "Less accurate on noisy or fast speech.",
        factory=lambda dtype, dev: WhisperTinyHinglishEngine("abhishekgautamm/whisper-tiny-hinglish", dev),
    ),
    "Tara (Whisper large-v3 based, heavy)": dict(
        script="devanagari", uses_dtype=True, uses_gpu=True,
        note="1.55B parameters (about 3 GB of VRAM in 16-bit). Strong Hindi/Hinglish benchmarks. "
        "Needs a GPU to be practical.",
        factory=lambda dtype, dev: TaraEngine("Trelis/tara", dtype, dev),
    ),
    "Shruti (tiny 30 MB, offline)": dict(
        script="devanagari", uses_dtype=False, uses_gpu=False,
        note="Runs on its own CPU runtime (not PyTorch), so it does not use the GPU. "
        "Hindi in Devanagari, English words in Latin. Smallest; less accurate on pure Hindi.",
        factory=lambda dtype, dev: ShrutiEngine(),
    ),
}

_current = {"key": None, "engine": None}


def unload_engine():
    _current.update(key=None, engine=None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_engine(label, dtype_name, device_choice="auto"):
    cfg = ENGINES[label]
    device = resolve_device(device_choice) if cfg["uses_gpu"] else "cpu"
    dtype = resolve_dtype(dtype_name, device) if cfg["uses_dtype"] else None
    key = (label, dtype, device)
    if _current["key"] != key:
        unload_engine()  # free the previous model (and its VRAM) before loading the next
        try:
            _current["engine"] = cfg["factory"](dtype, device)
        except torch.cuda.OutOfMemoryError as exc:
            unload_engine()
            raise RuntimeError(OOM_MESSAGE) from exc
        _current["key"] = key
    return _current["engine"]


def transcribe_chunk(engine, chunk, language):
    """Transcribe one chunk; returns '' for silent/too-short audio."""
    if len(chunk) < 0.3 * SR or float(np.sqrt(np.mean(chunk ** 2))) < SILENCE_RMS:
        return ""
    try:
        return engine.transcribe(chunk, language)
    except torch.cuda.OutOfMemoryError as exc:
        gc.collect()
        torch.cuda.empty_cache()
        raise RuntimeError(OOM_MESSAGE) from exc
