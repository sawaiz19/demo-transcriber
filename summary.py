"""LLM Summarization & Key Points Generation.

Supports:
1. Ollama (local server, e.g. qwen2.5:7b, llama3.2:3b, qwen3:1.7b, mistral)
2. Local Hugging Face Transformers pipeline (in-process, e.g. Qwen/Qwen2.5-1.5B-Instruct)
3. OpenAI-compatible Cloud APIs (Groq, OpenAI, OpenRouter, Gemini)
"""
import json
import os
import requests
import torch
import streamlit as st

OLLAMA_DEFAULT_HOST = "http://localhost:11434"

SYSTEM_PROMPT = """You are a sharp, concise meeting analyst. Your job is to produce a brief, honest summary of what was actually discussed — nothing inflated, no padding.

Rules:
- Be factual and direct. Only report what is clearly in the transcript.
- Do NOT exaggerate, speculate, or add content that is not in the transcript.
- Keep the summary to 2-4 sentences maximum.
- Key points should be tight, plain-English bullet points — 3 to 6 maximum.
- If something is unclear, skip it rather than guess.
- The transcript may be in Roman Urdu or mixed Urdu-English — write the output in plain English.

Output format (use exactly these headings, nothing else):

### Summary
2–4 sentences covering what was talked about and what was decided or concluded.

### Key Points
- Bullet 1
- Bullet 2
- (3–6 bullets maximum, each one sentence)

### Speakers
- **Speaker name**: one sentence on what they contributed or said.
(only include speakers who actually spoke)
"""


def check_ollama_alive(host=OLLAMA_DEFAULT_HOST, timeout=2.0):
    """Check if the local Ollama daemon is reachable."""
    try:
        r = requests.get(f"{host.rstrip('/')}/api/tags", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def list_ollama_models(host=OLLAMA_DEFAULT_HOST, timeout=2.0):
    """Retrieve list of locally installed Ollama models."""
    try:
        r = requests.get(f"{host.rstrip('/')}/api/tags", timeout=timeout)
        if r.status_code == 200:
            data = r.json()
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        pass
    return []


def query_ollama(prompt: str, model_name: str, host=OLLAMA_DEFAULT_HOST, timeout=180.0) -> str:
    """Send prompt to Ollama's chat or generate endpoint."""
    url = f"{host.rstrip('/')}/api/chat"
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {
            "temperature": 0.3,
            "top_p": 0.9,
        },
    }
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("message", {}).get("content", "").strip()
        else:
            raise RuntimeError(f"Ollama returned HTTP {resp.status_code}: {resp.text}")
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            "Could not connect to Ollama. Ensure the Ollama app is running locally (http://localhost:11434)."
        ) from exc


@st.cache_resource(show_spinner="Loading local Hugging Face LLM pipeline...")
def load_hf_llm(model_id: str, device: str = "auto"):
    """Load an in-process instruction-tuned model via Hugging Face."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    torch_device = "cuda" if device in ("auto", "cuda") and torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (
        torch.float16 if torch_device == "cuda" else torch.float32
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=torch_device,
    )
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.3,
        top_p=0.9,
    )


def query_hf_local(prompt: str, model_id: str = "Qwen/Qwen2.5-1.5B-Instruct", device: str = "auto") -> str:
    """Generate summary using a local Hugging Face model in Python."""
    generator = load_hf_llm(model_id, device)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    prompt_formatted = generator.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    outputs = generator(prompt_formatted)
    full_text = outputs[0]["generated_text"]
    # Extract only assistant reply after the prompt
    assistant_reply = full_text[len(prompt_formatted):].strip()
    return assistant_reply


def query_openai_api(prompt: str, api_key: str, model_name: str = "gpt-4o-mini", base_url: str = None) -> str:
    """Query an OpenAI-compatible API (e.g. OpenAI, Groq, OpenRouter)."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    url = (base_url.rstrip("/") if base_url else "https://api.openai.com/v1") + "/chat/completions"
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=90.0)
    if resp.status_code == 200:
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    else:
        raise RuntimeError(f"API Error ({resp.status_code}): {resp.text}")


def format_transcript_for_llm(segments: list) -> str:
    """Format dialogue chunks with timestamps and speaker tags into clean text for LLM ingestion."""
    lines = []
    for seg in segments:
        spk = seg.get("speaker", "Unknown")
        txt = seg.get("text", "").strip()
        start = seg.get("start", 0)
        end = seg.get("end", 0)
        start_min, start_sec = divmod(int(start), 60)
        end_min, end_sec = divmod(int(end), 60)
        time_tag = f"[{start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}]"
        if txt:
            lines.append(f"{time_tag} {spk}: {txt}")
    return "\n".join(lines)


def generate_summary(
    transcript_text_or_segments,
    provider: str,
    model_name: str,
    api_key: str = "",
    device: str = "auto",
    custom_host: str = OLLAMA_DEFAULT_HOST,
) -> str:
    """Master entry point for generating summary and key points."""
    if isinstance(transcript_text_or_segments, list):
        transcript_str = format_transcript_for_llm(transcript_text_or_segments)
    else:
        transcript_str = str(transcript_text_or_segments).strip()

    if not transcript_str:
        return "No transcript text available to summarize."

    user_prompt = f"Here is the meeting / video transcript to analyze:\n\n---\n{transcript_str}\n---\n\nPlease generate the executive summary, key discussion points, speaker breakdown, and action items."

    if provider == "Ollama (Local)":
        return query_ollama(user_prompt, model_name=model_name, host=custom_host)
    elif provider == "Local Transformers (Hugging Face)":
        return query_hf_local(user_prompt, model_id=model_name, device=device)
    elif provider in ("OpenAI / Groq / Cloud API", "Cloud API"):
        if not api_key:
            raise ValueError("API Key is required for Cloud API provider.")
        base_url = "https://api.groq.com/openai/v1" if "groq" in model_name.lower() or "llama-3" in model_name.lower() else None
        return query_openai_api(user_prompt, api_key=api_key, model_name=model_name, base_url=base_url)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
