# 🎙️ Urdu & English AI Transcriber & Meeting Analyst

An end-to-end local speech recognition and intelligence application built with **Streamlit**, featuring:

1. **State-of-the-Art Urdu & English ASR**:
   - **Urdu Whisper Large-v3 Turbo (Recommended)**: OpenAI's highest-accuracy, fastest multilingual Whisper model with native Urdu support.
   - **Urdu Whisper Large-v3 (kingabzpro)**: Fine-tuned specifically on Pakistani Urdu corpora (Mozilla Common Voice & FLEURS).
   - **Roman Urdu Whisper Turbo (Farhan-SE)**: Specialized model for direct Roman Urdu transcription.
   - **Qwen3-ASR (1.7B & 0.6B)**, **Whisper-tiny Hinglish**, and **Tara**.

2. **Automatic Continuous Speaker Identification**:
   - Integrated directly into the main transcription workflow.
   - Compares voice embeddings against enrolled profiles using **SpeechBrain ECAPA-TDNN**.
   - If a speaker is enrolled, uses their real name (e.g., `Sawaiz Ahmad`).
   - If un-enrolled, dynamically clusters voice signatures into `Anonymous 1`, `Anonymous 2`, etc.
   - Standalone "Recognize speaker" page has been replaced by seamless automatic speaker tracking in the main transcript.

3. **Natural Roman Urdu Transliteration**:
   - High-coverage conversational Urdu dictionary + phonetic ligature transliterator.
   - Automatically converts native Urdu script or Hindi Devanagari into natural, standardized Roman Urdu.
   - Editable Roman spelling standard and custom vocabulary/word lists.

4. **AI Executive Summary & Key Points Generation**:
   - Built-in LLM integration generating:
     - 📌 **Executive Summary**
     - 🎯 **Key Points & Discussion Topics**
     - 👥 **Speaker-by-Speaker Contributions**
     - ⚡ **Action Items & Next Steps**
   - Supports **Ollama** (e.g. `qwen2.5:7b`, `llama3.2:3b`, `qwen3:1.7b`), in-process **Local Hugging Face Transformers** (`Qwen/Qwen2.5-1.5B-Instruct`), and **Cloud APIs** (Groq / OpenAI).

5. **Appealing & Intuitive UI**:
   - Modern conversation view with speaker avatars and timestamp badges.
   - Clear multi-stage visual pipeline progress indicator.
   - One-click export for TXT transcripts, structured JSON, and Markdown summaries.

---

## 🚀 Quick Start

### 1. Launch with Batch Script
Double-click `run_app.bat` or run in PowerShell:
```powershell
.\run_app.bat
```

### 2. Manual Launch
```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

The app opens automatically in your browser at `http://localhost:8501`.

---

## 📋 Features Overview

- **🎙️ Transcribe & Analyze**: Upload audio or video files (MP3, WAV, MP4, M4A, WEBM, etc.). Real-time chunking, progress bar, live speech preview, continuous speaker labeling, and AI summarization.
- **👥 Enroll Speaker**: Record or upload 5–15 seconds of clean speech to create an enrolled profile.
- **🗂️ Saved Profiles**: View and manage enrolled voice identities in the local SQLite database.
- **⚖️ Compare & Cross-check**: Evaluate two models or configurations on the same audio clip with word-level disagreement analysis.
- **📖 Word Lists & Spelling**: Customize find-and-replace terms and standardized Roman spellings.
