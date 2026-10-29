# Urdu + English Qwen Transcriber

A local Streamlit prototype that uses a two-stage workflow:

1. **Qwen3-ASR** produces the raw speech transcript, with Hindi selected by default as a workaround for Urdu-like speech.
2. **Ollama + a local text LLM** converts the Hindi/Devanagari transcript into Urdu script. The app offers Mixed Urdu-English and Natural Urdu styles.
3. Existing speaker-profile enrollment and similarity matching are retained as an optional feature.

> Important: Qwen3-ASR's published language list includes Hindi but does not list Urdu. This workaround is experimental; test it on your own Pakistani Urdu accents and code-switching examples. The converter cannot recover words Qwen failed to recognize.

## Recommended setup on Windows

**Use Python 3.12** in a fresh environment. The user's previous Python 3.15 setup may not be supported by all model dependencies yet.

1. Install Python 3.12 from the official Python website and make sure the Python launcher or `python` command works.
2. Open PowerShell in this project folder.
3. Create and activate a virtual environment:

   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

   If PowerShell blocks activation, use this command in the same terminal session:

   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   .\.venv\Scripts\Activate.ps1
   ```

4. Install [Ollama](https://ollama.com/download) and open it.
5. In a second PowerShell window, download the lightweight conversion model:

   ```powershell
   ollama pull qwen3:1.7b
   ```

6. Return to the project environment and start Streamlit:

   ```powershell
   streamlit run app.py
   ```

7. Open the local URL Streamlit prints, usually `http://localhost:8501`.

## Model notes for an 8 GB RAM PC

- The app defaults to `Qwen/Qwen3-ASR-1.7B`, because this is the quality-first model the user tested.
- If it runs out of memory, choose `Qwen/Qwen3-ASR-0.6B` in the sidebar, close other memory-heavy apps, or use a machine with more RAM.
- Keep **Unload ASR before conversion** enabled. The app clears the cached ASR model before it asks Ollama to load the text model. This reduces peak RAM use, but Qwen will need to reload on the next recording.
- The first run downloads model files and can take a while. Later runs are faster if the model stays cached; the unload option trades speed for lower RAM use.
- Qwen's model package and its dependencies are evolving. If installation fails, check the official Qwen3-ASR README: https://github.com/QwenLM/Qwen3-ASR

## Troubleshooting

### “Could not connect to Ollama”
Open the Ollama application and verify it is running. Then run:

```powershell
ollama list
ollama pull qwen3:1.7b
```

If you changed the model name in the sidebar, pull that exact model first.

### ASR runs out of memory or is extremely slow
Try `Qwen/Qwen3-ASR-0.6B`, close other applications, and keep the unload option enabled. CPU-only transcription on long recordings can be slow.

### Speaker profile fails on Windows
The speaker feature uses SpeechBrain and downloads a separate model. Run the app inside the Python 3.12 virtual environment above. If model file copying fails, check that the project folder is writable and try running from a normal user-owned folder (not `Program Files`). Profiles are stored in `speakers.sqlite3`.

## Files

- `app.py` — Streamlit application.
- `requirements.txt` — Python dependencies.
- `.gitignore` — excludes virtual environments, model files, databases and temporary artifacts.

## Privacy

Audio is processed locally by the models in this setup. The first run downloads model weights from their model repositories. Do not enroll or process another person's voice without their consent. If you later configure a cloud model or API, review its privacy terms separately.
