# Urdu + English Speech-to-Text & Speaker Recognition

This local Streamlit prototype provides:
- Multilingual speech-to-text with Faster-Whisper (Urdu, English, or auto-detect).
- Speaker enrollment and returning-speaker comparison using SpeechBrain ECAPA-TDNN embeddings.
- Local SQLite storage of voice embeddings and profile labels.

## Limitations and privacy
- This is a prototype, not a production biometric authentication system.
- Urdu transcription quality varies with accent, noise, and Urdu-English code-switching. Test with representative recordings.
- The speaker checkpoint is trained mainly on VoxCeleb interview audio; Urdu/Pakistani accent performance is not guaranteed.
- Similarity scores are not probabilities. Calibrate a threshold on recordings from your intended users.
- This compares a single-speaker recording with profiles; it does not separate several speakers in one conversation.
- Voice embeddings are biometric data. Obtain consent, limit access, protect backups, and allow deletion. Do not use this demo as the sole login/authentication factor.

## Requirements
- Windows 10/11, macOS, or Linux.
- Python 3.11 recommended.
- Internet for installing dependencies and downloading models the first time.
- Microphone.
- A reasonably modern computer. Larger Whisper models need more memory and run more slowly.

## Windows setup

Open PowerShell in this folder.

### 1. Create a virtual environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell blocks activation, run this in the current terminal and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 2. Install PyTorch first

For CPU use, try:

```powershell
python -m pip install torch torchaudio
```

If installation fails, use the official selector: https://pytorch.org/get-started/locally/

### 3. Install project dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Run the application

```powershell
python -m streamlit run app.py
```

The browser should open at http://localhost:8501. Keep the terminal open. The first model load downloads files and may take a few minutes.

## How to use it

### Speech to text
1. Open **Speech to text**.
2. Choose Auto-detect, Urdu, or English.
3. Record speech and click **Transcribe audio**.
4. Review and download the transcript.

The UI microphone recorder returns WAV audio. If the first run is slow, choose `tiny` or `base` in the sidebar; `small` is a reasonable quality/speed compromise for many computers.

### Speaker recognition
1. Obtain consent from the person.
2. Open **Enroll speaker**, enter a label, check the consent box, and record 5–10 seconds.
3. Open **Recognize speaker**, record a fresh sample, and click **Identify speaker**.
4. Review the score. A low score means no profile passed the selected threshold; it is not definitive proof that the speaker is new.

### Delete profiles
Use **Saved profiles**, tick **Confirm delete**, then click **Delete**.

## References
- Faster-Whisper: https://github.com/SYSTRAN/faster-whisper
- SpeechBrain speaker model: https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb
- Streamlit microphone widget: https://docs.streamlit.io/develop/api-reference/widgets/st.audio_input
- PyTorch install guide: https://pytorch.org/get-started/locally/

## Before production
Add authentication, HTTPS, access controls, encryption at rest, retention/deletion policies, monitoring, validation of false accepts/false rejects, and anti-spoofing if needed. Do not expose this prototype publicly as-is.
