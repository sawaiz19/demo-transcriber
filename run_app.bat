@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\streamlit.exe" (
  echo Virtual environment not found. Follow README.md setup instructions first.
  pause
  exit /b 1
)
call ".venv\Scripts\activate.bat"
streamlit run app.py
