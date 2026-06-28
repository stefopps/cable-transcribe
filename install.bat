@echo off
echo Cable Transcribe — install dependencies
cd /d "%~dp0"
python --version >nul 2>&1
if errorlevel 1 (
    echo Python not found. Install Python 3.11+ from python.org and retry.
    pause
    exit /b 1
)
echo Installing pip packages...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
echo.
echo Optional — Llama summaries need Ollama:
echo   ollama pull llama3.2:3b
echo.
echo Optional — live meetings need VB-Audio Virtual Cable.
echo.
echo Launch: open_meeting.bat  or  python cable_transcribe.py
pause
