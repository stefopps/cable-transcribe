@echo off
echo Stopping any running Cable Transcribe (meeting) instances...
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Where-Object { $_.CommandLine -like '*cable_transcribe*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
timeout /t 2 /nobreak >nul
echo Starting Cable Transcribe (meeting tool)...
cd /d "%~dp0"
python cable_transcribe.py
