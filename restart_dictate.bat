@echo off
echo Stopping any running Live Dictate instances...
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Where-Object { $_.CommandLine -like '*live_dictate*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
timeout /t 2 /nobreak >nul
echo Starting Live Dictate...
cd /d "%~dp0"
python live_dictate.py
