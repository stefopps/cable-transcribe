@echo off
REM Drag an audio/video file onto this .bat, or run: import_recording.bat "path\to\file.mp4"
cd /d "%~dp0"
if "%~1"=="" (
    echo Usage: drag a file onto this bat, or:
    echo   import_recording.bat "C:\path\to\recording.mp4" [meeting name]
    pause
    exit /b 1
)
if "%~2"=="" (
    python transcribe_recording.py "%~1" --format
) else (
    python transcribe_recording.py "%~1" "%~2" --format
)
echo.
echo Done. Open cable_transcribe.py to review, or check meetings\ folder.
pause
