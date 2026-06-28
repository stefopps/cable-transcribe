# Cable Transcribe — setup on another PC

## 1. Copy the bundle

Copy the ZIP from Steve's other machine (or USB) and unzip to:

```
C:\Users\<you>\cable-transcribe
```

Or any folder you prefer — paths in `.bat` files are relative.

## 2. Install Python (once)

1. [python.org](https://www.python.org/downloads/) — Python **3.11+**
2. Check **"Add python.exe to PATH"** during install

## 3. Install dependencies

Double-click **`install.bat`** or:

```bat
cd C:\Users\<you>\cable-transcribe
python -m pip install -r requirements.txt
```

First Whisper run downloads the speech model (~150 MB for `base`).

## 4. Optional — Llama summaries

For "Ask Llama" and "Finalize Meeting":

1. Install [Ollama](https://ollama.com)
2. In a terminal: `ollama pull llama3.2:3b`

Without Ollama you can still transcribe (`--no-finalize`).

## 5. Optional — live meetings (VB-Audio CABLE)

Only needed for **real-time** meeting capture (not pre-recorded files):

1. Install [VB-Audio Virtual Cable](https://vb-audio.com/Cable/)
2. Route Teams/Zoom playback to **CABLE Input**
3. App listens on **CABLE Output**

## 6. Launch

| Task | How |
|------|-----|
| Live meeting | `open_meeting.bat` |
| Mic dictation | `restart_dictate.bat` |
| Pre-recorded file | Drag file onto `import_recording.bat` |
| CLI file import | `python transcribe_recording.py "file.mp4" "Meeting name" --format` |
| Format existing folder | `python format_meeting.py meetings\<folder> --package` |

## 7. Bring your meetings

Meeting data lives in `meetings\`. Copy specific folders from the old PC:

```
meetings\2026-06-03_203457_whatsapp-...
```

Or zip one meeting folder and unzip on the new PC under `meetings\`.

## Outputs per meeting

| File | What |
|------|------|
| `transcript_log.txt` | Timestamped transcript |
| `meeting_finalize.txt` | Llama summary |
| `Full Transcript.docx` | Word — full transcript |
| `Meeting Summary.docx` | Word — formatted summary |
| `Meeting-Package\` | Client-ready folder + `.bat` shortcuts |
