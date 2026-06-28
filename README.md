# Cable Transcribe

Local speech-to-text tools for Windows. No cloud required for transcription.

| App | Purpose |
|-----|---------|
| **`live_dictate.py`** | Microphone dictation → editable text on screen |
| **`cable_transcribe.py`** | Meeting audio via VB-Audio CABLE + Llama summaries |

---

## Quick Start (new machine)

```powershell
# 1. Clone the repo
git clone https://github.com/stefopps/cable-transcribe.git
cd cable-transcribe

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Install Ollama and pull a model (for meeting summaries)
winget install Ollama.Ollama
ollama pull llama3.2:3b

# 4. Done — verify
python --version
python list_devices.py
```

Tkinter is included with Python on Windows.

---

## Live Dictate (recommended for dictation)

Speak into your mic; Whisper writes text into an editable transcript. No Llama, no meeting chat.

```bash
python live_dictate.py
```

### UI (top bar)

| Control | Action |
|---------|--------|
| **● Record** / **■ Stop** | Start / stop capture |
| **Copy** | Copy full transcript to clipboard |
| **Pause** | Pause while recording (enabled only during capture) |

### Collapsible sections

| Section | Default | Contents |
|---------|---------|----------|
| **TRANSCRIPT** | Open | Your words (editable) |
| **MICROPHONE & SETTINGS** | Collapsed | Mic dropdown, pin on top |
| **AUDIO MONITOR** | Collapsed (opens while recording) | Status, mic level bar |
| **ACTIONS & LOG** | Collapsed | Clear transcript, log file note |

### Which microphone?

- Gold line under the title shows the active device, e.g. `Mic: [1] Microphone (Brio 101)`.
- While recording: `● RECORDING FROM: [index] name`.
- Pick a device under **▶ MICROPHONE & SETTINGS** before you hit Record.

```bash
python live_dictate.py --list-devices   # all input devices
python live_dictate.py --device "Brio"  # match by name substring
python live_dictate.py --cli            # terminal only
```

### Defaults (`live_dictate.py` header)

| Setting | Value | Notes |
|---------|-------|-------|
| `CHUNK_DURATION` | `3` | Seconds per Whisper chunk |
| `WHISPER_MODEL` | `small` | OpenAI Whisper; use `medium` for best accuracy (slower) |
| Log file | `dictate_log.txt` | Appended each chunk |

### Smoke test

```bash
python smoke_test_dictate.py
```

---

## Cable Transcribe (meetings + Llama)

Real-time transcription from **VB-Audio CABLE Output**, with optional Ollama summaries.

```bash
python cable_transcribe.py
```

Terminal mode: `python cable_transcribe.py --cli`

### Why silence? (Cable vs ChatGPT vs Loopback)

| Method | What it captures | Setup |
|--------|------------------|-------|
| **ChatGPT browser Record** | Tab/system audio via browser (`getDisplayMedia` + "Share system audio") | None — Chrome exposes loopback to the page |
| **Cable Transcribe** (`cable_transcribe.py`) | Audio routed into **CABLE Input** | Windows playback → **CABLE Input**; app records **CABLE Output** |
| **Loopback Transcribe** (`loopback_transcribe.py`) | Whatever plays on **default speakers** (WASAPI loopback) | None — speakers stay on Realtek/normal output |

**Silence on Cable Transcribe** means playback is going to your speakers/headphones, not through VB-Cable. CABLE Output only hears what was sent to CABLE Input.

**Prefer loopback for "hear what's playing"** (ChatGPT-like):

```bat
C:\Users\steve\tools\personal-assistants\launchers\LAUNCH-MEETING-TRANSCRIBE.bat
C:\Users\steve\tools\personal-assistants\launchers\LAUNCH-MEETING-TRANSCRIBE.bat "Mountainside orientation"
```

Or:

```bash
python loopback_transcribe.py
python loopback_transcribe.py --meeting-name "HIPAA Day 3"
python loopback_transcribe.py --list-devices
python loopback_transcribe.py --device "Speakers (Realtek"
```

**On stop (Ctrl+C or close window):** transcript saves to `meetings/<timestamp>_<name>/transcript_log.txt`, then Ollama writes `meeting_finalize.txt` + `MEETING-ROUNDUP.md` (same Llama prompt as **Finalize Meeting** in the GUI).

Live mirror log (optional): `loopback_live_transcript.txt` in this folder (opened in Notepad while recording).

**Roundup an existing log without live audio:**

```bash
python meeting_roundup.py --transcript loopback_live_transcript.txt --name "My meeting"
```

### New meeting (important)

Each meeting gets its **own folder** under `meetings/` (name + timestamp). The UI has:

- **Meeting** — type a name (e.g. `HIPAA Day 3`, `Benefits Q&A`)
- **New Meeting** — archives the current session on disk and starts a **fresh** empty folder with that name

Llama and Finalize only read files for the **active** meeting — not old logs.

Old root-level files (`transcript_log.txt`, etc.) are moved to `meetings/archived_*` on startup automatically.

### Usage

1. Enter a **meeting name** and click **New Meeting** (or use the session created on launch).
2. Route meeting audio to **CABLE Input** (playback).
3. App listens on **CABLE Output** (recording).
4. Every **20s** (default), a transcript line appears.
5. **Ask Llama So Far** — running notes from full transcript + chat.
6. **Finalize Meeting** — end-of-session report.

Outputs per meeting folder: `transcript_log.txt`, `llama_notes.json`, `meeting_finalize.txt`, `meeting_meta.json`, etc.

### Pre-recorded file (no CABLE needed)

Drag an mp3/mp4/wav onto **`import_recording.bat`**, or:

```bash
python transcribe_recording.py "recording.mp4" "Meeting name" --format
```

This transcribes the file, runs Llama finalize, and builds Word docs + `Meeting-Package/`.

In the GUI: **Import Recording** (next to New Meeting).

Format an existing folder:

```bash
python format_meeting.py meetings\<folder-name> --package
```

### Transfer to another PC

```powershell
cd C:\Users\steve\cable-transcribe
.\bundle_for_transfer.ps1 -IncludeMeeting "2026-06-03_203457_whatsapp-2026-06-03-505-pm"
```

ZIP lands on Desktop. On the other PC: unzip → `install.bat` → `SETUP_OTHER_PC.md`.

### Config (`cable_transcribe.py`)

| Variable | Default |
|----------|---------|
| `CHUNK_DURATION` | `20` |
| `WHISPER_MODEL` | `base` |
| `OLLAMA_MODEL` | `llama3.2:3b` |

---

## Files

| File | Role |
|------|------|
| `live_dictate.py` | Mic dictation GUI |
| `loopback_transcribe.py` | WASAPI loopback — PC playback, auto roundup on stop |
| `meeting_roundup.py` | Ollama finalize for loopback or imported transcripts |
| `cable_transcribe.py` | CABLE + Llama meeting app |
| `transcribe_recording.py` | Pre-recorded file → transcript + finalize |
| `format_meeting.py` | Meeting folder → Word docs + client package |
| `import_recording.bat` | Drag-drop file import |
| `bundle_for_transfer.ps1` | ZIP portable pack for another PC |
| `install.bat` | `pip install -r requirements.txt` on new machine |
| `smoke_test_dictate.py` | Automated checks for Live Dictate |
| `list_devices.py` | Quick CABLE device listing |
| `chat_cleaner.py` | Meeting chat paste cleaner (used by cable app) |
| `requirements.txt` | Python dependencies |
