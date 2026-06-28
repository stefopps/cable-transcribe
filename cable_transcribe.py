#!/usr/bin/env python3
"""
Real-time transcription from VB-Audio CABLE Output.
Whisper transcribes on a fixed chunk interval; Llama summarizes on demand via "Ask Llama So Far".
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import shutil
import tempfile
import threading
import time
import wave
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import sounddevice as sd
import whisper

from chat_cleaner import clean_meeting_chat

# ── Configuration ──────────────────────────────────────────────────────────────
CHUNK_DURATION = 20               # seconds per chunk (set to 120 for 2-minute chunks)
SAMPLE_RATE = 16000
REQUIRED_SAMPLES = CHUNK_DURATION * SAMPLE_RATE  # 320_000 samples @ 16 kHz for 20s
SILENCE_RMS_THRESHOLD = 0.01
WHISPER_MODEL = "base"
INPUT_DEVICE_NAME = "CABLE Output"
OLLAMA_MODEL = "llama3.2:3b"      # falls back to first llama* model in `ollama list`
LOG_FILE = "transcript_log.txt"
CHAT_LOG_FILE = "meeting_chat_clean.txt"
CHAT_ARCHIVE_FILE = "meeting_chat_log.txt"
NOTES_JSON = "llama_notes.json"
QUESTIONS_JSON = "llama_questions.json"
FINALIZE_TXT = "meeting_finalize.txt"
FINALIZE_JSON = "meeting_finalize.json"
MEETING_META_FILE = "meeting_meta.json"
LLAMA_AUTO_INTERVAL_SEC = 60    # auto-update Llama every 1 minute
TEMP_DIR = tempfile.mkdtemp(prefix="cable_transcribe_")

APP_ROOT = Path(__file__).resolve().parent
MEETINGS_DIR = APP_ROOT / "meetings"
CURRENT_MEETING_FILE = APP_ROOT / "current_meeting.json"
LEGACY_SESSION_FILES = (
    LOG_FILE,
    CHAT_LOG_FILE,
    CHAT_ARCHIVE_FILE,
    NOTES_JSON,
    QUESTIONS_JSON,
    FINALIZE_TXT,
    FINALIZE_JSON,
)


@dataclass
class MeetingSession:
    name: str
    folder: Path
    started: str


_active_session: MeetingSession | None = None


def _slugify(name: str) -> str:
    s = re.sub(r"[^\w\s-]", "", name.lower())
    s = re.sub(r"[\s_]+", "-", s).strip("-")
    return (s[:48] or "meeting")


def _default_meeting_name() -> str:
    return datetime.now().strftime("Meeting %Y-%m-%d %H:%M")


def _unique_meeting_folder(name: str) -> Path:
    MEETINGS_DIR.mkdir(parents=True, exist_ok=True)
    slug = _slugify(name)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    folder = MEETINGS_DIR / f"{stamp}_{slug}"
    n = 2
    while folder.exists():
        folder = MEETINGS_DIR / f"{stamp}_{slug}-{n}"
        n += 1
    return folder


def _set_active_session(session: MeetingSession) -> None:
    global _active_session
    _active_session = session


def get_active_session() -> MeetingSession:
    if _active_session is None:
        raise RuntimeError("No active meeting session")
    return _active_session


def session_path(filename: str) -> Path:
    return get_active_session().folder / filename


def _save_current_meeting_pointer(session: MeetingSession) -> None:
    rel = session.folder.relative_to(APP_ROOT).as_posix()
    CURRENT_MEETING_FILE.write_text(
        json.dumps(
            {
                "name": session.name,
                "folder": rel,
                "started": session.started,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def load_current_meeting() -> MeetingSession | None:
    if not CURRENT_MEETING_FILE.exists():
        return None
    try:
        data = json.loads(CURRENT_MEETING_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    folder = APP_ROOT / str(data.get("folder", "")).replace("/", os.sep)
    if not folder.is_dir():
        return None
    name = str(data.get("name", "")).strip() or folder.name
    started = str(data.get("started", ""))
    return MeetingSession(name=name, folder=folder, started=started)


def archive_legacy_root_files() -> Path | None:
    """Move old root-level session files into meetings/archived_*."""
    present = [f for f in LEGACY_SESSION_FILES if (APP_ROOT / f).exists()]
    if not present:
        return None
    MEETINGS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    dest = MEETINGS_DIR / f"archived_{stamp}"
    dest.mkdir(parents=True, exist_ok=True)
    for fname in present:
        shutil.move(str(APP_ROOT / fname), str(dest / fname))
    return dest


def start_new_meeting(name: str) -> MeetingSession:
    """Create a fresh meeting folder and make it the active session."""
    clean = (name or "").strip() or _default_meeting_name()
    folder = _unique_meeting_folder(clean)
    folder.mkdir(parents=True, exist_ok=True)
    started = datetime.now().isoformat(timespec="seconds")
    meta = {"name": clean, "started": started, "slug": _slugify(clean)}
    (folder / MEETING_META_FILE).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    session = MeetingSession(name=clean, folder=folder, started=started)
    _set_active_session(session)
    _save_current_meeting_pointer(session)
    return session


def init_session_on_startup() -> MeetingSession:
    archive_legacy_root_files()
    loaded = load_current_meeting()
    if loaded is not None:
        _set_active_session(loaded)
        return loaded
    return start_new_meeting(_default_meeting_name())


def _meeting_label(folder: Path) -> str:
    meta_path = folder / MEETING_META_FILE
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            name = str(meta.get("name", folder.name)).strip() or folder.name
            started = str(meta.get("started", ""))[:10]
            if started:
                return f"{name} ({started})"
            return name
        except (json.JSONDecodeError, OSError):
            pass
    return folder.name.replace("_", " ")


def list_all_meetings() -> list[tuple[str, Path]]:
    """All meeting folders, newest first — (dropdown label, folder path)."""
    if not MEETINGS_DIR.exists():
        return []
    items: list[tuple[str, Path, float]] = []
    for folder in MEETINGS_DIR.iterdir():
        if not folder.is_dir():
            continue
        items.append((_meeting_label(folder), folder, folder.stat().st_mtime))
    items.sort(key=lambda row: row[2], reverse=True)
    seen: set[str] = set()
    out: list[tuple[str, Path]] = []
    for label, folder, _ in items:
        key = label
        n = 2
        while key in seen:
            key = f"{label} [{folder.name}]"
            n += 1
            if n > 20:
                break
        seen.add(key)
        out.append((key, folder))
    return out


def switch_to_meeting(folder: Path) -> MeetingSession:
    """Point active session at an existing meeting folder."""
    meta_path = folder / MEETING_META_FILE
    name = folder.name
    started = ""
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            name = str(meta.get("name", name)).strip() or folder.name
            started = str(meta.get("started", started))
        except (json.JSONDecodeError, OSError):
            pass
    if not started:
        started = datetime.fromtimestamp(folder.stat().st_mtime).isoformat(
            timespec="seconds",
        )
    session = MeetingSession(name=name, folder=folder, started=started)
    _set_active_session(session)
    _save_current_meeting_pointer(session)
    return session

LLAMA_SO_FAR_PROMPT = """You are taking running notes on a live session (meeting chat + audio transcript).

Previous notes (update and extend — keep what's still accurate):
{previous_notes}

Session content so far (chat + audio):
{transcript}

Write updated running notes:
- Bullet list of everything important said so far
- End with exactly one plain-text line (no markdown, no bold):
MOST_IMPORTANT: [one clear sentence only — do not repeat the label in the sentence]
"""

LLAMA_QUESTION_PROMPT = """Answer the question using the session files below. The SEARCH MATCHES section lists lines that contain words from the question — use them.

=== SEARCH MATCHES (lines containing question keywords) ===
{excerpts}

=== AUDIO TRANSCRIPT (recent portion of full log) ===
{transcript}

=== LLAMA SUMMARIES (recent notes) ===
{summaries}

=== MEETING CHAT ===
{chat}

Question: {question}

Answer from the content above. If SEARCH MATCHES or the transcript mention the topic, use that text.
"""

LLAMA_FINALIZE_PROMPT = """The meeting has ended. Write a complete final summary using ALL session material below.

=== AUDIO TRANSCRIPT (full) ===
{transcript}

=== MEETING CHAT (cleaned) ===
{chat}

=== MEETING CHAT (archive log) ===
{chat_archive}

=== RUNNING NOTES (Llama summaries during session) ===
{summaries}

=== Q&A ASKED DURING SESSION ===
{questions}

Write a polished end-of-meeting report (plain text, no markdown tables).

MEETING OVERVIEW
2-4 sentences on what this session was about.

TOPIC GROUPS
Split the meeting into separate topic groups (e.g. HIPAA/PHI, benefits, IT systems, onboarding logistics).
For EACH group use exactly this format — one group per block, blank line between groups:

GROUP: [short topic name]
HIGHLIGHT: [one clear sentence — the main takeaway for this group only]
- [bullet detail]
- [bullet detail]

Include every major subject that had meaningful discussion. Do not merge unrelated topics into one group.

ACTION ITEMS & NEXT STEPS
Bullet list — emails, follow-up training, Day 3 agenda, logins, Symplr/NetLearning, etc.

UNANSWERED QUESTIONS
Bullet list from chat or transcript not clearly answered (or "None noted").

MOST_IMPORTANT: [one clear sentence — the single takeaway for the whole meeting]
"""


# ── Audio helpers ──────────────────────────────────────────────────────────────

def find_cable_output_device(name: str = INPUT_DEVICE_NAME) -> tuple[int, str]:
    matches = []
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] < 1:
            continue
        if name.lower() in d["name"].lower():
            matches.append((i, d["name"], d["max_input_channels"]))
    if not matches:
        raise RuntimeError(
            f'No input device matching "{name}" found.\n'
            + "\n".join(
                f"  {i}: {d['name']}"
                for i, d in enumerate(sd.query_devices())
                if d["max_input_channels"] > 0
            )
        )
    matches.sort(key=lambda m: (len(m[1]), m[2]), reverse=True)
    return matches[0][0], matches[0][1]


def to_mono_float32(audio: np.ndarray) -> np.ndarray:
    """Convert sounddevice block (frames, channels) to mono float32 in [-1, 1]."""
    if audio.ndim == 1:
        mono = audio.astype(np.float32, copy=False)
    else:
        mono = np.mean(audio, axis=1, dtype=np.float32)
    return np.clip(mono, -1.0, 1.0)


def float32_to_int16(audio: np.ndarray) -> np.ndarray:
    return (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)


def chunk_rms(audio_float: np.ndarray) -> float:
    if audio_float.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio_float.astype(np.float64) ** 2)))


def save_wav(path: str, audio: np.ndarray) -> None:
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())


def format_timestamp(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 60:02d}:{s % 60:02d}"


def resolve_ollama_model() -> str:
    import ollama

    try:
        models = [m.model for m in ollama.list().models]
    except Exception:
        return OLLAMA_MODEL

    for candidate in (OLLAMA_MODEL, "llama3", "llama3.2:3b"):
        if candidate in models:
            return candidate
    for name in models:
        if "llama" in name.split(":")[0].lower():
            return name
    return OLLAMA_MODEL


def ask_llama_question(
    question: str,
    excerpts: str,
    transcript: str,
    summaries: str,
    chat: str,
) -> str:
    import ollama

    model = resolve_ollama_model()
    prompt = LLAMA_QUESTION_PROMPT.format(
        question=question.strip(),
        excerpts=excerpts,
        transcript=transcript or "(empty)",
        summaries=summaries.strip() or "(no summaries yet)",
        chat=chat.strip() or "(no meeting chat)",
    )
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"].strip()


def ask_llama_finalize(
    transcript: str,
    chat: str,
    chat_archive: str,
    summaries: str,
    questions: str,
) -> str:
    import ollama

    model = resolve_ollama_model()
    prompt = LLAMA_FINALIZE_PROMPT.format(
        transcript=transcript.strip() or "(no audio transcript)",
        chat=chat.strip() or "(no cleaned chat)",
        chat_archive=chat_archive.strip() or "(no chat archive)",
        summaries=summaries.strip() or "(no running notes)",
        questions=questions.strip() or "(no questions asked)",
    )
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"].strip()


def ask_llama_so_far(session_context: str, previous_notes: str) -> str:
    import ollama

    model = resolve_ollama_model()
    prev = previous_notes.strip() or "(none yet — this is the first summary)"
    prompt = LLAMA_SO_FAR_PROMPT.format(
        previous_notes=prev,
        transcript=session_context,
    )
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"].strip()


def clean_hero_fact(text: str) -> str:
    """Strip markdown and repeated 'MOST IMPORTANT' labels for the hero card."""
    s = re.sub(r"\*+", "", text.strip())
    label = re.compile(
        r"^(?:most[_\s-]*important(?:\s+so\s+far)?)\s*:?\s*",
        re.IGNORECASE,
    )
    for _ in range(3):
        s = label.sub("", s, count=1).strip()
    return s


def extract_most_important(answer: str) -> tuple[str, str]:
    """Split Llama response into hero line + remaining notes body (no duplicate labels)."""
    text = answer.strip()
    patterns = [
        r"(?im)^\*{0,2}MOST_?IMPORTANT\*{0,2}:?\s*(.+)$",
        r"(?im)^Most important so far:\s*(.+)$",
        r"(?im)^Most important:\s*(.+)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            important = clean_hero_fact(match.group(1))
            body = re.sub(pattern, "", text, count=1).strip()
            body = re.sub(
                r"(?im)^\*{0,2}MOST_?IMPORTANT\*{0,2}:?.*$",
                "",
                body,
            ).strip()
            return important, body
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) >= 2:
        return clean_hero_fact(lines[-1]), "\n".join(lines[:-1])
    return clean_hero_fact(text), ""


def save_notes_json(notes: str, transcript_chars: int, most_important: str = "") -> None:
    path = session_path(NOTES_JSON)
    data: dict = {"entries": []}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            data = {"entries": []}
    data["entries"].append({
        "time": datetime.now().isoformat(timespec="seconds"),
        "transcript_chars": transcript_chars,
        "most_important": most_important,
        "notes": notes,
    })
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


MAX_LLAMA_CONTEXT_CHARS = 100_000  # cap per section to fit model context


def _read_text_file(name: str) -> str:
    path = session_path(name)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def _tail_if_too_long(text: str, label: str) -> str:
    if len(text) <= MAX_LLAMA_CONTEXT_CHARS:
        return text
    return (
        f"[{label}: truncated to last {MAX_LLAMA_CONTEXT_CHARS:,} chars]\n"
        + text[-MAX_LLAMA_CONTEXT_CHARS:]
    )


def load_full_transcript() -> str:
    """Always load the entire transcript log from disk."""
    return _tail_if_too_long(_read_text_file(LOG_FILE), "transcript")


def load_chat_archive() -> str:
    return _tail_if_too_long(_read_text_file(CHAT_ARCHIVE_FILE), "chat archive")


def load_all_questions() -> str:
    path = session_path(QUESTIONS_JSON)
    if not path.exists():
        return ""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return ""
    blocks: list[str] = []
    for item in data.get("questions", []):
        stamp = item.get("time", "")
        q = item.get("question", "").strip()
        a = item.get("answer", "").strip()
        if q:
            blocks.append(f"--- {stamp} ---\nQ: {q}\nA: {a}")
    combined = "\n\n".join(blocks)
    if len(combined) <= 30_000:
        return combined
    return f"[Q&A: last 30,000 chars]\n" + combined[-30_000:]


def load_full_chat(in_memory_chat: str = "") -> str:
    """Load cleaned chat from disk; use in-memory if newer/longer."""
    disk = _read_text_file(CHAT_LOG_FILE)
    mem = in_memory_chat.strip()
    chat = disk if len(disk) >= len(mem) else mem
    return _tail_if_too_long(chat, "chat")


def load_full_notes_summaries(max_entries: int | None = None) -> str:
    """Load saved Llama summaries from llama_notes.json."""
    path = session_path(NOTES_JSON)
    if not path.exists():
        return ""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return ""
    entries = data.get("entries", [])
    if max_entries is not None:
        entries = entries[-max_entries:]
    blocks: list[str] = []
    for entry in entries:
        stamp = entry.get("time", "")
        most = entry.get("most_important", "").strip()
        notes = entry.get("notes", "").strip()
        block = f"--- {stamp} ---"
        if most:
            block += f"\nKey: {most}"
        if notes:
            block += f"\n{notes}"
        blocks.append(block)
    combined = "\n\n".join(blocks)
    cap = 40_000 if max_entries else MAX_LLAMA_CONTEXT_CHARS
    if len(combined) <= cap:
        return combined
    return f"[summaries: last {cap:,} chars]\n" + combined[-cap:]


_QUESTION_STOPWORDS = frozenset({
    "the", "and", "for", "are", "but", "not", "you", "all", "can", "her", "was",
    "one", "our", "out", "day", "get", "has", "him", "his", "how", "its", "may",
    "new", "now", "old", "see", "two", "way", "who", "boy", "did", "its", "let",
    "put", "say", "she", "too", "use", "what", "when", "with", "they", "this",
    "that", "from", "have", "been", "were", "said", "about", "into", "your",
    "any", "there", "their", "would", "could", "should", "them", "then", "than",
})


def _is_transcript_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if s.startswith("SUMMARY:") or s.startswith("TRANSCRIPT:"):
        return False
    if s.startswith("2026-") and "SUMMARY" in s:
        return False
    return True


def _word_match(word: str, line: str) -> bool:
    return re.search(rf"\b{re.escape(word)}\b", line, re.IGNORECASE) is not None


def search_transcript_excerpts(question: str, raw_transcript: str | None = None) -> str:
    """Pull transcript lines that match question keywords (whole words only)."""
    if raw_transcript is None:
        raw_transcript = _read_text_file(LOG_FILE)
    if not raw_transcript:
        return "(no transcript file yet)"

    words = [
        w.lower()
        for w in re.findall(r"[a-zA-Z]{3,}", question)
        if w.lower() not in _QUESTION_STOPWORDS
    ]
    words = list(dict.fromkeys(words))
    if not words:
        return "(no search terms)"

    lines = raw_transcript.splitlines()
    seen: set[str] = set()
    hits: list[str] = []

    for i, line in enumerate(lines):
        if not _is_transcript_line(line):
            continue
        if not any(_word_match(w, line) for w in words):
            continue
        for ctx in lines[max(0, i - 1) : min(len(lines), i + 2)]:
            if ctx.strip() and _is_transcript_line(ctx) and ctx not in seen:
                seen.add(ctx)
                hits.append(ctx)

    if not hits:
        return f"(no transcript lines matched: {', '.join(words)})"
    return "\n".join(hits[:80])


def build_llama_context_for_question(question: str, meeting_chat: str) -> tuple[str, str, str, str]:
    """Build question context: search hits + transcript tail + recent summaries + chat."""
    raw = _read_text_file(LOG_FILE)
    excerpts = search_transcript_excerpts(question, raw)
    transcript = _tail_if_too_long(raw, "transcript") if raw else ""
    summaries = load_full_notes_summaries(max_entries=12)
    chat = load_full_chat(meeting_chat)
    return excerpts, transcript, summaries, chat


def build_finalize_context(meeting_chat: str) -> tuple[str, str, str, str, str]:
    """All sources for end-of-meeting finalize."""
    transcript = load_full_transcript()
    chat = load_full_chat(meeting_chat)
    chat_archive = load_chat_archive()
    summaries = load_full_notes_summaries()
    questions = load_all_questions()
    return transcript, chat, chat_archive, summaries, questions


def parse_topic_groups(report: str) -> list[dict]:
    """Extract GROUP / HIGHLIGHT blocks from a finalize report."""
    groups: list[dict] = []
    current: dict | None = None
    for line in report.splitlines():
        stripped = line.strip()
        upper = stripped.upper()
        if upper.startswith("GROUP:"):
            if current:
                groups.append(current)
            current = {
                "name": stripped.split(":", 1)[1].strip(),
                "highlight": "",
                "details": [],
            }
        elif upper.startswith("HIGHLIGHT:") and current is not None:
            current["highlight"] = stripped.split(":", 1)[1].strip()
        elif current is not None and stripped:
            if stripped.startswith("-") or stripped.startswith("•"):
                current["details"].append(stripped.lstrip("-•").strip())
            elif not upper.startswith("GROUP:"):
                current["details"].append(stripped)
    if current:
        groups.append(current)
    return groups


def save_finalize_report(report: str, most_important: str) -> Path:
    txt_path = session_path(FINALIZE_TXT)
    txt_path.write_text(report + "\n", encoding="utf-8")
    json_path = session_path(FINALIZE_JSON)
    data: dict = {"finalized": []}
    if json_path.exists():
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            data = {"finalized": []}
    data.setdefault("finalized", []).append({
        "time": datetime.now().isoformat(timespec="seconds"),
        "most_important": most_important,
        "topic_groups": parse_topic_groups(report),
        "report": report,
    })
    json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return txt_path


def build_llama_context(in_memory_transcript: str, meeting_chat: str) -> str:
    """Load full transcript + chat files from disk for broad Llama context."""
    parts: list[str] = []
    summaries = load_full_notes_summaries()
    if summaries:
        parts.append(f"=== LLAMA SUMMARIES (all saved) ===\n{summaries}")
    chat = load_full_chat(meeting_chat)
    if chat:
        parts.append(f"=== MEETING CHAT (cleaned) ===\n{chat}")
    audio = load_full_transcript()
    if audio:
        parts.append(f"=== AUDIO TRANSCRIPT (full log) ===\n{audio}")
    return "\n\n".join(parts)


def save_clean_chat(cleaned: str) -> Path:
    path = session_path(CHAT_LOG_FILE)
    path.write_text(cleaned + "\n", encoding="utf-8")
    return path


def save_question_json(question: str, answer: str) -> None:
    path = session_path(QUESTIONS_JSON)
    data: dict = {"questions": []}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            data = {"questions": []}
    data.setdefault("questions", []).append({
        "time": datetime.now().isoformat(timespec="seconds"),
        "question": question,
        "answer": answer,
    })
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def transcribe_chunk(model: whisper.Whisper, wav_path: str) -> str:
    result = model.transcribe(wav_path, fp16=False, language="en")
    return result.get("text", "").strip()


# ── Capture pipeline ─────────────────────────────────────────────────────────

def run_capture(
    stop_event: threading.Event,
    on_status: Callable[[str], None],
    on_chunk: Callable[[str, str], None],
    on_progress: Callable[[float], None] | None = None,
    pause_event: threading.Event | None = None,
) -> None:
    """
    Record fixed-length chunks from CABLE Output.
    Callback only appends audio; Whisper runs only after REQUIRED_SAMPLES collected.
    """
    assert REQUIRED_SAMPLES == CHUNK_DURATION * SAMPLE_RATE

    device_idx, device_name = find_cable_output_device()
    dev = sd.query_devices(device_idx)
    channels = min(2, int(dev["max_input_channels"])) or 1
    on_status(f"Loading Whisper ({WHISPER_MODEL})...")
    whisper_model = whisper.load_model(WHISPER_MODEL)
    on_status(
        f"Listening on {device_name} "
        f"({CHUNK_DURATION}s chunks = {REQUIRED_SAMPLES:,} samples)"
    )

    if pause_event is None:
        pause_event = threading.Event()

    # Thread-safe rolling buffer — callback ONLY appends, never calls Whisper
    buffer_lock = threading.Lock()
    buffer_parts: list[np.ndarray] = []
    total_samples = 0
    chunk_index = 0

    def audio_callback(indata, _frames, _time, status) -> None:
        nonlocal total_samples
        if pause_event.is_set():
            return
        if status:
            print(status, flush=True)
        block = to_mono_float32(indata)
        with buffer_lock:
            buffer_parts.append(block)
            total_samples += len(block)

    blocksize = 4096  # small blocks; chunking is driven by sample count, not block size

    with sd.InputStream(
        device=device_idx,
        channels=channels,
        samplerate=SAMPLE_RATE,
        dtype="float32",
        blocksize=blocksize,
        callback=audio_callback,
    ):
        while not stop_event.is_set():
            time.sleep(0.25)

            if pause_event.is_set():
                with buffer_lock:
                    buffer_parts.clear()
                    total_samples = 0
                on_status("Paused — on break (not recording)")
                if on_progress:
                    on_progress(-1.0)  # signal paused to UI
                continue

            with buffer_lock:
                samples_ready = total_samples

            if on_progress and samples_ready < REQUIRED_SAMPLES:
                on_progress(samples_ready / REQUIRED_SAMPLES)

            if samples_ready < REQUIRED_SAMPLES:
                continue

            # Extract exactly one full chunk (CHUNK_DURATION seconds of audio)
            with buffer_lock:
                parts: list[np.ndarray] = []
                need = REQUIRED_SAMPLES
                while need > 0 and buffer_parts:
                    part = buffer_parts[0]
                    if len(part) <= need:
                        parts.append(part)
                        need -= len(part)
                        buffer_parts.pop(0)
                    else:
                        parts.append(part[:need])
                        buffer_parts[0] = part[need:]
                        need = 0

                chunk_audio = np.concatenate(parts)
                total_samples = sum(len(p) for p in buffer_parts)

            assert len(chunk_audio) == REQUIRED_SAMPLES, (
                f"Chunk size wrong: {len(chunk_audio)} != {REQUIRED_SAMPLES}"
            )

            ts = format_timestamp(chunk_index * CHUNK_DURATION)

            if chunk_rms(chunk_audio) < SILENCE_RMS_THRESHOLD:
                on_status(f"[{ts}] silence — skipping Whisper")
                on_chunk(ts, "[silence]")
                with session_path(LOG_FILE).open("a", encoding="utf-8") as f:
                    f.write(f"\n[{ts}] [silence]\n")
            else:
                wav_path = os.path.join(TEMP_DIR, f"chunk_{chunk_index:05d}.wav")
                save_wav(wav_path, float32_to_int16(chunk_audio))
                on_status(f"Transcribing [{ts}] ({CHUNK_DURATION}s chunk)...")
                text = transcribe_chunk(whisper_model, wav_path)
                on_chunk(ts, text)
                with session_path(LOG_FILE).open("a", encoding="utf-8") as f:
                    f.write(f"\n[{ts}] {text}\n")
                try:
                    os.remove(wav_path)
                except OSError:
                    pass

            chunk_index += 1
            if on_progress:
                on_progress(0.0)

    on_status("Stopped")


# ── UI ─────────────────────────────────────────────────────────────────────────

def _load_transcript_into_memory() -> str:
    raw = _read_text_file(LOG_FILE)
    if not raw:
        return ""
    lines: list[str] = []
    for line in raw.splitlines():
        s = line.strip()
        if not s:
            continue
        if re.match(r"^\[\d{2}:\d{2}\]", s):
            lines.append(f"{s}\n\n")
        else:
            lines.append(f"{line}\n")
    return "".join(lines)


def run_ui() -> None:
    import tkinter as tk
    from tkinter import font as tkfont
    from tkinter import scrolledtext
    from tkinter import simpledialog
    from tkinter import ttk

    class CollapsibleSection:
        """Click header to expand/collapse a panel."""

        def __init__(
            self,
            parent,
            title: str,
            *,
            bg: str,
            card: str,
            gold: str,
            font,
            start_open: bool = True,
        ) -> None:
            self.title = title
            self._open = start_open
            self.outer = tk.Frame(parent, bg=bg)
            self.header = tk.Button(
                self.outer,
                text=self._header_text(),
                command=self.toggle,
                font=font,
                fg=gold,
                bg=card,
                activebackground=card,
                activeforeground=gold,
                anchor="w",
                relief=tk.FLAT,
                padx=10,
                pady=6,
                cursor="hand2",
            )
            self.header.pack(fill=tk.X)
            self.body = tk.Frame(self.outer, bg=bg)
            if start_open:
                self.body.pack(fill=tk.BOTH, expand=True)

        def _header_text(self) -> str:
            arrow = "▼" if self._open else "▶"
            return f" {arrow}  {self.title}"

        def toggle(self) -> None:
            self._open = not self._open
            self.header.configure(text=self._header_text())
            if self._open:
                self.body.pack(fill=tk.BOTH, expand=True)
            else:
                self.body.pack_forget()

        def grid(self, **kwargs) -> None:
            self.outer.grid(**kwargs)

    session = init_session_on_startup()

    root = tk.Tk()
    root.title(f"Cable Transcribe — {session.name}")
    root.geometry("580x900")
    root.minsize(400, 640)
    root.configure(bg="#0c0c0c")

    # BEIZA-inspired dark theme
    BG = "#0c0c0c"
    CARD = "#161616"
    BORDER = "#2a2a2a"
    TEXT = "#f5f5f5"
    MUTED = "#8a8a8a"
    GOLD = "#d4af37"
    GREEN = "#7dd87d"
    PILL_BG = "#f0f0f0"
    PILL_FG = "#0c0c0c"

    stop_event = threading.Event()
    pause_event = threading.Event()
    events: queue.Queue = queue.Queue()
    transcript_lock = threading.Lock()
    full_transcript = _load_transcript_into_memory()
    llama_notes_so_far = ""
    meeting_chat_clean = _read_text_file(CHAT_LOG_FILE)

    status_var = tk.StringVar(value="Starting...")
    meeting_folder_var = tk.StringVar(
        value=session.folder.relative_to(APP_ROOT).as_posix(),
    )
    progress_var = tk.StringVar(value="")

    # Responsive root grid
    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(0, weight=1)

    main = tk.Frame(root, bg=BG, padx=12, pady=10)
    main.grid(row=0, column=0, sticky="nsew")
    main.grid_rowconfigure(7, weight=5)   # live transcript — primary growing area
    main.grid_rowconfigure(8, weight=2)   # notes — directly under transcript
    main.grid_rowconfigure(9, weight=0)   # meeting chat — compact
    main.grid_columnconfigure(0, weight=1)

    title_font = tkfont.Font(family="Segoe UI", size=10, weight="bold")
    body_font = tkfont.Font(family="Segoe UI", size=10)
    mono_font = tkfont.Font(family="Consolas", size=10)
    small_font = tkfont.Font(family="Segoe UI", size=9)
    hero_font = tkfont.Font(family="Segoe UI", size=16, weight="bold")

    llama_busy = False
    auto_llama_enabled = tk.BooleanVar(value=True)
    pin_on_top = tk.BooleanVar(value=True)
    highlight_var = tk.StringVar(value="Waiting for first Llama update...")
    auto_var = tk.StringVar(value=f"Auto Llama: ON (every {LLAMA_AUTO_INTERVAL_SEC}s)")

    def on_toggle_pin() -> None:
        root.attributes("-topmost", pin_on_top.get())

    title_frame = tk.Frame(main, bg=BG)
    title_frame.grid(row=0, column=0, sticky="ew")
    title_frame.grid_columnconfigure(0, weight=1)
    tk.Label(title_frame, text="CABLE TRANSCRIBE", font=title_font, fg=TEXT, bg=BG).grid(
        row=0, column=0, sticky="w",
    )
    tk.Checkbutton(
        title_frame,
        text="Pin on top",
        variable=pin_on_top,
        command=on_toggle_pin,
        font=small_font,
        fg=TEXT,
        bg=BG,
        activebackground=BG,
        activeforeground=TEXT,
        selectcolor=CARD,
        cursor="hand2",
    ).grid(row=0, column=1, sticky="e", padx=(0, 8))

    def toggle_pause() -> None:
        if pause_event.is_set():
            pause_event.clear()
            pause_btn.configure(text="Pause", bg="#c45c5c", fg="white")
            status_var.set("Listening...")
        else:
            pause_event.set()
            pause_btn.configure(text="Resume", bg=GREEN, fg=PILL_FG)
            progress_var.set("Paused — on break")
            status_var.set("Paused — not recording")

    pause_btn = tk.Button(
        title_frame,
        text="Pause",
        command=toggle_pause,
        font=tkfont.Font(family="Segoe UI", size=10, weight="bold"),
        bg="#c45c5c",
        fg="white",
        activebackground="#a84a4a",
        relief=tk.FLAT,
        padx=14,
        pady=4,
        cursor="hand2",
    )
    pause_btn.grid(row=0, column=2, sticky="e")
    root.attributes("-topmost", True)

    meeting_frame = tk.Frame(main, bg=BG)
    meeting_frame.grid(row=1, column=0, sticky="ew", pady=(6, 0))
    meeting_frame.grid_columnconfigure(1, weight=1)
    tk.Label(
        meeting_frame, text="Meeting", font=small_font, fg=MUTED, bg=BG,
    ).grid(row=0, column=0, sticky="w", padx=(0, 8))
    meeting_name_var = tk.StringVar()
    meeting_by_label: dict[str, Path] = {}

    style = ttk.Style()
    style.theme_use("clam")
    style.configure(
        "Meeting.TCombobox",
        fieldbackground=CARD,
        background=CARD,
        foreground=TEXT,
        arrowcolor=GOLD,
        bordercolor=BORDER,
    )

    meeting_combo = ttk.Combobox(
        meeting_frame,
        textvariable=meeting_name_var,
        state="readonly",
        font=body_font,
        style="Meeting.TCombobox",
    )
    meeting_combo.grid(row=0, column=1, sticky="ew", ipady=4)

    def refresh_meeting_dropdown(*, select_folder: Path | None = None) -> None:
        meeting_by_label.clear()
        labels: list[str] = []
        active = get_active_session().folder
        pick = select_folder or active
        for label, folder in list_all_meetings():
            meeting_by_label[label] = folder
            labels.append(label)
        meeting_combo["values"] = labels
        chosen = ""
        for label, folder in meeting_by_label.items():
            if folder == pick:
                chosen = label
                break
        if chosen:
            meeting_name_var.set(chosen)
        elif labels:
            meeting_name_var.set(labels[0])

    def load_session_from_disk() -> None:
        nonlocal full_transcript, llama_notes_so_far, meeting_chat_clean
        full_transcript = _load_transcript_into_memory()
        meeting_chat_clean = _read_text_file(CHAT_LOG_FILE)
        llama_notes_so_far = load_full_notes_summaries(max_entries=1) or ""
        most = ""
        if llama_notes_so_far:
            most, _ = extract_most_important(llama_notes_so_far)
        highlight_var.set(most or "Waiting for first Llama update...")

        transcript_box.configure(state=tk.NORMAL)
        transcript_box.delete("1.0", tk.END)
        if full_transcript.strip():
            transcript_box.insert(tk.END, full_transcript)
            transcript_box.see(tk.END)
        transcript_box.configure(state=tk.DISABLED)

        chat_raw_box.delete("1.0", tk.END)
        chat_raw_box.insert(tk.END, "Paste Zoom/Teams chat here, then click Clean & Add...")
        chat_clean_box.configure(state=tk.NORMAL)
        chat_clean_box.delete("1.0", tk.END)
        if meeting_chat_clean.strip():
            chat_clean_box.insert(tk.END, meeting_chat_clean)
        chat_clean_box.configure(state=tk.DISABLED)

        notes_box.configure(state=tk.NORMAL)
        notes_box.delete("1.0", tk.END)
        summaries = load_full_notes_summaries()
        if summaries:
            notes_box.insert(tk.END, summaries + "\n")
        else:
            notes_box.insert(
                tk.END,
                f"Meeting: {get_active_session().name}\n(no Llama notes yet)\n\n",
            )
        notes_box.configure(state=tk.DISABLED)

        fin_path = session_path(FINALIZE_TXT)
        if fin_path.exists():
            show_finalize(fin_path.read_text(encoding="utf-8").strip())
        else:
            show_finalize(
                "When the meeting ends, click Finalize Meeting below.\n"
                "Uses only this meeting's transcript + chat files.\n",
            )

    def on_meeting_selected(_event: tk.Event | None = None) -> None:
        label = meeting_name_var.get().strip()
        folder = meeting_by_label.get(label)
        if folder is None or folder == get_active_session().folder:
            return
        switch_to_meeting(folder)
        sess = get_active_session()
        meeting_folder_var.set(sess.folder.relative_to(APP_ROOT).as_posix())
        root.title(f"Cable Transcribe — {sess.name}")
        load_session_from_disk()
        status_var.set(f"Switched to: {sess.name}")

    meeting_combo.bind("<<ComboboxSelected>>", on_meeting_selected)

    def reset_session_ui() -> None:
        nonlocal full_transcript, llama_notes_so_far, meeting_chat_clean
        full_transcript = ""
        llama_notes_so_far = ""
        meeting_chat_clean = ""
        highlight_var.set("Waiting for first Llama update...")
        transcript_box.configure(state=tk.NORMAL)
        transcript_box.delete("1.0", tk.END)
        transcript_box.configure(state=tk.DISABLED)
        notes_box.configure(state=tk.NORMAL)
        notes_box.delete("1.0", tk.END)
        notes_box.insert(
            tk.END,
            f"New meeting: {get_active_session().name}\n"
            f"Folder: {meeting_folder_var.get()}\n\n",
        )
        notes_box.configure(state=tk.DISABLED)
        chat_raw_box.delete("1.0", tk.END)
        chat_raw_box.insert(tk.END, "Paste Zoom/Teams chat here, then click Clean & Add...")
        chat_clean_box.configure(state=tk.NORMAL)
        chat_clean_box.delete("1.0", tk.END)
        chat_clean_box.configure(state=tk.DISABLED)
        answer_box.configure(state=tk.NORMAL)
        answer_box.delete("1.0", tk.END)
        answer_box.insert(tk.END, "Ask anything about what has been said so far...")
        answer_box.configure(state=tk.DISABLED)
        show_finalize(
            "When the meeting ends, click Finalize Meeting below.\n"
            "Uses only this meeting's transcript + chat files.\n",
        )

    def on_new_meeting() -> None:
        name = simpledialog.askstring(
            "New Meeting",
            "Meeting name:",
            initialvalue=_default_meeting_name(),
            parent=root,
        )
        if not name or not name.strip():
            status_var.set("New meeting cancelled.")
            return
        start_new_meeting(name.strip())
        sess = get_active_session()
        meeting_folder_var.set(sess.folder.relative_to(APP_ROOT).as_posix())
        root.title(f"Cable Transcribe — {sess.name}")
        reset_session_ui()
        refresh_meeting_dropdown(select_folder=sess.folder)
        status_var.set(f"New meeting started: {sess.name}")

    def on_import_recording() -> None:
        nonlocal llama_busy
        if llama_busy:
            status_var.set("Wait — Llama is busy.")
            return
        from tkinter import filedialog

        from transcribe_recording import AUDIO_EXTENSIONS, transcribe_recording

        path = filedialog.askopenfilename(
            title="Import pre-recorded meeting",
            filetypes=[
                ("Audio/Video", " ".join(f"*{e}" for e in sorted(AUDIO_EXTENSIONS))),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        name = simpledialog.askstring(
            "Meeting name",
            "Name for this recording:",
            initialvalue=Path(path).stem,
            parent=root,
        )
        if not name or not name.strip():
            status_var.set("Import cancelled.")
            return
        if not pause_event.is_set():
            pause_event.set()
            pause_btn.configure(text="Resume", bg=GREEN, fg=PILL_FG)
        llama_busy = True
        set_llama_buttons_enabled(False)
        status_var.set(f"Importing {Path(path).name}...")
        progress_var.set("Transcribing file — live capture paused")

        def worker() -> None:
            try:
                session = transcribe_recording(
                    Path(path),
                    name.strip(),
                    finalize=True,
                    copy_source=True,
                    on_status=lambda m: events.put(("status", m)),
                    on_chunk=lambda _line: None,
                )
                events.put(("import_done", session.folder))
            except Exception as exc:
                events.put(("import_error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    btn_row = tk.Frame(meeting_frame, bg=BG)
    btn_row.grid(row=0, column=2, sticky="e", padx=(8, 0))
    tk.Button(
        btn_row,
        text="New Meeting",
        command=on_new_meeting,
        font=tkfont.Font(family="Segoe UI", size=9, weight="bold"),
        bg=GOLD,
        fg=PILL_FG,
        activebackground="#b8962e",
        relief=tk.FLAT,
        padx=10,
        pady=4,
        cursor="hand2",
    ).pack(side=tk.LEFT, padx=(0, 4))
    tk.Button(
        btn_row,
        text="Import Recording",
        command=on_import_recording,
        font=tkfont.Font(family="Segoe UI", size=9, weight="bold"),
        bg="#4a7c9e",
        fg="white",
        activebackground="#3d6885",
        relief=tk.FLAT,
        padx=10,
        pady=4,
        cursor="hand2",
    ).pack(side=tk.LEFT)
    tk.Label(
        meeting_frame, textvariable=meeting_folder_var, font=small_font,
        fg=MUTED, bg=BG, anchor="w",
    ).grid(row=1, column=0, columnspan=3, sticky="ew", pady=(4, 0))
    refresh_meeting_dropdown(select_folder=session.folder)

    status_lbl = tk.Label(
        main, textvariable=status_var, font=small_font,
        fg=MUTED, bg=BG, justify=tk.LEFT, anchor="w",
    )
    status_lbl.grid(row=2, column=0, sticky="ew", pady=(4, 0))
    tk.Label(
        main, textvariable=progress_var, font=small_font,
        fg=GREEN, bg=BG, anchor="w",
    ).grid(row=3, column=0, sticky="w", pady=(2, 4))
    mode_frame = tk.Frame(main, bg=BG)
    mode_frame.grid(row=4, column=0, sticky="ew", pady=(0, 6))
    mode_frame.grid_columnconfigure(1, weight=1)

    def update_auto_label() -> None:
        if auto_llama_enabled.get():
            auto_var.set(f"Auto Llama: ON (every {LLAMA_AUTO_INTERVAL_SEC}s)")
        else:
            auto_var.set("Manual mode — click Ask Llama So Far")

    def on_toggle_auto() -> None:
        update_auto_label()

    tk.Checkbutton(
        mode_frame,
        text="Auto Llama",
        variable=auto_llama_enabled,
        command=on_toggle_auto,
        font=small_font,
        fg=TEXT,
        bg=BG,
        activebackground=BG,
        activeforeground=TEXT,
        selectcolor=CARD,
        cursor="hand2",
    ).grid(row=0, column=0, sticky="w")
    tk.Label(
        mode_frame, textvariable=auto_var, font=small_font,
        fg=GREEN, bg=BG, anchor="w",
    ).grid(row=0, column=1, sticky="w", padx=(8, 0))

    sec_hero = CollapsibleSection(
        main, "MOST IMPORTANT", bg=BG, card=CARD, gold=GOLD,
        font=small_font, start_open=False,
    )
    sec_hero.grid(row=5, column=0, sticky="ew", pady=(0, 4))
    hero_inner = tk.Frame(sec_hero.body, bg=CARD, padx=12, pady=10)
    hero_inner.pack(fill=tk.X)
    hero_inner.grid_columnconfigure(0, weight=1)
    hero_lbl = tk.Label(
        hero_inner, textvariable=highlight_var, font=hero_font,
        fg=TEXT, bg=CARD, justify=tk.LEFT, anchor="w", wraplength=460,
    )
    hero_lbl.grid(row=0, column=0, sticky="ew")

    sec_question = CollapsibleSection(
        main, "ASK A QUESTION", bg=BG, card=CARD, gold=GOLD,
        font=small_font, start_open=False,
    )
    sec_question.grid(row=6, column=0, sticky="ew", pady=(0, 4))
    question_frame = tk.Frame(sec_question.body, bg=BG)
    question_frame.pack(fill=tk.X, pady=(0, 8))
    question_frame.grid_columnconfigure(0, weight=1)

    question_var = tk.StringVar()
    question_entry = tk.Entry(
        question_frame,
        textvariable=question_var,
        font=body_font,
        bg=CARD,
        fg=TEXT,
        insertbackground=TEXT,
        relief=tk.FLAT,
        highlightthickness=1,
        highlightbackground=BORDER,
        highlightcolor=GOLD,
    )
    question_entry.grid(row=0, column=0, sticky="ew", ipady=10, padx=(0, 8))

    answer_frame = tk.Frame(sec_question.body, bg=CARD, padx=12, pady=8)
    answer_frame.pack(fill=tk.BOTH, expand=True)
    answer_frame.grid_columnconfigure(0, weight=1)
    tk.Label(answer_frame, text="Answer", font=small_font, fg=MUTED, bg=CARD).grid(
        row=0, column=0, sticky="w",
    )
    answer_box = scrolledtext.ScrolledText(
        answer_frame, wrap=tk.WORD, font=body_font, height=4,
        bg=CARD, fg=TEXT, insertbackground=TEXT,
        relief=tk.FLAT, padx=0, pady=6,
    )
    answer_box.grid(row=1, column=0, sticky="nsew")
    answer_frame.grid_rowconfigure(1, weight=1)
    answer_box.insert(tk.END, "Ask anything about what has been said so far...")
    answer_box.configure(state=tk.DISABLED)

    sec_transcript = CollapsibleSection(
        main, "LIVE TRANSCRIPT", bg=BG, card=CARD, gold=GOLD,
        font=small_font, start_open=True,
    )
    sec_transcript.grid(row=7, column=0, sticky="nsew", pady=(0, 4))
    transcript_wrap = tk.Frame(sec_transcript.body, bg=BG)
    transcript_wrap.pack(fill=tk.BOTH, expand=True)
    transcript_wrap.grid_rowconfigure(0, weight=1)
    transcript_wrap.grid_columnconfigure(0, weight=1)
    transcript_box = scrolledtext.ScrolledText(
        transcript_wrap, wrap=tk.WORD, font=mono_font, height=18,
        bg=CARD, fg=TEXT, insertbackground=TEXT,
        relief=tk.FLAT, padx=8, pady=8,
    )
    transcript_box.grid(row=0, column=0, sticky="nsew")
    transcript_box.configure(state=tk.DISABLED)

    _resize_drag: dict[str, int] = {"lines": 18, "y": 0}

    def _transcript_resize_start(event: tk.Event) -> None:
        _resize_drag["lines"] = int(transcript_box.cget("height"))
        _resize_drag["y"] = event.y_root

    def _transcript_resize_move(event: tk.Event) -> None:
        delta = event.y_root - _resize_drag["y"]
        lines = max(10, min(72, _resize_drag["lines"] + delta // 17))
        transcript_box.configure(height=lines)

    resize_grip = tk.Frame(
        transcript_wrap, bg=BORDER, height=8, cursor="size_ns",
    )
    resize_grip.grid(row=1, column=0, sticky="ew")
    resize_lbl = tk.Label(
        resize_grip, text="⇕ drag to resize transcript", font=small_font,
        fg=MUTED, bg=BORDER, cursor="size_ns",
    )
    resize_lbl.pack(fill=tk.X)
    for widget in (resize_grip, resize_lbl):
        widget.bind("<ButtonPress-1>", _transcript_resize_start)
        widget.bind("<B1-Motion>", _transcript_resize_move)

    sec_chat = CollapsibleSection(
        main, "MEETING CHAT", bg=BG, card=CARD, gold=GOLD,
        font=small_font, start_open=False,
    )
    sec_chat.grid(row=9, column=0, sticky="nsew", pady=(0, 4))
    chat_raw_box = scrolledtext.ScrolledText(
        sec_chat.body, wrap=tk.WORD, font=small_font, height=3,
        bg=CARD, fg=MUTED, insertbackground=TEXT, relief=tk.FLAT, padx=8, pady=6,
    )
    chat_raw_box.pack(fill=tk.X)
    chat_raw_box.insert(tk.END, "Paste Zoom/Teams chat here, then click Clean & Add...")
    chat_clean_box = scrolledtext.ScrolledText(
        sec_chat.body, wrap=tk.WORD, font=mono_font, height=4,
        bg=CARD, fg=TEXT, insertbackground=TEXT, relief=tk.FLAT, padx=8, pady=6,
    )
    chat_clean_box.pack(fill=tk.BOTH, expand=True, pady=(6, 0))
    chat_clean_box.configure(state=tk.DISABLED)
    if meeting_chat_clean.strip():
        chat_clean_box.configure(state=tk.NORMAL)
        chat_clean_box.insert(tk.END, meeting_chat_clean)
        chat_clean_box.configure(state=tk.DISABLED)

    chat_btn_row = tk.Frame(sec_chat.body, bg=BG)
    chat_btn_row.pack(fill=tk.X, pady=(6, 0))

    def on_clean_chat() -> None:
        nonlocal meeting_chat_clean
        raw = chat_raw_box.get("1.0", tk.END).strip()
        if not raw or raw.startswith("Paste Zoom"):
            status_var.set("Paste meeting chat first.")
            return
        cleaned = clean_meeting_chat(raw)
        meeting_chat_clean = cleaned
        save_clean_chat(cleaned)
        chat_clean_box.configure(state=tk.NORMAL)
        chat_clean_box.delete("1.0", tk.END)
        if cleaned:
            chat_clean_box.insert(tk.END, cleaned)
        else:
            chat_clean_box.insert(tk.END, "(no messages found — check paste format)")
        chat_clean_box.configure(state=tk.DISABLED)
        n = len(cleaned.splitlines()) if cleaned else 0
        status_var.set(f"Chat cleaned: {n} message(s) — included in Llama processing")

    tk.Button(
        chat_btn_row,
        text="Clean & Add to Processing",
        command=on_clean_chat,
        font=tkfont.Font(family="Segoe UI", size=10, weight="bold"),
        bg=PILL_BG,
        fg=PILL_FG,
        relief=tk.FLAT,
        padx=12,
        pady=6,
        cursor="hand2",
    ).pack(side=tk.LEFT)

    btn_frame = tk.Frame(main, bg=BG)
    btn_frame.grid(row=10, column=0, sticky="ew", pady=(0, 4))
    btn_frame.grid_columnconfigure(0, weight=1)

    def append_transcript(ts: str, text: str) -> None:
        nonlocal full_transcript
        display = text if text else "[silence]"
        line = f"[{ts}] {display}\n\n"
        with transcript_lock:
            full_transcript += line
        transcript_box.configure(state=tk.NORMAL)
        transcript_box.insert(tk.END, line)
        transcript_box.see(tk.END)
        transcript_box.configure(state=tk.DISABLED)

    def append_notes(header: str, body: str) -> None:
        if not body.strip():
            return
        block = f"--- {header} ---\n{body}\n\n"
        notes_box.configure(state=tk.NORMAL)
        notes_box.insert(tk.END, block)
        notes_box.see(tk.END)
        notes_box.configure(state=tk.DISABLED)

    def apply_llama_result(answer: str, source: str) -> None:
        nonlocal llama_notes_so_far
        llama_notes_so_far = answer
        most, body = extract_most_important(answer)
        highlight_var.set(most or clean_hero_fact(answer[:280]))
        tag = "Auto" if source == "auto" else "Manual"
        append_notes(f"{tag} {datetime.now().strftime('%H:%M:%S')}", body or answer)
        with transcript_lock:
            chars = len(full_transcript)
        save_notes_json(answer, chars, most_important=most)

    def request_llama(source: str = "manual") -> None:
        nonlocal llama_busy
        if llama_busy:
            return
        with transcript_lock:
            text = build_llama_context(full_transcript, meeting_chat_clean)
            prev_notes = load_full_notes_summaries() or llama_notes_so_far
        if not text:
            if source == "manual":
                append_notes("No transcript", f"Wait for the first {CHUNK_DURATION}s chunk.")
            return

        llama_busy = True
        set_llama_buttons_enabled(False)
        label = "Auto-updating Llama..." if source == "auto" else "Asking Llama (full files)..."
        status_var.set(f"{label} ({len(text):,} chars from disk)")

        def worker() -> None:
            try:
                answer = ask_llama_so_far(text, prev_notes)
                events.put(("llama_so_far", (answer, source)))
            except Exception as exc:
                events.put(("llama_error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def on_ask_llama_so_far() -> None:
        request_llama("manual")

    def set_llama_buttons_enabled(enabled: bool) -> None:
        state = tk.NORMAL if enabled else tk.DISABLED
        ask_btn.configure(state=state)
        ask_q_btn.configure(state=state)
        finalize_btn.configure(state=state)

    def show_answer(text: str) -> None:
        answer_box.configure(state=tk.NORMAL)
        answer_box.delete("1.0", tk.END)
        answer_box.insert(tk.END, text)
        answer_box.see(tk.END)
        answer_box.configure(state=tk.DISABLED)

    def on_ask_question() -> None:
        nonlocal llama_busy
        question = question_var.get().strip()
        if not question:
            show_answer("Type a question first.")
            return
        if llama_busy:
            return
        with transcript_lock:
            excerpts, transcript, summaries, chat = build_llama_context_for_question(
                question, meeting_chat_clean,
            )
        if not transcript and excerpts.startswith("(no"):
            show_answer(
                "No transcript on disk yet. Record audio first, or check transcript_log.txt."
            )
            return

        llama_busy = True
        set_llama_buttons_enabled(False)
        n = len(transcript) + len(excerpts)
        status_var.set(f"Asking Llama (search + {n:,} chars from transcript_log.txt)...")
        show_answer("Thinking...")

        def worker() -> None:
            try:
                answer = ask_llama_question(
                    question, excerpts, transcript, summaries, chat,
                )
                events.put(("llama_question", (question, answer)))
            except Exception as exc:
                events.put(("llama_question_error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    ask_q_btn = tk.Button(
        question_frame,
        text="Ask",
        command=on_ask_question,
        font=tkfont.Font(family="Segoe UI", size=10, weight="bold"),
        bg=PILL_BG,
        fg=PILL_FG,
        activebackground="#d0d0d0",
        activeforeground=PILL_FG,
        relief=tk.FLAT,
        padx=20,
        pady=8,
        cursor="hand2",
    )
    ask_q_btn.grid(row=0, column=1, sticky="e")
    question_entry.bind("<Return>", lambda _e: on_ask_question())

    def auto_llama_tick() -> None:
        if not stop_event.is_set():
            if auto_llama_enabled.get():
                request_llama("auto")
            root.after(LLAMA_AUTO_INTERVAL_SEC * 1000, auto_llama_tick)

    ask_btn = tk.Button(
        btn_frame, text="Ask Llama So Far", command=on_ask_llama_so_far,
        font=tkfont.Font(family="Segoe UI", size=11, weight="bold"),
        bg=GOLD, fg=PILL_FG, activebackground="#b8962e",
        activeforeground=PILL_FG, relief=tk.FLAT, padx=12, pady=8,
        cursor="hand2",
    )
    ask_btn.grid(row=0, column=0, sticky="ew")

    def on_finalize_meeting() -> None:
        nonlocal llama_busy
        if llama_busy:
            return
        # Meeting over — stop auto updates and pause recording
        auto_llama_enabled.set(False)
        update_auto_label()
        if not pause_event.is_set():
            pause_event.set()
            pause_btn.configure(text="Resume", bg=GREEN, fg=PILL_FG)
            progress_var.set("Paused — finalizing meeting")

        transcript, chat, chat_archive, summaries, questions = build_finalize_context(
            meeting_chat_clean,
        )
        total_chars = sum(len(s) for s in (transcript, chat, chat_archive, summaries, questions))
        if total_chars < 20:
            show_finalize(
                "Nothing to finalize yet.\n\n"
                "Record audio and/or paste meeting chat, then try again."
            )
            status_var.set("Finalize: no session content found")
            return

        llama_busy = True
        set_llama_buttons_enabled(False)
        status_var.set(f"Finalizing meeting ({total_chars:,} chars of context)...")
        show_finalize("Generating final meeting summary...\n\nThis may take a minute.")

        def worker() -> None:
            try:
                answer = ask_llama_finalize(
                    transcript, chat, chat_archive, summaries, questions,
                )
                events.put(("llama_finalize", answer))
            except Exception as exc:
                events.put(("llama_finalize_error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    finalize_btn = tk.Button(
        btn_frame,
        text="Finalize Meeting",
        command=on_finalize_meeting,
        font=tkfont.Font(family="Segoe UI", size=11, weight="bold"),
        bg="#4a7c9e",
        fg="white",
        activebackground="#3d6885",
        activeforeground="white",
        relief=tk.FLAT,
        padx=12,
        pady=8,
        cursor="hand2",
    )
    finalize_btn.grid(row=1, column=0, sticky="ew", pady=(6, 0))

    sec_finalize = CollapsibleSection(
        main,
        "FINAL MEETING SUMMARY",
        bg=BG,
        card=CARD,
        gold=GOLD,
        font=small_font,
        start_open=True,
    )
    sec_finalize.grid(row=11, column=0, sticky="nsew", pady=(0, 4))
    finalize_box = scrolledtext.ScrolledText(
        sec_finalize.body,
        wrap=tk.WORD,
        font=body_font,
        height=10,
        bg=CARD,
        fg=TEXT,
        insertbackground=TEXT,
        relief=tk.FLAT,
        padx=8,
        pady=8,
    )
    finalize_box.pack(fill=tk.BOTH, expand=True)
    finalize_header_font = tkfont.Font(family="Segoe UI", size=11, weight="bold")
    finalize_highlight_font = tkfont.Font(family="Segoe UI", size=10, weight="bold")
    finalize_box.tag_configure("section", foreground=GOLD, font=finalize_header_font)
    finalize_box.tag_configure("group", foreground=GOLD, font=finalize_header_font)
    finalize_box.tag_configure(
        "highlight", foreground=GREEN, font=finalize_highlight_font,
    )
    finalize_box.tag_configure("most", foreground=GOLD, font=hero_font)
    finalize_box.tag_configure("bullet", foreground=MUTED)

    def _finalize_line_tag(line: str) -> str | None:
        s = line.strip()
        if not s:
            return None
        u = s.upper()
        if u.startswith("MOST_IMPORTANT:") or u.startswith("MOST IMPORTANT:"):
            return "most"
        if u.startswith("GROUP:"):
            return "group"
        if u.startswith("HIGHLIGHT:"):
            return "highlight"
        if s.isupper() and len(s) > 3 and ":" not in s:
            return "section"
        if s.startswith("-") or s.startswith("•"):
            return "bullet"
        return None

    def show_finalize(text: str) -> None:
        finalize_box.configure(state=tk.NORMAL)
        finalize_box.delete("1.0", tk.END)
        for line in text.splitlines():
            tag = _finalize_line_tag(line)
            finalize_box.insert(tk.END, line + "\n", tag if tag else ())
        finalize_box.see("1.0")
        finalize_box.configure(state=tk.DISABLED)

    finalize_box.insert(
        tk.END,
        "When the meeting ends, click Finalize Meeting above.\n"
        "Each topic group gets its own HIGHLIGHT line (shown in green).\n",
    )
    finalize_box.configure(state=tk.DISABLED)

    sec_notes = CollapsibleSection(
        main, "NOTES SO FAR (LLAMA)", bg=BG, card=CARD, gold=GOLD,
        font=small_font, start_open=True,
    )
    sec_notes.grid(row=8, column=0, sticky="nsew", pady=(0, 4))
    notes_box = scrolledtext.ScrolledText(
        sec_notes.body, wrap=tk.WORD, font=body_font, height=6,
        bg=CARD, fg=TEXT, insertbackground=TEXT,
        relief=tk.FLAT, padx=8, pady=8,
    )
    notes_box.pack(fill=tk.BOTH, expand=True)
    notes_box.insert(
        tk.END,
        f"Toggle Auto Llama for 1-min updates, or use manual mode.\n"
        'Click "Ask Llama So Far" to refresh notes anytime.\n'
        "Click section headers to collapse/expand.\n\n",
    )
    notes_box.configure(state=tk.DISABLED)

    load_session_from_disk()

    def on_resize(event: tk.Event) -> None:
        if event.widget is root:
            wrap = max(200, event.width - 56)
            status_lbl.configure(wraplength=wrap)
            hero_lbl.configure(wraplength=wrap)

    root.bind("<Configure>", on_resize)

    def poll() -> None:
        nonlocal llama_busy
        try:
            while True:
                kind, payload = events.get_nowait()
                if kind == "status":
                    status_var.set(payload)
                elif kind == "progress":
                    if payload < 0:
                        progress_var.set("Paused — on break")
                    else:
                        secs = int(payload * CHUNK_DURATION)
                        progress_var.set(f"Recording chunk: {secs}s / {CHUNK_DURATION}s")
                elif kind == "chunk":
                    progress_var.set("")
                    ts, text = payload
                    append_transcript(ts, text)
                    status_var.set("Listening...")
                elif kind == "llama_so_far":
                    answer, source = payload
                    apply_llama_result(answer, source)
                    status_var.set("Listening...")
                    llama_busy = False
                    set_llama_buttons_enabled(True)
                elif kind == "llama_error":
                    append_notes("Error", str(payload))
                    status_var.set("Listening...")
                    llama_busy = False
                    set_llama_buttons_enabled(True)
                elif kind == "llama_question":
                    q, answer = payload
                    show_answer(answer)
                    save_question_json(q, answer)
                    append_notes(f"Q {datetime.now().strftime('%H:%M:%S')}", f"Q: {q}\nA: {answer}")
                    status_var.set("Listening...")
                    llama_busy = False
                    set_llama_buttons_enabled(True)
                elif kind == "llama_question_error":
                    show_answer(f"Error: {payload}")
                    status_var.set("Listening...")
                    llama_busy = False
                    set_llama_buttons_enabled(True)
                elif kind == "llama_finalize":
                    report = payload
                    most, body = extract_most_important(report)
                    show_finalize(report)
                    highlight_var.set(most or clean_hero_fact(report[:280]))
                    path = save_finalize_report(report, most)
                    append_notes(
                        f"FINALIZED {datetime.now().strftime('%H:%M:%S')}",
                        report,
                    )
                    status_var.set(f"Meeting finalized — saved to {path.name}")
                    llama_busy = False
                    set_llama_buttons_enabled(True)
                elif kind == "llama_finalize_error":
                    show_finalize(f"Finalize error:\n\n{payload}")
                    status_var.set("Finalize failed")
                    llama_busy = False
                    set_llama_buttons_enabled(True)
                elif kind == "import_done":
                    folder = payload
                    switch_to_meeting(folder)
                    sess = get_active_session()
                    meeting_folder_var.set(sess.folder.relative_to(APP_ROOT).as_posix())
                    root.title(f"Cable Transcribe — {sess.name}")
                    refresh_meeting_dropdown(select_folder=sess.folder)
                    load_session_from_disk()
                    try:
                        from format_meeting import format_meeting_folder

                        format_meeting_folder(sess.folder, package=True)
                        status_var.set(
                            f"Imported + formatted: {sess.name} — see Meeting-Package/",
                        )
                    except Exception as fmt_exc:
                        status_var.set(f"Imported (format skipped: {fmt_exc})")
                    progress_var.set("")
                    llama_busy = False
                    set_llama_buttons_enabled(True)
                elif kind == "import_error":
                    status_var.set(f"Import failed: {payload}")
                    progress_var.set("")
                    llama_busy = False
                    set_llama_buttons_enabled(True)
        except queue.Empty:
            pass
        if not stop_event.is_set():
            root.after(100, poll)

    def on_close() -> None:
        stop_event.set()
        status_var.set("Stopping...")
        root.after(400, root.destroy)

    root.protocol("WM_DELETE_WINDOW", on_close)

    def capture_worker() -> None:
        try:
            run_capture(
                stop_event,
                on_status=lambda m: events.put(("status", m)),
                on_chunk=lambda ts, t: events.put(("chunk", (ts, t))),
                on_progress=lambda p: events.put(("progress", p)),
                pause_event=pause_event,
            )
        except Exception as exc:
            events.put(("status", f"Error: {exc}"))

    threading.Thread(target=capture_worker, daemon=True).start()
    root.after(100, poll)
    root.after(LLAMA_AUTO_INTERVAL_SEC * 1000, auto_llama_tick)
    root.mainloop()


def run_cli() -> None:
    stop = threading.Event()
    lines: list[str] = []

    def on_chunk(ts: str, text: str) -> None:
        line = f"[{ts}] {text}"
        lines.append(line)
        print(line, flush=True)
        print(flush=True)

    try:
        run_capture(stop, on_status=print, on_chunk=on_chunk)
    except KeyboardInterrupt:
        stop.set()
        print("\nStopped.")
        if lines:
            print("\n--- Ask Llama manually with: ollama run llama3 ---")
            print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", action="store_true", help="Terminal mode (no UI)")
    args = parser.parse_args()
    if args.cli:
        run_cli()
    else:
        run_ui()


if __name__ == "__main__":
    main()
