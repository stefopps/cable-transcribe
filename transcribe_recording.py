#!/usr/bin/env python3
"""Transcribe a pre-recorded audio/video file into a meeting folder."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import numpy as np
import whisper

from cable_transcribe import (
    CHUNK_DURATION,
    FINALIZE_TXT,
    LOG_FILE,
    MEETING_META_FILE,
    SAMPLE_RATE,
    WHISPER_MODEL,
    MeetingSession,
    _save_current_meeting_pointer,
    _set_active_session,
    _slugify,
    _unique_meeting_folder,
    ask_llama_finalize,
    extract_most_important,
    float32_to_int16,
    save_finalize_report,
    save_wav,
    session_path,
    transcribe_chunk,
)

AUDIO_EXTENSIONS = {".mp3", ".mp4", ".m4a", ".wav", ".webm", ".ogg", ".flac", ".mkv", ".mov"}


def format_timestamp(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 60:02d}:{s % 60:02d}"


def transcribe_file_chunks(
    model: whisper.Whisper,
    audio_path: Path,
    chunk_sec: int = CHUNK_DURATION,
    on_chunk: Callable[[str], None] | None = None,
) -> list[str]:
    """Load file once, transcribe fixed-length slices."""
    print(f"Loading audio from {audio_path}...", flush=True)
    audio = whisper.load_audio(str(audio_path))
    total_sec = len(audio) / SAMPLE_RATE
    chunk_samples = chunk_sec * SAMPLE_RATE
    n_chunks = max(1, int(np.ceil(len(audio) / chunk_samples)))
    print(
        f"Duration ~{total_sec:.0f}s — {n_chunks} chunk(s) of {chunk_sec}s",
        flush=True,
    )

    lines: list[str] = []
    temp_dir = Path(tempfile.mkdtemp(prefix="file_transcribe_"))

    for i in range(n_chunks):
        start = i * chunk_samples
        end = min(start + chunk_samples, len(audio))
        chunk = audio[start:end]
        if chunk.size == 0:
            continue

        wav_path = str(temp_dir / f"chunk_{i:04d}.wav")
        save_wav(wav_path, float32_to_int16(chunk.astype(np.float32)))

        t0 = start / SAMPLE_RATE
        print(f"  Chunk {i + 1}/{n_chunks} @ {format_timestamp(t0)}...", flush=True)
        text = transcribe_chunk(model, wav_path)
        if text:
            line = f"[{format_timestamp(t0)}] {text}"
            lines.append(line)
            if on_chunk:
                on_chunk(line)

    return lines


def create_meeting_for_recording(
    name: str,
    source_path: Path | None = None,
) -> MeetingSession:
    """Create meeting folder, meta, and optionally copy source recording."""
    folder = _unique_meeting_folder(name)
    folder.mkdir(parents=True, exist_ok=True)
    started = datetime.now().isoformat(timespec="seconds")
    meta = {
        "name": name,
        "started": started,
        "slug": _slugify(name),
        "source": "pre-recorded",
    }
    if source_path is not None:
        dest_name = f"recording{source_path.suffix.lower()}"
        dest = folder / dest_name
        shutil.copy2(source_path, dest)
        meta["recording_file"] = dest_name
    (folder / MEETING_META_FILE).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    session = MeetingSession(name=name, folder=folder, started=started)
    _set_active_session(session)
    _save_current_meeting_pointer(session)
    return session


def transcribe_recording(
    audio_path: Path,
    meeting_name: str | None = None,
    *,
    chunk_sec: int = CHUNK_DURATION,
    finalize: bool = True,
    copy_source: bool = True,
    on_status: Callable[[str], None] | None = None,
    on_chunk: Callable[[str], None] | None = None,
) -> MeetingSession:
    """Full pipeline: file → meeting folder → transcript → optional Llama finalize."""
    audio_path = audio_path.resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"File not found: {audio_path}")
    if audio_path.suffix.lower() not in AUDIO_EXTENSIONS:
        raise ValueError(
            f"Unsupported format {audio_path.suffix}. "
            f"Use: {', '.join(sorted(AUDIO_EXTENSIONS))}",
        )

    def status(msg: str) -> None:
        print(msg, flush=True)
        if on_status:
            on_status(msg)

    name = (meeting_name or audio_path.stem).strip() or "Recording"
    status(f"Creating meeting: {name}")
    session = create_meeting_for_recording(
        name,
        source_path=audio_path if copy_source else None,
    )
    log_path = session_path(LOG_FILE)
    log_path.write_text("", encoding="utf-8")

    status(f"Loading Whisper ({WHISPER_MODEL})...")
    model = whisper.load_model(WHISPER_MODEL)

    def append_line(line: str) -> None:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        if on_chunk:
            on_chunk(line)

    lines = transcribe_file_chunks(
        model,
        audio_path,
        chunk_sec=chunk_sec,
        on_chunk=append_line,
    )
    formatted = "\n".join(lines)
    if not formatted.strip():
        raise RuntimeError("No speech detected in recording.")

    if finalize:
        status("Generating meeting summary (Llama)...")
        report = ask_llama_finalize(
            transcript=formatted,
            chat="(none — pre-recorded file)",
            chat_archive="(none)",
            summaries="(none)",
            questions="(none)",
        )
        most, _body = extract_most_important(report)
        save_finalize_report(report, most)
        status(f"Saved summary to {FINALIZE_TXT}")

    return session


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe a pre-recorded meeting audio/video file",
    )
    parser.add_argument("audio_path", type=Path, help="mp3, mp4, wav, m4a, etc.")
    parser.add_argument("meeting_name", nargs="?", default=None)
    parser.add_argument(
        "--chunk-sec",
        type=int,
        default=CHUNK_DURATION,
        help=f"Seconds per Whisper chunk (default {CHUNK_DURATION})",
    )
    parser.add_argument(
        "--no-finalize",
        action="store_true",
        help="Transcript only — skip Llama summary",
    )
    parser.add_argument(
        "--no-copy",
        action="store_true",
        help="Do not copy source file into meeting folder",
    )
    parser.add_argument(
        "--format",
        action="store_true",
        help="After transcribe, build Word docs (format_meeting.py)",
    )
    args = parser.parse_args()

    try:
        session = transcribe_recording(
            args.audio_path,
            args.meeting_name,
            chunk_sec=args.chunk_sec,
            finalize=not args.no_finalize,
            copy_source=not args.no_copy,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    log_path = session.folder / LOG_FILE
    print("--- SAVED ---")
    print("folder:", session.folder)
    print("transcript:", log_path)
    fin = session.folder / FINALIZE_TXT
    if fin.exists():
        print("summary:", fin)

    if args.format:
        from format_meeting import format_meeting_folder

        print("--- FORMATTING ---")
        format_meeting_folder(session.folder, package=True)


if __name__ == "__main__":
    main()
