#!/usr/bin/env python3
"""End-of-meeting roundup — reuses cable_transcribe Llama finalize (no duplicate prompts)."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

from cable_transcribe import (
    FINALIZE_TXT,
    LOG_FILE,
    MEETING_META_FILE,
    MeetingSession,
    _default_meeting_name,
    _save_current_meeting_pointer,
    _set_active_session,
    _slugify,
    _unique_meeting_folder,
    ask_llama_finalize,
    extract_most_important,
    save_finalize_report,
)

ROUNDUP_MD = "MEETING-ROUNDUP.md"
APP_ROOT = Path(__file__).resolve().parent


def write_roundup_md(folder: Path, report: str, meeting_name: str) -> Path:
    md_path = folder / ROUNDUP_MD
    header = f"# {meeting_name} — Meeting Roundup\n\n"
    md_path.write_text(header + report.strip() + "\n", encoding="utf-8")
    return md_path


def finalize_meeting_session(
    session: MeetingSession,
    *,
    skip_ollama: bool = False,
) -> Path | None:
    """Run Llama finalize for a meeting folder. Returns path to MEETING-ROUNDUP.md."""
    _set_active_session(session)
    _save_current_meeting_pointer(session)

    transcript_path = session.folder / LOG_FILE
    if not transcript_path.is_file():
        print(f"No transcript at {transcript_path}", file=sys.stderr)
        return None

    transcript = transcript_path.read_text(encoding="utf-8").strip()
    if len(transcript) < 20:
        print("Transcript too short to finalize.", file=sys.stderr)
        return None

    if skip_ollama:
        return transcript_path

    print("Generating meeting roundup (Ollama)...", flush=True)
    report = ask_llama_finalize(
        transcript=transcript,
        chat="(none — loopback transcription only)",
        chat_archive="(none)",
        summaries="(none)",
        questions="(none)",
    )
    most, _body = extract_most_important(report)
    save_finalize_report(report, most)
    md_path = write_roundup_md(session.folder, report, session.name)
    print(f"Saved {FINALIZE_TXT} and {ROUNDUP_MD} in {session.folder}", flush=True)
    return md_path


def import_transcript_to_meeting(
    transcript_path: Path,
    meeting_name: str | None = None,
    *,
    skip_ollama: bool = False,
    source: str = "loopback-import",
) -> MeetingSession:
    """Copy a transcript into a new dated meeting folder and optionally finalize."""
    transcript_path = transcript_path.resolve()
    if not transcript_path.is_file():
        raise FileNotFoundError(f"Transcript not found: {transcript_path}")

    name = (meeting_name or _default_meeting_name()).strip()
    folder = _unique_meeting_folder(name)
    folder.mkdir(parents=True, exist_ok=True)
    started = datetime.now().isoformat(timespec="seconds")
    meta = {
        "name": name,
        "started": started,
        "slug": _slugify(name),
        "source": source,
        "imported_from": transcript_path.name,
    }
    (folder / MEETING_META_FILE).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    session = MeetingSession(name=name, folder=folder, started=started)
    shutil.copy2(transcript_path, folder / LOG_FILE)
    _set_active_session(session)
    _save_current_meeting_pointer(session)

    if not skip_ollama:
        finalize_meeting_session(session)

    return session


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate meeting roundup from transcript (Ollama via cable_transcribe)",
    )
    parser.add_argument(
        "--transcript",
        type=Path,
        help="Transcript file to import (e.g. loopback_live_transcript.txt)",
    )
    parser.add_argument(
        "--folder",
        type=Path,
        help="Existing meeting folder with transcript_log.txt",
    )
    parser.add_argument(
        "--name",
        help="Meeting name (for --transcript import only)",
    )
    parser.add_argument(
        "--no-ollama",
        action="store_true",
        help="Save transcript only — skip Ollama roundup",
    )
    args = parser.parse_args()

    if args.folder:
        folder = args.folder.resolve()
        meta_path = folder / MEETING_META_FILE
        name = folder.name
        started = datetime.now().isoformat(timespec="seconds")
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                name = str(meta.get("name", name))
                started = str(meta.get("started", started))
            except (json.JSONDecodeError, OSError):
                pass
        session = MeetingSession(name=name, folder=folder, started=started)
        result = finalize_meeting_session(session, skip_ollama=args.no_ollama)
    elif args.transcript:
        session = import_transcript_to_meeting(
            args.transcript,
            args.name,
            skip_ollama=args.no_ollama,
        )
        result = session.folder / ROUNDUP_MD if not args.no_ollama else session.folder / LOG_FILE
    else:
        parser.error("Provide --transcript or --folder")
        return

    print("--- SAVED ---")
    print("folder:", session.folder)
    print("transcript:", session.folder / LOG_FILE)
    if result and result.exists():
        print("roundup:", result)


if __name__ == "__main__":
    main()
