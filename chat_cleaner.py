"""Clean pasted Zoom/Teams meeting chat for Llama processing."""

from __future__ import annotations

import re

TIME_RE = re.compile(r"^\d{1,2}:\d{2}\s*(?:AM|PM)$", re.IGNORECASE)
INITIALS_RE = re.compile(r"^[A-Z]{2,4}$")
NEW_MSG_RE = re.compile(r"^\d+\s+new\s+message", re.IGNORECASE)

NOISE_EXACT = {
    "",
    "to",
    "everyone",
    "to:",
    "meeting chat",
    "collapse all",
    "who can see your messages?",
}


def is_collapse(line: str) -> bool:
    return line.strip().lower() == "collapse all"


def is_noise(line: str) -> bool:
    low = line.strip().lower()
    if low in NOISE_EXACT or is_collapse(line):
        return True
    if NEW_MSG_RE.search(low):
        return True
    if low in ("red heart",) or re.fullmatch(r"\d+", low):
        return True
    if low.startswith("↓") or low.startswith("↑"):
        return True
    if "new message" in low:
        return True
    if low in ("thumbs up", "red heart"):
        return True
    return False


def _is_attendee_list_line(line: str, lines: list[str], index: int) -> bool:
    """Skip bare names repeated in the participant list (not chat messages)."""
    if " " not in line.strip() or len(line) > 60:
        return False
    if index + 1 < len(lines) and lines[index + 1].strip().lower() in ("to", "meeting chat"):
        return False
    if index + 1 < len(lines) and lines[index + 1].strip() == line.strip():
        return True
    if index + 1 < len(lines) and " " in lines[index + 1] and lines[index + 1] != line:
        nxt = lines[index + 1]
        if nxt.lower() not in ("to", "everyone", "meeting chat") and not TIME_RE.match(nxt):
            if index + 2 >= len(lines) or lines[index + 2].lower() != "to":
                return True
    return False


def clean_meeting_chat(raw: str) -> str:
    """
    Parse pasted meeting chat into clean lines:
    [10:30 AM] Monica Viola (MV): i cannot reset my password
    """
    lines = [ln.strip() for ln in raw.splitlines()]
    out: list[str] = []
    seen: set[tuple[str, str]] = set()
    i = 0

    while i < len(lines):
        line = lines[i]
        if not line or is_noise(line):
            i += 1
            continue

        if _is_attendee_list_line(line, lines, i):
            i += 1
            continue

        # Zoom: Name / to / Everyone / TIME / INITIALS / message
        if (
            i + 3 < len(lines)
            and lines[i + 1].lower() == "to"
            and lines[i + 2].lower() == "everyone"
            and TIME_RE.match(lines[i + 3])
        ):
            name = line
            time_s = lines[i + 3]
            j = i + 4
            initials = ""
            if j < len(lines) and INITIALS_RE.match(lines[j]):
                initials = lines[j]
                j += 1
            msg_parts: list[str] = []
            while j < len(lines):
                if is_collapse(lines[j]):
                    j += 1
                    break
                if (
                    j + 2 < len(lines)
                    and lines[j + 1].lower() == "to"
                    and lines[j + 2].lower() == "everyone"
                ):
                    break
                if is_noise(lines[j]):
                    j += 1
                    continue
                if TIME_RE.match(lines[j]):
                    break
                msg_parts.append(lines[j])
                j += 1
            if msg_parts:
                msg = " ".join(msg_parts).strip()
                key = (name.lower(), msg.lower()[:100])
                if key not in seen and msg:
                    seen.add(key)
                    who = f"{name} ({initials})" if initials else name
                    out.append(f"[{time_s}] {who}: {msg}")
            i = j
            continue

        # Reply thread: Name + title / TIME / message
        if i + 1 < len(lines) and TIME_RE.match(lines[i + 1]):
            name = line
            time_s = lines[i + 1]
            j = i + 2
            while j < len(lines) and not lines[j].strip():
                j += 1
            initials = ""
            if j < len(lines) and INITIALS_RE.match(lines[j]):
                initials = lines[j]
                j += 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            msg_parts = []
            while j < len(lines):
                if is_collapse(lines[j]):
                    j += 1
                    break
                if TIME_RE.match(lines[j]):
                    break
                if (
                    j + 2 < len(lines)
                    and lines[j + 1].lower() == "to"
                    and lines[j + 2].lower() == "everyone"
                ):
                    break
                if is_noise(lines[j]):
                    j += 1
                    continue
                # Next line is a timestamp → new speaker, stop this message
                if j + 1 < len(lines) and TIME_RE.match(lines[j + 1]):
                    break
                msg_parts.append(lines[j])
                j += 1
            if msg_parts:
                msg = " ".join(msg_parts).strip()
                key = (name.lower(), msg.lower()[:100])
                if key not in seen and msg:
                    seen.add(key)
                    who = f"{name} ({initials})" if initials else name
                    out.append(f"[{time_s}] {who}: {msg}")
            i = j
            continue

        i += 1

    return "\n".join(out)


if __name__ == "__main__":
    import sys

    raw = sys.stdin.read() if not sys.argv[1:] else open(sys.argv[1], encoding="utf-8").read()
    print(clean_meeting_chat(raw))
