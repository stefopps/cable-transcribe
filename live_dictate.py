#!/usr/bin/env python3
"""
Live dictation: microphone → Whisper → text on screen (and dictate_log.txt).
No Llama, no meeting chat — just speak and transcribe.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import tempfile
import threading
import time
import wave
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import sounddevice as sd
import whisper

# ── Configuration ──────────────────────────────────────────────────────────────
CHUNK_DURATION = 3
SAMPLE_RATE = 16000
REQUIRED_SAMPLES = CHUNK_DURATION * SAMPLE_RATE
SILENCE_RMS_THRESHOLD = 0.006
UI_HEARING_THRESHOLD = 0.02
WHISPER_MODEL = "small"  # much better than tiny; use "medium" for best (slower)
LOG_FILE = "dictate_log.txt"
RECORDINGS_ROOT = "voice_training"
TEMP_DIR = tempfile.mkdtemp(prefix="live_dictate_")


def default_input_index() -> int:
    """Input index from sd.default.device (tuple or sounddevice._InputOutputPair)."""
    default = sd.default.device
    try:
        idx = default[0]
    except (TypeError, IndexError, KeyError):
        idx = default
    if idx is None:
        raise RuntimeError("No default input device. Use --device or --list-devices.")
    idx = int(idx)
    if idx < 0:
        raise RuntimeError("No default input device. Use --device or --list-devices.")
    return idx


def find_input_device(name: str | None = None) -> tuple[int, str]:
    """Default mic when name is None; otherwise first input device matching name."""
    if name is None:
        idx = default_input_index()
        dev = sd.query_devices(idx)
        if dev["max_input_channels"] < 1:
            raise RuntimeError(f'Default device "{dev["name"]}" has no input channels.')
        return idx, dev["name"]

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
    _save_wav_at_rate(path, audio, SAMPLE_RATE)


def _save_wav_at_rate(path: str, audio: np.ndarray, rate: int) -> None:
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(audio.tobytes())


def format_timestamp(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 60:02d}:{s % 60:02d}"


def use_fp16() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def transcribe_chunk(model: whisper.Whisper, wav_path: str) -> str:
    result = model.transcribe(
        wav_path,
        fp16=use_fp16(),
        language="en",
        condition_on_previous_text=False,
    )
    return result.get("text", "").strip()


def near_duplicate(prev: str, new: str) -> bool:
    """Skip repeated lines when Whisper echoes the prior chunk."""
    a = " ".join(prev.lower().split())
    b = " ".join(new.lower().split())
    if not a or not b:
        return False
    if a == b:
        return True
    if len(b) >= 8 and (b in a or a in b):
        return True
    aw, bw = set(a.split()), set(b.split())
    if len(aw) >= 3 and len(aw & bw) / max(len(aw), 1) > 0.85:
        return True
    return False


def _open_stream_with_supported_rate(
    device_idx: int,
    channels: int,
    blocksize: int,
    callback,
) -> tuple["sd.InputStream", int]:
    """Open InputStream at a rate the device supports; return stream + rate."""
    dev = sd.query_devices(device_idx)
    device_default = int(dev.get("default_samplerate") or 48000)
    candidates: list[int] = []
    for r in (SAMPLE_RATE, device_default, 48000, 44100, 32000, 22050, 16000, 8000):
        if r not in candidates:
            candidates.append(r)
    last_exc: Exception | None = None
    for rate in candidates:
        try:
            stream = sd.InputStream(
                device=device_idx,
                channels=channels,
                samplerate=rate,
                dtype="float32",
                blocksize=blocksize,
                callback=callback,
            )
            return stream, rate
        except Exception as exc:
            last_exc = exc
    raise RuntimeError(
        f"No supported sample rate. Last error: {last_exc}"
    )


def _resample_to_16k(audio: np.ndarray, src_rate: int) -> np.ndarray:
    if src_rate == SAMPLE_RATE:
        return audio
    try:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(src_rate, SAMPLE_RATE)
        return resample_poly(audio, SAMPLE_RATE // g, src_rate // g).astype(np.float32)
    except Exception:
        n_out = int(round(len(audio) * SAMPLE_RATE / src_rate))
        x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False, dtype=np.float64)
        x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False, dtype=np.float64)
        return np.interp(x_new, x_old, audio).astype(np.float32)


def run_capture(
    stop_event: threading.Event,
    on_status: Callable[[str], None],
    on_chunk: Callable[[str, str], None],
    on_progress: Callable[[float], None] | None = None,
    on_level: Callable[[float], None] | None = None,
    on_transcribing: Callable[[str], None] | None = None,
    on_device: Callable[[int, str], None] | None = None,
    on_saved_clip: Callable[[str, int], None] | None = None,
    pause_event: threading.Event | None = None,
    device_name: str | None = None,
    save_clips_event: threading.Event | None = None,
    save_session_dir: Path | None = None,
) -> None:
    device_idx, device_label = find_input_device(device_name)
    if on_device:
        on_device(device_idx, device_label)
    dev = sd.query_devices(device_idx)
    channels = min(2, int(dev["max_input_channels"])) or 1
    log_path = Path(__file__).resolve().parent / LOG_FILE

    on_status(f"Loading Whisper ({WHISPER_MODEL})...")
    whisper_model = whisper.load_model(WHISPER_MODEL)
    on_status(
        f"Listening on {device_label} "
        f"({CHUNK_DURATION}s chunks)"
    )

    if pause_event is None:
        pause_event = threading.Event()

    buffer_lock = threading.Lock()
    buffer_parts: list[np.ndarray] = []
    total_samples = 0
    chunk_index = 0
    transcribe_q: queue.Queue[tuple[str, str, str]] = queue.Queue()
    transcribe_lock = threading.Lock()

    def transcribe_worker() -> None:
        while not stop_event.is_set():
            try:
                ts, wav_path, saved_clip_path = transcribe_q.get(timeout=0.3)
            except queue.Empty:
                continue
            try:
                with transcribe_lock:
                    text = transcribe_chunk(whisper_model, wav_path)
                if is_likely_hallucination(text):
                    text = ""
                on_chunk(ts, text)
                if text:
                    with log_path.open("a", encoding="utf-8") as f:
                        f.write(f"[{ts}] {text}\n")
                if saved_clip_path and save_session_dir is not None:
                    try:
                        manifest = save_session_dir / "manifest.jsonl"
                        entry = {
                            "audio_file": Path(saved_clip_path).name,
                            "timestamp": ts,
                            "transcript": text,
                            "sample_rate": actual_rate,
                            "captured_at": datetime.now().isoformat(timespec="seconds"),
                        }
                        with manifest.open("a", encoding="utf-8") as f:
                            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    except Exception as exc:
                        on_status(f"Manifest write failed: {exc}")
            except Exception as exc:
                on_status(f"Transcribe error [{ts}]: {exc}")
                on_chunk(ts, "")
            finally:
                try:
                    os.remove(wav_path)
                except OSError:
                    pass
                transcribe_q.task_done()

    threading.Thread(target=transcribe_worker, daemon=True).start()

    def peek_mic_level() -> float:
        with buffer_lock:
            if not buffer_parts:
                return 0.0
            tail_samples = int(SAMPLE_RATE * 0.25)
            collected: list[np.ndarray] = []
            need = tail_samples
            for part in reversed(buffer_parts):
                if len(part) <= need:
                    collected.insert(0, part)
                    need -= len(part)
                else:
                    collected.insert(0, part[-need:])
                    break
            if not collected:
                return 0.0
            return chunk_rms(np.concatenate(collected))

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

    blocksize = 4096

    stream, actual_rate = _open_stream_with_supported_rate(
        device_idx, channels, blocksize, audio_callback,
    )
    required_samples = CHUNK_DURATION * actual_rate
    print(
        f"[live_dictate] opened {device_label} at {actual_rate} Hz "
        f"(will resample to {SAMPLE_RATE} Hz for Whisper)",
        flush=True,
    )
    if actual_rate != SAMPLE_RATE:
        on_status(
            f"Listening on {device_label} "
            f"({CHUNK_DURATION}s chunks, {actual_rate}→{SAMPLE_RATE} Hz)"
        )
    with stream:
        while not stop_event.is_set():
            time.sleep(0.25)

            if pause_event.is_set():
                with buffer_lock:
                    buffer_parts.clear()
                    total_samples = 0
                on_status("Paused — not recording")
                if on_progress:
                    on_progress(-1.0)
                continue

            with buffer_lock:
                samples_ready = total_samples

            if on_level:
                on_level(peek_mic_level())

            if on_progress and samples_ready < required_samples:
                on_progress(samples_ready / required_samples)

            if samples_ready < required_samples:
                continue

            with buffer_lock:
                parts: list[np.ndarray] = []
                need = required_samples
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
                chunk_audio_raw = np.concatenate(parts)
                total_samples = sum(len(p) for p in buffer_parts)

            chunk_audio = _resample_to_16k(chunk_audio_raw, actual_rate)
            ts = format_timestamp(chunk_index * CHUNK_DURATION)

            rms = chunk_rms(chunk_audio)
            if rms < SILENCE_RMS_THRESHOLD:
                on_chunk(ts, "")
            else:
                # Save raw native-rate audio for voice training
                saved_clip_path = ""
                if (
                    save_clips_event is not None
                    and save_clips_event.is_set()
                    and save_session_dir is not None
                ):
                    try:
                        save_session_dir.mkdir(parents=True, exist_ok=True)
                        clip_name = f"clip_{chunk_index:05d}.wav"
                        clip_path = save_session_dir / clip_name
                        _save_wav_at_rate(
                            str(clip_path),
                            float32_to_int16(chunk_audio_raw),
                            actual_rate,
                        )
                        saved_clip_path = str(clip_path)
                        if on_saved_clip:
                            on_saved_clip(clip_name, actual_rate)
                    except Exception as exc:
                        on_status(f"Save clip failed: {exc}")

                wav_path = os.path.join(TEMP_DIR, f"chunk_{chunk_index:05d}.wav")
                save_wav(wav_path, float32_to_int16(chunk_audio))
                if on_transcribing:
                    on_transcribing(ts)
                transcribe_q.put((ts, wav_path, saved_clip_path))

            chunk_index += 1
            if on_progress:
                on_progress(0.0)

    on_status("Stopped")


def enumerate_input_devices() -> list[tuple[int, str]]:
    devices: list[tuple[int, str]] = []
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] < 1:
            continue
        devices.append((i, d["name"]))
    return devices


def format_device_choice(index: int, name: str) -> str:
    return f"[{index}] {name}"


def parse_device_choice(choice: str) -> str | None:
    """Return device name substring from combobox value '[12] Mic name'."""
    choice = choice.strip()
    if "] " in choice:
        return choice.split("] ", 1)[1]
    return choice or None


def list_input_devices() -> None:
    try:
        default_in = default_input_index()
    except RuntimeError:
        default_in = -1
    print("Input devices:\n")
    for i, name in enumerate_input_devices():
        mark = " (default)" if i == default_in else ""
        print(f"  {i}: {name}{mark}")


# Whisper often invents these on quiet/noisy chunks.
_HALLUCINATION_PHRASES = frozenset({
    "thanks for watching",
    "thank you for watching",
    "thanks for listening",
    "subscribe",
    "thank you",
    "thanks",
    "bye",
    "you",
})


def is_likely_hallucination(text: str) -> bool:
    t = " ".join(text.lower().split())
    if not t:
        return True
    if t in _HALLUCINATION_PHRASES:
        return True
    if len(t) < 20 and t.endswith("!") and "thank" in t:
        return True
    return False


def run_ui(device_name: str | None) -> None:
    import tkinter as tk
    from tkinter import font as tkfont
    from tkinter import scrolledtext
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
            start_open: bool = False,
        ) -> None:
            self.title = title
            self._open = start_open
            self.outer = tk.Frame(parent, bg=bg)
            self.header = tk.Button(
                self.outer,
                text=self._header_text(title),
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

        def _header_text(self, title: str) -> str:
            arrow = "▼" if self._open else "▶"
            return f" {arrow}  {title}"

        def toggle(self) -> None:
            self._open = not self._open
            self.header.configure(text=self._header_text(self.title))
            if self._open:
                self.body.pack(fill=tk.BOTH, expand=True)
            else:
                self.body.pack_forget()

        def set_open(self, open_: bool) -> None:
            if open_ != self._open:
                self.toggle()

        def grid(self, **kwargs) -> None:
            self.outer.grid(**kwargs)

    BG = "#0c0c0c"
    CARD = "#161616"
    BORDER = "#2a2a2a"
    TEXT = "#f5f5f5"
    MUTED = "#8a8a8a"
    GOLD = "#d4af37"
    GREEN = "#7dd87d"
    PILL_BG = "#f0f0f0"
    PILL_FG = "#0c0c0c"

    root = tk.Tk()
    root.title("Live Dictate")
    root.geometry("520x520")
    root.minsize(360, 400)
    root.configure(bg=BG)
    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(0, weight=1)

    stop_event = threading.Event()
    pause_event = threading.Event()
    save_clips_event = threading.Event()
    events: queue.Queue = queue.Queue()
    full_text_lock = threading.Lock()
    full_text = ""
    capture_thread: threading.Thread | None = None
    is_recording = False
    capture_id = 0
    current_session_dir: Path | None = None
    saved_clips_count = 0

    status_var = tk.StringVar(value="Ready — pick your mic, then Start recording")
    progress_var = tk.StringVar(value="")
    live_var = tk.StringVar(value="")
    mic_var = tk.StringVar(value="Mic: not selected")
    pin_on_top = tk.BooleanVar(value=True)
    save_clips_var = tk.BooleanVar(value=False)
    last_line = ""
    level_display = 0.0

    main = tk.Frame(root, bg=BG, padx=12, pady=10)
    main.grid(row=0, column=0, sticky="nsew")
    main.grid_rowconfigure(3, weight=1)
    main.grid_columnconfigure(0, weight=1)
    ui_row = 0

    title_font = tkfont.Font(family="Segoe UI", size=11, weight="bold")
    body_font = tkfont.Font(family="Segoe UI", size=11)
    mono_font = tkfont.Font(family="Consolas", size=11)
    small_font = tkfont.Font(family="Segoe UI", size=9)

    header = tk.Frame(main, bg=BG)
    header.grid(row=ui_row, column=0, sticky="ew")
    ui_row += 1
    header.grid_columnconfigure(0, weight=1)

    tk.Label(header, text="LIVE DICTATE", font=title_font, fg=TEXT, bg=BG).grid(
        row=0, column=0, sticky="w",
    )

    def toggle_pause() -> None:
        if not is_recording:
            return
        if pause_event.is_set():
            pause_event.clear()
            pause_btn.configure(text="Pause", bg="#c45c5c", fg="white")
            status_var.set("Recording...")
            progress_var.set("")
        else:
            pause_event.set()
            pause_btn.configure(text="Resume", bg=GREEN, fg=PILL_FG)
            progress_var.set("")
            status_var.set("Paused — click Resume to continue")

    header_btns = tk.Frame(header, bg=BG)
    header_btns.grid(row=0, column=1, sticky="e")

    btn_font = tkfont.Font(family="Segoe UI", size=10, weight="bold")

    record_slot = tk.Frame(header_btns, bg=BG)
    record_slot.pack(side=tk.LEFT, padx=(0, 6))

    record_btn = tk.Button(
        record_slot,
        text="● Record",
        font=btn_font,
        bg=GREEN,
        fg=PILL_FG,
        activebackground="#6bc46b",
        activeforeground=PILL_FG,
        relief=tk.FLAT,
        padx=12,
        pady=4,
        cursor="hand2",
    )
    record_btn.pack(fill=tk.BOTH)

    stop_btn = tk.Button(
        record_slot,
        text="■ Stop",
        font=btn_font,
        bg="#c45c5c",
        fg="white",
        activebackground="#a84a4a",
        activeforeground="white",
        relief=tk.FLAT,
        padx=12,
        pady=4,
        cursor="hand2",
    )

    pause_btn = tk.Button(
        header_btns,
        text="Pause",
        command=toggle_pause,
        font=btn_font,
        bg="#c45c5c",
        fg="white",
        activebackground="#a84a4a",
        relief=tk.FLAT,
        padx=12,
        pady=4,
        cursor="hand2",
        state=tk.DISABLED,
    )
    root.attributes("-topmost", True)

    tk.Label(
        main, textvariable=mic_var, font=small_font,
        fg=GOLD, bg=BG, anchor="w", wraplength=480,
    ).grid(row=ui_row, column=0, sticky="ew", pady=(0, 2))
    ui_row += 1

    status_strip = tk.Frame(main, bg=BG)
    status_strip.grid(row=ui_row, column=0, sticky="ew", pady=(0, 4))
    ui_row += 1
    tk.Label(
        status_strip, textvariable=progress_var, font=small_font,
        fg=GREEN, bg=BG, anchor="w",
    ).pack(side=tk.LEFT)
    tk.Label(
        status_strip, textvariable=live_var, font=small_font,
        fg=MUTED, bg=BG, anchor="w",
    ).pack(side=tk.LEFT, padx=(12, 0))

    sec_transcript = CollapsibleSection(
        main, "TRANSCRIPT", bg=BG, card=CARD, gold=GOLD,
        font=small_font, start_open=True,
    )
    sec_transcript.grid(row=ui_row, column=0, sticky="nsew", pady=(0, 4))
    ui_row += 1

    sec_setup = CollapsibleSection(
        main, "MICROPHONE & SETTINGS", bg=BG, card=CARD, gold=GOLD,
        font=small_font, start_open=False,
    )
    sec_setup.grid(row=ui_row, column=0, sticky="ew", pady=(0, 4))
    ui_row += 1

    sec_monitor = CollapsibleSection(
        main, "AUDIO MONITOR", bg=BG, card=CARD, gold=GOLD,
        font=small_font, start_open=False,
    )
    sec_monitor.grid(row=ui_row, column=0, sticky="ew", pady=(0, 4))
    ui_row += 1

    sec_actions = CollapsibleSection(
        main, "ACTIONS & LOG", bg=BG, card=CARD, gold=GOLD,
        font=small_font, start_open=False,
    )
    sec_actions.grid(row=ui_row, column=0, sticky="ew")
    ui_row += 1

    device_options = [
        format_device_choice(i, n) for i, n in enumerate_input_devices()
    ]

    def pick_default_mic_in_combo() -> None:
        try:
            idx, name = find_input_device(device_name)
            choice = format_device_choice(idx, name)
            if choice in device_options:
                mic_combo.set(choice)
            elif device_options:
                mic_combo.current(0)
            mic_var.set(f"Mic: [{idx}] {name}")
        except RuntimeError as exc:
            mic_var.set(f"Mic: {exc}")
            if device_options:
                mic_combo.current(0)

    def get_selected_device_name() -> str | None:
        if device_name:
            return device_name
        sel = mic_combo.get().strip()
        return parse_device_choice(sel) if sel else None

    def on_mic_changed(_event=None) -> None:
        if is_recording:
            return
        try:
            idx, name = find_input_device(get_selected_device_name())
            mic_var.set(f"Mic: [{idx}] {name}")
        except RuntimeError as exc:
            mic_var.set(f"Mic: {exc}")

    mic_frame = tk.Frame(sec_setup.body, bg=BG)
    mic_frame.pack(fill=tk.X, padx=4, pady=4)
    mic_frame.grid_columnconfigure(1, weight=1)
    tk.Label(mic_frame, text="Device", font=small_font, fg=MUTED, bg=BG).grid(
        row=0, column=0, sticky="w", padx=(0, 8),
    )
    mic_combo = ttk.Combobox(
        mic_frame, values=device_options, state="readonly", font=small_font,
    )
    mic_combo.grid(row=0, column=1, sticky="ew")

    def on_toggle_pin() -> None:
        root.attributes("-topmost", pin_on_top.get())

    setup_opts = tk.Frame(sec_setup.body, bg=BG)
    setup_opts.pack(fill=tk.X, padx=4, pady=(0, 6))
    tk.Checkbutton(
        setup_opts,
        text="Pin window on top",
        variable=pin_on_top,
        command=on_toggle_pin,
        font=small_font,
        fg=TEXT,
        bg=BG,
        activebackground=BG,
        activeforeground=TEXT,
        selectcolor=CARD,
        cursor="hand2",
    ).pack(anchor="w")

    def on_toggle_save_clips() -> None:
        if save_clips_var.get():
            save_clips_event.set()
            live_var.set(
                "Saving audio clips for voice training "
                f"→ {RECORDINGS_ROOT}/"
            )
        else:
            save_clips_event.clear()
            live_var.set("Audio clip saving OFF")

    tk.Checkbutton(
        setup_opts,
        text=f"Save audio clips for voice training (→ {RECORDINGS_ROOT}/)",
        variable=save_clips_var,
        command=on_toggle_save_clips,
        font=small_font,
        fg=GOLD,
        bg=BG,
        activebackground=BG,
        activeforeground=GOLD,
        selectcolor=CARD,
        cursor="hand2",
    ).pack(anchor="w", pady=(4, 0))

    mic_combo.bind("<<ComboboxSelected>>", on_mic_changed)
    pick_default_mic_in_combo()

    monitor_inner = tk.Frame(sec_monitor.body, bg=BG, padx=4, pady=4)
    monitor_inner.pack(fill=tk.X)
    monitor_inner.grid_columnconfigure(0, weight=1)
    tk.Label(
        monitor_inner, textvariable=status_var, font=small_font,
        fg=MUTED, bg=BG, anchor="w", justify=tk.LEFT, wraplength=460,
    ).grid(row=0, column=0, sticky="ew", pady=(0, 6))
    level_frame = tk.Frame(monitor_inner, bg=BG)
    level_frame.grid(row=1, column=0, sticky="ew")
    level_frame.grid_columnconfigure(1, weight=1)
    tk.Label(level_frame, text="Level", font=small_font, fg=MUTED, bg=BG).grid(
        row=0, column=0, sticky="w", padx=(0, 6),
    )
    level_bar = tk.Canvas(level_frame, height=10, bg=CARD, highlightthickness=0)
    level_bar.grid(row=0, column=1, sticky="ew")

    transcript_box = scrolledtext.ScrolledText(
        sec_transcript.body,
        wrap=tk.WORD,
        font=mono_font,
        bg=CARD,
        fg=TEXT,
        insertbackground=TEXT,
        relief=tk.FLAT,
        padx=10,
        pady=10,
        highlightthickness=1,
        highlightbackground=BORDER,
        highlightcolor=GOLD,
    )
    transcript_box.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
    transcript_box.insert(
        tk.END,
        f"Your words appear here (editable). Updates every {CHUNK_DURATION}s.\n\n",
    )

    btn_row = tk.Frame(sec_actions.body, bg=BG)
    btn_row.pack(fill=tk.X, padx=4, pady=4)
    btn_row.grid_columnconfigure(0, weight=1)
    btn_row.grid_columnconfigure(1, weight=1)

    def get_plain_text() -> str:
        return transcript_box.get("1.0", tk.END).strip()

    def draw_level(rms: float) -> None:
        level_bar.delete("all")
        w = max(level_bar.winfo_width(), 120)
        h = 10
        fill = min(1.0, rms * 12.0)
        bar_w = int(w * fill)
        color = GREEN if fill > 0.08 else GOLD if fill > 0.02 else BORDER
        if bar_w > 0:
            level_bar.create_rectangle(0, 0, bar_w, h, fill=color, outline="")

    def append_chunk(ts: str, text: str) -> None:
        nonlocal full_text, last_line
        if not text:
            live_var.set("")
            return
        if near_duplicate(last_line, text):
            live_var.set("(skipped duplicate)")
            return
        last_line = text
        paragraph = text + " "
        with full_text_lock:
            full_text += paragraph
        transcript_box.insert(tk.END, paragraph)
        transcript_box.see(tk.END)
        live_var.set(f"Added: {text[:72]}{'…' if len(text) > 72 else ''}")

    def on_copy() -> None:
        text = get_plain_text()
        hint = (
            f"Your words appear here (editable). Updates every {CHUNK_DURATION}s."
        )
        if text.startswith(hint):
            text = text[len(hint):].strip()
        if not text:
            live_var.set("Nothing to copy yet.")
            return
        root.clipboard_clear()
        root.clipboard_append(text)
        live_var.set("Copied to clipboard.")

    copy_btn = tk.Button(
        header_btns,
        text="Copy",
        command=on_copy,
        font=btn_font,
        bg=PILL_BG,
        fg=PILL_FG,
        activebackground="#d0d0d0",
        activeforeground=PILL_FG,
        relief=tk.FLAT,
        padx=12,
        pady=4,
        cursor="hand2",
    )
    copy_btn.pack(side=tk.LEFT, padx=(0, 6))
    pause_btn.pack(side=tk.LEFT)

    def on_clear() -> None:
        nonlocal full_text, last_line
        with full_text_lock:
            full_text = ""
        last_line = ""
        transcript_box.delete("1.0", tk.END)
        live_var.set("")
        status_var.set("Cleared.")

    tk.Button(
        btn_row,
        text="Clear transcript",
        command=on_clear,
        font=body_font,
        bg=CARD,
        fg=TEXT,
        relief=tk.FLAT,
        padx=12,
        pady=8,
        cursor="hand2",
    ).pack(fill=tk.X)

    log_path = Path(__file__).resolve().parent / LOG_FILE
    tk.Label(
        sec_actions.body,
        text=f"Auto-saved to {log_path.name}",
        font=small_font,
        fg=MUTED,
        bg=BG,
        anchor="w",
    ).pack(anchor="w", padx=4, pady=(0, 6))

    def drain_events() -> None:
        try:
            while True:
                events.get_nowait()
        except queue.Empty:
            pass

    def set_recording_ui(active: bool) -> None:
        nonlocal is_recording
        is_recording = active
        if active:
            record_btn.pack_forget()
            stop_btn.pack(fill=tk.BOTH)
            pause_btn.configure(state=tk.NORMAL, text="Pause", bg="#c45c5c", fg="white")
            mic_combo.configure(state="disabled")
            sec_monitor.set_open(True)
        else:
            stop_btn.pack_forget()
            record_btn.pack(fill=tk.BOTH)
            pause_btn.configure(state=tk.DISABLED, text="Pause", bg="#c45c5c", fg="white")
            mic_combo.configure(state="readonly")
            progress_var.set("")
            pause_event.clear()
            sec_monitor.set_open(False)

    def kill_stale_mic_holders() -> None:
        """Kill other live_dictate python processes that may hold the mic."""
        import subprocess
        my_pid = os.getpid()
        try:
            out = subprocess.check_output(
                ["wmic", "process", "where",
                 "name='python.exe'", "get", "processid,commandline"],
                text=True, stderr=subprocess.DEVNULL,
            )
            for line in out.splitlines():
                if "live_dictate" in line.lower():
                    parts = line.strip().split()
                    pid = int(parts[-1])
                    if pid != my_pid:
                        try:
                            subprocess.call(
                                ["taskkill", "/PID", str(pid), "/F"],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                            )
                        except Exception:
                            pass
        except Exception:
            pass

    def start_recording() -> None:
        nonlocal capture_thread, capture_id, current_session_dir, saved_clips_count
        if is_recording:
            return
        # Kill stale mic-holding processes, then wait for mic to be released
        kill_stale_mic_holders()
        if capture_thread is not None and capture_thread.is_alive():
            stop_event.set()
            capture_thread.join(timeout=5.0)
        time.sleep(0.4)
        capture_id += 1
        my_id = capture_id
        drain_events()
        stop_event.clear()
        pause_event.clear()
        # Create a session folder for saved clips (used only if toggle is on)
        session_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        current_session_dir = (
            Path(__file__).resolve().parent / RECORDINGS_ROOT / session_stamp
        )
        saved_clips_count = 0
        set_recording_ui(True)
        status_var.set("Starting microphone and Whisper...")
        progress_var.set("")

        def capture_worker(run_id: int) -> None:
            import traceback
            err_log = Path(__file__).resolve().parent / "dictate_errors.log"
            try:
                run_capture(
                    stop_event,
                    on_status=lambda m: events.put(("status", m)),
                    on_chunk=lambda ts, t: events.put(("chunk", (ts, t))),
                    on_progress=lambda p: events.put(("progress", p)),
                    on_level=lambda r: events.put(("level", r)),
                    on_transcribing=lambda ts: events.put(("transcribing", ts)),
                    on_device=lambda i, n: events.put(("device", (i, n))),
                    on_saved_clip=lambda name, rate: events.put(
                        ("saved_clip", (name, rate))
                    ),
                    pause_event=pause_event,
                    device_name=get_selected_device_name(),
                    save_clips_event=save_clips_event,
                    save_session_dir=current_session_dir,
                )
                events.put(("stopped", run_id))
            except Exception as exc:
                tb = traceback.format_exc()
                try:
                    with err_log.open("a", encoding="utf-8") as f:
                        f.write(
                            f"\n--- {datetime.now().isoformat()} run_id={run_id} ---\n{tb}\n"
                        )
                except OSError:
                    pass
                print(f"[live_dictate ERROR] {tb}", flush=True)
                events.put(("error", (run_id, f"{type(exc).__name__}: {exc}")))

        capture_thread = threading.Thread(
            target=capture_worker, args=(my_id,), daemon=True,
        )
        capture_thread.start()

    def stop_recording() -> None:
        if not is_recording:
            return
        stop_event.set()
        status_var.set("Stopping...")
        progress_var.set("")

    record_btn.configure(command=start_recording)
    stop_btn.configure(command=stop_recording)

    def poll() -> None:
        nonlocal level_display, saved_clips_count
        try:
            while True:
                kind, payload = events.get_nowait()
                if kind == "device":
                    idx, name = payload
                    mic_var.set(f"● RECORDING FROM: [{idx}] {name}")
                elif kind == "status":
                    if payload != "Stopped":
                        status_var.set(payload)
                elif kind == "progress":
                    if not is_recording:
                        continue
                    if payload < 0:
                        progress_var.set("")
                        live_var.set("")
                    else:
                        secs = int(payload * CHUNK_DURATION)
                        progress_var.set(f"● Recording… {secs}s / {CHUNK_DURATION}s")
                elif kind == "level":
                    level_display = payload
                    draw_level(payload)
                    if payload > UI_HEARING_THRESHOLD:
                        live_var.set("🔊 Hearing you…")
                elif kind == "transcribing":
                    live_var.set(f"⏳ Transcribing [{payload}]…")
                elif kind == "saved_clip":
                    name, rate = payload
                    saved_clips_count += 1
                    live_var.set(
                        f"💾 Saved clip #{saved_clips_count} ({rate} Hz) → {name}"
                    )
                elif kind == "chunk":
                    ts, text = payload
                    if text:
                        append_chunk(ts, text)
                        status_var.set("Recording")
                        progress_var.set("Listening…")
                    else:
                        if level_display > UI_HEARING_THRESHOLD:
                            live_var.set("Processing…")
                        else:
                            live_var.set("Very quiet — speak a little louder")
                        progress_var.set("Listening…")
                elif kind == "error":
                    run_id, err = payload
                    if run_id != capture_id:
                        continue
                    status_var.set(f"Error: {err}")
                    mic_var.set("Mic busy? Close other Live Dictate windows, then Record again.")
                    set_recording_ui(False)
                elif kind == "stopped":
                    run_id = payload
                    if run_id != capture_id:
                        continue
                    if is_recording:
                        set_recording_ui(False)
                        if "Error" not in status_var.get():
                            status_var.set("Stopped — click Record to continue")
        except queue.Empty:
            pass
        root.after(100, poll)

    def on_close() -> None:
        stop_event.set()
        status_var.set("Stopping...")
        root.after(400, root.destroy)

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.after(100, poll)
    root.mainloop()


def run_cli(device_name: str | None) -> None:
    stop = threading.Event()

    def on_chunk(_ts: str, text: str) -> None:
        if text:
            print(text, flush=True)
            print(flush=True)

    try:
        run_capture(stop, on_status=print, on_chunk=on_chunk, device_name=device_name)
    except KeyboardInterrupt:
        stop.set()
        print("\nStopped.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live microphone dictation with Whisper (local).",
    )
    parser.add_argument("--cli", action="store_true", help="Print transcript to terminal")
    parser.add_argument(
        "--device",
        metavar="NAME",
        help='Input device substring (default: system default mic). Example: "CABLE Output"',
    )
    parser.add_argument("--list-devices", action="store_true", help="List input devices and exit")
    args = parser.parse_args()

    if args.list_devices:
        list_input_devices()
        return

    if args.cli:
        run_cli(args.device)
    else:
        run_ui(args.device)


if __name__ == "__main__":
    main()
