#!/usr/bin/env python3

"""

Live transcription from Windows WASAPI loopback (default speakers).



Captures whatever is playing on the PC — same idea as ChatGPT browser Record

with "Share system audio" — without routing playback through VB-Cable.



On stop (Ctrl+C or window close), saves to a dated meeting folder and runs

Ollama roundup via meeting_roundup.py (reuses cable_transcribe finalize).

"""



from __future__ import annotations



import argparse

import atexit

import os

import queue

import signal

import sys

import tempfile

import threading

import time

import wave

from datetime import datetime

from pathlib import Path



import numpy as np

import pyaudiowpatch as pyaudio

import whisper

from scipy import signal



from cable_transcribe import LOG_FILE, start_new_meeting, session_path

from meeting_roundup import finalize_meeting_session



APP_ROOT = Path(__file__).resolve().parent

DEFAULT_LOG = APP_ROOT / "loopback_live_transcript.txt"



CHUNK_DURATION = 20

WHISPER_SAMPLE_RATE = 16000

WHISPER_MODEL = "base"

SILENCE_RMS_THRESHOLD = 0.01

BLOCK_SIZE = 4096





def find_default_loopback_device(p: pyaudio.PyAudio) -> dict:

    """Return WASAPI loopback device matching the default output speakers."""

    try:

        wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)

    except OSError as exc:

        raise RuntimeError("WASAPI is not available on this system") from exc



    default_speakers = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])

    if default_speakers.get("isLoopbackDevice"):

        return default_speakers



    for loopback in p.get_loopback_device_info_generator():

        if default_speakers["name"] in loopback["name"]:

            return loopback



    raise RuntimeError(

        "No WASAPI loopback device found for default output "

        f'"{default_speakers["name"]}". Run: python -m pyaudiowpatch'

    )





def find_loopback_by_name(p: pyaudio.PyAudio, name_substring: str) -> dict:

    needle = name_substring.lower()

    matches = [

        dev

        for dev in p.get_loopback_device_info_generator()

        if needle in dev["name"].lower()

    ]

    if matches:

        return matches[0]

    raise RuntimeError(

        f'No loopback device matching "{name_substring}". '

        "Run: python -m pyaudiowpatch"

    )





def bytes_to_mono_float32(raw: bytes, channels: int) -> np.ndarray:

    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    if channels > 1:

        samples = samples.reshape(-1, channels).mean(axis=1)

    return np.clip(samples, -1.0, 1.0)





def resample_to_whisper(audio: np.ndarray, source_rate: int) -> np.ndarray:

    if source_rate == WHISPER_SAMPLE_RATE:

        return audio

    n_out = int(len(audio) * WHISPER_SAMPLE_RATE / source_rate)

    if n_out < 1:

        return np.array([], dtype=np.float32)

    return signal.resample(audio, n_out).astype(np.float32)





def chunk_rms(audio: np.ndarray) -> float:

    if audio.size == 0:

        return 0.0

    return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))





def save_wav(path: str, audio: np.ndarray) -> None:

    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)

    with wave.open(path, "wb") as wf:

        wf.setnchannels(1)

        wf.setsampwidth(2)

        wf.setframerate(WHISPER_SAMPLE_RATE)

        wf.writeframes(pcm.tobytes())





def format_timestamp(seconds: float) -> str:

    s = int(seconds)

    return f"{s // 60:02d}:{s % 60:02d}"





def append_log(log_path: Path, line: str, mirror: Path | None = None) -> None:

    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("a", encoding="utf-8") as f:

        f.write(line + "\n")

    if mirror is not None and mirror != log_path:

        mirror.parent.mkdir(parents=True, exist_ok=True)

        with mirror.open("a", encoding="utf-8") as f:

            f.write(line + "\n")





def run_loopback_transcribe(

    log_path: Path,

    *,

    device_name: str | None = None,

    chunk_duration: int = CHUNK_DURATION,

    whisper_model_name: str = WHISPER_MODEL,

    meeting_name: str | None = None,

    use_meeting_folder: bool = True,

    finalize: bool = True,

    mirror_log: Path | None = DEFAULT_LOG,

) -> None:

    required_samples = chunk_duration * WHISPER_SAMPLE_RATE

    temp_dir = tempfile.mkdtemp(prefix="loopback_transcribe_")



    session = None

    if use_meeting_folder:

        session = start_new_meeting(meeting_name or "")

        log_path = session_path(LOG_FILE)

        print(f"Meeting folder: {session.folder}", flush=True)



    shutdown_done = False



    def shutdown_roundup() -> None:

        nonlocal shutdown_done

        if shutdown_done:

            return

        shutdown_done = True

        if session is None or not finalize:

            return

        ended = datetime.now().isoformat(timespec="seconds")

        append_log(

            log_path,

            f"\n=== Loopback transcribe ended {ended} ===\n",

            mirror=mirror_log,

        )

        print("\n--- MEETING END — generating roundup ---", flush=True)

        try:

            roundup = finalize_meeting_session(session)

            if roundup:

                print(f"Roundup: {roundup}", flush=True)

            print(f"Transcript: {log_path}", flush=True)

            print(f"Folder: {session.folder}", flush=True)

        except Exception as exc:

            print(f"Roundup failed: {exc}", file=sys.stderr, flush=True)



    atexit.register(shutdown_roundup)



    def _signal_handler(_signum, _frame) -> None:

        raise KeyboardInterrupt



    signal.signal(signal.SIGINT, _signal_handler)

    if hasattr(signal, "SIGTERM"):

        signal.signal(signal.SIGTERM, _signal_handler)

    if hasattr(signal, "SIGBREAK"):

        signal.signal(signal.SIGBREAK, _signal_handler)



    header = (

        f"\n=== Loopback transcribe started {datetime.now().isoformat(timespec='seconds')} ===\n"

    )

    if session:

        header = (

            f"\n=== {session.name} — loopback started "

            f"{datetime.now().isoformat(timespec='seconds')} ===\n"

        )

    append_log(log_path, header, mirror=mirror_log)

    print(header.strip(), flush=True)



    try:

        with pyaudio.PyAudio() as pa:

            device = (

                find_loopback_by_name(pa, device_name)

                if device_name

                else find_default_loopback_device(pa)

            )

            capture_rate = int(device["defaultSampleRate"])

            channels = int(device["maxInputChannels"]) or 2

            device_label = f"({device['index']}) {device['name']}"



            print(f"Loading Whisper ({whisper_model_name})...", flush=True)

            model = whisper.load_model(whisper_model_name)

            print(

                f"Listening via WASAPI loopback on {device_label} "

                f"({chunk_duration}s chunks @ {WHISPER_SAMPLE_RATE} Hz for Whisper)",

                flush=True,

            )

            print(f"Writing transcript to: {log_path}", flush=True)

            if mirror_log and mirror_log != log_path:

                print(f"Mirror log: {mirror_log}", flush=True)

            print("Press Ctrl+C to stop — roundup runs automatically.\n", flush=True)



            audio_q: queue.Queue[bytes] = queue.Queue()

            stop_event = threading.Event()



            def callback(in_data, _frame_count, _time_info, status) -> tuple[bytes, int]:

                if status:

                    print(status, flush=True)

                if not stop_event.is_set():

                    audio_q.put(in_data)

                return in_data, pyaudio.paContinue



            buffer_parts: list[np.ndarray] = []

            total_samples = 0

            chunk_index = 0



            with pa.open(

                format=pyaudio.paInt16,

                channels=channels,

                rate=capture_rate,

                frames_per_buffer=BLOCK_SIZE,

                input=True,

                input_device_index=int(device["index"]),

                stream_callback=callback,

            ):

                while not stop_event.is_set():

                    try:

                        raw = audio_q.get(timeout=0.25)

                    except queue.Empty:

                        continue



                    block = resample_to_whisper(

                        bytes_to_mono_float32(raw, channels),

                        capture_rate,

                    )

                    buffer_parts.append(block)

                    total_samples += len(block)



                    if total_samples < required_samples:

                        continue



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



                    chunk_audio = np.concatenate(parts)

                    total_samples = sum(len(p) for p in buffer_parts)

                    ts = format_timestamp(chunk_index * chunk_duration)



                    if chunk_rms(chunk_audio) < SILENCE_RMS_THRESHOLD:

                        line = f"[{ts}] [silence]"

                        print(line, flush=True)

                        append_log(log_path, line, mirror=mirror_log)

                    else:

                        wav_path = os.path.join(temp_dir, f"chunk_{chunk_index:05d}.wav")

                        save_wav(wav_path, chunk_audio)

                        print(f"Transcribing [{ts}]...", flush=True)

                        result = model.transcribe(wav_path, fp16=False, language="en")

                        text = result.get("text", "").strip() or "[no speech detected]"

                        line = f"[{ts}] {text}"

                        print(line, flush=True)

                        append_log(log_path, line, mirror=mirror_log)

                        try:

                            os.remove(wav_path)

                        except OSError:

                            pass



                    chunk_index += 1

    except KeyboardInterrupt:

        print("\nStopping...", flush=True)

    finally:

        shutdown_roundup()





def main() -> None:

    parser = argparse.ArgumentParser(

        description="Transcribe PC playback via WASAPI loopback; auto-roundup on stop.",

    )

    parser.add_argument(

        "--log",

        type=Path,

        default=DEFAULT_LOG,

        help=f"Legacy mirror log (default: {DEFAULT_LOG}); primary log is in meetings/",

    )

    parser.add_argument(

        "--meeting-name",

        help="Meeting name for folder under meetings/ (default: auto timestamp)",

    )

    parser.add_argument(

        "--no-meeting-folder",

        action="store_true",

        help="Write only to --log (no dated folder, no roundup)",

    )

    parser.add_argument(

        "--no-finalize",

        action="store_true",

        help="Skip Ollama roundup on stop",

    )

    parser.add_argument(

        "--no-mirror",

        action="store_true",

        help="Do not mirror to loopback_live_transcript.txt",

    )

    parser.add_argument(

        "--device",

        help='Loopback device name substring (default: match default speakers). '

        'Example: "Speakers (Realtek"',

    )

    parser.add_argument(

        "--chunk",

        type=int,

        default=CHUNK_DURATION,

        help=f"Seconds per Whisper chunk (default: {CHUNK_DURATION})",

    )

    parser.add_argument(

        "--model",

        default=WHISPER_MODEL,

        help=f"Whisper model name (default: {WHISPER_MODEL})",

    )

    parser.add_argument(

        "--list-devices",

        action="store_true",

        help="List WASAPI loopback devices and exit",

    )

    args = parser.parse_args()



    if args.list_devices:

        with pyaudio.PyAudio() as pa:

            print("WASAPI loopback devices:")

            for dev in pa.get_loopback_device_info_generator():

                print(f"  [{dev['index']}] {dev['name']}")

        return



    mirror = None if args.no_mirror or args.no_meeting_folder else args.log.resolve()

    log_path = args.log.resolve()



    run_loopback_transcribe(

        log_path,

        device_name=args.device,

        chunk_duration=args.chunk,

        whisper_model_name=args.model,

        meeting_name=args.meeting_name,

        use_meeting_folder=not args.no_meeting_folder,

        finalize=not args.no_finalize,

        mirror_log=mirror,

    )





if __name__ == "__main__":

    main()

