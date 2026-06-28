#!/usr/bin/env python3
"""Quick smoke test for live_dictate (no GUI). Run: python smoke_test_dictate.py"""

from __future__ import annotations

import sys
import threading
import time

import numpy as np
import sounddevice as sd

from live_dictate import (
    CHUNK_DURATION,
    REQUIRED_SAMPLES,
    SAMPLE_RATE,
    default_input_index,
    find_input_device,
    run_capture,
    to_mono_float32,
)


def test_default_device() -> None:
    idx = default_input_index()
    dev = sd.query_devices(idx)
    assert dev["max_input_channels"] >= 1, dev
    print(f"OK  default input: [{idx}] {dev['name']}")


def test_find_by_name() -> None:
    idx, name = find_input_device()
    assert isinstance(idx, int) and idx >= 0
    print(f"OK  find_input_device: [{idx}] {name}")


def test_mic_stream() -> None:
    idx = default_input_index()
    secs = 0.5
    frames = int(SAMPLE_RATE * secs)
    audio = sd.rec(frames, samplerate=SAMPLE_RATE, channels=1, device=idx, dtype="float32")
    sd.wait()
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    print(f"OK  mic capture {secs}s RMS={rms:.4f} (any level means stream works)")


def test_capture_pipeline_short() -> None:
    """Run capture for one chunk duration; accept silence or text."""
    stop = threading.Event()
    chunks: list[tuple[str, str]] = []
    statuses: list[str] = []

    def on_chunk(ts: str, text: str) -> None:
        chunks.append((ts, text))
        print(f"OK  chunk [{ts}]: {text!r}" if text else f"OK  chunk [{ts}]: (silence)")

    t = threading.Thread(
        target=run_capture,
        args=(stop,),
        kwargs={
            "on_status": lambda m: statuses.append(m),
            "on_chunk": on_chunk,
        },
        daemon=True,
    )
    t.start()
    # Whisper load + one full chunk + transcribe margin
    wait = 90
    deadline = time.time() + wait
    while time.time() < deadline and not chunks:
        time.sleep(0.5)
        if statuses:
            last = statuses[-1]
            if "Listening on" in last:
                print(f"    {last}")
    stop.set()
    t.join(timeout=15)

    if not any("Listening on" in s for s in statuses):
        raise AssertionError(f"Never reached listening state. Statuses: {statuses[-5:]}")
    if not chunks:
        raise AssertionError(
            f"No chunk completed in {wait}s. Last statuses: {statuses[-8:]}"
        )
    print(f"OK  capture pipeline ({len(chunks)} chunk(s))")


def test_long_run_30s() -> None:
    """Record for ~30s; require multiple chunks, no crash, capture loop survives."""
    record_secs = 30
    expected_min_chunks = max(2, (record_secs // 3) - 2)
    stop = threading.Event()
    chunks: list[tuple[str, str]] = []
    statuses: list[str] = []
    errors: list[str] = []
    progress_seen = 0
    crashed = {"yes": False, "exc": None}

    def on_chunk(ts: str, text: str) -> None:
        chunks.append((ts, text))
        kind = "text" if text else "silence"
        print(f"    chunk #{len(chunks)} [{ts}] {kind}: {text[:60]!r}" if text else f"    chunk #{len(chunks)} [{ts}] silence")

    def on_progress(p: float) -> None:
        nonlocal progress_seen
        progress_seen += 1

    def worker() -> None:
        try:
            run_capture(
                stop,
                on_status=lambda m: statuses.append(m),
                on_chunk=on_chunk,
                on_progress=on_progress,
            )
        except Exception as exc:
            crashed["yes"] = True
            crashed["exc"] = exc
            errors.append(str(exc))

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    # Wait for Whisper load + listening state
    listen_deadline = time.time() + 60
    while time.time() < listen_deadline:
        if any("Listening on" in s for s in statuses):
            break
        if crashed["yes"]:
            raise AssertionError(f"Worker crashed before listening: {crashed['exc']}")
        time.sleep(0.3)
    else:
        raise AssertionError(
            f"Never started listening. Last statuses: {statuses[-5:]}"
        )

    print(f"    listening; recording for {record_secs}s…")
    record_deadline = time.time() + record_secs
    while time.time() < record_deadline:
        if crashed["yes"]:
            raise AssertionError(
                f"Worker crashed mid-recording after {len(chunks)} chunk(s): "
                f"{crashed['exc']}"
            )
        time.sleep(0.5)

    stop.set()
    t.join(timeout=15)

    if crashed["yes"]:
        raise AssertionError(f"Worker crashed: {crashed['exc']}")
    if t.is_alive():
        raise AssertionError("Capture thread did not exit after stop")
    if len(chunks) < expected_min_chunks:
        raise AssertionError(
            f"Expected >= {expected_min_chunks} chunks in {record_secs}s, got {len(chunks)}. "
            f"Last statuses: {statuses[-5:]}"
        )
    if progress_seen < 5:
        raise AssertionError(
            f"Progress callback fired only {progress_seen} times — capture loop stalled?"
        )
    print(
        f"OK  long run: {len(chunks)} chunks over {record_secs}s "
        f"({progress_seen} progress ticks), clean shutdown"
    )


def main() -> int:
    print("Live Dictate smoke test\n")
    tests = [
        ("default device", test_default_device),
        ("find input device", test_find_by_name),
        ("mic stream", test_mic_stream),
        ("capture + whisper (one chunk)", test_capture_pipeline_short),
        ("long run (30s, multiple chunks)", test_long_run_30s),
    ]
    failed = 0
    for label, fn in tests:
        print(f"\n--- {label} ---")
        try:
            fn()
        except Exception as exc:
            failed += 1
            print(f"FAIL {label}: {exc}", file=sys.stderr)
    print()
    if failed:
        print(f"FAILED ({failed}/{len(tests)} tests)")
        return 1
    print(f"All {len(tests)} tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
