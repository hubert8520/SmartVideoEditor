#!/usr/bin/env python3
"""Check local SmartVideoEditor setup without calling external APIs."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(
    os.environ.get("SMART_VIDEO_EDITOR_WORKSPACE", Path(__file__).resolve().parents[1])
).expanduser().resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from smart_video_editor.env import load_env_file, looks_like_placeholder
from smart_video_editor.ffmpeg import resolve_ffmpeg_executable
from smart_video_editor.paths import RAW_DIR, supported_raw_media_files


def status_line(ok: bool, message: str) -> None:
    marker = "OK" if ok else "WARN"
    print(f"[{marker}] {message}")


def main() -> None:
    loaded_env = load_env_file(None)
    status_line(loaded_env is not None, f".env loaded: {loaded_env}" if loaded_env else ".env not found")

    try:
        ffmpeg = resolve_ffmpeg_executable()
        status_line(True, f"ffmpeg available: {ffmpeg}")
    except Exception as exc:
        status_line(False, f"ffmpeg unavailable: {exc}")

    raw_files = supported_raw_media_files(RAW_DIR)
    status_line(bool(raw_files), f"raw media files: {len(raw_files)}")
    for path in raw_files:
        print(f"  - {path.name}")

    for key in ("OPENAI_API_KEY", "DEEPGRAM_API_KEY"):
        value = os.getenv(key)
        status_line(not looks_like_placeholder(value), f"{key}: {'set' if value else 'missing'}")


if __name__ == "__main__":
    main()
