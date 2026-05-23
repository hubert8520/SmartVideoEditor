#!/usr/bin/env python3
'''Apply Phase 2 refactor to scripts/transcribe_video.py.

This is intentionally conservative:
- it does not change the CLI;
- it does not change Deepgram/OpenAI provider behavior;
- it does not change artifacts/raw_transcription.json;
- it creates a backup before editing;
- it only redirects stable utility functions to smart_video_editor.transcription.runtime.
'''

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRANSCRIBE_PATH = PROJECT_ROOT / "scripts" / "transcribe_video.py"
BACKUP_PATH = PROJECT_ROOT / "scripts" / "transcribe_video.py.phase2.bak"

PATCH_MARKER = "# Phase 2 shared runtime helpers"
PATCH_BLOCK = '''


# Phase 2 shared runtime helpers
#
# Keep this block close to main() while the file is being migrated. Functions
# defined above still exist for now, but the names below are rebound to shared
# package implementations before the CLI runs. This keeps behavior stable while
# reducing duplication step by step.
from smart_video_editor.transcription.runtime import (  # noqa: E402
    ARTIFACTS_DIR as ARTIFACTS_DIR,
    PROJECT_ROOT as PROJECT_ROOT,
    RAW_DIR as RAW_DIR,
    RAW_MEDIA_EXTENSIONS as RAW_MEDIA_EXTENSIONS,
    RAW_TRANSCRIPTION_PATH as RAW_TRANSCRIPTION_PATH,
    get_media_duration as get_media_duration,
    resolve_ffmpeg_executable as resolve_ffmpeg_executable,
    resolve_raw_media_path as resolve_raw_media_path,
    seconds_to_timestamp as seconds_to_timestamp,
    supported_raw_media_files as supported_raw_media_files,
    timestamp_to_seconds as timestamp_to_seconds,
)
'''


def fail(message: str) -> None:
    raise SystemExit(f"Error: {message}")


def main() -> None:
    if not TRANSCRIBE_PATH.exists():
        fail(f"Missing file: {TRANSCRIBE_PATH}")

    source = TRANSCRIBE_PATH.read_text(encoding="utf-8")

    if PATCH_MARKER in source:
        print("Phase 2 patch already present in scripts/transcribe_video.py")
        return

    insertion_point = source.find("\ndef main() -> None:")
    if insertion_point == -1:
        fail("Could not find 'def main() -> None:' in scripts/transcribe_video.py")

    if not BACKUP_PATH.exists():
        BACKUP_PATH.write_text(source, encoding="utf-8")
        print(f"Backup written: {BACKUP_PATH}")
    else:
        print(f"Backup already exists: {BACKUP_PATH}")

    patched = source[:insertion_point] + PATCH_BLOCK + source[insertion_point:]
    TRANSCRIBE_PATH.write_text(patched, encoding="utf-8")
    print("Patched scripts/transcribe_video.py")

    subprocess.run(
        [sys.executable, "-m", "py_compile", str(TRANSCRIBE_PATH)],
        cwd=PROJECT_ROOT,
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "py_compile", str(PROJECT_ROOT / "smart_video_editor" / "transcription" / "runtime.py")],
        cwd=PROJECT_ROOT,
        check=True,
    )
    print("Python compile check passed.")


if __name__ == "__main__":
    main()
