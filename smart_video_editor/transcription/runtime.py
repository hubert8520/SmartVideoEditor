"""Runtime helpers for scripts/transcribe_video.py.

This module provides shared implementations with signatures compatible with the
existing script. Phase 2 intentionally keeps the script's CLI and output format
unchanged while moving stable utility behavior into the package.
"""

from __future__ import annotations

from pathlib import Path

from smart_video_editor.ffmpeg import probe_duration_seconds, resolve_ffmpeg_executable
from smart_video_editor.paths import (
    ARTIFACTS_DIR,
    PROJECT_ROOT,
    RAW_DIR,
    RAW_MEDIA_EXTENSIONS,
    RAW_TRANSCRIPTION_PATH,
    resolve_single_raw_media,
    supported_raw_media_files as _supported_raw_media_files,
)
from smart_video_editor.timecode import seconds_to_timestamp, timestamp_to_seconds


def supported_raw_media_files() -> list[Path]:
    """Return supported media files from the project raw/ directory."""
    return _supported_raw_media_files(RAW_DIR)


def resolve_raw_media_path(video_name: str | None) -> Path:
    """Resolve a single raw media path using the original transcribe_video.py semantics."""
    return resolve_single_raw_media(video_name, RAW_DIR)


def get_media_duration(ffmpeg_executable: str, media_path: Path) -> float:
    """Compatibility wrapper matching the original get_media_duration signature."""
    return probe_duration_seconds(media_path, ffmpeg_executable)
