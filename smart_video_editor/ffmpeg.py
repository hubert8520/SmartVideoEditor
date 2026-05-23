"""FFmpeg discovery and subprocess helpers."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path


def resolve_ffmpeg_executable() -> str:
    """Resolve ffmpeg from PATH or imageio-ffmpeg."""
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg

    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError("Missing ffmpeg. Install dependencies with: pip install -r requirements.txt") from exc

    return imageio_ffmpeg.get_ffmpeg_exe()


def run_ffmpeg(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Run ffmpeg command and raise on failure."""
    return subprocess.run(command, capture_output=True, text=True, check=True)


def probe_duration_seconds(media_path: Path, ffmpeg_executable: str | None = None) -> float:
    """Read media duration using ffmpeg stderr metadata."""
    ffmpeg = ffmpeg_executable or resolve_ffmpeg_executable()
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", str(media_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", result.stderr)
    if not match:
        raise RuntimeError(f"Could not read media duration from ffmpeg output for: {media_path}")

    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
