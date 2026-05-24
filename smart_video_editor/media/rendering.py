"""Media duration, silence detection, and raw-video rendering."""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path


def fail(message: str) -> None:
    print(f"Error: {message}", file=sys.stderr)
    raise SystemExit(1)


def get_media_duration(ffmpeg: str, media_path: Path) -> float:
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", str(media_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", result.stderr)
    if not match:
        fail("Could not read media duration from ffmpeg output.")

    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def detect_silences(
    ffmpeg: str,
    media_path: Path,
    threshold: str,
    min_duration: float,
) -> list[tuple[float, float]]:
    result = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-i",
            str(media_path),
            "-af",
            f"silencedetect=noise={threshold}:d={min_duration}",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    starts: list[float] = []
    silences: list[tuple[float, float]] = []
    for line in result.stderr.splitlines():
        start_match = re.search(r"silence_start:\s*([0-9.]+)", line)
        if start_match:
            starts.append(float(start_match.group(1)))
            continue

        end_match = re.search(r"silence_end:\s*([0-9.]+)", line)
        if end_match and starts:
            silences.append((starts.pop(0), float(end_match.group(1))))

    return silences


def speech_intervals_from_silences(
    silences: list[tuple[float, float]],
    duration: float,
) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    cursor = 0.0

    for start, end in sorted(silences):
        start = max(0.0, min(start, duration))
        end = max(start, min(end, duration))
        if start > cursor:
            intervals.append((cursor, start))
        cursor = max(cursor, end)

    if cursor < duration:
        intervals.append((cursor, duration))

    return intervals


def render_video(
    ffmpeg: str,
    media_path: Path,
    keep_intervals: list[tuple[float, float]],
    output_path: Path,
) -> None:
    if not keep_intervals:
        fail("No intervals left to render after analysis.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="edited_video_") as temp_name:
        temp_dir = Path(temp_name)
        segment_paths: list[Path] = []

        for index, (start, end) in enumerate(keep_intervals, start=1):
            segment_path = temp_dir / f"segment_{index:04d}.mp4"
            duration = end - start
            command = [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{start:.3f}",
                "-i",
                str(media_path),
                "-t",
                f"{duration:.3f}",
                "-map",
                "0:v:0?",
                "-map",
                "0:a:0?",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                "-c:a",
                "aac",
                "-b:a",
                "160k",
                "-movflags",
                "+faststart",
                str(segment_path),
            ]
            subprocess.run(command, check=True)
            segment_paths.append(segment_path)

        concat_list = temp_dir / "concat.txt"
        concat_list.write_text(
            "".join(f"file '{path.as_posix()}'\n" for path in segment_paths),
            encoding="utf-8",
        )
        subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_list),
                "-c",
                "copy",
                str(output_path),
            ],
            check=True,
        )
