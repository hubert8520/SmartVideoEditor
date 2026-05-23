"""Timestamp and timecode utilities.

The project uses timestamps formatted as hh:mm:ss:ms, for example:
00:01:23:456
"""

from __future__ import annotations

import re


TIMESTAMP_RE = re.compile(r"^\d{2}:\d{2}:\d{2}:\d{3}$")


def validate_timestamp(value: str, label: str = "timestamp") -> None:
    """Validate a project timestamp."""
    if not TIMESTAMP_RE.fullmatch(value):
        raise ValueError(f"Invalid {label}: {value!r}. Expected hh:mm:ss:ms.")


def timestamp_to_seconds(timestamp: str) -> float:
    """Convert hh:mm:ss:ms to seconds."""
    validate_timestamp(timestamp)
    hours, minutes, seconds, milliseconds = timestamp.strip().split(":")
    return (
        int(hours) * 3600
        + int(minutes) * 60
        + int(seconds)
        + int(milliseconds.ljust(3, "0")[:3]) / 1000
    )


def seconds_to_timestamp(seconds: float) -> str:
    """Convert seconds to hh:mm:ss:ms."""
    total_milliseconds = max(0, int(round(seconds * 1000)))
    total_seconds, milliseconds = divmod(total_milliseconds, 1000)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}:{milliseconds:03d}"


def clamp_interval(start: float, end: float, *, min_value: float = 0.0) -> tuple[float, float]:
    """Normalize a numeric time interval."""
    normalized_start = max(min_value, start)
    normalized_end = max(normalized_start, end)
    return normalized_start, normalized_end
