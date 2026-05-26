"""Local edit candidate detectors."""

from smart_video_editor.detection.local import (
    EditCandidate,
    detect_bad_marker_takes,
    detect_local_candidates,
    detect_partial_repeats,
    detect_repeated_attempts,
    detect_repeated_take_prefixes,
    detect_truncated_word_restarts,
)

__all__ = [
    "EditCandidate",
    "detect_bad_marker_takes",
    "detect_local_candidates",
    "detect_partial_repeats",
    "detect_repeated_attempts",
    "detect_repeated_take_prefixes",
    "detect_truncated_word_restarts",
]
