"""Word, phrase, thought, and take segmentation helpers."""

from smart_video_editor.segmentation.attempts import (
    AttemptSpan,
    RepeatedAttemptGroup,
    find_repeated_attempt_groups,
)
from smart_video_editor.segmentation.takes import TakeSegment, segment_take_indices, segment_takes
from smart_video_editor.segmentation.words import PhraseSegment, segment_phrases

__all__ = [
    "AttemptSpan",
    "PhraseSegment",
    "RepeatedAttemptGroup",
    "TakeSegment",
    "find_repeated_attempt_groups",
    "segment_phrases",
    "segment_take_indices",
    "segment_takes",
]
