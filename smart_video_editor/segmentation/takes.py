"""Take-level segmentation helpers."""

from __future__ import annotations

from dataclasses import dataclass

from smart_video_editor.domain.models import TranscriptWord


@dataclass(frozen=True, slots=True)
class TakeSegment:
    id: int
    start_index: int
    end_index: int
    start_word_id: int
    end_word_id: int
    start: float
    end: float
    text: str
    word_ids: tuple[int, ...]


def segment_take_indices(
    words: list[TranscriptWord],
    *,
    max_gap: float = 1.2,
) -> list[tuple[int, int]]:
    """Return inclusive word-index spans split by long pauses."""
    if not words:
        return []

    spans: list[tuple[int, int]] = []
    start_index = 0
    for index in range(1, len(words)):
        previous = words[index - 1]
        current = words[index]
        if current.timestamp - previous.end > max_gap:
            spans.append((start_index, index - 1))
            start_index = index
    spans.append((start_index, len(words) - 1))
    return spans


def segment_takes(
    words: list[TranscriptWord],
    *,
    max_gap: float = 1.2,
) -> list[TakeSegment]:
    """Group continuous speech into take-like spans."""
    segments: list[TakeSegment] = []
    for segment_id, (start_index, end_index) in enumerate(segment_take_indices(words, max_gap=max_gap)):
        group = words[start_index : end_index + 1]
        segments.append(
            TakeSegment(
                id=segment_id,
                start_index=start_index,
                end_index=end_index,
                start_word_id=group[0].id,
                end_word_id=group[-1].id,
                start=group[0].timestamp,
                end=group[-1].end,
                text=" ".join(word.text for word in group).strip(),
                word_ids=tuple(word.id for word in group),
            )
        )
    return segments

