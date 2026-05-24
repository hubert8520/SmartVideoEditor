"""Small word-level segmentation primitives."""

from __future__ import annotations

from dataclasses import dataclass

from smart_video_editor.domain.models import TranscriptWord


@dataclass(frozen=True, slots=True)
class PhraseSegment:
    id: int
    start_word_id: int
    end_word_id: int
    start: float
    end: float
    text: str


def segment_phrases(words: list[TranscriptWord], *, max_gap: float = 0.65) -> list[PhraseSegment]:
    """Split words into simple phrase-like groups by timing gaps."""
    if not words:
        return []

    groups: list[list[TranscriptWord]] = []
    current: list[TranscriptWord] = [words[0]]
    for word in words[1:]:
        previous = current[-1]
        if word.timestamp - previous.end > max_gap:
            groups.append(current)
            current = []
        current.append(word)
    groups.append(current)

    return [
        PhraseSegment(
            id=index,
            start_word_id=group[0].id,
            end_word_id=group[-1].id,
            start=group[0].timestamp,
            end=group[-1].end,
            text=" ".join(word.text for word in group),
        )
        for index, group in enumerate(groups)
    ]
