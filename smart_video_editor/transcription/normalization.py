"""Transcript normalization helpers shared by segmentation and detectors."""

from __future__ import annotations

from typing import Any

from smart_video_editor.domain.models import TranscriptWord
from smart_video_editor.text import normalize_text
from smart_video_editor.timecode import timestamp_to_seconds


def normalize_word_payload(item: dict[str, Any]) -> TranscriptWord:
    """Convert a raw transcript word object to the shared domain model."""
    text = str(item.get("word", item.get("text", ""))).strip()
    return TranscriptWord(
        id=int(item["id"]),
        timestamp=timestamp_to_seconds(str(item["timestamp"])),
        end=timestamp_to_seconds(str(item["end"])),
        text=text,
        normalized=normalize_text(text),
        speaker=item.get("speaker"),
        confidence=float(item["confidence"]) if item.get("confidence") is not None else None,
    )


def normalize_words_payload(items: list[dict[str, Any]]) -> list[TranscriptWord]:
    return [normalize_word_payload(item) for item in items]
