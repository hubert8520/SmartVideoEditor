"""Shared domain models for transcript, edit decisions and repair planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


Action = Literal["KEEP", "CUT", "REVIEW"]
DetectionSource = Literal["heuristic", "llm", "qa", "repair"]


@dataclass(slots=True)
class TranscriptWord:
    id: int
    timestamp: float
    end: float
    text: str
    normalized: str
    speaker: str | None = None
    confidence: float | None = None


@dataclass(slots=True)
class TranscriptSegment:
    id: int
    timestamp: float
    end: float
    text: str
    word_ids: list[int] = field(default_factory=list)
    speaker: str | None = None


@dataclass(slots=True)
class DropWindow:
    start: float
    end: float
    reason: str
    source_text: str
    word_ids: list[int] = field(default_factory=list)
    source: DetectionSource = "heuristic"
    confidence: float | None = None
    force: bool = False


@dataclass(slots=True)
class ReviewWindow:
    start: float
    end: float
    reason: str
    source_text: str
    word_ids: list[int] = field(default_factory=list)
    source: DetectionSource = "heuristic"
    confidence: float | None = None


@dataclass(slots=True)
class ThoughtBlock:
    id: int
    start: float
    end: float
    word_ids: list[int]
    text: str
    role: str = "thought"
    must_keep: bool = True


@dataclass(slots=True)
class TimelineMapItem:
    id: int
    raw_start: float
    raw_end: float
    final_start: float
    final_end: float

    @property
    def duration_seconds(self) -> float:
        return max(0.0, self.raw_end - self.raw_start)
