"""Boundary checks that run before a planner approves cuts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from smart_video_editor.domain.models import TranscriptWord
from smart_video_editor.text import normalize_text


BoundaryAction = Literal["BLOCK", "REVIEW"]
BRIDGE_TOKENS = {"bo", "ale", "czyli", "wiec", "więc", "zeby", "żeby", "jesli", "jeśli"}
CONTINUATION_TOKENS = {"w", "z", "do", "dla", "ktory", "który", "ktora", "która", "ktore", "które"}


@dataclass(frozen=True, slots=True)
class BoundaryIssue:
    kind: str
    action: BoundaryAction
    reason: str
    start: float
    end: float
    left_word_id: int | None = None
    right_word_id: int | None = None
    removed_word_ids: tuple[int, ...] = ()


def _token(word: TranscriptWord) -> str:
    return normalize_text(word.text)


def _overlaps(boundary: float, word: TranscriptWord) -> bool:
    return word.timestamp < boundary < word.end


def _kept_words(words: list[TranscriptWord], drop_windows: list[tuple[float, float]]) -> list[TranscriptWord]:
    kept: list[TranscriptWord] = []
    for word in words:
        if any(word.end > start and word.timestamp < end for start, end in drop_windows):
            continue
        kept.append(word)
    return kept


def _removed_words_between(
    words: list[TranscriptWord],
    left_word_id: int,
    right_word_id: int,
) -> list[TranscriptWord]:
    return [word for word in words if left_word_id < word.id < right_word_id]


def validate_cut_boundaries(
    words: list[TranscriptWord],
    drop_windows: list[tuple[float, float]],
) -> list[BoundaryIssue]:
    """Return BLOCK/REVIEW issues for unsafe word or join boundaries."""
    issues: list[BoundaryIssue] = []

    for start, end in drop_windows:
        for boundary_name, boundary in (("start", start), ("end", end)):
            for word in words:
                if not _overlaps(boundary, word):
                    continue
                issues.append(
                    BoundaryIssue(
                        kind="mid_word_cut",
                        action="BLOCK",
                        reason=f"{boundary_name}_boundary_inside_word",
                        start=boundary,
                        end=boundary,
                        left_word_id=word.id,
                        right_word_id=word.id,
                    )
                )

    kept = _kept_words(words, drop_windows)
    for left, right in zip(kept, kept[1:]):
        if right.id <= left.id + 1:
            continue

        removed = _removed_words_between(words, left.id, right.id)
        removed_tokens = {_token(word) for word in removed}
        left_token = _token(left)
        right_token = _token(right)

        if left_token in BRIDGE_TOKENS:
            issues.append(
                BoundaryIssue(
                    kind="unnatural_join",
                    action="REVIEW",
                    reason="join_after_bridge_word",
                    start=left.end,
                    end=right.timestamp,
                    left_word_id=left.id,
                    right_word_id=right.id,
                    removed_word_ids=tuple(word.id for word in removed),
                )
            )
        if removed_tokens & BRIDGE_TOKENS:
            issues.append(
                BoundaryIssue(
                    kind="removed_bridge",
                    action="REVIEW",
                    reason="cut_removed_logical_connector",
                    start=left.end,
                    end=right.timestamp,
                    left_word_id=left.id,
                    right_word_id=right.id,
                    removed_word_ids=tuple(word.id for word in removed if _token(word) in BRIDGE_TOKENS),
                )
            )
        if right_token in CONTINUATION_TOKENS:
            issues.append(
                BoundaryIssue(
                    kind="unnatural_join",
                    action="REVIEW",
                    reason="join_before_continuation_word",
                    start=left.end,
                    end=right.timestamp,
                    left_word_id=left.id,
                    right_word_id=right.id,
                    removed_word_ids=tuple(word.id for word in removed),
                )
            )

    return issues


def trim_short_block_tail_to_silence(
    last_word: TranscriptWord,
    original_end: float,
    silences: list[tuple[float, float]],
    *,
    min_spoken_before_trim: float = 0.25,
    silence_window: float = 0.45,
    tail_padding: float = 0.02,
) -> float:
    """Trim a short heading tail to nearby silence without touching the word."""
    earliest_silence = last_word.timestamp + min_spoken_before_trim
    latest_silence = last_word.end + silence_window
    for silence_start, _silence_end in silences:
        if silence_start < earliest_silence:
            continue
        if silence_start > latest_silence:
            continue
        return min(original_end, max(last_word.end, silence_start + tail_padding))
    return original_end
