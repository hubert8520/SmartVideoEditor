"""Conservative local detectors for edit candidates.

Detectors do not cut media. They produce candidate ranges that a planner can
accept, downgrade to REVIEW, or reject after boundary validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from smart_video_editor.domain.models import TranscriptWord
from smart_video_editor.text import normalize_text


CandidateAction = Literal["DROP", "REVIEW"]

BAD_MARKER_PHRASES: tuple[tuple[str, ...], ...] = (
    ("kurwa",),
    ("jeszcze", "raz"),
    ("od", "poczatku"),
    ("od", "nowa"),
    ("nie", "tak"),
    ("zle",),
    ("stop",),
    ("pomylilem", "sie"),
)
COMMON_STARTERS = {"a", "ale", "bo", "czyli", "i", "jak", "no", "to", "wiec", "więc", "zeby", "żeby"}


@dataclass(frozen=True, slots=True)
class EditCandidate:
    category: str
    start_word_id: int
    end_word_id: int
    start: float
    end: float
    text: str
    reason: str
    confidence: float
    recommended_action: CandidateAction = "REVIEW"
    source: str = "local"


def _token(word: TranscriptWord) -> str:
    return normalize_text(word.text)


def _candidate_text(words: list[TranscriptWord]) -> str:
    return " ".join(word.text for word in words).strip()


def _make_candidate(
    words: list[TranscriptWord],
    *,
    category: str,
    reason: str,
    confidence: float,
    recommended_action: CandidateAction,
) -> EditCandidate:
    return EditCandidate(
        category=category,
        start_word_id=words[0].id,
        end_word_id=words[-1].id,
        start=words[0].timestamp,
        end=words[-1].end,
        text=_candidate_text(words),
        reason=reason,
        confidence=confidence,
        recommended_action=recommended_action,
    )


def _is_prefix_token(fragment: str, full: str) -> bool:
    return len(fragment) >= 3 and fragment != full and full.startswith(fragment)


def _dedupe(candidates: list[EditCandidate]) -> list[EditCandidate]:
    seen: set[tuple[str, int, int]] = set()
    result: list[EditCandidate] = []
    for candidate in candidates:
        key = (candidate.category, candidate.start_word_id, candidate.end_word_id)
        if key in seen:
            continue
        seen.add(key)
        result.append(candidate)
    return result


def detect_partial_repeats(
    words: list[TranscriptWord],
    *,
    max_restart_words: int = 5,
    max_restart_gap: float = 4.0,
) -> list[EditCandidate]:
    """Find abandoned prefixes such as ``jak skonfi... jak skonfigurowac``."""
    tokens = [_token(word) for word in words]
    candidates: list[EditCandidate] = []

    for start_index in range(len(words)):
        starter = tokens[start_index]
        if not starter:
            continue

        max_next = min(len(words), start_index + max_restart_words + 2)
        for repeat_index in range(start_index + 2, max_next):
            if words[repeat_index].timestamp - words[start_index].timestamp > max_restart_gap:
                break
            if tokens[repeat_index] != starter:
                continue

            first_attempt = tokens[start_index:repeat_index]
            second_attempt = tokens[repeat_index : repeat_index + len(first_attempt)]
            if len(first_attempt) < 2 or len(second_attempt) < 2:
                continue

            has_truncated_word = any(
                _is_prefix_token(left, right)
                for left, right in zip(first_attempt[1:], second_attempt[1:])
            )
            if not has_truncated_word:
                continue

            candidate_words = words[start_index:repeat_index]
            candidates.append(
                _make_candidate(
                    candidate_words,
                    category="partial_repeat",
                    reason="abandoned_partial_phrase_before_complete_restart",
                    confidence=0.92,
                    recommended_action="DROP",
                )
            )
            break

    return _dedupe(candidates)


def detect_repeated_take_prefixes(
    words: list[TranscriptWord],
    *,
    min_phrase_words: int = 2,
    max_phrase_words: int = 8,
    max_restart_gap: float = 8.0,
) -> list[EditCandidate]:
    """Find earlier short takes that are prefixes of a later fuller take."""
    tokens = [_token(word) for word in words]
    candidates: list[EditCandidate] = []

    for start_index in range(len(words)):
        for phrase_len in range(max_phrase_words, min_phrase_words - 1, -1):
            end_index = start_index + phrase_len
            if end_index >= len(words):
                continue
            phrase = tokens[start_index:end_index]
            if not all(phrase):
                continue
            if len(phrase) == 1 and phrase[0] in COMMON_STARTERS:
                continue

            search_end = min(len(words), end_index + max_phrase_words + 4)
            for repeat_index in range(end_index, search_end):
                if repeat_index + phrase_len > len(words):
                    continue
                if words[repeat_index].timestamp - words[start_index].timestamp > max_restart_gap:
                    break
                if tokens[repeat_index : repeat_index + phrase_len] != phrase:
                    continue
                has_fuller_continuation = repeat_index + phrase_len < len(words)
                if not has_fuller_continuation:
                    continue

                candidate_words = words[start_index:end_index]
                candidates.append(
                    _make_candidate(
                        candidate_words,
                        category="repeated_take",
                        reason="earlier_take_is_prefix_of_later_fuller_take",
                        confidence=0.88,
                        recommended_action="DROP",
                    )
                )
                break
            else:
                continue
            break

    return _dedupe(candidates)


def _phrase_at(tokens: list[str], index: int, phrase: tuple[str, ...]) -> bool:
    return tuple(tokens[index : index + len(phrase)]) == phrase


def detect_bad_marker_takes(
    words: list[TranscriptWord],
    *,
    max_take_words_before_marker: int = 16,
    max_take_gap: float = 1.2,
) -> list[EditCandidate]:
    """Expand bad markers to the failed take that precedes them."""
    tokens = [_token(word) for word in words]
    candidates: list[EditCandidate] = []

    for index in range(len(words)):
        matched_phrase: tuple[str, ...] | None = None
        for phrase in BAD_MARKER_PHRASES:
            if _phrase_at(tokens, index, phrase):
                matched_phrase = phrase
                break
        if matched_phrase is None:
            continue

        marker_end_index = index + len(matched_phrase) - 1
        start_index = index
        floor = max(0, index - max_take_words_before_marker)
        for cursor in range(index - 1, floor - 1, -1):
            gap = words[cursor + 1].timestamp - words[cursor].end
            if gap > max_take_gap:
                break
            start_index = cursor

        candidate_words = words[start_index : marker_end_index + 1]
        has_failed_take_context = start_index < index
        candidates.append(
            _make_candidate(
                candidate_words,
                category="bad_marker_take",
                reason="bad_marker_expanded_to_failed_take",
                confidence=0.9 if has_failed_take_context else 0.72,
                recommended_action="DROP" if has_failed_take_context else "REVIEW",
            )
        )

    return _dedupe(candidates)


def detect_local_candidates(words: list[TranscriptWord]) -> list[EditCandidate]:
    """Run all deterministic local detectors."""
    return _dedupe(
        [
            *detect_partial_repeats(words),
            *detect_repeated_take_prefixes(words),
            *detect_bad_marker_takes(words),
        ]
    )
