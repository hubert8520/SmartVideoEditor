"""Conservative local detectors for edit candidates.

Detectors do not cut media. They produce candidate ranges that a planner can
accept, downgrade to REVIEW, or reject after boundary validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, cast

from smart_video_editor.domain.models import TranscriptWord
from smart_video_editor.segmentation.attempts import (
    AttemptCompleteness,
    AttemptSpan,
    RepeatedAttemptGroup,
    classify_repeated_attempt,
    find_repeated_attempt_groups,
    score_attempt_completeness,
)
from smart_video_editor.segmentation.takes import segment_take_indices
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
FILLER_TOKENS = {"e", "em", "eee", "yyy", "hmm", "mhm", "no", "dobra", "znaczy"}
STRONG_RETAKE_MARKERS = {
    ("jeszcze", "raz"),
    ("od", "poczatku"),
    ("od", "nowa"),
    ("nie", "tak"),
    ("zle",),
    ("stop",),
    ("pomylilem", "sie"),
}


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
    evidence: dict[str, object] = field(default_factory=dict)


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
    evidence: dict[str, object] | None = None,
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
        evidence=evidence or {},
    )


def _completeness_evidence(completeness: AttemptCompleteness) -> dict[str, object]:
    return {
        "token_count": completeness.token_count,
        "score": completeness.score,
        "markers": list(completeness.markers),
        "is_complete": completeness.is_complete,
    }


def _span_evidence(span: AttemptSpan, completeness: AttemptCompleteness) -> dict[str, object]:
    return {
        "text": span.text,
        "word_ids": list(span.word_ids),
        "tokens": list(span.tokens),
        "completeness": _completeness_evidence(completeness),
    }


def _attempt_group_evidence(group: RepeatedAttemptGroup) -> dict[str, object]:
    return {
        "attempt_group_id": group.id,
        "shared_prefix": list(group.shared_prefix),
        "shared_prefix_word_count": group.shared_prefix_word_count,
        "later_extra_word_count": group.later_extra_word_count,
        "earlier": _span_evidence(group.earlier, group.earlier_completeness),
        "later": _span_evidence(group.later, group.later_completeness),
    }


def _prefix_attempt_evidence(
    earlier_words: list[TranscriptWord],
    earlier_tokens: tuple[str, ...],
    later_words: list[TranscriptWord],
    later_tokens: tuple[str, ...],
    *,
    shared_prefix_word_count: int,
    later_extra_word_count: int,
) -> dict[str, object]:
    return {
        "shared_prefix": list(earlier_tokens[:shared_prefix_word_count]),
        "shared_prefix_word_count": shared_prefix_word_count,
        "later_extra_word_count": later_extra_word_count,
        "earlier": {
            "text": _candidate_text(earlier_words),
            "word_ids": [word.id for word in earlier_words],
            "tokens": list(earlier_tokens),
            "completeness": _completeness_evidence(score_attempt_completeness(earlier_tokens)),
        },
        "later": {
            "text": _candidate_text(later_words),
            "word_ids": [word.id for word in later_words],
            "tokens": list(later_tokens),
            "completeness": _completeness_evidence(score_attempt_completeness(later_tokens)),
        },
    }


def _is_prefix_token(fragment: str, full: str) -> bool:
    return len(fragment) >= 3 and fragment != full and full.startswith(fragment)


def _ranges_overlap(left: EditCandidate, right: EditCandidate) -> bool:
    return not (left.end_word_id < right.start_word_id or right.end_word_id < left.start_word_id)


def _contains(left: EditCandidate, right: EditCandidate) -> bool:
    return left.start_word_id <= right.start_word_id and left.end_word_id >= right.end_word_id


def _candidate_rank(candidate: EditCandidate) -> tuple[int, int, float, int]:
    category_rank = {
        "bad_marker_take": 5,
        "partial_repeat": 4,
        "repeated_attempt": 3,
        "repeated_take": 2,
    }.get(candidate.category, 1)
    action_rank = 1 if candidate.recommended_action == "DROP" else 0
    length = candidate.end_word_id - candidate.start_word_id + 1
    return category_rank, action_rank, candidate.confidence, length


def _dedupe(candidates: list[EditCandidate]) -> list[EditCandidate]:
    seen: set[tuple[str, int, int]] = set()
    exact: list[EditCandidate] = []
    for candidate in candidates:
        key = (candidate.category, candidate.start_word_id, candidate.end_word_id)
        if key in seen:
            continue
        seen.add(key)
        exact.append(candidate)

    result: list[EditCandidate] = []
    for candidate in exact:
        should_add = True
        for index, existing in enumerate(result):
            if not _ranges_overlap(candidate, existing):
                continue
            if not (_contains(candidate, existing) or _contains(existing, candidate)):
                continue
            if _candidate_rank(candidate) > _candidate_rank(existing):
                result[index] = candidate
            should_add = False
            break
        if should_add:
            result.append(candidate)
    return result


def _take_bounds_by_index(
    words: list[TranscriptWord],
    *,
    max_gap: float,
) -> dict[int, tuple[int, int]]:
    bounds: dict[int, tuple[int, int]] = {}
    for start_index, end_index in segment_take_indices(words, max_gap=max_gap):
        for index in range(start_index, end_index + 1):
            bounds[index] = (start_index, end_index)
    return bounds


def _match_phrase(tokens: list[str], index: int) -> tuple[str, ...] | None:
    matches = [
        phrase
        for phrase in BAD_MARKER_PHRASES
        if _phrase_at(tokens, index, phrase)
    ]
    if not matches:
        return None
    return max(matches, key=len)


def _marker_cluster_end(
    words: list[TranscriptWord],
    tokens: list[str],
    index: int,
    *,
    max_marker_gap: float,
) -> tuple[int, tuple[tuple[str, ...], ...]]:
    phrases: list[tuple[str, ...]] = []
    current = index
    end_index = index
    while current < len(words):
        probe = current
        while phrases and probe < len(words) and tokens[probe] in FILLER_TOKENS:
            gap = words[probe].timestamp - words[end_index].end
            if gap > max_marker_gap:
                break
            end_index = probe
            probe += 1
        current = probe
        phrase = _match_phrase(tokens, current)
        if phrase is None:
            break
        if phrases:
            gap = words[current].timestamp - words[end_index].end
            if gap > max_marker_gap:
                break
        phrases.append(phrase)
        end_index = current + len(phrase) - 1
        current = end_index + 1
    return end_index, tuple(phrases)


def _has_strong_retake_marker(phrases: tuple[tuple[str, ...], ...]) -> bool:
    return any(phrase in STRONG_RETAKE_MARKERS for phrase in phrases)


def _restart_prefix_length(
    tokens: list[str],
    failed_start_index: int,
    failed_end_index: int,
    restart_index: int,
    *,
    max_prefix_words: int,
) -> int:
    failed_tokens = [
        token
        for token in tokens[failed_start_index:failed_end_index]
        if token and token not in FILLER_TOKENS
    ]
    restart_tokens = [
        token
        for token in tokens[restart_index : restart_index + max_prefix_words]
        if token and token not in FILLER_TOKENS
    ]
    match_count = 0
    for left, right in zip(failed_tokens[:max_prefix_words], restart_tokens):
        if left == right or _is_prefix_token(left, right):
            match_count += 1
            continue
        break
    return match_count


def _find_restart_after_marker(
    words: list[TranscriptWord],
    tokens: list[str],
    *,
    failed_start_index: int,
    marker_start_index: int,
    marker_end_index: int,
    min_prefix_words: int,
    max_prefix_words: int,
    max_restart_scan_words: int,
    max_restart_gap: float,
) -> tuple[int, int] | None:
    search_end = min(len(words) - 1, marker_end_index + max_restart_scan_words)
    for restart_index in range(marker_end_index + 1, search_end + 1):
        if words[restart_index].timestamp - words[marker_end_index].end > max_restart_gap:
            break
        prefix_length = _restart_prefix_length(
            tokens,
            failed_start_index,
            marker_start_index,
            restart_index,
            max_prefix_words=max_prefix_words,
        )
        if prefix_length >= min_prefix_words:
            return restart_index, prefix_length
    return None


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


def detect_truncated_word_restarts(
    words: list[TranscriptWord],
    *,
    max_gap: float = 0.8,
) -> list[EditCandidate]:
    """Find direct word-level restarts such as ``skonfi skonfigurowac``."""
    tokens = [_token(word) for word in words]
    candidates: list[EditCandidate] = []

    for index in range(len(words) - 1):
        left = tokens[index]
        right = tokens[index + 1]
        if not left or not right:
            continue
        if left in COMMON_STARTERS or left in FILLER_TOKENS:
            continue
        if words[index + 1].timestamp - words[index].end > max_gap:
            continue
        if not _is_prefix_token(left, right):
            continue
        if len(right) - len(left) < 2:
            continue

        candidates.append(
            _make_candidate(
                [words[index]],
                category="partial_repeat",
                reason="truncated_word_before_immediate_restart",
                confidence=0.9,
                recommended_action="DROP",
            )
        )

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

                later_words = words[repeat_index:search_end]
                later_tokens = tuple(tokens[repeat_index:search_end])
                later_extra = sum(1 for token in tokens[repeat_index + phrase_len : search_end] if token)
                boundary_gap = words[repeat_index].timestamp - words[end_index - 1].end
                action, reason, confidence = classify_repeated_attempt(
                    tuple(phrase),
                    later_tokens=later_tokens,
                    shared_prefix_word_count=phrase_len,
                    later_extra_word_count=later_extra,
                    boundary_gap=boundary_gap,
                )
                candidate_words = words[start_index:end_index]
                candidates.append(
                    _make_candidate(
                        candidate_words,
                        category="repeated_take",
                        reason=reason,
                        confidence=confidence,
                        recommended_action=action,
                        evidence=_prefix_attempt_evidence(
                            candidate_words,
                            tuple(phrase),
                            later_words,
                            later_tokens,
                            shared_prefix_word_count=phrase_len,
                            later_extra_word_count=later_extra,
                        ),
                    )
                )
                break
            else:
                continue
            break

    return _dedupe(candidates)


def detect_repeated_attempts(words: list[TranscriptWord]) -> list[EditCandidate]:
    """Find repeated spoken attempts and classify uncertain repeats as REVIEW."""
    candidates: list[EditCandidate] = []
    for group in find_repeated_attempt_groups(words):
        candidate_words = words[group.earlier.start_index : group.earlier.end_index + 1]
        candidates.append(
            _make_candidate(
                candidate_words,
                category="repeated_attempt",
                reason=group.reason,
                confidence=group.confidence,
                recommended_action=cast(CandidateAction, group.recommended_action),
                evidence=_attempt_group_evidence(group),
            )
        )
    return _dedupe(candidates)


def _phrase_at(tokens: list[str], index: int, phrase: tuple[str, ...]) -> bool:
    return tuple(tokens[index : index + len(phrase)]) == phrase


def detect_bad_marker_takes(
    words: list[TranscriptWord],
    *,
    max_take_words_before_marker: int = 16,
    max_take_gap: float = 1.2,
    max_marker_gap: float = 0.55,
    min_restart_prefix_words: int = 2,
    max_restart_prefix_words: int = 8,
    max_restart_scan_words: int = 24,
    max_restart_gap: float = 8.0,
) -> list[EditCandidate]:
    """Expand bad markers to the failed take that precedes them."""
    tokens = [_token(word) for word in words]
    candidates: list[EditCandidate] = []
    take_bounds = _take_bounds_by_index(words, max_gap=max_take_gap)
    consumed_until = -1

    for index in range(len(words)):
        if index <= consumed_until:
            continue
        matched_phrase = _match_phrase(tokens, index)
        if matched_phrase is None:
            continue

        marker_end_index, marker_phrases = _marker_cluster_end(
            words,
            tokens,
            index,
            max_marker_gap=max_marker_gap,
        )
        consumed_until = marker_end_index

        take_start_index, _ = take_bounds.get(index, (0, len(words) - 1))
        start_index = index
        floor = max(take_start_index, index - max_take_words_before_marker)
        for cursor in range(index - 1, floor - 1, -1):
            gap = words[cursor + 1].timestamp - words[cursor].end
            if gap > max_take_gap:
                break
            start_index = cursor

        has_failed_take_context = start_index < index
        restart = None
        if has_failed_take_context:
            restart = _find_restart_after_marker(
                words,
                tokens,
                failed_start_index=start_index,
                marker_start_index=index,
                marker_end_index=marker_end_index,
                min_prefix_words=min_restart_prefix_words,
                max_prefix_words=max_restart_prefix_words,
                max_restart_scan_words=max_restart_scan_words,
                max_restart_gap=max_restart_gap,
            )

        strong_retake_marker = _has_strong_retake_marker(marker_phrases)
        if restart is not None:
            candidate_words = words[start_index : marker_end_index + 1]
            reason = "bad_marker_failed_take_before_confirmed_restart"
            confidence = 0.93
            recommended_action: CandidateAction = "DROP"
        elif has_failed_take_context and strong_retake_marker:
            candidate_words = words[start_index : marker_end_index + 1]
            reason = "bad_marker_failed_take_without_confirmed_restart"
            confidence = 0.78
            recommended_action = "REVIEW"
        else:
            candidate_words = words[index : marker_end_index + 1]
            reason = "ambiguous_bad_marker_without_confirmed_restart"
            confidence = 0.68
            recommended_action = "REVIEW"

        candidates.append(
            _make_candidate(
                candidate_words,
                category="bad_marker_take",
                reason=reason,
                confidence=confidence,
                recommended_action=recommended_action,
            )
        )

    return _dedupe(candidates)


def detect_local_candidates(words: list[TranscriptWord]) -> list[EditCandidate]:
    """Run all deterministic local detectors."""
    return _dedupe(
        [
            *detect_partial_repeats(words),
            *detect_truncated_word_restarts(words),
            *detect_repeated_attempts(words),
            *detect_repeated_take_prefixes(words),
            *detect_bad_marker_takes(words),
        ]
    )
