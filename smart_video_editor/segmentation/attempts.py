"""Repeated-attempt grouping for spoken takes."""

from __future__ import annotations

from dataclasses import dataclass

from smart_video_editor.domain.models import TranscriptWord
from smart_video_editor.text import normalize_text


FILLER_TOKENS = {"e", "em", "eee", "yyy", "hmm", "mhm", "no", "dobra", "znaczy"}
INCOMPLETE_END_TOKENS = {
    "bo",
    "ci",
    "czyli",
    "dla",
    "do",
    "jak",
    "na",
    "o",
    "po",
    "przez",
    "to",
    "w",
    "wam",
    "wiec",
    "więc",
    "z",
    "ze",
    "że",
    "zeby",
    "żeby",
}
OPEN_ENDED_VERBS = {
    "omowie",
    "omówię",
    "opowiem",
    "pokaze",
    "pokaże",
    "pokażę",
    "przejde",
    "przejdę",
    "wyjasnie",
    "wyjaśnię",
}


@dataclass(frozen=True, slots=True)
class AttemptSpan:
    start_index: int
    end_index: int
    start_word_id: int
    end_word_id: int
    start: float
    end: float
    text: str
    tokens: tuple[str, ...]
    word_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class AttemptCompleteness:
    token_count: int
    score: float
    markers: tuple[str, ...]
    is_complete: bool


@dataclass(frozen=True, slots=True)
class RepeatedAttemptGroup:
    id: int
    earlier: AttemptSpan
    later: AttemptSpan
    shared_prefix: tuple[str, ...]
    shared_prefix_word_count: int
    later_extra_word_count: int
    earlier_completeness: AttemptCompleteness
    later_completeness: AttemptCompleteness
    confidence: float
    recommended_action: str
    reason: str


def _token(word: TranscriptWord) -> str:
    return normalize_text(word.text)


def _is_prefix_token(fragment: str, full: str) -> bool:
    return len(fragment) >= 3 and fragment != full and full.startswith(fragment)


def _content_tokens(
    words: list[TranscriptWord],
    start_index: int,
    end_index: int,
) -> list[tuple[int, str]]:
    tokens: list[tuple[int, str]] = []
    for index in range(start_index, min(end_index, len(words))):
        token = _token(words[index])
        if token and token not in FILLER_TOKENS:
            tokens.append((index, token))
    return tokens


def _common_prefix(
    left: list[tuple[int, str]],
    right: list[tuple[int, str]],
) -> list[tuple[int, str]]:
    prefix: list[tuple[int, str]] = []
    for left_item, right_item in zip(left, right):
        left_token = left_item[1]
        right_token = right_item[1]
        if left_token == right_token or _is_prefix_token(left_token, right_token):
            prefix.append(left_item)
            continue
        break
    return prefix


def _span(
    words: list[TranscriptWord],
    start_index: int,
    end_index: int,
    tokens: list[tuple[int, str]],
) -> AttemptSpan:
    group = words[start_index : end_index + 1]
    return AttemptSpan(
        start_index=start_index,
        end_index=end_index,
        start_word_id=group[0].id,
        end_word_id=group[-1].id,
        start=group[0].timestamp,
        end=group[-1].end,
        text=" ".join(word.text for word in group).strip(),
        tokens=tuple(token for _, token in tokens),
        word_ids=tuple(word.id for word in group),
    )


def is_incomplete_attempt_tokens(tokens: tuple[str, ...] | list[str]) -> bool:
    """Return whether token sequence looks like an unfinished spoken thought."""
    return not score_attempt_completeness(tokens).is_complete


def score_attempt_completeness(tokens: tuple[str, ...] | list[str]) -> AttemptCompleteness:
    """Score whether a token sequence looks like a complete spoken attempt."""
    content_tokens = tuple(
        normalized
        for token in tokens
        if (normalized := normalize_text(token)) and normalized not in FILLER_TOKENS
    )
    markers: list[str] = []
    if not content_tokens:
        return AttemptCompleteness(
            token_count=0,
            score=0.0,
            markers=("empty_attempt",),
            is_complete=False,
        )

    score = min(1.0, 0.38 + 0.09 * len(content_tokens))
    last = content_tokens[-1]
    if last in INCOMPLETE_END_TOKENS:
        markers.append("ends_with_incomplete_token")
        score -= 0.45
    if last in OPEN_ENDED_VERBS:
        markers.append("ends_with_open_ended_verb")
        score -= 0.38
    if len(content_tokens) <= 2:
        markers.append("very_short_attempt")
        score -= 0.08
    if any(_is_prefix_token(last, token) for token in content_tokens[:-1]):
        markers.append("ends_with_truncated_prefix")
        score -= 0.35

    score = max(0.0, min(1.0, score))
    blocking_markers = {
        "empty_attempt",
        "ends_with_incomplete_token",
        "ends_with_open_ended_verb",
        "ends_with_truncated_prefix",
    }
    return AttemptCompleteness(
        token_count=len(content_tokens),
        score=round(score, 3),
        markers=tuple(markers),
        is_complete=score >= 0.62 and not (set(markers) & blocking_markers),
    )


def _legacy_incomplete_attempt_tokens(tokens: tuple[str, ...] | list[str]) -> bool:
    normalized_tokens = tuple(
        normalized for token in tokens if (normalized := normalize_text(token))
    )
    if not normalized_tokens:
        return False
    last = normalized_tokens[-1]
    if last in INCOMPLETE_END_TOKENS or last in OPEN_ENDED_VERBS:
        return True
    if any(_is_prefix_token(last, token) for token in normalized_tokens[:-1]):
        return True
    return False


def classify_repeated_attempt(
    earlier_tokens: tuple[str, ...] | list[str],
    *,
    later_tokens: tuple[str, ...] | list[str] = (),
    shared_prefix_word_count: int,
    later_extra_word_count: int,
    boundary_gap: float,
    pause_signal: float = 0.65,
) -> tuple[str, str, float]:
    """Classify a repeated attempt as DROP or REVIEW with a conservative reason."""
    earlier_completeness = score_attempt_completeness(earlier_tokens)
    later_completeness = score_attempt_completeness(later_tokens) if later_tokens else None

    if later_extra_word_count < 2:
        return "REVIEW", "repeated_attempt_without_fuller_later_take", 0.62

    if later_completeness is not None and not later_completeness.is_complete:
        return "REVIEW", "later_attempt_also_incomplete_needs_review", 0.7

    if not earlier_completeness.is_complete or _legacy_incomplete_attempt_tokens(earlier_tokens):
        return "DROP", "earlier_incomplete_attempt_before_fuller_restart", 0.9

    if len(earlier_tokens) <= 2 and shared_prefix_word_count >= 2:
        return "DROP", "short_starter_attempt_before_fuller_restart", 0.86

    if boundary_gap >= pause_signal:
        return "REVIEW", "possible_repeated_attempt_after_pause_needs_review", 0.74

    return "REVIEW", "possible_rhetorical_or_intentional_repeat", 0.68


def find_repeated_attempt_groups(
    words: list[TranscriptWord],
    *,
    min_prefix_words: int = 2,
    max_attempt_words: int = 18,
    max_restart_scan_words: int = 24,
    max_restart_gap: float = 10.0,
) -> list[RepeatedAttemptGroup]:
    """Find earlier attempts followed by a later fuller restart."""
    groups: list[RepeatedAttemptGroup] = []
    seen_starts: set[tuple[int, int]] = set()

    for start_index in range(len(words)):
        start_token = _token(words[start_index])
        if not start_token or start_token in FILLER_TOKENS:
            continue

        scan_end = min(len(words), start_index + max_restart_scan_words + 1)
        for restart_index in range(start_index + min_prefix_words, scan_end):
            if words[restart_index].timestamp - words[start_index].timestamp > max_restart_gap:
                break
            restart_token = _token(words[restart_index])
            if restart_token != start_token:
                continue
            if (
                start_index > 0
                and restart_index > 0
                and _token(words[start_index - 1]) == _token(words[restart_index - 1])
            ):
                continue

            earlier_tokens = _content_tokens(words, start_index, restart_index)
            later_tokens = _content_tokens(
                words,
                restart_index,
                min(len(words), restart_index + max_attempt_words),
            )
            prefix = _common_prefix(earlier_tokens, later_tokens)
            if len(prefix) < min_prefix_words:
                continue

            later_extra = max(0, len(later_tokens) - len(prefix))
            boundary_gap = words[restart_index].timestamp - words[restart_index - 1].end
            action, reason, confidence = classify_repeated_attempt(
                tuple(token for _, token in earlier_tokens),
                later_tokens=tuple(token for _, token in later_tokens),
                shared_prefix_word_count=len(prefix),
                later_extra_word_count=later_extra,
                boundary_gap=boundary_gap,
            )
            earlier_completeness = score_attempt_completeness(
                tuple(token for _, token in earlier_tokens)
            )
            later_completeness = score_attempt_completeness(
                tuple(token for _, token in later_tokens)
            )
            key = (start_index, restart_index)
            if key in seen_starts:
                continue
            seen_starts.add(key)
            groups.append(
                RepeatedAttemptGroup(
                    id=len(groups),
                    earlier=_span(words, start_index, restart_index - 1, earlier_tokens),
                    later=_span(
                        words,
                        restart_index,
                        min(len(words) - 1, restart_index + max_attempt_words - 1),
                        later_tokens,
                    ),
                    shared_prefix=tuple(token for _, token in prefix),
                    shared_prefix_word_count=len(prefix),
                    later_extra_word_count=later_extra,
                    earlier_completeness=earlier_completeness,
                    later_completeness=later_completeness,
                    confidence=confidence,
                    recommended_action=action,
                    reason=reason,
                )
            )
            break

    return groups
