"""Conservative candidate-to-EDL decision planning.

Detectors and LLM/repair readers produce candidates. This module is the layer
that decides whether a candidate becomes an approved drop window, a blocked
window, or a review item.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from smart_video_editor.detection.local import detect_local_candidates
from smart_video_editor.editing.intervals import merge_intervals
from smart_video_editor.planning.boundary import BoundaryIssue, validate_cut_boundaries
from smart_video_editor.text import normalize_text
from smart_video_editor.timecode import seconds_to_timestamp


CandidateDecision = Literal["DROP", "REVIEW"]


@dataclass(frozen=True, slots=True)
class PlannerCandidate:
    start: float
    end: float
    reason: str
    source_text: str
    word_ids: tuple[int, ...] = ()
    source: str = "heuristic"
    force: bool = False
    confidence: float | None = None
    category: str = "other"
    recommended_action: CandidateDecision = "DROP"
    evidence: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class DecisionPlannerResult:
    drop_windows: list[tuple[float, float]]
    applied_windows: list[dict[str, object]] = field(default_factory=list)
    blocked_windows: list[dict[str, object]] = field(default_factory=list)
    review_windows: list[dict[str, object]] = field(default_factory=list)
    boundary_issues: list[dict[str, object]] = field(default_factory=list)
    simulated_text: str = ""


def _word_text(words: list[Any]) -> str:
    return " ".join(str(word.text) for word in words).strip()


def _token(word: Any) -> str:
    return normalize_text(str(word.text))


def _is_filler_token(token: str) -> bool:
    return token in {"e", "y", "a", "m", "hm", "em", "um", "uh", "eh", "yyy", "eee", "aaaa", "mmm", "hmm"}


def _candidate_words(
    candidate: PlannerCandidate,
    words: list[Any],
    words_by_id: dict[int, Any],
) -> list[Any]:
    if candidate.word_ids:
        return [words_by_id[word_id] for word_id in candidate.word_ids if word_id in words_by_id]
    return [word for word in words if word.end > candidate.start and word.timestamp < candidate.end]


def _keep_note_word_ranges(llm_summary: Any, words_by_id: dict[int, Any]) -> list[set[int]]:
    ranges: list[set[int]] = []
    for note in getattr(llm_summary, "keep_notes", []):
        try:
            start_id = int(note.get("start_word_id"))
            end_id = int(note.get("end_word_id"))
        except (AttributeError, TypeError, ValueError):
            continue
        if end_id < start_id:
            start_id, end_id = end_id, start_id
        ids = {word_id for word_id in range(start_id, end_id + 1) if word_id in words_by_id}
        if ids:
            ranges.append(ids)
    return ranges


def _should_block_for_semantics(
    candidate: PlannerCandidate,
    selected_words: list[Any],
    thought_blocks: list[Any],
    keep_ranges: list[set[int]],
) -> tuple[bool, str]:
    if not selected_words:
        return False, ""

    selected_ids = {int(word.id) for word in selected_words}
    reason = candidate.reason.lower()
    text = normalize_text(candidate.source_text)
    category_is_filler = "filler" in reason or (
        bool(text) and all(_is_filler_token(token) for token in text.split())
    )
    if category_is_filler:
        return False, ""

    for keep_range in keep_ranges:
        if selected_ids & keep_range:
            return True, "overlaps_llm_keep_note"

    first_section_heading_start = min(
        (
            min(block.word_ids)
            for block in thought_blocks
            if getattr(block, "role", "") == "section_heading" and getattr(block, "word_ids", [])
        ),
        default=None,
    )

    for block in thought_blocks:
        block_ids = {int(word_id) for word_id in getattr(block, "word_ids", [])}
        overlap = selected_ids & block_ids
        if not overlap:
            continue
        covers_block = overlap == block_ids
        if (
            covers_block
            and getattr(block, "role", "") == "section_heading"
            and first_section_heading_start is not None
            and min(block.word_ids) == first_section_heading_start
        ):
            return True, "first_section_heading_protected"
        if covers_block:
            continue
        if getattr(block, "role", "") in {"section_heading", "question", "structure_step"}:
            return True, f"partial_{block.role}_cut"

        overlap_ratio = len(overlap) / max(1, len(block_ids))
        source_can_cut_partial_thought = candidate.source in {"llm", "local"} or candidate.force
        if not source_can_cut_partial_thought and overlap_ratio > 0.15:
            return True, "heuristic_partial_thought_cut"

    return False, ""


def _snap_drop_boundary_to_silence(
    value: float,
    silences: list[tuple[float, float]],
    snap_window: float,
    direction: str,
) -> float:
    best_value = value
    best_distance = snap_window
    for start, end in silences:
        for candidate in (start, end):
            if direction == "start" and candidate > value:
                continue
            if direction == "end" and candidate < value:
                continue
            distance = abs(value - candidate)
            if distance <= best_distance:
                best_distance = distance
                best_value = candidate
    return best_value


def _safe_interval_for_candidate(
    candidate: PlannerCandidate,
    selected_words: list[Any],
    words: list[Any],
    duration: float,
    cut_safety_margin: float,
    silences: list[tuple[float, float]],
    silence_snap_window: float,
) -> tuple[float, float]:
    if selected_words:
        selected_ids = {int(word.id) for word in selected_words}
        first_word = min(selected_words, key=lambda word: word.timestamp)
        last_word = max(selected_words, key=lambda word: word.end)
        start_floor = 0.0
        end_ceiling = duration

        previous_words = [
            word
            for word in words
            if int(word.id) not in selected_ids and word.end <= first_word.timestamp
        ]
        next_words = [
            word
            for word in words
            if int(word.id) not in selected_ids and word.timestamp >= last_word.end
        ]
        if previous_words:
            start_floor = max(word.end for word in previous_words) + 0.01
        if next_words:
            end_ceiling = min(word.timestamp for word in next_words) - 0.01

        start = max(start_floor, first_word.timestamp - cut_safety_margin)
        end = min(end_ceiling, last_word.end + cut_safety_margin)
    else:
        start_floor = 0.0
        end_ceiling = duration
        start = candidate.start
        end = candidate.end

    start = max(0.0, min(start, duration))
    end = max(start, min(end, duration))
    start = _snap_drop_boundary_to_silence(start, silences, silence_snap_window, "start")
    end = _snap_drop_boundary_to_silence(end, silences, silence_snap_window, "end")
    if selected_words:
        start = max(start_floor, start)
        end = min(end_ceiling, end)
    start = max(0.0, min(start, duration))
    end = max(start, min(end, duration))
    return start, end


def _kept_words_after_drops(words: list[Any], drop_windows: list[tuple[float, float]]) -> list[Any]:
    kept: list[Any] = []
    for word in words:
        if any(word.end > start and word.timestamp < end for start, end in drop_windows):
            continue
        kept.append(word)
    return kept


def _issue_involves_candidate(issue: BoundaryIssue, candidate_word_ids: set[int]) -> bool:
    if not candidate_word_ids:
        return True
    if issue.removed_word_ids and candidate_word_ids & set(issue.removed_word_ids):
        return True
    if issue.left_word_id in candidate_word_ids or issue.right_word_id in candidate_word_ids:
        return True
    if issue.left_word_id is not None and issue.right_word_id is not None:
        between = set(range(min(issue.left_word_id, issue.right_word_id) + 1, max(issue.left_word_id, issue.right_word_id)))
        return bool(candidate_word_ids & between)
    return False


def _boundary_issue_record(issue: BoundaryIssue, words: list[Any], context_words: int) -> dict[str, object]:
    by_id = {int(word.id): word for word in words}
    left_id = issue.left_word_id
    right_id = issue.right_word_id
    before_words = []
    after_words = []
    if left_id is not None:
        before_words = [
            word
            for word in words
            if max(0, left_id - context_words + 1) <= int(word.id) <= left_id
        ]
    if right_id is not None:
        after_words = [
            word
            for word in words
            if right_id <= int(word.id) <= right_id + context_words - 1
        ]

    return {
        "kind": issue.kind,
        "action": issue.action,
        "reason": issue.reason,
        "start": seconds_to_timestamp(issue.start),
        "end": seconds_to_timestamp(issue.end),
        "left_word_id": left_id,
        "right_word_id": right_id,
        "left_word": str(by_id[left_id].text) if left_id in by_id else "",
        "right_word": str(by_id[right_id].text) if right_id in by_id else "",
        "removed_word_ids": list(issue.removed_word_ids),
        "before_text": _word_text(before_words),
        "after_text": _word_text(after_words),
    }


def _candidate_record(
    candidate: PlannerCandidate,
    start: float,
    end: float,
    selected_words: list[Any],
    **extra: object,
) -> dict[str, object]:
    record: dict[str, object] = {
        "start": seconds_to_timestamp(start),
        "end": seconds_to_timestamp(end),
        "candidate_start": seconds_to_timestamp(candidate.start),
        "candidate_end": seconds_to_timestamp(candidate.end),
        "reason": candidate.reason,
        "category": candidate.category,
        "source_text": candidate.source_text,
        "source": candidate.source,
        "confidence": candidate.confidence,
        "word_ids": [int(word.id) for word in selected_words],
        "force": candidate.force,
        "recommended_action": candidate.recommended_action,
    }
    if candidate.evidence:
        record["evidence"] = candidate.evidence
    record.update(extra)
    return record


def _candidate_boundary_issues(
    words: list[Any],
    planned_windows: list[tuple[float, float]],
    candidate_window: tuple[float, float],
    candidate_word_ids: set[int],
) -> list[BoundaryIssue]:
    issues = validate_cut_boundaries(words, [*planned_windows, candidate_window])
    return [
        issue
        for issue in issues
        if _issue_involves_candidate(issue, candidate_word_ids)
    ]


def validate_boundaries(
    words: list[Any],
    drop_windows: list[tuple[float, float]],
    context_words: int,
) -> list[dict[str, object]]:
    return [
        _boundary_issue_record(issue, words, context_words)
        for issue in validate_cut_boundaries(words, drop_windows)
    ]


def simulated_text_after_drops(words: list[Any], drop_windows: list[tuple[float, float]]) -> str:
    return _word_text(_kept_words_after_drops(words, drop_windows))


def plan_candidates(
    candidates: list[PlannerCandidate],
    *,
    words: list[Any],
    silences: list[tuple[float, float]],
    thought_blocks: list[Any],
    keep_ranges: list[set[int]] | None = None,
    duration: float,
    cut_safety_margin: float,
    silence_snap_window: float,
    context_words: int,
    disable_boundary_validator: bool = False,
) -> DecisionPlannerResult:
    words_by_id = {int(word.id): word for word in words}
    keep_ranges = keep_ranges or []

    planned: list[tuple[float, float]] = []
    unmerged_planned: list[tuple[float, float]] = []
    applied: list[dict[str, object]] = []
    blocked: list[dict[str, object]] = []
    review: list[dict[str, object]] = []

    for candidate in candidates:
        selected_words = _candidate_words(candidate, words, words_by_id)
        if candidate.recommended_action == "REVIEW":
            review.append(
                _candidate_record(
                    candidate,
                    candidate.start,
                    candidate.end,
                    selected_words,
                    review_reason="candidate_marked_review",
                )
            )
            continue

        block, block_reason = (False, "")
        if not candidate.force:
            block, block_reason = _should_block_for_semantics(
                candidate,
                selected_words,
                thought_blocks,
                keep_ranges,
            )
        if block:
            blocked.append(
                _candidate_record(
                    candidate,
                    candidate.start,
                    candidate.end,
                    selected_words,
                    block_reason=block_reason,
                )
            )
            continue

        if selected_words and not candidate.word_ids and words and not disable_boundary_validator:
            source_boundary_issues = validate_cut_boundaries(words, [(candidate.start, candidate.end)])
            blocking_source_issues = [issue for issue in source_boundary_issues if issue.action == "BLOCK"]
            if blocking_source_issues:
                blocked.append(
                    _candidate_record(
                        candidate,
                        candidate.start,
                        candidate.end,
                        selected_words,
                        block_reason="boundary_validator",
                        boundary_issues=[
                            _boundary_issue_record(issue, words, context_words)
                            for issue in blocking_source_issues
                        ],
                    )
                )
                continue

        start, end = _safe_interval_for_candidate(
            candidate,
            selected_words,
            words,
            duration,
            cut_safety_margin,
            silences,
            silence_snap_window,
        )
        if end <= start:
            blocked.append(
                _candidate_record(
                    candidate,
                    start,
                    end,
                    selected_words,
                    block_reason="empty_or_invalid_range",
                )
            )
            continue

        candidate_word_ids = {int(word.id) for word in selected_words}
        boundary_issues = []
        if words and not disable_boundary_validator:
            boundary_issues = _candidate_boundary_issues(
                words,
                planned,
                (start, end),
                candidate_word_ids,
            )
        blocking_issues = [
            issue
            for issue in boundary_issues
            if issue.action == "BLOCK" or (issue.action == "REVIEW" and not candidate.force)
        ]
        if blocking_issues:
            blocked.append(
                _candidate_record(
                    candidate,
                    start,
                    end,
                    selected_words,
                    block_reason="boundary_validator",
                    boundary_issues=[
                        _boundary_issue_record(issue, words, context_words)
                        for issue in blocking_issues
                    ],
                )
            )
            continue

        unmerged_planned.append((start, end))
        planned = merge_intervals(unmerged_planned)
        applied.append(_candidate_record(candidate, start, end, selected_words))

    boundary_issues: list[dict[str, object]] = []
    if words and not disable_boundary_validator:
        boundary_issues = validate_boundaries(words, planned, context_words)

    return DecisionPlannerResult(
        drop_windows=planned,
        applied_windows=applied,
        blocked_windows=blocked,
        review_windows=review,
        boundary_issues=boundary_issues,
        simulated_text=simulated_text_after_drops(words, planned) if words else "",
    )


def _entry_drop_candidates(entries: list[Any]) -> list[PlannerCandidate]:
    candidates: list[PlannerCandidate] = []
    for entry in entries:
        if not getattr(entry, "drop", False):
            continue
        candidates.append(
            PlannerCandidate(
                start=entry.timestamp,
                end=entry.end,
                reason=";".join(getattr(entry, "reasons", [])),
                source_text=entry.text,
                word_ids=tuple(int(word_id) for word_id in getattr(entry, "word_ids", [])),
                source="heuristic",
                category="entry_drop",
            )
        )
    return candidates


def _partial_window_candidates(partial_drop_windows: list[Any]) -> list[PlannerCandidate]:
    candidates: list[PlannerCandidate] = []
    for window in partial_drop_windows:
        candidates.append(
            PlannerCandidate(
                start=window.start,
                end=window.end,
                reason=window.reason,
                source_text=window.source_text,
                word_ids=tuple(int(word_id) for word_id in getattr(window, "word_ids", [])),
                source=getattr(window, "source", "heuristic"),
                force=bool(getattr(window, "force", False)),
                category=str(getattr(window, "source", "candidate")),
            )
        )
    return candidates


def _local_candidates(words: list[Any]) -> list[PlannerCandidate]:
    candidates: list[PlannerCandidate] = []
    for candidate in detect_local_candidates(words):
        candidates.append(
            PlannerCandidate(
                start=candidate.start,
                end=candidate.end,
                reason=candidate.reason,
                source_text=candidate.text,
                word_ids=tuple(range(candidate.start_word_id, candidate.end_word_id + 1)),
                source=candidate.source,
                confidence=candidate.confidence,
                category=candidate.category,
                recommended_action=candidate.recommended_action,
                evidence=candidate.evidence,
            )
        )
    return candidates


def plan_drop_windows(
    entries: list[Any],
    partial_drop_windows: list[Any],
    words: list[Any],
    silences: list[tuple[float, float]],
    thought_blocks: list[Any],
    llm_summary: Any,
    duration: float,
    cut_safety_margin: float,
    silence_snap_window: float,
    context_words: int,
    disable_boundary_validator: bool,
) -> DecisionPlannerResult:
    words_by_id = {int(word.id): word for word in words}
    keep_ranges = _keep_note_word_ranges(llm_summary, words_by_id)
    candidates = [
        *_entry_drop_candidates(entries),
        *_partial_window_candidates(partial_drop_windows),
        *_local_candidates(words),
    ]
    return plan_candidates(
        candidates,
        words=words,
        silences=silences,
        thought_blocks=thought_blocks,
        keep_ranges=keep_ranges,
        duration=duration,
        cut_safety_margin=cut_safety_margin,
        silence_snap_window=silence_snap_window,
        context_words=context_words,
        disable_boundary_validator=disable_boundary_validator,
    )
