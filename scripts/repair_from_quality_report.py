#!/usr/bin/env python3
"""Create a conservative repair plan from the final quality report."""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from smart_video_editor.reporting.quality import normalize_repair_suggestion  # noqa: E402
from smart_video_editor.reporting.timeline import raw_range_tuples  # noqa: E402

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_QUALITY_REPORT = ARTIFACTS_DIR / "final_quality_report.json"
DEFAULT_EDIT_DECISIONS = ARTIFACTS_DIR / "edit_decisions.json"
DEFAULT_RAW_TRANSCRIPT = ARTIFACTS_DIR / "raw_transcription.json"
DEFAULT_EDITED_TRANSCRIPT = ARTIFACTS_DIR / "edited_transcription.json"
DEFAULT_OUTPUT = ARTIFACTS_DIR / "repair_plan.json"

SEVERITY_RANK = {"low": 1, "medium": 2, "high": 3}
CONTENT_MIN_LENGTH = 4
COMMON_SHORT_TOKENS = {
    "albo",
    "ale",
    "bo",
    "czyli",
    "oraz",
    "taki",
    "taka",
    "takie",
    "tego",
    "to",
    "wiec",
    "więc",
    "zeby",
    "żeby",
}
AUTO_DROP_CATEGORIES = {"repetition", "off_topic", "noise_or_setup"}
AUTO_KEEP_CATEGORIES = {"cut_word", "dangling_thought", "logic_gap"}
CONNECTOR_TOKENS = {
    "ale",
    "bo",
    "czyli",
    "dlatego",
    "natomiast",
    "poniewaz",
    "ponieważ",
    "wiec",
    "więc",
    "zeby",
    "żeby",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read final_quality_report.json and create artifacts/repair_plan.json "
            "with conservative operations for the next render iteration."
        )
    )
    parser.add_argument("--quality-report", type=Path, default=DEFAULT_QUALITY_REPORT)
    parser.add_argument("--edit-decisions", type=Path, default=DEFAULT_EDIT_DECISIONS)
    parser.add_argument("--raw-transcript", type=Path, default=DEFAULT_RAW_TRANSCRIPT)
    parser.add_argument("--edited-transcript", type=Path, default=DEFAULT_EDITED_TRANSCRIPT)
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--iteration",
        type=int,
        default=2,
        help="Iteration number written to the repair plan. Default: 2.",
    )
    parser.add_argument(
        "--min-severity",
        choices=("low", "medium", "high"),
        default="medium",
        help="Lowest QA severity repaired automatically. Default: medium.",
    )
    parser.add_argument(
        "--gap-threshold",
        type=float,
        default=0.6,
        help="Minimum raw gap treated as a possible missing connector. Default: 0.6.",
    )
    parser.add_argument(
        "--max-gap-repair",
        type=float,
        default=2.25,
        help="Maximum raw gap to auto-keep as possible unrecognized speech. Default: 2.25.",
    )
    parser.add_argument(
        "--word-padding",
        type=float,
        default=0.12,
        help="Padding for force_keep_words repairs. Default: 0.12.",
    )
    parser.add_argument(
        "--gap-padding",
        type=float,
        default=0.04,
        help="Padding for force_keep_interval gap repairs. Default: 0.04.",
    )
    return parser.parse_args()


def fail(message: str) -> None:
    raise SystemExit(f"Error: {message}")


def load_json(path: Path) -> Any:
    if not path.exists():
        fail(f"Missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def timestamp_to_seconds(timestamp: str) -> float:
    parts = timestamp.strip().split(":")
    if len(parts) != 4:
        fail(f"Invalid timestamp: {timestamp!r}")
    hours, minutes, seconds, milliseconds = parts
    return (
        int(hours) * 3600
        + int(minutes) * 60
        + int(seconds)
        + int(milliseconds.ljust(3, "0")[:3]) / 1000
    )


def seconds_to_timestamp(seconds: float) -> str:
    total_milliseconds = max(0, int(round(seconds * 1000)))
    total_seconds, milliseconds = divmod(total_milliseconds, 1000)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}:{milliseconds:03d}"


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def normalize_text(text: str) -> str:
    text = strip_accents(text.lower())
    text = re.sub(r"[^a-z0-9ąćęłńóśźż]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return [token for token in normalize_text(text).split() if token]


def raw_word_text(word: dict[str, Any]) -> str:
    return str(word.get("word", "")).strip()


def raw_word_token(word: dict[str, Any]) -> str:
    tokens = tokenize(raw_word_text(word))
    return tokens[0] if tokens else ""


def word_start(word: dict[str, Any]) -> float:
    return timestamp_to_seconds(str(word["timestamp"]))


def word_end(word: dict[str, Any]) -> float:
    return timestamp_to_seconds(str(word["end"]))


def build_timeline_from_keep_intervals(decisions: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(decisions.get("timeline_map"), list) and decisions["timeline_map"]:
        return decisions["timeline_map"]

    timeline: list[dict[str, Any]] = []
    cursor = 0.0
    for index, item in enumerate(decisions.get("keep_intervals", [])):
        raw_start = timestamp_to_seconds(str(item["start"]))
        raw_end = timestamp_to_seconds(str(item["end"]))
        duration = raw_end - raw_start
        timeline.append(
            {
                "id": index,
                "raw_start": seconds_to_timestamp(raw_start),
                "raw_end": seconds_to_timestamp(raw_end),
                "final_start": seconds_to_timestamp(cursor),
                "final_end": seconds_to_timestamp(cursor + duration),
                "duration_seconds": round(duration, 6),
            }
        )
        cursor += duration
    return timeline


def map_final_range_to_raw_ranges(
    final_start: float,
    final_end: float,
    timeline: list[dict[str, Any]],
) -> list[tuple[float, float, int]]:
    ranges: list[tuple[float, float, int]] = []
    for item in timeline:
        item_final_start = timestamp_to_seconds(str(item["final_start"]))
        item_final_end = timestamp_to_seconds(str(item["final_end"]))
        overlap_start = max(final_start, item_final_start)
        overlap_end = min(final_end, item_final_end)
        if overlap_end <= overlap_start:
            continue

        raw_start = timestamp_to_seconds(str(item["raw_start"]))
        mapped_start = raw_start + (overlap_start - item_final_start)
        mapped_end = raw_start + (overlap_end - item_final_start)
        ranges.append((mapped_start, mapped_end, int(item.get("id", len(ranges)))))
    return ranges


def words_in_ranges(
    words: list[dict[str, Any]],
    ranges: list[tuple[float, float, int]],
    context: float = 0.0,
    center_only: bool = False,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[int] = set()
    for start, end, _ in ranges:
        for word in words:
            word_id = int(word["id"])
            if word_id in seen:
                continue
            if center_only:
                midpoint = (word_start(word) + word_end(word)) / 2
                in_range = start - context <= midpoint <= end + context
            else:
                in_range = word_end(word) > start - context and word_start(word) < end + context
            if in_range:
                selected.append(word)
                seen.add(word_id)
    return sorted(selected, key=lambda word: int(word["id"]))


def issue_word_ids(issue: dict[str, Any], edited_words: list[dict[str, Any]]) -> set[int]:
    try:
        start_id = int(issue["start_word_id"])
        end_id = int(issue["end_word_id"])
    except (KeyError, TypeError, ValueError):
        return set()
    if end_id < start_id:
        start_id, end_id = end_id, start_id
    edited_ids = {int(word["id"]) for word in edited_words}
    return {word_id for word_id in range(start_id, end_id + 1) if word_id in edited_ids}


def source_issue_summary(issue: dict[str, Any], raw_ranges: list[tuple[float, float, int]]) -> dict[str, Any]:
    return {
        "start": issue.get("start"),
        "end": issue.get("end"),
        "issue_category": issue.get("issue_category"),
        "severity": issue.get("severity"),
        "affected_text": issue.get("affected_text"),
        "description": issue.get("description"),
        "raw_ranges": [
            {
                "raw_start": seconds_to_timestamp(start),
                "raw_end": seconds_to_timestamp(end),
                "timeline_id": timeline_id,
            }
            for start, end, timeline_id in raw_ranges
        ],
        "repair_suggestion": normalize_repair_suggestion(issue),
    }


def raw_ranges_for_issue(
    issue: dict[str, Any],
    final_start: float,
    final_end: float,
    timeline: list[dict[str, Any]],
) -> list[tuple[float, float, int]]:
    embedded = issue.get("raw_ranges")
    if isinstance(embedded, list) and embedded:
        tuples = raw_range_tuples([item for item in embedded if isinstance(item, dict)])
        if tuples:
            return tuples
    return map_final_range_to_raw_ranges(final_start, final_end, timeline)


def repair_action_skip_reason(category: str, suggestion: dict[str, Any]) -> str | None:
    action = str(suggestion.get("action", "manual_review"))
    confidence = str(suggestion.get("confidence", "low"))
    if action == "manual_review":
        return "qa_requested_manual_review"
    if action == "no_auto_repair":
        return "qa_requested_no_auto_repair"
    if action == "force_drop":
        if category not in AUTO_DROP_CATEGORIES:
            return "inconsistent_qa_repair_action"
        if confidence != "high":
            return "force_drop_requires_high_confidence"
        return None
    if action == "force_keep":
        if category not in AUTO_KEEP_CATEGORIES:
            return "inconsistent_qa_repair_action"
        return None
    return "unsupported_qa_repair_action"


def skip_record(
    issue_index: int,
    skip_reason: str,
    issue: dict[str, Any],
    raw_ranges: list[tuple[float, float, int]],
) -> dict[str, Any]:
    return {
        "issue_index": issue_index,
        "skip_reason": skip_reason,
        "issue": source_issue_summary(issue, raw_ranges),
    }


def make_force_drop_operation(
    start_word_id: int,
    end_word_id: int,
    reason: str,
    issue: dict[str, Any],
    raw_ranges: list[tuple[float, float, int]],
) -> dict[str, Any]:
    return {
        "type": "force_drop_words",
        "start_word_id": start_word_id,
        "end_word_id": end_word_id,
        "reason": reason,
        "affected_text": str(issue.get("affected_text", "")),
        "source_issue": source_issue_summary(issue, raw_ranges),
        "repair_action": "force_drop",
    }


def make_force_keep_words_operation(
    start_word_id: int,
    end_word_id: int,
    reason: str,
    issue: dict[str, Any],
    raw_ranges: list[tuple[float, float, int]],
    padding: float,
) -> dict[str, Any]:
    return {
        "type": "force_keep_words",
        "start_word_id": start_word_id,
        "end_word_id": end_word_id,
        "padding": padding,
        "reason": reason,
        "source_issue": source_issue_summary(issue, raw_ranges),
        "repair_action": "force_keep",
    }


def make_force_keep_interval_operation(
    raw_start: float,
    raw_end: float,
    reason: str,
    issue: dict[str, Any],
    raw_ranges: list[tuple[float, float, int]],
    padding: float,
    before_word_id: int,
    after_word_id: int,
) -> dict[str, Any]:
    return {
        "type": "force_keep_interval",
        "raw_start": seconds_to_timestamp(raw_start),
        "raw_end": seconds_to_timestamp(raw_end),
        "padding": padding,
        "before_word_id": before_word_id,
        "after_word_id": after_word_id,
        "reason": reason,
        "source_issue": source_issue_summary(issue, raw_ranges),
        "repair_action": "force_keep",
    }


def find_blocked_repair_drop(
    issue_words: list[dict[str, Any]],
    blocked_windows: list[dict[str, Any]],
) -> tuple[int, int] | None:
    issue_ids = {int(word["id"]) for word in issue_words}
    if not issue_ids:
        return None

    best_overlap = 0
    best_ids: list[int] = []
    for window in blocked_windows:
        raw_ids = [int(word_id) for word_id in window.get("word_ids", [])]
        if not raw_ids:
            continue
        overlap = len(issue_ids & set(raw_ids))
        if overlap > best_overlap:
            best_overlap = overlap
            best_ids = raw_ids

    if best_ids and best_overlap > 0:
        return min(best_ids), max(best_ids)
    return None


def find_repeated_phrase_drop(issue_words: list[dict[str, Any]]) -> tuple[int, int] | None:
    tokens = [raw_word_token(word) for word in issue_words]
    if not tokens:
        return None

    for phrase_length in (3, 2, 1):
        if len(tokens) < phrase_length * 2:
            continue
        for index in range(0, len(tokens) - phrase_length):
            phrase = tokens[index : index + phrase_length]
            if not all(phrase):
                continue
            if phrase_length == 1 and (len(phrase[0]) < CONTENT_MIN_LENGTH or phrase[0] in COMMON_SHORT_TOKENS):
                continue

            cursor = index + phrase_length
            repeat_count = 1
            while cursor + phrase_length <= len(tokens) and tokens[cursor : cursor + phrase_length] == phrase:
                repeat_count += 1
                cursor += phrase_length

            if repeat_count >= 2:
                drop_words = issue_words[index + phrase_length : cursor]
                return int(drop_words[0]["id"]), int(drop_words[-1]["id"])

    return None


def missing_raw_word_repairs(
    issue: dict[str, Any],
    issue_words: list[dict[str, Any]],
    raw_ranges: list[tuple[float, float, int]],
    word_padding: float,
) -> list[dict[str, Any]]:
    affected_tokens = Counter(tokenize(str(issue.get("affected_text", ""))))
    hint_tokens = set(
        tokenize(
            " ".join(
                [
                    str(issue.get("description", "")),
                    str(issue.get("suggested_action", "")),
                ]
            )
        )
    )
    affected_token_list = tokenize(str(issue.get("affected_text", "")))
    missing: list[dict[str, Any]] = []
    for word in issue_words:
        token = raw_word_token(word)
        if not token:
            continue
        if affected_tokens[token] > 0:
            affected_tokens[token] -= 1
            continue
        if len(token) < CONTENT_MIN_LENGTH or token in COMMON_SHORT_TOKENS:
            continue
        if token not in hint_tokens:
            continue
        if any(difflib.SequenceMatcher(None, token, affected).ratio() >= 0.84 for affected in affected_token_list):
            continue
        missing.append(word)

    if not missing:
        return []

    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = [missing[0]]
    for word in missing[1:]:
        previous_id = int(current[-1]["id"])
        current_id = int(word["id"])
        if current_id == previous_id + 1:
            current.append(word)
        else:
            groups.append(current)
            current = [word]
    groups.append(current)

    repairs: list[dict[str, Any]] = []
    for group in groups:
        words_text = " ".join(raw_word_text(word) for word in group)
        repairs.append(
            make_force_keep_words_operation(
                int(group[0]["id"]),
                int(group[-1]["id"]),
                f"QA reported a possible missing word; keep raw recognized word(s): {words_text}",
                issue,
                raw_ranges,
                word_padding,
            )
        )
    return repairs


def gap_keep_repairs(
    issue: dict[str, Any],
    issue_words: list[dict[str, Any]],
    raw_ranges: list[tuple[float, float, int]],
    gap_threshold: float,
    max_gap_repair: float,
    gap_padding: float,
) -> list[dict[str, Any]]:
    repairs: list[dict[str, Any]] = []
    affected_tokens = set(tokenize(str(issue.get("affected_text", ""))))
    for left, right in zip(issue_words, issue_words[1:]):
        left_token = raw_word_token(left)
        right_token = raw_word_token(right)
        if left_token not in affected_tokens or right_token not in affected_tokens:
            continue
        gap = word_start(right) - word_end(left)
        if gap < gap_threshold or gap > max_gap_repair:
            continue
        repairs.append(
            make_force_keep_interval_operation(
                word_end(left),
                word_start(right),
                (
                    "QA reported a logic gap; keep raw audio between neighboring "
                    f"words {left['id']} and {right['id']} in case Deepgram missed a connector."
                ),
                issue,
                raw_ranges,
                gap_padding,
                int(left["id"]),
                int(right["id"]),
            )
        )
    return repairs


def dangling_bridge_tail_repair(
    issue: dict[str, Any],
    context_words: list[dict[str, Any]],
    raw_ranges: list[tuple[float, float, int]],
) -> dict[str, Any] | None:
    if "bo inaczej" not in normalize_text(str(issue.get("affected_text", ""))):
        return None

    tokens = [raw_word_token(word) for word in context_words]
    for index in range(len(tokens) - 1):
        if tokens[index : index + 2] != ["bo", "inaczej"]:
            continue

        next_words = context_words[index + 2 : index + 6]
        if not any(raw_word_token(word) in {"drugi", "druga", "trzeci", "trzecia", "punkt"} for word in next_words):
            continue

        start_index = index
        search_floor = max(0, index - 8)
        for cursor in range(index - 1, search_floor - 1, -1):
            if raw_word_text(context_words[cursor]).endswith((".", "?", "!")):
                start_index = cursor + 1
                break
            start_index = cursor

        return make_force_drop_operation(
            int(context_words[start_index]["id"]),
            int(context_words[index + 1]["id"]),
            (
                "QA reported dangling 'bo inaczej' before the next section; "
                "drop the repeated tail and keep the earlier completed question."
            ),
            issue,
            raw_ranges,
        )

    return None


def direct_drop_repair(
    issue: dict[str, Any],
    issue_words: list[dict[str, Any]],
    raw_ranges: list[tuple[float, float, int]],
) -> dict[str, Any] | None:
    if not issue_words:
        return None
    if len(issue_words) > 12:
        return None

    tokens = [raw_word_token(word) for word in issue_words]
    if any(token in CONNECTOR_TOKENS for token in tokens):
        return None

    start = min(word_start(word) for word in issue_words)
    end = max(word_end(word) for word in issue_words)
    if end - start > 2.5:
        return None

    return make_force_drop_operation(
        int(issue_words[0]["id"]),
        int(issue_words[-1]["id"]),
        (
            "QA requested a high-confidence force_drop for a short off-topic/noise "
            "range mapped back to raw words."
        ),
        issue,
        raw_ranges,
    )


def repair_key(repair: dict[str, Any]) -> tuple[Any, ...]:
    repair_type = repair.get("type")
    if repair_type in {"force_drop_words", "force_keep_words"}:
        return (repair_type, repair.get("start_word_id"), repair.get("end_word_id"))
    return (repair_type, repair.get("raw_start"), repair.get("raw_end"))


def main() -> None:
    args = parse_args()
    quality_report = load_json(args.quality_report)
    edit_decisions = load_json(args.edit_decisions)
    raw_transcript = load_json(args.raw_transcript)
    edited_transcript = load_json(args.edited_transcript)

    timeline = build_timeline_from_keep_intervals(edit_decisions)
    raw_words = raw_transcript.get("words", [])
    edited_words = edited_transcript.get("words", [])
    if not timeline:
        fail("Edit decisions do not contain keep_intervals/timeline_map.")
    if not isinstance(raw_words, list) or not raw_words:
        fail("Raw transcript does not contain words.")

    min_rank = SEVERITY_RANK[args.min_severity]
    blocked_windows = edit_decisions.get("cut_planner_review", {}).get("blocked_windows", [])
    repairs: list[dict[str, Any]] = []
    skipped_issues: list[dict[str, Any]] = []
    mapped_issues: list[dict[str, Any]] = []
    manual_review_issues: list[dict[str, Any]] = []

    for issue_index, issue in enumerate(quality_report.get("issues", [])):
        severity = str(issue.get("severity", "low"))
        category = str(issue.get("issue_category", "other"))
        suggestion = normalize_repair_suggestion(issue)
        final_start = timestamp_to_seconds(str(issue.get("start", "00:00:00:000")))
        final_end = timestamp_to_seconds(str(issue.get("end", "00:00:00:000")))
        raw_ranges = raw_ranges_for_issue(issue, final_start, final_end, timeline)
        issue_words = words_in_ranges(raw_words, raw_ranges, center_only=True)
        context_words = words_in_ranges(raw_words, raw_ranges, context=0.65)
        mapped_issues.append(source_issue_summary(issue, raw_ranges))

        if SEVERITY_RANK.get(severity, 1) < min_rank:
            skipped_issues.append(skip_record(issue_index, f"severity_below_{args.min_severity}", issue, raw_ranges))
            continue

        action_skip_reason = repair_action_skip_reason(category, suggestion)
        if action_skip_reason:
            record = skip_record(issue_index, action_skip_reason, issue, raw_ranges)
            skipped_issues.append(record)
            if action_skip_reason in {"qa_requested_manual_review", "qa_requested_no_auto_repair"}:
                manual_review_issues.append(record)
            continue

        issue_repairs: list[dict[str, Any]] = []
        if category == "repetition" and suggestion["action"] == "force_drop":
            blocked_drop = find_blocked_repair_drop(context_words, blocked_windows)
            repeated_drop = blocked_drop or find_repeated_phrase_drop(context_words)
            if repeated_drop:
                issue_repairs.append(
                    make_force_drop_operation(
                        repeated_drop[0],
                        repeated_drop[1],
                        "QA requested high-confidence force_drop for this exact repeated raw range.",
                        issue,
                        raw_ranges,
                    )
                )

        if category in {"off_topic", "noise_or_setup"} and suggestion["action"] == "force_drop":
            direct_drop = direct_drop_repair(issue, issue_words, raw_ranges)
            if direct_drop:
                issue_repairs.append(direct_drop)

        if category in {"cut_word", "dangling_thought", "logic_gap"} and suggestion["action"] == "force_keep":
            bridge_repair = dangling_bridge_tail_repair(issue, context_words, raw_ranges)
            if bridge_repair:
                issue_repairs.append(bridge_repair)
            issue_repairs.extend(
                missing_raw_word_repairs(
                    issue,
                    issue_words,
                    raw_ranges,
                    args.word_padding,
                )
            )
            issue_repairs.extend(
                gap_keep_repairs(
                    issue,
                    issue_words,
                    raw_ranges,
                    args.gap_threshold,
                    args.max_gap_repair,
                    args.gap_padding,
                )
            )

        if issue_repairs:
            repairs.extend(issue_repairs)
        else:
            skipped_issues.append(skip_record(issue_index, "no_conservative_auto_repair", issue, raw_ranges))

    deduped_repairs: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for repair in repairs:
        key = repair_key(repair)
        if key in seen:
            continue
        deduped_repairs.append(repair)
        seen.add(key)

    payload = {
        "version": "1.0",
        "iteration": args.iteration,
        "max_auto_iterations": 3,
        "source_quality_report": str(args.quality_report),
        "source_edit_decisions": str(args.edit_decisions),
        "source_raw_transcript": str(args.raw_transcript),
        "source_edited_transcript": str(args.edited_transcript),
        "render_source_policy": "repair renders from the original raw video through edit_video.py --repair-plan",
        "repairs": deduped_repairs,
        "skipped_issues": skipped_issues,
        "manual_review_issues": manual_review_issues,
        "mapped_issues": mapped_issues,
        "notes": (
            "This plan is conservative. It only auto-applies actionable QA suggestions "
            "with raw_ranges: high-confidence force_drop for exact short ranges, "
            "force_keep for missing words/connectors, and short raw gap restoration."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Repair plan saved to: {args.output}")
    print(f"Repairs: {len(deduped_repairs)}")
    print(f"Skipped issues: {len(skipped_issues)}")
    print(f"Manual review issues: {len(manual_review_issues)}")


if __name__ == "__main__":
    main()
