"""Actionable QA report enrichment."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from smart_video_editor.reporting.timeline import (
    compact_words_for_report,
    map_final_range_to_raw_ranges,
    normalize_transcript_words,
    timeline_from_edit_decisions,
    words_in_raw_ranges,
)
from smart_video_editor.timecode import seconds_to_timestamp, timestamp_to_seconds


QA_ACTIONS = {"force_keep", "force_drop", "manual_review", "no_auto_repair"}
AUTO_DROP_ACTIONS = {"force_drop"}
AUTO_KEEP_ACTIONS = {"force_keep"}
DEFAULT_ACTION_BY_CATEGORY = {
    "cut_word": "force_keep",
    "logic_gap": "force_keep",
    "dangling_thought": "manual_review",
    "repetition": "force_drop",
    "off_topic": "manual_review",
    "noise_or_setup": "manual_review",
}


def _safe_timestamp(value: Any, default: str = "00:00:00:000") -> str:
    text = str(value or default)
    try:
        timestamp_to_seconds(text)
    except ValueError:
        return default
    return text


def _fallback_repair_suggestion(issue: dict[str, Any]) -> dict[str, Any]:
    category = str(issue.get("issue_category", "other"))
    action = DEFAULT_ACTION_BY_CATEGORY.get(category, "manual_review")
    confidence = "high" if action == "force_drop" and category == "repetition" else "medium"
    return {
        "action": action,
        "confidence": confidence,
        "rationale": str(issue.get("suggested_action", "Legacy QA issue without repair_suggestion.")),
        "requires_manual_review": action in {"manual_review", "no_auto_repair"},
    }


def normalize_repair_suggestion(issue: dict[str, Any]) -> dict[str, Any]:
    raw = issue.get("repair_suggestion")
    if not isinstance(raw, dict):
        return _fallback_repair_suggestion(issue)

    action = str(raw.get("action", "")).strip()
    if action not in QA_ACTIONS:
        action = "manual_review"
    confidence = str(raw.get("confidence", "low")).strip()
    if confidence not in {"low", "medium", "high"}:
        confidence = "low"

    return {
        "action": action,
        "confidence": confidence,
        "rationale": str(raw.get("rationale", "")).strip() or str(issue.get("suggested_action", "")),
        "requires_manual_review": bool(raw.get("requires_manual_review", action in {"manual_review", "no_auto_repair"})),
    }


def _downgrade_to_review(suggestion: dict[str, Any], reason: str) -> dict[str, Any]:
    downgraded = dict(suggestion)
    downgraded["original_action"] = suggestion.get("action")
    downgraded["action"] = "manual_review"
    downgraded["requires_manual_review"] = True
    rationale = str(suggestion.get("rationale", "")).strip()
    downgraded["rationale"] = f"{rationale} {reason}".strip()
    return downgraded


def enrich_quality_report(
    report: dict[str, Any],
    *,
    edit_decisions: dict[str, Any] | None = None,
    raw_transcript: dict[str, Any] | None = None,
    edit_decisions_path: Path | None = None,
    raw_transcript_path: Path | None = None,
    raw_context_seconds: float = 0.45,
) -> dict[str, Any]:
    """Add raw-time mapping and conservative repair suggestions to a QA report."""
    enriched = deepcopy(report)
    timeline = timeline_from_edit_decisions(edit_decisions or {}) if edit_decisions else tuple()
    raw_words = normalize_transcript_words(raw_transcript or {}) if raw_transcript else []

    mapped_count = 0
    issues = enriched.get("issues", [])
    if not isinstance(issues, list):
        enriched["issues"] = []
        issues = []

    for issue_index, issue in enumerate(issues):
        if not isinstance(issue, dict):
            continue

        final_start_text = _safe_timestamp(issue.get("start"))
        final_end_text = _safe_timestamp(issue.get("end"), final_start_text)
        final_start = timestamp_to_seconds(final_start_text)
        final_end = timestamp_to_seconds(final_end_text)
        if final_end < final_start:
            final_start, final_end = final_end, final_start
            final_start_text = seconds_to_timestamp(final_start)
            final_end_text = seconds_to_timestamp(final_end)

        raw_ranges = map_final_range_to_raw_ranges(final_start, final_end, timeline) if timeline else []
        if raw_ranges:
            mapped_count += 1

        suggestion = normalize_repair_suggestion(issue)
        if suggestion["action"] == "force_drop" and suggestion["confidence"] != "high":
            suggestion = _downgrade_to_review(
                suggestion,
                "Auto force_drop requires high QA confidence.",
            )
        if suggestion["action"] in AUTO_DROP_ACTIONS | AUTO_KEEP_ACTIONS and not raw_ranges:
            suggestion = _downgrade_to_review(
                suggestion,
                "No final-to-raw timeline mapping is available, so repair must be reviewed manually.",
            )

        context_words = words_in_raw_ranges(
            raw_words,
            raw_ranges,
            context=raw_context_seconds,
        )
        issue["issue_index"] = issue_index
        issue["final_range"] = {
            "start": final_start_text,
            "end": final_end_text,
            "start_seconds": round(final_start, 6),
            "end_seconds": round(final_end, 6),
        }
        issue["raw_ranges"] = raw_ranges
        issue["raw_context"] = compact_words_for_report(context_words)
        issue["repair_suggestion"] = suggestion
        issue["actionability"] = {
            "mapped_to_raw": bool(raw_ranges),
            "timeline_ids": [item["timeline_id"] for item in raw_ranges],
            "requires_manual_review": bool(suggestion["requires_manual_review"]),
        }

    enriched["action_contract"] = {
        "version": "1.0",
        "repair_policy": (
            "Repair planner may only auto-apply force_keep/force_drop suggestions "
            "that have raw_ranges and pass conservative local validation."
        ),
        "edit_decisions_path": str(edit_decisions_path) if edit_decisions_path else "",
        "raw_transcript_path": str(raw_transcript_path) if raw_transcript_path else "",
        "mapped_issue_count": mapped_count,
        "issue_count": len([issue for issue in issues if isinstance(issue, dict)]),
    }
    return enriched


def quality_mapping_context_for_prompt(
    *,
    edit_decisions: dict[str, Any] | None,
    raw_transcript: dict[str, Any] | None,
) -> dict[str, Any]:
    timeline = timeline_from_edit_decisions(edit_decisions or {}) if edit_decisions else tuple()
    raw_words = normalize_transcript_words(raw_transcript or {}) if raw_transcript else []
    return {
        "timeline_available": bool(timeline),
        "timeline_item_count": len(timeline),
        "raw_words_available": bool(raw_words),
        "raw_word_count": len(raw_words),
        "mapping_note": (
            "Do not invent raw timestamps. The script maps final issue ranges to "
            "raw_ranges after your response using edit_decisions.timeline_map."
        ),
    }

