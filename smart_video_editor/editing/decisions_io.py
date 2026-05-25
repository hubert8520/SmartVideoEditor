"""Edit-decision artifact writing helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from smart_video_editor.timecode import seconds_to_timestamp


def build_timeline_map(keep_intervals: list[tuple[float, float]]) -> list[dict[str, object]]:
    timeline: list[dict[str, object]] = []
    cursor = 0.0
    for index, (raw_start, raw_end) in enumerate(keep_intervals):
        duration = raw_end - raw_start
        final_start = cursor
        final_end = cursor + duration
        timeline.append(
            {
                "id": index,
                "raw_start": seconds_to_timestamp(raw_start),
                "raw_end": seconds_to_timestamp(raw_end),
                "final_start": seconds_to_timestamp(final_start),
                "final_end": seconds_to_timestamp(final_end),
                "duration_seconds": round(duration, 6),
            }
        )
        cursor = final_end
    return timeline


def write_decisions(
    transcript_path: Path,
    transcript_metadata: dict[str, object],
    media_path: Path,
    output_path: Path,
    edit_decisions_path: Path,
    duration: float,
    silences: list[tuple[float, float]],
    keep_source: str,
    llm_summary: Any,
    repair_summary: Any,
    keep_intervals: list[tuple[float, float]],
    drop_windows: list[tuple[float, float]],
    partial_drop_windows: list[Any],
    review_windows: list[Any],
    planner_result: Any,
    thought_blocks: list[Any],
    short_block_trim_records: list[dict[str, object]],
    entries: list[Any],
    words: list[Any],
    padding: float,
    keep_settings: Any,
    word_gap_merge: float,
    cut_safety_margin: float,
    silence_snap_window: float,
    dry_run: bool,
) -> None:
    payload = {
        "source_video": str(media_path),
        "transcript": str(transcript_path),
        "transcript_metadata": transcript_metadata,
        "output_video": str(output_path),
        "duration": seconds_to_timestamp(duration),
        "padding_seconds": padding,
        "keep_planner": {
            "use_word_mask": keep_settings.use_word_mask,
            "word_head_padding": keep_settings.word_head_padding,
            "word_tail_padding": keep_settings.word_tail_padding,
            "short_block_tail_padding": keep_settings.short_block_tail_padding,
            "short_block_max_words": keep_settings.short_block_max_words,
            "short_block_silence_trim": keep_settings.short_block_silence_trim,
            "short_block_silence_min_duration": keep_settings.short_block_silence_min_duration,
            "short_block_silence_window": keep_settings.short_block_silence_window,
            "short_block_min_spoken_before_trim": keep_settings.short_block_min_spoken_before_trim,
            "short_block_silence_trims": short_block_trim_records,
        },
        "keep_source": keep_source,
        "word_count": len(words),
        "segment_count": len(entries),
        "cut_planner": {
            "word_gap_merge": word_gap_merge,
            "cut_safety_margin": cut_safety_margin,
            "silence_snap_window": silence_snap_window,
        },
        "thought_blocks": [
            {
                "id": block.id,
                "start": seconds_to_timestamp(block.start),
                "end": seconds_to_timestamp(block.end),
                "role": block.role,
                "text": block.text,
                "word_ids": block.word_ids,
            }
            for block in thought_blocks
        ],
        "cut_planner_review": {
            "applied_windows": planner_result.applied_windows,
            "blocked_windows": planner_result.blocked_windows,
            "review_windows": getattr(planner_result, "review_windows", []),
            "boundary_issues": planner_result.boundary_issues,
            "simulated_text": planner_result.simulated_text,
        },
        "dry_run": dry_run,
        "llm_decisions": {
            "path": llm_summary.path,
            "status": llm_summary.status,
            "min_confidence": llm_summary.min_confidence,
            "applied_drop_ranges": llm_summary.applied_drop_ranges,
            "skipped_drop_ranges": llm_summary.skipped_drop_ranges,
            "review_ranges": llm_summary.review_ranges,
            "keep_notes": llm_summary.keep_notes,
            "thought_blocks": llm_summary.thought_blocks,
        },
        "repair_plan": {
            "path": repair_summary.path,
            "status": repair_summary.status,
            "repairs": repair_summary.repairs,
            "skipped_repairs": repair_summary.skipped_repairs,
            "forced_keep_intervals": repair_summary.forced_keep_records,
        },
        "removed_entries": [
            {
                "timestamp": seconds_to_timestamp(entry.timestamp),
                "end": seconds_to_timestamp(entry.end),
                "transcription": entry.text,
                "reasons": entry.reasons,
            }
            for entry in entries
            if entry.drop
        ],
        "review_entries": [
            {
                "timestamp": seconds_to_timestamp(entry.timestamp),
                "end": seconds_to_timestamp(entry.end),
                "transcription": entry.text,
                "reasons": entry.review_reasons,
            }
            for entry in entries
            if entry.review_reasons
        ],
        "partial_drop_windows": [
            {
                "start": seconds_to_timestamp(window.start),
                "end": seconds_to_timestamp(window.end),
                "source_text": window.source_text,
                "reason": window.reason,
                "word_ids": window.word_ids,
                "source": window.source,
                "force": window.force,
            }
            for window in partial_drop_windows
        ],
        "review_windows": [
            {
                "start": seconds_to_timestamp(window.start),
                "end": seconds_to_timestamp(window.end),
                "source_text": window.source_text,
                "reason": window.reason,
                "word_ids": window.word_ids,
                "source": window.source,
            }
            for window in review_windows
        ],
        "silences": [
            {
                "start": seconds_to_timestamp(start),
                "end": seconds_to_timestamp(end),
            }
            for start, end in silences
        ],
        "drop_windows": [
            {
                "start": seconds_to_timestamp(start),
                "end": seconds_to_timestamp(end),
            }
            for start, end in drop_windows
        ],
        "keep_intervals": [
            {
                "start": seconds_to_timestamp(start),
                "end": seconds_to_timestamp(end),
            }
            for start, end in keep_intervals
        ],
        "timeline_map": build_timeline_map(keep_intervals),
    }
    edit_decisions_path.parent.mkdir(parents=True, exist_ok=True)
    edit_decisions_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
