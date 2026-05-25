"""Final-to-raw timeline mapping helpers."""

from __future__ import annotations

from typing import Any

from smart_video_editor.domain.models import TimelineMapItem, TranscriptWord
from smart_video_editor.timecode import seconds_to_timestamp, timestamp_to_seconds
from smart_video_editor.transcription.normalization import normalize_words_payload


def _read_time(item: dict[str, Any], *keys: str) -> float:
    for key in keys:
        if key not in item:
            continue
        value = item[key]
        if isinstance(value, int | float):
            return float(value)
        return timestamp_to_seconds(str(value))
    raise KeyError(f"Missing time field; expected one of: {', '.join(keys)}")


def timeline_from_edit_decisions(decisions: dict[str, Any]) -> tuple[TimelineMapItem, ...]:
    """Build a timeline map from edit_decisions.json with legacy fallbacks."""
    timeline_payload = decisions.get("timeline_map")
    if isinstance(timeline_payload, list) and timeline_payload:
        items: list[TimelineMapItem] = []
        for index, item in enumerate(timeline_payload):
            if not isinstance(item, dict):
                continue
            items.append(
                TimelineMapItem(
                    id=int(item.get("id", index)),
                    raw_start=_read_time(item, "raw_start", "start"),
                    raw_end=_read_time(item, "raw_end", "end"),
                    final_start=_read_time(item, "final_start"),
                    final_end=_read_time(item, "final_end"),
                )
            )
        return tuple(items)

    keep_intervals = decisions.get("keep_intervals", [])
    cursor = 0.0
    items = []
    for index, item in enumerate(keep_intervals):
        if not isinstance(item, dict):
            continue
        raw_start = _read_time(item, "raw_start", "start")
        raw_end = _read_time(item, "raw_end", "end")
        duration = max(0.0, raw_end - raw_start)
        items.append(
            TimelineMapItem(
                id=index,
                raw_start=raw_start,
                raw_end=raw_end,
                final_start=cursor,
                final_end=cursor + duration,
            )
        )
        cursor += duration
    return tuple(items)


def map_final_range_to_raw_ranges(
    final_start: float,
    final_end: float,
    timeline: tuple[TimelineMapItem, ...] | list[TimelineMapItem],
) -> list[dict[str, Any]]:
    """Map a final-video time range onto one or more raw-video ranges."""
    if final_end < final_start:
        final_start, final_end = final_end, final_start

    ranges: list[dict[str, Any]] = []
    for item in timeline:
        overlap_start = max(final_start, item.final_start)
        overlap_end = min(final_end, item.final_end)
        if overlap_end <= overlap_start:
            continue

        raw_start = item.raw_start + (overlap_start - item.final_start)
        raw_end = item.raw_start + (overlap_end - item.final_start)
        ranges.append(
            {
                "timeline_id": item.id,
                "final_start": seconds_to_timestamp(overlap_start),
                "final_end": seconds_to_timestamp(overlap_end),
                "final_start_seconds": round(overlap_start, 6),
                "final_end_seconds": round(overlap_end, 6),
                "raw_start": seconds_to_timestamp(raw_start),
                "raw_end": seconds_to_timestamp(raw_end),
                "raw_start_seconds": round(raw_start, 6),
                "raw_end_seconds": round(raw_end, 6),
                "duration_seconds": round(raw_end - raw_start, 6),
            }
        )
    return ranges


def map_final_time_to_raw_time(
    final_time: float,
    timeline: tuple[TimelineMapItem, ...] | list[TimelineMapItem],
) -> float | None:
    for item in timeline:
        if item.final_start <= final_time <= item.final_end:
            return item.raw_start + (final_time - item.final_start)
    return None


def raw_range_tuples(raw_ranges: list[dict[str, Any]]) -> list[tuple[float, float, int]]:
    tuples: list[tuple[float, float, int]] = []
    for index, item in enumerate(raw_ranges):
        try:
            if "raw_start_seconds" in item:
                start = float(item["raw_start_seconds"])
            else:
                start = timestamp_to_seconds(str(item["raw_start"]))
            if "raw_end_seconds" in item:
                end = float(item["raw_end_seconds"])
            else:
                end = timestamp_to_seconds(str(item["raw_end"]))
        except (KeyError, TypeError, ValueError):
            continue
        tuples.append((start, end, int(item.get("timeline_id", index))))
    return tuples


def normalize_transcript_words(transcript: dict[str, Any]) -> list[TranscriptWord]:
    words = transcript.get("words", [])
    if not isinstance(words, list):
        return []
    return normalize_words_payload(words)


def words_in_raw_ranges(
    words: list[TranscriptWord],
    raw_ranges: list[dict[str, Any]],
    *,
    context: float = 0.0,
    center_only: bool = False,
) -> list[TranscriptWord]:
    selected: list[TranscriptWord] = []
    seen: set[int] = set()
    for start, end, _ in raw_range_tuples(raw_ranges):
        for word in words:
            if word.id in seen:
                continue
            if center_only:
                midpoint = (word.timestamp + word.end) / 2
                in_range = start - context <= midpoint <= end + context
            else:
                in_range = word.end > start - context and word.timestamp < end + context
            if in_range:
                selected.append(word)
                seen.add(word.id)
    return sorted(selected, key=lambda item: item.id)


def compact_words_for_report(words: list[TranscriptWord]) -> list[dict[str, Any]]:
    return [
        {
            "id": word.id,
            "start": seconds_to_timestamp(word.timestamp),
            "end": seconds_to_timestamp(word.end),
            "word": word.text,
        }
        for word in words
    ]
