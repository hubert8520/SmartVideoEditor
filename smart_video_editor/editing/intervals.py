"""Interval operations used by edit planning and rendering."""

from __future__ import annotations


Interval = tuple[float, float]


def merge_intervals(intervals: list[Interval], gap: float = 0.0) -> list[Interval]:
    """Merge overlapping or nearby intervals."""
    if not intervals:
        return []

    sorted_intervals = sorted((max(0.0, start), max(start, end)) for start, end in intervals)
    merged: list[Interval] = [sorted_intervals[0]]

    for start, end in sorted_intervals[1:]:
        last_start, last_end = merged[-1]
        if start - last_end <= gap:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))

    return merged


def subtract_intervals(base: list[Interval], drops: list[Interval]) -> list[Interval]:
    """Subtract drop intervals from base intervals."""
    result = merge_intervals(base)
    for drop_start, drop_end in merge_intervals(drops):
        next_result: list[Interval] = []
        for start, end in result:
            if drop_end <= start or drop_start >= end:
                next_result.append((start, end))
                continue
            if drop_start > start:
                next_result.append((start, drop_start))
            if drop_end < end:
                next_result.append((drop_end, end))
        result = next_result
    return [(start, end) for start, end in result if end > start]


def filter_short_intervals(intervals: list[Interval], min_duration: float) -> list[Interval]:
    """Remove intervals shorter than min_duration."""
    return [(start, end) for start, end in intervals if end - start >= min_duration]


def clamp_intervals(intervals: list[Interval], min_value: float = 0.0, max_value: float | None = None) -> list[Interval]:
    """Clamp intervals to a time range."""
    clamped: list[Interval] = []
    for start, end in intervals:
        start = max(min_value, start)
        end = max(start, end)
        if max_value is not None:
            start = min(start, max_value)
            end = min(end, max_value)
        if end > start:
            clamped.append((start, end))
    return clamped
