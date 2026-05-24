"""Edit decision list and timeline mapping helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from smart_video_editor.timecode import seconds_to_timestamp, timestamp_to_seconds


@dataclass(frozen=True, slots=True)
class KeepInterval:
    raw_start: float
    raw_end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.raw_end - self.raw_start)


@dataclass(frozen=True, slots=True)
class TimelineItem:
    id: int
    raw_start: float
    raw_end: float
    final_start: float
    final_end: float


@dataclass(frozen=True, slots=True)
class EditDecisionList:
    keep_intervals: tuple[KeepInterval, ...]
    version: str = "1.0"

    def timeline_map(self) -> tuple[TimelineItem, ...]:
        cursor = 0.0
        items: list[TimelineItem] = []
        for index, keep in enumerate(self.keep_intervals):
            duration = keep.duration
            items.append(
                TimelineItem(
                    id=index,
                    raw_start=keep.raw_start,
                    raw_end=keep.raw_end,
                    final_start=cursor,
                    final_end=cursor + duration,
                )
            )
            cursor += duration
        return tuple(items)

    def map_final_time_to_raw(self, final_time: float) -> float | None:
        for item in self.timeline_map():
            if item.final_start <= final_time <= item.final_end:
                return item.raw_start + (final_time - item.final_start)
        return None

    def to_json_dict(self) -> dict[str, Any]:
        timeline = self.timeline_map()
        return {
            "version": self.version,
            "keep_intervals": [
                {
                    "raw_start": seconds_to_timestamp(interval.raw_start),
                    "raw_end": seconds_to_timestamp(interval.raw_end),
                    "duration_seconds": round(interval.duration, 6),
                }
                for interval in self.keep_intervals
            ],
            "timeline_map": [
                {
                    "id": item.id,
                    "raw_start": seconds_to_timestamp(item.raw_start),
                    "raw_end": seconds_to_timestamp(item.raw_end),
                    "final_start": seconds_to_timestamp(item.final_start),
                    "final_end": seconds_to_timestamp(item.final_end),
                    "duration_seconds": round(item.final_end - item.final_start, 6),
                }
                for item in timeline
            ],
        }

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> "EditDecisionList":
        intervals = []
        for item in payload.get("keep_intervals", []):
            intervals.append(
                KeepInterval(
                    raw_start=timestamp_to_seconds(str(item["raw_start"])),
                    raw_end=timestamp_to_seconds(str(item["raw_end"])),
                )
            )
        return cls(keep_intervals=tuple(intervals), version=str(payload.get("version", "1.0")))
