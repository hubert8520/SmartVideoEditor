#!/usr/bin/env python3
"""Generate a human-readable review brief for a non-technical editor."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
EDITED_DIR = PROJECT_ROOT / "edited"
BASE_QUALITY_REPORT = ARTIFACTS_DIR / "final_quality_report.json"
BASE_EDIT_DECISIONS = ARTIFACTS_DIR / "edit_decisions.json"
BASE_EDITED_VIDEO = EDITED_DIR / "edited_video.mp4"
VERSION_RE = re.compile(r"_v(\d+)\.")

SEVERITY_LABELS = {
    "high": "WYSOKI",
    "medium": "ŚREDNI",
    "low": "NISKI",
}

CATEGORY_LABELS = {
    "cut_word": "ucięte słowo",
    "repetition": "powtórzenie",
    "dangling_thought": "urwana myśl",
    "logic_gap": "problem logiczny",
    "off_topic": "off-topic",
    "noise_or_setup": "hałas/setup",
    "other": "inne",
}


def parse_args() -> argparse.Namespace:
    latest_quality = latest_versioned_path("final_quality_report", BASE_QUALITY_REPORT)
    iteration = iteration_from_path(latest_quality)
    default_edit_decisions = versioned_or_base("edit_decisions", "json", iteration, BASE_EDIT_DECISIONS)
    default_edited_video = versioned_or_base("edited_video", "mp4", iteration, BASE_EDITED_VIDEO, EDITED_DIR)
    suffix = f"_v{iteration}" if iteration is not None else ""

    parser = argparse.ArgumentParser(
        description=(
            "Create Markdown/CSV instructions for a non-technical video editor from "
            "a final quality report and edit timeline map."
        )
    )
    parser.add_argument("--quality-report", type=Path, default=latest_quality)
    parser.add_argument("--edit-decisions", type=Path, default=default_edit_decisions)
    parser.add_argument("--edited-video", type=Path, default=default_edited_video)
    parser.add_argument("--raw-video", type=Path, help="Raw video path. Defaults to source_video from edit decisions.")
    parser.add_argument("--output-md", type=Path, default=ARTIFACTS_DIR / f"editor_review{suffix}.md")
    parser.add_argument("--output-csv", type=Path, default=ARTIFACTS_DIR / f"editor_review{suffix}.csv")
    parser.add_argument(
        "--clip-margin",
        type=float,
        default=1.0,
        help="Seconds added before/after each issue when creating comparison clips. Default: 1.0.",
    )
    parser.add_argument(
        "--make-clips",
        action="store_true",
        help="Also create short edited/raw comparison clips for every issue.",
    )
    parser.add_argument(
        "--clips-dir",
        type=Path,
        default=ARTIFACTS_DIR / f"editor_review{suffix}_clips",
        help="Output folder for optional comparison clips.",
    )
    return parser.parse_args()


def latest_versioned_path(prefix: str, base: Path) -> Path:
    candidates: list[tuple[int, Path]] = []
    for path in ARTIFACTS_DIR.glob(f"{prefix}_v*.json"):
        iteration = iteration_from_path(path)
        if iteration is not None:
            candidates.append((iteration, path))
    if candidates:
        return sorted(candidates)[-1][1]
    return base


def iteration_from_path(path: Path) -> int | None:
    match = VERSION_RE.search(path.name)
    if not match:
        return None
    return int(match.group(1))


def versioned_or_base(
    stem: str,
    suffix: str,
    iteration: int | None,
    base: Path,
    folder: Path = ARTIFACTS_DIR,
) -> Path:
    if iteration is None:
        return base
    candidate = folder / f"{stem}_v{iteration}.{suffix}"
    return candidate if candidate.exists() else base


def fail(message: str) -> None:
    raise SystemExit(f"Error: {message}")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        fail(f"Missing file: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        fail(f"Expected JSON object: {path}")
    return data


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


def format_range(start: float, end: float) -> str:
    return f"{seconds_to_timestamp(start)} - {seconds_to_timestamp(end)}"


def clip_range(start: float, end: float, margin: float) -> tuple[float, float]:
    return max(0.0, start - margin), max(start, end + margin)


def build_timeline_from_keep_intervals(decisions: dict[str, Any]) -> list[dict[str, Any]]:
    timeline_map = decisions.get("timeline_map")
    if isinstance(timeline_map, list) and timeline_map:
        return timeline_map

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
) -> list[dict[str, Any]]:
    ranges: list[dict[str, Any]] = []
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
        ranges.append(
            {
                "timeline_id": int(item.get("id", len(ranges))),
                "raw_start_seconds": mapped_start,
                "raw_end_seconds": mapped_end,
                "raw_range": format_range(mapped_start, mapped_end),
            }
        )
    return ranges


def resolve_ffmpeg_executable() -> str:
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg

    try:
        import imageio_ffmpeg
    except ImportError:
        fail("Missing ffmpeg. Install dependencies with: pip install -r requirements.txt")

    return imageio_ffmpeg.get_ffmpeg_exe()


def render_clip(
    ffmpeg: str,
    input_path: Path,
    output_path: Path,
    start: float,
    end: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.05, end - start)
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-i",
            str(input_path),
            "-t",
            f"{duration:.3f}",
            "-map",
            "0:v:0?",
            "-map",
            "0:a:0?",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            str(output_path),
        ],
        check=True,
    )


def relative_label(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def severity_label(issue: dict[str, Any]) -> str:
    return SEVERITY_LABELS.get(str(issue.get("severity", "")).lower(), str(issue.get("severity", "")))


def category_label(issue: dict[str, Any]) -> str:
    category = str(issue.get("issue_category", ""))
    return CATEGORY_LABELS.get(category, category)


def issue_rows(
    quality_report: dict[str, Any],
    timeline: list[dict[str, Any]],
    edited_video: Path,
    raw_video: Path,
    clip_margin: float,
    make_clips: bool,
    clips_dir: Path,
) -> list[dict[str, Any]]:
    ffmpeg = resolve_ffmpeg_executable() if make_clips else ""
    rows: list[dict[str, Any]] = []
    for index, issue in enumerate(quality_report.get("issues", []), start=1):
        final_start = timestamp_to_seconds(str(issue["start"]))
        final_end = timestamp_to_seconds(str(issue["end"]))
        edited_clip_start, edited_clip_end = clip_range(final_start, final_end, clip_margin)
        raw_ranges = map_final_range_to_raw_ranges(final_start, final_end, timeline)
        raw_compare_ranges = []
        raw_clip_paths = []

        if make_clips:
            edited_clip_path = clips_dir / f"issue_{index:02d}_edited.mp4"
            render_clip(ffmpeg, edited_video, edited_clip_path, edited_clip_start, edited_clip_end)
        else:
            edited_clip_path = None

        for raw_index, raw_range in enumerate(raw_ranges, start=1):
            raw_start = float(raw_range["raw_start_seconds"])
            raw_end = float(raw_range["raw_end_seconds"])
            compare_start, compare_end = clip_range(raw_start, raw_end, clip_margin)
            raw_compare_ranges.append(format_range(compare_start, compare_end))
            if make_clips:
                raw_clip_path = clips_dir / f"issue_{index:02d}_raw_{raw_index}.mp4"
                render_clip(ffmpeg, raw_video, raw_clip_path, compare_start, compare_end)
                raw_clip_paths.append(raw_clip_path)

        rows.append(
            {
                "index": index,
                "priority": severity_label(issue),
                "category": category_label(issue),
                "edited_range": format_range(final_start, final_end),
                "edited_compare_range": format_range(edited_clip_start, edited_clip_end),
                "raw_ranges": raw_ranges,
                "raw_compare_ranges": raw_compare_ranges,
                "description": str(issue.get("description", "")),
                "affected_text": str(issue.get("affected_text", "")),
                "suggested_action": str(issue.get("suggested_action", "")),
                "edited_clip_path": edited_clip_path,
                "raw_clip_paths": raw_clip_paths,
            }
        )
    return rows


def write_markdown(
    path: Path,
    rows: list[dict[str, Any]],
    quality_report: dict[str, Any],
    edited_video: Path,
    raw_video: Path,
    make_clips: bool,
) -> None:
    lines = [
        "# Brief montażowy",
        "",
        f"Status QA: **{quality_report.get('status', '')}**",
        "",
        f"Film edytowany: `{relative_label(edited_video)}`",
        f"Film raw: `{relative_label(raw_video)}`",
        "",
        "Instrukcja dla montażysty:",
        "",
        "1. Otwórz film edytowany i przejdź do czasu z pola `Film edytowany`.",
        "2. Otwórz raw i przejdź do czasu z pola `Raw do porównania`.",
        "3. Porównaj, czy w edicie nie zniknęło słowo, łącznik, sens zdania albo czy nie zostało powtórzenie.",
        "4. Jeśli są klipy porównawcze, możesz pracować na nich zamiast ręcznie szukać timecode'ów.",
        "",
        f"Liczba miejsc do sprawdzenia: **{len(rows)}**",
        "",
    ]
    notes = str(quality_report.get("overall_notes", "")).strip()
    if notes:
        lines.extend(["## Ogólna uwaga QA", "", notes, ""])

    for row in rows:
        lines.extend(
            [
                f"## {row['index']}. Priorytet {row['priority']} - {row['category']}",
                "",
                f"Film edytowany: `{row['edited_range']}`",
                f"Do odsłuchu w edicie z marginesem: `{row['edited_compare_range']}`",
                "",
                "Raw do porównania:",
            ]
        )
        if row["raw_compare_ranges"]:
            for raw_index, raw_range in enumerate(row["raw_compare_ranges"], start=1):
                timeline_id = row["raw_ranges"][raw_index - 1]["timeline_id"]
                lines.append(f"- `{raw_range}` (fragment raw/timeline #{timeline_id})")
        else:
            lines.append("- brak mapowania na raw; sprawdzić ręcznie")

        if make_clips:
            lines.extend(["", "Klipy porównawcze:"])
            if row["edited_clip_path"]:
                lines.append(f"- edit: `{relative_label(row['edited_clip_path'])}`")
            for raw_clip_path in row["raw_clip_paths"]:
                lines.append(f"- raw: `{relative_label(raw_clip_path)}`")

        lines.extend(
            [
                "",
                "Co brzmi podejrzanie:",
                "",
                row["description"],
                "",
                "Podejrzany tekst:",
                "",
                f"> {row['affected_text']}",
                "",
                "Sugestia dla montażysty:",
                "",
                row["suggested_action"],
                "",
            ]
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], edited_video: Path, raw_video: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "nr",
                "priorytet",
                "kategoria",
                "film_edytowany_czas",
                "film_edytowany_do_odsluchu",
                "raw_do_porownania",
                "co_brzmi_podejrzanie",
                "podejrzany_tekst",
                "sugestia",
                "plik_edytowany",
                "plik_raw",
            ],
            delimiter=";",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "nr": row["index"],
                    "priorytet": row["priority"],
                    "kategoria": row["category"],
                    "film_edytowany_czas": row["edited_range"],
                    "film_edytowany_do_odsluchu": row["edited_compare_range"],
                    "raw_do_porownania": " | ".join(row["raw_compare_ranges"]),
                    "co_brzmi_podejrzanie": row["description"],
                    "podejrzany_tekst": row["affected_text"],
                    "sugestia": row["suggested_action"],
                    "plik_edytowany": relative_label(edited_video),
                    "plik_raw": relative_label(raw_video),
                }
            )


def main() -> None:
    args = parse_args()
    quality_report = load_json(args.quality_report)
    edit_decisions = load_json(args.edit_decisions)
    timeline = build_timeline_from_keep_intervals(edit_decisions)
    if not timeline:
        fail("Edit decisions do not contain timeline_map or keep_intervals.")

    raw_video = args.raw_video or Path(str(edit_decisions.get("source_video", "")))
    if not raw_video.exists():
        fail(f"Raw video does not exist: {raw_video}")
    if not args.edited_video.exists():
        fail(f"Edited video does not exist: {args.edited_video}")

    rows = issue_rows(
        quality_report,
        timeline,
        args.edited_video,
        raw_video,
        args.clip_margin,
        args.make_clips,
        args.clips_dir,
    )
    write_markdown(args.output_md, rows, quality_report, args.edited_video, raw_video, args.make_clips)
    write_csv(args.output_csv, rows, args.edited_video, raw_video)

    print(f"Editor brief saved to: {args.output_md}")
    print(f"Editor CSV saved to: {args.output_csv}")
    if args.make_clips:
        print(f"Comparison clips saved to: {args.clips_dir}")


if __name__ == "__main__":
    main()
