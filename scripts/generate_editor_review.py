#!/usr/bin/env python3
"""Generate a human-readable review brief for a non-technical editor."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from smart_video_editor.reporting import editor_review as review_reporting  # noqa: E402
from smart_video_editor.reporting.timeline import timeline_from_edit_decisions  # noqa: E402

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
EDITED_DIR = PROJECT_ROOT / "edited"
BASE_QUALITY_REPORT = ARTIFACTS_DIR / "final_quality_report.json"
BASE_EDIT_DECISIONS = ARTIFACTS_DIR / "edit_decisions.json"
BASE_EDITED_VIDEO = EDITED_DIR / "edited_video.mp4"
VERSION_RE = re.compile(r"_v(\d+)\.")


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


def attach_comparison_clips(
    rows: list[dict[str, Any]],
    edited_video: Path,
    raw_video: Path,
    clip_margin: float,
    clips_dir: Path,
) -> None:
    ffmpeg = resolve_ffmpeg_executable()
    for row in rows:
        index = int(row["index"])
        edited_clip_start, edited_clip_end = review_reporting.clip_range(
            float(row["final_start_seconds"]),
            float(row["final_end_seconds"]),
            clip_margin,
        )
        edited_clip_path = clips_dir / f"issue_{index:02d}_edited.mp4"
        render_clip(ffmpeg, edited_video, edited_clip_path, edited_clip_start, edited_clip_end)
        row["edited_clip_path"] = relative_label(edited_clip_path)

        raw_clip_paths = []
        for raw_index, raw_range in enumerate(row["raw_ranges"], start=1):
            compare_start, compare_end = review_reporting.clip_range(
                float(raw_range["raw_start_seconds"]),
                float(raw_range["raw_end_seconds"]),
                clip_margin,
            )
            raw_clip_path = clips_dir / f"issue_{index:02d}_raw_{raw_index}.mp4"
            render_clip(ffmpeg, raw_video, raw_clip_path, compare_start, compare_end)
            raw_clip_paths.append(relative_label(raw_clip_path))
        row["raw_clip_paths"] = raw_clip_paths


def main() -> None:
    args = parse_args()
    quality_report = load_json(args.quality_report)
    edit_decisions = load_json(args.edit_decisions)
    timeline = timeline_from_edit_decisions(edit_decisions)
    if not timeline:
        fail("Edit decisions do not contain timeline_map or keep_intervals.")

    raw_video = args.raw_video or Path(str(edit_decisions.get("source_video", "")))
    if not raw_video.exists():
        fail(f"Raw video does not exist: {raw_video}")
    if not args.edited_video.exists():
        fail(f"Edited video does not exist: {args.edited_video}")

    rows = review_reporting.build_editor_review_rows(
        quality_report,
        edit_decisions,
        clip_margin=args.clip_margin,
    )
    if args.make_clips:
        attach_comparison_clips(rows, args.edited_video, raw_video, args.clip_margin, args.clips_dir)

    review_reporting.write_editor_review_markdown(
        args.output_md,
        rows,
        quality_report,
        relative_label(args.edited_video),
        relative_label(raw_video),
        make_clips=args.make_clips,
        planner_evidence=review_reporting.planner_evidence_rows(edit_decisions),
    )
    review_reporting.write_editor_review_csv(
        args.output_csv,
        rows,
        relative_label(args.edited_video),
        relative_label(raw_video),
    )

    print(f"Editor brief saved to: {args.output_md}")
    print(f"Editor CSV saved to: {args.output_csv}")
    if args.make_clips:
        print(f"Comparison clips saved to: {args.clips_dir}")


if __name__ == "__main__":
    main()
