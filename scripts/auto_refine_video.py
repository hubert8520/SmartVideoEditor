#!/usr/bin/env python3
"""Run bounded QA -> repair -> render iterations."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(
    os.environ.get("SMART_VIDEO_EDITOR_WORKSPACE", Path(__file__).resolve().parents[1])
).expanduser().resolve()
SCRIPTS_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
EDITED_DIR = PROJECT_ROOT / "edited"
DEFAULT_INITIAL_QUALITY_REPORT = ARTIFACTS_DIR / "final_quality_report.json"
DEFAULT_INITIAL_EDIT_DECISIONS = ARTIFACTS_DIR / "edit_decisions.json"
DEFAULT_INITIAL_EDITED_TRANSCRIPT = ARTIFACTS_DIR / "edited_transcription.json"
DEFAULT_RAW_TRANSCRIPT = ARTIFACTS_DIR / "raw_transcription.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Start from an existing quality report and run bounded repair/render/QA "
            "iterations from the original raw media."
        )
    )
    parser.add_argument("--quality-report", type=Path, default=DEFAULT_INITIAL_QUALITY_REPORT)
    parser.add_argument("--edit-decisions", type=Path, default=DEFAULT_INITIAL_EDIT_DECISIONS)
    parser.add_argument("--edited-transcript", type=Path, default=DEFAULT_INITIAL_EDITED_TRANSCRIPT)
    parser.add_argument("--raw-transcript", type=Path, default=DEFAULT_RAW_TRANSCRIPT)
    parser.add_argument(
        "--start-iteration",
        type=int,
        default=2,
        help="First repaired iteration number. Default: 2.",
    )
    parser.add_argument(
        "--max-iteration",
        type=int,
        default=3,
        help="Last automatic iteration number. Default: 3.",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.1,
        help="Padding passed to edit_video.py. Default: 0.1.",
    )
    parser.add_argument(
        "--quality-language",
        default="pl",
        help="Language passed to post-render QA. Default: pl.",
    )
    parser.add_argument(
        "--min-severity",
        choices=("low", "medium", "high"),
        default="medium",
        help="Lowest QA issue severity repaired automatically. Default: medium.",
    )
    parser.add_argument("--video-name", help="Optional media file name inside raw/.")
    return parser.parse_args()


def fail(message: str) -> None:
    raise SystemExit(f"Error: {message}")


def load_status(path: Path) -> str:
    if not path.exists():
        fail(f"Missing quality report: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return str(data.get("status", "fail"))


def run_command(command: list[str]) -> None:
    print("$ " + " ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def count_repairs(path: Path) -> int:
    data = json.loads(path.read_text(encoding="utf-8"))
    repairs = data.get("repairs", [])
    return len(repairs) if isinstance(repairs, list) else 0


def main() -> None:
    args = parse_args()
    if args.max_iteration < args.start_iteration:
        fail("--max-iteration must be greater than or equal to --start-iteration.")

    current_quality_report = args.quality_report
    current_edit_decisions = args.edit_decisions
    current_edited_transcript = args.edited_transcript

    status = load_status(current_quality_report)
    if status == "pass":
        print(f"Initial quality report is pass: {current_quality_report}")
        return

    for iteration in range(args.start_iteration, args.max_iteration + 1):
        print(f"Starting repair iteration v{iteration} from status: {status}")

        repair_plan = ARTIFACTS_DIR / f"repair_plan_v{iteration}.json"
        run_command(
            [
                sys.executable,
                str(SCRIPTS_DIR / "repair_from_quality_report.py"),
                "--quality-report",
                str(current_quality_report),
                "--edit-decisions",
                str(current_edit_decisions),
                "--raw-transcript",
                str(args.raw_transcript),
                "--edited-transcript",
                str(current_edited_transcript),
                "--iteration",
                str(iteration),
                "--min-severity",
                args.min_severity,
                "--output",
                str(repair_plan),
            ]
        )

        repairs = count_repairs(repair_plan)
        if repairs == 0:
            print("No conservative repairs available. Manual review required.")
            return

        output_video = EDITED_DIR / f"edited_video_v{iteration}.mp4"
        next_edit_decisions = ARTIFACTS_DIR / f"edit_decisions_v{iteration}.json"
        next_edited_transcript = ARTIFACTS_DIR / f"edited_transcription_v{iteration}.json"
        next_quality_report = ARTIFACTS_DIR / f"final_quality_report_v{iteration}.json"

        edit_command = [
            sys.executable,
            str(SCRIPTS_DIR / "edit_video.py"),
            "--padding",
            str(args.padding),
            "--repair-plan",
            str(repair_plan),
            "--output",
            str(output_video),
            "--edit-decisions-output",
            str(next_edit_decisions),
            "--quality-language",
            args.quality_language,
            "--quality-transcript-output",
            str(next_edited_transcript),
            "--quality-report-output",
            str(next_quality_report),
        ]
        if args.video_name:
            edit_command.extend(["--video-name", args.video_name])
        run_command(edit_command)

        current_quality_report = next_quality_report
        current_edit_decisions = next_edit_decisions
        current_edited_transcript = next_edited_transcript
        status = load_status(current_quality_report)
        if status == "pass":
            print(f"Quality passed after iteration v{iteration}: {output_video}")
            return

    print(
        f"Stopped after v{args.max_iteration} with status '{status}'. "
        "Manual review is recommended before more automatic edits."
    )


if __name__ == "__main__":
    main()
