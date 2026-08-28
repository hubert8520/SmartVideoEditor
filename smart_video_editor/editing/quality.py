"""Post-render quality-check orchestration."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from smart_video_editor.paths import PROJECT_ROOT


def fail(message: str) -> None:
    print(f"Error: {message}", file=sys.stderr)
    raise SystemExit(1)


def run_quality_check(args: argparse.Namespace) -> None:
    if args.skip_quality_check:
        print("Post-render quality check skipped.")
        return

    script_path = Path(__file__).resolve().parents[2] / "scripts" / "quality_check_edited_video.py"
    command = [
        sys.executable,
        str(script_path),
        "--video",
        str(args.output),
        "--provider",
        args.quality_provider,
        "--transcript-output",
        str(args.quality_transcript_output),
        "--output",
        str(args.quality_report_output),
    ]
    if args.quality_language:
        command.extend(["--language", args.quality_language])
    if args.quality_model:
        command.extend(["--model", args.quality_model])

    print("Running post-render quality check...")
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.returncode == 0:
        print(f"Quality report saved to: {args.quality_report_output}")
        return

    message = result.stderr.strip() or result.stdout.strip() or "unknown error"
    if args.quality_check_strict:
        fail(f"Post-render quality check failed: {message}")
    print(f"Warning: post-render quality check failed: {message}", file=sys.stderr)
