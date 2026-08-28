"""Unified command-line interface for the Smart Video Editor pipeline."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from smart_video_editor import __version__
from smart_video_editor.paths import WORKSPACE_ENV


INSTALL_ROOT = Path(__file__).resolve().parents[2]
BUNDLED_SCRIPTS_DIR = INSTALL_ROOT / "scripts"
WORKSPACE_DIRS = ("raw", "artifacts", "edited")


@dataclass(frozen=True)
class ScriptCommand:
    script: str
    summary: str


SCRIPT_COMMANDS = {
    "doctor": ScriptCommand(
        "doctor.py",
        "Check the local setup without calling external APIs.",
    ),
    "transcribe": ScriptCommand(
        "transcribe_video.py",
        "Create a word-level transcript from raw media.",
    ),
    "analyze": ScriptCommand(
        "analyze_transcript_llm.py",
        "Generate semantic edit decisions with an LLM.",
    ),
    "edit": ScriptCommand(
        "edit_video.py",
        "Plan cuts, render the video, and optionally run QA.",
    ),
    "quality": ScriptCommand(
        "quality_check_edited_video.py",
        "Run post-render transcription and quality assurance.",
    ),
    "repair": ScriptCommand(
        "repair_from_quality_report.py",
        "Create a conservative repair plan from a QA report.",
    ),
    "refine": ScriptCommand(
        "auto_refine_video.py",
        "Run bounded repair, render, and QA iterations.",
    ),
    "review": ScriptCommand(
        "generate_editor_review.py",
        "Generate a Markdown and CSV brief for manual review.",
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smart-video-editor",
        description=(
            "Transcribe, analyze, edit, and quality-check talking-head videos "
            "from one command-line interface."
        ),
        epilog=(
            "Use 'smart-video-editor <command> --help' for stage-specific options. "
            "Global options must appear before the command."
        ),
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="Directory containing raw/, artifacts/, edited/, and .env. Default: current directory.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.add_parser(
        "init",
        help="Create the workspace directory structure.",
        description="Create raw/, artifacts/, and edited/ in the selected workspace.",
    )
    add_run_parser(subparsers)
    for name, command in SCRIPT_COMMANDS.items():
        subparsers.add_parser(name, add_help=False, help=command.summary)
    return parser


def add_run_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "run",
        help="Run the end-to-end editing pipeline.",
        description="Run transcription, semantic analysis, rendering, and optional refinement.",
    )
    parser.add_argument(
        "--video-name",
        help="Media file name inside raw/. The only media file is selected when omitted.",
    )
    parser.add_argument(
        "--language",
        help="Optional ISO language code used for transcription and QA, for example pl or en.",
    )
    parser.add_argument(
        "--provider",
        choices=("deepgram", "openai"),
        default="deepgram",
        help="Raw transcription provider. Default: deepgram.",
    )
    parser.add_argument("--transcription-model", help="Model used for raw transcription.")
    parser.add_argument(
        "--analysis-model",
        default="gpt-5.2",
        help="OpenAI model used for semantic edit decisions. Default: gpt-5.2.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("minimal", "low", "medium", "high"),
        default="medium",
        help="Reasoning effort used for semantic analysis. Default: medium.",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.05,
        help="Seconds kept around spoken fragments. Default: 0.05.",
    )
    parser.add_argument(
        "--quality-provider",
        choices=("deepgram", "openai"),
        default="deepgram",
        help="Provider used to transcribe the rendered video. Default: deepgram.",
    )
    parser.add_argument("--quality-model", help="OpenAI model used for post-render QA.")
    parser.add_argument(
        "--from-stage",
        choices=("transcribe", "analyze", "edit"),
        default="transcribe",
        help="Resume the pipeline from an existing stage. Default: transcribe.",
    )
    parser.add_argument("--diarize", action="store_true", help="Enable speaker diarization.")
    parser.add_argument(
        "--allow-heuristic-drops",
        action="store_true",
        help="Allow local heuristics to apply cuts instead of only flagging them for review.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Write edit decisions without rendering or post-render QA.",
    )
    quality_group = parser.add_mutually_exclusive_group()
    quality_group.add_argument(
        "--skip-quality-check",
        action="store_true",
        help="Render without post-render quality assurance.",
    )
    quality_group.add_argument(
        "--auto-refine",
        action="store_true",
        help="Run bounded repair iterations when the initial QA does not pass.",
    )


def resolve_workspace(path: Path) -> Path:
    return path.expanduser().resolve()


def initialize_workspace(workspace: Path) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    for directory in WORKSPACE_DIRS:
        (workspace / directory).mkdir(exist_ok=True)
    print(f"Workspace ready: {workspace}")
    print("Created or verified: raw/, artifacts/, edited/")


def script_path(script_name: str) -> Path:
    path = BUNDLED_SCRIPTS_DIR / script_name
    if not path.exists():
        raise RuntimeError(f"Bundled command script is missing: {path}")
    return path


def run_script(script_name: str, arguments: Sequence[str], workspace: Path) -> int:
    environment = os.environ.copy()
    environment[WORKSPACE_ENV] = str(workspace)
    command = [sys.executable, str(script_path(script_name)), *arguments]
    result = subprocess.run(command, cwd=workspace, env=environment, check=False)
    return result.returncode


def append_option(arguments: list[str], name: str, value: object | None) -> None:
    if value is not None:
        arguments.extend([name, str(value)])


def pipeline_stages(args: argparse.Namespace) -> list[tuple[str, str, list[str]]]:
    transcribe_args = ["--provider", args.provider]
    append_option(transcribe_args, "--video-name", args.video_name)
    append_option(transcribe_args, "--model", args.transcription_model)
    append_option(transcribe_args, "--language", args.language)
    if args.diarize:
        transcribe_args.append("--diarize")

    analyze_args = [
        "--model",
        args.analysis_model,
        "--reasoning-effort",
        args.reasoning_effort,
    ]

    edit_args = [
        "--padding",
        str(args.padding),
        "--quality-provider",
        args.quality_provider,
    ]
    append_option(edit_args, "--video-name", args.video_name)
    append_option(edit_args, "--quality-language", args.language)
    append_option(edit_args, "--quality-model", args.quality_model)
    if args.allow_heuristic_drops:
        edit_args.append("--allow-heuristic-drops")
    if args.plan_only:
        edit_args.extend(["--dry-run", "--skip-quality-check"])
    elif args.skip_quality_check:
        edit_args.append("--skip-quality-check")
    elif args.auto_refine:
        edit_args.append("--quality-check-strict")

    stages = [
        ("transcribe", SCRIPT_COMMANDS["transcribe"].script, transcribe_args),
        ("analyze", SCRIPT_COMMANDS["analyze"].script, analyze_args),
        ("edit", SCRIPT_COMMANDS["edit"].script, edit_args),
    ]
    start_index = ("transcribe", "analyze", "edit").index(args.from_stage)
    stages = stages[start_index:]

    if args.auto_refine and not args.plan_only:
        refine_args: list[str] = []
        append_option(refine_args, "--video-name", args.video_name)
        append_option(refine_args, "--quality-language", args.language)
        stages.append(("refine", SCRIPT_COMMANDS["refine"].script, refine_args))
    return stages


def run_pipeline(args: argparse.Namespace, workspace: Path) -> int:
    if args.padding < 0:
        raise ValueError("--padding must be greater than or equal to 0.")
    if args.plan_only and args.auto_refine:
        raise ValueError("--plan-only cannot be combined with --auto-refine.")

    initialize_workspace(workspace)
    stages = pipeline_stages(args)
    for index, (name, script_name, stage_args) in enumerate(stages, start=1):
        print(f"\n[{index}/{len(stages)}] Running {name}")
        returncode = run_script(script_name, stage_args, workspace)
        if returncode:
            print(f"Pipeline stopped during '{name}' (exit code {returncode}).", file=sys.stderr)
            return returncode
    print("\nPipeline completed successfully.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args, forwarded = parser.parse_known_args(argv)
    if not args.command:
        parser.print_help()
        return 0

    workspace = resolve_workspace(args.workspace)
    if args.command == "init":
        if forwarded:
            parser.error(f"unrecognized arguments: {' '.join(forwarded)}")
        initialize_workspace(workspace)
        return 0
    if args.command == "run":
        if forwarded:
            parser.error(f"unrecognized arguments: {' '.join(forwarded)}")
        try:
            return run_pipeline(args, workspace)
        except (RuntimeError, ValueError) as exc:
            parser.error(str(exc))

    command = SCRIPT_COMMANDS[args.command]
    try:
        return run_script(command.script, forwarded, workspace)
    except RuntimeError as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
