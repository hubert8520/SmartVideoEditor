#!/usr/bin/env python3
"""Run a post-render quality check on the edited video."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(
    os.environ.get("SMART_VIDEO_EDITOR_WORKSPACE", Path(__file__).resolve().parents[1])
).expanduser().resolve()
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import transcribe_video  # noqa: E402

from smart_video_editor.llm.prompts import (  # noqa: E402
    QUALITY_CHECK_SYSTEM_PROMPT,
    QUALITY_CHECK_USER_PROMPT_TEMPLATE,
)
from smart_video_editor.reporting.quality import (  # noqa: E402
    enrich_quality_report,
    quality_mapping_context_for_prompt,
)


ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
EDITED_DIR = PROJECT_ROOT / "edited"
DEFAULT_VIDEO_PATH = EDITED_DIR / "edited_video.mp4"
DEFAULT_TRANSCRIPT_OUTPUT = ARTIFACTS_DIR / "edited_transcription.json"
DEFAULT_OUTPUT_PATH = ARTIFACTS_DIR / "final_quality_report.json"
DEFAULT_EDIT_DECISIONS = ARTIFACTS_DIR / "edit_decisions.json"
DEFAULT_RAW_TRANSCRIPT = ARTIFACTS_DIR / "raw_transcription.json"
DEFAULT_LLM_MODEL = "gpt-5.2"


QUALITY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "status": {
            "type": "string",
            "enum": ["pass", "needs_review", "fail"],
        },
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "start_word_id": {"type": "integer"},
                    "end_word_id": {"type": "integer"},
                    "start": {"type": "string"},
                    "end": {"type": "string"},
                    "issue_category": {
                        "type": "string",
                        "enum": [
                            "cut_word",
                            "repetition",
                            "dangling_thought",
                            "logic_gap",
                            "off_topic",
                            "noise_or_setup",
                            "other",
                        ],
                    },
                    "severity": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                    },
                    "description": {"type": "string"},
                    "affected_text": {"type": "string"},
                    "suggested_action": {"type": "string"},
                    "repair_suggestion": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": [
                                    "force_keep",
                                    "force_drop",
                                    "manual_review",
                                    "no_auto_repair",
                                ],
                            },
                            "confidence": {
                                "type": "string",
                                "enum": ["low", "medium", "high"],
                            },
                            "rationale": {"type": "string"},
                            "requires_manual_review": {"type": "boolean"},
                        },
                        "required": [
                            "action",
                            "confidence",
                            "rationale",
                            "requires_manual_review",
                        ],
                    },
                },
                "required": [
                    "start_word_id",
                    "end_word_id",
                    "start",
                    "end",
                    "issue_category",
                    "severity",
                    "description",
                    "affected_text",
                    "suggested_action",
                    "repair_suggestion",
                ],
            },
        },
        "overall_notes": {"type": "string"},
    },
    "required": ["status", "issues", "overall_notes"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe edited video and ask an LLM for QA.")
    parser.add_argument(
        "--video",
        type=Path,
        default=DEFAULT_VIDEO_PATH,
        help="Edited video path. Default: edited/edited_video.mp4.",
    )
    parser.add_argument(
        "--provider",
        choices=("deepgram", "openai"),
        default="deepgram",
        help="Provider used to transcribe the edited video. Default: deepgram.",
    )
    parser.add_argument(
        "--language",
        help="Optional ISO language code, e.g. pl. Leave empty for auto-detect.",
    )
    parser.add_argument(
        "--transcription-model",
        help="Provider model used for QA transcription.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_LLM_MODEL,
        help=f"OpenAI model used for QA analysis. Default: {DEFAULT_LLM_MODEL}.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("minimal", "low", "medium", "high"),
        default="medium",
        help="Reasoning effort for supported models. Default: medium.",
    )
    parser.add_argument(
        "--no-reasoning",
        action="store_true",
        help="Do not send the reasoning parameter. Useful for non-reasoning models.",
    )
    parser.add_argument(
        "--chunk-minutes",
        type=float,
        default=5.0,
        help="Length of each audio chunk in minutes. Default: 5.",
    )
    parser.add_argument(
        "--bitrate",
        default="64k",
        help="MP3 audio bitrate used for upload chunks. Default: 64k.",
    )
    parser.add_argument(
        "--transcript-output",
        type=Path,
        default=DEFAULT_TRANSCRIPT_OUTPUT,
        help="Output path for edited-video transcription.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output path for quality report.",
    )
    parser.add_argument(
        "--edit-decisions",
        type=Path,
        default=DEFAULT_EDIT_DECISIONS,
        help="edit_decisions.json used to map final issue ranges back to raw time.",
    )
    parser.add_argument(
        "--raw-transcript",
        type=Path,
        default=DEFAULT_RAW_TRANSCRIPT,
        help="raw_transcription.json used to add raw context to QA issues.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        help="Path to a .env file. Defaults to .env in the current directory or project root.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print intended work without calling APIs.",
    )
    return parser.parse_args()


def fail(message: str) -> None:
    print(f"Error: {message}", file=sys.stderr)
    raise SystemExit(1)


def load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"Invalid JSON in {path}: {exc}")
    if not isinstance(data, dict):
        fail(f"Expected JSON object in {path}.")
    return data


def compact_transcript_for_prompt(transcript: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": transcript.get("source", {}),
        "segments": [
            {
                "id": segment["id"],
                "start": segment["timestamp"],
                "end": segment["end"],
                "text": segment["transcription"],
                "word_ids": segment.get("word_ids", []),
            }
            for segment in transcript.get("segments", [])
        ],
        "words": [
            {
                "id": word["id"],
                "start": word["timestamp"],
                "end": word["end"],
                "word": word["word"],
            }
            for word in transcript.get("words", [])
        ],
    }


def call_openai_quality(
    api_key: str,
    model: str,
    reasoning_effort: str | None,
    transcript: dict[str, Any],
    qa_context: dict[str, Any],
) -> dict[str, Any]:
    try:
        from openai import OpenAI
    except ImportError:
        fail("Missing Python package 'openai'. Install dependencies with: pip install -r requirements.txt")

    transcript_json = json.dumps(compact_transcript_for_prompt(transcript), ensure_ascii=False, indent=2)
    qa_context_json = json.dumps(qa_context, ensure_ascii=False, indent=2)
    client = OpenAI(api_key=api_key)
    params: dict[str, Any] = {
        "model": model,
        "instructions": QUALITY_CHECK_SYSTEM_PROMPT,
        "input": QUALITY_CHECK_USER_PROMPT_TEMPLATE.format(
            qa_context_json=qa_context_json,
            transcript_json=transcript_json,
        ),
        "text": {
            "format": {
                "type": "json_schema",
                "name": "edited_video_quality_report",
                "strict": True,
                "schema": QUALITY_SCHEMA,
            }
        },
    }
    if reasoning_effort:
        params["reasoning"] = {"effort": reasoning_effort}

    response = client.responses.create(**params)
    if not response.output_text:
        fail("OpenAI response did not contain output_text.")

    try:
        report = json.loads(response.output_text)
    except json.JSONDecodeError as exc:
        fail(f"OpenAI response was not valid JSON: {exc}")

    if not isinstance(report, dict):
        fail("Quality report must be a JSON object.")
    return report


def main() -> None:
    args = parse_args()
    if not args.video.exists():
        fail(f"Video does not exist: {args.video}")
    if args.chunk_minutes <= 0:
        fail("--chunk-minutes must be greater than 0.")

    loaded_env = transcribe_video.load_env_file(args.env_file)
    transcribe_args = argparse.Namespace(
        provider=args.provider,
        model=args.transcription_model,
        language=args.language,
        prompt=None,
        chunk_minutes=args.chunk_minutes,
        bitrate=args.bitrate,
        diarize=False,
        allow_coarse_openai=False,
    )
    transcription_model = transcribe_video.resolve_model(transcribe_args)
    transcribe_video.validate_provider_options(transcribe_args, transcription_model)
    active_model = transcribe_video.display_model(args.provider, transcription_model, False)
    edit_decisions = load_optional_json(args.edit_decisions)
    raw_transcript = load_optional_json(args.raw_transcript)
    qa_context = quality_mapping_context_for_prompt(
        edit_decisions=edit_decisions,
        raw_transcript=raw_transcript,
    )

    if args.dry_run:
        print(f"Video: {args.video}")
        print(f"QA transcription: {args.provider}/{active_model}")
        print(f"QA LLM model: {args.model}")
        print(f"Edit decisions mapping: {'available' if edit_decisions else 'missing'} ({args.edit_decisions})")
        print(f"Raw transcript context: {'available' if raw_transcript else 'missing'} ({args.raw_transcript})")
        print("Dry run complete. No API call was made.")
        return

    transcription_key = transcribe_video.get_required_api_key(args.provider, loaded_env)
    transcription_client = transcribe_video.create_client(args.provider, transcription_key)
    ffmpeg = transcribe_video.resolve_ffmpeg_executable()
    transcript = transcribe_video.transcribe_media(
        args.video,
        transcribe_args,
        transcription_key,
        transcription_model,
        active_model,
        transcription_client,
        ffmpeg,
    )

    args.transcript_output.parent.mkdir(parents=True, exist_ok=True)
    args.transcript_output.write_text(
        json.dumps(transcript, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Edited transcript saved to: {args.transcript_output}")

    openai_key = transcribe_video.os.getenv("OPENAI_API_KEY")
    if not openai_key or any(marker in openai_key.lower() for marker in transcribe_video.PLACEHOLDER_KEY_MARKERS):
        fail("OPENAI_API_KEY is missing or still looks like a placeholder.")

    reasoning_effort = None if args.no_reasoning else args.reasoning_effort
    report = call_openai_quality(openai_key, args.model, reasoning_effort, transcript, qa_context)
    report = enrich_quality_report(
        report,
        edit_decisions=edit_decisions,
        raw_transcript=raw_transcript,
        edit_decisions_path=args.edit_decisions,
        raw_transcript_path=args.raw_transcript,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Quality status: {report.get('status', 'unknown')}")
    print(f"Issues: {len(report.get('issues', []))}")


if __name__ == "__main__":
    main()
