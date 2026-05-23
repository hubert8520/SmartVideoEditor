#!/usr/bin/env python3
"""Run a post-render quality check on the edited video."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import transcribe_video  # noqa: E402


ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
EDITED_DIR = PROJECT_ROOT / "edited"
DEFAULT_VIDEO_PATH = EDITED_DIR / "edited_video.mp4"
DEFAULT_TRANSCRIPT_OUTPUT = ARTIFACTS_DIR / "edited_transcription.json"
DEFAULT_OUTPUT_PATH = ARTIFACTS_DIR / "final_quality_report.json"
DEFAULT_LLM_MODEL = "gpt-5.2"


SYSTEM_PROMPT = """Jesteś kontrolerem jakości montażu krótkich filmów edukacyjnych po polsku.

Analizujesz transkrypcję finalnego, już zmontowanego filmu. Twoje zadanie to wykryć problemy, które powinny wrócić do poprawki montażowej.

Szukaj:
- urwanych słów albo słów brzmiących jak ucięte, np. "zaa", "któ", "żeb",
- nienaturalnych skoków logicznych między zdaniami,
- powtórek, które powinny zostać wycięte, np. "żeby post" i zaraz "żeby post trafił",
- zawieszonych myśli bez dokończenia,
- resztek setupu nagrania, przekleństw, testowych tekstów, dźwięków opisanych w transkrypcji jako off-topic,
- fragmentów, które brzmią jak błąd montażowy, a nie naturalny styl mówienia.

Nie oznaczaj jako problem:
- normalnych, celowych powtórzeń retorycznych,
- krótkich łączników typu "no", "więc", "ale", jeśli zdanie ma sens,
- lekkiej potoczności, jeśli film nadal brzmi naturalnie.

Decyzje zakotwiczaj w word_id. Jeśli problem dotyczy przejścia między dwoma fragmentami, wskaż najbliższy zakres słów wokół problemu.
"""


USER_PROMPT_TEMPLATE = """Przeanalizuj finalną transkrypcję zmontowanego filmu.

Zwróć:
- status: pass, needs_review albo fail
- issues: konkretne problemy do poprawki
- overall_notes: krótki opis jakości finalnego montażu

Finalna transkrypcja:
{transcript_json}
"""


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
) -> dict[str, Any]:
    try:
        from openai import OpenAI
    except ImportError:
        fail("Missing Python package 'openai'. Install dependencies with: pip install -r requirements.txt")

    transcript_json = json.dumps(compact_transcript_for_prompt(transcript), ensure_ascii=False, indent=2)
    client = OpenAI(api_key=api_key)
    params: dict[str, Any] = {
        "model": model,
        "instructions": SYSTEM_PROMPT,
        "input": USER_PROMPT_TEMPLATE.format(transcript_json=transcript_json),
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

    if args.dry_run:
        print(f"Video: {args.video}")
        print(f"QA transcription: {args.provider}/{active_model}")
        print(f"QA LLM model: {args.model}")
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
    report = call_openai_quality(openai_key, args.model, reasoning_effort, transcript)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Quality status: {report.get('status', 'unknown')}")
    print(f"Issues: {len(report.get('issues', []))}")


if __name__ == "__main__":
    main()
