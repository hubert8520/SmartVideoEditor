#!/usr/bin/env python3
"""Analyze word-level raw_transcription.json with an OpenAI LLM."""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(
    os.environ.get("SMART_VIDEO_EDITOR_WORKSPACE", Path(__file__).resolve().parents[1])
).expanduser().resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from smart_video_editor.detection.local import detect_local_candidates  # noqa: E402
from smart_video_editor.llm.prompts import (  # noqa: E402
    EDIT_ANALYSIS_SYSTEM_PROMPT,
    EDIT_ANALYSIS_USER_PROMPT_TEMPLATE,
)
from smart_video_editor.transcription.normalization import normalize_words_payload  # noqa: E402

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_TRANSCRIPT_PATH = ARTIFACTS_DIR / "raw_transcription.json"
LEGACY_TRANSCRIPT_PATH = ARTIFACTS_DIR / "raw_transcrpition.json"
DEFAULT_OUTPUT_PATH = ARTIFACTS_DIR / "llm_edit_decisions.json"
DEFAULT_MODEL = "gpt-5.2"
PLACEHOLDER_KEY_MARKERS = ("your-", "your_", "wklej", "tutaj", "...")
TIMESTAMP_RE = re.compile(r"^\d{2}:\d{2}:\d{2}:\d{3}$")
SYSTEM_PROMPT = EDIT_ANALYSIS_SYSTEM_PROMPT
USER_PROMPT_TEMPLATE = EDIT_ANALYSIS_USER_PROMPT_TEMPLATE


EDIT_DECISIONS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "thought_blocks": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "integer"},
                    "start_word_id": {"type": "integer"},
                    "end_word_id": {"type": "integer"},
                    "start": {"type": "string"},
                    "end": {"type": "string"},
                    "role": {
                        "type": "string",
                        "enum": [
                            "section_heading",
                            "transition_question",
                            "structure_step",
                            "core_explanation",
                            "case_study",
                            "result",
                            "aside",
                            "other",
                        ],
                    },
                    "text": {"type": "string"},
                    "must_keep": {"type": "boolean"},
                    "note": {"type": "string"},
                },
                "required": [
                    "id",
                    "start_word_id",
                    "end_word_id",
                    "start",
                    "end",
                    "role",
                    "text",
                    "must_keep",
                    "note",
                ],
            },
        },
        "drop_ranges": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "start_word_id": {"type": "integer"},
                    "end_word_id": {"type": "integer"},
                    "start": {"type": "string"},
                    "end": {"type": "string"},
                    "reason": {"type": "string"},
                    "reason_category": {
                        "type": "string",
                        "enum": [
                            "false_start",
                            "repeated_take",
                            "filler",
                            "dangling_thought",
                            "off_topic",
                            "flow_break",
                            "cut_safety",
                            "other",
                        ],
                    },
                    "confidence": {"type": "number"},
                    "affected_text": {"type": "string"},
                    "candidate_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "cut_risk": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                    },
                    "preserves_meaning": {"type": "boolean"},
                    "safety_basis": {"type": "string"},
                },
                "required": [
                    "start_word_id",
                    "end_word_id",
                    "start",
                    "end",
                    "reason",
                    "reason_category",
                    "confidence",
                    "affected_text",
                    "candidate_ids",
                    "cut_risk",
                    "preserves_meaning",
                    "safety_basis",
                ],
            },
        },
        "review_ranges": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "start_word_id": {"type": "integer"},
                    "end_word_id": {"type": "integer"},
                    "start": {"type": "string"},
                    "end": {"type": "string"},
                    "reason": {"type": "string"},
                    "confidence": {"type": "number"},
                    "affected_text": {"type": "string"},
                    "question": {"type": "string"},
                    "candidate_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "uncertainty_reason": {"type": "string"},
                    "suggested_action": {
                        "type": "string",
                        "enum": ["manual_review", "audio_review", "llm_context_review", "keep"],
                    },
                },
                "required": [
                    "start_word_id",
                    "end_word_id",
                    "start",
                    "end",
                    "reason",
                    "confidence",
                    "affected_text",
                    "question",
                    "candidate_ids",
                    "uncertainty_reason",
                    "suggested_action",
                ],
            },
        },
        "candidate_reviews": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "candidate_id": {"type": "string"},
                    "decision": {
                        "type": "string",
                        "enum": ["approve_drop", "review", "reject"],
                    },
                    "reason": {"type": "string"},
                    "safety_basis": {"type": "string"},
                    "target": {
                        "type": "string",
                        "enum": ["drop_ranges", "review_ranges", "keep_notes", "none"],
                    },
                },
                "required": [
                    "candidate_id",
                    "decision",
                    "reason",
                    "safety_basis",
                    "target",
                ],
            },
        },
        "keep_notes": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "start_word_id": {"type": "integer"},
                    "end_word_id": {"type": "integer"},
                    "start": {"type": "string"},
                    "end": {"type": "string"},
                    "note": {"type": "string"},
                    "affected_text": {"type": "string"},
                },
                "required": [
                    "start_word_id",
                    "end_word_id",
                    "start",
                    "end",
                    "note",
                    "affected_text",
                ],
            },
        },
        "overall_notes": {"type": "string"},
    },
    "required": [
        "thought_blocks",
        "drop_ranges",
        "review_ranges",
        "candidate_reviews",
        "keep_notes",
        "overall_notes",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use an OpenAI LLM to analyze raw_transcription.json for edit decisions."
    )
    parser.add_argument(
        "--transcript",
        type=Path,
        help="Transcript path. Defaults to artifacts/raw_transcription.json.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output path. Default: artifacts/llm_edit_decisions.json.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenAI model used for analysis. Default: {DEFAULT_MODEL}.",
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
        "--env-file",
        type=Path,
        help="Path to a .env file. Defaults to .env in the current directory or project root.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate input and show the request size without calling the API.",
    )
    return parser.parse_args()


def fail(message: str) -> None:
    print(f"Error: {message}", file=sys.stderr)
    raise SystemExit(1)


def warn_if_env_file_is_too_open(env_path: Path) -> None:
    if os.name != "posix":
        return

    mode = stat.S_IMODE(env_path.stat().st_mode)
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        print(
            f"Warning: {env_path} can be accessed by group/other users. "
            f"Secure it with: chmod 600 {env_path}",
            file=sys.stderr,
        )


def load_env_file(env_file: Path | None) -> Path | None:
    candidates = [env_file] if env_file else [Path.cwd() / ".env", PROJECT_ROOT / ".env"]
    seen: set[Path] = set()

    for candidate in candidates:
        if candidate is None:
            continue

        candidate = candidate.expanduser().resolve()
        if candidate in seen:
            continue
        seen.add(candidate)

        if not candidate.exists():
            if env_file:
                fail(f"Env file does not exist: {candidate}")
            continue

        try:
            from dotenv import load_dotenv
        except ImportError:
            fail(
                "Missing Python package 'python-dotenv'. "
                "Install dependencies with: pip install -r requirements.txt"
            )

        load_dotenv(candidate, override=False)
        warn_if_env_file_is_too_open(candidate)
        return candidate

    return None


def get_openai_api_key(loaded_env: Path | None) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key and not any(marker in api_key.lower() for marker in PLACEHOLDER_KEY_MARKERS):
        return api_key

    location_hint = loaded_env or (PROJECT_ROOT / ".env")
    fail(
        "OPENAI_API_KEY is missing or still looks like a placeholder. "
        f"Add it to {location_hint} as: OPENAI_API_KEY=sk-..."
    )


def resolve_transcript_path(path: Path | None) -> Path:
    if path:
        if not path.exists():
            fail(f"Transcript file does not exist: {path}")
        return path

    if DEFAULT_TRANSCRIPT_PATH.exists():
        return DEFAULT_TRANSCRIPT_PATH
    if LEGACY_TRANSCRIPT_PATH.exists():
        return LEGACY_TRANSCRIPT_PATH

    fail(
        "No transcript found. Expected artifacts/raw_transcription.json "
        "or artifacts/raw_transcrpition.json."
    )


def validate_timestamp(value: str, label: str) -> None:
    if not TIMESTAMP_RE.fullmatch(value):
        fail(f"Invalid {label}: {value!r}. Expected hh:mm:ss:ms.")


def timestamp_to_seconds(timestamp: str) -> float:
    validate_timestamp(timestamp, "timestamp")
    hours, minutes, seconds, milliseconds = timestamp.split(":")
    return (
        int(hours) * 3600
        + int(minutes) * 60
        + int(seconds)
        + int(milliseconds) / 1000
    )


def seconds_to_timestamp(seconds: float) -> str:
    total_milliseconds = max(0, int(round(seconds * 1000)))
    total_seconds, milliseconds = divmod(total_milliseconds, 1000)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}:{milliseconds:03d}"


def legacy_transcript_to_word_level(data: list[Any]) -> dict[str, Any]:
    segments: list[dict[str, Any]] = []
    words: list[dict[str, Any]] = []
    word_id = 0

    for segment_id, item in enumerate(data):
        if not isinstance(item, dict):
            fail(f"Transcript item #{segment_id} is not an object.")
        for key in ("timestamp", "end", "transcription"):
            if key not in item:
                fail(f"Transcript item #{segment_id} is missing {key!r}.")

        start = str(item["timestamp"])
        end = str(item["end"])
        text = str(item["transcription"]).strip()
        validate_timestamp(start, f"timestamp in item #{segment_id}")
        validate_timestamp(end, f"end in item #{segment_id}")

        start_seconds = timestamp_to_seconds(start)
        end_seconds = timestamp_to_seconds(end)
        tokens = [token for token in text.split() if token]
        token_count = max(1, len(tokens))
        segment_word_ids: list[int] = []
        for index, token in enumerate(tokens):
            token_start = start_seconds + (end_seconds - start_seconds) * index / token_count
            token_end = start_seconds + (end_seconds - start_seconds) * (index + 1) / token_count
            words.append(
                {
                    "id": word_id,
                    "timestamp": seconds_to_timestamp(token_start),
                    "end": seconds_to_timestamp(token_end),
                    "word": token,
                }
            )
            segment_word_ids.append(word_id)
            word_id += 1

        segments.append(
            {
                "id": segment_id,
                "timestamp": start,
                "end": end,
                "transcription": text,
                "word_ids": segment_word_ids,
            }
        )

    return {
        "version": "1.0-upgraded",
        "source": {"word_level": False},
        "segments": segments,
        "words": words,
    }


def load_transcript(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        data = legacy_transcript_to_word_level(data)
    if not isinstance(data, dict):
        fail("Transcript JSON must be an object with segments and words.")

    segments = data.get("segments")
    words = data.get("words")
    if not isinstance(segments, list):
        fail("Transcript JSON must contain a 'segments' list.")
    if not isinstance(words, list):
        fail("Transcript JSON must contain a 'words' list.")

    previous_word_id = -1
    for index, word in enumerate(words):
        if not isinstance(word, dict):
            fail(f"Transcript words[{index}] is not an object.")
        for key in ("id", "timestamp", "end", "word"):
            if key not in word:
                fail(f"Transcript words[{index}] is missing {key!r}.")
        word_id = int(word["id"])
        if word_id <= previous_word_id:
            fail(f"Transcript word ids are not sorted at words[{index}].")
        previous_word_id = word_id
        validate_timestamp(str(word["timestamp"]), f"words[{index}].timestamp")
        validate_timestamp(str(word["end"]), f"words[{index}].end")

    valid_word_ids = {int(word["id"]) for word in words}
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            fail(f"Transcript segments[{index}] is not an object.")
        for key in ("id", "timestamp", "end", "transcription"):
            if key not in segment:
                fail(f"Transcript segments[{index}] is missing {key!r}.")
        validate_timestamp(str(segment["timestamp"]), f"segments[{index}].timestamp")
        validate_timestamp(str(segment["end"]), f"segments[{index}].end")
        for word_id in segment.get("word_ids", []):
            if int(word_id) not in valid_word_ids:
                fail(f"Transcript segments[{index}] references missing word id {word_id}.")

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
            for segment in transcript["segments"]
        ],
        "words": [
            {
                "id": word["id"],
                "start": word["timestamp"],
                "end": word["end"],
                "word": word["word"],
                **({"speaker": word["speaker"]} if "speaker" in word else {}),
            }
            for word in transcript["words"]
        ],
    }


def local_candidates_for_prompt(transcript: dict[str, Any]) -> list[dict[str, Any]]:
    domain_words = normalize_words_payload(transcript["words"])
    candidates = []
    for index, candidate in enumerate(detect_local_candidates(domain_words), start=1):
        record = {
            "id": f"local-{index:03d}",
            "category": candidate.category,
            "recommended_action": candidate.recommended_action,
            "confidence": candidate.confidence,
            "start_word_id": candidate.start_word_id,
            "end_word_id": candidate.end_word_id,
            "start": seconds_to_timestamp(candidate.start),
            "end": seconds_to_timestamp(candidate.end),
            "text": candidate.text,
            "reason": candidate.reason,
            "source": candidate.source,
        }
        if candidate.evidence:
            record["evidence"] = candidate.evidence
        candidates.append(record)
    return candidates


def build_user_prompt(transcript: dict[str, Any]) -> str:
    transcript_json = json.dumps(compact_transcript_for_prompt(transcript), ensure_ascii=False, indent=2)
    candidate_json = json.dumps(local_candidates_for_prompt(transcript), ensure_ascii=False, indent=2)
    return USER_PROMPT_TEMPLATE.format(
        candidate_json=candidate_json,
        transcript_json=transcript_json,
    )


def known_candidate_ids(transcript: dict[str, Any]) -> set[str]:
    return {str(candidate["id"]) for candidate in local_candidates_for_prompt(transcript)}


def validate_word_range(
    item: dict[str, Any],
    top_key: str,
    index: int,
    valid_word_ids: set[int],
) -> None:
    start_word_id = int(item["start_word_id"])
    end_word_id = int(item["end_word_id"])
    if start_word_id not in valid_word_ids:
        fail(f"Model output {top_key}[{index}].start_word_id does not exist.")
    if end_word_id not in valid_word_ids:
        fail(f"Model output {top_key}[{index}].end_word_id does not exist.")
    if end_word_id < start_word_id:
        fail(f"Model output {top_key}[{index}] has end_word_id < start_word_id.")
    validate_timestamp(str(item["start"]), f"{top_key}[{index}].start")
    validate_timestamp(str(item["end"]), f"{top_key}[{index}].end")
    if timestamp_to_seconds(str(item["end"])) <= timestamp_to_seconds(str(item["start"])):
        fail(f"Model output {top_key}[{index}] has end <= start.")


def validate_output(decisions: dict[str, Any], transcript: dict[str, Any]) -> None:
    valid_word_ids = {int(word["id"]) for word in transcript["words"]}
    transcript_range_keys = {
        "thought_blocks": (
            "id",
            "start_word_id",
            "end_word_id",
            "start",
            "end",
            "role",
            "text",
            "must_keep",
            "note",
        ),
        "drop_ranges": (
            "start_word_id",
            "end_word_id",
            "start",
            "end",
            "reason",
            "reason_category",
            "confidence",
            "affected_text",
            "candidate_ids",
            "cut_risk",
            "preserves_meaning",
            "safety_basis",
        ),
        "review_ranges": (
            "start_word_id",
            "end_word_id",
            "start",
            "end",
            "reason",
            "confidence",
            "affected_text",
            "question",
            "candidate_ids",
            "uncertainty_reason",
            "suggested_action",
        ),
        "keep_notes": (
            "start_word_id",
            "end_word_id",
            "start",
            "end",
            "note",
            "affected_text",
        ),
    }

    for top_key, required_keys in transcript_range_keys.items():
        value = decisions.get(top_key)
        if not isinstance(value, list):
            fail(f"Model output {top_key!r} must be a list.")
        for index, item in enumerate(value):
            if not isinstance(item, dict):
                fail(f"Model output {top_key}[{index}] must be an object.")
            for key in required_keys:
                if key not in item:
                    fail(f"Model output {top_key}[{index}] is missing {key!r}.")
            validate_word_range(item, top_key, index, valid_word_ids)

            if top_key == "drop_ranges":
                if item["cut_risk"] != "low":
                    fail(f"Model output drop_ranges[{index}] must use review_ranges unless cut_risk is low.")
                if not bool(item["preserves_meaning"]):
                    fail(f"Model output drop_ranges[{index}] must preserve meaning.")
                if not str(item["safety_basis"]).strip():
                    fail(f"Model output drop_ranges[{index}].safety_basis must be non-empty.")

            if top_key == "review_ranges":
                if not str(item["uncertainty_reason"]).strip():
                    fail(f"Model output review_ranges[{index}].uncertainty_reason must be non-empty.")

    candidate_ids = known_candidate_ids(transcript)
    candidate_reviews = decisions.get("candidate_reviews")
    if not isinstance(candidate_reviews, list):
        fail("Model output 'candidate_reviews' must be a list.")
    required_candidate_review_keys = ("candidate_id", "decision", "reason", "safety_basis", "target")
    for index, item in enumerate(candidate_reviews):
        if not isinstance(item, dict):
            fail(f"Model output candidate_reviews[{index}] must be an object.")
        for key in required_candidate_review_keys:
            if key not in item:
                fail(f"Model output candidate_reviews[{index}] is missing {key!r}.")
        candidate_id = str(item.get("candidate_id", ""))
        if candidate_id not in candidate_ids:
            fail(f"Model output candidate_reviews[{index}] references unknown candidate_id {candidate_id!r}.")
        if not str(item.get("safety_basis", "")).strip():
            fail(f"Model output candidate_reviews[{index}].safety_basis must be non-empty.")

    for top_key in ("drop_ranges", "review_ranges"):
        for index, item in enumerate(decisions[top_key]):
            ids = item.get("candidate_ids")
            if not isinstance(ids, list):
                fail(f"Model output {top_key}[{index}].candidate_ids must be a list.")
            unknown = [str(candidate_id) for candidate_id in ids if str(candidate_id) not in candidate_ids]
            if unknown:
                fail(f"Model output {top_key}[{index}] references unknown candidate_ids: {unknown}")

    if not isinstance(decisions.get("overall_notes"), str):
        fail("Model output 'overall_notes' must be a string.")


def call_openai(
    api_key: str,
    model: str,
    reasoning_effort: str | None,
    user_prompt: str,
) -> dict[str, Any]:
    try:
        from openai import OpenAI
    except ImportError:
        fail("Missing Python package 'openai'. Install dependencies with: pip install -r requirements.txt")

    client = OpenAI(api_key=api_key)
    params: dict[str, Any] = {
        "model": model,
        "instructions": SYSTEM_PROMPT,
        "input": user_prompt,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "video_edit_decisions",
                "strict": True,
                "schema": EDIT_DECISIONS_SCHEMA,
            }
        },
    }
    if reasoning_effort:
        params["reasoning"] = {"effort": reasoning_effort}

    response = client.responses.create(**params)
    output_text = response.output_text
    if not output_text:
        fail("OpenAI response did not contain output_text.")

    try:
        decisions = json.loads(output_text)
    except json.JSONDecodeError as exc:
        fail(f"OpenAI response was not valid JSON: {exc}")

    if not isinstance(decisions, dict):
        fail("OpenAI response JSON must be an object.")
    return decisions


def main() -> None:
    args = parse_args()
    loaded_env = load_env_file(args.env_file)
    transcript_path = resolve_transcript_path(args.transcript)
    transcript = load_transcript(transcript_path)
    user_prompt = build_user_prompt(transcript)

    if args.dry_run:
        print(f"Transcript: {transcript_path}")
        print(f"Segments: {len(transcript['segments'])}")
        print(f"Words: {len(transcript['words'])}")
        print(f"Local candidates: {len(local_candidates_for_prompt(transcript))}")
        print(f"Model: {args.model}")
        print(f"Reasoning: {'disabled' if args.no_reasoning else args.reasoning_effort}")
        print(f"Prompt characters: {len(SYSTEM_PROMPT) + len(user_prompt)}")
        print("Dry run complete. No API call was made.")
        return

    api_key = get_openai_api_key(loaded_env)
    reasoning_effort = None if args.no_reasoning else args.reasoning_effort
    decisions = call_openai(api_key, args.model, reasoning_effort, user_prompt)
    validate_output(decisions, transcript)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(decisions, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"LLM edit decisions saved to: {args.output}")


if __name__ == "__main__":
    main()
