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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_TRANSCRIPT_PATH = ARTIFACTS_DIR / "raw_transcription.json"
LEGACY_TRANSCRIPT_PATH = ARTIFACTS_DIR / "raw_transcrpition.json"
DEFAULT_OUTPUT_PATH = ARTIFACTS_DIR / "llm_edit_decisions.json"
DEFAULT_MODEL = "gpt-5.2"
PLACEHOLDER_KEY_MARKERS = ("your-", "your_", "wklej", "tutaj", "...")
TIMESTAMP_RE = re.compile(r"^\d{2}:\d{2}:\d{2}:\d{3}$")


SYSTEM_PROMPT = """Jesteś doświadczonym montażystą krótkich filmów edukacyjnych i sprzedażowych po polsku.

Analizujesz transkrypcję filmu z segmentami i słowami. Najważniejsze: decyzje montażowe mają być zakotwiczone w konkretnych word_id, bo późniejszy cut planner tnie po granicach słów.

Najpierw wyznacz thought_blocks, czyli całe jednostki sensu: nagłówki sekcji, pytania, przejścia logiczne, kroki struktury, kompletne myśli i case study. Potem dopiero proponuj drop_ranges. Cut planner będzie chronił te bloki przed przypadkowym cięciem w środku.

Usuwaj:
- false starty, czyli rozpoczęte i porzucone wypowiedzi,
- powtórzone próby tej samej myśli, jeśli późniejsza wersja jest pełniejsza lub brzmi naturalniej,
- krótkie prefixy pełniejszej frazy, np. "żeby post" przed "żeby post trafił...",
- wypełniacze typu "yyy", "eee", "aaaa", "kurde", "no" gdy nie pełnią funkcji stylistycznej,
- niedokończone mostki typu "bo inaczej", "więc", "żeby", "jeśli", jeśli nie prowadzą do logicznego wyjaśnienia,
- krótkie restartujące frazy przed poprawną wersją zdania,
- fragmenty, które rozbijają flow i nie wnoszą treści.

Nie usuwaj:
- celowych powtórzeń retorycznych,
- ważnych przejść logicznych,
- fragmentów potrzebnych do zrozumienia następnej myśli,
- końcówek zdań, które domykają sens, np. "zanim nagrasz",
- naturalnych potknięć, jeśli zdanie nadal jest zrozumiałe i brzmi autentycznie,
- fragmentu tylko dlatego, że zawiera słowo "no", "więc", "ale", "bo" albo "jeśli".

Zasady decyzji:
- Dla drop_ranges i review_ranges podawaj start_word_id i end_word_id jako inclusive range.
- Dla thought_blocks także podawaj start_word_id i end_word_id jako inclusive range.
- Wybieraj minimalny zakres słów do usunięcia. Nie wycinaj słów potrzebnych do sensu zdania.
- Jeśli wskazujesz false start, zakończ zakres na ostatnim słowie nieudanej próby i zostaw pełną wersję.
- Jeśli fragment jest częścią ważnego thought_block, ale nie jesteś pewien, czy można go bezpiecznie wyciąć, użyj review_ranges.
- Jeśli nie masz pewności, użyj review_ranges zamiast drop_ranges.
- Confidence podawaj jako liczbę od 0 do 1.
- Reason ma być konkretny i montażowy, nie ogólnikowy.
- Timestampy start/end mają odpowiadać wybranym word_id.
"""


USER_PROMPT_TEMPLATE = """Przeanalizuj poniższy raw_transcription.json.

Kontekst:
To jest transkrypcja surowego nagrania edukacyjnego/marketingowego. Osoba często nagrywa kilka prób tej samej myśli. Chcemy stworzyć dynamiczny, ale logiczny edit.

Zwróć decyzje montażowe:
- thought_blocks: pełne jednostki sensu, których nie należy rozcinać przypadkiem
- drop_ranges: fragmenty do wycięcia automatycznie
- review_ranges: fragmenty podejrzane, ale wymagające sprawdzenia
- keep_notes: istotne uwagi o fragmentach, których nie należy wycinać
- overall_notes: krótki opis najważniejszych problemów w nagraniu

Każda decyzja ma być oparta na sensie wypowiedzi, nie tylko na pojedynczych słowach. Używaj word_id, nie wymyślaj własnych identyfikatorów.

Role dla thought_blocks:
- section_heading
- transition_question
- structure_step
- core_explanation
- case_study
- result
- aside
- other

Transkrypcja:
{transcript_json}
"""


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
    "required": ["thought_blocks", "drop_ranges", "review_ranges", "keep_notes", "overall_notes"],
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


def build_user_prompt(transcript: dict[str, Any]) -> str:
    transcript_json = json.dumps(compact_transcript_for_prompt(transcript), ensure_ascii=False, indent=2)
    return USER_PROMPT_TEMPLATE.format(transcript_json=transcript_json)


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
