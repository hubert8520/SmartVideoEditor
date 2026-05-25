#!/usr/bin/env python3
"""Create an edited video by removing silence, fillers, and repeated takes."""

from __future__ import annotations

import argparse
import difflib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
RAW_DIR = PROJECT_ROOT / "raw"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
EDITED_DIR = PROJECT_ROOT / "edited"
DEFAULT_OUTPUT_PATH = EDITED_DIR / "edited_video.mp4"
EDIT_DECISIONS_PATH = ARTIFACTS_DIR / "edit_decisions.json"
LLM_EDIT_DECISIONS_PATH = ARTIFACTS_DIR / "llm_edit_decisions.json"
REPAIR_PLAN_PATH = ARTIFACTS_DIR / "repair_plan.json"
QUALITY_REPORT_PATH = ARTIFACTS_DIR / "final_quality_report.json"
QUALITY_TRANSCRIPT_PATH = ARTIFACTS_DIR / "edited_transcription.json"
TRANSCRIPT_CANDIDATES = [
    ARTIFACTS_DIR / "raw_transcription.json",
    ARTIFACTS_DIR / "raw_transcrpition.json",
]
RAW_MEDIA_EXTENSIONS = {
    ".aac",
    ".aiff",
    ".avi",
    ".flac",
    ".m4a",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".ogg",
    ".wav",
    ".webm",
}
FILLER_RE = re.compile(r"^(?:e+|y+|a{2,}|m+|hm+|em+|um+|uh+|eh+)$")
SOFT_FILLERS = {
    "kurde",
    "yyy",
    "yyyy",
    "eee",
    "eeee",
    "aaaa",
    "mmm",
    "hmm",
    "emmm",
    "kurwa",
}
FALSE_START_PHRASES = (
    "nie tak",
    "jeszcze raz",
    "od nowa",
    "od początku",
    "od poczatku",
    "wróć",
    "wroc",
    "powtórz",
    "powtorz",
    "do wycięcia",
    "do wyciecia",
    "stop",
)
FALSE_START_SINGLETONS = {
    "ciecie",
    "cięcie",
    "jeszcze",
    "powtorka",
    "powtórka",
    "stop",
}
SELF_CORRECTION_MARKERS = (
    "znaczy",
    "to znaczy",
    "sorry",
    "wróć",
    "wroc",
    "nie tak",
)
RESTART_CHAIN_TOKENS = {
    "ale",
    "bo",
    "czyli",
    "dobra",
    "i",
    "jak",
    "no",
    "to",
    "wiec",
    "więc",
    "zeby",
    "żeby",
}
EXPLANATION_BRIDGE_PHRASES = {
    ("bo",),
    ("bo", "inaczej"),
    ("ale",),
    ("czyli",),
}
SECTION_START_TOKENS = {
    "1",
    "2",
    "3",
    "druga",
    "drugi",
    "pierwsza",
    "pierwsze",
    "pierwszy",
    "punkt",
    "trzecia",
    "trzeci",
}
MAX_RESTART_CHAIN_WORDS = 2
MIN_RESTART_CHAIN_SPAN = 1.4
MIN_RESTART_CHAIN_ITEMS = 2
BRIDGE_CLUSTER_TAIL_CUT_SECONDS = 4.0
TRAILING_RESTART_TOKENS = {
    "ale",
    "bo",
    "czyli",
    "no",
    "to",
    "wiec",
    "więc",
    "zeby",
    "żeby",
}
TAIL_RESTART_MIN_SECONDS = 1.1
TAIL_RESTART_MAX_SECONDS = 2.1
PRE_RESTART_NOISE_SECONDS = 1.6
SHORT_BLOCK_SILENCE_TRIM_ROLES = {"section_heading", "structure_step"}


@dataclass
class TranscriptWord:
    id: int
    timestamp: float
    end: float
    text: str
    normalized: str
    speaker: str | None = None
    confidence: float | None = None


@dataclass
class TranscriptEntry:
    index: int
    timestamp: float
    text: str
    normalized: str
    tokens: list[str]
    content_tokens: list[str]
    word_ids: list[int] = field(default_factory=list)
    end: float = 0.0
    drop: bool = False
    reasons: list[str] = field(default_factory=list)
    review_reasons: list[str] = field(default_factory=list)


@dataclass
class PartialDropWindow:
    start: float
    end: float
    reason: str
    source_text: str
    word_ids: list[int] = field(default_factory=list)
    source: str = "heuristic"
    force: bool = False


@dataclass
class ReviewWindow:
    start: float
    end: float
    reason: str
    source_text: str
    word_ids: list[int] = field(default_factory=list)
    source: str = "heuristic"


@dataclass
class ThoughtBlock:
    id: int
    start: float
    end: float
    word_ids: list[int]
    text: str
    role: str = "thought"


@dataclass
class CutPlannerResult:
    drop_windows: list[tuple[float, float]]
    applied_windows: list[dict[str, object]] = field(default_factory=list)
    blocked_windows: list[dict[str, object]] = field(default_factory=list)
    boundary_issues: list[dict[str, object]] = field(default_factory=list)
    simulated_text: str = ""


@dataclass
class KeepPlannerSettings:
    padding: float
    word_head_padding: float
    word_tail_padding: float
    short_block_tail_padding: float
    short_block_max_words: int
    short_block_silence_trim: bool
    short_block_silence_min_duration: float
    short_block_silence_window: float
    short_block_min_spoken_before_trim: float
    use_word_mask: bool


@dataclass
class LlmDecisionSummary:
    path: str
    status: str
    min_confidence: float
    applied_drop_ranges: list[dict[str, object]] = field(default_factory=list)
    skipped_drop_ranges: list[dict[str, object]] = field(default_factory=list)
    review_ranges: list[dict[str, object]] = field(default_factory=list)
    keep_notes: list[dict[str, object]] = field(default_factory=list)
    thought_blocks: list[dict[str, object]] = field(default_factory=list)


@dataclass
class RepairPlanSummary:
    path: str
    status: str
    repairs: list[dict[str, object]] = field(default_factory=list)
    skipped_repairs: list[dict[str, object]] = field(default_factory=list)
    forced_keep_intervals: list[tuple[float, float]] = field(default_factory=list)
    forced_keep_records: list[dict[str, object]] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read artifacts/raw_transcription.json, remove silence/fillers/repeated "
            "takes from raw/ media, and write edited/edited_video.mp4."
        )
    )
    parser.add_argument(
        "--video-name",
        help=(
            "Media file name inside raw/. If omitted, the only supported media file "
            "from raw/ is selected automatically."
        ),
    )
    parser.add_argument(
        "--transcript",
        type=Path,
        help="Transcript JSON path. Defaults to artifacts/raw_transcription.json.",
    )
    parser.add_argument(
        "--llm-decisions",
        type=Path,
        default=LLM_EDIT_DECISIONS_PATH,
        help=(
            "LLM edit decisions JSON path. Defaults to artifacts/llm_edit_decisions.json "
            "and is used automatically when the file exists."
        ),
    )
    parser.add_argument(
        "--repair-plan",
        type=Path,
        help=(
            "Optional repair plan generated from final_quality_report.json. "
            "Uses raw word_id/timestamps and renders from the original media."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Edited video output path. Default: edited/edited_video.mp4.",
    )
    parser.add_argument(
        "--edit-decisions-output",
        type=Path,
        default=EDIT_DECISIONS_PATH,
        help="Edit decisions output path. Default: artifacts/edit_decisions.json.",
    )
    parser.add_argument(
        "--ignore-llm-decisions",
        action="store_true",
        help="Ignore artifacts/llm_edit_decisions.json and use only local heuristics.",
    )
    parser.add_argument(
        "--allow-heuristic-drops",
        action="store_true",
        help=(
            "Allow local heuristics to create automatic cuts. By default they only "
            "create review notes, except obvious filler-only fragments."
        ),
    )
    parser.add_argument(
        "--llm-min-confidence",
        type=float,
        default=0.75,
        help="Minimum confidence for automatic LLM drop ranges. Default: 0.75.",
    )
    parser.add_argument(
        "--keep-source",
        choices=("transcript", "audio"),
        default="transcript",
        help=(
            "Use transcript ranges as the main keep source, or raw non-silent audio. "
            "Default: transcript."
        ),
    )
    parser.add_argument(
        "--word-gap-merge",
        type=float,
        default=0.7,
        help=(
            "Merge neighboring word intervals separated by this many seconds or less. "
            "Default: 0.7."
        ),
    )
    parser.add_argument(
        "--thought-gap",
        type=float,
        default=1.0,
        help="Maximum gap between words inside one protected thought block. Default: 1.0.",
    )
    parser.add_argument(
        "--cut-safety-margin",
        type=float,
        default=0.06,
        help="Seconds added around word-based drop ranges. Default: 0.06.",
    )
    parser.add_argument(
        "--silence-snap-window",
        type=float,
        default=0.18,
        help="Snap cut boundaries to nearby silence boundaries within this window. Default: 0.18.",
    )
    parser.add_argument(
        "--boundary-context-words",
        type=int,
        default=8,
        help="Words before/after each planned join used by the boundary validator. Default: 8.",
    )
    parser.add_argument(
        "--disable-boundary-validator",
        action="store_true",
        help="Do not block cuts that appear to break sentence/thought boundaries.",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.2,
        help="Seconds to keep before and after each spoken kept fragment. Default: 0.2.",
    )
    parser.add_argument(
        "--word-head-padding",
        type=float,
        default=0.05,
        help="Seconds to keep before the first word in word-mask mode. Default: 0.05.",
    )
    parser.add_argument(
        "--word-tail-padding",
        type=float,
        default=0.06,
        help="Seconds to keep after the last word in word-mask mode. Default: 0.06.",
    )
    parser.add_argument(
        "--short-block-tail-padding",
        type=float,
        default=0.02,
        help="Tail padding for short headings/thoughts in word-mask mode. Default: 0.02.",
    )
    parser.add_argument(
        "--short-block-max-words",
        type=int,
        default=3,
        help="Word count treated as a short block for special tail padding. Default: 3.",
    )
    parser.add_argument(
        "--disable-short-block-silence-trim",
        action="store_true",
        help=(
            "Do not trim short block tails to the first detected short silence. "
            "Enabled by default."
        ),
    )
    parser.add_argument(
        "--short-block-silence-min-duration",
        type=float,
        default=0.08,
        help="Minimum silence duration used only for short-block tail trimming. Default: 0.08.",
    )
    parser.add_argument(
        "--short-block-silence-window",
        type=float,
        default=0.45,
        help=(
            "How far after the last word end to look for a trimming silence in "
            "short blocks. Default: 0.45."
        ),
    )
    parser.add_argument(
        "--short-block-min-spoken-before-trim",
        type=float,
        default=0.25,
        help=(
            "Minimum spoken duration inside the last word before a short-block "
            "silence can trim its tail. Default: 0.25."
        ),
    )
    parser.add_argument(
        "--disable-word-mask",
        action="store_true",
        help="Use interval padding instead of word-level keep masking.",
    )
    parser.add_argument(
        "--silence-threshold",
        default="-35dB",
        help="ffmpeg silencedetect threshold. Default: -35dB.",
    )
    parser.add_argument(
        "--min-silence-duration",
        type=float,
        default=0.35,
        help="Minimum silence duration to remove, in seconds. Default: 0.35.",
    )
    parser.add_argument(
        "--min-clip-duration",
        type=float,
        default=0.4,
        help="Discard final keep clips shorter than this many seconds. Default: 0.4.",
    )
    parser.add_argument(
        "--lookahead",
        type=int,
        default=3,
        help="How many following transcript items to inspect for repeated takes. Default: 3.",
    )
    parser.add_argument(
        "--max-repeat-gap",
        type=float,
        default=15.0,
        help="Only compare repeated takes within this many seconds. Default: 15.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write edit decisions only; do not render edited/edited_video.mp4.",
    )
    parser.add_argument(
        "--skip-quality-check",
        action="store_true",
        help="Do not run the post-render LLM quality check.",
    )
    parser.add_argument(
        "--quality-check-strict",
        action="store_true",
        help="Fail the command if the post-render quality check cannot run.",
    )
    parser.add_argument(
        "--quality-provider",
        choices=("deepgram", "openai"),
        default="deepgram",
        help="Transcription provider for post-render quality check. Default: deepgram.",
    )
    parser.add_argument(
        "--quality-language",
        help="Language code for post-render quality check. Defaults to auto-detect.",
    )
    parser.add_argument(
        "--quality-model",
        help="OpenAI LLM model for post-render quality check.",
    )
    parser.add_argument(
        "--quality-transcript-output",
        type=Path,
        default=QUALITY_TRANSCRIPT_PATH,
        help="Output path for post-render transcription.",
    )
    parser.add_argument(
        "--quality-report-output",
        type=Path,
        default=QUALITY_REPORT_PATH,
        help="Output path for post-render quality report.",
    )
    return parser.parse_args()


def fail(message: str) -> None:
    print(f"Error: {message}", file=sys.stderr)
    raise SystemExit(1)


def resolve_ffmpeg_executable() -> str:
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg

    try:
        import imageio_ffmpeg
    except ImportError:
        fail("Missing ffmpeg. Install dependencies with: pip install -r requirements.txt")

    return imageio_ffmpeg.get_ffmpeg_exe()


def supported_raw_media_files() -> list[Path]:
    if not RAW_DIR.exists():
        fail(f"Input folder does not exist: {RAW_DIR}")

    return sorted(
        path
        for path in RAW_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in RAW_MEDIA_EXTENSIONS
    )


def resolve_raw_media_path(video_name: str | None) -> Path:
    if video_name:
        requested = Path(video_name)
        if requested.is_absolute() or requested.parent != Path("."):
            fail("--video-name must be a file name from raw/, not a path.")

        media_path = RAW_DIR / requested.name
        if not media_path.exists():
            fail(f"Input file does not exist in raw/: {requested.name}")
        if media_path.suffix.lower() not in RAW_MEDIA_EXTENSIONS:
            fail(f"Unsupported media file extension: {media_path.suffix}")
        return media_path

    media_files = supported_raw_media_files()
    if not media_files:
        fail(f"No supported media files found in {RAW_DIR}")
    if len(media_files) > 1:
        names = ", ".join(path.name for path in media_files)
        fail(f"More than one media file found in raw/. Choose one with --video-name. Found: {names}")

    return media_files[0]


def resolve_transcript_path(path: Path | None) -> Path:
    if path:
        if not path.exists():
            fail(f"Transcript file does not exist: {path}")
        return path

    for candidate in TRANSCRIPT_CANDIDATES:
        if candidate.exists():
            return candidate

    names = ", ".join(str(path) for path in TRANSCRIPT_CANDIDATES)
    fail(f"No transcript file found. Expected one of: {names}")


def timestamp_to_seconds(timestamp: str) -> float:
    parts = timestamp.strip().split(":")
    if len(parts) != 4:
        fail(f"Invalid timestamp format: {timestamp!r}. Expected hh:mm:ss:ms.")

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


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def collapse_repeated_letters(text: str) -> str:
    return re.sub(r"([a-ząćęłńóśźż])\1{2,}", r"\1\1", text, flags=re.IGNORECASE)


def normalize_text(text: str) -> str:
    text = re.sub(r"^speaker\s+\S+:\s*", "", text.strip(), flags=re.IGNORECASE)
    text = strip_accents(text.lower())
    text = collapse_repeated_letters(text)
    text = re.sub(r"[^a-z0-9ąćęłńóśźż]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return [token for token in normalize_text(text).split() if token]


def is_filler_token(token: str) -> bool:
    return bool(FILLER_RE.fullmatch(token)) or token in SOFT_FILLERS


def content_tokens(tokens: list[str]) -> list[str]:
    return [token for token in tokens if not is_filler_token(token)]


def legacy_items_to_word_level(data: list[object], duration: float) -> dict[str, object]:
    segments: list[dict[str, object]] = []
    words: list[dict[str, object]] = []
    word_id = 0

    for index, item in enumerate(data):
        if not isinstance(item, dict):
            fail(f"Transcript item #{index} is not an object.")
        if "timestamp" not in item or "transcription" not in item:
            fail(f"Transcript item #{index} must have timestamp and transcription.")

        start = str(item["timestamp"])
        end = str(item.get("end", ""))
        text = str(item["transcription"]).strip()
        start_seconds = timestamp_to_seconds(start)
        end_seconds = timestamp_to_seconds(end) if end else 0.0
        if end_seconds <= start_seconds:
            end_seconds = duration

        tokens = [token for token in text.split() if token]
        segment_word_ids: list[int] = []
        token_count = max(1, len(tokens))
        for token_index, token in enumerate(tokens):
            token_start = start_seconds + (end_seconds - start_seconds) * token_index / token_count
            token_end = start_seconds + (end_seconds - start_seconds) * (token_index + 1) / token_count
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
                "id": index,
                "timestamp": start,
                "end": seconds_to_timestamp(end_seconds),
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


def load_transcript(
    path: Path,
    duration: float,
) -> tuple[list[TranscriptEntry], list[TranscriptWord], dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        data = legacy_items_to_word_level(data, duration)
    if not isinstance(data, dict):
        fail("Transcript JSON must be an object with segments and words.")

    raw_segments = data.get("segments")
    raw_words = data.get("words")
    if not isinstance(raw_segments, list):
        fail("Transcript JSON must contain a 'segments' list.")
    if not isinstance(raw_words, list):
        fail("Transcript JSON must contain a 'words' list.")

    words: list[TranscriptWord] = []
    for index, item in enumerate(raw_words):
        if not isinstance(item, dict):
            fail(f"Transcript word #{index} is not an object.")
        if "id" not in item or "timestamp" not in item or "end" not in item or "word" not in item:
            fail(f"Transcript word #{index} must have id, timestamp, end, and word.")
        text = str(item["word"]).strip()
        timestamp = timestamp_to_seconds(str(item["timestamp"]))
        end = timestamp_to_seconds(str(item["end"]))
        if end <= timestamp:
            end = timestamp + 0.05
        confidence = item.get("confidence")
        words.append(
            TranscriptWord(
                id=int(item["id"]),
                timestamp=max(0.0, min(timestamp, duration)),
                end=max(0.0, min(end, duration)),
                text=text,
                normalized=normalize_text(text),
                speaker=str(item["speaker"]) if item.get("speaker") is not None else None,
                confidence=float(confidence) if confidence is not None else None,
            )
        )

    words.sort(key=lambda word: (word.timestamp, word.end, word.id))
    word_ids = {word.id for word in words}

    entries: list[TranscriptEntry] = []
    for index, item in enumerate(raw_segments):
        if not isinstance(item, dict):
            fail(f"Transcript segment #{index} is not an object.")
        if "timestamp" not in item or "transcription" not in item:
            fail(f"Transcript segment #{index} must have timestamp and transcription.")

        text = str(item["transcription"]).strip()
        tokens = tokenize(text)
        timestamp = timestamp_to_seconds(str(item["timestamp"]))
        end = timestamp_to_seconds(str(item["end"])) if "end" in item else 0.0
        entry_word_ids = [
            int(word_id)
            for word_id in item.get("word_ids", [])
            if int(word_id) in word_ids
        ]
        entries.append(
            TranscriptEntry(
                index=int(item.get("id", index)),
                timestamp=timestamp,
                text=text,
                normalized=normalize_text(text),
                tokens=tokens,
                content_tokens=content_tokens(tokens),
                word_ids=entry_word_ids,
                end=end,
            )
        )

    entries.sort(key=lambda entry: entry.timestamp)
    for index, entry in enumerate(entries):
        next_timestamp = entries[index + 1].timestamp if index + 1 < len(entries) else duration
        if entry.end <= entry.timestamp:
            entry.end = next_timestamp
        entry.end = max(entry.timestamp + 0.05, min(entry.end, next_timestamp, duration))

    return entries, words, data


def load_llm_decisions(
    path: Path,
    duration: float,
    words: list[TranscriptWord],
    min_confidence: float,
    ignore: bool,
) -> tuple[list[PartialDropWindow], LlmDecisionSummary]:
    summary = LlmDecisionSummary(
        path=str(path),
        status="ignored" if ignore else "missing",
        min_confidence=min_confidence,
    )
    if ignore:
        return [], summary

    if not path.exists():
        return [], summary

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"Invalid LLM decisions JSON in {path}: {exc}")

    if not isinstance(data, dict):
        fail("LLM decisions JSON must be an object.")

    drop_ranges = data.get("drop_ranges", [])
    review_ranges = data.get("review_ranges", [])
    keep_notes = data.get("keep_notes", [])
    thought_blocks = data.get("thought_blocks", [])
    if not isinstance(drop_ranges, list):
        fail("LLM decisions field 'drop_ranges' must be a list.")
    if not isinstance(review_ranges, list):
        fail("LLM decisions field 'review_ranges' must be a list.")
    if not isinstance(keep_notes, list):
        fail("LLM decisions field 'keep_notes' must be a list.")
    if thought_blocks is None:
        thought_blocks = []
    if not isinstance(thought_blocks, list):
        fail("LLM decisions field 'thought_blocks' must be a list.")

    summary.status = "loaded"
    windows: list[PartialDropWindow] = []
    words_by_id = {word.id: word for word in words}
    for index, item in enumerate(drop_ranges):
        if not isinstance(item, dict):
            fail(f"LLM drop_ranges[{index}] must be an object.")

        try:
            confidence = float(item.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0

        raw_start = str(item.get("start", ""))
        raw_end = str(item.get("end", ""))
        reason = str(item.get("reason", "LLM drop range")).strip() or "LLM drop range"
        category = str(item.get("reason_category", "other")).strip() or "other"
        affected_text = str(item.get("affected_text", "")).strip()

        word_ids: list[int] = []
        if "start_word_id" in item and "end_word_id" in item and words_by_id:
            start_word_id = int(item["start_word_id"])
            end_word_id = int(item["end_word_id"])
            if end_word_id < start_word_id:
                start_word_id, end_word_id = end_word_id, start_word_id
            word_ids = [
                word_id
                for word_id in range(start_word_id, end_word_id + 1)
                if word_id in words_by_id
            ]

        if word_ids:
            selected_words = [words_by_id[word_id] for word_id in word_ids]
            start = min(word.timestamp for word in selected_words)
            end = max(word.end for word in selected_words)
        else:
            start = max(0.0, min(timestamp_to_seconds(raw_start), duration))
            end = max(start, min(timestamp_to_seconds(raw_end), duration))

        record = {
            "start": seconds_to_timestamp(start),
            "end": seconds_to_timestamp(end),
            "start_word_id": word_ids[0] if word_ids else item.get("start_word_id"),
            "end_word_id": word_ids[-1] if word_ids else item.get("end_word_id"),
            "confidence": confidence,
            "reason_category": category,
            "reason": reason,
            "affected_text": affected_text,
        }

        if end <= start:
            record["skip_reason"] = "empty_or_invalid_range"
            summary.skipped_drop_ranges.append(record)
            continue
        if confidence < min_confidence:
            record["skip_reason"] = "below_min_confidence"
            summary.skipped_drop_ranges.append(record)
            continue

        windows.append(
            PartialDropWindow(
                start=start,
                end=end,
                reason=f"llm_{category}: {reason}",
                source_text=affected_text,
                word_ids=word_ids,
                source="llm",
            )
        )
        summary.applied_drop_ranges.append(record)

    for index, item in enumerate(review_ranges):
        if not isinstance(item, dict):
            fail(f"LLM review_ranges[{index}] must be an object.")
        summary.review_ranges.append(
            {
                "start": str(item.get("start", "")),
                "end": str(item.get("end", "")),
                "start_word_id": item.get("start_word_id"),
                "end_word_id": item.get("end_word_id"),
                "confidence": item.get("confidence"),
                "reason": str(item.get("reason", "")),
                "affected_text": str(item.get("affected_text", "")),
                "question": str(item.get("question", "")),
            }
        )

    for item in keep_notes:
        if not isinstance(item, dict):
            continue
        summary.keep_notes.append(
            {
                "start": str(item.get("start", "")),
                "end": str(item.get("end", "")),
                "start_word_id": item.get("start_word_id"),
                "end_word_id": item.get("end_word_id"),
                "note": str(item.get("note", "")),
                "affected_text": str(item.get("affected_text", "")),
            }
        )

    for item in thought_blocks:
        if not isinstance(item, dict):
            continue
        summary.thought_blocks.append(
            {
                "id": item.get("id"),
                "start": str(item.get("start", "")),
                "end": str(item.get("end", "")),
                "start_word_id": item.get("start_word_id"),
                "end_word_id": item.get("end_word_id"),
                "role": str(item.get("role", "thought")),
                "text": str(item.get("text", "")),
                "must_keep": bool(item.get("must_keep", False)),
            }
        )

    return windows, summary


def word_ids_from_repair_item(
    item: dict[str, object],
    words_by_id: dict[int, TranscriptWord],
) -> list[int]:
    try:
        start_word_id = int(item["start_word_id"])
        end_word_id = int(item["end_word_id"])
    except (KeyError, TypeError, ValueError):
        return []

    if end_word_id < start_word_id:
        start_word_id, end_word_id = end_word_id, start_word_id
    return [
        word_id
        for word_id in range(start_word_id, end_word_id + 1)
        if word_id in words_by_id
    ]


def interval_record(
    start: float,
    end: float,
    reason: str,
    repair_type: str,
    word_ids: list[int] | None = None,
) -> dict[str, object]:
    return {
        "type": repair_type,
        "start": seconds_to_timestamp(start),
        "end": seconds_to_timestamp(end),
        "reason": reason,
        "word_ids": word_ids or [],
    }


def load_repair_plan(
    path: Path | None,
    duration: float,
    words: list[TranscriptWord],
) -> tuple[list[PartialDropWindow], RepairPlanSummary]:
    summary = RepairPlanSummary(
        path=str(path or REPAIR_PLAN_PATH),
        status="missing",
    )
    if path is None:
        return [], summary
    summary.path = str(path)
    if not path.exists():
        fail(f"Repair plan does not exist: {path}")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"Invalid repair plan JSON in {path}: {exc}")

    if not isinstance(data, dict):
        fail("Repair plan JSON must be an object.")
    repairs = data.get("repairs", [])
    if not isinstance(repairs, list):
        fail("Repair plan field 'repairs' must be a list.")

    summary.status = "loaded"
    words_by_id = {word.id: word for word in words}
    drop_windows: list[PartialDropWindow] = []

    for index, item in enumerate(repairs):
        if not isinstance(item, dict):
            summary.skipped_repairs.append(
                {"index": index, "skip_reason": "repair_is_not_an_object"}
            )
            continue

        repair_type = str(item.get("type", "")).strip()
        reason = str(item.get("reason", "repair plan")).strip() or "repair plan"
        source_issue = item.get("source_issue", {})
        if repair_type == "force_drop_words":
            word_ids = word_ids_from_repair_item(item, words_by_id)
            if not word_ids:
                skipped = dict(item)
                skipped["skip_reason"] = "missing_or_invalid_word_ids"
                summary.skipped_repairs.append(skipped)
                continue

            selected = [words_by_id[word_id] for word_id in word_ids]
            start = min(word.timestamp for word in selected)
            end = max(word.end for word in selected)
            drop_windows.append(
                PartialDropWindow(
                    start=start,
                    end=end,
                    reason=f"repair_force_drop: {reason}",
                    source_text=str(item.get("affected_text", "")),
                    word_ids=word_ids,
                    source="repair",
                    force=True,
                )
            )
            record = interval_record(start, end, reason, repair_type, word_ids)
            record["source_issue"] = source_issue
            summary.repairs.append(record)
            continue

        if repair_type in {"force_keep_words", "force_keep_interval", "expand_keep_between_words"}:
            word_ids = word_ids_from_repair_item(item, words_by_id)
            padding = float(item.get("padding", 0.0) or 0.0)
            if repair_type in {"force_keep_words", "expand_keep_between_words"} and word_ids:
                selected = [words_by_id[word_id] for word_id in word_ids]
                start = min(word.timestamp for word in selected) - padding
                end = max(word.end for word in selected) + padding
            else:
                try:
                    start = timestamp_to_seconds(str(item["raw_start"]))
                    end = timestamp_to_seconds(str(item["raw_end"]))
                except (KeyError, TypeError, ValueError):
                    skipped = dict(item)
                    skipped["skip_reason"] = "missing_or_invalid_raw_interval"
                    summary.skipped_repairs.append(skipped)
                    continue
                start -= padding
                end += padding

            start = max(0.0, min(start, duration))
            end = max(start, min(end, duration))
            if end <= start:
                skipped = dict(item)
                skipped["skip_reason"] = "empty_interval"
                summary.skipped_repairs.append(skipped)
                continue

            summary.forced_keep_intervals.append((start, end))
            record = interval_record(start, end, reason, repair_type, word_ids)
            record["source_issue"] = source_issue
            summary.forced_keep_records.append(record)
            summary.repairs.append(record)
            continue

        skipped = dict(item)
        skipped["skip_reason"] = f"unsupported_repair_type:{repair_type}"
        summary.skipped_repairs.append(skipped)

    return drop_windows, summary


def get_media_duration(ffmpeg: str, media_path: Path) -> float:
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", str(media_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", result.stderr)
    if not match:
        fail("Could not read media duration from ffmpeg output.")

    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def detect_silences(
    ffmpeg: str,
    media_path: Path,
    threshold: str,
    min_duration: float,
) -> list[tuple[float, float]]:
    result = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-i",
            str(media_path),
            "-af",
            f"silencedetect=noise={threshold}:d={min_duration}",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    starts: list[float] = []
    silences: list[tuple[float, float]] = []
    for line in result.stderr.splitlines():
        start_match = re.search(r"silence_start:\s*([0-9.]+)", line)
        if start_match:
            starts.append(float(start_match.group(1)))
            continue

        end_match = re.search(r"silence_end:\s*([0-9.]+)", line)
        if end_match and starts:
            silences.append((starts.pop(0), float(end_match.group(1))))

    return silences


def speech_intervals_from_silences(
    silences: list[tuple[float, float]],
    duration: float,
) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    cursor = 0.0

    for start, end in sorted(silences):
        start = max(0.0, min(start, duration))
        end = max(start, min(end, duration))
        if start > cursor:
            intervals.append((cursor, start))
        cursor = max(cursor, end)

    if cursor < duration:
        intervals.append((cursor, duration))

    return intervals


def has_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    normalized = normalize_text(text)
    return any(phrase in normalized for phrase in phrases)


def mark_drop(entry: TranscriptEntry, reason: str) -> None:
    entry.drop = True
    if reason not in entry.reasons:
        entry.reasons.append(reason)


def mark_review(entry: TranscriptEntry, reason: str) -> None:
    if reason not in entry.review_reasons:
        entry.review_reasons.append(reason)


def text_quality(entry: TranscriptEntry) -> float:
    filler_count = len([token for token in entry.tokens if is_filler_token(token)])
    quality = len(entry.content_tokens) - filler_count * 1.5
    if has_phrase(entry.text, FALSE_START_PHRASES):
        quality -= 5
    if has_phrase(entry.text, SELF_CORRECTION_MARKERS) and len(entry.tokens) <= 14:
        quality -= 3
    return quality


def similarity(a: TranscriptEntry, b: TranscriptEntry) -> float:
    a_text = " ".join(a.content_tokens)
    b_text = " ".join(b.content_tokens)
    if not a_text or not b_text:
        return 0.0

    sequence = difflib.SequenceMatcher(None, a_text, b_text).ratio()
    a_set = set(a.content_tokens)
    b_set = set(b.content_tokens)
    jaccard = len(a_set & b_set) / max(1, len(a_set | b_set))
    return max(sequence, jaccard)


def is_prefix_fragment(fragment: list[str], full: list[str]) -> bool:
    if not fragment or not full or len(fragment) > len(full):
        return False
    return fragment == full[: len(fragment)]


def is_standalone_repeat_of_previous(entry: TranscriptEntry, previous: TranscriptEntry) -> bool:
    current = entry.content_tokens
    prior = previous.content_tokens
    if not current or not prior or len(current) > 4:
        return False
    if tuple(current) in EXPLANATION_BRIDGE_PHRASES:
        return False

    if current == prior:
        return True
    if len(current) <= len(prior) and current == prior[-len(current) :]:
        return True
    if len(current) <= len(prior) and current == prior[: len(current)]:
        return True

    return False


def matching_bridge_phrase(tokens: list[str]) -> tuple[str, ...] | None:
    for phrase in sorted(EXPLANATION_BRIDGE_PHRASES, key=len, reverse=True):
        if len(tokens) >= len(phrase) and tuple(tokens[-len(phrase) :]) == phrase:
            return phrase
    return None


def starts_section(entry: TranscriptEntry) -> bool:
    return bool(entry.content_tokens) and entry.content_tokens[0] in SECTION_START_TOKENS


def is_restart_chain_entry(entry: TranscriptEntry) -> bool:
    return (
        0 < len(entry.content_tokens) <= MAX_RESTART_CHAIN_WORDS
        and all(token in RESTART_CHAIN_TOKENS for token in entry.content_tokens)
    )


def mark_restart_chains(entries: list[TranscriptEntry]) -> None:
    index = 0
    while index < len(entries):
        if not is_restart_chain_entry(entries[index]):
            index += 1
            continue

        chain_start = index
        while index < len(entries) and is_restart_chain_entry(entries[index]):
            index += 1

        chain = entries[chain_start:index]
        next_entry = entries[index] if index < len(entries) else None
        chain_span = (next_entry.timestamp if next_entry else chain[-1].end) - chain[0].timestamp
        has_real_continuation = next_entry is not None and len(next_entry.content_tokens) >= 3

        if (
            len(chain) >= MIN_RESTART_CHAIN_ITEMS
            and chain_span >= MIN_RESTART_CHAIN_SPAN
            and has_real_continuation
        ):
            for entry in chain:
                mark_review(entry, "restart_chain_before_better_take")


def trailing_restart_cut_seconds(entry: TranscriptEntry) -> float:
    duration = max(0.0, entry.end - entry.timestamp)
    return min(
        TAIL_RESTART_MAX_SECONDS,
        max(TAIL_RESTART_MIN_SECONDS, duration * 0.45),
    )


def entry_ends_with_restart(entry: TranscriptEntry) -> bool:
    return bool(entry.content_tokens) and entry.content_tokens[-1] in TRAILING_RESTART_TOKENS


def entry_starts_jest_taki(entry: TranscriptEntry) -> bool:
    tokens = entry.content_tokens
    return tokens[:2] == ["jest", "taki"] or tokens[:2] == ["jest", "tak"]


def is_jest_taki_before_schemat(entry: TranscriptEntry, next_entry: TranscriptEntry) -> bool:
    if not entry_starts_jest_taki(entry):
        return False

    next_tokens = next_entry.content_tokens
    return (
        next_tokens[:2] == ["1", "schemat"]
        or next_tokens[:2] == ["jeden", "schemat"]
        or next_tokens[:1] == ["schemat"]
    )


def detect_heuristic_review_windows(entries: list[TranscriptEntry]) -> list[ReviewWindow]:
    windows: list[ReviewWindow] = []

    for index, entry in enumerate(entries[:-1]):
        next_entry = entries[index + 1]
        bridge_phrase = matching_bridge_phrase(entry.content_tokens)
        if bridge_phrase and len(entry.content_tokens) > len(bridge_phrase):
            cluster_index = index + 1
            bridge_cluster: list[TranscriptEntry] = []
            while cluster_index < len(entries):
                candidate = entries[cluster_index]
                if tuple(candidate.content_tokens) != bridge_phrase:
                    break
                bridge_cluster.append(candidate)
                cluster_index += 1

            next_after_cluster = entries[cluster_index] if cluster_index < len(entries) else None
            if bridge_cluster and next_after_cluster and starts_section(next_after_cluster):
                windows.append(
                    ReviewWindow(
                        start=max(entry.timestamp, entry.end - BRIDGE_CLUSTER_TAIL_CUT_SECONDS),
                        end=entry.end,
                        reason="dangling_bridge_tail_before_section",
                        source_text=entry.text,
                    )
                )
                for bridge_entry in bridge_cluster:
                    mark_review(bridge_entry, "dangling_repeated_bridge_before_section")

        if entry_ends_with_restart(entry) and len(entry.content_tokens) >= 5:
            cut = trailing_restart_cut_seconds(entry)
            windows.append(
                ReviewWindow(
                    start=max(entry.timestamp, entry.end - cut),
                    end=entry.end,
                    reason="trailing_restart_tail",
                    source_text=entry.text,
                )
            )

        if is_jest_taki_before_schemat(entry, next_entry):
            mark_review(entry, "setup_restart_before_full_phrase")

    return windows


def analyze_entries(
    entries: list[TranscriptEntry],
    lookahead: int,
    max_repeat_gap: float,
    allow_heuristic_drops: bool,
) -> tuple[list[PartialDropWindow], list[ReviewWindow]]:
    for entry in entries:
        filler_count = len([token for token in entry.tokens if is_filler_token(token)])
        filler_ratio = filler_count / max(1, len(entry.tokens))
        if not entry.content_tokens:
            mark_drop(entry, "filler_only")
        elif filler_ratio >= 0.55:
            if allow_heuristic_drops:
                mark_drop(entry, "mostly_fillers")
            else:
                mark_review(entry, "mostly_fillers")
        elif has_phrase(entry.text, FALSE_START_PHRASES):
            mark_review(entry, "false_start_marker")
        elif has_phrase(entry.text, SELF_CORRECTION_MARKERS) and len(entry.tokens) <= 14:
            mark_review(entry, "self_correction_marker")
        elif entry.normalized in FALSE_START_SINGLETONS:
            mark_review(entry, "false_start_fragment")

    mark_restart_chains(entries)

    for index, entry in enumerate(entries):
        for previous in reversed(entries[max(0, index - lookahead) : index]):
            if previous.drop:
                continue
            if entry.timestamp - previous.timestamp > max_repeat_gap:
                continue
            if is_standalone_repeat_of_previous(entry, previous):
                if allow_heuristic_drops:
                    mark_drop(entry, "standalone_repeated_fragment")
                else:
                    mark_review(entry, "standalone_repeated_fragment")
                break

        future: list[TranscriptEntry] = []
        for candidate in entries[index + 1 : index + 1 + lookahead]:
            if candidate.drop:
                continue
            if candidate.timestamp - entry.timestamp <= max_repeat_gap:
                future.append(candidate)

        future_tokens: list[str] = []
        for candidate in future:
            future_tokens.extend(candidate.content_tokens)

        if (
            entry.content_tokens
            and len(entry.content_tokens) <= 5
            and tuple(entry.content_tokens) not in EXPLANATION_BRIDGE_PHRASES
            and is_prefix_fragment(entry.content_tokens, future_tokens)
        ):
            if allow_heuristic_drops:
                mark_drop(entry, "repeated_prefix_fragment")
            else:
                mark_review(entry, "repeated_prefix_fragment")

        for candidate in future:
            if not candidate.content_tokens:
                continue
            score = similarity(entry, candidate)
            if score < 0.45:
                continue

            if text_quality(candidate) > text_quality(entry):
                if allow_heuristic_drops:
                    mark_drop(entry, "lower_quality_repeated_take")
                else:
                    mark_review(entry, "lower_quality_repeated_take")
            elif (
                len(entry.content_tokens) <= 5
                and len(candidate.content_tokens) >= len(entry.content_tokens)
                and score >= 0.65
                and tuple(entry.content_tokens) not in EXPLANATION_BRIDGE_PHRASES
            ):
                if allow_heuristic_drops:
                    mark_drop(entry, "short_repeated_take")
                else:
                    mark_review(entry, "short_repeated_take")

    review_windows = detect_heuristic_review_windows(entries)
    return [], review_windows


def word_intervals(words: list[TranscriptWord], max_gap: float) -> list[tuple[float, float]]:
    if not words:
        return []

    intervals: list[tuple[float, float]] = []
    current_start = words[0].timestamp
    current_end = words[0].end
    for word in words[1:]:
        if word.timestamp <= current_end + max_gap:
            current_end = max(current_end, word.end)
            continue
        intervals.append((current_start, current_end))
        current_start = word.timestamp
        current_end = word.end

    intervals.append((current_start, current_end))
    return merge_intervals(intervals, min_gap=max_gap)


def transcript_intervals(
    entries: list[TranscriptEntry],
    words: list[TranscriptWord],
    word_gap_merge: float,
) -> list[tuple[float, float]]:
    if words:
        return word_intervals(words, word_gap_merge)
    return merge_intervals((entry.timestamp, entry.end) for entry in entries)


def merge_intervals(
    intervals: list[tuple[float, float]],
    min_gap: float = 0.03,
) -> list[tuple[float, float]]:
    cleaned = sorted((start, end) for start, end in intervals if end > start)
    if not cleaned:
        return []

    merged = [cleaned[0]]
    for start, end in cleaned[1:]:
        previous_start, previous_end = merged[-1]
        if start <= previous_end + min_gap:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))

    return merged


def subtract_intervals(
    intervals: list[tuple[float, float]],
    removals: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    result = intervals[:]

    for remove_start, remove_end in merge_intervals(removals):
        next_result: list[tuple[float, float]] = []
        for start, end in result:
            if remove_end <= start or remove_start >= end:
                next_result.append((start, end))
                continue

            if remove_start > start:
                next_result.append((start, remove_start))
            if remove_end < end:
                next_result.append((remove_end, end))

        result = next_result

    return merge_intervals(result)


def pad_intervals(
    intervals: list[tuple[float, float]],
    padding: float,
    duration: float,
) -> list[tuple[float, float]]:
    return merge_intervals(
        [
            (max(0.0, start - padding), min(duration, end + padding))
            for start, end in intervals
        ]
    )


def thought_blocks_by_word_id(thought_blocks: list[ThoughtBlock]) -> dict[int, ThoughtBlock]:
    lookup: dict[int, ThoughtBlock] = {}
    for block in thought_blocks:
        for word_id in block.word_ids:
            lookup[word_id] = block
    return lookup


def trim_short_block_end_to_silence(
    word: TranscriptWord,
    block: ThoughtBlock,
    original_end: float,
    silences: list[tuple[float, float]],
    settings: KeepPlannerSettings,
    duration: float,
) -> tuple[float, dict[str, object] | None]:
    if not settings.short_block_silence_trim:
        return original_end, None

    earliest_trim_start = word.timestamp + settings.short_block_min_spoken_before_trim
    latest_trim_start = word.end + settings.short_block_silence_window

    for silence_start, silence_end in silences:
        if silence_end <= word.timestamp:
            continue
        if silence_start < earliest_trim_start:
            continue
        if silence_start > latest_trim_start:
            continue

        trimmed_end = min(duration, silence_start + settings.short_block_tail_padding)
        if trimmed_end >= original_end:
            continue

        return trimmed_end, {
            "word_id": word.id,
            "word": word.text,
            "block_id": block.id,
            "block_role": block.role,
            "block_text": block.text,
            "original_end": seconds_to_timestamp(original_end),
            "trimmed_end": seconds_to_timestamp(trimmed_end),
            "silence_start": seconds_to_timestamp(silence_start),
            "silence_end": seconds_to_timestamp(silence_end),
        }

    return original_end, None


def word_mask_intervals(
    words: list[TranscriptWord],
    thought_blocks: list[ThoughtBlock],
    silences: list[tuple[float, float]],
    settings: KeepPlannerSettings,
    duration: float,
) -> tuple[list[tuple[float, float]], list[dict[str, object]]]:
    if not words:
        return [], []

    block_lookup = thought_blocks_by_word_id(thought_blocks)
    intervals: list[tuple[float, float]] = []
    trim_records: list[dict[str, object]] = []
    for index, word in enumerate(words):
        block = block_lookup.get(word.id)
        block_word_count = len(block.word_ids) if block else 1
        is_block_last_word = block is not None and word.id == max(block.word_ids)
        is_short_block = (
            block is not None
            and block_word_count <= settings.short_block_max_words
            and is_block_last_word
        )
        tail_padding = (
            settings.short_block_tail_padding
            if is_short_block
            else settings.word_tail_padding
        )

        start = max(0.0, word.timestamp - settings.word_head_padding)
        end = min(duration, word.end + tail_padding)
        trim_record = None
        if (
            is_short_block
            and block is not None
            and block.role in SHORT_BLOCK_SILENCE_TRIM_ROLES
        ):
            end, trim_record = trim_short_block_end_to_silence(
                word,
                block,
                end,
                silences,
                settings,
                duration,
            )
        if trim_record:
            trim_records.append(trim_record)

        if index > 0:
            previous = words[index - 1]
            if previous.end <= word.timestamp:
                start = max(start, previous.end + 0.01)
        if index + 1 < len(words):
            next_word = words[index + 1]
            if next_word.timestamp >= word.end:
                end = min(end, next_word.timestamp - 0.01)

        if end > start:
            intervals.append((start, end))

    return merge_intervals(intervals), trim_records


def words_for_window(
    window: PartialDropWindow,
    words: list[TranscriptWord],
    words_by_id: dict[int, TranscriptWord],
) -> list[TranscriptWord]:
    if window.word_ids:
        return [
            words_by_id[word_id]
            for word_id in window.word_ids
            if word_id in words_by_id
        ]

    return [
        word
        for word in words
        if word.end > window.start and word.timestamp < window.end
    ]


def snap_drop_boundary_to_silence(
    value: float,
    silences: list[tuple[float, float]],
    snap_window: float,
    direction: str,
) -> float:
    best_value = value
    best_distance = snap_window
    for start, end in silences:
        for candidate in (start, end):
            if direction == "start" and candidate > value:
                continue
            if direction == "end" and candidate < value:
                continue
            distance = abs(value - candidate)
            if distance <= best_distance:
                best_distance = distance
                best_value = candidate
    return best_value


def word_ids_from_range(start_id: object, end_id: object, words_by_id: dict[int, TranscriptWord]) -> list[int]:
    try:
        start_word_id = int(start_id)
        end_word_id = int(end_id)
    except (TypeError, ValueError):
        return []
    if end_word_id < start_word_id:
        start_word_id, end_word_id = end_word_id, start_word_id
    return [
        word_id
        for word_id in range(start_word_id, end_word_id + 1)
        if word_id in words_by_id
    ]


def word_text(words: list[TranscriptWord]) -> str:
    return " ".join(word.text for word in words).strip()


def clean_word_tail(word: str) -> str:
    return word.strip().strip(".,!?;:…").lower()


def build_thought_blocks(
    entries: list[TranscriptEntry],
    words: list[TranscriptWord],
    llm_summary: LlmDecisionSummary,
    thought_gap: float,
) -> list[ThoughtBlock]:
    words_by_id = {word.id: word for word in words}
    blocks: list[ThoughtBlock] = []
    used_ids: set[int] = set()

    for item in llm_summary.thought_blocks:
        ids = word_ids_from_range(item.get("start_word_id"), item.get("end_word_id"), words_by_id)
        if not ids:
            continue
        selected = [words_by_id[word_id] for word_id in ids]
        blocks.append(
            ThoughtBlock(
                id=len(blocks),
                start=min(word.timestamp for word in selected),
                end=max(word.end for word in selected),
                word_ids=ids,
                text=str(item.get("text") or word_text(selected)),
                role=str(item.get("role") or "thought"),
            )
        )
        used_ids.update(ids)

    if words:
        current: list[TranscriptWord] = [words[0]]
        for word in words[1:]:
            previous = current[-1]
            gap = word.timestamp - previous.end
            previous_tail = str(previous.text).strip()
            sentence_boundary = previous_tail.endswith((".", "!", "?")) and gap > 0.18
            if gap > thought_gap or sentence_boundary:
                blocks.append(make_thought_block(len(blocks), current, entries))
                current = []
            current.append(word)
        if current:
            blocks.append(make_thought_block(len(blocks), current, entries))

    return blocks


def make_thought_block(
    block_id: int,
    block_words: list[TranscriptWord],
    entries: list[TranscriptEntry],
) -> ThoughtBlock:
    text = word_text(block_words)
    normalized = normalize_text(text)
    role = "thought"
    tokens = normalized.split()
    if tokens[:2] == ["punkt", "pierwszy"] or tokens[:2] == ["punkt", "drugi"]:
        role = "section_heading"
    elif "?" in text or tokens[:1] in (["kto"], ["co"], ["jak"]):
        role = "question"
    elif any(token in {"pierwsze", "drugi", "trzecia", "element"} for token in tokens[:3]):
        role = "structure_step"
    elif any(entry.index >= 0 and set(entry.word_ids) & {word.id for word in block_words} for entry in entries):
        role = "thought"

    return ThoughtBlock(
        id=block_id,
        start=block_words[0].timestamp,
        end=block_words[-1].end,
        word_ids=[word.id for word in block_words],
        text=text,
        role=role,
    )


def keep_note_word_ranges(llm_summary: LlmDecisionSummary, words_by_id: dict[int, TranscriptWord]) -> list[set[int]]:
    ranges: list[set[int]] = []
    for note in llm_summary.keep_notes:
        ids = set(word_ids_from_range(note.get("start_word_id"), note.get("end_word_id"), words_by_id))
        if ids:
            ranges.append(ids)
    return ranges


def should_block_window(
    window: PartialDropWindow,
    selected_words: list[TranscriptWord],
    thought_blocks: list[ThoughtBlock],
    keep_ranges: list[set[int]],
) -> tuple[bool, str]:
    if not selected_words:
        return False, ""

    selected_ids = {word.id for word in selected_words}
    source = window.source
    reason = window.reason.lower()
    text = normalize_text(window.source_text)
    category_is_filler = "filler" in reason or all(is_filler_token(token) for token in text.split())

    if category_is_filler:
        return False, ""

    for keep_range in keep_ranges:
        if selected_ids & keep_range:
            return True, "overlaps_llm_keep_note"

    first_section_heading_start = min(
        (
            min(block.word_ids)
            for block in thought_blocks
            if block.role == "section_heading" and block.word_ids
        ),
        default=None,
    )

    for block in thought_blocks:
        block_ids = set(block.word_ids)
        overlap = selected_ids & block_ids
        if not overlap:
            continue
        covers_block = overlap == block_ids
        if (
            covers_block
            and block.role == "section_heading"
            and first_section_heading_start is not None
            and min(block.word_ids) == first_section_heading_start
        ):
            return True, "first_section_heading_protected"
        if covers_block:
            continue
        overlap_ratio = len(overlap) / max(1, len(block_ids))
        if block.role in {"section_heading", "question", "structure_step"}:
            return True, f"partial_{block.role}_cut"
        if source != "llm" and overlap_ratio > 0.15:
            return True, "heuristic_partial_thought_cut"

    return False, ""


def kept_words_after_drops(
    words: list[TranscriptWord],
    drop_ranges: list[tuple[float, float]],
) -> list[TranscriptWord]:
    kept: list[TranscriptWord] = []
    for word in words:
        if any(word.end > start and word.timestamp < end for start, end in drop_ranges):
            continue
        kept.append(word)
    return kept


def validate_boundaries(
    words: list[TranscriptWord],
    drop_windows: list[tuple[float, float]],
    context_words: int,
) -> list[dict[str, object]]:
    kept = kept_words_after_drops(words, drop_windows)
    if not kept:
        return []

    issues: list[dict[str, object]] = []
    for left, right in zip(kept, kept[1:]):
        original_gap = right.id - left.id
        if original_gap <= 1:
            continue

        before_words = [word for word in kept if max(0, left.id - context_words + 1) <= word.id <= left.id]
        after_words = [word for word in kept if right.id <= word.id <= right.id + context_words - 1]
        left_clean = clean_word_tail(left.text)
        right_clean = clean_word_tail(right.text)
        before_text = word_text(before_words)
        after_text = word_text(after_words)

        reason = ""
        if str(left.text).strip().endswith((",", "…")):
            reason = "join_after_comma_or_unfinished_clause"
        elif left_clean in {"bo", "ale", "czyli", "więc", "wiec", "żeby", "zeby", "jeśli", "jesli"}:
            reason = "join_after_bridge_word"
        elif right_clean in {
            "zacząć",
            "zaczac",
            "rozwiązywać",
            "rozwiazywac",
            "w",
            "z",
            "dla",
            "który",
            "ktory",
            "która",
            "ktora",
            "które",
            "ktore",
        }:
            reason = "join_before_continuation_word"

        if reason:
            issues.append(
                {
                    "left_word_id": left.id,
                    "right_word_id": right.id,
                    "start": seconds_to_timestamp(left.end),
                    "end": seconds_to_timestamp(right.timestamp),
                    "reason": reason,
                    "before_text": before_text,
                    "after_text": after_text,
                }
            )

    return issues


def simulated_text_after_drops(
    words: list[TranscriptWord],
    drop_windows: list[tuple[float, float]],
) -> str:
    return word_text(kept_words_after_drops(words, drop_windows))


def plan_drop_windows(
    entries: list[TranscriptEntry],
    partial_drop_windows: list[PartialDropWindow],
    words: list[TranscriptWord],
    silences: list[tuple[float, float]],
    thought_blocks: list[ThoughtBlock],
    llm_summary: LlmDecisionSummary,
    duration: float,
    cut_safety_margin: float,
    silence_snap_window: float,
    context_words: int,
    disable_boundary_validator: bool,
) -> CutPlannerResult:
    words_by_id = {word.id: word for word in words}
    keep_ranges = keep_note_word_ranges(llm_summary, words_by_id)
    raw_windows: list[PartialDropWindow] = [
        PartialDropWindow(
            start=entry.timestamp,
            end=entry.end,
            reason=";".join(entry.reasons),
            source_text=entry.text,
            word_ids=entry.word_ids,
            source="heuristic",
        )
        for entry in entries
        if entry.drop
    ]
    raw_windows.extend(partial_drop_windows)

    planned: list[tuple[float, float]] = []
    applied: list[dict[str, object]] = []
    blocked: list[dict[str, object]] = []
    for window in raw_windows:
        selected_words = words_for_window(window, words, words_by_id)
        block = False
        block_reason = ""
        if not window.force:
            block, block_reason = should_block_window(
                window,
                selected_words,
                thought_blocks,
                keep_ranges,
            )
        if block:
            blocked.append(
                {
                    "start": seconds_to_timestamp(window.start),
                    "end": seconds_to_timestamp(window.end),
                    "reason": window.reason,
                    "source_text": window.source_text,
                    "source": window.source,
                    "word_ids": [word.id for word in selected_words],
                    "force": window.force,
                    "block_reason": block_reason,
                }
            )
            continue

        if selected_words:
            selected_ids = {word.id for word in selected_words}
            first_word = min(selected_words, key=lambda word: word.timestamp)
            last_word = max(selected_words, key=lambda word: word.end)
            start_floor = 0.0
            end_ceiling = duration

            previous_words = [
                word
                for word in words
                if word.id not in selected_ids and word.end <= first_word.timestamp
            ]
            next_words = [
                word
                for word in words
                if word.id not in selected_ids and word.timestamp >= last_word.end
            ]
            if previous_words:
                start_floor = max(word.end for word in previous_words) + 0.01
            if next_words:
                end_ceiling = min(word.timestamp for word in next_words) - 0.01

            start = max(start_floor, first_word.timestamp - cut_safety_margin)
            end = min(end_ceiling, last_word.end + cut_safety_margin)
        else:
            start = window.start
            end = window.end

        start = max(0.0, min(start, duration))
        end = max(start, min(end, duration))
        start = snap_drop_boundary_to_silence(start, silences, silence_snap_window, "start")
        end = snap_drop_boundary_to_silence(end, silences, silence_snap_window, "end")
        if selected_words:
            start = max(start_floor, start)
            end = min(end_ceiling, end)
        start = max(0.0, min(start, duration))
        end = max(start, min(end, duration))
        if end > start:
            planned.append((start, end))
            applied.append(
                {
                    "start": seconds_to_timestamp(start),
                    "end": seconds_to_timestamp(end),
                    "reason": window.reason,
                    "source_text": window.source_text,
                    "source": window.source,
                    "word_ids": [word.id for word in selected_words],
                    "force": window.force,
                }
            )

    unmerged_planned = planned[:]
    planned = merge_intervals(unmerged_planned)
    boundary_issues: list[dict[str, object]] = []
    if words and not disable_boundary_validator:
        boundary_issues = validate_boundaries(words, planned, context_words)
        if boundary_issues:
            blocked_indices: set[int] = set()
            for issue in boundary_issues:
                left_id = int(issue["left_word_id"])
                right_id = int(issue["right_word_id"])
                removed_ids = set(range(left_id + 1, right_id))
                for index, record in enumerate(applied):
                    if record.get("force"):
                        continue
                    record_word_ids = {int(word_id) for word_id in record.get("word_ids", [])}
                    if record_word_ids & removed_ids:
                        blocked_indices.add(index)

            if blocked_indices:
                next_applied: list[dict[str, object]] = []
                next_planned: list[tuple[float, float]] = []
                for index, record in enumerate(applied):
                    if index in blocked_indices:
                        blocked_record = dict(record)
                        blocked_record["block_reason"] = "boundary_validator"
                        blocked.append(blocked_record)
                    else:
                        next_applied.append(record)
                        next_planned.append(unmerged_planned[index])
                applied = next_applied
                planned = merge_intervals(next_planned)
                boundary_issues = validate_boundaries(words, planned, context_words)

    return CutPlannerResult(
        drop_windows=planned,
        applied_windows=applied,
        blocked_windows=blocked,
        boundary_issues=boundary_issues,
        simulated_text=simulated_text_after_drops(words, planned) if words else "",
    )


def build_keep_intervals(
    base_intervals: list[tuple[float, float]],
    drop_windows: list[tuple[float, float]],
    settings: KeepPlannerSettings,
    duration: float,
    min_clip_duration: float,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    semantic_keep = subtract_intervals(base_intervals, drop_windows)
    padded_keep = (
        semantic_keep
        if settings.use_word_mask
        else pad_intervals(semantic_keep, settings.padding, duration)
    )
    final_keep = subtract_intervals(padded_keep, drop_windows)
    final_keep = [
        (start, end)
        for start, end in final_keep
        if end - start >= min_clip_duration
    ]
    return merge_intervals(final_keep), merge_intervals(drop_windows)


def build_timeline_map(keep_intervals: list[tuple[float, float]]) -> list[dict[str, object]]:
    timeline: list[dict[str, object]] = []
    cursor = 0.0
    for index, (raw_start, raw_end) in enumerate(keep_intervals):
        duration = raw_end - raw_start
        final_start = cursor
        final_end = cursor + duration
        timeline.append(
            {
                "id": index,
                "raw_start": seconds_to_timestamp(raw_start),
                "raw_end": seconds_to_timestamp(raw_end),
                "final_start": seconds_to_timestamp(final_start),
                "final_end": seconds_to_timestamp(final_end),
                "duration_seconds": round(duration, 6),
            }
        )
        cursor = final_end
    return timeline


def write_decisions(
    transcript_path: Path,
    transcript_metadata: dict[str, object],
    media_path: Path,
    output_path: Path,
    edit_decisions_path: Path,
    duration: float,
    silences: list[tuple[float, float]],
    keep_source: str,
    llm_summary: LlmDecisionSummary,
    repair_summary: RepairPlanSummary,
    keep_intervals: list[tuple[float, float]],
    drop_windows: list[tuple[float, float]],
    partial_drop_windows: list[PartialDropWindow],
    review_windows: list[ReviewWindow],
    planner_result: CutPlannerResult,
    thought_blocks: list[ThoughtBlock],
    short_block_trim_records: list[dict[str, object]],
    entries: list[TranscriptEntry],
    words: list[TranscriptWord],
    padding: float,
    keep_settings: KeepPlannerSettings,
    word_gap_merge: float,
    cut_safety_margin: float,
    silence_snap_window: float,
    dry_run: bool,
) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_video": str(media_path),
        "transcript": str(transcript_path),
        "transcript_metadata": transcript_metadata,
        "output_video": str(output_path),
        "duration": seconds_to_timestamp(duration),
        "padding_seconds": padding,
        "keep_planner": {
            "use_word_mask": keep_settings.use_word_mask,
            "word_head_padding": keep_settings.word_head_padding,
            "word_tail_padding": keep_settings.word_tail_padding,
            "short_block_tail_padding": keep_settings.short_block_tail_padding,
            "short_block_max_words": keep_settings.short_block_max_words,
            "short_block_silence_trim": keep_settings.short_block_silence_trim,
            "short_block_silence_min_duration": keep_settings.short_block_silence_min_duration,
            "short_block_silence_window": keep_settings.short_block_silence_window,
            "short_block_min_spoken_before_trim": (
                keep_settings.short_block_min_spoken_before_trim
            ),
            "short_block_silence_trims": short_block_trim_records,
        },
        "keep_source": keep_source,
        "word_count": len(words),
        "segment_count": len(entries),
        "cut_planner": {
            "word_gap_merge": word_gap_merge,
            "cut_safety_margin": cut_safety_margin,
            "silence_snap_window": silence_snap_window,
        },
        "thought_blocks": [
            {
                "id": block.id,
                "start": seconds_to_timestamp(block.start),
                "end": seconds_to_timestamp(block.end),
                "role": block.role,
                "text": block.text,
                "word_ids": block.word_ids,
            }
            for block in thought_blocks
        ],
        "cut_planner_review": {
            "applied_windows": planner_result.applied_windows,
            "blocked_windows": planner_result.blocked_windows,
            "boundary_issues": planner_result.boundary_issues,
            "simulated_text": planner_result.simulated_text,
        },
        "dry_run": dry_run,
        "llm_decisions": {
            "path": llm_summary.path,
            "status": llm_summary.status,
            "min_confidence": llm_summary.min_confidence,
            "applied_drop_ranges": llm_summary.applied_drop_ranges,
            "skipped_drop_ranges": llm_summary.skipped_drop_ranges,
            "review_ranges": llm_summary.review_ranges,
            "keep_notes": llm_summary.keep_notes,
            "thought_blocks": llm_summary.thought_blocks,
        },
        "repair_plan": {
            "path": repair_summary.path,
            "status": repair_summary.status,
            "repairs": repair_summary.repairs,
            "skipped_repairs": repair_summary.skipped_repairs,
            "forced_keep_intervals": repair_summary.forced_keep_records,
        },
        "removed_entries": [
            {
                "timestamp": seconds_to_timestamp(entry.timestamp),
                "end": seconds_to_timestamp(entry.end),
                "transcription": entry.text,
                "reasons": entry.reasons,
            }
            for entry in entries
            if entry.drop
        ],
        "review_entries": [
            {
                "timestamp": seconds_to_timestamp(entry.timestamp),
                "end": seconds_to_timestamp(entry.end),
                "transcription": entry.text,
                "reasons": entry.review_reasons,
            }
            for entry in entries
            if entry.review_reasons
        ],
        "partial_drop_windows": [
            {
                "start": seconds_to_timestamp(window.start),
                "end": seconds_to_timestamp(window.end),
                "source_text": window.source_text,
                "reason": window.reason,
                "word_ids": window.word_ids,
                "source": window.source,
                "force": window.force,
            }
            for window in partial_drop_windows
        ],
        "review_windows": [
            {
                "start": seconds_to_timestamp(window.start),
                "end": seconds_to_timestamp(window.end),
                "source_text": window.source_text,
                "reason": window.reason,
                "word_ids": window.word_ids,
                "source": window.source,
            }
            for window in review_windows
        ],
        "silences": [
            {
                "start": seconds_to_timestamp(start),
                "end": seconds_to_timestamp(end),
            }
            for start, end in silences
        ],
        "drop_windows": [
            {
                "start": seconds_to_timestamp(start),
                "end": seconds_to_timestamp(end),
            }
            for start, end in drop_windows
        ],
        "keep_intervals": [
            {
                "start": seconds_to_timestamp(start),
                "end": seconds_to_timestamp(end),
            }
            for start, end in keep_intervals
        ],
        "timeline_map": build_timeline_map(keep_intervals),
    }
    edit_decisions_path.parent.mkdir(parents=True, exist_ok=True)
    edit_decisions_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def render_video(
    ffmpeg: str,
    media_path: Path,
    keep_intervals: list[tuple[float, float]],
    output_path: Path,
) -> None:
    if not keep_intervals:
        fail("No intervals left to render after analysis.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="edited_video_") as temp_name:
        temp_dir = Path(temp_name)
        segment_paths: list[Path] = []

        for index, (start, end) in enumerate(keep_intervals, start=1):
            segment_path = temp_dir / f"segment_{index:04d}.mp4"
            duration = end - start
            command = [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{start:.3f}",
                "-i",
                str(media_path),
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
                "18",
                "-c:a",
                "aac",
                "-b:a",
                "160k",
                "-movflags",
                "+faststart",
                str(segment_path),
            ]
            subprocess.run(command, check=True)
            segment_paths.append(segment_path)

        concat_list = temp_dir / "concat.txt"
        concat_list.write_text(
            "".join(f"file '{path.as_posix()}'\n" for path in segment_paths),
            encoding="utf-8",
        )
        subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_list),
                "-c",
                "copy",
                str(output_path),
            ],
            check=True,
        )


def run_quality_check(args: argparse.Namespace) -> None:
    if args.skip_quality_check:
        print("Post-render quality check skipped.")
        return

    script_path = PROJECT_ROOT / "scripts" / "quality_check_edited_video.py"
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




# Phase 3 shared edit runtime helpers
#
# Keep this block close to main() while edit_video.py is being migrated. The
# original definitions remain above for now, but these stable names are rebound
# to shared package implementations before the CLI runs. This keeps the patch
# small and reversible.
from smart_video_editor.editing.runtime import (  # noqa: E402
    ARTIFACTS_DIR as ARTIFACTS_DIR,
    DEFAULT_OUTPUT_PATH as DEFAULT_OUTPUT_PATH,
    EDITED_DIR as EDITED_DIR,
    EDIT_DECISIONS_PATH as EDIT_DECISIONS_PATH,
    LLM_EDIT_DECISIONS_PATH as LLM_EDIT_DECISIONS_PATH,
    PROJECT_ROOT as PROJECT_ROOT,
    QUALITY_REPORT_PATH as QUALITY_REPORT_PATH,
    QUALITY_TRANSCRIPT_PATH as QUALITY_TRANSCRIPT_PATH,
    RAW_DIR as RAW_DIR,
    RAW_MEDIA_EXTENSIONS as RAW_MEDIA_EXTENSIONS,
    REPAIR_PLAN_PATH as REPAIR_PLAN_PATH,
    TRANSCRIPT_CANDIDATES as TRANSCRIPT_CANDIDATES,
    normalize_text as normalize_text,
    seconds_to_timestamp as seconds_to_timestamp,
    strip_accents as strip_accents,
    timestamp_to_seconds as timestamp_to_seconds,
    tokenize as tokenize,
)
from smart_video_editor.editing.decisions_io import (  # noqa: E402
    build_timeline_map as build_timeline_map,
    write_decisions as write_decisions,
)
from smart_video_editor.editing.quality import run_quality_check as run_quality_check  # noqa: E402
from smart_video_editor.media.rendering import (  # noqa: E402
    detect_silences as detect_silences,
    get_media_duration as get_media_duration,
    render_video as render_video,
    speech_intervals_from_silences as speech_intervals_from_silences,
)
from smart_video_editor.planning.decision_planner import (  # noqa: E402
    plan_drop_windows as plan_drop_windows,
    validate_boundaries as validate_boundaries,
)

def main() -> None:
    start_time = time.time()
    args = parse_args()
    if args.padding < 0:
        fail("--padding must be greater than or equal to 0.")
    if args.word_head_padding < 0:
        fail("--word-head-padding must be greater than or equal to 0.")
    if args.word_tail_padding < 0:
        fail("--word-tail-padding must be greater than or equal to 0.")
    if args.short_block_tail_padding < 0:
        fail("--short-block-tail-padding must be greater than or equal to 0.")
    if args.short_block_max_words < 1:
        fail("--short-block-max-words must be at least 1.")
    if args.short_block_silence_min_duration <= 0:
        fail("--short-block-silence-min-duration must be greater than 0.")
    if args.short_block_silence_window < 0:
        fail("--short-block-silence-window must be greater than or equal to 0.")
    if args.short_block_min_spoken_before_trim < 0:
        fail("--short-block-min-spoken-before-trim must be greater than or equal to 0.")
    if args.min_silence_duration <= 0:
        fail("--min-silence-duration must be greater than 0.")
    if args.min_clip_duration <= 0:
        fail("--min-clip-duration must be greater than 0.")
    if not 0 <= args.llm_min_confidence <= 1:
        fail("--llm-min-confidence must be between 0 and 1.")
    if args.word_gap_merge < 0:
        fail("--word-gap-merge must be greater than or equal to 0.")
    if args.cut_safety_margin < 0:
        fail("--cut-safety-margin must be greater than or equal to 0.")
    if args.silence_snap_window < 0:
        fail("--silence-snap-window must be greater than or equal to 0.")
    if args.thought_gap < 0:
        fail("--thought-gap must be greater than or equal to 0.")
    if args.boundary_context_words < 1:
        fail("--boundary-context-words must be at least 1.")

    ffmpeg = resolve_ffmpeg_executable()
    media_path = resolve_raw_media_path(args.video_name)
    transcript_path = resolve_transcript_path(args.transcript)

    duration = get_media_duration(ffmpeg, media_path)
    entries, words, transcript_payload = load_transcript(transcript_path, duration)
    transcript_metadata = {
        "version": transcript_payload.get("version"),
        "source": transcript_payload.get("source", {}),
    }
    heuristic_drop_windows, heuristic_review_windows = analyze_entries(
        entries,
        args.lookahead,
        args.max_repeat_gap,
        args.allow_heuristic_drops,
    )
    llm_drop_windows, llm_summary = load_llm_decisions(
        args.llm_decisions,
        duration,
        words,
        args.llm_min_confidence,
        args.ignore_llm_decisions,
    )
    repair_drop_windows, repair_summary = load_repair_plan(
        args.repair_plan,
        duration,
        words,
    )
    thought_blocks = build_thought_blocks(entries, words, llm_summary, args.thought_gap)
    keep_settings = KeepPlannerSettings(
        padding=args.padding,
        word_head_padding=args.word_head_padding,
        word_tail_padding=args.word_tail_padding,
        short_block_tail_padding=args.short_block_tail_padding,
        short_block_max_words=args.short_block_max_words,
        short_block_silence_trim=not args.disable_short_block_silence_trim,
        short_block_silence_min_duration=args.short_block_silence_min_duration,
        short_block_silence_window=args.short_block_silence_window,
        short_block_min_spoken_before_trim=args.short_block_min_spoken_before_trim,
        use_word_mask=bool(words) and not args.disable_word_mask and args.keep_source == "transcript",
    )
    partial_drop_windows = heuristic_drop_windows
    partial_drop_windows.extend(llm_drop_windows)
    partial_drop_windows.extend(repair_drop_windows)

    print(f"Analyzing silence in {media_path.name}...")
    silences = detect_silences(
        ffmpeg,
        media_path,
        args.silence_threshold,
        args.min_silence_duration,
    )
    short_block_silences = silences
    if (
        keep_settings.use_word_mask
        and keep_settings.short_block_silence_trim
        and args.short_block_silence_min_duration < args.min_silence_duration
    ):
        print("Analyzing short pauses for short-block tail trimming...")
        short_block_silences = detect_silences(
            ffmpeg,
            media_path,
            args.silence_threshold,
            args.short_block_silence_min_duration,
        )
    speech_intervals = speech_intervals_from_silences(silences, duration)
    short_block_trim_records: list[dict[str, object]] = []
    if args.keep_source == "audio":
        base_intervals = speech_intervals
    elif keep_settings.use_word_mask:
        base_intervals, short_block_trim_records = word_mask_intervals(
            words,
            thought_blocks,
            short_block_silences,
            keep_settings,
            duration,
        )
    else:
        base_intervals = transcript_intervals(entries, words, args.word_gap_merge)
    if repair_summary.forced_keep_intervals:
        base_intervals = merge_intervals(base_intervals + repair_summary.forced_keep_intervals)

    planner_result = plan_drop_windows(
        entries,
        partial_drop_windows,
        words,
        silences,
        thought_blocks,
        llm_summary,
        duration,
        args.cut_safety_margin,
        args.silence_snap_window,
        args.boundary_context_words,
        args.disable_boundary_validator,
    )

    keep_intervals, drop_windows = build_keep_intervals(
        base_intervals,
        planner_result.drop_windows,
        keep_settings,
        duration,
        args.min_clip_duration,
    )

    write_decisions(
        transcript_path,
        transcript_metadata,
        media_path,
        args.output,
        args.edit_decisions_output,
        duration,
        silences,
        args.keep_source,
        llm_summary,
        repair_summary,
        keep_intervals,
        drop_windows,
        partial_drop_windows,
        heuristic_review_windows,
        planner_result,
        thought_blocks,
        short_block_trim_records,
        entries,
        words,
        args.padding,
        keep_settings,
        args.word_gap_merge,
        args.cut_safety_margin,
        args.silence_snap_window,
        args.dry_run,
    )

    removed_count = len([entry for entry in entries if entry.drop])
    print(f"Marked {removed_count} transcript fragment(s) for removal.")
    print(f"Marked {len(heuristic_review_windows)} heuristic window(s) for review.")
    if planner_result.blocked_windows:
        print(f"Blocked {len(planner_result.blocked_windows)} unsafe cut(s).")
    if llm_summary.status == "loaded":
        print(f"Loaded {len(llm_summary.applied_drop_ranges)} LLM drop range(s).")
        print(f"Planner applied {len(planner_result.applied_windows)} cut window(s).")
    elif llm_summary.status == "missing":
        print(f"No LLM decisions found at: {args.llm_decisions}")
    if repair_summary.status == "loaded":
        print(f"Loaded {len(repair_summary.repairs)} repair operation(s).")
    if short_block_trim_records:
        print(f"Trimmed {len(short_block_trim_records)} short-block tail(s) to nearby silence.")
    print(f"Keeping {len(keep_intervals)} video interval(s).")
    print(f"Edit decisions saved to: {args.edit_decisions_output}")

    if args.dry_run:
        print("Dry run complete. Video was not rendered.")
        return

    print(f"Rendering {args.output}...")
    render_video(ffmpeg, media_path, keep_intervals, args.output)
    print(f"Edited video saved to: {args.output}")
    run_quality_check(args)
    print(f"Duration of editing video: {round(time.time()-start_time, 1)}s")


if __name__ == "__main__":
    main()
