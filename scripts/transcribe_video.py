#!/usr/bin/env python3
"""Transcribe media with word-level timestamps for the editing pipeline."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


DEFAULT_PROVIDER = "deepgram"
DEFAULT_OPENAI_MODEL = "whisper-1"
DEFAULT_DEEPGRAM_MODEL = "nova-3"
DEEPGRAM_LISTEN_URL = "https://api.deepgram.com/v1/listen"
OPENAI_COARSE_TRANSCRIPTION_MODELS = {
    "gpt-4o-transcribe",
    "gpt-4o-mini-transcribe",
}
MAX_UPLOAD_BYTES = 24 * 1024 * 1024
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "raw"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RAW_TRANSCRIPTION_PATH = ARTIFACTS_DIR / "raw_transcription.json"
PLACEHOLDER_KEY_MARKERS = ("your-", "your_", "wklej", "tutaj", "...")
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract audio from raw/ media and create artifacts/raw_transcription.json."
    )
    parser.add_argument(
        "--video-name",
        help=(
            "Media file name inside raw/. If omitted, the only supported media file "
            "from raw/ is selected automatically."
        ),
    )
    parser.add_argument(
        "--provider",
        choices=("openai", "deepgram"),
        default=DEFAULT_PROVIDER,
        help=f"Transcription provider to use. Default: {DEFAULT_PROVIDER}.",
    )
    parser.add_argument(
        "--model",
        help=(
            "Provider model. Defaults to "
            f"{DEFAULT_OPENAI_MODEL} for OpenAI and {DEFAULT_DEEPGRAM_MODEL} for Deepgram."
        ),
    )
    parser.add_argument(
        "--language",
        help="Optional ISO language code, e.g. pl, en, de. Leave empty for auto-detect.",
    )
    parser.add_argument(
        "--prompt",
        help="Optional OpenAI context prompt with names, terminology, or expected vocabulary.",
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
        "--diarize",
        action="store_true",
        help="Enable speaker-aware transcription when supported by the provider.",
    )
    parser.add_argument(
        "--allow-coarse-openai",
        action="store_true",
        help=(
            "Allow OpenAI models that return only chunk-level text, not segment or word "
            "timestamps. Not recommended for editing."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=RAW_TRANSCRIPTION_PATH,
        help="Output path. Default: artifacts/raw_transcription.json.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        help="Path to a .env file. Defaults to .env in the current directory or project root.",
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


def get_media_duration(ffmpeg_executable: str, media_path: Path) -> float:
    result = subprocess.run(
        [ffmpeg_executable, "-hide_banner", "-i", str(media_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", result.stderr)
    if not match:
        fail(f"Could not read media duration from ffmpeg output for: {media_path}")

    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def extract_audio_chunks(
    media_path: Path,
    output_dir: Path,
    chunk_seconds: int,
    bitrate: str,
    ffmpeg_executable: str,
) -> list[tuple[Path, float, float]]:
    output_pattern = output_dir / "chunk_%04d.mp3"
    command = [
        ffmpeg_executable,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(media_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-b:a",
        bitrate,
        "-f",
        "segment",
        "-segment_time",
        str(chunk_seconds),
        "-reset_timestamps",
        "1",
        str(output_pattern),
    ]
    subprocess.run(command, check=True)

    chunks: list[tuple[Path, float, float]] = []
    for index, chunk_path in enumerate(sorted(output_dir.glob("chunk_*.mp3"))):
        size = chunk_path.stat().st_size
        if size > MAX_UPLOAD_BYTES:
            fail(
                f"{chunk_path.name} is {size / (1024 * 1024):.1f} MB after compression. "
                "Use a smaller --chunk-minutes value or lower --bitrate."
            )

        start = index * chunk_seconds
        duration = get_media_duration(ffmpeg_executable, chunk_path)
        chunks.append((chunk_path, start, start + duration))

    if not chunks:
        fail("No audio chunks were created. Check that the input file has an audio track.")

    return chunks


def get_value(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def seconds_to_timestamp(seconds: float) -> str:
    total_milliseconds = max(0, int(round(seconds * 1000)))
    total_seconds, milliseconds = divmod(total_milliseconds, 1000)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}:{milliseconds:03d}"


def timestamp_to_seconds(timestamp: str) -> float:
    hours, minutes, seconds, milliseconds = timestamp.strip().split(":")
    return (
        int(hours) * 3600
        + int(minutes) * 60
        + int(seconds)
        + int(milliseconds.ljust(3, "0")[:3]) / 1000
    )


def response_text(response: Any) -> str:
    if isinstance(response, str):
        return response.strip()

    text = get_value(response, "text")
    if isinstance(text, str):
        return text.strip()

    return str(response).strip()


def raw_segment(
    start_seconds: float,
    end_seconds: float,
    text: str,
    speaker: str | None = None,
) -> dict[str, Any] | None:
    clean_text = text.strip()
    if not clean_text:
        return None
    return {
        "start_seconds": max(0.0, start_seconds),
        "end_seconds": max(start_seconds, end_seconds),
        "text": clean_text,
        "speaker": speaker,
    }


def raw_word(
    start_seconds: float,
    end_seconds: float,
    word: str,
    speaker: str | None = None,
    confidence: float | None = None,
) -> dict[str, Any] | None:
    clean_word = word.strip()
    if not clean_word:
        return None
    payload: dict[str, Any] = {
        "start_seconds": max(0.0, start_seconds),
        "end_seconds": max(start_seconds, end_seconds),
        "word": clean_word,
    }
    if speaker is not None:
        payload["speaker"] = str(speaker)
    if confidence is not None:
        payload["confidence"] = confidence
    return payload


def deepgram_plain_text(response: dict[str, Any]) -> str:
    channels = response.get("results", {}).get("channels", [])
    if not channels:
        return ""

    alternatives = channels[0].get("alternatives", [])
    if not alternatives:
        return ""

    return str(alternatives[0].get("transcript", "")).strip()


def deepgram_word_text(word: dict[str, Any]) -> str:
    return str(word.get("punctuated_word") or word.get("word") or "").strip()


def deepgram_words(response: dict[str, Any], offset_seconds: float) -> list[dict[str, Any]]:
    channels = response.get("results", {}).get("channels", [])
    alternatives = channels[0].get("alternatives", []) if channels else []
    words = alternatives[0].get("words", []) if alternatives else []

    raw_words: list[dict[str, Any]] = []
    for word in words:
        start = float(word.get("start", 0.0) or 0.0) + offset_seconds
        end = float(word.get("end", start) or start) + offset_seconds
        text = deepgram_word_text(word)
        speaker_value = word.get("speaker")
        speaker = f"Speaker {speaker_value}" if speaker_value is not None else None
        confidence = word.get("confidence")
        item = raw_word(
            start,
            end,
            text,
            speaker=speaker,
            confidence=float(confidence) if confidence is not None else None,
        )
        if item:
            raw_words.append(item)

    return raw_words


def segments_from_words(
    words: list[dict[str, Any]],
    max_gap: float = 0.75,
    max_duration: float = 7.0,
) -> list[dict[str, Any]]:
    if not words:
        return []

    segments: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = [words[0]]

    for word in words[1:]:
        previous = current[-1]
        gap = word["start_seconds"] - previous["end_seconds"]
        duration = word["end_seconds"] - current[0]["start_seconds"]
        speaker_changed = word.get("speaker") != previous.get("speaker")
        previous_text = str(previous.get("word", ""))
        sentence_boundary = previous_text.endswith((".", "!", "?")) and len(current) >= 4

        if speaker_changed or gap > max_gap or duration > max_duration or sentence_boundary:
            text = " ".join(str(item["word"]) for item in current)
            segment = raw_segment(
                current[0]["start_seconds"],
                current[-1]["end_seconds"],
                text,
                speaker=current[0].get("speaker"),
            )
            if segment:
                segments.append(segment)
            current = []

        current.append(word)

    if current:
        text = " ".join(str(item["word"]) for item in current)
        segment = raw_segment(
            current[0]["start_seconds"],
            current[-1]["end_seconds"],
            text,
            speaker=current[0].get("speaker"),
        )
        if segment:
            segments.append(segment)

    return segments


def deepgram_segments(
    response: dict[str, Any],
    offset_seconds: float,
    chunk_end_seconds: float,
    words: list[dict[str, Any]],
    include_speaker: bool,
) -> list[dict[str, Any]]:
    utterances = response.get("results", {}).get("utterances") or []
    segments: list[dict[str, Any]] = []
    for utterance in utterances:
        text = str(utterance.get("transcript", "")).strip()
        if not text:
            continue
        speaker_value = utterance.get("speaker")
        speaker = f"Speaker {speaker_value}" if speaker_value is not None else None
        prefix = f"{speaker}: " if include_speaker and speaker else ""
        start = float(utterance.get("start", 0.0) or 0.0) + offset_seconds
        end = float(utterance.get("end", start) or start) + offset_seconds
        segment = raw_segment(start, end, f"{prefix}{text}", speaker=speaker)
        if segment:
            segments.append(segment)

    if segments:
        return segments
    if words:
        return segments_from_words(words)

    fallback = raw_segment(offset_seconds, chunk_end_seconds, deepgram_plain_text(response))
    return [fallback] if fallback else []


def openai_words(response: Any, offset_seconds: float) -> list[dict[str, Any]]:
    words = get_value(response, "words", []) or []
    raw_words: list[dict[str, Any]] = []
    for word in words:
        start = float(get_value(word, "start", 0.0) or 0.0) + offset_seconds
        end = float(get_value(word, "end", start) or start) + offset_seconds
        text = str(get_value(word, "word", "")).strip()
        item = raw_word(start, end, text)
        if item:
            raw_words.append(item)
    return raw_words


def openai_segments(
    response: Any,
    offset_seconds: float,
    chunk_end_seconds: float,
    words: list[dict[str, Any]],
    include_speaker: bool,
) -> list[dict[str, Any]]:
    segments = get_value(response, "segments", []) or []
    raw_segments: list[dict[str, Any]] = []
    for segment in segments:
        speaker = get_value(segment, "speaker", None)
        text = str(get_value(segment, "text", "")).strip()
        start = float(get_value(segment, "start", 0.0) or 0.0) + offset_seconds
        end = float(get_value(segment, "end", start) or start) + offset_seconds
        prefix = f"{speaker}: " if include_speaker and speaker else ""
        item = raw_segment(start, end, f"{prefix}{text}", speaker=str(speaker) if speaker else None)
        if item:
            raw_segments.append(item)

    if raw_segments:
        return raw_segments
    if words:
        return segments_from_words(words)

    fallback = raw_segment(offset_seconds, chunk_end_seconds, response_text(response))
    return [fallback] if fallback else []


def resolve_model(args: argparse.Namespace) -> str:
    if args.model:
        return args.model
    if args.provider == "deepgram":
        return DEFAULT_DEEPGRAM_MODEL
    return DEFAULT_OPENAI_MODEL


def display_model(provider: str, model: str, diarize: bool) -> str:
    if provider == "openai" and diarize:
        return "gpt-4o-transcribe-diarize"
    return model


def validate_provider_options(args: argparse.Namespace, model: str) -> None:
    if (
        args.provider == "openai"
        and not args.diarize
        and model in OPENAI_COARSE_TRANSCRIPTION_MODELS
        and not args.allow_coarse_openai
    ):
        fail(
            f"{model} does not return segment/word timestamps, so it is not usable for "
            "precise video editing. Use Deepgram, the OpenAI model whisper-1, or pass "
            "--diarize for gpt-4o-transcribe-diarize. To force chunk-level text anyway, "
            "add --allow-coarse-openai."
        )


def required_api_key_name(provider: str) -> str:
    return "DEEPGRAM_API_KEY" if provider == "deepgram" else "OPENAI_API_KEY"


def get_required_api_key(provider: str, loaded_env: Path | None) -> str:
    key_name = required_api_key_name(provider)
    api_key = os.getenv(key_name)
    if api_key and not any(marker in api_key.lower() for marker in PLACEHOLDER_KEY_MARKERS):
        return api_key

    location_hint = loaded_env or (PROJECT_ROOT / ".env")
    example = "dg_..." if provider == "deepgram" else "sk-..."
    fail(
        f"{key_name} is missing or still looks like a placeholder. "
        f"Add it to {location_hint} as: {key_name}={example}"
    )


def transcribe_openai_chunk(
    client: Any,
    chunk_path: Path,
    args: argparse.Namespace,
    offset_seconds: float,
    chunk_end_seconds: float,
    model: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected_model = "gpt-4o-transcribe-diarize" if args.diarize else model
    response_format = "diarized_json" if args.diarize else "json"
    if selected_model == "whisper-1" and not args.diarize:
        response_format = "verbose_json"

    params: dict[str, Any] = {
        "model": selected_model,
        "file": None,
        "response_format": response_format,
    }
    if args.language:
        params["language"] = args.language
    if args.prompt and not args.diarize:
        params["prompt"] = args.prompt
    if args.diarize:
        params["chunking_strategy"] = "auto"
        if args.prompt:
            print("Ignoring --prompt because diarization does not support prompts.")
    elif response_format == "verbose_json":
        params["timestamp_granularities"] = ["segment", "word"]

    with chunk_path.open("rb") as audio_file:
        params["file"] = audio_file
        response = client.audio.transcriptions.create(**params)

    words = openai_words(response, offset_seconds)
    segments = openai_segments(
        response,
        offset_seconds,
        chunk_end_seconds,
        words,
        include_speaker=args.diarize,
    )
    return segments, words


def transcribe_deepgram_chunk(
    api_key: str,
    chunk_path: Path,
    args: argparse.Namespace,
    offset_seconds: float,
    chunk_end_seconds: float,
    model: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        import httpx
    except ImportError:
        fail("Missing Python package 'httpx'. Install dependencies with: pip install -r requirements.txt")

    if args.prompt:
        print("Ignoring --prompt because Deepgram does not support OpenAI-style prompts.")

    params: dict[str, str] = {
        "model": model,
        "smart_format": "true",
        "punctuate": "true",
        "utterances": "true",
    }
    if args.language:
        params["language"] = args.language
    else:
        params["detect_language"] = "true"
    if args.diarize:
        params["diarize"] = "true"

    with chunk_path.open("rb") as audio_file:
        response = httpx.post(
            DEEPGRAM_LISTEN_URL,
            params=params,
            headers={
                "Authorization": f"Token {api_key}",
                "Content-Type": "audio/mpeg",
            },
            content=audio_file.read(),
            timeout=600.0,
        )

    if response.status_code >= 400:
        fail(f"Deepgram API error {response.status_code}: {response.text[:500]}")

    payload = response.json()
    words = deepgram_words(payload, offset_seconds)
    segments = deepgram_segments(
        payload,
        offset_seconds,
        chunk_end_seconds,
        words,
        include_speaker=args.diarize,
    )
    return segments, words


def finalize_transcript(
    media_path: Path,
    provider: str,
    model: str,
    active_model: str,
    language: str | None,
    diarize: bool,
    raw_segments: list[dict[str, Any]],
    raw_words: list[dict[str, Any]],
) -> dict[str, Any]:
    sorted_words = sorted(raw_words, key=lambda item: (item["start_seconds"], item["end_seconds"]))
    words: list[dict[str, Any]] = []
    for word_id, item in enumerate(sorted_words):
        word: dict[str, Any] = {
            "id": word_id,
            "timestamp": seconds_to_timestamp(float(item["start_seconds"])),
            "end": seconds_to_timestamp(float(item["end_seconds"])),
            "word": str(item["word"]),
        }
        if item.get("speaker"):
            word["speaker"] = str(item["speaker"])
        if item.get("confidence") is not None:
            word["confidence"] = float(item["confidence"])
        words.append(word)

    word_seconds = [
        (
            int(word["id"]),
            timestamp_to_seconds(str(word["timestamp"])),
            timestamp_to_seconds(str(word["end"])),
        )
        for word in words
    ]

    sorted_segments = sorted(
        raw_segments,
        key=lambda item: (item["start_seconds"], item["end_seconds"]),
    )
    segments: list[dict[str, Any]] = []
    for segment_id, item in enumerate(sorted_segments):
        start = float(item["start_seconds"])
        end = max(start, float(item["end_seconds"]))
        word_ids = [
            word_id
            for word_id, word_start, word_end in word_seconds
            if word_end > start - 0.05 and word_start < end + 0.05
        ]
        segment: dict[str, Any] = {
            "id": segment_id,
            "timestamp": seconds_to_timestamp(start),
            "end": seconds_to_timestamp(end),
            "transcription": str(item["text"]).strip(),
            "word_ids": word_ids,
        }
        if item.get("speaker"):
            segment["speaker"] = str(item["speaker"])
        segments.append(segment)

    return {
        "version": "2.0",
        "source": {
            "media_path": str(media_path),
            "provider": provider,
            "model": active_model,
            "base_model": model,
            "language": language or "auto",
            "diarize": diarize,
            "word_level": bool(words),
        },
        "segments": segments,
        "words": words,
    }


def create_client(provider: str, api_key: str) -> Any:
    if provider != "openai":
        return None

    try:
        from openai import OpenAI
    except ImportError:
        fail("Missing Python package 'openai'. Install it with: pip install -r requirements.txt")
    return OpenAI(api_key=api_key)


def transcribe_media(
    media_path: Path,
    args: argparse.Namespace,
    api_key: str,
    model: str,
    active_model: str,
    client: Any,
    ffmpeg_executable: str,
) -> dict[str, Any]:
    if args.chunk_minutes <= 0:
        fail("--chunk-minutes must be greater than 0.")

    chunk_seconds = max(1, int(args.chunk_minutes * 60))
    raw_segments: list[dict[str, Any]] = []
    raw_words: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="video_transcribe_") as temp_name:
        temp_dir = Path(temp_name)
        print(f"Preparing audio chunks from {media_path}...")
        chunks = extract_audio_chunks(
            media_path,
            temp_dir,
            chunk_seconds,
            args.bitrate,
            ffmpeg_executable,
        )

        total = len(chunks)
        for index, (chunk_path, offset_seconds, chunk_end_seconds) in enumerate(chunks, start=1):
            print(
                f"Transcribing chunk {index}/{total} "
                f"from {seconds_to_timestamp(offset_seconds)} "
                f"with {args.provider}/{active_model}..."
            )
            if args.provider == "deepgram":
                part_segments, part_words = transcribe_deepgram_chunk(
                    api_key,
                    chunk_path,
                    args,
                    offset_seconds,
                    chunk_end_seconds,
                    model,
                )
            else:
                part_segments, part_words = transcribe_openai_chunk(
                    client,
                    chunk_path,
                    args,
                    offset_seconds,
                    chunk_end_seconds,
                    model,
                )
            raw_segments.extend(part_segments)
            raw_words.extend(part_words)

    return finalize_transcript(
        media_path=media_path,
        provider=args.provider,
        model=model,
        active_model=active_model,
        language=args.language,
        diarize=args.diarize,
        raw_segments=raw_segments,
        raw_words=raw_words,
    )


def main() -> None:
    args = parse_args()
    loaded_env = load_env_file(args.env_file)
    media_path = resolve_raw_media_path(args.video_name)
    model = resolve_model(args)
    validate_provider_options(args, model)
    active_model = display_model(args.provider, model, args.diarize)
    api_key = get_required_api_key(args.provider, loaded_env)
    client = create_client(args.provider, api_key)
    ffmpeg_executable = resolve_ffmpeg_executable()

    transcript = transcribe_media(
        media_path,
        args,
        api_key,
        model,
        active_model,
        client,
        ffmpeg_executable,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(transcript, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Transcript saved to: {args.output}")
    if transcript["source"]["provider"] != "deepgram":
        print("Tip: Deepgram is the default provider because it gives word-level timings for editing.")


if __name__ == "__main__":
    main()
