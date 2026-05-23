"""Project paths used by the local SmartVideoEditor pipeline."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "raw"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
EDITED_DIR = PROJECT_ROOT / "edited"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

RAW_TRANSCRIPTION_PATH = ARTIFACTS_DIR / "raw_transcription.json"
LEGACY_RAW_TRANSCRIPTION_PATH = ARTIFACTS_DIR / "raw_transcrpition.json"
LLM_EDIT_DECISIONS_PATH = ARTIFACTS_DIR / "llm_edit_decisions.json"
EDIT_DECISIONS_PATH = ARTIFACTS_DIR / "edit_decisions.json"
QUALITY_REPORT_PATH = ARTIFACTS_DIR / "final_quality_report.json"
QUALITY_TRANSCRIPT_PATH = ARTIFACTS_DIR / "edited_transcription.json"
REPAIR_PLAN_PATH = ARTIFACTS_DIR / "repair_plan.json"
DEFAULT_EDITED_VIDEO_PATH = EDITED_DIR / "edited_video.mp4"

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


def ensure_runtime_dirs() -> None:
    """Create local runtime directories if they do not exist."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    EDITED_DIR.mkdir(parents=True, exist_ok=True)


def supported_raw_media_files(raw_dir: Path = RAW_DIR) -> list[Path]:
    """Return supported media files from raw_dir."""
    if not raw_dir.exists():
        return []
    return sorted(
        path
        for path in raw_dir.iterdir()
        if path.is_file() and path.suffix.lower() in RAW_MEDIA_EXTENSIONS
    )


def resolve_single_raw_media(video_name: str | None, raw_dir: Path = RAW_DIR) -> Path:
    """Resolve a media file from raw_dir, preserving the current CLI semantics."""
    if video_name:
        requested = Path(video_name)
        if requested.is_absolute() or requested.parent != Path("."):
            raise ValueError("--video-name must be a file name from raw/, not a path.")

        media_path = raw_dir / requested.name
        if not media_path.exists():
            raise FileNotFoundError(f"Input file does not exist in raw/: {requested.name}")
        if media_path.suffix.lower() not in RAW_MEDIA_EXTENSIONS:
            raise ValueError(f"Unsupported media file extension: {media_path.suffix}")
        return media_path

    media_files = supported_raw_media_files(raw_dir)
    if not media_files:
        raise FileNotFoundError(f"No supported media files found in {raw_dir}")
    if len(media_files) > 1:
        names = ", ".join(path.name for path in media_files)
        raise ValueError(f"More than one media file found in raw/. Choose one with --video-name. Found: {names}")
    return media_files[0]
