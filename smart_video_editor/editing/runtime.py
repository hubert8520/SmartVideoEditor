"""Runtime helpers for scripts/edit_video.py.

Phase 3 keeps edit_video.py as the main entrypoint and does not change the edit
planning algorithm. This module centralizes stable paths and pure helpers so the
large script can be reduced safely in later phases.
"""

from __future__ import annotations

from smart_video_editor.paths import (
    ARTIFACTS_DIR,
    EDITED_DIR,
    PROJECT_ROOT,
    RAW_DIR,
    RAW_MEDIA_EXTENSIONS,
)
from smart_video_editor.text import normalize_text, strip_accents, tokenize
from smart_video_editor.timecode import seconds_to_timestamp, timestamp_to_seconds


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
