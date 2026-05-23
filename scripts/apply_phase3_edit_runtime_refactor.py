#!/usr/bin/env python3
'''Apply Phase 3 runtime refactor to scripts/edit_video.py.

This patch is intentionally narrow:
- no edit planning logic changes;
- no rendering logic changes;
- no CLI changes;
- creates a backup before editing;
- rebinds stable path/timecode/text helper names to shared package functions.
'''

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EDIT_PATH = PROJECT_ROOT / "scripts" / "edit_video.py"
BACKUP_PATH = PROJECT_ROOT / "scripts" / "edit_video.py.phase3.bak"

PATCH_MARKER = "# Phase 3 shared edit runtime helpers"
PATCH_BLOCK = '''


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
'''


def fail(message: str) -> None:
    raise SystemExit(f"Error: {message}")


def main() -> None:
    if not EDIT_PATH.exists():
        fail(f"Missing file: {EDIT_PATH}")

    source = EDIT_PATH.read_text(encoding="utf-8")

    if PATCH_MARKER in source:
        print("Phase 3 patch already present in scripts/edit_video.py")
        return

    insertion_point = source.find("\ndef main() -> None:")
    if insertion_point == -1:
        fail("Could not find 'def main() -> None:' in scripts/edit_video.py")

    if not BACKUP_PATH.exists():
        BACKUP_PATH.write_text(source, encoding="utf-8")
        print(f"Backup written: {BACKUP_PATH}")
    else:
        print(f"Backup already exists: {BACKUP_PATH}")

    patched = source[:insertion_point] + PATCH_BLOCK + source[insertion_point:]
    EDIT_PATH.write_text(patched, encoding="utf-8")
    print("Patched scripts/edit_video.py")

    subprocess.run(
        [sys.executable, "-m", "py_compile", str(EDIT_PATH)],
        cwd=PROJECT_ROOT,
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "py_compile", str(PROJECT_ROOT / "smart_video_editor" / "editing" / "runtime.py")],
        cwd=PROJECT_ROOT,
        check=True,
    )
    print("Python compile check passed.")


if __name__ == "__main__":
    main()
