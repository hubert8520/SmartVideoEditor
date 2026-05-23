#!/usr/bin/env python3
"""Revert Phase 2 transcribe_video.py patch from backup."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRANSCRIBE_PATH = PROJECT_ROOT / "scripts" / "transcribe_video.py"
BACKUP_PATH = PROJECT_ROOT / "scripts" / "transcribe_video.py.phase2.bak"


def main() -> None:
    if not BACKUP_PATH.exists():
        raise SystemExit(f"Error: backup does not exist: {BACKUP_PATH}")
    TRANSCRIBE_PATH.write_text(BACKUP_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Restored {TRANSCRIBE_PATH} from {BACKUP_PATH}")


if __name__ == "__main__":
    main()
