#!/usr/bin/env python3
"""Revert Phase 3 edit_video.py patch from backup."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EDIT_PATH = PROJECT_ROOT / "scripts" / "edit_video.py"
BACKUP_PATH = PROJECT_ROOT / "scripts" / "edit_video.py.phase3.bak"


def main() -> None:
    if not BACKUP_PATH.exists():
        raise SystemExit(f"Error: backup does not exist: {BACKUP_PATH}")
    EDIT_PATH.write_text(BACKUP_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Restored {EDIT_PATH} from {BACKUP_PATH}")


if __name__ == "__main__":
    main()
