#!/usr/bin/env bash
set -euo pipefail

# Run from repository root:
#   bash scripts/apply_phase1_refactor.sh
#
# This script intentionally does not overwrite your existing processing scripts.

if [ ! -d ".git" ]; then
  echo "Error: run this script from the SmartVideoEditor repository root." >&2
  exit 1
fi

if [ -f "READ.me" ] && [ ! -f "README.md" ]; then
  git mv READ.me README.md
  echo "Renamed READ.me to README.md"
fi

mkdir -p raw artifacts edited

echo "Phase 1 scaffold is applied. Next:"
echo "  ./venv/bin/python scripts/doctor.py"
echo "  git status"
