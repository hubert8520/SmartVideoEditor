# SmartVideoEditor phase 3: edit runtime extraction

This phase starts reducing `scripts/edit_video.py` without touching edit planning or rendering logic.

## Scope

Phase 3 only centralizes:

- project paths,
- artifact path constants,
- raw media extensions,
- timestamp helpers,
- text normalization helpers.

It does not change:

- CLI options,
- LLM decision handling,
- repair plan handling,
- word-mask behavior,
- boundary validation,
- rendering behavior.

## Apply

If you are inside the unzipped `smart_video_editor_phase3` directory, run:

```bash
cp -R smart_video_editor tests PHASE3_README.md ..
cp scripts/apply_phase3_edit_runtime_refactor.py scripts/revert_phase3_edit_runtime_refactor.py ../scripts/
cd ..
./venv/bin/python scripts/apply_phase3_edit_runtime_refactor.py
```

Then test:

```bash
./venv/bin/python -m py_compile scripts/edit_video.py
./venv/bin/python -m py_compile smart_video_editor/editing/runtime.py
./venv/bin/python scripts/edit_video.py --padding 0.05 --dry-run
```

If you want the broader smoke check:

```bash
./venv/bin/python scripts/doctor.py
```

## Revert

```bash
./venv/bin/python scripts/revert_phase3_edit_runtime_refactor.py
```

## Before commit

Remove the backup and unpacked patch directory:

```bash
rm scripts/edit_video.py.phase3.bak
rm -rf smart_video_editor_phase3
find . -type d -name "__pycache__" -prune -exec rm -rf {} +
```

## Commit

```bash
git add scripts/edit_video.py \
  scripts/apply_phase3_edit_runtime_refactor.py \
  scripts/revert_phase3_edit_runtime_refactor.py \
  smart_video_editor/editing/runtime.py \
  tests/test_editing_runtime.py \
  PHASE3_README.md

git commit -m "Extract edit runtime helpers"
git push
```
