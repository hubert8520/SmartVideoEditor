# SmartVideoEditor phase 1 refactor

This patch is intentionally conservative. It adds a reusable Python package and setup helpers without changing the current behavior of the existing scripts.

## Apply

From repository root:

```bash
unzip smart_video_editor_phase1.zip
cp -R smart_video_editor scripts tests .env.example .
bash scripts/apply_phase1_refactor.sh
```

Then:

```bash
./venv/bin/python scripts/doctor.py
git status
pytest tests
```

If `pytest` is not installed, either install it in your venv or skip tests for now.

## What this phase does

- Adds `smart_video_editor/` package.
- Adds shared path, timecode, env, ffmpeg, JSON and interval utilities.
- Adds domain models for future refactors.
- Adds `.env.example`.
- Adds `scripts/doctor.py`.
- Adds basic tests.
- Provides a safe rename path from `READ.me` to `README.md`.

## What this phase does not do

- It does not rewrite `edit_video.py`.
- It does not change rendering behavior.
- It does not alter Deepgram/OpenAI calls.
- It does not change generated artifacts.

This makes it a safe first commit before deeper refactors.
