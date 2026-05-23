# SmartVideoEditor phase 2: transcribe runtime extraction

This phase is intentionally conservative. It starts extracting shared runtime helpers from `scripts/transcribe_video.py` without changing the CLI, providers, or output format.

## Apply

If you are inside the unzipped `smart_video_editor_phase2` directory, run:

```bash
cp -R smart_video_editor tests PHASE2_README.md ..
cp scripts/apply_phase2_transcribe_refactor.py scripts/revert_phase2_transcribe_refactor.py ../scripts/
cd ..
./venv/bin/python scripts/apply_phase2_transcribe_refactor.py
```

Then test:

```bash
./venv/bin/python -m py_compile scripts/transcribe_video.py
./venv/bin/python -m py_compile smart_video_editor/transcription/runtime.py
./venv/bin/python scripts/doctor.py
```

`transcribe_video.py` currently calls external APIs when run normally, so do not run full transcription unless you intend to call Deepgram/OpenAI.

## Revert

```bash
./venv/bin/python scripts/revert_phase2_transcribe_refactor.py
```

## Commit

```bash
git add scripts/transcribe_video.py scripts/apply_phase2_transcribe_refactor.py scripts/revert_phase2_transcribe_refactor.py smart_video_editor/transcription tests/test_transcription_runtime.py PHASE2_README.md
git commit -m "Extract transcribe runtime helpers"
```
