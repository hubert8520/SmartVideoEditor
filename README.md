# Smart Video Editor

[![Tests](https://github.com/hubert8520/SmartVideoEditor/actions/workflows/tests.yml/badge.svg)](https://github.com/hubert8520/SmartVideoEditor/actions/workflows/tests.yml)

An AI-assisted command-line pipeline that turns raw talking-head footage into a tighter, reviewable edit.

Smart Video Editor combines word-level transcription, local candidate detection, semantic edit judgment, deterministic cut planning, FFmpeg rendering, and post-render quality assurance. It is designed for recordings where a speaker repeats takes, abandons sentences, uses filler words, or leaves setup noise and long pauses.

The current detectors are optimized for Polish spoken content. Transcription supports configurable language codes, while the documentation, CLI, model instructions, and generated review reports are in English.

## Why this project exists

Editing an educational or marketing monologue is repetitive but risky. Removing silence is easy; removing the wrong half-sentence is also easy. This project separates detection, semantic judgment, deterministic planning, rendering, and QA so every proposed cut can be traced to transcript word IDs and source-video timestamps.

The pipeline aims to be:

- precise: edits are planned around word-level timestamps;
- explainable: candidates carry evidence and every applied decision has a reason;
- conservative: uncertain changes become review items instead of silent cuts;
- recoverable: repair iterations always render from the original recording;
- auditable: artifacts preserve decisions and final-to-source timeline mappings.

## Pipeline

```text
raw media
    |
    v
word-level transcription
    |
    v
local candidate detection + semantic LLM judgment
    |
    v
deterministic planner + boundary validation
    |
    v
FFmpeg render
    |
    v
post-render transcription + quality assurance
    |
    +---- pass ---------> final video
    |
    +---- review/fail --> bounded repair loop or editor brief
```

## Engineering highlights

- Deepgram and OpenAI transcription providers with word-level normalization.
- First-class candidates for repeated attempts, failed-take markers, and isolated noise.
- Attempt grouping and completeness evidence for partial and repeated takes.
- Candidate-aware LLM decisions anchored to stable `word_id` ranges.
- Protected thought blocks and boundary validation around planned joins.
- Deterministic edit decision lists and timeline maps back to the source recording.
- Actionable QA reports with `force_keep`, `force_drop`, `manual_review`, and `no_auto_repair` intents.
- Conservative repair planning with bounded automatic refinement.
- Human-readable Markdown and CSV review briefs with optional comparison clips.
- Modular packages for detection, planning, editing, reporting, transcription, and media rendering.

## Requirements

- Python 3.11 or newer
- An OpenAI API key for semantic analysis and final quality assurance
- A Deepgram API key for the default word-level transcription workflow

FFmpeg is provided through `imageio-ffmpeg` when a system installation is unavailable.

## Installation

Create a virtual environment and install the project:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -e .
```

For development and tests, install the optional development dependencies:

```bash
python -m pip install -e ".[dev]"
```

Create the local environment file from the template and add your keys:

```bash
cp .env.example .env
```

```dotenv
OPENAI_API_KEY=sk-...
DEEPGRAM_API_KEY=...
```

The `.env` file, source media, generated artifacts, and rendered videos are excluded from Git.

## Quick start

Initialize the runtime directories and check the environment without calling any API:

```bash
smart-video-editor init
smart-video-editor doctor
```

Place one supported audio or video file in `raw/`, then run the complete pipeline:

```bash
smart-video-editor run --language pl
```

If `raw/` contains multiple media files, select one explicitly:

```bash
smart-video-editor run --video-name recording.mp4 --language pl
```

Run automatic QA repairs after the first render:

```bash
smart-video-editor run --video-name recording.mp4 --language pl --auto-refine
```

The main outputs are:

```text
artifacts/raw_transcription.json
artifacts/llm_edit_decisions.json
artifacts/edit_decisions.json
artifacts/edited_transcription.json
artifacts/final_quality_report.json
edited/edited_video.mp4
```

## CLI commands

| Command | Purpose |
| --- | --- |
| `init` | Create `raw/`, `artifacts/`, and `edited/`. |
| `doctor` | Check `.env`, FFmpeg, raw media, and API-key presence without API calls. |
| `run` | Execute transcription, analysis, rendering, QA, and optional refinement. |
| `transcribe` | Create a word-level transcript from source media. |
| `analyze` | Detect local candidates and generate semantic edit decisions. |
| `edit` | Plan cuts, render the video, and optionally run QA. |
| `quality` | Run post-render transcription and quality assurance. |
| `repair` | Build a conservative repair plan from a QA report. |
| `refine` | Run bounded repair, render, and QA iterations. |
| `review` | Generate Markdown and CSV instructions for manual review. |

Stage-specific arguments are forwarded to the underlying implementation:

```bash
smart-video-editor transcribe --help
smart-video-editor edit --help
smart-video-editor review --help
```

The original `python scripts/<stage>.py` entry points remain supported.

## Workspaces and resuming

Run against a different workspace without changing directories:

```bash
smart-video-editor --workspace /path/to/project run --language pl
```

Resume after a completed transcription:

```bash
smart-video-editor run --from-stage analyze --language pl
```

Resume directly from existing transcript and LLM decision artifacts:

```bash
smart-video-editor run --from-stage edit --language pl
```

Generate edit decisions without rendering the video:

```bash
smart-video-editor run --language pl --plan-only
```

`--plan-only` still runs API-backed transcription and semantic analysis when starting from those stages.

## Individual stages

### Transcription

Deepgram `nova-3` is the default because precise word timestamps are essential for editing:

```bash
smart-video-editor transcribe --provider deepgram --language pl
```

OpenAI transcription and speaker diarization remain available:

```bash
smart-video-editor transcribe --provider openai --language pl
smart-video-editor transcribe --language pl --diarize
```

### Semantic analysis

Validate the transcript and estimate request size without calling the API:

```bash
smart-video-editor analyze --dry-run
```

Generate candidate-aware edit decisions:

```bash
smart-video-editor analyze
```

The result contains protected `thought_blocks`, safe `drop_ranges`, uncertain `review_ranges`, `keep_notes`, local candidate evidence, and explicit safety explanations.

### Planning and rendering

Inspect the planned edit without rendering:

```bash
smart-video-editor edit --padding 0.05 --dry-run
```

Render the edit and run post-render QA:

```bash
smart-video-editor edit --padding 0.05 --quality-language pl
```

Useful safety and tuning options:

```bash
# Ignore semantic decisions and inspect local candidates only.
smart-video-editor edit --ignore-llm-decisions --dry-run

# Allow local heuristics to create automatic cuts.
smart-video-editor edit --allow-heuristic-drops

# Raise the minimum confidence for automatic LLM cuts.
smart-video-editor edit --llm-min-confidence 0.85

# Disable automatic post-render quality assurance.
smart-video-editor edit --skip-quality-check
```

### Repair and review

Run bounded repair iterations from the original source media:

```bash
smart-video-editor refine --quality-language pl
```

Generate a review brief when an issue still needs a human decision:

```bash
smart-video-editor review --make-clips
```

The brief includes final and source time ranges, QA intent, repair status, confidence, evidence, source-word context, and optional comparison clips. See [Editor Review Reports](docs/editor_review_reports.md) for the report contract.

## Architecture

```text
smart_video_editor/
  cli/             unified CLI and edit orchestration
  detection/       local candidate detection
  domain/          shared candidate and decision models
  editing/         runtime paths, decision I/O, intervals, and QA orchestration
  llm/             semantic-analysis and QA prompts
  media/           FFmpeg rendering helpers
  planning/        decision planner, boundaries, and edit decision lists
  reporting/       timeline mapping, actionable QA, and editor briefs
  segmentation/    word, take, and repeated-attempt grouping
  transcription/   normalized transcription runtime
```

The scripts in `scripts/` are thin or backward-compatible entry points around these packages.

## Artifact and safety model

The pipeline never treats a rendered output as the next source. Each repair iteration uses the original recording and a structured repair plan. Every `edit_decisions*.json` contains a `timeline_map` that maps final intervals back to raw-media intervals.

Smart Video Editor deliberately favors review over aggressive automation:

- local detectors produce evidence-backed candidates rather than unconditional cuts;
- low-confidence or ambiguous LLM ranges are not applied automatically;
- thought boundaries and source-word boundaries can block unsafe joins;
- automatic repair requires actionable QA intent and source mapping;
- refinement stops after a small, configured number of iterations;
- applied, reviewed, and blocked decisions are preserved for inspection.

Generated working data is stored in:

```text
raw/        local source media
artifacts/  transcripts, candidates, decisions, QA reports, and review briefs
edited/     rendered videos
```

## Tests

Run the complete test suite:

```bash
python -m pytest -q
```

The suite covers local detection, attempt grouping, planner safety, edit decision I/O, timeline mapping, actionable QA and repair contracts, editor reports, transcription helpers, CLI delegation, and script entry points.

## Current limitations

- Detection heuristics are currently tuned for Polish monologues.
- API-backed stages require network access and may incur provider costs.
- The tool focuses on content cuts, not captions, color grading, visual effects, or multi-camera editing.
- Human review is recommended before publishing the final render.
