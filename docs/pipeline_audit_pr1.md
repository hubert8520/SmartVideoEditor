# Pipeline quality audit - PR 1

This audit treats the current implementation as a prototype, not as a
requirement. The existing pipeline has useful pieces, but edit quality is
limited by mixed responsibilities and weak contracts between detection,
planning, validation, QA, and repair.

## Current failure modes

### Word boundary safety

- `scripts/edit_video.py` plans cuts from both segment timestamps and word
  ranges. Word-based windows are clamped between neighboring words, but
  timestamp-only windows can still be used when `word_ids` are missing.
- `snap_drop_boundary_to_silence()` can move a boundary toward silence and is
  then clamped only for word-backed windows. Timestamp-only windows have no
  explicit mid-word guard.
- The renderer trusts `keep_intervals`. It does not know whether a keep edge
  is aligned to word boundaries.

### Repetition and partial repetitions

- `analyze_entries()` marks repeats on transcript entries, not on a durable
  candidate model. It can identify prefix fragments and similar entries, but
  the detector result is stored as `entry.drop` or `entry.review_reasons`.
- By default most heuristic repeats become review notes, which is conservative
  but leaves repeated takes in the video unless the LLM catches them.
- The current similarity logic is entry-based, so cases such as
  `jak skonfi... jak skonfigurowac` depend on how Deepgram segmented the text.

### Bad markers and failed takes

- Phrases such as `kurwa`, `jeszcze raz`, `od poczatku`, and `nie tak` are
  detected as entry review markers, but the code does not consistently expand
  them to the full failed take.
- Cutting only the marker word is usually the wrong edit. The planner needs a
  take-level candidate with context before and after the marker.

### Noise handling

- Noise without recognized words is mostly removed as a side effect of
  word-mask keep intervals, which is useful but not explicit.
- Coughs, throat clearing, chair movement, and setup noise that overlap speech
  have no first-class candidate type. They should become REVIEW unless a safe
  silence boundary exists.
- Short-block tail trimming exists, but it is buried inside the monolithic
  editor and is hard to test in isolation.

### Boundary validator

- `validate_boundaries()` catches some suspicious joins, especially joins after
  bridge words like `bo`, `ale`, `czyli`, `wiec`, but it validates only after
  candidate windows have already been converted into drop windows.
- The validator blocks applied windows by matching removed word ids. This can
  block a good cut if the issue is caused by a neighboring merged window.
- It does not explicitly validate every raw cut edge against every word unless
  the window came from known words.

### LLM prompt

- The edit-analysis prompt is directionally good but too broad. It asks for
  thought blocks, drops, reviews, and keep notes in one pass without a strict
  candidate taxonomy.
- It does not force a clear distinction between local facts, uncertain
  judgment, and automatic cut decisions.
- It does not require an explicit `why_not_review` / `safety_basis` field for
  automatic drops.

### QA and repair

- QA happens after render, so it detects many problems too late.
- `final_quality_report.json` does not currently require raw-time mapping or a
  machine-actionable repair intent such as `force_keep`, `force_drop`, or
  `manual_review`.
- `repair_from_quality_report.py` is conservative and renders from raw through
  `edit_video.py`, which is good. However, it often treats symptoms detected in
  the final video instead of feeding clearer constraints back into the planner.

## What is worth keeping

- Deepgram word-level transcription as the main transcript source.
- The artifact flow: `raw_transcription.json`, `llm_edit_decisions.json`,
  `edit_decisions.json`, `edited_transcription.json`, `final_quality_report.json`,
  and optional `repair_plan.json`.
- Rendering from raw media with keep intervals.
- Timeline mapping concept.
- Conservative defaults: uncertain edits should be REVIEW.
- The idea of post-render QA and bounded repair iterations.

## What should be replaced

- Entry mutation as the detector interface. Detectors should return candidates.
- A planner that mixes local heuristics, LLM ranges, repair overrides, boundary
  validation, and reporting in one function.
- Timestamp-only drop windows without explicit word-boundary checks.
- QA reports that do not include raw mapping and a concrete repair intent.

## PR 1 scope

This PR does not rewrite the editor. It adds regression tests and small
contract modules for the next PRs:

- local candidate detection,
- boundary validation,
- EDL/timeline JSON roundtrip,
- script entrypoint import safety.

The monolithic `scripts/edit_video.py` remains the production CLI for now.
The next recommended step is to move its pure planning logic behind these
contracts, then make the script a thin orchestrator.
