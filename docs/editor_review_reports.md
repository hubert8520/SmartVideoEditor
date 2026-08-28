# Editor Review Reports

The editor brief is the final layer of the pipeline when automatic repair should stop making assumptions. It presents the same issue in final-video and source-media time, together with the QA rationale and repair status.

## Inputs

`smart-video-editor review` reads:

- `artifacts/final_quality_report*.json`;
- `artifacts/edit_decisions*.json`;
- `edited/edited_video*.mp4`;
- the source video referenced by `source_video` in the edit decisions.

Current QA reports can contain:

- `raw_ranges`: mappings from final-video issues to the source recording;
- `raw_context`: source transcript words around an issue;
- `repair_suggestion.action`: `force_keep`, `force_drop`, `manual_review`, or `no_auto_repair`;
- `actionability`: whether the issue is mapped and requires manual review.

When an older QA report has no `raw_ranges`, the brief falls back to `edit_decisions.timeline_map`.

## Outputs

The command writes:

- `artifacts/editor_review*.md`: a human-readable review brief;
- `artifacts/editor_review*.csv`: a sortable checklist;
- optionally `artifacts/editor_review*_clips/` when `--make-clips` is enabled.

## Report fields

- `QA action` records the repair path selected by QA.
- `Repair status` distinguishes an automatic repair candidate from a manual decision.
- `QA confidence` helps prioritize risk.
- `Edited video` identifies the exact problem range in the final render.
- `Source comparison` identifies the corresponding source range with listening margin.
- `Source context` includes nearby words from the raw transcript.
- `Instructions for the editor` explains whether to check a missing connector, repetition, noise, or an uncertain transition.

## Safety rule

The brief does not modify the edit. It explains why QA or the repair planner considers a location safe to repair or in need of review. Uncertain decisions must remain `manual_review` instead of forcing an aggressive cut.
