import json
from types import SimpleNamespace

from smart_video_editor.domain.models import TranscriptWord
from smart_video_editor.editing.decisions_io import write_decisions


def test_write_decisions_includes_planner_candidate_inventory(tmp_path):
    output_path = tmp_path / "edited.mp4"
    decisions_path = tmp_path / "edit_decisions.json"
    planner_result = SimpleNamespace(
        candidate_inventory=[
            {
                "planner_candidate_id": "planner-001",
                "category": "noise_or_setup",
                "source_text": "kaszel",
            }
        ],
        applied_windows=[],
        blocked_windows=[],
        review_windows=[],
        boundary_issues=[],
        simulated_text="intro dalej",
    )
    llm_summary = SimpleNamespace(
        path="",
        status="missing",
        min_confidence=0.75,
        applied_drop_ranges=[],
        skipped_drop_ranges=[],
        review_ranges=[],
        keep_notes=[],
        thought_blocks=[],
    )
    repair_summary = SimpleNamespace(
        path="",
        status="missing",
        repairs=[],
        skipped_repairs=[],
        forced_keep_records=[],
    )
    keep_settings = SimpleNamespace(
        use_word_mask=True,
        word_head_padding=0.0,
        word_tail_padding=0.0,
        short_block_tail_padding=0.0,
        short_block_max_words=3,
        short_block_silence_trim=True,
        short_block_silence_min_duration=0.08,
        short_block_silence_window=0.45,
        short_block_min_spoken_before_trim=0.25,
    )
    words = [
        TranscriptWord(
            id=0,
            timestamp=0.0,
            end=0.2,
            text="intro",
            normalized="intro",
        )
    ]

    write_decisions(
        transcript_path=tmp_path / "raw_transcription.json",
        transcript_metadata={"word_level": True},
        media_path=tmp_path / "raw.mp4",
        output_path=output_path,
        edit_decisions_path=decisions_path,
        duration=1.0,
        silences=[],
        keep_source="word_mask",
        llm_summary=llm_summary,
        repair_summary=repair_summary,
        keep_intervals=[(0.0, 1.0)],
        drop_windows=[],
        partial_drop_windows=[],
        review_windows=[],
        planner_result=planner_result,
        thought_blocks=[],
        short_block_trim_records=[],
        entries=[],
        words=words,
        padding=0.05,
        keep_settings=keep_settings,
        word_gap_merge=0.25,
        cut_safety_margin=0.04,
        silence_snap_window=0.18,
        dry_run=True,
    )

    payload = json.loads(decisions_path.read_text(encoding="utf-8"))

    assert payload["cut_planner_review"]["candidate_inventory"] == planner_result.candidate_inventory
