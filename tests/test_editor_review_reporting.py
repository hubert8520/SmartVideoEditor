import csv
import json
from pathlib import Path

from smart_video_editor.reporting.editor_review import (
    build_editor_review_rows,
    planner_evidence_rows,
    review_summary,
    write_editor_review_csv,
    write_editor_review_markdown,
)


def issue(action: str, *, requires_review: bool = False) -> dict[str, object]:
    return {
        "start_word_id": 1,
        "end_word_id": 2,
        "start": "00:00:00:300",
        "end": "00:00:01:600",
        "issue_category": "logic_gap",
        "severity": "high",
        "description": "Brakuje logicznego lacznika.",
        "affected_text": "dziala kampania",
        "suggested_action": "Sprawdz raw.",
        "repair_suggestion": {
            "action": action,
            "confidence": "high",
            "rationale": "Raw zawiera brakujacy lacznik.",
            "requires_manual_review": requires_review,
        },
        "actionability": {
            "mapped_to_raw": True,
            "timeline_ids": [0],
            "requires_manual_review": requires_review,
        },
        "raw_ranges": [
            {
                "timeline_id": 0,
                "raw_start": "00:00:10:300",
                "raw_end": "00:00:11:600",
                "raw_start_seconds": 10.3,
                "raw_end_seconds": 11.6,
            }
        ],
        "raw_context": [
            {"id": 1, "start": "00:00:10:300", "end": "00:00:10:500", "word": "dziala"},
            {"id": 2, "start": "00:00:11:400", "end": "00:00:11:600", "word": "kampania"},
        ],
    }


def decisions() -> dict[str, object]:
    return {
        "keep_intervals": [
            {"start": "00:00:10:000", "end": "00:00:12:000"},
        ]
    }


def test_editor_review_rows_include_actionability_and_raw_context():
    report = {
        "status": "needs_review",
        "issues": [issue("force_keep")],
        "overall_notes": "Jest luka.",
    }

    rows = build_editor_review_rows(report, decisions(), clip_margin=1.0)

    assert rows[0]["action"] == "force_keep"
    assert rows[0]["actionability"] == "auto_repair_candidate"
    assert rows[0]["raw_context"] == "dziala kampania"
    assert rows[0]["raw_compare_ranges"] == ["00:00:09:300 - 00:00:12:600"]
    assert "missing word" in rows[0]["editor_instruction"]


def test_editor_review_rows_fallback_to_timeline_when_report_has_no_raw_ranges():
    report = {
        "status": "fail",
        "issues": [
            {
                "start": "00:00:00:300",
                "end": "00:00:01:000",
                "issue_category": "repetition",
                "severity": "medium",
                "description": "Powtorka.",
                "affected_text": "test test",
                "suggested_action": "Usun powtorke.",
            }
        ],
        "overall_notes": "",
    }

    rows = build_editor_review_rows(report, decisions(), clip_margin=0.0)

    assert rows[0]["raw_ranges"][0]["raw_range"] == "00:00:10:300 - 00:00:11:000"
    assert rows[0]["action"] == "force_drop"
    assert rows[0]["actionability"] == "auto_repair_candidate"


def test_editor_review_outputs_markdown_and_csv_with_repair_columns(tmp_path: Path):
    report = {
        "status": "needs_review",
        "issues": [issue("manual_review", requires_review=True)],
        "overall_notes": "Do odsluchu.",
    }
    rows = build_editor_review_rows(report, decisions(), clip_margin=0.5)
    markdown_path = tmp_path / "review.md"
    csv_path = tmp_path / "review.csv"

    write_editor_review_markdown(
        markdown_path,
        rows,
        report,
        "edited/edited_video.mp4",
        "raw/source.mp4",
        make_clips=False,
    )
    write_editor_review_csv(csv_path, rows, "edited/edited_video.mp4", "raw/source.mp4")

    markdown = markdown_path.read_text(encoding="utf-8")
    assert "QA action: `manual_review`" in markdown
    assert "Source context" in markdown
    assert "Manual review required: **1**" in markdown

    with csv_path.open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle, delimiter=";"))
    assert csv_rows[0]["qa_action"] == "manual_review"
    assert csv_rows[0]["repair_status"] == "manual_review"
    assert csv_rows[0]["source_context"] == "dziala kampania"


def test_editor_review_markdown_can_include_planner_evidence(tmp_path: Path):
    report = {"status": "pass", "issues": [], "overall_notes": ""}
    edit_decisions = {
        "cut_planner_review": {
            "review_windows": [
                {
                    "candidate_start": "00:00:00:000",
                    "candidate_end": "00:00:01:000",
                    "category": "repeated_attempt",
                    "source_text": "Dzisiaj pokażę wam",
                    "reason": "later_attempt_also_incomplete_needs_review",
                    "evidence": {
                        "earlier": {
                            "completeness": {
                                "score": 0.2,
                                "is_complete": False,
                                "markers": ["ends_with_incomplete_token"],
                            }
                        },
                        "later": {
                            "completeness": {
                                "score": 0.5,
                                "is_complete": False,
                                "markers": ["ends_with_incomplete_token"],
                            }
                        },
                    },
                }
            ]
        }
    }
    markdown_path = tmp_path / "review.md"

    evidence = planner_evidence_rows(edit_decisions)
    write_editor_review_markdown(
        markdown_path,
        [],
        report,
        "edited/edited_video.mp4",
        "raw/source.mp4",
        make_clips=False,
        planner_evidence=evidence,
    )

    markdown = markdown_path.read_text(encoding="utf-8")
    assert evidence[0]["action"] == "REVIEW"
    assert "Planner evidence" in markdown
    assert "later_attempt_also_incomplete_needs_review" in markdown
    assert "ends_with_incomplete_token" in markdown


def test_editor_review_summarizes_bad_marker_evidence(tmp_path: Path):
    report = {"status": "pass", "issues": [], "overall_notes": ""}
    edit_decisions = {
        "cut_planner_review": {
            "applied_windows": [
                {
                    "candidate_start": "00:00:00:000",
                    "candidate_end": "00:00:01:500",
                    "category": "bad_marker_take",
                    "source_text": "ustawiam kampanie kurwa jeszcze raz",
                    "reason": "bad_marker_failed_take_before_confirmed_restart",
                    "evidence": {
                        "failed_take": {"text": "ustawiam kampanie"},
                        "marker": {"text": "kurwa jeszcze raz"},
                        "restart": {
                            "confirmed": True,
                            "prefix_word_count": 2,
                            "text": "ustawiam kampanie poprawnie",
                        },
                    },
                }
            ]
        }
    }
    markdown_path = tmp_path / "review.md"

    evidence = planner_evidence_rows(edit_decisions)
    write_editor_review_markdown(
        markdown_path,
        [],
        report,
        "edited/edited_video.mp4",
        "raw/source.mp4",
        make_clips=False,
        planner_evidence=evidence,
    )

    markdown = markdown_path.read_text(encoding="utf-8")
    assert evidence[0]["summary"].startswith("marker=kurwa jeszcze raz")
    assert "restart_confirmed=True" in markdown
    assert "ustawiam kampanie poprawnie" in markdown


def test_editor_review_summarizes_noise_evidence(tmp_path: Path):
    report = {"status": "pass", "issues": [], "overall_notes": ""}
    edit_decisions = {
        "cut_planner_review": {
            "applied_windows": [
                {
                    "candidate_start": "00:00:00:700",
                    "candidate_end": "00:00:00:900",
                    "category": "noise_or_setup",
                    "source_text": "kaszel",
                    "reason": "noise_marker_isolated_from_speech",
                    "evidence": {
                        "noise": {"text": "kaszel"},
                        "previous_gap_seconds": 0.48,
                        "next_gap_seconds": 0.48,
                        "overlaps_speech_context": False,
                    },
                }
            ]
        }
    }
    markdown_path = tmp_path / "review.md"

    evidence = planner_evidence_rows(edit_decisions)
    write_editor_review_markdown(
        markdown_path,
        [],
        report,
        "edited/edited_video.mp4",
        "raw/source.mp4",
        make_clips=False,
        planner_evidence=evidence,
    )

    markdown = markdown_path.read_text(encoding="utf-8")
    assert evidence[0]["summary"].startswith("noise=kaszel")
    assert "overlaps_speech_context=False" in markdown
    assert "previous_gap=0.48" in markdown


def test_review_summary_counts_actions():
    rows = [
        {"actionability": "auto_repair_candidate", "raw_ranges": [1]},
        {"actionability": "manual_review", "raw_ranges": []},
        {"actionability": "blocked_no_raw_mapping", "raw_ranges": []},
    ]

    assert review_summary(rows) == {
        "issues": 3,
        "manual_review": 2,
        "auto_repair_candidates": 1,
        "mapped_to_raw": 1,
    }


def test_editor_review_rows_are_json_serializable():
    report = {"status": "fail", "issues": [issue("force_keep")], "overall_notes": ""}

    rows = build_editor_review_rows(report, decisions(), clip_margin=0.5)

    json.dumps(rows)
