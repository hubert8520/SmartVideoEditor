import json
import subprocess
import sys
from pathlib import Path

from smart_video_editor.reporting.quality import enrich_quality_report


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def word(word_id: int, start: float, end: float, text: str) -> dict[str, object]:
    def ts(seconds: float) -> str:
        milliseconds = int(round(seconds * 1000))
        total_seconds, ms = divmod(milliseconds, 1000)
        minutes, sec = divmod(total_seconds, 60)
        hours, minute = divmod(minutes, 60)
        return f"{hours:02d}:{minute:02d}:{sec:02d}:{ms:03d}"

    return {"id": word_id, "timestamp": ts(start), "end": ts(end), "word": text}


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def base_decisions() -> dict[str, object]:
    return {
        "keep_intervals": [
            {"start": "00:00:10:000", "end": "00:00:12:000"},
            {"start": "00:00:20:000", "end": "00:00:23:000"},
        ]
    }


def issue(
    *,
    category: str,
    action: str,
    confidence: str = "high",
    start: str = "00:00:00:300",
    end: str = "00:00:01:600",
    affected_text: str = "działa kampania",
) -> dict[str, object]:
    return {
        "start_word_id": 1,
        "end_word_id": 2,
        "start": start,
        "end": end,
        "issue_category": category,
        "severity": "high",
        "description": "synthetic QA issue",
        "affected_text": affected_text,
        "suggested_action": "synthetic action",
        "repair_suggestion": {
            "action": action,
            "confidence": confidence,
            "rationale": "synthetic rationale",
            "requires_manual_review": action == "manual_review",
        },
    }


def test_quality_report_enrichment_maps_final_issue_to_raw_time():
    report = {
        "status": "fail",
        "issues": [issue(category="logic_gap", action="force_keep")],
        "overall_notes": "gap",
    }
    raw_transcript = {
        "words": [
            word(0, 10.0, 10.2, "to"),
            word(1, 10.3, 10.5, "działa"),
            word(2, 11.4, 11.6, "kampania"),
        ]
    }

    enriched = enrich_quality_report(report, edit_decisions=base_decisions(), raw_transcript=raw_transcript)

    enriched_issue = enriched["issues"][0]
    assert enriched_issue["raw_ranges"][0]["raw_start"] == "00:00:10:300"
    assert enriched_issue["raw_ranges"][0]["raw_end"] == "00:00:11:600"
    assert enriched_issue["actionability"]["mapped_to_raw"] is True
    assert [item["word"] for item in enriched_issue["raw_context"]] == ["to", "działa", "kampania"]


def test_quality_report_enrichment_downgrades_unmapped_auto_drop_to_review():
    report = {
        "status": "fail",
        "issues": [issue(category="repetition", action="force_drop")],
        "overall_notes": "repeat",
    }

    enriched = enrich_quality_report(report)

    suggestion = enriched["issues"][0]["repair_suggestion"]
    assert suggestion["action"] == "manual_review"
    assert suggestion["original_action"] == "force_drop"
    assert enriched["issues"][0]["actionability"]["requires_manual_review"] is True


def run_repair(tmp_path: Path, quality_report: dict[str, object]) -> dict[str, object]:
    quality_path = tmp_path / "quality.json"
    decisions_path = tmp_path / "decisions.json"
    raw_path = tmp_path / "raw.json"
    edited_path = tmp_path / "edited.json"
    output_path = tmp_path / "repair.json"

    write_json(quality_path, quality_report)
    write_json(decisions_path, base_decisions())
    write_json(
        raw_path,
        {
            "words": [
                word(0, 10.0, 10.2, "to"),
                word(1, 10.3, 10.5, "działa"),
                word(2, 11.4, 11.6, "kampania"),
                word(3, 20.2, 20.4, "kaszel"),
            ]
        },
    )
    write_json(edited_path, {"words": [word(0, 0.0, 0.2, "to")]})

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "repair_from_quality_report.py"),
            "--quality-report",
            str(quality_path),
            "--edit-decisions",
            str(decisions_path),
            "--raw-transcript",
            str(raw_path),
            "--edited-transcript",
            str(edited_path),
            "--output",
            str(output_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(output_path.read_text(encoding="utf-8"))


def test_repair_plan_uses_actionable_force_keep_for_raw_gap(tmp_path):
    report = {
        "status": "fail",
        "issues": [issue(category="logic_gap", action="force_keep")],
        "overall_notes": "gap",
    }
    enriched = enrich_quality_report(report, edit_decisions=base_decisions())

    repair = run_repair(tmp_path, enriched)

    assert repair["repairs"][0]["type"] == "force_keep_interval"
    assert repair["repairs"][0]["raw_start"] == "00:00:10:500"
    assert repair["repairs"][0]["raw_end"] == "00:00:11:400"
    assert repair["render_source_policy"].startswith("repair renders from the original raw video")


def test_repair_plan_respects_manual_review_action(tmp_path):
    report = {
        "status": "needs_review",
        "issues": [issue(category="repetition", action="manual_review")],
        "overall_notes": "review",
    }
    enriched = enrich_quality_report(report, edit_decisions=base_decisions())

    repair = run_repair(tmp_path, enriched)

    assert repair["repairs"] == []
    assert repair["skipped_issues"][0]["skip_reason"] == "qa_requested_manual_review"
    assert repair["manual_review_issues"]


def test_repair_plan_refuses_inconsistent_force_drop_for_cut_word(tmp_path):
    report = {
        "status": "fail",
        "issues": [issue(category="cut_word", action="force_drop")],
        "overall_notes": "bad action",
    }
    enriched = enrich_quality_report(report, edit_decisions=base_decisions())

    repair = run_repair(tmp_path, enriched)

    assert repair["repairs"] == []
    assert repair["skipped_issues"][0]["skip_reason"] == "inconsistent_qa_repair_action"


def test_repair_plan_allows_short_high_confidence_noise_drop(tmp_path):
    report = {
        "status": "fail",
        "issues": [
            issue(
                category="noise_or_setup",
                action="force_drop",
                start="00:00:02:200",
                end="00:00:02:400",
                affected_text="kaszel",
            )
        ],
        "overall_notes": "noise",
    }
    enriched = enrich_quality_report(report, edit_decisions=base_decisions())

    repair = run_repair(tmp_path, enriched)

    assert repair["repairs"][0]["type"] == "force_drop_words"
    assert repair["repairs"][0]["start_word_id"] == 3
    assert repair["repairs"][0]["end_word_id"] == 3

