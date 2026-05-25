import importlib.util
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "analyze_transcript_llm.py"


def load_analyzer():
    spec = importlib.util.spec_from_file_location("analyze_transcript_llm", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def sample_transcript():
    return {
        "version": "2.0",
        "source": {"provider": "deepgram", "word_level": True},
        "segments": [
            {
                "id": 0,
                "timestamp": "00:00:00:000",
                "end": "00:00:02:000",
                "transcription": "jak skonfi jak skonfigurować kampanię",
                "word_ids": [0, 1, 2, 3, 4],
            }
        ],
        "words": [
            {"id": 0, "timestamp": "00:00:00:000", "end": "00:00:00:200", "word": "jak"},
            {"id": 1, "timestamp": "00:00:00:350", "end": "00:00:00:550", "word": "skonfi"},
            {"id": 2, "timestamp": "00:00:00:700", "end": "00:00:00:900", "word": "jak"},
            {"id": 3, "timestamp": "00:00:01:050", "end": "00:00:01:350", "word": "skonfigurować"},
            {"id": 4, "timestamp": "00:00:01:500", "end": "00:00:01:760", "word": "kampanię"},
        ],
    }


def valid_decisions():
    return {
        "thought_blocks": [
            {
                "id": 0,
                "start_word_id": 2,
                "end_word_id": 4,
                "start": "00:00:00:700",
                "end": "00:00:01:760",
                "role": "core_explanation",
                "text": "jak skonfigurować kampanię",
                "must_keep": True,
                "note": "pełna wersja myśli",
            }
        ],
        "drop_ranges": [
            {
                "start_word_id": 0,
                "end_word_id": 1,
                "start": "00:00:00:000",
                "end": "00:00:00:550",
                "reason": "Urwana próba przed pełnym restartem.",
                "reason_category": "repeated_take",
                "confidence": 0.93,
                "affected_text": "jak skonfi",
                "candidate_ids": ["local-001"],
                "cut_risk": "low",
                "preserves_meaning": True,
                "safety_basis": "Pełna wersja zaczyna się od word_id 2 i zawiera dokończone słowo.",
            }
        ],
        "review_ranges": [],
        "candidate_reviews": [
            {
                "candidate_id": "local-001",
                "decision": "approve_drop",
                "reason": "To urwana wersja prefiksu.",
                "safety_basis": "Zostaje kompletne powtórzenie bez utraty sensu.",
                "target": "drop_ranges",
            }
        ],
        "keep_notes": [],
        "overall_notes": "Jedna bezpieczna częściowa powtórka.",
    }


def test_user_prompt_includes_local_candidates_and_safety_instructions():
    analyzer = load_analyzer()
    transcript = sample_transcript()

    candidates = analyzer.local_candidates_for_prompt(transcript)
    prompt = analyzer.build_user_prompt(transcript)

    assert candidates[0]["id"] == "local-001"
    assert candidates[0]["category"] == "partial_repeat"
    assert "Local candidates" in prompt
    assert "safety_basis" in analyzer.SYSTEM_PROMPT
    assert "jak skonfi" in prompt


def test_validate_output_accepts_candidate_aware_schema():
    analyzer = load_analyzer()

    analyzer.validate_output(valid_decisions(), sample_transcript())


def test_validate_output_rejects_non_low_risk_drop():
    analyzer = load_analyzer()
    decisions = valid_decisions()
    decisions["drop_ranges"][0]["cut_risk"] = "medium"

    try:
        analyzer.validate_output(decisions, sample_transcript())
    except SystemExit:
        return
    raise AssertionError("validate_output should reject medium-risk automatic drops")


def test_validate_output_rejects_unknown_candidate_id():
    analyzer = load_analyzer()
    decisions = valid_decisions()
    decisions["drop_ranges"][0]["candidate_ids"] = ["missing-candidate"]

    try:
        analyzer.validate_output(decisions, sample_transcript())
    except SystemExit:
        return
    raise AssertionError("validate_output should reject unknown candidate ids")


def test_analyze_transcript_dry_run_reports_local_candidates(tmp_path):
    transcript_path = tmp_path / "raw_transcription.json"
    transcript_path.write_text(__import__("json").dumps(sample_transcript()), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--transcript", str(transcript_path), "--dry-run"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Local candidates: 1" in result.stdout


def test_edit_video_loader_preserves_and_enforces_llm_safety_fields(tmp_path):
    from smart_video_editor.cli import edit_video as edit_video_cli

    path = tmp_path / "llm_edit_decisions.json"
    decisions = valid_decisions()
    decisions["drop_ranges"].append(
        {
            **decisions["drop_ranges"][0],
            "start_word_id": 2,
            "end_word_id": 3,
            "start": "00:00:00:700",
            "end": "00:00:01:350",
            "affected_text": "jak skonfigurować",
            "cut_risk": "medium",
        }
    )
    path.write_text(__import__("json").dumps(decisions), encoding="utf-8")
    words = [
        edit_video_cli.TranscriptWord(
            id=word["id"],
            timestamp=edit_video_cli.timestamp_to_seconds(word["timestamp"]),
            end=edit_video_cli.timestamp_to_seconds(word["end"]),
            text=word["word"],
            normalized=word["word"].lower(),
        )
        for word in sample_transcript()["words"]
    ]

    windows, summary = edit_video_cli.load_llm_decisions(
        path,
        duration=2.0,
        words=words,
        min_confidence=0.75,
        ignore=False,
    )

    assert len(windows) == 1
    assert summary.applied_drop_ranges[0]["safety_basis"]
    assert summary.skipped_drop_ranges[0]["skip_reason"] == "unsafe_llm_drop_contract"
