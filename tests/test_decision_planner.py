from types import SimpleNamespace

from smart_video_editor.domain.models import TranscriptWord
from smart_video_editor.planning.decision_planner import (
    PlannerCandidate,
    plan_candidates,
    plan_drop_windows,
)
from smart_video_editor.text import normalize_text


def make_words(
    text: str,
    *,
    start: float = 0.0,
    step: float = 0.35,
    gaps: dict[int, float] | None = None,
) -> list[TranscriptWord]:
    words = []
    cursor = start
    gaps = gaps or {}
    for index, raw_word in enumerate(text.split()):
        clean = raw_word.strip(".!,?…")
        words.append(
            TranscriptWord(
                id=index,
                timestamp=cursor,
                end=cursor + 0.22,
                text=clean,
                normalized=normalize_text(clean),
            )
        )
        cursor += gaps.get(index, step)
    return words


def plan(candidates: list[PlannerCandidate], words: list[TranscriptWord]):
    return plan_candidates(
        candidates,
        words=words,
        silences=[],
        thought_blocks=[],
        keep_ranges=[],
        duration=words[-1].end + 1.0,
        cut_safety_margin=0.04,
        silence_snap_window=0.0,
        context_words=4,
    )


def plan_with_local_detectors(words: list[TranscriptWord]):
    return plan_drop_windows(
        entries=[],
        partial_drop_windows=[],
        words=words,
        silences=[],
        thought_blocks=[],
        llm_summary=SimpleNamespace(keep_notes=[]),
        duration=words[-1].end + 1.0,
        cut_safety_margin=0.04,
        silence_snap_window=0.0,
        context_words=4,
        disable_boundary_validator=False,
    )


def test_planner_approves_safe_partial_repeat_candidate():
    words = make_words("jak skonfi jak skonfigurować kampanię")
    candidate = PlannerCandidate(
        start=words[0].timestamp,
        end=words[1].end,
        reason="abandoned_partial_phrase_before_complete_restart",
        source_text="jak skonfi",
        word_ids=(0, 1),
        source="local",
        category="partial_repeat",
        confidence=0.92,
    )

    result = plan([candidate], words)

    assert len(result.applied_windows) == 1
    assert not result.blocked_windows
    assert result.drop_windows
    assert "jak skonfigurować kampanię" in result.simulated_text


def test_planner_applies_local_repeated_attempt_candidate():
    words = make_words("Dzisiaj pokażę wam Dzisiaj pokażę wam jak ustawić kampanię")

    result = plan_with_local_detectors(words)

    assert any(window["category"] == "repeated_attempt" for window in result.applied_windows)
    assert result.drop_windows
    assert result.simulated_text == "Dzisiaj pokażę wam jak ustawić kampanię"


def test_planner_records_local_candidate_evidence():
    words = make_words("Dzisiaj pokażę wam Dzisiaj pokażę wam jak ustawić kampanię")

    result = plan_with_local_detectors(words)
    repeated_attempt = next(
        window for window in result.applied_windows if window["category"] == "repeated_attempt"
    )

    assert repeated_attempt["evidence"]["earlier"]["completeness"]["is_complete"] is False
    assert repeated_attempt["evidence"]["later"]["completeness"]["is_complete"] is True


def test_planner_records_candidate_inventory_before_decisions():
    words = make_words("Dzisiaj pokażę wam Dzisiaj pokażę wam jak ustawić kampanię")

    result = plan_with_local_detectors(words)
    inventory_item = next(
        candidate
        for candidate in result.candidate_inventory
        if candidate["category"] == "repeated_attempt"
    )
    applied_item = next(
        window for window in result.applied_windows if window["category"] == "repeated_attempt"
    )

    assert inventory_item["planner_candidate_id"] == applied_item["planner_candidate_id"]
    assert inventory_item["candidate_index"] == applied_item["candidate_index"]
    assert inventory_item["recommended_action"] == "DROP"
    assert inventory_item["evidence"]["later"]["completeness"]["is_complete"] is True


def test_planner_records_bad_marker_evidence():
    words = make_words("ustawiam kampanię kurwa jeszcze raz ustawiam kampanię poprawnie")

    result = plan_with_local_detectors(words)
    marker = next(window for window in result.applied_windows if window["category"] == "bad_marker_take")

    assert marker["evidence"]["failed_take"]["text"] == "ustawiam kampanię"
    assert marker["evidence"]["marker"]["text"] == "kurwa jeszcze raz"
    assert marker["evidence"]["restart"]["confirmed"] is True


def test_planner_applies_isolated_noise_candidate():
    words = make_words("intro kaszel dalej", gaps={0: 0.7, 1: 0.7})

    result = plan_with_local_detectors(words)
    noise = next(window for window in result.applied_windows if window["category"] == "noise_or_setup")

    assert noise["evidence"]["noise"]["text"] == "kaszel"
    assert noise["evidence"]["overlaps_speech_context"] is False
    assert result.simulated_text == "intro dalej"


def test_planner_reviews_noise_touching_speech_without_cutting():
    words = make_words("intro kaszel dalej")

    result = plan_with_local_detectors(words)

    assert not result.applied_windows
    assert not result.drop_windows
    assert any(window["category"] == "noise_or_setup" for window in result.review_windows)


def test_planner_reviews_repeated_attempt_when_later_take_is_incomplete():
    words = make_words("Dzisiaj pokażę wam Dzisiaj pokażę wam jak to")

    result = plan_with_local_detectors(words)

    assert not result.applied_windows
    assert not result.drop_windows
    assert any(
        window["category"] == "repeated_attempt"
        and window["review_reason"] == "candidate_marked_review"
        for window in result.review_windows
    )


def test_planner_reviews_rhetorical_repeat_candidate_without_cutting():
    words = make_words("to jest ważne to jest ważne dla mnie")

    result = plan_with_local_detectors(words)

    assert not result.applied_windows
    assert not result.drop_windows
    assert any(window["category"] == "repeated_attempt" for window in result.review_windows)


def test_planner_blocks_removed_logical_connector():
    words = make_words("działa bo inaczej kampania nie ruszy")
    candidate = PlannerCandidate(
        start=words[1].timestamp,
        end=words[1].end,
        reason="candidate_would_remove_connector",
        source_text="bo",
        word_ids=(1,),
        source="local",
        category="connector",
        confidence=0.9,
    )

    result = plan([candidate], words)

    assert not result.applied_windows
    assert result.blocked_windows
    assert result.blocked_windows[0]["block_reason"] == "boundary_validator"
    assert result.blocked_windows[0]["boundary_issues"][0]["reason"] == "cut_removed_logical_connector"


def test_planner_blocks_timestamp_candidate_that_cuts_mid_word():
    words = make_words("to jest test")
    candidate = PlannerCandidate(
        start=words[1].timestamp + 0.04,
        end=words[1].end + 0.1,
        reason="timestamp_only_mid_word_cut",
        source_text="est",
        source="llm",
        category="timestamp_only",
        confidence=0.95,
    )

    result = plan([candidate], words)

    assert not result.applied_windows
    assert result.blocked_windows
    assert result.blocked_windows[0]["boundary_issues"][0]["kind"] == "mid_word_cut"


def test_planner_honors_review_candidates_without_cutting():
    words = make_words("stop ustawiam kampanię")
    candidate = PlannerCandidate(
        start=words[0].timestamp,
        end=words[0].end,
        reason="marker_without_context",
        source_text="stop",
        word_ids=(0,),
        source="local",
        category="bad_marker_take",
        confidence=0.72,
        recommended_action="REVIEW",
    )

    result = plan([candidate], words)

    assert not result.applied_windows
    assert not result.drop_windows
    assert result.review_windows
    assert result.review_windows[0]["review_reason"] == "candidate_marked_review"


def test_planner_blocks_heuristic_partial_thought_cut_but_allows_local_candidate():
    words = make_words("dzisiaj pokażę jak ustawić kampanię")
    thought_block = SimpleNamespace(
        role="thought",
        word_ids=[word.id for word in words],
    )
    heuristic_candidate = PlannerCandidate(
        start=words[2].timestamp,
        end=words[3].end,
        reason="legacy_heuristic_partial_cut",
        source_text="jak ustawić",
        word_ids=(2, 3),
        source="heuristic",
        category="legacy",
    )
    local_candidate = PlannerCandidate(
        start=words[2].timestamp,
        end=words[3].end,
        reason="high_confidence_local_partial_repeat",
        source_text="jak ustawić",
        word_ids=(2, 3),
        source="local",
        category="partial_repeat",
        confidence=0.92,
    )

    heuristic_result = plan_candidates(
        [heuristic_candidate],
        words=words,
        silences=[],
        thought_blocks=[thought_block],
        keep_ranges=[],
        duration=words[-1].end + 1.0,
        cut_safety_margin=0.04,
        silence_snap_window=0.0,
        context_words=4,
    )
    local_result = plan_candidates(
        [local_candidate],
        words=words,
        silences=[],
        thought_blocks=[thought_block],
        keep_ranges=[],
        duration=words[-1].end + 1.0,
        cut_safety_margin=0.04,
        silence_snap_window=0.0,
        context_words=4,
    )

    assert heuristic_result.blocked_windows[0]["block_reason"] == "heuristic_partial_thought_cut"
    assert local_result.applied_windows
