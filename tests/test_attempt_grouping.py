from smart_video_editor.detection.local import detect_local_candidates
from smart_video_editor.domain.models import TranscriptWord
from smart_video_editor.segmentation.attempts import (
    find_repeated_attempt_groups,
    score_attempt_completeness,
)
from smart_video_editor.text import normalize_text


def make_words(text: str, *, start: float = 0.0, step: float = 0.35) -> list[TranscriptWord]:
    words = []
    cursor = start
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
        cursor += step
    return words


def repeated_attempt_candidates(words: list[TranscriptWord]):
    return [candidate for candidate in detect_local_candidates(words) if candidate.category == "repeated_attempt"]


def test_attempt_group_drops_short_starter_before_fuller_restart():
    words = make_words("Dzisiaj pokażę Dzisiaj pokażę wam jak ustawić kampanię")

    groups = find_repeated_attempt_groups(words)
    candidates = repeated_attempt_candidates(words)

    assert groups[0].earlier.text == "Dzisiaj pokażę"
    assert groups[0].later.text == "Dzisiaj pokażę wam jak ustawić kampanię"
    assert groups[0].recommended_action == "DROP"
    assert groups[0].reason == "earlier_incomplete_attempt_before_fuller_restart"
    assert candidates[0].text == "Dzisiaj pokażę"
    assert candidates[0].recommended_action == "DROP"


def test_attempt_completeness_scores_finished_and_unfinished_attempts():
    unfinished = score_attempt_completeness(("Dzisiaj", "pokażę", "wam", "jak"))
    finished = score_attempt_completeness(
        ("Dzisiaj", "pokażę", "wam", "jak", "ustawić", "kampanię")
    )

    assert not unfinished.is_complete
    assert "ends_with_incomplete_token" in unfinished.markers
    assert finished.is_complete
    assert finished.score > unfinished.score


def test_attempt_group_drops_incomplete_pronoun_tail_before_restart():
    words = make_words("Dzisiaj pokażę wam Dzisiaj pokażę wam jak ustawić kampanię")

    groups = find_repeated_attempt_groups(words)
    candidates = repeated_attempt_candidates(words)

    assert groups[0].earlier.text == "Dzisiaj pokażę wam"
    assert groups[0].recommended_action == "DROP"
    assert groups[0].reason == "earlier_incomplete_attempt_before_fuller_restart"
    assert candidates[0].recommended_action == "DROP"


def test_attempt_group_reviews_when_later_attempt_is_still_incomplete():
    words = make_words("Dzisiaj pokażę wam Dzisiaj pokażę wam jak to")

    groups = find_repeated_attempt_groups(words)
    candidates = repeated_attempt_candidates(words)

    assert groups[0].recommended_action == "REVIEW"
    assert groups[0].reason == "later_attempt_also_incomplete_needs_review"
    assert not groups[0].later_completeness.is_complete
    assert candidates[0].recommended_action == "REVIEW"


def test_repeated_attempt_candidate_carries_completeness_evidence():
    words = make_words("Dzisiaj pokażę wam Dzisiaj pokażę wam jak ustawić kampanię")

    candidates = repeated_attempt_candidates(words)
    evidence = candidates[0].evidence

    assert evidence["later_extra_word_count"] == 3
    assert evidence["earlier"]["completeness"]["is_complete"] is False
    assert "ends_with_incomplete_token" in evidence["earlier"]["completeness"]["markers"]
    assert evidence["later"]["completeness"]["is_complete"] is True


def test_attempt_group_reviews_possible_rhetorical_repeat():
    words = make_words("to jest ważne to jest ważne dla mnie")

    groups = find_repeated_attempt_groups(words)
    candidates = repeated_attempt_candidates(words)

    assert groups[0].earlier.text == "to jest ważne"
    assert groups[0].recommended_action == "REVIEW"
    assert groups[0].reason == "possible_rhetorical_or_intentional_repeat"
    assert candidates[0].recommended_action == "REVIEW"
    assert not any(
        candidate.text == "to jest ważne" and candidate.recommended_action == "DROP"
        for candidate in detect_local_candidates(words)
    )


def test_attempt_group_ignores_paraphrase_without_same_restart_prefix():
    words = make_words("dzisiaj pokażę kampanię zaraz omówię ustawienia kampanii")

    assert find_repeated_attempt_groups(words) == []
    assert repeated_attempt_candidates(words) == []


def test_attempt_group_reviews_exact_repeat_without_fuller_later_take():
    words = make_words("to działa to działa")

    groups = find_repeated_attempt_groups(words)

    assert groups[0].recommended_action == "REVIEW"
    assert groups[0].reason == "repeated_attempt_without_fuller_later_take"
