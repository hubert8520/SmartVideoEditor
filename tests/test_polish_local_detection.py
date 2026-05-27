from smart_video_editor.detection.local import detect_local_candidates
from smart_video_editor.domain.models import TranscriptWord
from smart_video_editor.segmentation.takes import segment_takes
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


def candidates_by_category(words: list[TranscriptWord], category: str):
    return [candidate for candidate in detect_local_candidates(words) if candidate.category == category]


def test_take_segmentation_splits_on_long_pause():
    words = make_words("pierwszy take drugi take", gaps={1: 1.8})

    takes = segment_takes(words, max_gap=1.2)

    assert [take.text for take in takes] == ["pierwszy take", "drugi take"]
    assert takes[0].word_ids == (0, 1)
    assert takes[1].word_ids == (2, 3)


def test_bad_marker_cluster_drops_failed_take_only_when_restart_is_confirmed():
    words = make_words("dzisiaj pokazuje kampanie kurwa jeszcze raz dzisiaj pokazuje kampanie poprawnie")

    markers = candidates_by_category(words, "bad_marker_take")

    assert len(markers) == 1
    assert markers[0].text == "dzisiaj pokazuje kampanie kurwa jeszcze raz"
    assert markers[0].reason == "bad_marker_failed_take_before_confirmed_restart"
    assert markers[0].recommended_action == "DROP"
    assert markers[0].evidence["failed_take"]["text"] == "dzisiaj pokazuje kampanie"
    assert markers[0].evidence["marker"]["text"] == "kurwa jeszcze raz"
    assert markers[0].evidence["marker_phrases"] == ["kurwa", "jeszcze raz"]
    assert markers[0].evidence["restart"]["confirmed"] is True
    assert markers[0].evidence["restart"]["prefix_word_count"] == 3
    assert markers[0].evidence["restart"]["text"] == "dzisiaj pokazuje kampanie poprawnie"


def test_bad_marker_cluster_keeps_filler_between_marker_words():
    words = make_words("dzisiaj pokazuje kampanie kurwa no jeszcze raz dzisiaj pokazuje kampanie poprawnie")

    markers = candidates_by_category(words, "bad_marker_take")

    assert len(markers) == 1
    assert markers[0].text == "dzisiaj pokazuje kampanie kurwa no jeszcze raz"
    assert markers[0].recommended_action == "DROP"


def test_lone_profanity_without_restart_is_review_not_failed_take_drop():
    words = make_words("to jest kurwa wazne dla kampanii")

    markers = candidates_by_category(words, "bad_marker_take")

    assert len(markers) == 1
    assert markers[0].text == "kurwa"
    assert markers[0].reason == "ambiguous_bad_marker_without_confirmed_restart"
    assert markers[0].recommended_action == "REVIEW"
    assert markers[0].evidence["has_failed_take_context"] is False
    assert markers[0].evidence["marker"]["text"] == "kurwa"
    assert markers[0].evidence["restart"]["confirmed"] is False


def test_retake_marker_without_successful_restart_is_review():
    words = make_words("ustawiam kampanie jeszcze raz teraz przechodze dalej")

    markers = candidates_by_category(words, "bad_marker_take")

    assert len(markers) == 1
    assert markers[0].text == "ustawiam kampanie jeszcze raz"
    assert markers[0].reason == "bad_marker_failed_take_without_confirmed_restart"
    assert markers[0].recommended_action == "REVIEW"
    assert markers[0].evidence["strong_retake_marker"] is True
    assert markers[0].evidence["failed_take"]["text"] == "ustawiam kampanie"
    assert markers[0].evidence["marker"]["text"] == "jeszcze raz"
    assert markers[0].evidence["restart"]["confirmed"] is False


def test_bad_marker_does_not_expand_across_previous_take_pause():
    words = make_words(
        "pierwszy zamkniety take ustawiam kampanie kurwa jeszcze raz ustawiam kampanie dobrze",
        gaps={2: 1.8},
    )

    markers = candidates_by_category(words, "bad_marker_take")

    assert markers[0].text == "ustawiam kampanie kurwa jeszcze raz"
    assert markers[0].recommended_action == "DROP"


def test_bad_marker_can_confirm_restart_after_pause():
    words = make_words(
        "ustawiam kampanie kurwa jeszcze raz ustawiam kampanie dobrze",
        gaps={4: 1.8},
    )

    markers = candidates_by_category(words, "bad_marker_take")

    assert markers[0].text == "ustawiam kampanie kurwa jeszcze raz"
    assert markers[0].recommended_action == "DROP"


def test_isolated_noise_marker_is_drop_candidate_with_gap_evidence():
    words = make_words("intro kaszel dalej", gaps={0: 0.7, 1: 0.7})

    noises = candidates_by_category(words, "noise_or_setup")

    assert len(noises) == 1
    assert noises[0].text == "kaszel"
    assert noises[0].reason == "noise_marker_isolated_from_speech"
    assert noises[0].recommended_action == "DROP"
    assert noises[0].evidence["noise"]["text"] == "kaszel"
    assert noises[0].evidence["previous_gap_seconds"] == 0.48
    assert noises[0].evidence["next_gap_seconds"] == 0.48
    assert noises[0].evidence["overlaps_speech_context"] is False


def test_noise_marker_touching_speech_is_review_candidate():
    words = make_words("intro kaszel dalej")

    noises = candidates_by_category(words, "noise_or_setup")

    assert len(noises) == 1
    assert noises[0].text == "kaszel"
    assert noises[0].reason == "noise_marker_overlaps_speech_context"
    assert noises[0].recommended_action == "REVIEW"
    assert noises[0].evidence["overlaps_speech_context"] is True


def test_multiword_setup_noise_marker_is_detected():
    words = make_words("start szuranie krzeslem dalej", gaps={0: 0.7, 2: 0.7})

    noises = candidates_by_category(words, "noise_or_setup")

    assert len(noises) == 1
    assert noises[0].text == "szuranie krzeslem"
    assert noises[0].evidence["marker_phrase"] == "szuranie krzeslem"


def test_truncated_word_restart_without_repeated_starter_is_drop_candidate():
    words = make_words("jak skonfi skonfigurowac kampanie")

    partial = candidates_by_category(words, "partial_repeat")

    assert any(candidate.text == "skonfi" for candidate in partial)
    assert all(candidate.recommended_action == "DROP" for candidate in partial)


def test_partial_repeat_with_filler_before_restart_is_candidate():
    words = make_words("jak skonfi yyy jak skonfigurowac kampanie")

    partial = candidates_by_category(words, "partial_repeat")

    assert any(candidate.text == "jak skonfi yyy" for candidate in partial)
