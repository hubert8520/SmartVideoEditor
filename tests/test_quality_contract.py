import json

from smart_video_editor.detection.local import detect_local_candidates
from smart_video_editor.domain.models import TranscriptWord
from smart_video_editor.planning.boundary import (
    trim_short_block_tail_to_silence,
    validate_cut_boundaries,
)
from smart_video_editor.planning.edl import EditDecisionList, KeepInterval
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


def test_partial_repeat_candidate_for_truncated_restart():
    words = make_words("jak skonfi jak skonfigurować kampanię")

    candidates = detect_local_candidates(words)

    partial = [candidate for candidate in candidates if candidate.category == "partial_repeat"]
    assert partial
    assert partial[0].text == "jak skonfi"
    assert partial[0].recommended_action == "DROP"


def test_repeated_take_prefix_candidate_keeps_fuller_later_take():
    words = make_words("Dzisiaj pokażę Dzisiaj pokażę wam jak ustawić kampanię")

    candidates = detect_local_candidates(words)

    repeated = [candidate for candidate in candidates if candidate.category == "repeated_take"]
    assert repeated
    assert repeated[0].text == "Dzisiaj pokażę"
    assert repeated[0].recommended_action == "DROP"


def test_bad_marker_expands_to_failed_take():
    words = make_words("ustawiam kampanię kurwa jeszcze raz ustawiam kampanię poprawnie")

    candidates = detect_local_candidates(words)

    marker = [candidate for candidate in candidates if candidate.category == "bad_marker_take"]
    assert marker
    assert marker[0].text == "ustawiam kampanię kurwa"
    assert marker[0].recommended_action == "DROP"


def test_short_heading_tail_trims_to_safe_silence():
    words = make_words("Punkt pierwszy")
    last_word = words[-1]

    trimmed = trim_short_block_tail_to_silence(
        last_word,
        original_end=last_word.end + 0.5,
        silences=[(last_word.end + 0.08, last_word.end + 0.4)],
        min_spoken_before_trim=0.1,
        silence_window=0.45,
        tail_padding=0.02,
    )

    assert trimmed == last_word.end + 0.1


def test_boundary_validator_blocks_mid_word_cut():
    words = make_words("to jest test")

    issues = validate_cut_boundaries(words, [(words[1].timestamp + 0.04, words[1].end + 0.1)])

    assert any(issue.kind == "mid_word_cut" and issue.action == "BLOCK" for issue in issues)


def test_boundary_validator_reviews_removed_logical_connector():
    words = make_words("działa bo inaczej kampania nie ruszy")

    issues = validate_cut_boundaries(words, [(words[1].timestamp - 0.01, words[2].end + 0.01)])

    assert any(issue.reason == "cut_removed_logical_connector" for issue in issues)


def test_timeline_map_maps_final_time_to_raw_time():
    edl = EditDecisionList(
        keep_intervals=(
            KeepInterval(10.0, 12.0),
            KeepInterval(20.0, 23.0),
        )
    )

    assert edl.map_final_time_to_raw(2.5) == 20.5


def test_edl_json_roundtrip():
    edl = EditDecisionList(
        keep_intervals=(
            KeepInterval(1.0, 2.5),
            KeepInterval(5.0, 7.0),
        )
    )

    payload = json.loads(json.dumps(edl.to_json_dict()))
    restored = EditDecisionList.from_json_dict(payload)

    assert restored == edl
