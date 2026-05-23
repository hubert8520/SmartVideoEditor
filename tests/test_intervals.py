from smart_video_editor.editing.intervals import merge_intervals, subtract_intervals


def test_merge_intervals_with_gap():
    assert merge_intervals([(0.0, 1.0), (1.1, 2.0)], gap=0.2) == [(0.0, 2.0)]


def test_subtract_intervals():
    assert subtract_intervals([(0.0, 10.0)], [(2.0, 4.0), (6.0, 8.0)]) == [
        (0.0, 2.0),
        (4.0, 6.0),
        (8.0, 10.0),
    ]
