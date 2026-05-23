from smart_video_editor.timecode import seconds_to_timestamp, timestamp_to_seconds


def test_timestamp_roundtrip():
    value = "00:01:02:345"
    assert seconds_to_timestamp(timestamp_to_seconds(value)) == value


def test_seconds_to_timestamp_clamps_negative():
    assert seconds_to_timestamp(-1.0) == "00:00:00:000"
