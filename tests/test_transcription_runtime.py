from smart_video_editor.transcription import runtime


def test_runtime_exports_project_paths():
    assert runtime.PROJECT_ROOT.name == "SmartVideoEditor"
    assert runtime.RAW_DIR == runtime.PROJECT_ROOT / "raw"
    assert runtime.ARTIFACTS_DIR == runtime.PROJECT_ROOT / "artifacts"


def test_get_media_duration_signature_is_compatible():
    assert callable(runtime.get_media_duration)


def test_timestamp_helpers_are_reexported():
    assert runtime.seconds_to_timestamp(1.234) == "00:00:01:234"
    assert runtime.timestamp_to_seconds("00:00:01:234") == 1.234
