from smart_video_editor.editing import runtime


def test_editing_runtime_paths():
    assert runtime.PROJECT_ROOT.name == "SmartVideoEditor"
    assert runtime.RAW_DIR == runtime.PROJECT_ROOT / "raw"
    assert runtime.EDITED_DIR == runtime.PROJECT_ROOT / "edited"
    assert runtime.ARTIFACTS_DIR == runtime.PROJECT_ROOT / "artifacts"


def test_editing_runtime_artifact_paths():
    assert runtime.DEFAULT_OUTPUT_PATH == runtime.EDITED_DIR / "edited_video.mp4"
    assert runtime.EDIT_DECISIONS_PATH == runtime.ARTIFACTS_DIR / "edit_decisions.json"
    assert runtime.LLM_EDIT_DECISIONS_PATH == runtime.ARTIFACTS_DIR / "llm_edit_decisions.json"
    assert runtime.REPAIR_PLAN_PATH == runtime.ARTIFACTS_DIR / "repair_plan.json"


def test_editing_runtime_timecode_helpers():
    assert runtime.seconds_to_timestamp(62.345) == "00:01:02:345"
    assert runtime.timestamp_to_seconds("00:01:02:345") == 62.345


def test_editing_runtime_text_helpers():
    assert runtime.normalize_text("Zażółć gęślą JAŹŃ!") == "zazołc gesla jazn"
    assert runtime.tokenize("Ala, ma kota.") == ["ala", "ma", "kota"]
