import py_compile
import subprocess
import sys
from pathlib import Path

from smart_video_editor.cli import edit_video as edit_video_cli


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATHS = [
    PROJECT_ROOT / "scripts" / "edit_video.py",
    PROJECT_ROOT / "scripts" / "transcribe_video.py",
    PROJECT_ROOT / "scripts" / "analyze_transcript_llm.py",
    PROJECT_ROOT / "scripts" / "quality_check_edited_video.py",
    PROJECT_ROOT / "scripts" / "repair_from_quality_report.py",
    PROJECT_ROOT / "scripts" / "auto_refine_video.py",
    PROJECT_ROOT / "scripts" / "generate_editor_review.py",
]


def test_script_help_imports_package_from_repo_root():
    for script in SCRIPT_PATHS:
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr


def test_py_compile_all_cli_scripts():
    for script in SCRIPT_PATHS:
        py_compile.compile(str(script), doraise=True)


def test_edit_video_cli_uses_extracted_modules():
    assert edit_video_cli.detect_silences.__module__ == "smart_video_editor.media.rendering"
    assert edit_video_cli.render_video.__module__ == "smart_video_editor.media.rendering"
    assert edit_video_cli.write_decisions.__module__ == "smart_video_editor.editing.decisions_io"
    assert edit_video_cli.run_quality_check.__module__ == "smart_video_editor.editing.quality"


def test_edit_video_script_is_thin_wrapper():
    script = PROJECT_ROOT / "scripts" / "edit_video.py"
    assert len(script.read_text(encoding="utf-8").splitlines()) < 80
