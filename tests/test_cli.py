from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

from smart_video_editor import cli


def parse_run(*arguments: str) -> argparse.Namespace:
    parser = cli.build_parser()
    return parser.parse_args(["run", *arguments])


def test_initialize_workspace_creates_expected_directories(tmp_path: Path) -> None:
    workspace = tmp_path / "project"

    cli.initialize_workspace(workspace)

    assert (workspace / "raw").is_dir()
    assert (workspace / "artifacts").is_dir()
    assert (workspace / "edited").is_dir()


def test_pipeline_builds_all_default_stages() -> None:
    args = parse_run("--video-name", "demo.mp4", "--language", "pl")

    stages = cli.pipeline_stages(args)

    assert [stage[0] for stage in stages] == ["transcribe", "analyze", "edit"]
    assert "demo.mp4" in stages[0][2]
    assert "--quality-language" in stages[2][2]


def test_pipeline_can_resume_from_edit() -> None:
    args = parse_run("--from-stage", "edit", "--skip-quality-check")

    stages = cli.pipeline_stages(args)

    assert [stage[0] for stage in stages] == ["edit"]
    assert "--skip-quality-check" in stages[0][2]


def test_pipeline_stops_after_a_failed_stage(tmp_path: Path) -> None:
    args = parse_run()

    with patch.object(cli, "initialize_workspace"), patch.object(
        cli, "run_script", side_effect=[0, 7]
    ) as run_script:
        returncode = cli.run_pipeline(args, tmp_path)

    assert returncode == 7
    assert run_script.call_count == 2


def test_auto_refine_requires_a_quality_report() -> None:
    args = parse_run("--auto-refine", "--language", "pl")

    stages = cli.pipeline_stages(args)

    assert stages[-1][0] == "refine"
    assert "--quality-check-strict" in stages[-2][2]
    assert "--quality-language" in stages[-1][2]


def test_stage_arguments_are_forwarded_unchanged(tmp_path: Path) -> None:
    with patch.object(cli, "run_script", return_value=0) as run_script:
        returncode = cli.main(
            [
                "--workspace",
                str(tmp_path),
                "transcribe",
                "--language",
                "pl",
                "--diarize",
            ]
        )

    assert returncode == 0
    run_script.assert_called_once_with(
        "transcribe_video.py",
        ["--language", "pl", "--diarize"],
        tmp_path.resolve(),
    )
