from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load():
    spec = importlib.util.spec_from_file_location(
        "start_demo_under_test", ROOT / "scripts" / "start_demo.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_plan_points_at_the_skin_lesion_models():
    demo = _load()
    plan = demo.build_launch_plan(demo.parse_args([]))
    assert plan["env"]["CRYPTOVIT_CONFIG"] == "dermamnist"
    assert plan["env"]["CRYPTOVIT_EXPERIMENTS"] == "experiments/dermamnist"
    assert plan["env"]["CRYPTOVIT_DEVICE"] == "cpu"
    assert plan["env"]["CRYPTOVIT_VARIANT"] == "D"
    assert "api.main:app" in plan["api"] and "8000" in plan["api"]
    assert "dashboard/app.py" in plan["ui"] and "8501" in plan["ui"]
    assert plan["ui_url"] == "http://localhost:8501"


def test_block_option_adds_the_block_config_and_directory():
    demo = _load()
    plan = demo.build_launch_plan(demo.parse_args(["--block", "2"]))
    assert plan["env"]["CRYPTOVIT_CONFIG"] == "dermamnist,block2"
    assert plan["env"]["CRYPTOVIT_EXPERIMENTS"] == "experiments/dermamnist_block2"


def test_explicit_experiments_directory_wins():
    demo = _load()
    plan = demo.build_launch_plan(
        demo.parse_args(["--block", "2", "--experiments", "experiments/custom"])
    )
    assert plan["env"]["CRYPTOVIT_EXPERIMENTS"] == "experiments/custom"


def test_invalid_block_size_is_rejected():
    demo = _load()
    with pytest.raises(SystemExit):
        demo.build_launch_plan(demo.parse_args(["--block", "0"]))


def test_dry_run_starts_nothing(capsys):
    demo = _load()
    assert demo.main(["--dry-run", "--block", "2", "--device", "cuda"]) == 0
    out = capsys.readouterr().out
    assert "CRYPTOVIT_CONFIG=dermamnist,block2" in out
    assert "CRYPTOVIT_DEVICE=cuda" in out
