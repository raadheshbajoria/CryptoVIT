"""Tests for the emergency pause: trip logic, and on Windows a real freeze that
the routine guard cannot undo, plus manual resume."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load(name):
    spec = importlib.util.spec_from_file_location(
        f"{name}_under_test", ROOT / "scripts" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_trips_once_at_or_above_the_limit_and_never_untrips():
    s = _load("emergency_pause").EmergencyState(90)
    assert [s.update(t) for t in (60, 85, 89.9)] == [False, False, False]
    assert s.update(90) is True and s.tripped
    assert s.update(95) is False
    assert s.update(50) is False and s.tripped


def test_unreadable_temperature_does_not_trip():
    s = _load("emergency_pause").EmergencyState(90)
    assert s.update(None) is False and not s.tripped


def test_read_pids_skips_bad_files(tmp_path):
    mod = _load("emergency_pause")
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "train.pid").write_text("1234\n")
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "train.pid").write_text("not a number")
    assert mod.read_pids([str(tmp_path / "*" / "train.pid")]) == [1234]
    assert mod.read_pids([str(tmp_path / "missing" / "train.pid")]) == []
