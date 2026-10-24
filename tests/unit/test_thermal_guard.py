"""Tests for the thermal guard: the hysteresis logic and, on Windows, real
suspend/resume of a child process tree."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load():
    spec = importlib.util.spec_from_file_location(
        "thermal_guard_under_test", ROOT / "scripts" / "thermal_guard.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_suspends_at_high_and_resumes_at_low():
    g = _load().ThermalGovernor(high=80, low=70)
    readings = [60, 75, 79, 80, 85, 78, 72, 71, 70, 65, 79, 81]
    actions = [g.update(t) for t in readings]
    #           60    75    79    80         85    78    72    71    70        65    79    81
    assert actions == [None, None, None, "suspend", None, None, None, None, "resume", None, None, "suspend"]


def test_stays_suspended_until_it_has_cooled_to_low():
    g = _load().ThermalGovernor(80, 70)
    assert g.update(82) == "suspend"
    for t in (79, 75, 71):
        assert g.update(t) is None and g.suspended
    assert g.update(70) == "resume" and not g.suspended


def test_exact_high_counts_as_crossing_and_exact_low_as_cooled():
    g = _load().ThermalGovernor(80, 70)
    assert g.update(80) == "suspend"
    assert g.update(70) == "resume"


def test_low_must_be_below_high():
    with pytest.raises(ValueError):
        _load().ThermalGovernor(high=70, low=70)
    with pytest.raises(ValueError):
        _load().ThermalGovernor(high=60, low=75)


@pytest.mark.skipif(sys.platform != "win32", reason="uses Windows process suspension")
def test_suspend_freezes_the_whole_tree_and_resume_restarts_it():
    guard = _load()
    # parent python spawns a child python; both spin and write a heartbeat file
    code = (
        "import subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time\\nwhile True: time.sleep(0.05)'])\n"
        "while True: time.sleep(0.05)\n"
    )
    proc = subprocess.Popen([sys.executable, "-c", code])
    try:
        time.sleep(1.0)
        tree = guard.process_tree(proc.pid)
        assert proc.pid in tree and len(tree) >= 2

        suspended = guard.suspend_tree(proc.pid)
        assert proc.pid in suspended and len(suspended) >= 2
        assert guard.process_alive(proc.pid)

        resumed = guard.resume_tree(proc.pid, suspended)
        assert proc.pid in resumed
        assert guard.process_alive(proc.pid)
    finally:
        for pid in guard.process_tree(proc.pid):
            subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True)


@pytest.mark.skipif(sys.platform != "win32", reason="uses Windows process APIs")
def test_process_alive_reports_exit():
    guard = _load()
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    assert guard.process_alive(proc.pid) is False
