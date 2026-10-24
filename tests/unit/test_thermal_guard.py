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


def test_paced_pause_follows_the_duty_cycle():
    pause = _load().paced_pause
    # duty 0.7 on a 60 s cycle: run 0-42 s, paused 42-60 s, then repeat
    assert [pause(t, 0.7, 60) for t in (0, 41, 41.9)] == [False, False, False]
    assert [pause(t, 0.7, 60) for t in (42, 50, 59.9)] == [True, True, True]
    assert pause(60, 0.7, 60) is False and pause(102, 0.7, 60) is True


def test_full_duty_never_pauses():
    pause = _load().paced_pause
    assert not any(pause(t, 1.0, 60) for t in range(0, 300, 7))


def test_pause_share_matches_the_duty():
    pause = _load().paced_pause
    paused = sum(pause(t / 10, 0.7, 60) for t in range(0, 6000))
    assert paused / 6000 == pytest.approx(0.3, abs=0.01)


@pytest.mark.parametrize("duty, cycle", [(0.0, 60), (-0.5, 60), (1.5, 60), (0.7, 0), (0.7, -1)])
def test_invalid_pacing_arguments(duty, cycle):
    if duty >= 1.0 and cycle > 0:
        assert _load().paced_pause(1.0, duty, cycle) is False   # >= 1 means no pacing
    else:
        with pytest.raises(ValueError):
            _load().paced_pause(1.0, duty, cycle)


@pytest.mark.skipif(sys.platform != "win32", reason="uses Windows process suspension")
def test_pacing_really_freezes_the_job_for_part_of_each_cycle(tmp_path, monkeypatch):
    import threading

    guard = _load()
    monkeypatch.setattr(guard, "gpu_temperature", lambda: 50.0)   # never "hot"
    beat = tmp_path / "beat.txt"
    code = (
        "import time\n"
        f"f = open(r'{beat}', 'a', buffering=1)\n"
        "while True:\n"
        "    f.write(repr(time.time()) + '\\n'); time.sleep(0.05)\n"
    )
    proc = subprocess.Popen([sys.executable, "-c", code])
    result = {}
    thread = threading.Thread(
        target=lambda: result.update(rc=guard.run(proc.pid, 80, 70, 0.1, duty=0.5, cycle=2.0)),
        daemon=True,
    )
    try:
        time.sleep(1.0)
        thread.start()
        time.sleep(7.0)
    finally:
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/F"], capture_output=True)
        thread.join(timeout=15)

    stamps = [float(x) for x in beat.read_text().split()]
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    assert len(stamps) > 40
    assert max(gaps) >= 0.7          # frozen for about 1 s of each 2 s cycle
    assert result.get("rc") == 0 and not thread.is_alive()


@pytest.mark.skipif(sys.platform != "win32", reason="uses Windows process APIs")
def test_process_alive_reports_exit():
    guard = _load()
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    assert guard.process_alive(proc.pid) is False
