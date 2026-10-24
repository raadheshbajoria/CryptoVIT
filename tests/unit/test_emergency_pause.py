"""Tests for the emergency pause: trip logic, and on Windows a real freeze that
the routine guard cannot undo, plus manual resume."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load(name):
    spec = importlib.util.spec_from_file_location(
        f"{name}_under_test", ROOT / "scripts" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(sys.platform != "win32", reason="uses Windows process suspension")
def test_a_guard_started_during_an_emergency_pause_does_not_unfreeze_the_job(tmp_path, monkeypatch):
    guard = _load("thermal_guard")
    monkeypatch.setattr(guard, "gpu_temperature", lambda: 50.0)
    beat = tmp_path / "beat.txt"
    code = (
        "import time\n"
        f"f = open(r'{beat}', 'a', buffering=1)\n"
        "while True:\n"
        "    f.write(repr(time.time()) + '\\n'); time.sleep(0.05)\n"
    )
    proc = subprocess.Popen([sys.executable, "-c", code])
    latch = tmp_path / "LATCH.txt"
    try:
        time.sleep(1.0)
        guard.suspend_tree(proc.pid)                  # the emergency freeze
        latch.write_text("EMERGENCY PAUSE")
        time.sleep(0.3)
        size = beat.stat().st_size
        thread = threading.Thread(
            target=lambda: guard.run(proc.pid, 80, 70, 0.1, duty=1.0, cycle=60, latch=latch),
            daemon=True,
        )
        thread.start()
        time.sleep(1.5)
        assert beat.stat().st_size == size, "guard start-up must not lift an emergency pause"
    finally:
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
        thread.join(timeout=15)


def test_trips_once_at_or_above_the_limit_and_never_untrips():
    s = _load("emergency_pause").EmergencyState(90)
    assert [s.update(t) for t in (60, 85, 89.9)] == [False, False, False]
    assert s.update(90) is True and s.tripped
    assert s.update(95) is False          # already tripped
    assert s.update(50) is False and s.tripped   # cooling does not clear it


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


@pytest.mark.skipif(sys.platform != "win32", reason="uses Windows process suspension")
def test_emergency_freeze_survives_the_guards_resume_and_manual_resume_works(tmp_path):
    em, guard = _load("emergency_pause"), _load("thermal_guard")
    beat = tmp_path / "beat.txt"
    code = (
        "import time\n"
        f"f = open(r'{beat}', 'a', buffering=1)\n"
        "while True:\n"
        "    f.write(repr(time.time()) + '\\n'); time.sleep(0.05)\n"
    )
    proc = subprocess.Popen([sys.executable, "-c", code])
    pid_dir = tmp_path / "run" / "ablation"
    pid_dir.mkdir(parents=True)
    (pid_dir / "train.pid").write_text(str(proc.pid))
    latch = tmp_path / "LATCH.txt"
    pid_files = [str(tmp_path / "*" / "ablation" / "train.pid")]

    temps = iter([50.0, 60.0, 91.0])           # the third reading trips it
    last = [91.0]

    def fake_temp():
        last[0] = next(temps, last[0])
        return last[0]

    watcher = threading.Thread(
        target=lambda: em.watch(90, 0.1, pid_files, latch, temperature=fake_temp, max_hours=0.0014),
        daemon=True,
    )
    try:
        time.sleep(1.0)
        watcher.start()
        time.sleep(1.5)
        assert latch.exists() and "EMERGENCY PAUSE" in latch.read_text()

        # the routine guard pauses and resumes the job (its own freeze/unfreeze
        # pair): the emergency freeze underneath must stay in force
        guard.suspend_tree(proc.pid)
        guard.resume_tree(proc.pid, set(guard.process_tree(proc.pid)))
        size = beat.stat().st_size
        time.sleep(1.0)
        assert beat.stat().st_size == size, "job must stay frozen after the guard's resume"

        # manual resume brings it back and clears the latch
        em.resume(pid_files, latch)
        time.sleep(1.0)
        assert beat.stat().st_size > size
        assert not latch.exists()
    finally:
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
        watcher.join(timeout=15)
