from __future__ import annotations

import importlib.util
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
    assert actions == [
        None,
        None,
        None,
        "suspend",
        None,
        None,
        None,
        None,
        "resume",
        None,
        None,
        "suspend",
    ]


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


def test_paced_pause_follows_the_duty_cycle():
    pause = _load().paced_pause
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
        assert _load().paced_pause(1.0, duty, cycle) is False
    else:
        with pytest.raises(ValueError):
            _load().paced_pause(1.0, duty, cycle)
