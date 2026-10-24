#!/usr/bin/env python
"""
scripts/emergency_pause.py
===========================
Hard safety net: if the GPU ever reaches the limit (default 90 C), freeze every
guarded training job and keep it frozen until a person resumes it.

    python scripts/emergency_pause.py --watch      # run the watchdog
    python scripts/emergency_pause.py --resume     # undo a pause by hand

It is independent of ``thermal_guard.py`` (the routine 80/70 C pacing). Freezing
uses ``NtSuspendProcess``, whose suspend counts stack, so the guard's own
resume cannot unfreeze a job this watchdog has frozen. Jobs are found through
``train.pid`` files (re-read every cycle, so a job started later is frozen too
while the pause is active). On a trigger a latch file is written; nothing
resumes automatically.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import logging
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

HERE = Path(__file__).resolve().parent
logger = logging.getLogger("emergency_pause")


def _load_guard():
    spec = importlib.util.spec_from_file_location("thermal_guard_lib", HERE / "thermal_guard.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_pids(pid_files: List[str]) -> List[int]:
    """PIDs written in the given files / globs (unreadable entries are skipped)."""
    pids = []
    for pattern in pid_files:
        for path in glob.glob(pattern):
            try:
                pids.append(int(Path(path).read_text().strip()))
            except (OSError, ValueError):
                continue
    return pids


class EmergencyState:
    """Decision logic: trip once at/above the limit; never un-trip by itself."""

    def __init__(self, limit: float = 90.0):
        self.limit = limit
        self.tripped = False

    def update(self, temperature: Optional[float]) -> bool:
        """True exactly when this reading trips the pause for the first time."""
        if not self.tripped and temperature is not None and temperature >= self.limit:
            self.tripped = True
            return True
        return False


def freeze_new_trees(guard, pids: List[int], already: Dict[int, Set[int]]) -> int:
    """Freeze the process tree of every alive PID not frozen yet; returns the count."""
    newly = 0
    for pid in pids:
        if pid in already or not guard.process_alive(pid):
            continue
        already[pid] = guard.suspend_tree(pid)
        newly += 1
        logger.info("Frozen process tree of PID %d (%d process(es)).", pid, len(already[pid]))
    return newly


def watch(limit: float, interval: float, pid_files: List[str], latch: Path,
          temperature: Callable[[], Optional[float]] = None, max_hours: float = 12.0) -> int:
    guard = _load_guard()
    temperature = temperature or guard.gpu_temperature
    state = EmergencyState(limit)
    frozen: Dict[int, Set[int]] = {}
    end = time.time() + max_hours * 3600
    logger.info("Watching the GPU: everything pauses at >= %.0f C and stays paused. Latch: %s",
                limit, latch)

    while time.time() < end:
        temp = temperature()
        if state.update(temp):
            latch.parent.mkdir(parents=True, exist_ok=True)
            latch.write_text(f"EMERGENCY PAUSE at {time.strftime('%Y-%m-%d %H:%M:%S')}: GPU {temp:.0f} C\n"
                             "Everything is frozen. Resume with: python scripts/emergency_pause.py --resume\n")
            logger.error("GPU at %.0f C (limit %.0f): EMERGENCY PAUSE.", temp, limit)
        if state.tripped:
            # keep freezing anything that appears later (e.g. a job started after the trip)
            freeze_new_trees(guard, read_pids(pid_files), frozen)
        time.sleep(interval)
    logger.info("Watchdog time limit reached; exiting%s.",
                " (the pause is still active)" if state.tripped else "")
    return 0


def resume(pid_files: List[str], latch: Path) -> int:
    """Undo an emergency pause: unfreeze the guarded jobs and remove the latch."""
    guard = _load_guard()
    resumed = 0
    for pid in read_pids(pid_files):
        if guard.process_alive(pid):
            resumed += len(guard.resume_tree(pid, set()))
    if latch.exists():
        latch.unlink()
    logger.info("Resumed %d process(es); latch removed.", resumed)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Freeze all guarded jobs if the GPU gets too hot.")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--watch", action="store_true")
    mode.add_argument("--resume", action="store_true")
    p.add_argument("--limit", type=float, default=90.0)
    p.add_argument("--interval", type=float, default=2.0)
    p.add_argument("--pid-file", action="append", default=None,
                   help="train.pid file or glob (repeatable)")
    p.add_argument("--latch", default="experiments/EMERGENCY_PAUSE.txt")
    p.add_argument("--log", default="experiments/emergency_pause.log")
    p.add_argument("--max-hours", type=float, default=12.0)
    args = p.parse_args(argv)

    pid_files = args.pid_file or ["experiments/*/ablation/train.pid"]
    handlers = [logging.StreamHandler(sys.stdout)]
    if args.log:
        Path(args.log).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(args.log, encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, handlers=handlers,
                        format="%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    latch = Path(args.latch)
    if args.resume:
        return resume(pid_files, latch)
    return watch(args.limit, args.interval, pid_files, latch, max_hours=args.max_hours)


if __name__ == "__main__":
    raise SystemExit(main())
