#!/usr/bin/env python
"""
scripts/thermal_guard.py
=========================
Pause a running job when the GPU gets too hot and resume it once it has cooled.

    python scripts/thermal_guard.py --pid 12345 --high 80 --low 70

When the GPU temperature reaches ``--high`` the job (the process and all of its
child processes, e.g. data-loader workers) is *suspended*: it freezes in place,
loses no progress and stops using the GPU, so the card cools. When the
temperature falls to ``--low`` or below, the job is resumed. The guard exits
when the job exits. Windows only (uses ``NtSuspendProcess``); the GPU
temperature comes from ``nvidia-smi``.

Every pause and resume is written to the log with the temperature and time.
"""

from __future__ import annotations

import argparse
import ctypes
import logging
import subprocess
import sys
import time
from ctypes import wintypes
from pathlib import Path
from typing import List, Optional, Set

logger = logging.getLogger("thermal_guard")


# ---------------------------------------------------------------------------
# Decision logic (pure, unit-tested)
# ---------------------------------------------------------------------------

class ThermalGovernor:
    """Hysteresis controller: suspend at ``high``, resume at ``low`` or below."""

    def __init__(self, high: float = 80.0, low: float = 70.0):
        if low >= high:
            raise ValueError(f"low ({low}) must be below high ({high}).")
        self.high, self.low = high, low
        self.suspended = False

    def update(self, temperature: float) -> Optional[str]:
        """Return ``"suspend"``, ``"resume"`` or ``None`` for this reading."""
        if not self.suspended and temperature >= self.high:
            self.suspended = True
            return "suspend"
        if self.suspended and temperature <= self.low:
            self.suspended = False
            return "resume"
        return None


# ---------------------------------------------------------------------------
# Windows process helpers
# ---------------------------------------------------------------------------

_TH32CS_SNAPPROCESS = 0x2
_PROCESS_SUSPEND_RESUME = 0x0800


class _PROCESSENTRY32(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
        ("th32ProcessID", wintypes.DWORD), ("th32DefaultHeapID", ctypes.c_void_p),
        ("th32ModuleID", wintypes.DWORD), ("cntThreads", wintypes.DWORD),
        ("th32ParentProcessID", wintypes.DWORD), ("pcPriClassBase", ctypes.c_long),
        ("dwFlags", wintypes.DWORD), ("szExeFile", ctypes.c_char * 260),
    ]


def process_tree(root_pid: int) -> List[int]:
    """``root_pid`` plus all of its descendants (Windows toolhelp snapshot)."""
    kernel32 = ctypes.windll.kernel32
    snapshot = kernel32.CreateToolhelp32Snapshot(_TH32CS_SNAPPROCESS, 0)
    children = {}
    entry = _PROCESSENTRY32()
    entry.dwSize = ctypes.sizeof(_PROCESSENTRY32)
    ok = kernel32.Process32First(snapshot, ctypes.byref(entry))
    while ok:
        children.setdefault(entry.th32ParentProcessID, []).append(entry.th32ProcessID)
        ok = kernel32.Process32Next(snapshot, ctypes.byref(entry))
    kernel32.CloseHandle(snapshot)

    found, queue = [], [root_pid]
    while queue:
        pid = queue.pop()
        if pid in found:
            continue
        found.append(pid)
        queue.extend(children.get(pid, []))
    return found


def _nt_call(pid: int, function: str) -> bool:
    kernel32, ntdll = ctypes.windll.kernel32, ctypes.windll.ntdll
    handle = kernel32.OpenProcess(_PROCESS_SUSPEND_RESUME, False, pid)
    if not handle:
        return False
    try:
        return getattr(ntdll, function)(handle) == 0
    finally:
        kernel32.CloseHandle(handle)


def suspend_tree(root_pid: int) -> Set[int]:
    """Suspend the whole process tree; returns the PIDs that were suspended."""
    pids = {pid for pid in process_tree(root_pid) if _nt_call(pid, "NtSuspendProcess")}
    return pids


def resume_tree(root_pid: int, remembered: Set[int]) -> Set[int]:
    """Resume everything suspended earlier plus any current descendants."""
    pids = set(remembered) | set(process_tree(root_pid))
    return {pid for pid in pids if _nt_call(pid, "NtResumeProcess")}


def process_alive(pid: int) -> bool:
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(0x1000, False, pid)   # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return False
    code = wintypes.DWORD()
    kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
    kernel32.CloseHandle(handle)
    return code.value == 259   # STILL_ACTIVE


def gpu_temperature() -> Optional[float]:
    """Hottest GPU temperature in Celsius, or None if nvidia-smi fails."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=True,
        ).stdout
        return max(float(line) for line in out.split() if line.strip())
    except (subprocess.SubprocessError, ValueError, OSError):
        return None


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(pid: int, high: float, low: float, interval: float) -> int:
    governor = ThermalGovernor(high, low)
    suspended_pids: Set[int] = set()
    paused_since = 0.0
    total_paused = 0.0
    pauses = 0
    logger.info("Watching PID %d: suspend at >= %.0f C, resume at <= %.0f C", pid, high, low)

    try:
        while process_alive(pid):
            temp = gpu_temperature()
            if temp is None:
                logger.warning("Could not read the GPU temperature; will retry.")
            else:
                action = governor.update(temp)
                if action == "suspend":
                    suspended_pids = suspend_tree(pid)
                    paused_since = time.time()
                    pauses += 1
                    logger.info("%.0f C reached: SUSPENDED %d process(es) (pause #%d)",
                                temp, len(suspended_pids), pauses)
                elif action == "resume":
                    resumed = resume_tree(pid, suspended_pids)
                    waited = time.time() - paused_since
                    total_paused += waited
                    logger.info("%.0f C: RESUMED %d process(es) after %.0f s paused "
                                "(total paused %.1f min)", temp, len(resumed), waited,
                                total_paused / 60)
            time.sleep(interval)
    except KeyboardInterrupt:
        pass
    finally:
        if governor.suspended:     # never leave the job frozen when the guard stops
            resume_tree(pid, suspended_pids)
            logger.info("Guard stopping: resumed the job.")
    logger.info("Job has ended (pauses: %d, total paused %.1f min). Guard exiting.",
                pauses, total_paused / 60)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Pause a job while the GPU is too hot.")
    p.add_argument("--pid", type=int, required=True, help="PID of the job to guard")
    p.add_argument("--high", type=float, default=80.0, help="suspend at this temperature (C)")
    p.add_argument("--low", type=float, default=70.0, help="resume at or below this (C)")
    p.add_argument("--interval", type=float, default=10.0, help="seconds between readings")
    p.add_argument("--log", default=None, help="also write the log to this file")
    args = p.parse_args(argv)

    handlers = [logging.StreamHandler(sys.stdout)]
    if args.log:
        Path(args.log).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(args.log, encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, handlers=handlers,
                        format="%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    return run(args.pid, args.high, args.low, args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
