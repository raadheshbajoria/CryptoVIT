#!/usr/bin/env python
"""
scripts/start_demo.py
======================
Start the API and the dashboard for the skin-lesion demo with one command.

    python scripts/start_demo.py                  # per-pixel scheme (default models)
    python scripts/start_demo.py --block 2        # models trained with the block2 scheme
    python scripts/start_demo.py --dry-run        # only print what would be started

The API runs on CPU by default so it does not compete with training for the
GPU; use ``--device cuda`` to change that. Press Ctrl+C to stop both.

Research prototype only. Not a medical device; not for clinical use.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from typing import Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Start the CryptoViT API and dashboard.")
    p.add_argument("--dataset-config", default="dermamnist",
                   help="dataset config in configs/ (default: dermamnist)")
    p.add_argument("--block", type=int, default=None, metavar="B",
                   help="use the block-wise scheme with B x B blocks (adds config 'block2' for B=2)")
    p.add_argument("--experiments", default=None,
                   help="experiments directory holding ablation/variant_X "
                        "(default: experiments/dermamnist, or experiments/dermamnist_block<B>)")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--variant", default="D", choices=["B", "C", "D"],
                   help="encrypted-image model used for the main prediction (default D)")
    p.add_argument("--api-port", type=int, default=8000)
    p.add_argument("--ui-port", type=int, default=8501)
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def build_launch_plan(args: argparse.Namespace) -> Dict:
    """Environment variables and the two commands to run (no side effects)."""
    configs: List[str] = [args.dataset_config]
    experiments = args.experiments or f"experiments/{args.dataset_config}"
    if args.block is not None:
        if args.block < 1:
            raise SystemExit("--block must be >= 1")
        configs.append(f"block{args.block}")
        if args.experiments is None:
            experiments = f"experiments/{args.dataset_config}_block{args.block}"

    env = {
        "CRYPTOVIT_CONFIG": ",".join(configs),
        "CRYPTOVIT_EXPERIMENTS": experiments,
        "CRYPTOVIT_DEVICE": args.device,
        "CRYPTOVIT_VARIANT": args.variant,
    }
    api = [sys.executable, "-m", "uvicorn", "api.main:app", "--port", str(args.api_port)]
    ui = [sys.executable, "-m", "streamlit", "run", "dashboard/app.py",
          "--server.port", str(args.ui_port), "--server.headless", "true",
          "--browser.gatherUsageStats", "false"]
    return {"env": env, "api": api, "ui": ui,
            "ui_url": f"http://localhost:{args.ui_port}",
            "api_url": f"http://127.0.0.1:{args.api_port}"}


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    plan = build_launch_plan(args)

    print("Environment:")
    for k, v in plan["env"].items():
        print(f"  {k}={v}")
    print("API      :", " ".join(plan["api"]))
    print("Dashboard:", " ".join(plan["ui"]))
    if args.dry_run:
        return 0

    exp_dir = PROJECT_ROOT / plan["env"]["CRYPTOVIT_EXPERIMENTS"] / "ablation"
    if not exp_dir.is_dir():
        print(f"\nNote: {exp_dir} does not exist yet; models will be untrained.")

    env = {**os.environ, **plan["env"]}
    procs = [subprocess.Popen(plan["api"], cwd=PROJECT_ROOT, env=env),
             subprocess.Popen(plan["ui"], cwd=PROJECT_ROOT, env=env)]
    print(f"\nDashboard: {plan['ui_url']}   (API: {plan['api_url']})   Ctrl+C to stop")
    if not args.no_browser:
        time.sleep(4)
        webbrowser.open(plan["ui_url"])
    try:
        while all(p.poll() is None for p in procs):
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
