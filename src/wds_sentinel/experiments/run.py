"""Small utilities for auditable experiment execution."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from wds_sentinel.experiments.ids import make_experiment_id
from wds_sentinel.utils.logging import configure_logging
from wds_sentinel.utils.seeding import seed_everything

RESULTS_RAW_DIR = Path("results/raw")


def prepare_run(config_text: str, seed: int, log_level: str = "INFO") -> Path:
    """Create a unique run directory from config text and seed."""
    configure_logging(log_level)
    seed_everything(seed)
    experiment_id = make_experiment_id(config_text)
    run_dir = RESULTS_RAW_DIR / experiment_id
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one project experiment script and capture stdout")
    parser.add_argument("script", type=Path, help="experiment script to execute")
    parser.add_argument("--log", type=Path, default=None, help="output log path")
    args = parser.parse_args()

    script = args.script.resolve()
    if not script.exists():
        raise SystemExit(f"script not found: {script}")
    log = args.log or (RESULTS_RAW_DIR / f"{script.stem}.log")
    log.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [sys.executable, str(script)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log.write_text(proc.stdout)
    print(proc.stdout, end="")
    if proc.returncode:
        raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
