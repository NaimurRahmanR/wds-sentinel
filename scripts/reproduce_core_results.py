"""Re-run the core experiment/validation scripts and capture console logs.

This wrapper does not change any scientific setting. It exists so a fresh
checkout can regenerate the principal tables/figures/traces and retain the
stdout that the original scripts historically printed.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "raw"
OUT.mkdir(parents=True, exist_ok=True)

SCRIPTS = [
    "run_reliability_experiment.py",
    "validate_agents_vs_frozen.py",
    "coverage_aware_evaluation.py",
    "run_hazard_integration_scenarios.py",
    "run_final_2019_evaluation.py",
    "run_linked_extreme_weather_validation.py",
]


def run_one(script_name: str) -> None:
    script = ROOT / "scripts" / script_name
    log = OUT / f"{script.stem}.log"
    env = dict(os.environ)
    env.setdefault("PYTHONHASHSEED", "0")
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log.write_text(proc.stdout)
    if proc.returncode != 0:
        raise SystemExit(f"{script_name} failed with exit code {proc.returncode}; see {log}")
    print(f"OK  {script_name} -> {log.relative_to(ROOT)}")


if __name__ == "__main__":
    for name in SCRIPTS:
        run_one(name)
