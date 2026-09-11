"""Audit 2018/2019 onset structure directly from bundled leakage CSVs."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from wds_sentinel.data.battledim import derive_onsets, load_scada_csv  # noqa: E402

DATA = ROOT / "data" / "raw"

for year in (2018, 2019):
    leak = load_scada_csv(DATA / f"{year}_Leakages.csv")
    onsets, carryovers = derive_onsets(leak)
    print(f"\n=== {year} ===")
    print(f"new onsets: {len(onsets)}")
    print(f"carryovers: {sorted(carryovers)}")
    for pipe, timestamp in sorted(onsets.items(), key=lambda kv: kv[1]):
        print(f"  {timestamp}  {pipe}")
