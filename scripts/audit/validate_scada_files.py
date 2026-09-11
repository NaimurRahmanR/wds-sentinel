"""Validate the bundled BattLeDIM/L-Town multimodal CSVs in-place.

No model training is performed. This script checks timestamp grids,
missingness, duplicate timestamps, schema consistency, and cross-modality
alignment for both 2018 and 2019.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from wds_sentinel.data.battledim import load_scada_csv  # noqa: E402

DATA = ROOT / "data" / "raw"


def report_frame(name: str, df: pd.DataFrame) -> None:
    print(f"\n--- {name} ---")
    print(f"Shape: {df.shape}")
    print(f"Timestamp range: {df.index.min()} -> {df.index.max()}")
    freq = df.index.to_series().diff().dropna().value_counts()
    print(f"Timestep deltas (top 5):\n{freq.head(5)}")
    full_range = pd.date_range(df.index.min(), df.index.max(), freq="5min")
    print(f"Missing timestamps vs complete 5-min grid: {len(full_range.difference(df.index))}")
    print(f"Duplicate timestamps: {df.index.duplicated().sum()}")
    nan_counts = df.isna().sum()
    print(f"Columns with any NaN: {(nan_counts > 0).sum()} / {df.shape[1]}")


streams: dict[tuple[int, str], pd.DataFrame] = {}
for year in (2018, 2019):
    names = {
        "pressure": f"{year}_SCADA_Pressures.csv",
        "flow": f"{year}_SCADA_Flows.csv",
        "level": f"{year}_SCADA_Levels.csv",
        "leakage": f"{year}_Leakages.csv",
    }
    for kind, filename in names.items():
        df = load_scada_csv(DATA / filename)
        streams[(year, kind)] = df
        report_frame(filename, df)

for year in (2018, 2019):
    p = streams[(year, "pressure")]
    f = streams[(year, "flow")]
    lv = streams[(year, "level")]
    leak = streams[(year, "leakage")]
    print(f"\n{year}: pressure/flow/level/leakage timestamps identical? "
          f"{p.index.equals(f.index) and p.index.equals(lv.index) and p.index.equals(leak.index)}")

print(f"\nPressure schema identical 2018 vs 2019? "
      f"{list(streams[(2018, 'pressure')].columns) == list(streams[(2019, 'pressure')].columns)}")
print(f"Flow schema identical 2018 vs 2019? "
      f"{list(streams[(2018, 'flow')].columns) == list(streams[(2019, 'flow')].columns)}")
print(f"Level schema identical 2018 vs 2019? "
      f"{list(streams[(2018, 'level')].columns) == list(streams[(2019, 'level')].columns)}")
