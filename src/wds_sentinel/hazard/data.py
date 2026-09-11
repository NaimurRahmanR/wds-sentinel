"""Loading for the independent hydro-meteorological evidence stream.

The bundled CSVs are a third-party rebuild of Taiwan Central Weather
Administration (CWA) / CODIS observations for station 466920 (Taipei),
2013-2018. They are not geographically or causally linked to the
BattLeDIM/L-Town WDS data. See data/PROVENANCE.md and
THIRD_PARTY_DATA_NOTICE.md.

The source rebuild explicitly documents CWA precipitation value ``-9.8``
as trace precipitation (<0.1 mm) and converts it to 0.09 mm. We reproduce
that documented conversion here instead of treating trace precipitation as
missing.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

TRACE_SENTINEL_VALUE = -9.8
TRACE_PRECIP_MM = 0.09


def load_daily_precipitation(paths: list[str | Path]) -> pd.Series:
    """Load and concatenate daily precipitation CSVs.

    Returns a sorted daily ``precip_mm`` series. The source-documented
    ``-9.8`` trace value is converted to 0.09 mm. Duplicate dates are an
    error rather than being silently collapsed.
    """
    frames = []
    for p in paths:
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        if "Precp" not in df.columns:
            raise ValueError(f"missing Precp column in {p}")
        frames.append(df[["Precp"]])

    combined = pd.concat(frames).sort_index()
    precip = pd.to_numeric(combined["Precp"], errors="coerce").copy()
    precip = precip.mask(precip == TRACE_SENTINEL_VALUE, TRACE_PRECIP_MM)
    if precip.index.duplicated().any():
        raise ValueError("duplicate dates in hazard source data — refusing to silently collapse them")
    return precip.rename("precip_mm")
