"""
A modest, deliberately simple hazard component: a training-calibrated,
multi-window precipitation-extreme indicator — the current 1-day/3-day/
7-day cumulative precipitation is compared against percentile cutoffs
computed once from a training period, and flagged "elevated" if any
window exceeds its own cutoff. This is a calibrated statistical
convention specific to this project (not an invented domain threshold,
and not a named external standard index), and not a fitted ML classifier
— chosen deliberately over a more complex model per "prefer a simple
interpretable baseline over unnecessary model complexity".

Underlying observations: Taiwan Central Weather Administration (CWA) /
CODIS station network. Processed copy used here: the Raingel/
historical_weather GitHub rebuild (see data/PROVENANCE.md for licence
status, checksums, and the sentinel-value data-quality note). No
geographic or causal relationship between this hazard source and
L-Town/BattLeDIM is claimed.

"Training" here means computing frozen percentile cutoffs from a training
period only, exactly the same pattern already used for the WDS
reliability cutoffs (fit once, applied unchanged to validation) — not
fitting a predictive model with learned weights.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

PERCENTILE = 95.0
ANTECEDENT_WINDOWS_DAYS = (1, 3, 7)  # "1" = the day itself, 3-day and 7-day trailing cumulative


def antecedent_precipitation(precip: pd.Series, windows: tuple[int, ...] = ANTECEDENT_WINDOWS_DAYS) -> pd.DataFrame:
    """Causal (backward-only) trailing cumulative precipitation over each
    window in `windows`, in days. pandas' default .rolling() is backward-
    looking (window ending at, and including, the current row), so no
    value at row t depends on any row after t — checked directly in
    tests/test_hazard_no_future_leakage.py."""
    out = {}
    for w in windows:
        out[f"precip_sum_{w}d"] = precip.rolling(window=w, min_periods=w).sum()
    return pd.DataFrame(out, index=precip.index)


@dataclass(frozen=True)
class HazardThresholds:
    percentile: float
    cutoffs: dict[str, float]  # column name (from antecedent_precipitation) -> frozen cutoff value


def fit_hazard_thresholds(train_antecedent: pd.DataFrame, percentile: float = PERCENTILE) -> HazardThresholds:
    """Computed ONCE on a training period only, frozen; never recomputed
    per-scenario or re-tuned against downstream results."""
    cutoffs = {col: float(train_antecedent[col].dropna().quantile(percentile / 100.0))
               for col in train_antecedent.columns}
    return HazardThresholds(percentile=percentile, cutoffs=cutoffs)


def hazard_elevated_flag(antecedent: pd.DataFrame, thresholds: HazardThresholds) -> pd.Series:
    """Elevated iff ANY antecedent window's value exceeds its own frozen
    training-period cutoff. NaN (missing precipitation input) propagates
    to NaN here — never silently treated as "not elevated"."""
    exceeds = pd.DataFrame(
        {col: antecedent[col] > cutoff for col, cutoff in thresholds.cutoffs.items()}
    )
    any_exceeds = exceeds.any(axis=1)
    # if every underlying column is NaN for a row, the flag is undefined, not False
    all_nan = antecedent[list(thresholds.cutoffs.keys())].isna().all(axis=1)
    return any_exceeds.mask(all_nan, other=pd.NA)
