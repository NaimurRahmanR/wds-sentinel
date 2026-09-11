"""
Five controlled, reproducible degradation conditions, each a pure function
of (data, fixed seed/config) — same seed always produces the identical
degraded output (tested directly in tests/test_degradation.py). None of
these functions ever see or take a ground-truth label; they only
transform sensor readings.

Fixed a priori and never re-tuned against A/B/C's results:
  - SEED = 20260910 (arbitrary but fixed and documented, chosen once)
  - missing subset fraction = 0.30 of sensors, entire validation window
  - additive noise = N(0, 0.10 * that sensor's own std, computed from
    the clean 2018 TRAINING period only — never from validation/test)
  - conflicting evidence = a fixed +3-training-std offset applied to the
    flow group only, for the whole window
  - dropout = each sensor independently missing with probability 0.05 at
    each timestep (i.i.d. Bernoulli, fixed seed)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SEED = 20260910
MISSING_SUBSET_FRACTION = 0.30
NOISE_STD_FRACTION = 0.10
CONFLICT_OFFSET_STDS = 3.0
DROPOUT_PROB = 0.05


@dataclass
class DegradedStreams:
    pressure: pd.DataFrame
    flow: pd.DataFrame
    level: pd.DataFrame
    condition: str


def clean(pressure: pd.DataFrame, flow: pd.DataFrame, level: pd.DataFrame) -> DegradedStreams:
    return DegradedStreams(pressure.copy(), flow.copy(), level.copy(), "clean")


def missing_sensor_subset(
    pressure: pd.DataFrame, flow: pd.DataFrame, level: pd.DataFrame, seed: int = SEED
) -> DegradedStreams:
    """A fixed, randomly-chosen 30% of ALL sensors (across all three
    streams) go entirely missing for the whole window — e.g. a subset of
    sensors taken offline for the period."""
    rng = np.random.default_rng(seed)
    p, f, lv = pressure.copy(), flow.copy(), level.copy()
    all_cols = [("p", c) for c in p.columns] + [("f", c) for c in f.columns] + [("l", c) for c in lv.columns]
    n_missing = int(round(MISSING_SUBSET_FRACTION * len(all_cols)))
    chosen = rng.choice(len(all_cols), size=n_missing, replace=False)
    for idx in chosen:
        stream, col = all_cols[idx]
        {"p": p, "f": f, "l": lv}[stream][col] = np.nan
    return DegradedStreams(p, f, lv, "missing_sensor_subset")


def additive_noise(
    pressure: pd.DataFrame,
    flow: pd.DataFrame,
    level: pd.DataFrame,
    train_stds: dict[str, pd.Series],
    seed: int = SEED,
) -> DegradedStreams:
    """Gaussian noise added to every reading, scaled to 10% of that
    sensor's OWN standard deviation as measured on the clean training
    period (train_stds), never on the data being degraded itself."""
    rng = np.random.default_rng(seed)
    p, f, lv = pressure.copy(), flow.copy(), level.copy()
    for name, df in [("pressure", p), ("flow", f), ("level", lv)]:
        for col in df.columns:
            noise = rng.normal(0, NOISE_STD_FRACTION * train_stds[name][col], size=len(df))
            df[col] = df[col] + noise
    return DegradedStreams(p, f, lv, "additive_noise")


def conflicting_evidence(
    pressure: pd.DataFrame,
    flow: pd.DataFrame,
    level: pd.DataFrame,
    train_stds: dict[str, pd.Series],
) -> DegradedStreams:
    """The flow group is shifted by a fixed +3-training-std offset for the
    whole window, while pressure/level are untouched — one modality now
    tells a systematically different story than the others, with no
    randomness (fully deterministic, no seed needed)."""
    p, f, lv = pressure.copy(), flow.copy(), level.copy()
    for col in f.columns:
        f[col] = f[col] + CONFLICT_OFFSET_STDS * train_stds["flow"][col]
    return DegradedStreams(p, f, lv, "conflicting_evidence")


def sensor_dropout(
    pressure: pd.DataFrame, flow: pd.DataFrame, level: pd.DataFrame, seed: int = SEED
) -> DegradedStreams:
    """Each sensor, at each timestep, independently missing with a fixed
    small probability — transient/intermittent, unlike the permanent
    subset in missing_sensor_subset."""
    rng = np.random.default_rng(seed)
    p, f, lv = pressure.copy(), flow.copy(), level.copy()
    for df in (p, f, lv):
        mask = rng.random(df.shape) < DROPOUT_PROB
        df.mask(mask, other=np.nan, inplace=True)
    return DegradedStreams(p, f, lv, "sensor_dropout")


CONDITIONS = ["clean", "missing_sensor_subset", "additive_noise", "conflicting_evidence", "sensor_dropout"]


def apply_condition(
    name: str, pressure: pd.DataFrame, flow: pd.DataFrame, level: pd.DataFrame,
    train_stds: dict[str, pd.Series] | None = None,
) -> DegradedStreams:
    if name == "clean":
        return clean(pressure, flow, level)
    if name == "missing_sensor_subset":
        return missing_sensor_subset(pressure, flow, level)
    if name == "additive_noise":
        return additive_noise(pressure, flow, level, train_stds)
    if name == "conflicting_evidence":
        return conflicting_evidence(pressure, flow, level, train_stds)
    if name == "sensor_dropout":
        return sensor_dropout(pressure, flow, level)
    raise ValueError(f"unknown condition: {name}")
