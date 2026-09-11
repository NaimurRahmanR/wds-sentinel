"""Causal, pressure-only feature construction.

Every feature here uses pandas' default .rolling() behaviour, which is
backward-looking (window ending at, and including, the current row) — so
no feature at row t can depend on any row after t. This is the property
tests/test_temporal_leakage.py checks directly, not just asserts.
"""
from __future__ import annotations

import pandas as pd


def build_causal_features(pressure_df: pd.DataFrame, window_steps: int = 12) -> pd.DataFrame:
    """For each sensor: raw value at t, rolling mean over [t-window+1, t],
    rolling std over the same window. Rows without a full window (the
    first window_steps-1 rows of the whole series) are left as NaN and
    should be dropped by the caller — never imputed, since that would
    fabricate history that doesn't exist.

    NOTE: this is the feature set used by the (documented null-result)
    1-hour-ahead forecasting baseline. Kept as-is; build_change_features
    below is what the live detection target uses.
    """
    raw = pressure_df.add_suffix("_raw")
    roll = pressure_df.rolling(window=window_steps, min_periods=window_steps)
    mean = roll.mean().add_suffix(f"_mean{window_steps}")
    std = roll.std().add_suffix(f"_std{window_steps}")
    return pd.concat([raw, mean, std], axis=1)


def build_change_features(
    pressure_df: pd.DataFrame,
    diff_steps: tuple[int, ...] = (1, 3, 12),
    baseline_window: int = 24,
    z_outlier_cutoff: float = -2.0,
) -> pd.DataFrame:
    """Causal, change-based features for the early-onset-detection target.

    All of these use only pandas' default backward-looking .diff()/.rolling()
    — no feature at row t depends on any row after t (checked directly by
    tests/test_causal_features.py's future-perturbation test, which also
    covers this function).

    Per sensor:
      - raw value
      - diff over each window in diff_steps (5-min/15-min/1h changes, by
        default) — the "first differences" and "short-window changes" asked
        for; a longer baseline_window (default 24 steps = 2h) is used for
        the z-score reference specifically so the reference mean/std isn't
        immediately contaminated by the very onset we're trying to detect.
      - z-score relative to a `baseline_window`-step rolling mean/std
        (past-only)

    Network-wide, cross-sensor summaries (not per-sensor):
      - mean and min of the 1-step diff across all sensors
      - mean and min of the z-score across all sensors
      - count of sensors with z-score below z_outlier_cutoff (a standard
        statistical convention, e.g. ~2 std below the sensor's own recent
        baseline — not a WDS-domain operational threshold)

    Fixed a priori (not tuned by search): diff_steps, baseline_window, and
    z_outlier_cutoff are single, reasoned choices, not swept.
    """
    parts = [pressure_df.add_suffix("_raw")]

    diffs = {}
    for k in diff_steps:
        d = pressure_df.diff(k)
        diffs[k] = d
        parts.append(d.add_suffix(f"_diff{k}"))

    roll = pressure_df.rolling(window=baseline_window, min_periods=baseline_window)
    mean = roll.mean()
    std = roll.std()
    zero_std = std == 0  # a real, computed zero (not the warmup NaN period, where std is NaN, not 0)
    zscore = (pressure_df - mean) / std
    # A sensor with no local variability (e.g. PUMP_1 sitting at exactly 0
    # for hours while off) has nothing to normalise against — treat that as
    # "not currently anomalous relative to its own flat baseline" (z=0)
    # rather than a NaN/inf that would silently drop the row.
    zscore = zscore.mask(zero_std, 0.0)
    parts.append(zscore.add_suffix(f"_z{baseline_window}"))

    diff1 = diffs[diff_steps[0]]
    cross = pd.DataFrame(
        {
            "cross_mean_diff1": diff1.mean(axis=1),
            "cross_min_diff1": diff1.min(axis=1),
            "cross_mean_z": zscore.mean(axis=1),
            "cross_min_z": zscore.min(axis=1),
            "cross_count_below_z_cutoff": (zscore < z_outlier_cutoff).sum(axis=1),
        }
    )
    parts.append(cross)

    return pd.concat(parts, axis=1)


def build_seasonal_residual_features(
    df: pd.DataFrame, lag: pd.Timedelta = pd.Timedelta("7D"), rolling_window: int = 12
) -> pd.DataFrame:
    """Causal weekly-seasonal residual features: value(t) - value(t - lag),
    plus a rolling mean/std of that residual series (no z-score division
    here, deliberately — avoids re-introducing the zero-std hazard fixed
    in build_change_features, since a residual can legitimately sit at
    exactly 0 for a while, e.g. a pump that's off at the same hour on
    both the current and the reference week).

    Exact-lag by timestamp, not by an assumed step count: value(t - lag) is
    looked up by reindexing on the actual shifted timestamps, so this is
    correct even if the true sampling grid ever had a gap (it would simply
    produce NaN for that lookup, not a silently wrong value from a
    different time offset).

    Rows where t - lag falls before the series' own start (no 7-day-old
    value exists yet) are NaN and must be dropped by the caller, not
    imputed.
    """
    lagged = df.reindex(df.index - lag)
    lagged.index = df.index
    residual = df - lagged

    parts = [residual.add_suffix(f"_wresid_{lag.days}d")]
    roll = residual.rolling(window=rolling_window, min_periods=rolling_window)
    parts.append(roll.mean().add_suffix(f"_wresid_mean{rolling_window}"))
    parts.append(roll.std().add_suffix(f"_wresid_std{rolling_window}"))
    return pd.concat(parts, axis=1)
