"""Tests required for the seasonal-residual pivot: exact lag, no future
information, identical row alignment, and no silent (unexplained) row loss
beyond the expected warmup period."""
import numpy as np
import pandas as pd

from wds_sentinel.prediction.features import build_seasonal_residual_features


def _index(n, freq="5min"):
    return pd.date_range("2020-01-01", periods=n, freq=freq)


def test_lag_is_exactly_seven_days_by_timestamp():
    idx = _index(3000)  # > 7 days @ 5-min steps (2016 steps/week)
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"n1": rng.normal(50, 1, len(idx))}, index=idx)

    feats = build_seasonal_residual_features(df, lag=pd.Timedelta("7D"), rolling_window=12)

    # Spot-check several rows well past the warmup: residual must equal
    # value(t) - value(t - 7 days) using genuine timestamp arithmetic, not
    # an assumed step count.
    for t in [idx[2500], idx[2800], idx[2999]]:
        t_minus_7d = t - pd.Timedelta("7D")
        assert t_minus_7d in df.index
        expected = df.loc[t, "n1"] - df.loc[t_minus_7d, "n1"]
        actual = feats.loc[t, "n1_wresid_7d"]
        assert np.isclose(actual, expected), f"{t}: expected {expected}, got {actual}"


def test_no_future_information_is_used():
    idx = _index(3000)
    rng = np.random.default_rng(4)
    df = pd.DataFrame({"n1": rng.normal(50, 1, len(idx))}, index=idx)

    feats_original = build_seasonal_residual_features(df, lag=pd.Timedelta("7D"), rolling_window=12)

    cutoff = 2500
    df_perturbed = df.copy()
    df_perturbed.iloc[cutoff:] += 1000.0
    feats_perturbed = build_seasonal_residual_features(
        df_perturbed, lag=pd.Timedelta("7D"), rolling_window=12
    )

    # Everything strictly before the perturbation, INCLUDING rows whose
    # lag lookup reaches back near (but not past) the cutoff, must be
    # identical. Only rows at/after cutoff, or whose rolling window/lag
    # reaches into [cutoff, end), may differ.
    safe_end = cutoff - 12  # rolling window margin
    pd.testing.assert_frame_equal(feats_original.iloc[:safe_end], feats_perturbed.iloc[:safe_end])


def test_row_alignment_identical_to_input():
    idx = _index(3000)
    df = pd.DataFrame({"n1": range(3000), "n2": range(3000, 6000)}, index=idx, dtype=float)
    feats = build_seasonal_residual_features(df, lag=pd.Timedelta("7D"), rolling_window=12)
    assert feats.index.equals(df.index)
    assert len(feats) == len(df)


def test_no_silent_row_loss_beyond_expected_warmup():
    idx = _index(3000)
    rng = np.random.default_rng(5)
    df = pd.DataFrame({"n1": rng.normal(50, 1, len(idx))}, index=idx)
    feats = build_seasonal_residual_features(df, lag=pd.Timedelta("7D"), rolling_window=12)

    warmup_steps = 2016 + 12 - 1  # 7 days of lag + (rolling_window - 1)
    assert feats.iloc[:warmup_steps].isna().any(axis=1).all(), "warmup rows should be NaN"
    assert feats.iloc[warmup_steps:].notna().all().all(), (
        "no NaN should remain after the expected warmup — any further NaN "
        "would be a silent, unexplained row loss (as the zero-std bug was)"
    )


def test_flat_repeating_values_do_not_produce_nan_or_inf():
    """A pump that's off at the same hour both weeks gives a long run of
    exactly-zero residuals — must stay 0, never NaN/inf (no division here,
    unlike build_change_features' z-score, but worth locking in)."""
    idx = _index(3000)
    values = np.zeros(3000)
    values[2500:2600] = 5.0  # a burst of real variation later on
    df = pd.DataFrame({"pump": values}, index=idx)
    feats = build_seasonal_residual_features(df, lag=pd.Timedelta("7D"), rolling_window=12)
    checked = feats.iloc[2016 + 11 : 2490]  # comfortably inside the flat-residual region
    assert np.isfinite(checked.to_numpy()).all()
    assert (checked == 0.0).all().all()
