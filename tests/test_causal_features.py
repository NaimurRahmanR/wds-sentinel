import numpy as np
import pandas as pd

from wds_sentinel.prediction.features import build_causal_features


def test_features_at_t_unaffected_by_perturbing_the_future():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=50, freq="5min")
    df = pd.DataFrame({"n1": rng.normal(50, 1, size=50)}, index=idx)

    feats_original = build_causal_features(df, window_steps=12)

    df_perturbed = df.copy()
    cutoff = 30
    df_perturbed.iloc[cutoff:] = df_perturbed.iloc[cutoff:] + 1000.0  # blow up the future
    feats_perturbed = build_causal_features(df_perturbed, window_steps=12)

    # Every row strictly before the perturbation must be identical: if it
    # isn't, some feature is looking into the future.
    pd.testing.assert_frame_equal(
        feats_original.iloc[: cutoff - 1], feats_perturbed.iloc[: cutoff - 1]
    )


def test_first_window_rows_are_nan_not_fabricated():
    idx = pd.date_range("2020-01-01", periods=20, freq="5min")
    df = pd.DataFrame({"n1": range(20)}, index=idx)
    feats = build_causal_features(df, window_steps=12)
    assert feats["n1_mean12"].iloc[:11].isna().all()
    assert feats["n1_mean12"].iloc[11:].notna().all()


# --- build_change_features ---
from wds_sentinel.prediction.features import build_change_features


def test_change_features_unaffected_by_perturbing_the_future():
    rng = np.random.default_rng(1)
    idx = pd.date_range("2020-01-01", periods=80, freq="5min")
    df = pd.DataFrame(
        {"n1": rng.normal(50, 1, size=80), "n2": rng.normal(40, 1, size=80)}, index=idx
    )
    feats_original = build_change_features(df, diff_steps=(1, 3, 12), baseline_window=24)

    df_perturbed = df.copy()
    cutoff = 50
    df_perturbed.iloc[cutoff:] += 1000.0
    feats_perturbed = build_change_features(df_perturbed, diff_steps=(1, 3, 12), baseline_window=24)

    pd.testing.assert_frame_equal(
        feats_original.iloc[: cutoff - 12], feats_perturbed.iloc[: cutoff - 12]
    )


def test_change_features_cross_sensor_columns_present():
    idx = pd.date_range("2020-01-01", periods=40, freq="5min")
    df = pd.DataFrame({"n1": range(40), "n2": range(40, 80)}, index=idx, dtype=float)
    feats = build_change_features(df, diff_steps=(1, 3, 12), baseline_window=24)
    for col in ["cross_mean_diff1", "cross_min_diff1", "cross_mean_z", "cross_min_z",
                "cross_count_below_z_cutoff"]:
        assert col in feats.columns


def test_zscore_is_zero_not_nan_during_flat_baseline():
    """A sensor sitting at a constant value for a full baseline window
    (e.g. a pump that's off) must not produce NaN/inf in its z-score."""
    idx = pd.date_range("2020-01-01", periods=40, freq="5min")
    values = [0.0] * 30 + [5.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0]
    df = pd.DataFrame({"pump": values}, index=idx)
    feats = build_change_features(df, diff_steps=(1, 3, 12), baseline_window=24)
    z = feats["pump_z24"]
    assert z.iloc[24:30].notna().all(), "flat-baseline rows should not be NaN"
    assert (z.iloc[24:30] == 0.0).all(), "flat-baseline rows should z-score to exactly 0, not NaN/inf"
    assert np.isfinite(z.iloc[30:]).all(), "no inf/NaN once real variation resumes"
