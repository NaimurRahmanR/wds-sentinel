import numpy as np
import pandas as pd

from wds_sentinel.prediction.degradation import (
    additive_noise,
    conflicting_evidence,
    missing_sensor_subset,
    sensor_dropout,
)


def _streams(n=200):
    idx = pd.date_range("2018-01-01", periods=n, freq="5min")
    rng = np.random.default_rng(0)
    pressure = pd.DataFrame({f"n{i}": rng.normal(50, 1, n) for i in range(5)}, index=idx)
    flow = pd.DataFrame({f"p{i}": rng.normal(100, 5, n) for i in range(3)}, index=idx)
    level = pd.DataFrame({"T1": rng.normal(3, 0.1, n)}, index=idx)
    return pressure, flow, level


def test_missing_subset_is_deterministic_given_same_seed():
    p, f, lv = _streams()
    d1 = missing_sensor_subset(p, f, lv, seed=42)
    d2 = missing_sensor_subset(p, f, lv, seed=42)
    pd.testing.assert_frame_equal(d1.pressure, d2.pressure)
    pd.testing.assert_frame_equal(d1.flow, d2.flow)
    pd.testing.assert_frame_equal(d1.level, d2.level)


def test_missing_subset_different_seed_gives_different_pattern():
    p, f, lv = _streams()
    d1 = missing_sensor_subset(p, f, lv, seed=42)
    d2 = missing_sensor_subset(p, f, lv, seed=99)
    cols_missing_1 = set(d1.pressure.columns[d1.pressure.isna().all()])
    cols_missing_2 = set(d2.pressure.columns[d2.pressure.isna().all()])
    assert cols_missing_1 != cols_missing_2 or set(d1.flow.columns[d1.flow.isna().all()]) != set(
        d2.flow.columns[d2.flow.isna().all()]
    )


def test_dropout_is_deterministic_given_same_seed():
    p, f, lv = _streams()
    d1 = sensor_dropout(p, f, lv, seed=7)
    d2 = sensor_dropout(p, f, lv, seed=7)
    pd.testing.assert_frame_equal(d1.pressure, d2.pressure)


def test_additive_noise_is_deterministic_given_same_seed():
    p, f, lv = _streams()
    train_stds = {"pressure": p.std(), "flow": f.std(), "level": lv.std()}
    d1 = additive_noise(p, f, lv, train_stds, seed=7)
    d2 = additive_noise(p, f, lv, train_stds, seed=7)
    pd.testing.assert_frame_equal(d1.pressure, d2.pressure)


def test_additive_noise_actually_changes_values():
    p, f, lv = _streams()
    train_stds = {"pressure": p.std(), "flow": f.std(), "level": lv.std()}
    d = additive_noise(p, f, lv, train_stds, seed=7)
    assert not d.pressure.equals(p)


def test_conflicting_evidence_only_touches_flow():
    p, f, lv = _streams()
    train_stds = {"pressure": p.std(), "flow": f.std(), "level": lv.std()}
    d = conflicting_evidence(p, f, lv, train_stds)
    pd.testing.assert_frame_equal(d.pressure, p)
    pd.testing.assert_frame_equal(d.level, lv)
    assert not d.flow.equals(f)


def test_conflicting_evidence_is_fully_deterministic_no_seed_needed():
    p, f, lv = _streams()
    train_stds = {"pressure": p.std(), "flow": f.std(), "level": lv.std()}
    d1 = conflicting_evidence(p, f, lv, train_stds)
    d2 = conflicting_evidence(p, f, lv, train_stds)
    pd.testing.assert_frame_equal(d1.flow, d2.flow)


def test_missing_subset_fraction_matches_config():
    from wds_sentinel.prediction.degradation import MISSING_SUBSET_FRACTION

    p, f, lv = _streams()
    d = missing_sensor_subset(p, f, lv, seed=42)
    total_cols = p.shape[1] + f.shape[1] + lv.shape[1]
    n_fully_missing = (
        d.pressure.isna().all().sum() + d.flow.isna().all().sum() + d.level.isna().all().sum()
    )
    assert n_fully_missing == round(MISSING_SUBSET_FRACTION * total_cols)
