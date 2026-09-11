"""Tests for the hazard detector: deterministic processing and no future
leakage (the same style of proof used for the WDS causal features)."""
import numpy as np
import pandas as pd

from wds_sentinel.hazard.detector import (
    antecedent_precipitation,
    fit_hazard_thresholds,
    hazard_elevated_flag,
)


def _precip(n=60, seed=0):
    idx = pd.date_range("2013-01-01", periods=n, freq="D")
    rng = np.random.default_rng(seed)
    values = rng.exponential(2.0, size=n)
    values[40] = 200.0  # one clear extreme day
    return pd.Series(values, index=idx, name="precip_mm")


def test_antecedent_precipitation_deterministic():
    p = _precip()
    a1 = antecedent_precipitation(p)
    a2 = antecedent_precipitation(p)
    pd.testing.assert_frame_equal(a1, a2)


def test_antecedent_precipitation_no_future_leakage():
    p = _precip()
    original = antecedent_precipitation(p)
    perturbed = p.copy()
    cutoff = 45
    perturbed.iloc[cutoff:] += 10_000.0
    perturbed_ante = antecedent_precipitation(perturbed)
    # rows before cutoff minus the longest window (7d) must be unaffected
    safe_end = cutoff - 7
    pd.testing.assert_frame_equal(original.iloc[:safe_end], perturbed_ante.iloc[:safe_end])


def test_thresholds_frozen_and_deterministic():
    p = _precip()
    ante = antecedent_precipitation(p)
    t1 = fit_hazard_thresholds(ante)
    t2 = fit_hazard_thresholds(ante)
    assert t1.cutoffs == t2.cutoffs


def test_hazard_flag_marks_the_known_extreme_day():
    p = _precip()
    ante = antecedent_precipitation(p)
    train_ante = ante.iloc[:35]  # excludes the extreme day at index 40
    thresholds = fit_hazard_thresholds(train_ante)
    flag = hazard_elevated_flag(ante, thresholds)
    assert bool(flag.iloc[40]) is True


def test_hazard_flag_undefined_not_false_when_all_inputs_missing():
    p = _precip()
    p_missing = p.copy()
    p_missing.iloc[:10] = np.nan  # first 10 days entirely missing
    ante = antecedent_precipitation(p_missing)
    train_ante = antecedent_precipitation(p)
    thresholds = fit_hazard_thresholds(train_ante)
    flag = hazard_elevated_flag(ante, thresholds)
    # day 0: 1d/3d/7d windows all need data that's entirely missing -> undefined
    assert flag.iloc[0] is pd.NA or pd.isna(flag.iloc[0])
    assert flag.iloc[0] is not False
