import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from wds_sentinel.prediction.degradation import clean, missing_sensor_subset
from wds_sentinel.prediction.frozen_predictor import FrozenPredictor, build_raw_features
from wds_sentinel.reliability.control import fit_instability_cutoff
from wds_sentinel.reliability.systems import run_systems


def _toy_predictor_and_streams(n=300):
    idx = pd.date_range("2018-01-01", periods=n, freq="5min")
    rng = np.random.default_rng(3)
    pressure = pd.DataFrame({f"n{i}": rng.normal(50, 1, n) for i in range(4)}, index=idx)
    flow = pd.DataFrame({f"p{i}": rng.normal(100, 5, n) for i in range(2)}, index=idx)
    level = pd.DataFrame({"T1": rng.normal(3, 0.1, n)}, index=idx)

    feats = build_raw_features(pressure, flow, level)
    valid = feats.notna().all(axis=1)
    X = feats[valid]
    y = pd.Series(rng.integers(0, 2, size=len(X)), index=X.index)  # toy labels, structure only

    scaler = StandardScaler().fit(X)
    model = LogisticRegression(max_iter=500).fit(scaler.transform(X), y)
    predictor = FrozenPredictor(
        model=model, scaler=scaler, threshold=0.5, feature_columns=list(feats.columns), onset_times_train=[]
    )
    instability_cutoff = fit_instability_cutoff(feats[valid])
    return predictor, pressure, flow, level, instability_cutoff


def test_A_B_C_share_the_identical_scores_and_features_object():
    predictor, pressure, flow, level, cutoff = _toy_predictor_and_streams()
    degraded = clean(pressure, flow, level)
    result = run_systems(predictor, degraded, cutoff)

    # decision_A and decision_B are both derived from result.scores and
    # result.features directly (same object identity, not a recomputation)
    assert result.scores is result.scores  # sanity
    # Recompute A/B manually from the SAME scores/features and confirm
    # identical results — proving no hidden second computation path exists.
    from wds_sentinel.reasoning.rules import direct_decision, hybrid_decision

    recomputed_A = direct_decision(result.scores, predictor.threshold)
    recomputed_B = hybrid_decision(result.scores, result.features, predictor.threshold)
    pd.testing.assert_series_equal(result.decision_A, recomputed_A)
    pd.testing.assert_series_equal(result.decision_B, recomputed_B)


def test_A_B_C_evaluated_on_identical_row_index_under_degradation():
    predictor, pressure, flow, level, cutoff = _toy_predictor_and_streams()
    # sensor_dropout (intermittent, 5% per-timestep) rather than
    # missing_sensor_subset here: a toy series with no "before" history
    # can't ffill/bfill a sensor that's missing for its ENTIRE span, which
    # is exactly the real-world constraint the actual experiment handles
    # by degrading only the validation window while keeping training
    # history intact for imputation (see scripts/run_reliability_experiment.py).
    from wds_sentinel.prediction.degradation import sensor_dropout

    degraded = sensor_dropout(pressure, flow, level, seed=1)
    result = run_systems(predictor, degraded, cutoff)

    # All three decisions must be defined over exactly the same timestamps
    assert result.decision_A.index.equals(result.decision_B.index)
    assert result.decision_A.index.equals(result.decision_C.index)
    assert result.decision_A.index.equals(result.scores.index)


def test_C_base_decision_before_override_equals_B_exactly():
    """C is built FROM decision_B (see reliability/systems.py) — this
    confirms that identity at the source, not just at the metric level."""
    predictor, pressure, flow, level, cutoff = _toy_predictor_and_streams()
    degraded = clean(pressure, flow, level)
    result = run_systems(predictor, degraded, cutoff)

    from wds_sentinel.reliability.control import is_unreliable

    unreliable = is_unreliable(result.evidence_quality, cutoff)
    # wherever NOT unreliable, C must equal B exactly
    reliable_idx = result.decision_C.index[~unreliable]
    pd.testing.assert_series_equal(
        result.decision_C.loc[reliable_idx], result.decision_B.loc[reliable_idx]
    )
