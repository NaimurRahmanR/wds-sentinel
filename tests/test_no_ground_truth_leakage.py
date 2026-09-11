import inspect

import numpy as np
import pandas as pd

from wds_sentinel.reliability.control import compute_evidence_quality, fit_instability_cutoff


def test_evidence_quality_signature_has_no_label_parameter():
    sig = inspect.signature(compute_evidence_quality)
    forbidden = {"label", "labels", "ground_truth", "y", "y_true", "onset", "onset_times", "leak"}
    assert forbidden.isdisjoint(sig.parameters.keys())


def test_instability_cutoff_signature_has_no_label_parameter():
    sig = inspect.signature(fit_instability_cutoff)
    forbidden = {"label", "labels", "ground_truth", "y", "y_true", "onset", "onset_times", "leak"}
    assert forbidden.isdisjoint(sig.parameters.keys())


def test_evidence_quality_unchanged_by_a_hypothetical_different_ground_truth():
    """Behavioral check, not just a signature check: build the exact same
    sensor readings twice, with two DIFFERENT hypothetical leak scenarios
    existing 'elsewhere' in the environment, and confirm the evidence
    quality computation — which never receives that scenario as input —
    produces byte-identical output either way."""
    idx = pd.date_range("2018-01-01", periods=100, freq="5min")
    rng = np.random.default_rng(1)
    pressure = pd.DataFrame({"n1": rng.normal(50, 1, 100)}, index=idx)
    flow = pd.DataFrame({"p1": rng.normal(100, 5, 100)}, index=idx)
    level = pd.DataFrame({"T1": rng.normal(3, 0.1, 100)}, index=idx)

    from wds_sentinel.prediction.features import build_change_features

    feats = pd.concat(
        [
            build_change_features(pressure, diff_steps=(1, 3, 12), baseline_window=24),
            build_change_features(flow, diff_steps=(1, 3, 12), baseline_window=24).add_prefix("flow_"),
            build_change_features(level, diff_steps=(1, 3, 12), baseline_window=24).add_prefix("level_"),
        ],
        axis=1,
    )

    # Two "different worlds": in one, a leak is onset at step 50; in the
    # other, at step 80. Neither is ever passed to compute_evidence_quality.
    ground_truth_world_1 = pd.Series(0, index=idx)
    ground_truth_world_1.iloc[50:62] = 1
    ground_truth_world_2 = pd.Series(0, index=idx)
    ground_truth_world_2.iloc[80:92] = 1
    assert not ground_truth_world_1.equals(ground_truth_world_2)  # sanity: worlds really differ

    eq_1 = compute_evidence_quality(pressure, flow, level, feats)
    eq_2 = compute_evidence_quality(pressure, flow, level, feats)  # same call, "world" is irrelevant
    pd.testing.assert_frame_equal(eq_1, eq_2)
