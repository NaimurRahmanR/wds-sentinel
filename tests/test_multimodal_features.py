"""Multimodal (pressure+flow+level) feature assembly is just concatenating
per-stream build_change_features() outputs — this tests that concatenation
doesn't break causality or accidentally misalign streams with different
column namespaces."""
import numpy as np
import pandas as pd

from wds_sentinel.prediction.features import build_change_features


def test_multimodal_concat_stays_causal():
    rng = np.random.default_rng(2)
    idx = pd.date_range("2020-01-01", periods=80, freq="5min")
    pressure = pd.DataFrame({"n1": rng.normal(50, 1, 80)}, index=idx)
    flow = pd.DataFrame({"p227": rng.normal(100, 5, 80)}, index=idx)
    level = pd.DataFrame({"T1": rng.normal(3, 0.1, 80)}, index=idx)

    def build_all(p, f, lv):
        return pd.concat(
            [
                build_change_features(p, diff_steps=(1, 3, 12), baseline_window=24),
                build_change_features(f, diff_steps=(1, 3, 12), baseline_window=24).add_prefix("flow_"),
                build_change_features(lv, diff_steps=(1, 3, 12), baseline_window=24).add_prefix("level_"),
            ],
            axis=1,
        )

    original = build_all(pressure, flow, level)

    cutoff = 50
    p2, f2, lv2 = pressure.copy(), flow.copy(), level.copy()
    p2.iloc[cutoff:] += 1000.0
    f2.iloc[cutoff:] += 1000.0
    lv2.iloc[cutoff:] += 1000.0
    perturbed = build_all(p2, f2, lv2)

    pd.testing.assert_frame_equal(original.iloc[: cutoff - 12], perturbed.iloc[: cutoff - 12])


def test_multimodal_column_namespaces_dont_collide():
    idx = pd.date_range("2020-01-01", periods=40, freq="5min")
    pressure = pd.DataFrame({"n1": range(40)}, index=idx, dtype=float)
    flow = pd.DataFrame({"n1": range(40, 80)}, index=idx, dtype=float)  # deliberately same col name
    feats_p = build_change_features(pressure, diff_steps=(1, 3, 12), baseline_window=24)
    feats_f = build_change_features(flow, diff_steps=(1, 3, 12), baseline_window=24).add_prefix("flow_")
    combined = pd.concat([feats_p, feats_f], axis=1)
    assert combined.shape[1] == feats_p.shape[1] + feats_f.shape[1]
    assert not combined["n1_raw"].equals(combined["flow_n1_raw"])
