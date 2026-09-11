"""
The frozen predictive evidence source for the reliability experiment.

Per instructions this is NOT tuned further: exact same recipe as the
"A: existing best (press+flow+level)" / Logistic configuration from the
seasonal-residual comparison (scripts/compare_seasonal_residuals.py) —
chosen over Random Forest specifically because every RF configuration in
prior rounds had an operationally unusable false-alarm rate (14-27/day),
while Logistic stayed in a usable range (0.5-3.5/day) throughout. This is
a documented selection among already-built candidates, not new tuning.

All three systems (A/B/C) call predict_scores() on identically-prepared
input — this module has no knowledge of which system is calling it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from wds_sentinel.data.battledim import derive_onsets, load_scada_csv
from wds_sentinel.experiments.split import split_masks_detection
from wds_sentinel.prediction.features import build_change_features
from wds_sentinel.prediction.onset_target import build_detection_label

DIFF_STEPS = (1, 3, 12)
BASELINE_WINDOW = 24
WINDOW = pd.Timedelta("1h")


@dataclass
class FrozenPredictor:
    model: LogisticRegression
    scaler: StandardScaler
    threshold: float
    feature_columns: list[str]
    onset_times_train: list[pd.Timestamp]

    def predict_scores(self, features: pd.DataFrame) -> pd.Series:
        """features must have exactly self.feature_columns, in any row
        order/index; NaN rows are not silently handled here — the caller
        (the degradation/imputation step) is responsible for producing a
        complete feature frame before calling this."""
        X = features[self.feature_columns]
        scores = self.model.predict_proba(self.scaler.transform(X))[:, 1]
        return pd.Series(scores, index=features.index)


def build_raw_features(pressure: pd.DataFrame, flow: pd.DataFrame, level: pd.DataFrame) -> pd.DataFrame:
    feat_pressure = build_change_features(pressure, diff_steps=DIFF_STEPS, baseline_window=BASELINE_WINDOW)
    feat_flow = build_change_features(flow, diff_steps=DIFF_STEPS, baseline_window=BASELINE_WINDOW).add_prefix("flow_")
    feat_level = build_change_features(level, diff_steps=DIFF_STEPS, baseline_window=BASELINE_WINDOW).add_prefix("level_")
    return pd.concat([feat_pressure, feat_flow, feat_level], axis=1)


def train_frozen_predictor(data_dir: Path) -> tuple[FrozenPredictor, dict]:
    """Reproduces exactly the training recipe already validated in
    scripts/compare_seasonal_residuals.py's Config A / Logistic. Returns
    the frozen predictor plus a dict of context (onset times, val rows)
    useful for downstream reliability-experiment scripts."""
    pressure = load_scada_csv(data_dir / "2018_SCADA_Pressures.csv")
    flow = load_scada_csv(data_dir / "2018_SCADA_Flows.csv")
    level = load_scada_csv(data_dir / "2018_SCADA_Levels.csv")
    leak = load_scada_csv(data_dir / "2018_Leakages.csv")

    onsets_18, carry_18 = derive_onsets(leak)
    assert carry_18 == []
    onset_times = sorted(onsets_18.values())

    label = build_detection_label(pressure.index, onset_times, window=WINDOW)
    masks = split_masks_detection(pressure.index, window=WINDOW)

    feats = build_raw_features(pressure, flow, level)
    valid = feats.notna().all(axis=1)
    train_idx = masks["train"] & valid
    val_idx = masks["validation"] & valid

    X_train, y_train = feats[train_idx], label[train_idx]
    scaler = StandardScaler().fit(X_train)
    model = LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0)
    model.fit(scaler.transform(X_train), y_train)

    # threshold: reproduce the same max-F1-on-validation rule
    from sklearn.metrics import precision_score, recall_score

    X_val, y_val = feats[val_idx], label[val_idx]
    val_scores = model.predict_proba(scaler.transform(X_val))[:, 1]
    thresholds = np.linspace(0.01, 0.99, 197)
    best_f1, best_th = -1, 0.5
    for th in thresholds:
        pred = (val_scores >= th).astype(int)
        p = precision_score(y_val, pred, zero_division=0)
        r = recall_score(y_val, pred, zero_division=0)
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        if f1 > best_f1:
            best_f1, best_th = f1, th

    predictor = FrozenPredictor(
        model=model,
        scaler=scaler,
        threshold=best_th,
        feature_columns=list(feats.columns),
        onset_times_train=onset_times,
    )
    val_onsets = [t for t in onset_times if masks["validation"].get(t, False)]
    context = dict(
        pressure=pressure, flow=flow, level=level, label=label, masks=masks,
        val_onsets=val_onsets, valid_mask=valid, onset_times=onset_times,
    )
    return predictor, context
