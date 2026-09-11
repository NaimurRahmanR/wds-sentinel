"""
A/B/C multimodal comparison for the (unchanged) early-onset-detection
target and (unchanged) 2018 train/validation split:
  A. pressure only (existing baseline)
  B. pressure + flow
  C. pressure + flow + level

Same causal change-feature construction, same two model configs, same
threshold-selection rule as the pressure-only baseline. 2018 only — 2019
is not touched here regardless of outcome, per instructions.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wds_sentinel.data.battledim import derive_onsets, load_scada_csv  # noqa: E402
from wds_sentinel.experiments.split import split_masks_detection  # noqa: E402
from wds_sentinel.prediction.event_metrics import (  # noqa: E402
    detection_rate_and_delay,
    false_alerts_per_day_detection,
)
from wds_sentinel.prediction.features import build_change_features  # noqa: E402
from wds_sentinel.prediction.onset_target import build_detection_label  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data" / "raw"
WINDOW = pd.Timedelta("1h")
DIFF_STEPS = (1, 3, 12)
BASELINE_WINDOW = 24

print("Loading 2018 streams...")
pressure = load_scada_csv(DATA / "2018_SCADA_Pressures.csv")
flow = load_scada_csv(DATA / "2018_SCADA_Flows.csv")
level = load_scada_csv(DATA / "2018_SCADA_Levels.csv")
leak = load_scada_csv(DATA / "2018_Leakages.csv")

assert pressure.index.equals(flow.index) and pressure.index.equals(level.index), \
    "streams are not on an identical timestamp grid — do not proceed on a misaligned join"

onsets_18, carry_18 = derive_onsets(leak)
assert carry_18 == []
onset_times = sorted(onsets_18.values())

label = build_detection_label(pressure.index, onset_times, window=WINDOW)
masks = split_masks_detection(pressure.index, window=WINDOW)
val_onsets = [t for t in onset_times if masks["validation"].get(t, False)]

feat_pressure = build_change_features(pressure, diff_steps=DIFF_STEPS, baseline_window=BASELINE_WINDOW)
feat_flow = build_change_features(flow, diff_steps=DIFF_STEPS, baseline_window=BASELINE_WINDOW).add_prefix("flow_")
feat_level = build_change_features(level, diff_steps=DIFF_STEPS, baseline_window=BASELINE_WINDOW).add_prefix("level_")

configs = {
    "A: pressure only": feat_pressure,
    "B: pressure + flow": pd.concat([feat_pressure, feat_flow], axis=1),
    "C: pressure + flow + level": pd.concat([feat_pressure, feat_flow, feat_level], axis=1),
}


def pick_threshold(y_true, scores):
    thresholds = np.linspace(0.01, 0.99, 197)
    best_f1, best_th = -1, 0.5
    for th in thresholds:
        pred = (scores >= th).astype(int)
        p = precision_score(y_true, pred, zero_division=0)
        r = recall_score(y_true, pred, zero_division=0)
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        if f1 > best_f1:
            best_f1, best_th = f1, th
    return best_th, best_f1


def evaluate(name, scores, y_true, index, threshold):
    pred = pd.Series((scores >= threshold).astype(int), index=index)
    auprc = average_precision_score(y_true, scores)
    auroc = roc_auc_score(y_true, scores)
    det_rate, delays, missed = detection_rate_and_delay(pred, val_onsets, window=WINDOW)
    false_per_day = false_alerts_per_day_detection(pred, val_onsets, window=WINDOW)
    delay_min = [d.total_seconds() / 60 for d in delays]
    med_delay = np.median(delay_min) if delay_min else float("nan")
    print(f"  {name}: AUPRC={auprc:.4f} AUROC={auroc:.4f} "
          f"detection_rate={det_rate:.2f} ({len(delays)}/{len(delays)+len(missed)}) "
          f"false/day={false_per_day:.2f} delay_med={med_delay:.1f}min")
    return dict(auprc=auprc, auroc=auroc, det_rate=det_rate, false_per_day=false_per_day,
                n_detected=len(delays), n_events=len(delays) + len(missed), med_delay=med_delay)


results = {}
for cfg_name, feats in configs.items():
    valid = feats.notna().all(axis=1)
    train_idx = masks["train"] & valid
    val_idx = masks["validation"] & valid

    X_train, y_train = feats[train_idx], label[train_idx]
    X_val, y_val = feats[val_idx], label[val_idx]

    scaler = StandardScaler().fit(X_train)
    Xs_train, Xs_val = scaler.transform(X_train), scaler.transform(X_val)

    print(f"\n=== {cfg_name} (n_features={feats.shape[1]}, train_rows={len(y_train)}, val_rows={len(y_val)}) ===")

    logit = LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0)
    logit.fit(Xs_train, y_train)
    val_scores = logit.predict_proba(Xs_val)[:, 1]
    th, _ = pick_threshold(y_val, val_scores)
    results[(cfg_name, "Logistic")] = evaluate("Logistic", val_scores, y_val, X_val.index, th)

    rf = RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced", n_jobs=-1, random_state=0)
    rf.fit(X_train, y_train)
    val_scores_rf = rf.predict_proba(X_val)[:, 1]
    th_rf, _ = pick_threshold(y_val, val_scores_rf)
    results[(cfg_name, "RandomForest")] = evaluate("RandomForest", val_scores_rf, y_val, X_val.index, th_rf)

print("\n=== Summary table (2018 validation, 5 onset events) ===")
print(f"{'Config':<28}{'Model':<14}{'AUPRC':>8}{'AUROC':>8}{'DetRate':>9}{'False/day':>11}{'Delay(med)':>12}")
for (cfg, model), r in results.items():
    print(f"{cfg:<28}{model:<14}{r['auprc']:>8.4f}{r['auroc']:>8.4f}"
          f"{r['det_rate']:>9.2f}{r['false_per_day']:>11.2f}{r['med_delay']:>12.1f}")
