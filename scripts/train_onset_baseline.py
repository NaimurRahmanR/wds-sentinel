"""
First baseline for the frozen target: at timestep t, will a new leak onset
occur anywhere in L-Town within the next hour (12 x 5-min steps)?

Pressure-only, causal features. Threshold chosen on validation only. Test
(full 2019) touched exactly once, at the end, for reporting.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wds_sentinel.data.battledim import derive_onsets, load_scada_csv  # noqa: E402
from wds_sentinel.experiments.split import split_masks  # noqa: E402
from wds_sentinel.prediction.event_metrics import (  # noqa: E402
    event_detection_rate,
    false_alerts_per_day,
    warning_lead_times,
)
from wds_sentinel.prediction.features import build_causal_features  # noqa: E402
from wds_sentinel.prediction.onset_target import build_onset_label  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data" / "raw"
HORIZON = 12

print("Loading data...")
p18 = load_scada_csv(DATA / "2018_SCADA_Pressures.csv")
p19 = load_scada_csv(DATA / "2019_SCADA_Pressures.csv")
l18 = load_scada_csv(DATA / "2018_Leakages.csv")
l19 = load_scada_csv(DATA / "2019_Leakages.csv")

pressures = pd.concat([p18, p19]).sort_index()
assert pressures.index.is_unique and pressures.index.is_monotonic_increasing

onsets_18, carry_18 = derive_onsets(l18)
onsets_19, carry_19 = derive_onsets(l19)
assert carry_18 == [], f"unexpected carryovers in 2018: {carry_18}"
assert set(carry_19) == {"p257", "p427", "p654", "p810"}, f"carryover set changed: {carry_19}"

all_onset_times = sorted(list(onsets_18.values()) + list(onsets_19.values()))
print(f"Onset events used for labeling: {len(all_onset_times)} "
      f"(2018: {len(onsets_18)}, 2019 genuine: {len(onsets_19)}, "
      f"2019 carryovers excluded: {sorted(carry_19)})")

print("Building label and features (causal only)...")
label = build_onset_label(pressures.index, all_onset_times, horizon_steps=HORIZON)
features = build_causal_features(pressures, window_steps=HORIZON)

valid = features.notna().all(axis=1)  # drop the first (window-1) rows of the whole series
masks = split_masks(pressures.index, horizon_steps=HORIZON)

train_idx = masks["train"] & valid
val_idx = masks["validation"] & valid
test_idx = masks["test"] & valid

print(f"Rows: train={train_idx.sum()}  val={val_idx.sum()}  test={test_idx.sum()}")
print(f"Positive rate: train={label[train_idx].mean():.4%}  "
      f"val={label[val_idx].mean():.4%}  test={label[test_idx].mean():.4%}")

X_train, y_train = features[train_idx], label[train_idx]
X_val, y_val = features[val_idx], label[val_idx]
X_test, y_test = features[test_idx], label[test_idx]

scaler = StandardScaler().fit(X_train)
Xs_train = scaler.transform(X_train)
Xs_val = scaler.transform(X_val)
Xs_test = scaler.transform(X_test)

print("Training logistic regression (class_weight='balanced')...")
clf = LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0)
clf.fit(Xs_train, y_train)

val_scores = clf.predict_proba(Xs_val)[:, 1]
test_scores = clf.predict_proba(Xs_test)[:, 1]

# --- threshold selection on validation only ---
thresholds = np.linspace(0.01, 0.99, 197)
f1s = []
for th in thresholds:
    pred = (val_scores >= th).astype(int)
    p = precision_score(y_val, pred, zero_division=0)
    r = recall_score(y_val, pred, zero_division=0)
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    f1s.append(f1)
best_th = thresholds[int(np.argmax(f1s))]
print(f"Threshold chosen on validation (max F1): {best_th:.3f} (val F1={max(f1s):.3f})")

# --- test-set evaluation (touched once) ---
test_pred = pd.Series((test_scores >= best_th).astype(int), index=X_test.index)

auprc = average_precision_score(y_test, test_scores)
auroc = roc_auc_score(y_test, test_scores)
precision = precision_score(y_test, test_pred, zero_division=0)
recall = recall_score(y_test, test_pred, zero_division=0)

test_onsets_2019 = [t for t in onsets_19.values() if t in test_pred.index]
det_rate, missed = event_detection_rate(test_pred, test_onsets_2019, horizon_steps=HORIZON)
false_per_day = false_alerts_per_day(test_pred, test_onsets_2019, horizon_steps=HORIZON)
lead_times = warning_lead_times(test_pred, test_onsets_2019, horizon_steps=HORIZON)
lead_minutes = [lt.total_seconds() / 60 for lt in lead_times]

print("\n=== TEST SET RESULTS (full 2019, genuine onsets only, touched once) ===")
print(f"n_test_rows={len(y_test)}  n_test_onset_events={len(test_onsets_2019)}")
print(f"AUPRC:  {auprc:.4f}")
print(f"AUROC:  {auroc:.4f}")
print(f"Precision @ threshold: {precision:.4f}")
print(f"Recall @ threshold:    {recall:.4f}")
print(f"Event detection rate:  {det_rate:.4f}  ({len(test_onsets_2019) - len(missed)}/{len(test_onsets_2019)} events)")
print(f"Missed onsets: {missed}")
print(f"False alerts/day:      {false_per_day:.3f}")
if lead_minutes:
    print(f"Warning lead time (minutes): mean={np.mean(lead_minutes):.1f}  "
          f"median={np.median(lead_minutes):.1f}  min={np.min(lead_minutes):.1f}  max={np.max(lead_minutes):.1f}")
else:
    print("Warning lead time: no detected events to measure")

# same set of metrics on validation, for reference / overfitting check
val_pred = pd.Series((val_scores >= best_th).astype(int), index=X_val.index)
val_onsets = [t for t in onsets_18.values() if t in val_pred.index]
val_auprc = average_precision_score(y_val, val_scores)
val_auroc = roc_auc_score(y_val, val_scores)
val_det_rate, val_missed = event_detection_rate(val_pred, val_onsets, horizon_steps=HORIZON)
print("\n=== VALIDATION SET RESULTS (for reference) ===")
print(f"n_val_onset_events={len(val_onsets)}")
print(f"AUPRC: {val_auprc:.4f}  AUROC: {val_auroc:.4f}  "
      f"detection_rate: {val_det_rate:.4f} ({len(val_onsets)-len(val_missed)}/{len(val_onsets)})")
