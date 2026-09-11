"""
Early onset detection: at timestep t, using only pressure data at or
before t, detect a leak that began within the previous 1 hour.

Development (feature/model/threshold selection) uses 2018 train/validation
ONLY. 2019 is touched at most once, at the end, and only if 2018
validation shows genuine detection signal — per the pivot instructions,
2019 has already been used once (for the now-frozen, rejected 1-hour-ahead
forecasting baseline) and must not be used again for selection.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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
from wds_sentinel.experiments.split import split_masks_detection  # noqa: E402
from wds_sentinel.prediction.event_metrics import (  # noqa: E402
    detection_rate_and_delay,
    false_alerts_per_day_detection,
)
from wds_sentinel.prediction.features import build_change_features  # noqa: E402
from wds_sentinel.prediction.onset_target import build_detection_label  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data" / "raw"
WINDOW = pd.Timedelta("1h")

print("Loading 2018 data (development only)...")
p18 = load_scada_csv(DATA / "2018_SCADA_Pressures.csv")
l18 = load_scada_csv(DATA / "2018_Leakages.csv")

onsets_18, carry_18 = derive_onsets(l18)
assert carry_18 == [], f"unexpected carryovers in 2018: {carry_18}"
onset_times_18 = sorted(onsets_18.values())
print(f"2018 onset events: {len(onset_times_18)}")

print("Building detection label and change features (causal only)...")
label = build_detection_label(p18.index, onset_times_18, window=WINDOW)
features = build_change_features(p18, diff_steps=(1, 3, 12), baseline_window=24)

valid = features.notna().all(axis=1)
masks = split_masks_detection(p18.index, window=WINDOW)
train_idx = masks["train"] & valid
val_idx = masks["validation"] & valid

print(f"Rows: train={train_idx.sum()}  val={val_idx.sum()}")
print(f"Positive rate: train={label[train_idx].mean():.4%}  val={label[val_idx].mean():.4%}")

X_train, y_train = features[train_idx], label[train_idx]
X_val, y_val = features[val_idx], label[val_idx]
val_onsets = [t for t in onset_times_18 if masks["validation"].get(t, False)]

scaler = StandardScaler().fit(X_train)
Xs_train, Xs_val = scaler.transform(X_train), scaler.transform(X_val)


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


def evaluate(name, scores, y_true, index, onset_times, threshold):
    pred = pd.Series((scores >= threshold).astype(int), index=index)
    auprc = average_precision_score(y_true, scores)
    auroc = roc_auc_score(y_true, scores)
    precision = precision_score(y_true, pred, zero_division=0)
    recall = recall_score(y_true, pred, zero_division=0)
    det_rate, delays, missed = detection_rate_and_delay(pred, onset_times, window=WINDOW)
    false_per_day = false_alerts_per_day_detection(pred, onset_times, window=WINDOW)
    delay_min = [d.total_seconds() / 60 for d in delays]
    print(f"\n--- {name} ---")
    print(f"AUPRC={auprc:.4f}  AUROC={auroc:.4f}  precision={precision:.4f}  recall={recall:.4f}")
    print(f"detection_rate={det_rate:.4f} ({len(delays)}/{len(delays)+len(missed)} events)  "
          f"false_alerts/day={false_per_day:.3f}")
    if delay_min:
        print(f"detection delay (min): mean={np.mean(delay_min):.1f} median={np.median(delay_min):.1f} "
              f"min={np.min(delay_min):.1f} max={np.max(delay_min):.1f}")
    else:
        print("detection delay: no detected events")
    return dict(auprc=auprc, auroc=auroc, precision=precision, recall=recall,
                det_rate=det_rate, false_per_day=false_per_day, delay_min=delay_min)


print("\n=== Model 1: Logistic Regression (class_weight='balanced') ===")
logit = LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0)
logit.fit(Xs_train, y_train)
val_scores_logit = logit.predict_proba(Xs_val)[:, 1]
th_logit, f1_logit = pick_threshold(y_val, val_scores_logit)
print(f"Validation-selected threshold: {th_logit:.3f} (val F1={f1_logit:.3f})")
res_logit = evaluate("Logistic — 2018 VALIDATION", val_scores_logit, y_val, X_val.index,
                      val_onsets, th_logit)

print("\n=== Model 2: Random Forest (class_weight='balanced', n_estimators=200, max_depth=8) ===")
rf = RandomForestClassifier(
    n_estimators=200, max_depth=8, class_weight="balanced", n_jobs=-1, random_state=0
)
rf.fit(X_train, y_train)  # tree model: no scaling needed
val_scores_rf = rf.predict_proba(X_val)[:, 1]
th_rf, f1_rf = pick_threshold(y_val, val_scores_rf)
print(f"Validation-selected threshold: {th_rf:.3f} (val F1={f1_rf:.3f})")
res_rf = evaluate("Random Forest — 2018 VALIDATION", val_scores_rf, y_val, X_val.index,
                   val_onsets, th_rf)

print("\n=== Model comparison (2018 validation only; 2019 NOT used for this decision) ===")
print(f"Logistic:      AUPRC={res_logit['auprc']:.4f}  AUROC={res_logit['auroc']:.4f}")
print(f"Random Forest: AUPRC={res_rf['auprc']:.4f}  AUROC={res_rf['auroc']:.4f}")

best_name, best_res = ("Random Forest", res_rf) if res_rf["auprc"] >= res_logit["auprc"] else ("Logistic", res_logit)
print(f"\nBetter model on validation AUPRC: {best_name}")

BASE_RATE = y_val.mean()
USEFUL_SIGNAL = best_res["auprc"] > 3 * BASE_RATE and best_res["auroc"] > 0.65
print(f"\nValidation base rate: {BASE_RATE:.4%}")
print(f"Useful-signal check (AUPRC > 3x base rate AND AUROC > 0.65): {USEFUL_SIGNAL}")

if not USEFUL_SIGNAL:
    print("\n>>> STOPPING before 2019: 2018 validation does not demonstrate useful detection "
          "signal by the stated criterion. Per instructions, not running the 2019 check.")
else:
    print("\n>>> 2018 validation demonstrates useful signal — running the ONE final 2019 check.")
    print("Loading 2019 data for the single final evaluation...")
    p19 = load_scada_csv(DATA / "2019_SCADA_Pressures.csv")
    l19 = load_scada_csv(DATA / "2019_Leakages.csv")
    onsets_19, carry_19 = derive_onsets(l19)
    assert set(carry_19) == {"p257", "p427", "p654", "p810"}
    onset_times_19 = sorted(onsets_19.values())

    label_19 = build_detection_label(p19.index, onset_times_19, window=WINDOW)
    features_19 = build_change_features(p19, diff_steps=(1, 3, 12), baseline_window=24)
    valid_19 = features_19.notna().all(axis=1)
    test_mask = split_masks_detection(p19.index, window=WINDOW)["test"] & valid_19

    X_test, y_test = features_19[test_mask], label_19[test_mask]
    if best_name == "Random Forest":
        test_scores = rf.predict_proba(X_test)[:, 1]
        threshold = th_rf
    else:
        test_scores = logit.predict_proba(scaler.transform(X_test))[:, 1]
        threshold = th_logit

    test_onsets = [t for t in onset_times_19 if test_mask.get(t, False)]
    evaluate(f"{best_name} — 2019 FINAL TEST (single evaluation)", test_scores, y_test,
              X_test.index, test_onsets, threshold)
