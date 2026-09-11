"""Final frozen multimodal temporal evaluation on 2019 BattLeDIM data.

No model, feature, threshold, degradation, KBS, or reliability setting is
selected or tuned from 2019. The predictor is trained exactly by the frozen
2018 recipe, including its already-frozen 2018 validation threshold. The
reliability cutoffs and degradation scales are derived from clean 2018
training data exactly as in the frozen 2018 reliability experiment.

Important historical note: 2019 pressure/leakage data were inspected once by
an earlier rejected 1-hour-ahead forecasting baseline. They were not used for
subsequent detection-model, hybrid-architecture, or reliability-controller
development. This run is therefore a final temporal evaluation of the frozen
multimodal system, but is not described as a pristine/blind test.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wds_sentinel.data.battledim import derive_onsets, load_scada_csv  # noqa: E402
from wds_sentinel.prediction.degradation import CONDITIONS, DegradedStreams, apply_condition  # noqa: E402
from wds_sentinel.prediction.frozen_predictor import (  # noqa: E402
    BASELINE_WINDOW,
    build_raw_features,
    train_frozen_predictor,
)
from wds_sentinel.prediction.onset_target import build_detection_label  # noqa: E402
from wds_sentinel.experiments.split import split_masks_detection  # noqa: E402
from wds_sentinel.reliability.control import fit_instability_cutoff  # noqa: E402
from wds_sentinel.reliability.evaluation import (  # noqa: E402
    abstention_escalation_rate,
    coverage,
    false_alerts_per_day,
    missed_event_rate,
    selective_unsafe_risk,
    unsafe_autonomous_decision_rate,
)
from wds_sentinel.reliability.systems import run_systems  # noqa: E402

DATA = ROOT / "data" / "raw"
OUT = ROOT / "results" / "derived"
WINDOW = pd.Timedelta("1h")
HISTORY_ROWS = BASELINE_WINDOW  # 24 x 5 min = 2 h, sufficient for every frozen causal feature.

required = [
    "2018_SCADA_Pressures.csv", "2018_SCADA_Flows.csv", "2018_SCADA_Levels.csv", "2018_Leakages.csv",
    "2019_SCADA_Pressures.csv", "2019_SCADA_Flows.csv", "2019_SCADA_Levels.csv", "2019_Leakages.csv",
]
missing = [name for name in required if not (DATA / name).exists()]
if missing:
    raise FileNotFoundError(f"Final 2019 evaluation requires: {missing}")

print("Training frozen predictor from the unchanged 2018 recipe...")
predictor, ctx = train_frozen_predictor(DATA)
p18, f18, lv18 = ctx["pressure"], ctx["flow"], ctx["level"]
masks18 = ctx["masks"]
train_mask = masks18["train"] & ctx["valid_mask"]
train_stds = {
    "pressure": p18[train_mask].std(),
    "flow": f18[train_mask].std(),
    "level": lv18[train_mask].std(),
}
clean_train_feats = build_raw_features(p18[train_mask], f18[train_mask], lv18[train_mask])
instability_cutoff = fit_instability_cutoff(clean_train_feats)
print(f"Frozen threshold={predictor.threshold:.6f}; instability cutoff={instability_cutoff:.6f}")

p19 = load_scada_csv(DATA / "2019_SCADA_Pressures.csv")
f19 = load_scada_csv(DATA / "2019_SCADA_Flows.csv")
lv19 = load_scada_csv(DATA / "2019_SCADA_Levels.csv")
leak19 = load_scada_csv(DATA / "2019_Leakages.csv")

if not (p19.index.equals(f19.index) and p19.index.equals(lv19.index) and p19.index.equals(leak19.index)):
    raise ValueError("2019 pressure/flow/level/leakage timestamps are not exactly aligned")
if list(p19.columns) != list(p18.columns):
    raise ValueError("2019 pressure schema differs from the frozen 2018 predictor schema")
if list(f19.columns) != list(f18.columns):
    raise ValueError("2019 flow schema differs from the frozen 2018 predictor schema")
if list(lv19.columns) != list(lv18.columns):
    raise ValueError("2019 level schema differs from the frozen 2018 predictor schema")

onsets19_by_pipe, carryovers19 = derive_onsets(leak19)
onsets19 = sorted(onsets19_by_pipe.values())
label19 = build_detection_label(p19.index, onsets19, window=WINDOW)
test_mask19 = split_masks_detection(p19.index, window=WINDOW)["test"]
test_index19 = p19.index[test_mask19]
print(f"2019 raw rows={len(p19)}; frozen scored test rows={len(test_index19)}; genuine new onsets={len(onsets19)}; carryovers excluded={len(carryovers19)} {sorted(carryovers19)}")
print(f"Positive detection-window rows in scored test={int(label19.loc[test_index19].sum())} ({float(label19.loc[test_index19].mean()):.6%})")

# Causal history immediately preceding 2019. It is never degraded and is only
# used so the first 2019 rows can compute the same past-only 2-hour features.
hp, hf, hl = p18.tail(HISTORY_ROWS), f18.tail(HISTORY_ROWS), lv18.tail(HISTORY_ROWS)

rows: list[dict] = []
predictive_rows: list[dict] = []
for condition in CONDITIONS:
    print(f"\n=== Final 2019 condition: {condition} ===")
    if condition == "clean":
        test_streams = DegradedStreams(p19.copy(), f19.copy(), lv19.copy(), condition)
    else:
        test_streams = apply_condition(condition, p19, f19, lv19, train_stds)

    combined = DegradedStreams(
        pd.concat([hp, test_streams.pressure]),
        pd.concat([hf, test_streams.flow]),
        pd.concat([hl, test_streams.level]),
        condition,
    )
    result = run_systems(predictor, combined, instability_cutoff)
    test_idx = result.scores.index[result.scores.index.isin(test_index19)]
    if len(test_idx) != len(test_index19):
        raise AssertionError(f"Expected all {len(test_index19)} frozen test rows to be scoreable; got {len(test_idx)}")

    scores = result.scores.loc[test_idx]
    y = label19.reindex(test_idx).astype(int)
    binary = (scores >= predictor.threshold).astype(int)
    pred_metrics = {
        "condition": condition,
        "n_rows": len(y),
        "positive_rows": int(y.sum()),
        "positive_rate": float(y.mean()),
        "auprc": float(average_precision_score(y, scores)),
        "auroc": float(roc_auc_score(y, scores)),
        "precision_at_frozen_threshold": float(precision_score(y, binary, zero_division=0)),
        "recall_at_frozen_threshold": float(recall_score(y, binary, zero_division=0)),
        "f1_at_frozen_threshold": float(f1_score(y, binary, zero_division=0)),
        "frozen_threshold": float(predictor.threshold),
    }
    predictive_rows.append(pred_metrics)
    print("  predictor " + "  ".join(f"{k}={v:.6f}" for k, v in pred_metrics.items() if isinstance(v, float) and k not in {"positive_rate", "frozen_threshold"}))

    for system_name, decisions in [
        ("A_DIRECT", result.decision_A),
        ("B_HYBRID", result.decision_B),
        ("C_RELIABILITY_AWARE", result.decision_C),
    ]:
        dec = decisions.loc[test_idx]
        cov = coverage(dec)
        abst_esc = abstention_escalation_rate(dec)
        risk, n_unsafe, n_autonomous = selective_unsafe_risk(dec, label19)
        unsafe_rate = unsafe_autonomous_decision_rate(dec, label19)
        missed_rate, missed = missed_event_rate(dec, onsets19, window=WINDOW)
        det_rate = 1.0 - missed_rate if not np.isnan(missed_rate) else float("nan")
        # Relative to the NEW-ONSET target. Alerts during carryover-leak periods
        # can therefore count as non-onset episodes; this is named explicitly in
        # the output to avoid implying they are necessarily operationally false.
        non_onset_alerts_day = false_alerts_per_day(dec, onsets19, window=WINDOW)
        row = {
            "condition": condition,
            "system": system_name,
            "n_rows": len(dec),
            "autonomous_coverage": cov,
            "abstain_escalate_rate": abst_esc,
            "unsafe_autonomous_count": n_unsafe,
            "autonomous_count": n_autonomous,
            "selective_unsafe_risk": risk,
            "unsafe_timestep_rate": unsafe_rate,
            "missed_onset_events": len(missed),
            "n_onset_events": len(onsets19),
            "event_detection_rate": det_rate,
            "non_onset_alert_episodes_per_day": non_onset_alerts_day,
        }
        rows.append(row)
        risk_str = "undefined" if np.isnan(risk) else f"{risk:.6f}"
        print(
            f"  {system_name:<22} coverage={cov:.4f} risk={risk_str} "
            f"detected={len(onsets19)-len(missed)}/{len(onsets19)} "
            f"non-onset-alert-episodes/day={non_onset_alerts_day:.3f} "
            f"abstain/escalate={abst_esc:.4f}"
        )

OUT.mkdir(parents=True, exist_ok=True)
summary = pd.DataFrame(rows)
pred_summary = pd.DataFrame(predictive_rows)
summary_path = OUT / "final_2019_temporal_evaluation.csv"
pred_path = OUT / "final_2019_predictive_metrics.csv"
summary.to_csv(summary_path, index=False)
pred_summary.to_csv(pred_path, index=False)

clean = summary[summary["condition"] == "clean"].copy()
clean_pred = pred_summary[pred_summary["condition"] == "clean"].iloc[0]
md = [
    "# Final frozen 2019 temporal evaluation",
    "",
    "All model, threshold, feature, degradation and reliability settings were frozen before this run.",
    f"The four 2019 modalities contain {len(p19):,} aligned rows. Consistent with the frozen backward-looking detection split, the first one-hour boundary embargo leaves {len(test_index19):,} scored test rows. There are 19 genuine new leak onsets; four leaks already active at 2019-01-01 are treated as carryovers and excluded from the new-onset target.",
    "",
    "2019 is not described as a pristine blind test: pressure/leakage were previously inspected once by an earlier rejected forecasting baseline, but 2019 was not used for subsequent detection, KBS/MAS or reliability-controller development.",
    "",
    f"Clean predictor: AUPRC={clean_pred['auprc']:.6f}, AUROC={clean_pred['auroc']:.6f}, precision={clean_pred['precision_at_frozen_threshold']:.6f}, recall={clean_pred['recall_at_frozen_threshold']:.6f} at the frozen threshold {clean_pred['frozen_threshold']:.6f}.",
    "",
    "## Clean-condition system results",
    "",
    "| System | Coverage | Selective unsafe risk | Events detected | Non-onset alert episodes/day | Abstain/escalate |",
    "|---|---:|---:|---:|---:|---:|",
]
for _, r in clean.iterrows():
    risk = "undefined" if pd.isna(r["selective_unsafe_risk"]) else f"{r['selective_unsafe_risk']:.6f}"
    detected = int(r["n_onset_events"] - r["missed_onset_events"])
    md.append(
        f"| {r['system']} | {r['autonomous_coverage']:.4f} | {risk} | {detected}/{int(r['n_onset_events'])} | "
        f"{r['non_onset_alert_episodes_per_day']:.3f} | {r['abstain_escalate_rate']:.4f} |"
    )
md += [
    "",
    "`non_onset_alert_episodes_per_day` is defined relative to the new-onset detection target. Because four leaks carry over from 2018 into 2019, an episode outside a new-onset window is not necessarily an operationally false leak indication.",
    "",
    "The five controlled degradation conditions are reported in `final_2019_temporal_evaluation.csv`; no setting was changed after observing these results.",
]
(OUT / "final_2019_temporal_summary.md").write_text("\n".join(md) + "\n")
print(f"\nWrote {summary_path.relative_to(ROOT)}")
print(f"Wrote {pred_path.relative_to(ROOT)}")
print(f"Wrote results/derived/final_2019_temporal_summary.md")
