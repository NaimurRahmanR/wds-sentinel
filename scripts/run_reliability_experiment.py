"""
Core reliability experiment. 2018 validation only — 2019 is not touched.

Degradation is injected ONLY into the validation-period rows of each
stream, leaving the training-period rows of the same continuous series
untouched — so the shared ffill/bfill imputation step (reliability/systems.py)
can always fall back on real pre-validation history rather than
collapsing to an unrecoverable all-NaN column, which is exactly what a
sensor "taken offline starting July 1" would look like to a real system
that remembers its own history.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wds_sentinel.prediction.degradation import CONDITIONS, DegradedStreams, apply_condition  # noqa: E402
from wds_sentinel.prediction.frozen_predictor import build_raw_features, train_frozen_predictor  # noqa: E402
from wds_sentinel.reliability.control import fit_instability_cutoff  # noqa: E402
from wds_sentinel.reliability.evaluation import (  # noqa: E402
    abstention_escalation_rate,
    coverage,
    false_alerts_per_day,
    missed_event_rate,
    unsafe_autonomous_decision_rate,
)
from wds_sentinel.reliability.systems import run_systems  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data" / "raw"
WINDOW = pd.Timedelta("1h")

print("Training the frozen predictor (unchanged recipe)...")
predictor, ctx = train_frozen_predictor(DATA)
pressure, flow, level = ctx["pressure"], ctx["flow"], ctx["level"]
label, masks = ctx["label"], ctx["masks"]
val_onsets = ctx["val_onsets"]
val_mask = masks["validation"]

print(f"Frozen predictor: threshold={predictor.threshold:.3f}, "
      f"n_features={len(predictor.feature_columns)}, val onset events={len(val_onsets)}")

# train_stds and instability cutoff: computed ONCE from clean training
# data only, frozen, reused unchanged across every condition.
train_mask = masks["train"] & ctx["valid_mask"]
train_stds = {
    "pressure": pressure[train_mask].std(),
    "flow": flow[train_mask].std(),
    "level": level[train_mask].std(),
}
clean_train_feats = build_raw_features(pressure[train_mask], flow[train_mask], level[train_mask])
instability_cutoff = fit_instability_cutoff(clean_train_feats)
print(f"Instability cutoff (99th pct of clean training instability): {instability_cutoff:.4f}")

results = []
for condition in CONDITIONS:
    print(f"\n=== Condition: {condition} ===")
    full_pressure, full_flow, full_level = pressure.copy(), flow.copy(), level.copy()

    if condition != "clean":
        degraded_val_only = apply_condition(
            condition, pressure[val_mask], flow[val_mask], level[val_mask], train_stds
        )
        full_pressure.loc[val_mask] = degraded_val_only.pressure
        full_flow.loc[val_mask] = degraded_val_only.flow
        full_level.loc[val_mask] = degraded_val_only.level

    degraded_full = DegradedStreams(full_pressure, full_flow, full_level, condition)
    result = run_systems(predictor, degraded_full, instability_cutoff)

    # restrict evaluation strictly to the validation window
    val_idx = result.decision_A.index[result.decision_A.index.isin(pressure.index[val_mask])]

    for system_name, decisions in [("A_DIRECT", result.decision_A),
                                     ("B_HYBRID", result.decision_B),
                                     ("C_RELIABILITY_AWARE", result.decision_C)]:
        dec = decisions.loc[val_idx]
        unsafe_rate = unsafe_autonomous_decision_rate(dec, label)
        cov = coverage(dec)
        missed_rate, missed_events = missed_event_rate(dec, val_onsets, window=WINDOW)
        false_per_day = false_alerts_per_day(dec, val_onsets, window=WINDOW)
        abst_esc_rate = abstention_escalation_rate(dec)
        results.append(dict(
            condition=condition, system=system_name, unsafe_rate=unsafe_rate, coverage=cov,
            missed_event_rate=missed_rate, n_missed=len(missed_events), n_events=len(val_onsets),
            false_per_day=false_per_day, abst_esc_rate=abst_esc_rate,
        ))
        print(f"  {system_name:<22} unsafe_rate={unsafe_rate:.4%}  coverage={cov:.4f}  "
              f"missed={missed_rate:.2f} ({len(missed_events)}/{len(val_onsets)})  "
              f"false/day={false_per_day:.2f}  abstain/escalate={abst_esc_rate:.4%}")

results_df = pd.DataFrame(results)
print("\n=== Full results table ===")
print(results_df.to_string(index=False))
