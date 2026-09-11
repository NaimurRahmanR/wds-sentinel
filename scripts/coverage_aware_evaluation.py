"""
Coverage-aware evaluation, using the frozen 2018 validation outputs only.
Nothing frozen is touched: predictors, thresholds, degradation functions,
KBS rules, agent logic, and the already-reported A/B/C results are all
recomputed identically (same code, same inputs) purely to expose their
already-existing intermediate values (decision_B, evidence_quality) for
this additional descriptive analysis — no new decision logic is applied
to the frozen operating point itself.

The sensitivity sweep for C is DESCRIPTIVE ONLY: it shows how selective
risk and coverage move together as the instability cutoff is varied
across percentiles of the same clean-training instability distribution
that produced the frozen 99th-percentile cutoff. It does not select,
recommend, or imply a "better" cutoff — the frozen operating point (99th
percentile) is marked, not chosen from among the alternatives shown.

2019 is not touched.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wds_sentinel.prediction.degradation import CONDITIONS, DegradedStreams, apply_condition  # noqa: E402
from wds_sentinel.prediction.frozen_predictor import build_raw_features, train_frozen_predictor  # noqa: E402
from wds_sentinel.reliability.control import fit_instability_cutoff, reliability_aware_decision  # noqa: E402
from wds_sentinel.reliability.evaluation import (  # noqa: E402
    abstention_escalation_rate,
    coverage,
    false_alerts_per_day,
    missed_event_rate,
    selective_unsafe_risk,
)
from wds_sentinel.reliability.systems import run_systems  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data" / "raw"
OUT = Path(__file__).resolve().parents[1] / "results" / "derived"
WINDOW = pd.Timedelta("1h")
FROZEN_PERCENTILE = 99.0
SWEEP_PERCENTILES = [50, 75, 90, 95, 97.5, 99.0, 99.5, 99.9, 99.99]

print("Training the frozen predictor (unchanged recipe)...")
predictor, ctx = train_frozen_predictor(DATA)
pressure, flow, level = ctx["pressure"], ctx["flow"], ctx["level"]
label, masks = ctx["label"], ctx["masks"]
val_onsets = ctx["val_onsets"]
val_mask = masks["validation"]

train_mask = masks["train"] & ctx["valid_mask"]
train_stds = {
    "pressure": pressure[train_mask].std(), "flow": flow[train_mask].std(), "level": level[train_mask].std(),
}
clean_train_feats = build_raw_features(pressure[train_mask], flow[train_mask], level[train_mask])
train_diff1_cols = [c for c in clean_train_feats.columns if c.endswith("_diff1")]
train_instability_dist = clean_train_feats[train_diff1_cols].std(axis=1, skipna=True)
frozen_instability_cutoff = fit_instability_cutoff(clean_train_feats)
print(f"Frozen instability cutoff (99th pct of clean training instability): {frozen_instability_cutoff:.4f}")

main_rows = []
sweep_rows = []

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

    result = run_systems(predictor, degraded_full, frozen_instability_cutoff)
    val_idx = result.decision_A.index[result.decision_A.index.isin(pressure.index[val_mask])]

    for system_name, decisions in [("A_DIRECT", result.decision_A),
                                     ("B_HYBRID", result.decision_B),
                                     ("C_RELIABILITY_AWARE", result.decision_C)]:
        dec = decisions.loc[val_idx]
        cov = coverage(dec)
        abst_esc = abstention_escalation_rate(dec)
        risk, n_unsafe, n_autonomous = selective_unsafe_risk(dec, label)
        missed_rate, missed_events = missed_event_rate(dec, val_onsets, window=WINDOW)
        det_rate = 1.0 - missed_rate if missed_rate == missed_rate else float("nan")
        false_per_day = false_alerts_per_day(dec, val_onsets, window=WINDOW)
        main_rows.append(dict(
            condition=condition, system=system_name, autonomous_coverage=cov,
            abstain_escalate_rate=abst_esc, unsafe_autonomous_count=n_unsafe,
            autonomous_count=n_autonomous, selective_unsafe_risk=risk,
            missed_onset_events=len(missed_events), n_onset_events=len(val_onsets),
            event_detection_rate=det_rate, false_alerts_per_day=false_per_day,
        ))
        risk_str = f"{risk:.4f}" if risk == risk else "undefined (0 autonomous decisions)"
        print(f"  {system_name:<22} coverage={cov:.4f}  abstain/escalate={abst_esc:.4f}  "
              f"unsafe_count={n_unsafe}/{n_autonomous}  selective_risk={risk_str}  "
              f"missed={len(missed_events)}/{len(val_onsets)}  false/day={false_per_day:.2f}")

    # --- Descriptive sensitivity sweep for C only: vary the instability
    # cutoff percentile; missingness/disagreement cutoffs stay at their
    # frozen fixed values throughout (not part of this sweep). ---
    for pct in SWEEP_PERCENTILES:
        candidate_cutoff = float(train_instability_dist.quantile(pct / 100.0))
        candidate_C = reliability_aware_decision(result.decision_B, result.evidence_quality, candidate_cutoff)
        dec = candidate_C.loc[val_idx]
        cov = coverage(dec)
        risk, n_unsafe, n_autonomous = selective_unsafe_risk(dec, label)
        sweep_rows.append(dict(
            condition=condition, instability_percentile=pct, instability_cutoff=candidate_cutoff,
            is_frozen_operating_point=(pct == FROZEN_PERCENTILE),
            coverage=cov, selective_unsafe_risk=risk, autonomous_count=n_autonomous, unsafe_count=n_unsafe,
        ))

main_df = pd.DataFrame(main_rows)
sweep_df = pd.DataFrame(sweep_rows)

main_csv = OUT / "coverage_aware_evaluation_summary.csv"
sweep_csv = OUT / "reliability_sensitivity_sweep_C.csv"
main_df.to_csv(main_csv, index=False)
sweep_df.to_csv(sweep_csv, index=False)
print(f"\nWrote {main_csv}")
print(f"Wrote {sweep_csv}")

# --- Figure: selective risk vs coverage, descriptive sensitivity only ---
fig, ax = plt.subplots(figsize=(8, 6))
colors = plt.cm.tab10.colors
for i, condition in enumerate(CONDITIONS):
    sub = sweep_df[sweep_df["condition"] == condition].sort_values("coverage")
    plottable = sub[sub["selective_unsafe_risk"].notna()]
    ax.plot(plottable["coverage"], plottable["selective_unsafe_risk"], marker="o", markersize=4,
            color=colors[i % len(colors)], label=condition, alpha=0.8)
    frozen_pt = sub[sub["is_frozen_operating_point"]]
    if len(frozen_pt) and frozen_pt["selective_unsafe_risk"].notna().iloc[0]:
        ax.scatter(frozen_pt["coverage"], frozen_pt["selective_unsafe_risk"], marker="*", s=250,
                   color=colors[i % len(colors)], edgecolor="black", zorder=5)
    zero_cov = sub[(sub["coverage"] == 0.0)]
    if len(zero_cov):
        ax.axvline(0.0, color=colors[i % len(colors)], linestyle=":", alpha=0.3)

ax.scatter([], [], marker="*", s=250, color="gray", edgecolor="black", label="Frozen operating point (99th pct)")
ax.annotate(
    "missing_sensor_subset:\ncoverage = 0 at EVERY\ntested cutoff (50th-99.99th pct)\nselective risk UNDEFINED\n(missingness gate dominates,\nunaffected by this sweep)",
    xy=(0.0, ax.get_ylim()[0]), xytext=(0.05, ax.get_ylim()[0] + 0.15 * (ax.get_ylim()[1] - ax.get_ylim()[0])),
    fontsize=7.5, color="darkorange",
    arrowprops=dict(arrowstyle="->", color="darkorange"),
)
ax.set_xlabel("Autonomous coverage (fraction of rows with ALERT/NO_ALERT, not ABSTAIN/ESCALATE)")
ax.set_ylabel("Selective unsafe risk (unsafe autonomous decisions / autonomous decisions)")
ax.set_title("DESCRIPTIVE SENSITIVITY ANALYSIS ONLY — not a cutoff selection\n"
              "System C: selective risk vs. coverage as instability-cutoff percentile varies\n"
              "(points at coverage=0 have undefined risk and are excluded from the curves — see notes)",
              fontsize=9)
ax.legend(fontsize=7, loc="best")
ax.grid(alpha=0.3)
fig.tight_layout()
fig_path = OUT / "reliability_sensitivity_risk_vs_coverage.png"
fig.savefig(fig_path, dpi=150)
print(f"Wrote {fig_path}")

print("\n=== Rows with coverage == 0 in the sweep (risk undefined, shown explicitly, not omitted) ===")
zero_cov_rows = sweep_df[sweep_df["coverage"] == 0.0]
print(zero_cov_rows.to_string(index=False) if len(zero_cov_rows) else "(none)")
