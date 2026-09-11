"""
Validates the new agent-based architecture (Evidence/Prediction/
Reasoning/Reliability/Supervisory Agents) against the already-frozen
reliability experiment (scripts/run_reliability_experiment.py), on the
EXACT same 2018 validation rows and the same 5 degradation conditions.

Nothing frozen is touched: the frozen predictor, thresholds, degradation
functions, corroboration/hybrid-decision logic, and reliability cutoffs
are all imported and reused unchanged. This script only adds a new,
independent code path (the agents) and compares its output to the old one.

2019 is not touched.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wds_sentinel.agents.evidence_agent import EvidenceAgent  # noqa: E402
from wds_sentinel.agents.prediction_agent import PredictionAgent  # noqa: E402
from wds_sentinel.agents.reasoning_agent import KnowledgeReasoningAgent  # noqa: E402
from wds_sentinel.agents.reliability_agent import ReliabilityAgent  # noqa: E402
from wds_sentinel.agents.supervisory_agent import SupervisoryDecisionAgent  # noqa: E402
from wds_sentinel.evidence.adapters import DataFrameEvidenceAdapter, PredictionEvidenceAdapter  # noqa: E402
from wds_sentinel.knowledge.kbs import KnowledgeBase  # noqa: E402
from wds_sentinel.prediction.degradation import CONDITIONS, DegradedStreams, apply_condition  # noqa: E402
from wds_sentinel.prediction.frozen_predictor import build_raw_features, train_frozen_predictor  # noqa: E402
from wds_sentinel.reasoning.rules import corroboration_count  # noqa: E402
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

print("Training the frozen predictor (unchanged recipe — identical to run_reliability_experiment.py)...")
predictor, ctx = train_frozen_predictor(DATA)
pressure, flow, level = ctx["pressure"], ctx["flow"], ctx["level"]
label, masks = ctx["label"], ctx["masks"]
val_onsets = ctx["val_onsets"]
val_mask = masks["validation"]

train_mask = masks["train"] & ctx["valid_mask"]
train_stds = {
    "pressure": pressure[train_mask].std(),
    "flow": flow[train_mask].std(),
    "level": level[train_mask].std(),
}
clean_train_feats = build_raw_features(pressure[train_mask], flow[train_mask], level[train_mask])
instability_cutoff = fit_instability_cutoff(clean_train_feats)
print(f"threshold={predictor.threshold:.3f}  instability_cutoff={instability_cutoff:.4f}  "
      f"val onset events={len(val_onsets)}")

all_mismatches = []
per_condition_summary = []

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

    # --- FROZEN path (unchanged) ---
    frozen_result = run_systems(predictor, degraded_full, instability_cutoff)
    val_idx = frozen_result.decision_A.index[frozen_result.decision_A.index.isin(pressure.index[val_mask])]
    frozen_C = frozen_result.decision_C.loc[val_idx]

    # --- NEW agent path, built from the SAME imputed features/scores/evidence_quality
    # the frozen path already computed (no recomputation, no drift) ---
    corrob = corroboration_count(frozen_result.features)
    evidence_agent = EvidenceAgent(
        adapters=[
            DataFrameEvidenceAdapter(name="pressure", modality="pressure", raw=degraded_full.pressure.loc[val_idx]),
            DataFrameEvidenceAdapter(name="flow", modality="flow", raw=degraded_full.flow.loc[val_idx]),
            DataFrameEvidenceAdapter(name="level", modality="level", raw=degraded_full.level.loc[val_idx]),
            PredictionEvidenceAdapter(name="predictor", scores=frozen_result.scores.loc[val_idx]),
        ],
        evidence_quality=frozen_result.evidence_quality,
    )
    prediction_agent = PredictionAgent(scores=frozen_result.scores, threshold=predictor.threshold)
    reasoning_agent = KnowledgeReasoningAgent(kb=KnowledgeBase(), corroboration=corrob)
    reliability_agent = ReliabilityAgent(instability_cutoff=instability_cutoff)
    supervisor = SupervisoryDecisionAgent(evidence_agent, prediction_agent, reasoning_agent, reliability_agent)

    agent_decisions = {}
    for t in val_idx:
        agent_decisions[t] = supervisor.decide(t).decision
    agent_C = pd.Series(agent_decisions).reindex(val_idx)

    matches = (agent_C.values == frozen_C.values)
    n_rows = len(val_idx)
    n_match = int(matches.sum())
    mismatch_idx = val_idx[~matches]
    for t in mismatch_idx:
        all_mismatches.append(dict(condition=condition, timestamp=t, frozen=frozen_C.loc[t], agent=agent_C.loc[t]))

    print(f"  rows={n_rows}  matches={n_match}  match_rate={n_match/n_rows:.6%}  mismatches={len(mismatch_idx)}")

    # metric reproduction check: agent-based decisions vs frozen decisions,
    # both scored with the identical evaluation functions
    def metric_row(dec, tag):
        cov = coverage(dec)
        missed_rate, missed = missed_event_rate(dec, val_onsets, window=WINDOW)
        false_per_day = false_alerts_per_day(dec, val_onsets, window=WINDOW)
        abst_esc = abstention_escalation_rate(dec)
        unsafe = unsafe_autonomous_decision_rate(dec, label)
        return dict(condition=condition, source=tag, coverage=cov, missed_event_rate=missed_rate,
                    n_missed=len(missed), false_per_day=false_per_day, abst_esc_rate=abst_esc,
                    unsafe_rate=unsafe)

    per_condition_summary.append(metric_row(frozen_C, "frozen_C"))
    per_condition_summary.append(metric_row(agent_C, "agent_C"))

summary_df = pd.DataFrame(per_condition_summary)
print("\n=== Metric reproduction: frozen C vs agent-based C, per condition ===")
print(summary_df.to_string(index=False))

mismatch_df = pd.DataFrame(all_mismatches)
total_rows = 0
total_match = 0
print("\n=== Overall equivalence ===")
for condition in CONDITIONS:
    cond_mismatches = mismatch_df[mismatch_df["condition"] == condition] if len(mismatch_df) else pd.DataFrame()
    print(f"{condition}: {len(cond_mismatches)} mismatches")

if len(mismatch_df):
    print("\nFirst 10 mismatch examples:")
    print(mismatch_df.head(10).to_string(index=False))
else:
    print("\nNo mismatches found in any condition.")

mismatch_df.to_csv(Path(__file__).resolve().parents[1] / "results" / "derived" / "agent_vs_frozen_mismatches.csv", index=False)
summary_df.to_csv(Path(__file__).resolve().parents[1] / "results" / "derived" / "agent_vs_frozen_metric_summary.csv", index=False)
