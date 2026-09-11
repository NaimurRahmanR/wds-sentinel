"""
Exact-equivalence regression test between the new agent pipeline and the
frozen reliability_aware_decision path, on a bounded slice of the REAL
2018 validation data (not synthetic) — catches any future drift between
the two code paths that the full validate_agents_vs_frozen.py script
(run manually; ~2 min over the complete validation set) already confirmed
is currently zero.

Skipped automatically if the bundled BattLeDIM CSVs are absent. The test
uses a genuinely bounded feature window so the presence of the data does
not accidentally turn normal CI into a full-year recomputation.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

DATA = Path(__file__).resolve().parents[1] / "data" / "raw"
REQUIRED_FILES = [
    "2018_SCADA_Pressures.csv", "2018_SCADA_Flows.csv",
    "2018_SCADA_Levels.csv", "2018_Leakages.csv",
]
DATA_AVAILABLE = all((DATA / f).exists() for f in REQUIRED_FILES)

pytestmark = pytest.mark.skipif(
    not DATA_AVAILABLE, reason="real BattLeDIM 2018 CSVs not present locally"
)


@pytest.fixture(scope="module")
def frozen_setup():
    from wds_sentinel.prediction.frozen_predictor import build_raw_features, train_frozen_predictor
    from wds_sentinel.reliability.control import fit_instability_cutoff

    predictor, ctx = train_frozen_predictor(DATA)
    train_mask = ctx["masks"]["train"] & ctx["valid_mask"]
    clean_train_feats = build_raw_features(
        ctx["pressure"][train_mask], ctx["flow"][train_mask], ctx["level"][train_mask]
    )
    instability_cutoff = fit_instability_cutoff(clean_train_feats)
    return predictor, ctx, instability_cutoff


def test_agent_pipeline_exactly_matches_frozen_system_c_on_a_real_data_slice(frozen_setup):
    from wds_sentinel.agents.evidence_agent import EvidenceAgent
    from wds_sentinel.agents.prediction_agent import PredictionAgent
    from wds_sentinel.agents.reasoning_agent import KnowledgeReasoningAgent
    from wds_sentinel.agents.reliability_agent import ReliabilityAgent
    from wds_sentinel.agents.supervisory_agent import SupervisoryDecisionAgent
    from wds_sentinel.evidence.adapters import DataFrameEvidenceAdapter, PredictionEvidenceAdapter
    from wds_sentinel.knowledge.kbs import KnowledgeBase
    from wds_sentinel.prediction.degradation import DegradedStreams, missing_sensor_subset
    from wds_sentinel.reasoning.rules import corroboration_count
    from wds_sentinel.reliability.systems import run_systems

    predictor, ctx, instability_cutoff = frozen_setup
    pressure, flow, level = ctx["pressure"], ctx["flow"], ctx["level"]
    val_mask = ctx["masks"]["validation"]

    # First 2,000 validation timestamps plus a short CLEAN history buffer.
    # This is genuinely bounded computationally: run_systems() sees only
    # ~2,030 rows, not the full year. The buffer exceeds the longest
    # frozen feature lookback (24 five-minute samples).
    val_slice_idx = pressure.index[val_mask][:2000]
    first_pos = pressure.index.get_loc(val_slice_idx[0])
    history_idx = pressure.index[max(0, first_pos - 30):first_pos]
    bounded_idx = history_idx.append(val_slice_idx)

    degraded_val = missing_sensor_subset(
        pressure.loc[val_slice_idx], flow.loc[val_slice_idx], level.loc[val_slice_idx]
    )
    full_pressure = pressure.loc[bounded_idx].copy()
    full_flow = flow.loc[bounded_idx].copy()
    full_level = level.loc[bounded_idx].copy()
    full_pressure.loc[val_slice_idx] = degraded_val.pressure
    full_flow.loc[val_slice_idx] = degraded_val.flow
    full_level.loc[val_slice_idx] = degraded_val.level
    degraded_full = DegradedStreams(full_pressure, full_flow, full_level, "missing_sensor_subset")

    frozen_result = run_systems(predictor, degraded_full, instability_cutoff)
    eval_idx = frozen_result.decision_C.index[frozen_result.decision_C.index.isin(val_slice_idx)]
    frozen_C = frozen_result.decision_C.loc[eval_idx]

    corrob = corroboration_count(frozen_result.features)
    evidence_agent = EvidenceAgent(
        adapters=[
            DataFrameEvidenceAdapter("pressure", "pressure", degraded_full.pressure.loc[eval_idx]),
            DataFrameEvidenceAdapter("flow", "flow", degraded_full.flow.loc[eval_idx]),
            DataFrameEvidenceAdapter("level", "level", degraded_full.level.loc[eval_idx]),
            PredictionEvidenceAdapter("predictor", frozen_result.scores.loc[eval_idx]),
        ],
        evidence_quality=frozen_result.evidence_quality,
    )
    prediction_agent = PredictionAgent(scores=frozen_result.scores, threshold=predictor.threshold)
    reasoning_agent = KnowledgeReasoningAgent(kb=KnowledgeBase(), corroboration=corrob)
    reliability_agent = ReliabilityAgent(instability_cutoff=instability_cutoff)
    supervisor = SupervisoryDecisionAgent(evidence_agent, prediction_agent, reasoning_agent, reliability_agent)

    agent_C = pd.Series({t: supervisor.decide(t).decision for t in eval_idx})

    assert len(eval_idx) > 0
    mismatches = (agent_C.values != frozen_C.loc[eval_idx].values).sum()
    assert mismatches == 0, f"{mismatches} mismatches on a {len(eval_idx)}-row real-data slice"
