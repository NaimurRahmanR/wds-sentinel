"""Helpers for constructing the agent architecture from real experiment outputs."""
from __future__ import annotations

import pandas as pd

from wds_sentinel.agents.evidence_agent import EvidenceAgent
from wds_sentinel.agents.prediction_agent import PredictionAgent
from wds_sentinel.agents.reasoning_agent import KnowledgeReasoningAgent
from wds_sentinel.agents.reliability_agent import ReliabilityAgent
from wds_sentinel.agents.supervisory_agent import SupervisoryDecisionAgent
from wds_sentinel.evidence.adapters import DataFrameEvidenceAdapter
from wds_sentinel.knowledge.kbs import KnowledgeBase
from wds_sentinel.prediction.degradation import DegradedStreams
from wds_sentinel.reasoning.rules import corroboration_count
from wds_sentinel.reliability.systems import SystemRunResult


def build_supervisor_from_real_run(
    *,
    run: SystemRunResult,
    raw_streams: DegradedStreams,
    predictor_threshold: float,
    instability_cutoff: float,
) -> SupervisoryDecisionAgent:
    """Build the MAS/IDSS path from already-computed real experiment data.

    No scores, corroboration values, evidence-quality values, or sensor
    readings are hand-authored. Every downstream agent reads the same
    objects produced by ``run_systems`` and the corresponding raw stream.
    """
    adapters = [
        DataFrameEvidenceAdapter("battledim_pressure", "pressure", raw_streams.pressure),
        DataFrameEvidenceAdapter("battledim_flow", "flow", raw_streams.flow),
        DataFrameEvidenceAdapter("battledim_level", "level", raw_streams.level),
    ]
    return SupervisoryDecisionAgent(
        evidence_agent=EvidenceAgent(adapters=adapters, evidence_quality=run.evidence_quality),
        prediction_agent=PredictionAgent(scores=run.scores, threshold=predictor_threshold),
        reasoning_agent=KnowledgeReasoningAgent(
            kb=KnowledgeBase(), corroboration=corroboration_count(run.features)
        ),
        reliability_agent=ReliabilityAgent(instability_cutoff=instability_cutoff),
    )


def first_real_case(
    decisions: pd.Series,
    label: pd.Series,
    *,
    decision: str,
    label_value: int | None = None,
) -> pd.Timestamp:
    """Deterministically select the first timestamp meeting case criteria."""
    mask = decisions.eq(decision)
    if label_value is not None:
        aligned = label.reindex(decisions.index).fillna(0).astype(int)
        mask &= aligned.eq(label_value)
    matches = decisions.index[mask]
    if len(matches) == 0:
        raise RuntimeError(
            f"no real timestamp found for decision={decision!r}, label_value={label_value!r}"
        )
    return pd.Timestamp(matches[0])
