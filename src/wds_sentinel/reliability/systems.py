"""
Ties frozen predictor + reasoning + reliability control together. Systems
A, B, and C are computed from the SAME `scores` and `features` objects —
not three separately-computed copies — so "A/B/C receive identical
underlying evidence" is true by construction (same Python object), not
just checked after the fact for equal values.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from wds_sentinel.prediction.degradation import DegradedStreams
from wds_sentinel.prediction.frozen_predictor import FrozenPredictor, build_raw_features
from wds_sentinel.reasoning.rules import direct_decision, hybrid_decision
from wds_sentinel.reliability.control import compute_evidence_quality, reliability_aware_decision


@dataclass
class SystemRunResult:
    scores: pd.Series
    features: pd.DataFrame
    evidence_quality: pd.DataFrame
    decision_A: pd.Series
    decision_B: pd.Series
    decision_C: pd.Series


def impute_for_predictor(degraded: DegradedStreams) -> DegradedStreams:
    """The one, shared, deterministic imputation policy every system's
    predictor input goes through — forward-fill then back-fill (handles
    any leading gap). Simple and generic, not privileged information: a
    forward-filled sensor carries no more information than "nothing new
    since last reading", which is the least a real deployed system could
    assume, and is applied identically regardless of which system
    (A, B, or C) will eventually consume the resulting score."""
    return DegradedStreams(
        pressure=degraded.pressure.ffill().bfill(),
        flow=degraded.flow.ffill().bfill(),
        level=degraded.level.ffill().bfill(),
        condition=degraded.condition,
    )


def run_systems(
    predictor: FrozenPredictor,
    degraded: DegradedStreams,
    instability_cutoff: float,
) -> SystemRunResult:
    imputed = impute_for_predictor(degraded)
    features = build_raw_features(imputed.pressure, imputed.flow, imputed.level)
    features = features.reindex(columns=predictor.feature_columns)

    valid = features.notna().all(axis=1)
    features = features[valid]

    scores = predictor.predict_scores(features)

    decision_A = direct_decision(scores, predictor.threshold)
    decision_B = hybrid_decision(scores, features, predictor.threshold)

    evidence_quality = compute_evidence_quality(
        degraded.pressure.loc[features.index],
        degraded.flow.loc[features.index],
        degraded.level.loc[features.index],
        features,
    )
    decision_C = reliability_aware_decision(decision_B, evidence_quality, instability_cutoff)

    return SystemRunResult(
        scores=scores,
        features=features,
        evidence_quality=evidence_quality,
        decision_A=decision_A,
        decision_B=decision_B,
        decision_C=decision_C,
    )
