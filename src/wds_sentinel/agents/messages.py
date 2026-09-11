"""
Typed message contracts between agents. Each agent takes one of these in
and produces one out — the contract is the dataclass, not an implicit
convention, so a routing mistake (e.g. passing a PredictionMessage where
an EvidenceMessage is expected) fails immediately and visibly rather than
producing a silently wrong decision.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from wds_sentinel.evidence.schema import EvidenceBundle
from wds_sentinel.knowledge.kbs import ReasoningTrace


@dataclass(frozen=True)
class EvidenceMessage:
    """Produced by the Evidence Agent."""
    timestamp: pd.Timestamp
    bundle: EvidenceBundle
    missingness: float
    availability: float
    disagreement: float
    instability: float


@dataclass(frozen=True)
class PredictionMessage:
    """Produced by the Prediction Agent."""
    timestamp: pd.Timestamp
    score: float
    threshold: float
    predictor_version: str


@dataclass(frozen=True)
class ReasoningMessage:
    """Produced by the Knowledge/Reasoning Agent."""
    timestamp: pd.Timestamp
    trace: ReasoningTrace
    decision: str  # "ALERT" | "NO_ALERT" — reasoning-layer decision, pre-reliability


@dataclass(frozen=True)
class ReliabilityMessage:
    """Produced by the Reliability Agent."""
    timestamp: pd.Timestamp
    unreliable: bool
    reasons: list[str] = field(default_factory=list)
    final_decision: str = "NO_ALERT"  # "NO_ALERT" | "ALERT" | "ABSTAIN" | "ESCALATE"


@dataclass(frozen=True)
class HazardMessage:
    """Produced by the Hazard Agent. Independent of the WDS evidence
    chain — carries its own evidence item(s), availability, and elevated
    flag (True/False/None-if-undefined), never a WDS decision value."""
    timestamp: pd.Timestamp
    evidence: tuple  # tuple[EvidenceState, ...]
    available: bool
    elevated: bool | None  # None means undefined for this timestep (missing input)
