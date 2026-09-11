"""Supervisory orchestration for the WDS decision path.

The agent executes Evidence -> Prediction -> Knowledge/Reasoning ->
Reliability in a fixed order. ``decide()`` preserves the original API and
returns only the final DecisionRecord. ``execute()`` exposes the typed
intermediate messages as an auditable execution trace without changing
any decision logic.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from wds_sentinel.agents.decision_record import DecisionRecord, generate_explanation
from wds_sentinel.agents.evidence_agent import EvidenceAgent
from wds_sentinel.agents.messages import (
    EvidenceMessage,
    PredictionMessage,
    ReasoningMessage,
    ReliabilityMessage,
)
from wds_sentinel.agents.prediction_agent import PredictionAgent
from wds_sentinel.agents.reasoning_agent import KnowledgeReasoningAgent
from wds_sentinel.agents.reliability_agent import ReliabilityAgent


@dataclass(frozen=True)
class SupervisoryExecutionTrace:
    timestamp: pd.Timestamp
    evidence_message: EvidenceMessage
    prediction_message: PredictionMessage
    reasoning_message: ReasoningMessage
    reliability_message: ReliabilityMessage
    record: DecisionRecord


@dataclass
class SupervisoryDecisionAgent:
    evidence_agent: EvidenceAgent
    prediction_agent: PredictionAgent
    reasoning_agent: KnowledgeReasoningAgent
    reliability_agent: ReliabilityAgent

    def execute(self, timestamp: pd.Timestamp) -> SupervisoryExecutionTrace:
        evidence_msg = self.evidence_agent.collect(timestamp)
        prediction_msg = self.prediction_agent.predict(timestamp)
        reasoning_msg = self.reasoning_agent.reason(prediction_msg, evidence_msg)
        reliability_msg = self.reliability_agent.assess(evidence_msg, reasoning_msg)

        fired_rules = tuple(reasoning_msg.trace.fired_rules)
        reliability_reasons = tuple(reliability_msg.reasons)
        evidence_refs = evidence_msg.bundle.items

        explanation = generate_explanation(
            timestamp=timestamp,
            decision=reliability_msg.final_decision,
            fired_rules=fired_rules,
            reliability_reasons=reliability_reasons,
            evidence_refs=evidence_refs,
        )

        record = DecisionRecord(
            timestamp=timestamp,
            decision=reliability_msg.final_decision,
            evidence_refs=evidence_refs,
            fired_rules=fired_rules,
            reliability_reasons=reliability_reasons,
            explanation=explanation,
        )
        return SupervisoryExecutionTrace(
            timestamp=timestamp,
            evidence_message=evidence_msg,
            prediction_message=prediction_msg,
            reasoning_message=reasoning_msg,
            reliability_message=reliability_msg,
            record=record,
        )

    def decide(self, timestamp: pd.Timestamp) -> DecisionRecord:
        return self.execute(timestamp).record
