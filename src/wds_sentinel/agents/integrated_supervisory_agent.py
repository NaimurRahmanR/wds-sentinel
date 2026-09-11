"""Integrated WDS + independent hazard supervisory orchestration.

The WDS and precipitation sources are intentionally independent: the
hazard stream is not geographically or causally linked to L-Town. The
integrated record therefore keeps the WDS operational decision unchanged
and adds a monitoring-level hazard context. This demonstrates real
heterogeneous evidence integration without inventing rainfall -> hydraulic
failure causality.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from wds_sentinel.agents.decision_record import DecisionRecord
from wds_sentinel.agents.hazard_agent import HazardAgent
from wds_sentinel.agents.messages import HazardMessage
from wds_sentinel.agents.supervisory_agent import SupervisoryDecisionAgent, SupervisoryExecutionTrace
from wds_sentinel.knowledge.kbs import HAZARD_RULES, KnowledgeBase, ReasoningTrace

HAZARD_STATES = {"NORMAL", "ELEVATED", "UNKNOWN"}
RELIABILITY_STATES = {"RELIABLE", "UNRELIABLE"}
INTEGRATED_ACTIONS = {
    "ROUTINE",
    "HAZARD_WATCH",
    "ALERT",
    "ALERT_ELEVATED_HAZARD_CONTEXT",
    "ABSTAIN",
    "ESCALATE",
}


def hazard_state_label(hazard_message: HazardMessage) -> str:
    if not hazard_message.available or hazard_message.elevated is None:
        return "UNKNOWN"
    return "ELEVATED" if hazard_message.elevated else "NORMAL"


def evidence_reliability_state(wds_record: DecisionRecord) -> str:
    return "UNRELIABLE" if wds_record.reliability_reasons else "RELIABLE"


def integrated_supervisory_action(wds_decision: str, hazard_state: str) -> str:
    """Combine an unchanged WDS decision with independent hazard context.

    ABSTAIN/ESCALATE from the WDS reliability gate always pass through.
    ``HAZARD_WATCH`` is a contextual monitoring action only and never a
    claim that precipitation caused or predicts a WDS failure.
    """
    if wds_decision in ("ABSTAIN", "ESCALATE"):
        return wds_decision
    if wds_decision == "ALERT":
        return "ALERT_ELEVATED_HAZARD_CONTEXT" if hazard_state == "ELEVATED" else "ALERT"
    if wds_decision == "NO_ALERT":
        return "HAZARD_WATCH" if hazard_state == "ELEVATED" else "ROUTINE"
    raise ValueError(f"unrecognised WDS decision: {wds_decision!r}")


@dataclass(frozen=True)
class IntegratedDecisionRecord:
    wds_timestamp: pd.Timestamp
    hazard_timestamp: pd.Timestamp
    wds_execution: SupervisoryExecutionTrace
    hazard_message: HazardMessage
    hazard_trace: ReasoningTrace
    wds_operational_state: str
    hazard_state: str
    evidence_reliability_state: str
    integrated_action: str
    combined_explanation: str

    def __post_init__(self):
        if self.hazard_state not in HAZARD_STATES:
            raise ValueError(f"hazard_state must be one of {HAZARD_STATES}, got {self.hazard_state!r}")
        if self.evidence_reliability_state not in RELIABILITY_STATES:
            raise ValueError(f"evidence_reliability_state must be one of {RELIABILITY_STATES}")
        if self.integrated_action not in INTEGRATED_ACTIONS:
            raise ValueError(f"integrated_action must be one of {INTEGRATED_ACTIONS}, got {self.integrated_action!r}")

    @property
    def timestamp(self) -> pd.Timestamp:
        """Backward-compatible alias for the WDS decision timestamp."""
        return self.wds_timestamp

    @property
    def wds_record(self) -> DecisionRecord:
        return self.wds_execution.record

    @property
    def decision(self) -> str:
        return self.wds_execution.record.decision


def generate_combined_explanation(
    wds_record: DecisionRecord,
    hazard_message: HazardMessage,
    hazard_trace: ReasoningTrace,
    hazard_state: str,
    reliability_state: str,
    action: str,
) -> str:
    parts = [
        f"Knowledge/Reasoning Agent (WDS): {wds_record.explanation}",
        f"Reliability Agent: evidence reliability = {reliability_state}"
        + (f" ({'; '.join(wds_record.reliability_reasons)})" if wds_record.reliability_reasons else "."),
    ]
    if hazard_message.available:
        parts.append(
            "Hazard Agent: independent CWA/CODIS precipitation context "
            f"= {hazard_state}. Knowledge Agent (hazard): rules fired: "
            f"{', '.join(hazard_trace.fired_rules) if hazard_trace.fired_rules else 'none'}."
        )
    else:
        parts.append(
            "Hazard Agent: hazard evidence unavailable for the selected hazard date "
            f"(hazard_state = UNKNOWN). Knowledge Agent (hazard): rules fired: "
            f"{', '.join(hazard_trace.fired_rules) if hazard_trace.fired_rules else 'none'}."
        )
    parts.append(
        f"Supervisory Agent: integrated action = {action}. Hazard context is independent of the "
        "BattLeDIM WDS evidence and does not alter the WDS operational state or imply a hydraulic causal link."
    )
    return " ".join(parts)


@dataclass
class IntegratedSupervisoryAgent:
    wds_supervisor: SupervisoryDecisionAgent
    hazard_agent: HazardAgent

    def decide(
        self,
        wds_timestamp: pd.Timestamp,
        hazard_timestamp: pd.Timestamp | None = None,
    ) -> IntegratedDecisionRecord:
        """Execute both real evidence chains through one orchestrator.

        ``hazard_timestamp`` may differ from ``wds_timestamp`` because the
        bundled demonstration streams are deliberately independent. When
        omitted, both agents use the same timestamp for backwards
        compatibility with synthetic/unit tests.
        """
        hazard_timestamp = wds_timestamp if hazard_timestamp is None else hazard_timestamp
        wds_execution = self.wds_supervisor.execute(wds_timestamp)
        wds_record = wds_execution.record
        hazard_msg = self.hazard_agent.collect(hazard_timestamp)

        hazard_trace = KnowledgeBase(HAZARD_RULES).evaluate(
            {"hazard_elevated": hazard_msg.elevated, "hazard_available": hazard_msg.available}
        )
        h_state = hazard_state_label(hazard_msg)
        r_state = evidence_reliability_state(wds_record)
        action = integrated_supervisory_action(wds_record.decision, h_state)
        explanation = generate_combined_explanation(
            wds_record, hazard_msg, hazard_trace, h_state, r_state, action
        )

        return IntegratedDecisionRecord(
            wds_timestamp=wds_timestamp,
            hazard_timestamp=hazard_timestamp,
            wds_execution=wds_execution,
            hazard_message=hazard_msg,
            hazard_trace=hazard_trace,
            wds_operational_state=wds_record.decision,
            hazard_state=h_state,
            evidence_reliability_state=r_state,
            integrated_action=action,
            combined_explanation=explanation,
        )
