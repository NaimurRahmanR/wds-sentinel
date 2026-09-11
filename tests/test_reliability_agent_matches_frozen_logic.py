import pandas as pd

from wds_sentinel.agents.evidence_agent import EvidenceAgent
from wds_sentinel.agents.messages import ReasoningMessage
from wds_sentinel.agents.reliability_agent import ReliabilityAgent
from wds_sentinel.evidence.adapters import DataFrameEvidenceAdapter
from wds_sentinel.knowledge.kbs import ReasoningTrace
from wds_sentinel.reliability.control import is_unreliable, reliability_aware_decision


def test_reliability_agent_final_decision_matches_frozen_function_exactly():
    idx = pd.date_range("2020-01-01", periods=6, freq="5min")
    eq = pd.DataFrame(
        {
            "missingness": [0.0, 0.5, 0.0, 0.0, 0.0, 0.5],
            "availability": [1.0, 0.5, 1.0, 1.0, 1.0, 0.5],
            "disagreement": [0.0, 0.0, 5.0, 0.0, 0.0, 0.0],
            "instability": [0.1, 0.1, 0.1, 10.0, 0.1, 10.0],
        },
        index=idx,
    )
    base_decisions = pd.Series(
        ["ALERT", "ALERT", "NO_ALERT", "NO_ALERT", "ALERT", "NO_ALERT"], index=idx
    )
    cutoff = 1.0

    # Frozen function's answer
    frozen_result = reliability_aware_decision(base_decisions, eq, cutoff)

    # Agent's answer, one timestamp at a time, via the typed message contract
    dummy_adapter = DataFrameEvidenceAdapter(name="x", modality="pressure", raw=pd.DataFrame({"n1": [1.0] * 6}, index=idx))
    evidence_agent = EvidenceAgent(adapters=[dummy_adapter], evidence_quality=eq)
    reliability_agent = ReliabilityAgent(instability_cutoff=cutoff)

    for t in idx:
        ev_msg = evidence_agent.collect(t)
        reasoning_msg = ReasoningMessage(timestamp=t, trace=ReasoningTrace(), decision=base_decisions.loc[t])
        rel_msg = reliability_agent.assess(ev_msg, reasoning_msg)
        assert rel_msg.final_decision == frozen_result.loc[t], f"mismatch at {t}"
        assert rel_msg.unreliable == bool(is_unreliable(eq, cutoff).loc[t])
