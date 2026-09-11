"""Regression/faithfulness tests for the corrected integrated experiment:
terminology, explicit states, action semantics, and four-agent trace
contribution."""
import pathlib
import re

import pandas as pd
import pytest

from wds_sentinel.agents.integrated_supervisory_agent import (
    evidence_reliability_state,
    hazard_state_label,
    integrated_supervisory_action,
)
from wds_sentinel.agents.messages import HazardMessage


def test_no_borrowed_standard_index_name_anywhere_in_source():
    """ETCCDI/R95p naming was explicitly retracted — must not reappear."""
    src_root = pathlib.Path(__file__).resolve().parents[1] / "src"
    forbidden = re.compile(r"ETCCDI|R95p", re.IGNORECASE)
    hits = []
    for f in src_root.rglob("*.py"):
        text = f.read_text()
        if forbidden.search(text):
            hits.append(str(f))
    assert hits == [], f"forbidden terminology found in: {hits}"


def test_no_unverified_specific_event_attribution_in_source_or_docs():
    """The September-cluster-as-typhoon claim was never verified and must
    not be asserted anywhere in the shipped project."""
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    forbidden = re.compile(r"typhoon", re.IGNORECASE)
    hits = []
    for pattern in ("src/**/*.py", "scripts/**/*.py", "data/**/*.md", "results/**/*.md"):
        for f in repo_root.glob(pattern):
            if f.is_file() and forbidden.search(f.read_text(errors="ignore")):
                hits.append(str(f))
    assert hits == [], f"unverified event attribution found in: {hits}"


def _msg(available, elevated):
    return HazardMessage(timestamp=pd.Timestamp("2020-01-01"), evidence=(), available=available, elevated=elevated)


def test_hazard_state_label_mapping():
    assert hazard_state_label(_msg(True, True)) == "ELEVATED"
    assert hazard_state_label(_msg(True, False)) == "NORMAL"
    assert hazard_state_label(_msg(False, None)) == "UNKNOWN"


@pytest.mark.parametrize(
    "wds_decision,hazard_state,expected",
    [
        ("NO_ALERT", "NORMAL", "ROUTINE"),
        ("NO_ALERT", "ELEVATED", "HAZARD_WATCH"),
        ("NO_ALERT", "UNKNOWN", "ROUTINE"),
        ("ALERT", "NORMAL", "ALERT"),
        ("ALERT", "ELEVATED", "ALERT_ELEVATED_HAZARD_CONTEXT"),
        ("ALERT", "UNKNOWN", "ALERT"),
        ("ABSTAIN", "NORMAL", "ABSTAIN"),
        ("ABSTAIN", "ELEVATED", "ABSTAIN"),
        ("ESCALATE", "NORMAL", "ESCALATE"),
        ("ESCALATE", "ELEVATED", "ESCALATE"),
    ],
)
def test_integrated_action_matches_specified_semantics(wds_decision, hazard_state, expected):
    assert integrated_supervisory_action(wds_decision, hazard_state) == expected


def test_reliability_gate_passthrough_never_reinterpreted_by_hazard():
    """The existing reliability gate's ABSTAIN/ESCALATE must never be
    overridden or softened by hazard state, in either direction."""
    for hazard_state in ("NORMAL", "ELEVATED", "UNKNOWN"):
        assert integrated_supervisory_action("ABSTAIN", hazard_state) == "ABSTAIN"
        assert integrated_supervisory_action("ESCALATE", hazard_state) == "ESCALATE"


def test_evidence_reliability_state_reflects_wds_record_reasons():
    from wds_sentinel.agents.decision_record import DecisionRecord

    t = pd.Timestamp("2020-01-01")
    reliable_record = DecisionRecord(t, "ALERT", (), ("R1",), (), "x")
    unreliable_record = DecisionRecord(t, "ESCALATE", (), ("R1",), ("missingness too high",), "x")
    assert evidence_reliability_state(reliable_record) == "RELIABLE"
    assert evidence_reliability_state(unreliable_record) == "UNRELIABLE"
