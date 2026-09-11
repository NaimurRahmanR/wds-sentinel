"""
Explicit knowledge-based reasoning. Every rule is identifiable (has a
fixed rule_id), deterministic (pure function of facts), and every
evaluation produces a full trace — which rules fired, what facts they
derived, and any contradictions noticed along the way.

Rule logic is NOT reinvented here: R2_CORROBORATED and R3_HYBRID_DECISION
call the exact same functions already frozen in reasoning/rules.py
(corroboration_count, the ALERT-iff-predictor-AND-corroborated logic).
This module adds identifiability and a trace on top of existing,
already-validated logic — it does not introduce new hydraulic thresholds
or new "expert" rules.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class Fact:
    name: str
    value: Any


@dataclass(frozen=True)
class Rule:
    rule_id: str
    description: str
    # pure function: facts dict -> Fact to add, or None if it doesn't fire
    apply: Callable[[dict[str, Any]], Fact | None]


@dataclass
class ReasoningTrace:
    fired_rules: list[str] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)
    contradictions: list[str] = field(default_factory=list)


def _r1_predictor_alert(facts: dict) -> Fact | None:
    if facts.get("predictor_score") is None:
        return None
    fired = facts["predictor_score"] >= facts["predictor_threshold"]
    return Fact("predictor_alert", bool(fired))


def _r2_corroborated(facts: dict) -> Fact | None:
    # Reuses reasoning/rules.py's exact corroboration convention: at least
    # one sensor below the standard -2 z cutoff, already computed upstream
    # and passed in as a fact (not recomputed here).
    if facts.get("corroboration_count") is None:
        return None
    return Fact("corroborated", facts["corroboration_count"] >= 1)


def _r3_hybrid_decision(facts: dict) -> Fact | None:
    # Identical to reasoning.rules.hybrid_decision's per-timestep logic,
    # expressed as an explicit, traceable rule.
    if "predictor_alert" not in facts or "corroborated" not in facts:
        return None
    decision = "ALERT" if (facts["predictor_alert"] and facts["corroborated"]) else "NO_ALERT"
    return Fact("hybrid_decision", decision)


def _r4_contradiction_check(facts: dict) -> Fact | None:
    # Not a decision rule — records an epistemic tension: the predictor
    # thinks something is wrong but no independent sensor corroborates
    # it. This is not treated as an error; R3's AND-based resolution
    # (require corroboration) is the documented, deterministic policy for
    # resolving it, and this rule exists so that resolution is visible in
    # the trace rather than silently applied.
    if "predictor_alert" not in facts or "corroborated" not in facts:
        return None
    if facts["predictor_alert"] and not facts["corroborated"]:
        return Fact("contradiction:predictor_alert_without_corroboration", True)
    return None


DEFAULT_RULES: list[Rule] = [
    Rule("R1_PREDICTOR_ALERT", "predictor score crosses the frozen threshold", _r1_predictor_alert),
    Rule("R2_CORROBORATED", "at least one sensor independently anomalous (z < -2)", _r2_corroborated),
    Rule("R3_HYBRID_DECISION", "ALERT iff predictor_alert AND corroborated (reasoning/rules.py)", _r3_hybrid_decision),
    Rule("R4_CONTRADICTION_CHECK", "flags predictor_alert without corroboration as a recorded tension", _r4_contradiction_check),
]


# --- Hazard-evidence rules (additive only — DEFAULT_RULES above and every
# existing KnowledgeBase() call with no arguments are completely
# unaffected; these rules only run in a KnowledgeBase explicitly
# constructed with DEFAULT_RULES + HAZARD_RULES, used solely by the
# hazard-integration scenarios). They read a precomputed hazard_elevated
# fact (from hazard.detector.hazard_elevated_flag, already computed
# upstream) and never touch WDS facts or the hybrid_decision output — no
# invented causal link between rainfall and the WDS decision. ---

def _r5_hazard_elevated(facts: dict) -> Fact | None:
    if facts.get("hazard_elevated") is True:
        return Fact("hazard_context_elevated", True)
    return None


def _r6_hazard_normal(facts: dict) -> Fact | None:
    if facts.get("hazard_elevated") is False:
        return Fact("hazard_context_normal", True)
    return None


def _r7_hazard_evidence_degraded(facts: dict) -> Fact | None:
    # Fires when the hazard evidence itself is unavailable/undefined for
    # this timestep (missing precipitation input) — a statement about
    # THAT evidence source's own quality, not about the WDS decision.
    if facts.get("hazard_available") is False:
        return Fact("hazard_evidence_degraded", True)
    return None


HAZARD_RULES: list[Rule] = [
    Rule("R5_HAZARD_ELEVATED", "records an elevated real-hazard context (does not alter the WDS decision)", _r5_hazard_elevated),
    Rule("R6_HAZARD_NORMAL", "records a normal real-hazard context", _r6_hazard_normal),
    Rule("R7_HAZARD_EVIDENCE_DEGRADED", "flags the hazard evidence source itself as unavailable this timestep", _r7_hazard_evidence_degraded),
]


class KnowledgeBase:
    def __init__(self, rules: list[Rule] | None = None):
        self.rules = rules if rules is not None else list(DEFAULT_RULES)

    def evaluate(self, input_facts: dict[str, Any]) -> ReasoningTrace:
        """Deterministic: same input_facts always produces the same
        trace — rules run once, in a fixed order, each seeing only facts
        already established (its own inputs plus any earlier rule's
        output), never in a random or state-dependent order."""
        trace = ReasoningTrace(facts=dict(input_facts))
        for rule in self.rules:
            result = rule.apply(trace.facts)
            if result is not None:
                trace.fired_rules.append(rule.rule_id)
                trace.facts[result.name] = result.value
                if result.name.startswith("contradiction:"):
                    trace.contradictions.append(result.name)
        return trace
