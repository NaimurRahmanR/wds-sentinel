"""
Evidence Agent. Responsibility: gather raw evidence from every adapter and
package it, together with observable evidence-quality signals, into one
typed EvidenceMessage. It does not interpret the evidence (no thresholds,
no decisions here) — that's the Reasoning and Reliability Agents' job.

The missingness/disagreement/instability values are read from a
precomputed evidence-quality frame (wds_sentinel.reliability.control.
compute_evidence_quality) rather than recomputed here, so this agent
cannot silently drift from the already-validated reliability experiment's
definition of those signals.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from wds_sentinel.agents.messages import EvidenceMessage
from wds_sentinel.evidence.adapters import EvidenceSourceAdapter
from wds_sentinel.evidence.schema import EvidenceBundle


@dataclass
class EvidenceAgent:
    adapters: list[EvidenceSourceAdapter]
    evidence_quality: pd.DataFrame  # columns: missingness, availability, disagreement, instability

    def collect(self, timestamp: pd.Timestamp) -> EvidenceMessage:
        items = []
        for adapter in self.adapters:
            items.extend(adapter.fetch(timestamp))
        bundle = EvidenceBundle(timestamp=timestamp, items=tuple(items))

        if timestamp in self.evidence_quality.index:
            row = self.evidence_quality.loc[timestamp]
            missingness, availability = float(row["missingness"]), float(row["availability"])
            disagreement, instability = float(row["disagreement"]), float(row["instability"])
        else:
            # no precomputed quality row: treat as maximally unreliable
            # rather than silently assuming everything is fine
            missingness, availability, disagreement, instability = 1.0, 0.0, float("inf"), float("inf")

        return EvidenceMessage(
            timestamp=timestamp, bundle=bundle, missingness=missingness,
            availability=availability, disagreement=disagreement, instability=instability,
        )
