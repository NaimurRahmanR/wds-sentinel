"""
Hazard Agent. Responsibility: report the real, independent hazard
evidence (from HazardEvidenceAdapter) for a timestamp. It never touches
WDS evidence, the frozen predictor, or the WDS decision — it only reports
what the hazard stream itself says, including when that stream's own
evidence is unavailable.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from wds_sentinel.agents.messages import HazardMessage
from wds_sentinel.evidence.adapters import HazardEvidenceAdapter


@dataclass
class HazardAgent:
    adapter: HazardEvidenceAdapter

    def collect(self, timestamp: pd.Timestamp) -> HazardMessage:
        evidence = self.adapter.fetch(timestamp)
        item = evidence[0]
        elevated: bool | None
        if not item.availability:
            elevated = None
        else:
            elevated = bool(item.value)
        return HazardMessage(timestamp=timestamp, evidence=evidence, available=item.availability, elevated=elevated)
