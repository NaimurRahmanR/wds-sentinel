"""
Adapter interface so additional evidence sources (a new sensor type, a
future ERA5-Land feed, etc.) can plug in without the Evidence Agent or
anything downstream changing. An adapter's only contract is
`fetch(timestamp) -> tuple[EvidenceState, ...]`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from wds_sentinel.evidence.schema import EvidenceState


class EvidenceSourceAdapter(Protocol):
    name: str
    modality: str

    def fetch(self, timestamp: pd.Timestamp) -> tuple[EvidenceState, ...]: ...


@dataclass
class DataFrameEvidenceAdapter:
    """Wraps a single (possibly degraded) sensor DataFrame as an evidence
    source. availability = the reading is not NaN at this timestamp;
    quality = 1.0 if available else 0.0 — deliberately the simplest
    possible, non-invented mapping; anything richer belongs in the
    Reliability Agent, not here."""
    name: str
    modality: str
    raw: pd.DataFrame

    def fetch(self, timestamp: pd.Timestamp) -> tuple[EvidenceState, ...]:
        if timestamp not in self.raw.index:
            return tuple(
                EvidenceState(
                    source=f"{self.name}:{col}", modality=self.modality, timestamp=timestamp,
                    value=None, availability=False, quality=0.0, provenance="timestamp not in source",
                )
                for col in self.raw.columns
            )
        row = self.raw.loc[timestamp]
        out = []
        for col in self.raw.columns:
            val = row[col]
            available = pd.notna(val)
            out.append(
                EvidenceState(
                    source=f"{self.name}:{col}", modality=self.modality, timestamp=timestamp,
                    value=float(val) if available else None,
                    availability=bool(available),
                    quality=1.0 if available else 0.0,
                    provenance="raw SCADA reading" if available else "missing at source",
                )
            )
        return tuple(out)


@dataclass
class PredictionEvidenceAdapter:
    """Wraps the frozen predictor's score series as an evidence source —
    same interface as any sensor adapter, so the Reasoning Agent doesn't
    need to know a prediction is structurally different from a sensor
    reading."""
    name: str
    scores: pd.Series
    predictor_version: str = "frozen_predictor_v1"

    modality: str = "prediction"

    def fetch(self, timestamp: pd.Timestamp) -> tuple[EvidenceState, ...]:
        if timestamp not in self.scores.index:
            return (
                EvidenceState(
                    source=self.name, modality="prediction", timestamp=timestamp,
                    value=None, availability=False, quality=0.0,
                    provenance=f"{self.predictor_version}: timestamp not scored",
                ),
            )
        score = self.scores.loc[timestamp]
        available = pd.notna(score)
        return (
            EvidenceState(
                source=self.name, modality="prediction", timestamp=timestamp,
                value=float(score) if available else None,
                availability=bool(available),
                quality=1.0 if available else 0.0,
                provenance=self.predictor_version,
            ),
        )


@dataclass
class HazardEvidenceAdapter:
    """Wraps a real, independent hydro-meteorological hazard indicator
    (e.g. the Taiwan CWA precipitation-derived hazard flag) as an evidence
    source. This is explicitly NOT geographically or causally linked to
    L-Town/BattLeDIM — it demonstrates the architecture processing a
    genuinely independent extreme-event evidence stream through the same
    EvidenceState formalism, not a physically-connected hazard signal for
    this specific WDS.

    availability is False, and quality 0.0, wherever the underlying
    elevated_flag is undefined (pandas.NA — all antecedent inputs missing
    for that day), never silently coerced to a "normal" reading.
    """
    name: str
    elevated_flag: pd.Series  # boolean or pandas.NA, from hazard.detector.hazard_elevated_flag
    raw_value: pd.Series        # the underlying precip_mm (or antecedent sum) for provenance/value
    provenance_note: str = "Taiwan CWA CODIS station 466920 (Raingel/historical_weather rebuild)"

    modality: str = "hazard"

    def fetch(self, timestamp: pd.Timestamp) -> tuple[EvidenceState, ...]:
        if timestamp not in self.elevated_flag.index:
            return (
                EvidenceState(
                    source=self.name, modality="hazard", timestamp=timestamp,
                    value=None, availability=False, quality=0.0,
                    provenance=f"{self.provenance_note}: timestamp not covered",
                ),
            )
        flag = self.elevated_flag.loc[timestamp]
        raw = self.raw_value.loc[timestamp] if timestamp in self.raw_value.index else None
        defined = flag is not pd.NA and pd.notna(flag)
        return (
            EvidenceState(
                source=self.name, modality="hazard", timestamp=timestamp,
                value=(1.0 if bool(flag) else 0.0) if defined else None,
                availability=bool(defined) and pd.notna(raw),
                quality=1.0 if (defined and pd.notna(raw)) else 0.0,
                provenance=self.provenance_note,
            ),
        )
