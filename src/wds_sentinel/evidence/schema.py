"""
Standardised evidence representation. Every piece of information that
flows between agents — a raw sensor reading, a predictor score, a
reliability signal — is carried as one of these, so downstream code never
has to special-case "is this a sensor value or a model output".

This is the first real content in evidence/ — deliberately minimal and
frozen-logic-agnostic: it carries values, it doesn't compute them.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class EvidenceState:
    source: str          # e.g. "pressure_sensor_n1", "frozen_predictor_v1", "reliability_monitor"
    modality: str         # "pressure" | "flow" | "level" | "prediction" | "reliability"
    timestamp: pd.Timestamp
    value: float | None   # raw reading, score, or signal value; None if unavailable
    availability: bool    # True iff this value was actually observed (not imputed/missing)
    quality: float         # 0.0-1.0 confidence in this specific evidence item
    provenance: str = ""   # free-text: how this was produced (e.g. "raw SCADA", "ffill-imputed")

    def __post_init__(self):
        if not (0.0 <= self.quality <= 1.0):
            raise ValueError(f"quality must be in [0,1], got {self.quality}")


@dataclass(frozen=True)
class EvidenceBundle:
    """All EvidenceState items collected for one decision timestep."""
    timestamp: pd.Timestamp
    items: tuple[EvidenceState, ...] = field(default_factory=tuple)

    def by_modality(self, modality: str) -> tuple[EvidenceState, ...]:
        return tuple(e for e in self.items if e.modality == modality)

    def availability_fraction(self) -> float:
        if not self.items:
            return 0.0
        return sum(1 for e in self.items if e.availability) / len(self.items)
