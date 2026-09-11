import pandas as pd
import pytest

from wds_sentinel.evidence.schema import EvidenceBundle, EvidenceState


def test_evidence_state_requires_quality_in_unit_interval():
    t = pd.Timestamp("2020-01-01")
    EvidenceState("s1", "pressure", t, 50.0, True, 1.0)  # ok
    with pytest.raises(ValueError):
        EvidenceState("s1", "pressure", t, 50.0, True, 1.5)
    with pytest.raises(ValueError):
        EvidenceState("s1", "pressure", t, 50.0, True, -0.1)


def test_evidence_bundle_by_modality_filters_correctly():
    t = pd.Timestamp("2020-01-01")
    items = (
        EvidenceState("p1", "pressure", t, 50.0, True, 1.0),
        EvidenceState("f1", "flow", t, 100.0, True, 1.0),
        EvidenceState("p2", "pressure", t, 51.0, True, 1.0),
    )
    bundle = EvidenceBundle(timestamp=t, items=items)
    pressure_items = bundle.by_modality("pressure")
    assert len(pressure_items) == 2
    assert all(e.modality == "pressure" for e in pressure_items)


def test_evidence_bundle_availability_fraction():
    t = pd.Timestamp("2020-01-01")
    items = (
        EvidenceState("p1", "pressure", t, 50.0, True, 1.0),
        EvidenceState("p2", "pressure", t, None, False, 0.0),
        EvidenceState("p3", "pressure", t, 51.0, True, 1.0),
        EvidenceState("p4", "pressure", t, None, False, 0.0),
    )
    bundle = EvidenceBundle(timestamp=t, items=items)
    assert bundle.availability_fraction() == 0.5


def test_empty_bundle_availability_fraction_is_zero_not_error():
    t = pd.Timestamp("2020-01-01")
    bundle = EvidenceBundle(timestamp=t, items=())
    assert bundle.availability_fraction() == 0.0
