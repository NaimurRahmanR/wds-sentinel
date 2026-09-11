import pandas as pd

from wds_sentinel.evidence.adapters import DataFrameEvidenceAdapter, PredictionEvidenceAdapter


def test_dataframe_adapter_reports_availability_from_nan_pattern():
    idx = pd.date_range("2020-01-01", periods=3, freq="5min")
    df = pd.DataFrame({"n1": [50.0, float("nan"), 52.0]}, index=idx)
    adapter = DataFrameEvidenceAdapter(name="pressure_stream", modality="pressure", raw=df)

    e0 = adapter.fetch(idx[0])[0]
    assert e0.availability is True and e0.value == 50.0 and e0.quality == 1.0

    e1 = adapter.fetch(idx[1])[0]
    assert e1.availability is False and e1.value is None and e1.quality == 0.0


def test_dataframe_adapter_missing_timestamp_is_unavailable_not_an_error():
    idx = pd.date_range("2020-01-01", periods=3, freq="5min")
    df = pd.DataFrame({"n1": [50.0, 51.0, 52.0]}, index=idx)
    adapter = DataFrameEvidenceAdapter(name="pressure_stream", modality="pressure", raw=df)
    far_future = pd.Timestamp("2030-01-01")
    result = adapter.fetch(far_future)
    assert len(result) == 1
    assert result[0].availability is False


def test_prediction_adapter_same_interface_as_dataframe_adapter():
    """Interchangeability: the Evidence Agent (or any caller) only relies
    on .fetch(timestamp) -> tuple[EvidenceState, ...] — this test swaps a
    sensor adapter for a prediction adapter and confirms both satisfy the
    exact same call contract."""
    idx = pd.date_range("2020-01-01", periods=3, freq="5min")
    df = pd.DataFrame({"n1": [50.0, 51.0, 52.0]}, index=idx)
    scores = pd.Series([0.1, 0.9, 0.2], index=idx)

    adapters = [
        DataFrameEvidenceAdapter(name="pressure_stream", modality="pressure", raw=df),
        PredictionEvidenceAdapter(name="predictor", scores=scores),
    ]

    def collect_all(t):
        out = []
        for a in adapters:
            out.extend(a.fetch(t))  # identical call, regardless of concrete adapter type
        return out

    items = collect_all(idx[1])
    assert len(items) == 2
    modalities = {e.modality for e in items}
    assert modalities == {"pressure", "prediction"}


def test_swapping_in_a_new_mock_adapter_requires_no_downstream_change():
    """A brand-new evidence source (e.g. a future ERA5-Land adapter) only
    needs to implement fetch() — nothing else in the pipeline changes."""
    from dataclasses import dataclass

    from wds_sentinel.evidence.schema import EvidenceState

    @dataclass
    class MockFutureAdapter:
        name: str
        modality: str

        def fetch(self, timestamp):
            return (EvidenceState(self.name, self.modality, timestamp, 1.0, True, 1.0, "mock"),)

    adapters = [MockFutureAdapter("era5_mock", "hazard")]

    def collect_all(t):
        out = []
        for a in adapters:
            out.extend(a.fetch(t))
        return out

    items = collect_all(pd.Timestamp("2020-01-01"))
    assert len(items) == 1
    assert items[0].modality == "hazard"
