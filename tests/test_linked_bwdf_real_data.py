from pathlib import Path

import pytest

from wds_sentinel.linked_weather import load_bwdf_dataset, evaluate_linked_weather_experiment

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "external" / "bwdf"


@pytest.mark.real_data
def test_bundled_bwdf_pair_and_frozen_linked_result():
    inflow = DATA_DIR / "InflowData.xlsx"
    weather = DATA_DIR / "WeatherData.xlsx"
    if not inflow.exists() or not weather.exists():
        pytest.skip("bundled BWDF workbooks are not present")

    ds = load_bwdf_dataset(DATA_DIR)
    assert len(ds.index) == 19679
    assert ds.index[0].strftime("%Y-%m-%d %H:%M") == "2021-01-01 00:00"
    assert ds.index[-1].strftime("%Y-%m-%d %H:%M") == "2023-03-31 23:00"

    result = evaluate_linked_weather_experiment(ds)
    assert result.rain_threshold_mm_day == pytest.approx(18.775, abs=1e-9)
    assert len(result.forecast_metrics) == 10
    assert int((result.forecast_metrics["weather_ridge_mae"] < result.forecast_metrics["naive_mae"]).sum()) == 9

    assert result.forecast_metrics.loc["DMA 2", "weather_ridge_mae"] == pytest.approx(1.472311, rel=1e-5)
    assert result.forecast_metrics.loc["DMA 3", "weather_ridge_mae"] == pytest.approx(1.333299, rel=1e-5)

    assert result.post_train_effects.loc["DMA 2", "difference_lps"] < 0
    assert result.post_train_effects.loc["DMA 3", "difference_lps"] < 0
    assert result.post_train_effects.loc["DMA 2", "bootstrap_95_hi"] < 0
    assert result.post_train_effects.loc["DMA 3", "bootstrap_95_hi"] < 0

    assert len(result.decisions) == 2
    for decision in result.decisions:
        assert decision.action == "WEATHER_LINKED_DEMAND_RESPONSE"
        assert "R8_EXTREME_RAIN" in decision.reasoning.fired_rules
        assert "R9_DEMAND_SUPPRESSION" in decision.reasoning.fired_rules
        assert "R10_LINKED_WEATHER_RESPONSE" in decision.reasoning.fired_rules
