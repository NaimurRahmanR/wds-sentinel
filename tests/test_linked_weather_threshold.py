import pandas as pd

from wds_sentinel.linked_weather.experiment import calibrate_extreme_rain_threshold


def test_extreme_rain_threshold_uses_positive_training_days_only():
    idx = pd.date_range("2021-01-01", periods=40 * 24, freq="h", tz="Europe/Rome")
    rain = pd.Series(0.0, index=idx)
    # 30 wet days with increasing daily totals 1..30 mm.
    for day in range(30):
        rain.loc[idx.normalize() == idx[day * 24].normalize()] = (day + 1) / 24.0
    weather = pd.DataFrame({"Rain": rain, "Temperature": 0.0, "Humidity": 0.0, "Windspeed": 0.0})
    threshold = calibrate_extreme_rain_threshold(weather)
    expected = pd.Series(range(1, 31), dtype=float).quantile(0.95)
    assert abs(threshold - expected) < 1e-9
