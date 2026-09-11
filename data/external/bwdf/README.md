# Battle of Water Demand Forecasting (BWDF) linked weather-demand data

This directory contains the two open supplementary files used by the physically linked extreme-weather validation:

- `InflowData.xlsx`
- `WeatherData.xlsx`

Source implementation: WaterFutures `wf4bwdf` / Alvisi et al. (2025), *Battle of Water Demand Forecasting*.

The data describe hourly net inflow for 10 DMAs in a real WDN in north-east Italy and weather observations from a station located within the same case-study WDN. The WaterFutures distribution states that the data are available under CC BY 4.0.

The validation protocol is pre-specified in `configs/linked_extreme_weather.yaml` and executed by `scripts/run_linked_extreme_weather_validation.py`.

Exact byte hashes and attribution are recorded in `data/PROVENANCE.md`.
