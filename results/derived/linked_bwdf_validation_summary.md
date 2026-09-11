# Physically linked extreme-weather validation

## Frozen protocol

- Dataset: Battle of Water Demand Forecasting (BWDF), real WDN in northeast Italy.
- Inputs: hourly net inflow from 10 DMAs plus co-located rainfall, temperature, humidity and wind.
- Training end: 2022-07-24 23:00 Europe/Rome.
- Evaluation: W1, 2022-07-25 through 2022-07-31.
- Extreme-rain calibration: 95th percentile of positive daily rainfall totals using training data only.
- Pre-specified weather-sensitive DMAs: DMA 2 and DMA 3.
- Forecast models: same-hour previous-week baseline versus Ridge(alpha=1) with 168h/336h lags, calendar cycles and W1 weather. The original BWDF protocol supplied W1 weather as a perfect forecast.
- Interpretation: observational operational response; no rainfall-to-hydraulic-failure causal claim.

## Results

Training-calibrated extreme-rain threshold: **18.775 mm/day**.

Weather-aware Ridge has lower W1 MAE than the previous-week baseline in **9 of 10 DMAs**.

| DMA | Baseline MAE (L/s) | Weather Ridge MAE (L/s) | MAE change |
|---|---:|---:|---:|
| DMA 2 | 1.8261 | 1.4723 | -19.4% |
| DMA 3 | 1.5626 | 1.3333 | -14.7% |

Post-training linked effect:

| DMA | Extreme-rain days | Dry days | Extreme mean weekly residual (L/s) | Dry mean weekly residual (L/s) | Difference (L/s) | Bootstrap 95% interval |
|---|---:|---:|---:|---:|---:|---|
| DMA 2 | 11 | 148 | -0.4177 | 0.0147 | -0.4324 | [-0.7781, -0.1411] |
| DMA 3 | 11 | 148 | -0.5243 | 0.0057 | -0.5301 | [-0.9663, -0.1880] |

On the first post-training extreme-rain day (2022-07-26, 32.6 mm), both pre-specified DMAs fire the explicit rule chain `R8_EXTREME_RAIN -> R9_DEMAND_SUPPRESSION -> R10_LINKED_WEATHER_RESPONSE` and produce `WEATHER_LINKED_DEMAND_RESPONSE`.

See the machine-readable tables and traces under `results/derived/` and figures under `results/figures/`.
