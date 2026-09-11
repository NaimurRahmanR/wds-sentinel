# Research protocol status

Last updated: 2026-09-11 after repository audit and real-evidence integration repair.

## Frozen / completed

- **Primary WDS task:** early leak-onset detection; positive label for timestamps from onset through +1 hour.
- **Chronological split:** 2018 Jan-Jun training, Jul-Dec validation, with the established causal feature handling/embargo rules in code and tests.
- **Predictor:** Logistic Regression with balanced class weights, raw multimodal causal change features, frozen validation-selected threshold. No further model tuning after the documented pressure/multimodal/seasonal comparisons.
- **Decision taxonomy:** WDS `NO_ALERT`, `ALERT`, `ABSTAIN`, `ESCALATE`; independent integrated monitoring actions `ROUTINE`, `HAZARD_WATCH`, `ALERT`, `ALERT_ELEVATED_HAZARD_CONTEXT`, `ABSTAIN`, `ESCALATE`.
- **Systems:** A Direct, B Hybrid, C Reliability-aware Hybrid.
- **Degradation conditions:** clean, missing sensor subset, additive noise, conflicting evidence, sensor dropout.
- **Reliability signals:** missingness, cross-group disagreement, instability.
- **Fixed reliability cutoffs:** missingness >0.20; disagreement >3.0; instability above the 99th percentile of clean training instability.
- **Coverage-aware interpretation:** selective unsafe risk is reported only at achieved autonomous coverage; zero-coverage risk is undefined.
- **KBS/MAS:** explicit deterministic WDS rules and hazard-context rules; typed agent messages; reliability gate cannot be bypassed by the supervisory path.
- **Agent equivalence:** full 2018 validation comparison previously produced 264,900/264,900 exact System-C decision matches across 5 conditions.
- **Hazard context:** CWA/CODIS-derived precipitation, 2013-2017 calibration and 2018 validation; `-9.8` trace precipitation is converted to 0.09 mm per the source rebuild's documented handling.
- **Integrated evidence demonstration:** real BattLeDIM rows and real precipitation observations executed through the same integrated orchestrator with independent clocks.
- **Final 2019 temporal evaluation:** all four 2019 modalities are present and aligned; the frozen 2018-developed multimodal predictor, A/B/C logic, degradation definitions and reliability cutoffs were evaluated without retuning. The frozen test mask excludes the first one-hour boundary window, leaving 105,108 scored rows, 19 genuine new onsets and 4 excluded carryovers.
- **Physically linked extreme-weather validation:** BWDF real-WDN hourly DMA inflows and co-located weather observations are bundled and verified against the public source. Training ends 2022-07-24 23:00 Europe/Rome; W1 (2022-07-25--31) is the frozen evaluation week. The rain-extreme threshold is calibrated on training data only. DMA 2/3 are pre-specified from the benchmark publication as summer rainfall-sensitive residential DMAs. W1 weather is used as an exogenous input under the original BWDF benchmark convention that supplied evaluation-week weather as a perfect forecast.

## Preserved negative / weak results

- one-hour-ahead onset forecasting showed no useful discrimination;
- early detection with pressure-only and multimodal SCADA remained weak;
- weekly seasonal-residual features did not materially rescue the detector;
- these results are retained and the predictive search is frozen rather than expanded post hoc.

## Open / not claimed complete

- independent flood/damage ground truth for the precipitation indicator: not available/used, so `ELEVATED` means statistically unusual precipitation relative to the calibration period, not flood occurrence;
- operational/utility validation in a real water utility: not claimed;
- geographic or causal linkage between the independent precipitation stream and L-Town: explicitly not claimed.

## Physically linked extreme-weather result

- BWDF training-calibrated positive-wet-day 95th-percentile daily-rain threshold: **18.775 mm/day**.
- During W1, the weather-aware Ridge model has lower MAE than the previous-week baseline in **9 of 10 DMAs**.
- For the pre-specified rainfall-sensitive DMAs, W1 MAE falls from **1.8261 to 1.4723 L/s** in DMA 2 (19.4% reduction) and from **1.5626 to 1.3333 L/s** in DMA 3 (14.7% reduction).
- Across the post-training period, 11 extreme-rain days and 148 dry days are available for DMA 2/3. Mean same-hour previous-week demand residuals are lower on extreme-rain days by **0.432 L/s** in DMA 2 (bootstrap 95% interval [-0.778, -0.141]) and **0.530 L/s** in DMA 3 ([-0.966, -0.188]).
- The first post-training extreme-rain day produces real typed evidence and KBS traces for both DMAs, with `R8_EXTREME_RAIN`, `R9_DEMAND_SUPPRESSION`, and `R10_LINKED_WEATHER_RESPONSE` firing and action `WEATHER_LINKED_DEMAND_RESPONSE`.
- These are observational weather-demand associations in a real co-located WDN; no rainfall-to-failure causal claim is made.
