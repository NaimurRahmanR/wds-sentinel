# Audited results summary

This summary is generated from the current reproducibility package after the 2026-09-11 repository audit. It reports only results regenerated from the bundled data and current code.

## 1. Frozen WDS reliability experiment — 2018 validation

Validation contains 52,980 scored five-minute rows and 5 leak-onset events under the chosen July-December split.

| Condition | System | Coverage | Selective unsafe risk | Events detected | False alert episodes/day |
|---|---|---:|---:|---:|---:|
| clean | A Direct | 1.000 | 0.001133 | 1/5 | 0.663 |
| clean | B Hybrid | 1.000 | 0.001208 | 1/5 | 0.261 |
| clean | C Reliability-aware | 0.866 | 0.001199 | 1/5 | 0.261 |
| missing sensor subset | A Direct | 1.000 | 0.001000 | 2/5 | 9.589 |
| missing sensor subset | B Hybrid | 1.000 | 0.001227 | 0/5 | 8.214 |
| missing sensor subset | C Reliability-aware | 0.000 | undefined | 0/5 | 8.214 |
| additive noise | A Direct | 1.000 | 0.000887 | 5/5 | 46.419 |
| additive noise | B Hybrid | 1.000 | 0.001133 | 4/5 | 16.977 |
| additive noise | C Reliability-aware | 0.833 | 0.001268 | 4/5 | 16.977 |
| conflicting evidence | A Direct | 1.000 | 0.001076 | 2/5 | 1.462 |
| conflicting evidence | B Hybrid | 1.000 | 0.001189 | 1/5 | 0.506 |
| conflicting evidence | C Reliability-aware | 0.866 | 0.001177 | 1/5 | 0.506 |
| sensor dropout | A Direct | 1.000 | 0.001114 | 2/5 | 21.000 |
| sensor dropout | B Hybrid | 1.000 | 0.001208 | 1/5 | 5.828 |
| sensor dropout | C Reliability-aware | 0.870 | 0.001214 | 1/5 | 5.828 |

**Interpretation:** C does not demonstrate uniformly lower selective risk when it acts. Its contribution is explicit failure awareness and selective withdrawal of autonomous decisions when fixed evidence-quality gates are violated. Coverage and selective risk must therefore be reported together. In the severe missing-sensor-subset condition, C has zero autonomous coverage, so selective risk is undefined rather than zero.

## 2. Agent architecture equivalence

The agent-based WDS path was compared row-by-row with the frozen System-C implementation on every 2018 validation row under all five degradation conditions:

- 52,980 rows × 5 conditions = **264,900 comparisons**;
- **264,900 exact decision matches**;
- **0 mismatches**.

This establishes that the typed Evidence → Prediction → KBS/Reasoning → Reliability → Supervisory architecture preserves the frozen decision behaviour rather than silently changing the experiment.

## 3. Independent precipitation-context component

Calibration: 2013–2017 daily CWA/CODIS-derived precipitation.  
Validation: 2018.  
Indicator: any 1-, 3- or 7-day trailing precipitation sum exceeds its frozen training-period 95th percentile.

Corrected source-documented trace handling (`-9.8 -> 0.09 mm`) produces:

- 1-day cutoff: **34.1 mm**;
- 3-day cutoff: **82.0 mm**;
- 7-day cutoff: **163.575 mm**;
- 2018 elevated days: **21 / 365**;
- undefined 2018 days: **0**.

This indicator is a statistically unusual precipitation-context flag only. It is not validated against flood/damage ground truth.

## 4. Real integrated evidence scenarios

The final scenario exporter no longer uses synthetic WDS scores or hand-set missingness/corroboration. It deterministically selects actual BattLeDIM validation rows and runs the real agent chain.

| Scenario | WDS evidence | Ground truth | Predictor score | WDS state | Hazard state | Integrated action |
|---|---|---:|---:|---|---|---|
| normal WDS + normal hazard | clean real row, 2018-07-01 01:00 | 0 | 0.5501 | NO_ALERT | NORMAL | ROUTINE |
| detected onset + normal hazard | clean real row, 2018-10-26 02:10 | 1 | 0.9969 | ALERT | NORMAL | ALERT |
| normal WDS + elevated hazard | same real normal row | 0 | 0.5501 | NO_ALERT | ELEVATED | HAZARD_WATCH |
| detected onset + elevated hazard | same real onset row | 1 | 0.9969 | ALERT | ELEVATED | ALERT_ELEVATED_HAZARD_CONTEXT |
| degraded onset + elevated hazard | real missing-sensor-subset row, 2018-07-07 09:10 | 1 | 0.0130 | ABSTAIN | ELEVATED | ABSTAIN |

All WDS scenario records contain the actual 37 pressure/flow/level evidence items, the actual model score, actual KBS facts/rules, actual evidence-quality signals and actual reliability decision. Hazard context uses actual precipitation dates through the Hazard Agent and hazard KBS. The WDS and hazard clocks are deliberately independent because the datasets are unrelated; no physical cross-domain causality is asserted.

## 5. Final frozen 2019 temporal evaluation

The final temporal run uses all four 2019 modalities with the frozen 2018-developed system. The four source files contain 105,120 aligned rows; the frozen detection split excludes the first one-hour boundary window, leaving **105,108 scored test rows**. The new-onset target contains **19 genuine 2019 onsets**; pipes `p257`, `p427`, `p654`, and `p810` are already active at the first 2019 row and are treated as carryovers rather than new events.

2019 is not described as a pristine blind test: pressure/leakage had been inspected once by an earlier rejected one-hour-ahead forecasting baseline. They were not used for subsequent detection-model, hybrid/KBS/MAS, or reliability-controller development. No parameter was changed after this final temporal run.

Clean predictive performance remained weak: **AUPRC 0.004041**, **AUROC 0.530564**, precision **0.022161**, recall **0.032389**, and F1 **0.026316** at the frozen threshold 0.99.

| System | Coverage | Selective unsafe risk | New onsets detected | Non-onset alert episodes/day |
|---|---:|---:|---:|---:|
| A Direct | 1.0000 | 0.002274 | 1/19 | 0.773 |
| B Hybrid | 1.0000 | 0.002331 | 1/19 | 0.406 |
| C Reliability-aware | 0.8617 | 0.002484 | 1/19 | 0.406 |

The result does **not** establish effective leak detection or lower conditional risk for C. It provides a temporal stress test showing that the explicit reliability gate continues to withdraw autonomous decisions at a similar clean-data coverage level (about 86%) while the underlying predictive signal remains poor. Under additive-noise degradation, event-detection rates rise only alongside a very large alert burden (A: 38.54 non-onset alert episodes/day; B/C: 14.53/day), so this is not interpreted as improved skill.

`non_onset_alert_episodes_per_day` is named relative to the new-onset target because carryover leaks can make an alert outside a new-onset window operationally ambiguous rather than necessarily false. Full results are in `results/derived/final_2019_temporal_evaluation.csv` and `results/derived/final_2019_predictive_metrics.csv`.

## 6. Reproducibility checks

- automated suite: **128 passed**;
- formerly unbounded 2,000-row real-data equivalence test now runs on a genuinely bounded feature window;
- core experiment logs, including the final temporal audit, are retained under `results/raw/`;
- raw BattLeDIM checksums in `data/PROVENANCE.md` match the official Zenodo file-level MD5 values for all eight bundled BattLeDIM CSVs.
