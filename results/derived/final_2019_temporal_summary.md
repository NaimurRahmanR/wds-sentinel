# Final frozen 2019 temporal evaluation

All model, feature, threshold, degradation and reliability settings were frozen before this run. No 2019 result was used for tuning.

The four 2019 modalities contain 105,120 exactly aligned rows. Consistent with the frozen backward-looking detection split, the first one-hour boundary embargo leaves **105,108 scored test rows**. There are 19 genuine new leak onsets; four leaks already active at 2019-01-01 (p257, p427, p654, p810) are carryovers and are excluded from the new-onset target.

2019 is **not** described as a pristine blind test: pressure/leakage were inspected once by an earlier rejected 1-hour-ahead forecasting baseline. They were not used for subsequent detection-model, KBS/MAS, or reliability-controller development.

Clean frozen predictor: AUPRC=0.004041, AUROC=0.530564, precision=0.022161, recall=0.032389, F1=0.026316 at threshold 0.99. Positive-row prevalence is 0.002350.

## Clean-condition system results

| System | Coverage | Selective unsafe risk | New onsets detected | Non-onset alert episodes/day | Abstain/escalate |
|---|---:|---:|---:|---:|---:|
| A_DIRECT | 1.0000 | 0.002274 | 1/19 | 0.773 | 0.0000 |
| B_HYBRID | 1.0000 | 0.002331 | 1/19 | 0.406 | 0.0000 |
| C_RELIABILITY_AWARE | 0.8617 | 0.002484 | 1/19 | 0.406 | 0.1383 |

`non_onset_alert_episodes_per_day` is relative to the **new-onset detection target**. Because four leaks carry over from 2018, an alert outside a new-onset window is not necessarily an operationally false leak indication.

Across controlled degradations, the final evaluation is descriptive robustness analysis only. No setting was changed after observing 2019. Additive noise yields broad alerting and therefore high apparent event detection alongside a very high alert burden; it is not evidence of improved detection skill.

Full condition-by-system results are in `final_2019_temporal_evaluation.csv`; score-level metrics are in `final_2019_predictive_metrics.csv`.
