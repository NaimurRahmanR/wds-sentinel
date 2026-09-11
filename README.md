# WDS Sentinel

WDS Sentinel is a research prototype for **reliability-aware hybrid AI supervision of water-distribution-system evidence**. It combines a frozen data-driven detector, explicit knowledge-based reasoning, typed multi-agent coordination, evidence-quality assessment, selective supervisory control, and grounded explanations.

The repository intentionally preserves negative predictive results. Its primary BattLeDIM/L-Town experiment evaluates reliability-aware supervision under degraded operational evidence. A secondary CWA/CODIS stream demonstrates independent hazard-context integration, while a final **physically linked BWDF validation** uses real DMA inflows and weather measured within the same real WDN to test extreme-rain demand response and weather-aware forecasting. No rainfall-induced pipe-failure claim is made.

## What is implemented

- real BattLeDIM/L-Town 2018 and 2019 SCADA evidence: pressure, flow, tank level and leakage ground truth;
- chronological train/validation methodology and onset-window targets;
- frozen Logistic Regression predictive component plus preserved null/weak baselines;
- controlled evidence degradation: missing sensor subset, additive noise, conflicting evidence and sensor dropout;
- three comparison systems:
  - **A — Direct:** predictor threshold only;
  - **B — Hybrid:** predictor + explicit statistical corroboration;
  - **C — Reliability-aware hybrid:** B plus evidence-quality gating with `ABSTAIN` / `ESCALATE`;
- typed `EvidenceState` representation and adapter protocol;
- explicit deterministic KBS with rule identifiers and contradiction traces;
- Evidence, Prediction, Knowledge/Reasoning, Reliability, Hazard and Supervisory agents;
- IDSS-style `DecisionRecord` objects and grounded explanation generation;
- coverage-aware reliability analysis and descriptive risk-vs-coverage sensitivity analysis;
- independent real precipitation context from CWA/CODIS observations;
- real-data integrated scenario traces using **actual BattLeDIM validation rows**, not hand-authored WDS scores or sensor values;
- final frozen multimodal 2019 temporal evaluation across the same clean/degraded evidence conditions, with no post-hoc tuning.
- physically linked extreme-weather validation on the real BWDF network: ten DMA inflow series + co-located rainfall/temperature/humidity/wind, frozen W1 forecast evaluation, training-only extreme-rain calibration, and KBS decision traces.

## Main empirical findings

The predictive component is deliberately not presented as a strong leak detector. Multiple pressure-only, multimodal and seasonal-residual variants were tested and preserved as weak/null results rather than tuned until a favourable result appeared.

For the frozen 2018 validation experiment, System C explicitly withdraws autonomous operation when evidence quality violates its fixed reliability criteria. This should be interpreted jointly with coverage: lower autonomous error is not claimed merely because the system abstains. Under the severe fixed missing-sensor-subset condition, C reaches **0 autonomous coverage**, so selective unsafe risk is correctly treated as undefined rather than zero.

The agent architecture has been checked against the frozen System-C path on the real 2018 validation experiment; the full validation run produced exact row-level agreement across all five degradation conditions. The bounded regression test in `tests/test_agent_vs_frozen_real_data.py` guards against future drift without recomputing the whole year inside normal development loops.

The final frozen 2019 temporal evaluation confirms that the underlying detector remains weak out of time: on clean 2019 data the frozen predictor obtains AUPRC **0.004041** and AUROC **0.530564**, and A/B/C each detect only **1 of 19** genuine new onsets. System C operates autonomously on **0.8617** of clean test rows and has selective unsafe risk **0.002484**; this is not lower than A/B. The temporal result therefore reinforces the project's reliability interpretation rather than providing evidence of strong detection performance.

The independent precipitation indicator is calibrated on 2013–2017 CWA/CODIS-derived daily observations and evaluated on 2018. It uses the 95th percentile of 1-, 3- and 7-day trailing precipitation sums from the training period. The source-documented `-9.8` trace-precipitation code is converted to **0.09 mm**. Current derived results are written to `results/derived/hazard_validation_summary.csv`.

## Physically linked extreme-weather validation

The final extension uses the Battle of Water Demand Forecasting (BWDF) supplementary data, which pair hourly net inflow for ten real DMAs with rainfall, temperature, humidity and wind measured at a station within the same case-study WDN. Training is frozen through 2022-07-24 23:00 Europe/Rome and W1 (2022-07-25--31) is the evaluation week. Under the original BWDF protocol, evaluation-week weather was supplied as a perfect weather forecast, so current W1 weather is a legitimate exogenous input for the forecast comparison.

The training-only 95th-percentile threshold over positive daily rainfall is **18.775 mm/day**. During W1, a simple weather-aware Ridge model improves MAE over the previous-week baseline in **9/10 DMAs**. For the two pre-specified rainfall-sensitive residential DMAs identified by the benchmark publication:

- DMA 2: MAE **1.8261 -> 1.4723 L/s** (19.4% reduction);
- DMA 3: MAE **1.5626 -> 1.3333 L/s** (14.7% reduction).

Across the post-training period, both DMAs show lower same-hour previous-week demand residuals on 11 extreme-rain days than on 148 dry days. The extreme-minus-dry differences are **-0.432 L/s** for DMA 2 (descriptive bootstrap 95% interval **[-0.778, -0.141]**) and **-0.530 L/s** for DMA 3 (**[-0.966, -0.188]**). Actual co-located evidence is exported through typed `EvidenceState` objects and explicit KBS rules into `WEATHER_LINKED_DEMAND_RESPONSE` decisions. This is observational evidence of a weather-linked operational demand response, not a causal rainfall-to-failure claim.

## Integrated real-evidence scenarios

`scripts/run_hazard_integration_scenarios.py` constructs five reproducible scenarios using actual WDS validation outputs and actual precipitation observations:

1. real normal WDS row + normal hazard context;
2. real detected leak-onset-window row + normal hazard context;
3. real normal WDS row + elevated hazard context;
4. real detected leak-onset-window row + elevated hazard context;
5. real degraded leak-onset-window row + elevated hazard context.

The WDS and precipitation timestamps are allowed to differ explicitly because the sources are independent. The integrated orchestrator keeps the WDS operational decision unchanged and adds context such as `HAZARD_WATCH` or `ALERT_ELEVATED_HAZARD_CONTEXT`. These labels are monitoring context only, not hydraulic causal claims.

## Repository layout

- `src/wds_sentinel/` — package code;
- `scripts/` — experiment, audit, validation and reproduction entry points;
- `configs/` — frozen protocol summaries and execution settings;
- `data/raw/` — bundled BattLeDIM CSVs used by the current experiments;
- `data/external/hazard_taiwan_precip/` — bundled precipitation CSVs used by the independent hazard-context experiment;
- `data/external/bwdf/` — bundled CC BY 4.0 BWDF inflow/weather workbooks used by the physically linked validation;
- `data/PROVENANCE.md` — source, checksum and rights/provenance notes;
- `THIRD_PARTY_DATA_NOTICE.md` — explicit third-party-data notice and licence separation;
- `results/derived/` — regenerated tables/figures/traces;
- `results/raw/` — captured console/run logs for reproducibility;
- `docs/` — architecture, protocol status, audit and limitations;
- `tests/` — unit, regression and real-data integration tests.

## Verification state

The final bundled-data suite passes **135/135 tests**, including real-data regressions for the BattLeDIM architecture and the physically linked BWDF result. A compiled technical report is available at `report/wds_sentinel_report.pdf`; its LaTeX source is retained alongside it.

## Environment

Recommended: Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest -q
```

A direct-dependency snapshot is also provided:

```bash
python -m pip install -r requirements/dev.txt
python -m pip install -e . --no-deps
pytest -q
```

## Reproduce the core outputs

```bash
python scripts/run_reliability_experiment.py
python scripts/validate_agents_vs_frozen.py
python scripts/coverage_aware_evaluation.py
python scripts/run_hazard_integration_scenarios.py
python scripts/run_final_2019_evaluation.py
python scripts/run_linked_extreme_weather_validation.py
```

Or capture the main reproducibility logs in one run:

```bash
python scripts/reproduce_core_results.py
```

The full agent-vs-frozen validation is intentionally heavier than the unit suite because it evaluates all 264,900 condition-row comparisons. The normal test suite contains a computationally bounded real-data regression instead.

## Data and third-party rights

The repository's original source code is licensed under the MIT License. **That licence does not apply to third-party datasets.** See `THIRD_PARTY_DATA_NOTICE.md` and `data/PROVENANCE.md` before redistributing or reusing data files.

BattLeDIM is publicly downloadable from Zenodo at DOI `10.5281/zenodo.4017659`. At the time of this audit, the record is marked as an open dataset but the displayed licence field is blank; this repository therefore does not claim that the BattLeDIM CSVs are public-domain or covered by the MIT code licence.

The precipitation observations ultimately originate from Taiwan's Central Weather Administration (CWA) / CODIS. The bundled copy was obtained from the `Raingel/historical_weather` rebuild. The source and transformation chain are documented separately because the rebuild repository and the underlying government observations have different provenance/rights considerations.

The BWDF workbooks are distributed by `WaterFutures/wf4bwdf` from the Alvisi et al. (2025) supplementary materials under **CC BY 4.0**. Exact bundled-file hashes and attribution are recorded in `data/PROVENANCE.md`.

## Important limitations

- the frozen WDS predictor is weak and should not be interpreted as operationally validated leak detection;
- only five 2018 validation onset events occur in the chosen split, so event-level estimates are small-sample;
- the reliability gate demonstrates explicit evidence-quality control, not universally lower conditional risk;
- the severe missing-sensor condition intentionally produces zero autonomous coverage under the fixed policy;
- the CWA/CODIS precipitation context is independent of L-Town and is not a flood/damage predictor; the separate BWDF validation provides a real co-located weather-demand response but still does not establish rainfall-caused hydraulic failure;
- the final 2019 temporal evaluation is not a pristine blind test: an earlier rejected forecasting baseline had already inspected 2019 pressure/leakage, although subsequent detection-model, hybrid/KBS/MAS, and reliability development used 2018 only;
- the 2019 target evaluates **new leak onsets**; four leaks already active on 2019-01-01 are treated as carryovers rather than new events.

See `docs/research_protocol_status.md` for the frozen/open protocol boundary.
