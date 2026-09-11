# Data provenance

This document records the exact third-party data included in this reproducibility package. It does not transfer ownership or relicense any dataset. See `../THIRD_PARTY_DATA_NOTICE.md`.

## BattLeDIM / L-Town SCADA

**Source record:** Zenodo, *Dataset of BattLeDIM: Battle of the Leakage Detection and Isolation Methods*  
**DOI:** `10.5281/zenodo.4017659`  
**Publisher:** Zenodo  
**Authors:** Stelios G. Vrachimis et al.  
**Access status:** publicly downloadable dataset.

The Zenodo record exposes file-level MD5 checksums. The bundled files below were independently hashed during the 2026-09-11 audit; each listed MD5 exactly matches the value shown by Zenodo.

| Bundled file | Bytes | MD5 | SHA-256 |
|---|---:|---|---|
| `2018_Leakages.csv` | 6,234,468 | `c2c5fab90420da44f02e29775050abfe` | `e1abd9d549a18880d8ae704dc90c06c8a3d5aecbd2541fa091548ef392084ba6` |
| `2018_SCADA_Flows.csv` | 4,023,420 | `d0602c06946b46287e956f007e4264ee` | `7f7e8625c9d92b083ee079a54cffb9ef803876de1e35017c1da0455968daf3ff` |
| `2018_SCADA_Levels.csv` | 2,721,000 | `92a512d6749ddf66fd6d99f2bdde6cc4` | `a1fc5f0229448a620b6220f4d079b3162a255b3070acdce448608fbfc760903c` |
| `2018_SCADA_Pressures.csv` | 22,593,812 | `d389d8541350c19ff0bfc6b80f246d35` | `7782d890a7195699be94ddc01cba392da7ca285908afff619065a9616e04dcae` |
| `2019_Leakages.csv` | 10,414,660 | `e1f0a43683813a90a9fec8562dde599b` | `b2ffe56b6c2f3b6cdb6aae9aeb30b9c425ba911fff8e3716ae543a56ff7576cd` |
| `2019_SCADA_Pressures.csv` | 22,606,417 | `5ea1e46d3f2f0a89a3f98d6fd39a851d` | `a4af6616059cc9b366bbab76d880b2c772df544970c6ff4d4a29a58704ce05b1` |
| `2019_SCADA_Flows.csv` | 4,130,511 | `28fc99fdcbf80fcd26079e7fe602d6dc` | `c709ea5ecc03197e1fe70cf4c2de7225f71015acb50a7445ed7765b5029f7f15` |
| `2019_SCADA_Levels.csv` | 2,720,780 | `e5a8050bd38729e4b7648b9d52abd5b1` | `00624e97b72e63233beeccb48db528d43e59a87b116d0295d5c2a206ee0e75a8` |

### Rights note

The Zenodo page is labelled **Dataset Open**, but during the audit the displayed `Rights -> License` field was blank. The files are therefore included for reproducibility with attribution and a clear separation from this repository's MIT code licence; no claim is made that they are public-domain or MIT-licensed. Users should consult the original record before redistribution or downstream reuse.

### Current completeness

Both 2018 and 2019 multimodal pressure/flow/level/leakage inputs required by the frozen experiment are bundled. The final 2019 temporal evaluation uses the 2019 files unchanged and records its outputs under `results/derived/final_2019_*`.

## CWA/CODIS precipitation context

**Underlying observations:** Taiwan Central Weather Administration (CWA), CODIS station observations.  
**Station:** 466920 (Taipei).  
**Bundled processed copy:** `https://github.com/Raingel/historical_weather` daily CSV rebuild.  
**Years used:** 2013-2018.

| File | SHA-256 |
|---|---|
| `466920_2013_daily.csv` | `c3d7a32db2841c88e36cfff11e47212ddf5e385d5e9f09b1d6db79653d5ae09d` |
| `466920_2014_daily.csv` | `83d7e2ad6507be281076cd02f97a015249bc21c2d460d675f760081d78b2301e` |
| `466920_2015_daily.csv` | `74ea7624c75d6455fcb79746ef084f5cf98f1e3b8e51d716dd206a46ed053823` |
| `466920_2016_daily.csv` | `784015f994993423258f7e86899d17110ba6ec928c1585e2d3502161438e17ad` |
| `466920_2017_daily.csv` | `efb0393948eca1cdfbaa78fe692f480046d58cfb4ddfcd4ea315319961345ef0` |
| `466920_2018_daily.csv` | `3cab7f5a0b17f530dc999c99a538d00047ece5d84b32137beb4f412a56a316c1` |

### Trace precipitation handling

The rebuild's source code explicitly documents precipitation value `-9.8` as **trace precipitation (<0.1 mm)** and converts it to `0.09 mm`. `wds_sentinel.hazard.data.load_daily_precipitation()` reproduces that documented conversion. It must not be treated as missing data.

### Rights note

CWA's open-data terms permit reuse of covered open government data subject to attribution, and CODIS asks users to acknowledge the source. The third-party rebuild repository itself does not carry a separate repository licence, so the bundled CSVs remain excluded from this project's MIT code licence and their transformation chain is recorded here.

### Scientific scope

Taipei precipitation and the BattLeDIM/L-Town WDS are unrelated in geography and time. The precipitation series is used as an **independent external hazard-context stream** for heterogeneous-evidence integration. It is not used to infer or validate rainfall-driven hydraulic consequences in L-Town.

## Battle of Water Demand Forecasting (BWDF): physically linked weather-demand validation

**Primary publication:** Alvisi et al. (2025), *Battle of Water Demand Forecasting*, Journal of Water Resources Planning and Management, DOI `10.1061/JWRMD5.WRENG-6887`.  
**Open implementation/distribution:** WaterFutures `wf4bwdf`, version 1.0.0.  
**Data licence stated by distributor:** CC BY 4.0.  
**Expected bundled files:** `data/external/bwdf/InflowData.xlsx`, `data/external/bwdf/WeatherData.xlsx`.

The WDN contains ten DMAs. The weather observations are measured at a station located within the case-study WDN and cover rain, temperature, humidity and wind speed on the same hourly timeline as the DMA net-inflow observations. This dataset is used only for the physically linked extreme-weather / operational-demand validation; it is not used to alter the frozen BattLeDIM leak-detection experiment.

The exact bundled files were verified byte-for-byte against the public `WaterFutures/wf4bwdf` repository using Git blob hashes.

| Bundled file | Bytes | Git blob SHA | SHA-256 |
|---|---:|---|---|
| `InflowData.xlsx` | 1,877,617 | `e3404426bd803a35343fe3a0da3b3b5bb35d7265` | `c239bdb5c98eee00d94fb68b6b0405c39c3ac9b032c6fd4c94bb8d12d00640c0` |
| `WeatherData.xlsx` | 642,069 | `9ec52e1744e88ab7d4075e17024d44b6405624ad` | `70729f184c0b47abde2ef2686e28e013693641782d6f1b26b3f28de9c8d0c62c` |

The paired files contain 19,679 aligned hourly rows from 2021-01-01 through 2023-03-31. The inflow workbook contains ten DMA net-inflow series; the weather workbook contains rainfall depth, air temperature, humidity and wind speed on the same time axis.

### Frozen linked-weather protocol

The final linked validation uses BWDF evaluation week W1 (2022-07-25 through 2022-07-31) after training through 2022-07-24 23:00 Europe/Rome. The original BWDF protocol supplied evaluation-week weather as a perfect weather forecast, so using W1 weather as an exogenous predictor does not constitute future-information leakage relative to that benchmark. Extreme rainfall is defined from training data only as the 95th percentile of positive daily rainfall totals. DMA 2 and DMA 3 are pre-specified as the weather-sensitive pair because the BWDF publication identifies their summer demand as susceptible to rainfall.

The linked experiment is observational. It demonstrates a physically co-located weather--WDN operational response and weather-aware demand forecasting; it does not claim rainfall causes pipe failure or validate flood damage.
