# Third-party data notice

This repository contains third-party data for research reproducibility. **No ownership of those datasets is claimed, and the repository's MIT licence does not apply to them.**

## BattLeDIM / L-Town SCADA data

Bundled files under `data/raw/` originate from the publicly accessible BattLeDIM dataset:

- **Dataset:** *Dataset of BattLeDIM: Battle of the Leakage Detection and Isolation Methods*
- **Publisher:** Zenodo
- **DOI:** `10.5281/zenodo.4017659`
- **Authors:** Stelios G. Vrachimis et al.

The Zenodo record is publicly downloadable and is labelled as an open dataset. During the 2026-09-11 audit, its displayed **Rights → License** field was blank. Accordingly:

- these files are not relicensed by this repository;
- this repository does not describe them as public-domain;
- users should consult the original Zenodo record and any applicable source terms before reuse or redistribution;
- exact checksums and file-level provenance are recorded in `data/PROVENANCE.md`.

## CWA/CODIS precipitation observations

Files under `data/external/hazard_taiwan_precip/` are a rebuild/copy of Taiwan Central Weather Administration (CWA) / CODIS observations obtained from:

`https://github.com/Raingel/historical_weather`

The underlying observations originate from CWA/CODIS. CWA's open-data terms permit broad reuse of covered open government data subject to attribution; CODIS also instructs users to acknowledge the source. The specific GitHub rebuild does not itself provide a separate repository licence, so this project preserves the provenance chain and does not apply the MIT code licence to those CSVs.

The rebuild source code explicitly documents precipitation value `-9.8` as **trace precipitation (<0.1 mm)** and converts it to `0.09 mm`. WDS Sentinel applies the same documented conversion during loading.

## Separation from project claims

The BattLeDIM and precipitation datasets come from unrelated domains/locations. The precipitation evidence is used as an independent external-hazard context to test heterogeneous evidence handling. **No geographic, temporal or causal relationship between Taipei precipitation and the L-Town network is asserted.**

## Battle of Water Demand Forecasting (BWDF) linked weather-demand data

The final physically linked extreme-weather validation uses the open supplementary data from Alvisi et al. (2025), *Battle of Water Demand Forecasting*, as distributed by the WaterFutures `wf4bwdf` implementation.

The dataset describes a real WDN in north-east Italy with hourly net inflow for ten district metered areas and weather observations (rain, temperature, humidity and wind speed) from a station located within the same case-study WDN. The WaterFutures distribution states that these data are licensed under **CC BY 4.0**.

The BWDF data remain third-party material and are excluded from this repository's MIT licence. The exact bundled workbooks were verified byte-for-byte against the public `WaterFutures/wf4bwdf` repository; source attribution, hashes and the frozen linked-weather protocol are recorded in `data/PROVENANCE.md`.
