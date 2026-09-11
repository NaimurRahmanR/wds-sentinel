> **Historical audit note (2026-09-11):** This document preserves early feasibility/audit reasoning. References below to an "untouched" 2019 holdout describe the original proposed protocol. In the final repository, 2019 pressure/leakage are disclosed as having been inspected once by an earlier rejected forecasting baseline; the final frozen multimodal temporal evaluation was run only after subsequent 2018-only system development was complete. Earlier environment-specific download blockers were later resolved.

# Historical data audit — Phase 0.5 snapshot

> This document preserves the initial acquisition/feasibility audit as historical research provenance. Statements such as "no model has been trained" describe that earlier phase and are **not the current repository status**. Current status is in `README.md`, `docs/research_protocol_status.md`, and `data/PROVENANCE.md`.

# Data audit — Phase 0.5 empirical feasibility gate

Scope: Tasks 1–9 of the Phase 0.5 brief. No model has been trained, no
target has been chosen, no threshold has been invented. Every fact below is
either directly verified by inspecting an official source in this
environment, or explicitly marked as unverified/blocked.

Sources inspected (both official, retrieved via GitHub's own archive
service, not a third-party mirror):
- https://github.com/KIOS-Research/BattLeDIM (master branch)
- https://github.com/KIOS-Research/EPANET-Benchmarks (master branch, `L-Town/` subfolder)

Reusable evidence script: `scripts/audit/audit_battledim_config.py`
(re-run: `python scripts/audit/audit_battledim_config.py /path/to/BattLeDIM-master`).

---

## Task 1 — Licence / redistribution audit

**WHAT:** Established provenance, version, authorship, and licence for every
candidate asset; separated code/config licensing from raw-measurement-data
licensing rather than treating them as one thing.

**HOW:** Downloaded both repositories via `codeload.github.com` (GitHub's
own archive endpoint), inspected `LICENSE.md`/absence of a licence file
directly, cross-checked against each repository's own GitHub page (which
shows GitHub's licence auto-detection under "Resources").

**WHY:** Licence and provenance must be established before any acquisition
or redistribution decision — inferring a licence from public
downloadability is explicitly disallowed by the brief.

**EVIDENCE:**

| Asset | Licence found | How verified |
|---|---|---|
| `KIOS-Research/BattLeDIM` (Dataset Generator code, Scoring Algorithm code, all ground-truth config/`.mat`/`.xlsx` files) | **EUPL-1.2** | Repository root `LICENSE.md` (full EUPL-1.2 text) + independent statement in `Dataset Generator/README.txt` ("Licensed under the EUPL... Copyright (c) 2020 KIOS..."); GitHub page explicitly badges "EUPL-1.2 license". |
| `KIOS-Research/BattLeDIM` L-Town `.inp` files (`L-TOWN_v2_Real.inp`, `L-TOWN_v2_Model.inp`) and `dataset_generator.exe` | Presumed same repository licence, **content unverified** | These are Git-LFS pointer files (~130 bytes each); actual object content could not be retrieved (see Task 8/engineering risks below). |
| `KIOS-Research/BattLeDIM` Zenodo-hosted historical/evaluation SCADA CSVs (DOI 10.5281/zenodo.4017659) | **UNRESOLVED** | Not present in the GitHub repository at all (verified by full directory listing); Zenodo itself unreachable from this environment (`HTTP 403` on direct request); targeted web searches did not surface the record's licence field either. **Not to be redistributed until resolved.** |
| `KIOS-Research/EPANET-Benchmarks` → `L-Town/L-TOWN.inp` (v1.2) | **No licence file present — UNRESOLVED, default copyright** | Verified: no `LICENSE`/`LICENSE.md` anywhere in the repository archive; GitHub's own repository page lists only "Readme" under Resources, no licence badge. **More restrictive than the BattLeDIM repo** — public downloadability confirmed, redistribution rights not established. |

---

## Task 2 — Raw file audit

**WHAT:** Inspected every file that is actually retrievable in this
environment, without training or labeling anything.

**HOW:** Direct file inspection (`file`, magic-byte checks, plain-text
reads, YAML parsing) on the extracted repository archives.

**WHY:** The brief requires inspecting file structure/content before any
target or split decision, and requires knowing exactly what is and isn't
available.

**EVIDENCE:**

Retrievable and inspected:
- `dataset_configuration.yalm`, `dataset_configuration_historical.yalm`,
  `dataset_configuration_evaluation.yalm` (BattLeDIM repo) — leak schedule
  + sensor node lists, plain text, parsed with PyYAML.
- `dataset_generator.py` (BattLeDIM repo) — 305 lines, Python, `import wntr`
  confirmed, pressure-dependent-demand (PDD) simulation mode, no explicit
  noise/randomness call found (`grep -i "random|seed|noise"` found nothing
  beyond an unrelated `np.arange` uncertainty-range variable and a leak
  diameter increment array — see Task 8).
- `Scoring Algorithm/competition_data/leakages_info.yalm` — plain text,
  23 entries.
- `Scoring Algorithm/competition_data/leak_info.mat`,
  `D_Mat_L-TOWN.mat` — valid MATLAB v5 binary files (verified via magic
  bytes `MATLAB 5.0 MAT-f`), internally timestamped 2020, consistent with
  contemporaneous competition artefacts.
- `Scoring Algorithm/competition_leakages/Leak_p*.xlsx` — 23 files,
  ~1.5 MB each, present and real (not LFS pointers).
- `L-Town/L-TOWN.inp` (EPANET-Benchmarks repo) — 406,569 bytes, plain ASCII
  EPANET INP format, fully parsed: 782 junctions, 905 pipes, 1 tank,
  2 reservoirs, 1 pump, 3 valves, 673 patterns, 2346 demand categories
  (see `data/PROVENANCE.md` for the full breakdown and SHA-256).

**Blocked — not retrievable in this environment:**
- The historical/evaluation **SCADA measurement CSVs**
  (`2018_SCADA_Demands.csv`, `2018_SCADA_Flows.csv`, `2018_SCADA_Levels.csv`,
  `2018_SCADA_Pressures.csv`, `2018_Leakages.csv`, and 2019 equivalents) —
  not present anywhere in either GitHub repository; authoritative location
  is Zenodo, which is unreachable.
- `L-TOWN_v2_Real.inp`, `L-TOWN_v2_Model.inp`, `dataset_generator.exe` (all
  in the BattLeDIM repo) — Git-LFS pointers only; the LFS object storage
  host (`github-cloud.githubusercontent.com`) returned `HTTP 403
  x-deny-reason: host_not_allowed` on direct test.

**Consequence:** row counts, missing values, duplicate timestamps, actual
sensor readings, and units-of-measurement for the real SCADA data **cannot
be audited yet** — this requires the blocked CSVs.

---

## Task 3 — Temporal / label audit

**WHAT:** Determined exactly how leakage events are represented, using the
verified ground-truth configuration (not the blocked CSVs).

**HOW:** Machine-parsed all three `dataset_configuration*.yalm` files plus
`leakages_info.yalm` with `scripts/audit/audit_battledim_config.py`.

**WHY:** The exact representation (interval vs. point, censored vs.
observed, single vs. concurrent) directly constrains which target
definitions (Task 6) are even well-posed.

**EVIDENCE (machine-computed, reproducible via the script above):**

- Representation: **interval-based per pipe** — `(linkID, startTime,
  endTime, leakDiameter_m, leakType ∈ {abrupt, incipient}, peakTime)`. Not
  point events.
- **33 total leak events, 33 unique pipes** across 2018-01-01 to
  2019-12-31 (14 in the 2018-only config, 23 in the 2019-only config, 4
  pipes common to both — see Task 7).
- Leak type split: 18 incipient, 15 abrupt (full period). For every abrupt
  entry, `peakTime == startTime` (0 exceptions found) — i.e. "abrupt"
  means instantaneous full severity, matching its name. For incipient
  leaks, start→peak ramp-up ranges from **3.56 to 97.81 days**.
- **16 of the 33 leaks have `endTime` equal to the file's own end-of-window
  boundary** (2019-12-31 23:55, or 2018-12-31 23:55 in the 2018-only file).
  This is a **genuine ambiguity, not resolved by the data**: it cannot be
  distinguished, from the config file alone, whether these leaks were
  actually still active at the true end of the study or whether the
  boundary value is simply a censoring convention. Treating it as "still
  leaking" vs. "repaired exactly at the boundary" is a labeling decision,
  not something I have decided.
- **212 overlapping (concurrent) leak-interval pairs** exist across the
  full 33-event set (38 within 2018 alone, 180 within 2019 alone).
  Concurrent multi-leak periods are the norm, not an edge case.
- Leak diameters range **0.0088–0.0229 m** (fixed per event, not a
  time-varying flow rate — actual flow depends on local pressure via the
  hydraulic simulation).
- No location field finer than the pipe/link ID is present in any
  inspected file — localisation ground truth is at pipe granularity.
- `endTime`/repair information is **not knowable in advance** at
  real-time decision-making time; it must never be used as a predictive
  feature, only for retrospective label construction (this is a Task 4
  leakage vector, noted here because it originates from the label
  structure itself).

---

## Task 4 — Leakage / split risk audit

**WHAT:** Catalogued the data-leakage mechanisms the brief lists, using
what is now verified about the label structure.

**HOW:** Reasoning from the verified interval/overlap/censoring facts
above, plus the verified sensor-node structure.

**WHY:** These risks constrain split design (Task 7) and must be surfaced
before, not after, a split is chosen.

**EVIDENCE / findings (each a risk requiring a protocol decision, none
resolved here):**

- **Row-level random splitting** at 5-minute sampling would place
  near-identical consecutive readings on both sides of a split — severe
  leakage. Requires block/temporal splitting.
- **Event-straddling:** verified — 4 leak events (p257, p427, p810, p654)
  span the 2018/2019 boundary with no repair in between (see Task 7). Any
  split at that boundary puts different phases of the *same* physical
  event on both sides.
- **Rolling/window features** using future timestamps (centred or
  backward-looking-with-future-leak) would leak information not available
  at decision time; only causal (past-only) windows are valid.
- **Whole-series normalisation** (e.g. z-scoring with statistics computed
  over the full 2018–2019 span) leaks test-period statistics into
  training.
- **`endTime`/repair time as a feature** (identified in Task 3) is
  unavailable at real-time decision time and must never be used as
  model input — only for constructing retrospective evaluation labels.
- **Interpolation crossing split boundaries** during missing-value
  handling is a leakage vector once the real CSVs are available and their
  missingness pattern is known (currently unauditable — CSVs blocked).
- **Duplicate/near-duplicate windows:** cannot be assessed without the
  real CSVs (blocked).

---

## Task 5 — Sensor / feature feasibility

**WHAT:** Catalogued which input families are specified, and separated
observed-SCADA vs. model-derived vs. ground-truth/evaluation information.

**HOW:** Parsed the sensor node lists in `dataset_configuration.yalm`
(pressure/flow/level/AMR) and cross-checked against the per-junction
`;AMR`/`;AMR & PRESSURE SENSOR` comments embedded directly in the verified
`L-TOWN.inp` (v1.2) file — two independently-sourced confirmations of the
same sensor layout.

**WHY:** The brief requires this separation explicitly, to prevent
ground-truth leak information from silently entering predictive features.

**EVIDENCE:**

| Category | Contents (verified from config + INP) |
|---|---|
| OBSERVED SCADA INPUT | 33 pressure-sensor nodes; 3 flow sensors (incl. one pump); 1 tank-level sensor (T1); 82 AMR (demand) nodes in the full-period config |
| MODEL-DERIVED INFORMATION | Anything computed by running a hydraulic simulation (WNTR/EPANET) over the (imperfect, participant-facing) network model — e.g. estimated pressure/flow at unmonitored nodes |
| GROUND-TRUTH / EVALUATION INFORMATION | The leak schedule itself (`dataset_configuration*.yalm`, `leakages_info.yalm`) — must never be a predictive feature, only a label/scoring source |

**Blocked:** actual sensor *values* (magnitude, noise level, missingness,
sampling gaps) require the CSVs, which are not accessible here.

---

## Task 6 — Target feasibility comparison

**WHAT:** Assessed what the *verified label structure* does and does not
support, without empirically scoring detectability (which needs the
blocked CSVs).

**HOW:** Mapped each of the five candidate target types (A–E) against the
verified facts from Tasks 3–5.

**WHY:** A qualitative, structure-level assessment is honest given current
blockers; claiming a quantitative score (e.g. actual detectability, SNR,
compute cost) would require data this environment does not have.

**EVIDENCE / qualitative assessment (structure-level only):**

| Candidate | Label support from verified structure | Caveat |
|---|---|---|
| A — active-leakage classification at time t | Directly supported — interval ground truth gives an unambiguous per-timestep active/inactive label per pipe, for any pipe with a known interval | Concurrent leaks (212 overlapping pairs) mean a single "any leak active" label conflates multiple simultaneous physical events |
| B — leakage-onset detection | Supported — `startTime` is explicit | `startTime` differs between the "evaluation-only" config and the official scoring ground truth for the 4 straddling events (Task 7) — onset definition needs a decision |
| C — short-horizon leak-event prediction | Structurally possible (interval start times exist) but detectability cannot be assessed without real sensor values | Fully blocked pending CSVs |
| D — leak localisation | Supported at pipe-ID granularity; the repo's own scoring code (`nodeTopologicalDistance.m`, `D_Mat_L-TOWN.mat`) implies partial credit for topologically-close misses in the original competition | Exact-match-only localisation may be an unnecessarily strict framing; topological partial-credit scoring is a metric choice, not decided here |
| E — abnormal-WDS-state detection | Directly supported, same ground truth as A | Same concurrency caveat as A |

I am not scoring computational cost, interpretability, or empirical label
quality/detectability — those require the blocked CSVs and are not
determined here.

---

## Task 7 — 2018 / 2019 role audit

**WHAT:** Investigated whether year-based (2018=train, 2019=test) is
natural and appropriate, using the verified label data.

**HOW:** Machine cross-checked `dataset_configuration_historical.yalm`
against `dataset_configuration_evaluation.yalm` and the officially
published competition scoring file `leakages_info.yalm`.

**WHY:** This is explicitly flagged in the brief as not to be adopted
automatically just because the original competition used it this way.

**EVIDENCE:**

- **Composition differs:** 14 leak events in 2018 vs. 23 in 2019 (8
  incipient/6 abrupt vs. 14 incipient/9 abrupt) — the two years are not
  drawn from a matched distribution of event types/counts.
- **4 pipes (p257, p427, p810, p654) have a leak event present in both
  years** — the *same physical leak*, continuing unrepaired across the
  boundary, not independent events (verified: `14 + 23 − 4 = 33`, matching
  the full-period total exactly).
- **Field-level discrepancy found and explained** (machine-verified) for
  those 4 pipes: `dataset_configuration_evaluation.yalm` (used to run the
  simulator standalone for 2019 only) re-parameterises each as if it
  *started* at 2019-01-01 00:00 (already at peak severity), whereas the
  official competition scoring ground truth (`leakages_info.yalm`)
  preserves each leak's **true original 2018 onset and peak time**. Example
  (p257): evaluation-config start/peak = `2019-01-01 00:00` / `2019-01-01
  00:00`; true onset per the scoring ground truth = `2018-01-08 13:30` /
  `2018-01-25 08:30`. All 4 discrepancies follow this same pattern. This
  means: **the true onset for these 4 events is in 2018, not 2019**, even
  though they are "active during" 2019 — a labeling subtlety a
  simulation-only reproduction using the evaluation config would get
  wrong if not corrected against the scoring ground truth.
- **2019 ground truth is already public**, has been since the competition
  concluded (the file is sitting in the public GitHub repo, dated 2020).
  A model-selection or hyperparameter process that is aware of published
  post-competition analyses of 2019 results is not a genuinely blind test
  in the way the live competition was — this is a **methodological risk**,
  not something I have resolved.

I am not recommending a specific split here; the brief's own suggestion
(validate inside 2018, keep 2019 as an untouched final holdout) is a
plausible mitigation for the public-ground-truth risk, but adopting it is
a protocol decision for the research-design process.

---

## Task 8 — WNTR / EPANET role audit

**WHAT:** Assessed WNTR/EPANET's role given what's now verified, without
installing WNTR (only static inspection of `dataset_generator.py`'s import
statement and logic was performed).

**HOW:** Read `dataset_generator.py` in full (305 lines); confirmed
`import wntr`, `wntr.network.WaterNetworkModel(inp_file)`, PDD simulation
mode; searched for randomness/noise injection (`grep -i
"random|seed|noise"`) and found none beyond an unrelated uncertainty-range
variable and a leak-diameter increment array used for incipient ramp-up.

**WHY:** Determines whether WNTR is a legitimate, protocol-consistent tool
for this project or an arbitrary addition.

**EVIDENCE / findings:**

- WNTR is not an arbitrary choice: it is **the same toolkit the original
  benchmark's own Python generator is built on**, which is a materially
  stronger justification than generic usefulness.
- No explicit noise-injection or random-seeding step was found in the
  inspected generator script. If the officially released SCADA CSVs
  contain measurement noise distinguishing them from raw hydraulic
  simulation output, that step is not in this script — it is either
  applied elsewhere (not found), or the base simulation itself is what was
  released. This is reported as an open question, not resolved.
- **Consequently, re-running this script would not be guaranteed to
  reproduce the officially released CSVs bit-for-bit**, even if the
  blocked network `.inp` file were obtained. Any locally-regenerated
  dataset must be labelled "BattLeDIM-protocol-derived," never presented
  as equivalent to "the BattLeDIM dataset."
- The v1.2 `L-TOWN.inp` (EPANET-Benchmarks repo) is real, retrievable, and
  gives full topology (782 junctions, 905 pipes) plus sensor tagging —
  useful for topology-aware reasoning/explanation work regardless of the
  measurement-data situation, but its licence is unresolved (Task 1) so it
  cannot be committed to this repository yet, and its version-identity
  relative to "v2" (used for the actual released measurements) is
  unverified.
- **Recommendation input (not a final decision):** USEFUL / CONTINGENT.
  Legitimate and protocol-consistent, but not required for a minimal first
  experiment restricted to the sensor CSVs once those are obtained, and
  currently blocked from a full role (network-based synthetic generation)
  by the unresolved licence and unreachable LFS objects.

---

## Task 9 — ERA5-Land role audit

**WHAT:** Assessed whether ERA5-Land integration is justified given
verified BattLeDIM materials.

**HOW:** Reviewed every file inspected in Tasks 1–8 for any reference to
weather, precipitation, or environmental-hazard variables.

**WHY:** The brief requires this to be justified by genuine heterogeneous-
integration value, not novelty for its own sake, and explicitly warns
against conflating environmental-hazard prediction with WDS operational-
risk inference.

**EVIDENCE:**

- **No reference to weather/precipitation/environmental data was found in
  any BattLeDIM or EPANET-Benchmarks material inspected.** Leaks are
  injected as independent engineering-fault events (pipe ID + time +
  diameter + type), not weather-triggered.
- ERA5-Land access is currently **fully blocked** in this environment:
  no network path to the Copernicus CDS API, and no credentials have been
  discussed.
- **Recommendation input (not a final decision):** given no natural
  linkage found in the primary benchmark materials, full current
  inaccessibility, and the 13 September freeze date, the evidence leans
  toward DEFER — but this is the research-design process's call, not mine.

---

## Cross-cutting note on scientific integrity

Nothing in this audit should be read as validating BattLeDIM/L-Town data as
real-world utility SCADA data. Every artefact inspected — the generator
script, the network model's own title block ("developed... for the
BattLeDIM 2020 competition"), and the absence of any real-utility
attribution anywhere in either repository — is consistent with this being
a **synthetic, EPANET-simulation-based benchmark**, not field-collected
data. Any future use of it must be described accordingly, and never as
operational validation.
