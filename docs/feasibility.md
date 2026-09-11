> **Historical audit note (2026-09-11):** This document preserves early feasibility/audit reasoning. References below to an "untouched" 2019 holdout describe the original proposed protocol. In the final repository, 2019 pressure/leakage are disclosed as having been inspected once by an earlier rejected forecasting baseline; the final frozen multimodal temporal evaluation was run only after subsequent 2018-only system development was complete. Earlier environment-specific download blockers were later resolved.

# Feasibility recommendation — Phase 0.5

See `docs/data_audit.md` for the full Task 1–9 investigation this is based
on. This document covers Task 10 and the final Task 11 recommendation.

## Task 10 — Unsafe-decision ground-truth feasibility

**WHAT:** Investigated whether an external, non-circular definition of
"unsafe recommendation" is possible.

**HOW:** Checked whether any verified ground truth exists that is
independent of any candidate system's own output (avoiding the circularity
the brief warns against).

**WHY:** A decision-quality metric defined relative to the system being
evaluated cannot demonstrate anything about that system.

**FINDING:** Yes, a non-circular definition is possible, but only at a
certain granularity. The verified leak-interval ground truth
(`dataset_configuration*.yalm`, independent of any model) can externally
define "a leak is actually active on pipe X at time t" — this supports
grounding **NO_ALERT / ALERT** (e.g. unsafe = ALERT-worthy ground truth
paired with a NO_ALERT decision, a false negative against the external
leak record) without circularity, with ABSTAIN/ESCALATE as controller-level
actions layered on top rather than ground-truth categories.

A finer **WATCH/WARNING/CRITICAL** severity taxonomy is **not supported**
by anything verified in this audit — the only severity-adjacent fields are
leak diameter and type (incipient/abrupt), and mapping those onto named
severity tiers would require inventing thresholds, which is explicitly out
of scope. This matches the brief's own suggested simplification.

---

## Task 11 — Feasibility recommendation

### A. Verified dataset facts
- BattLeDIM/L-Town, organised by KIOS Research and Innovation Centre of
  Excellence (University of Cyprus) with co-organisers at TU Delft,
  Technion, and Tsinghua University; competition held 2020.
- Official code/config repository: `github.com/KIOS-Research/BattLeDIM`,
  licensed EUPL-1.2, containing the Python/WNTR-based generator, MATLAB
  scoring code, and the complete, machine-verified 33-event leak ground
  truth for 2018–2019.
- A second official repository, `github.com/KIOS-Research/EPANET-Benchmarks`,
  contains a real, retrievable L-Town network model (v1.2): 782 junctions,
  905 pipes, 1 tank, 2 reservoirs, 1 pump, 3 valves (SHA-256 in
  `data/PROVENANCE.md`) — independently confirming the "782 nodes" figure
  otherwise seen only in secondary literature.
- The historical/evaluation **SCADA measurement CSVs** described in the
  original brief are **not present in either GitHub repository**; their
  authoritative location is Zenodo (DOI 10.5281/zenodo.4017659), which is
  unreachable from this environment.

### B. Licence / provenance status
- BattLeDIM generator/scoring code + all ground-truth config files: EUPL-1.2,
  **resolved**, redistribution permitted.
- BattLeDIM L-Town `.inp` files: Git-LFS-blocked, content unverified.
- BattLeDIM Zenodo SCADA CSVs: **unresolved**.
- EPANET-Benchmarks L-Town v1.2 `.inp`: **unresolved, no licence file
  present at all** — more restrictive than BattLeDIM's own repo; usable for
  local inspection only, not for redistribution in this repository.

### C. Available evidence streams
Specified (not yet valued): 33 pressure sensors, 3 flow sensors (one on a
pump), 1 tank-level sensor, 82 AMR/demand nodes — cross-confirmed from two
independent official sources. Actual sensor *values* remain inaccessible.

### D. Leakage-event representation
Interval-based per pipe (start, end, diameter, type, peak time); 33 events,
212 overlapping pairs, 16 right-censored at the observation-window
boundary with ambiguous "still active vs. repaired-at-boundary" meaning.
Full detail in `docs/data_audit.md` Task 3.

### E. Candidate targets
A (active-leakage classification), B (onset detection), D (localisation),
and E (abnormal-state detection) are structurally supported by the
verified label data. C (short-horizon prediction) is structurally possible
but its detectability is unassessable without the blocked CSVs. No target
has been selected.

### F. Temporal / split risks
Row-level random splitting, event-straddling across the 2018/2019
boundary (verified: 4 events), future-window/whole-series-normalisation
leakage, and repair-time-as-feature leakage are all identified. The
2018/2019 boundary additionally carries a **public-ground-truth risk**:
the 2019 answer key has been public since 2020.

### G. Recommended primary target
**Not recommended by me** — per the operating constraint that scientific
target selection belongs to the research-design process. What I can say:
the verified label structure most cleanly supports A/E without invented
thresholds, and least cleanly supports any severity-graded target beyond
NO_ALERT/ALERT.

### H. Recommended evaluation split design
**Not finalised.** The brief's own suggested mitigation — validate inside
2018 only, preserve 2019 as an untouched final holdout — directly
addresses the public-ground-truth risk found in F, but adopting it, and
deciding how to handle the 4 boundary-straddling events, is a protocol
decision, not made here.

### I. WNTR/EPANET role
**USEFUL / CONTINGENT.** Legitimate and protocol-consistent (it's the same
toolkit the original benchmark's generator uses), but not required for a
minimal first experiment restricted to sensor CSVs, and currently
constrained by the unresolved L-Town model licence and the blocked LFS
objects.

### J. ERA5-Land role
**Evidence leans DEFER.** No natural linkage found in any inspected
BattLeDIM material; currently fully inaccessible (network + no
credentials); tight timeline. Final call belongs to the research-design
process.

### K. External, non-circular unsafe-decision feasibility
**Yes, at NO_ALERT/ALERT granularity**, grounded in the verified,
model-independent leak-interval ground truth. Not supported at a
WATCH/WARNING/CRITICAL granularity without inventing thresholds.

### L. Scientific risks
- This is synthetic, EPANET-simulation-based benchmark data — must never
  be described as real-world operational validation.
- The 2019 evaluation-year ground truth is already public, undermining a
  naively "blind" 2019 holdout unless explicitly mitigated.
- 4 leak events straddle the 2018/2019 boundary as continuations of the
  same physical event, not independent occurrences.
- `endTime` semantics are ambiguous for 16 of 33 events (censoring vs.
  genuine end).
- Concurrent/overlapping leaks (212 pairs) complicate any single-leak
  framing.
- Two different network-model versions exist (v1.2 vs. v2) with unverified
  equivalence.
- Licence for the actual measurement CSVs, and for the alternative L-Town
  model source, remain unresolved.

### M. Engineering risks
- Zenodo is unreachable from this environment; GitHub LFS object storage
  is also blocked even though repository metadata and small files are
  reachable.
- The original Scoring Algorithm is MATLAB + EPANET-MATLAB-toolkit-only —
  not directly runnable in a Python-first stack without reimplementation.
- WNTR has not been installed or run in this environment yet (only
  statically inspected) — its actual compatibility here is unverified.
- Tight timeline (13 September freeze) against multiple unresolved
  acquisition/licence blockers.

### N. GO / MODIFY / NO-GO
**MODIFY.** The research direction is scientifically viable and is now far
better evidenced than at Phase 0 (33-event ground truth machine-verified,
network topology machine-verified from a primary source, sensor layout
cross-confirmed from two independent official repositories). But the
acquisition route for the two things a real experiment needs most — the
raw SCADA measurement CSVs and an unambiguous, licence-clear network model
— is currently **blocked** in this sandboxed environment. Per the Phase
0.5 stop conditions, I am not working around this by substituting
generator output, inventing labels, or proceeding past it. See below for
exactly what would need to be manually supplied.

---

## Minimum manual-download list

Per instruction, this is scoped to what the *label structure itself* most
directly supports (Task E: active-leakage classification / abnormal-state
detection at a timestep, using the verified sensor layout), not the full
archive. All items are from the single authoritative source, Zenodo DOI
**10.5281/zenodo.4017659**.

| File | Purpose | Approx. size* |
|---|---|---|
| `2018_SCADA_Pressures.csv` | Primary sensor stream (33 pressure nodes) for both training-side feature construction and detectability assessment | Est. tens of MB at 5-min resolution over 1 year × 33 columns |
| `2018_SCADA_Flows.csv` | Secondary sensor stream (3 flow sensors) | Est. low MB |
| `2018_SCADA_Levels.csv` | Tank-level stream (1 sensor) | Est. <1 MB |
| `2018_Leakages.csv` | Official released ground-truth labels for 2018, to cross-check against the already-verified `dataset_configuration_historical.yalm` | Small (event list) |
| `2019_SCADA_Pressures.csv`, `2019_SCADA_Flows.csv`, `2019_SCADA_Levels.csv`, `2019_Leakages.csv` | Same four items for 2019, needed to evaluate any split design (Task F/H) | Same order of magnitude as 2018 |

*Sizes are estimates based on the verified sampling frequency (5 minutes)
and sensor counts (Task C); I have not seen the actual files, so these are
not verified figures — please treat them as planning estimates only, not
audited facts.

Not included in this minimum list (deferred until a target is chosen):
`2018_SCADA_Demands.csv` / `2019_SCADA_Demands.csv` (82 AMR nodes — large,
and only clearly needed if a target uses demand-side features) and the
network `.inp` files (needed only if WNTR/EPANET involvement in I is
approved).

I could not find a published MD5/checksum for these individual files from
this environment (the Zenodo record itself is unreachable); please note
whatever Zenodo's own download page shows for each file so it can be
recorded in `data/PROVENANCE.md` once uploaded, and I'll verify each file's
SHA-256 against that on arrival.

---

## Remaining blockers

1. Zenodo licence for the SCADA CSVs — unresolved, needs a human to check
   the record page directly (unreachable from here).
2. The above CSVs themselves — need manual download + upload to
   a local raw-data directory, since the original audit environment could not retrieve Zenodo directly. This blocker was later resolved and the required files are now bundled.
3. L-Town network model version identity (v1.2 vs. v2) and the v1.2 file's
   licence — needs either KIOS clarification or a decision to proceed
   without redistributing it.
4. GitHub LFS object storage (`github-cloud.githubusercontent.com`) is
   blocked by this environment's network policy — relevant only if the
   `v2` `.inp`/`.exe` files are specifically needed later.
