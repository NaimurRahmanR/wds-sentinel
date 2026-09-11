# Coverage-aware evaluation of A/B/C — interpretation

Uses the frozen 2018 validation outputs only. Predictors, thresholds,
degradation definitions, KBS rules, agent logic, and the previously
reported A/B/C results are unchanged; this document only adds a
descriptive layer on top of them. 2019 not touched.

## Main summary (`coverage_aware_evaluation_summary.csv`)

A and B always have 100% autonomous coverage (they never abstain/escalate
by construction) and correspondingly a `selective_unsafe_risk` denominator
equal to the full row count. C trades coverage for a smaller autonomous
denominator every time it fires its reliability gate.

**The missing_sensor_subset row for C must be read as a genuinely
different kind of result, not a low-risk row:** coverage = 0.0000,
autonomous_count = 0, and `selective_unsafe_risk` is **undefined**, not
zero — with zero autonomous decisions there is nothing for the risk
fraction to describe. Reporting a risk number there would look like "0%
risk" and imply a favorable outcome; it would actually mean "this policy
made no autonomous claims at all under this condition." A and B, by
contrast, still make 52,980 autonomous decisions each under the same
condition, with selective risk of 0.0010 and 0.0012 respectively — worse
in an absolute sense than C's undefined figure only in the sense that they
made claims at all, not in the sense that a comparable risk number exists
to be worse.

## Sensitivity sweep (`reliability_sensitivity_sweep_C.csv`, figure `reliability_sensitivity_risk_vs_coverage.png`)

**This is a descriptive sensitivity analysis only. No cutoff is being
selected, recommended, or implied as better than the frozen one.** The
frozen operating point (99th percentile of clean-training instability,
the same cutoff used throughout this project) is marked with a star on
the figure and flagged in the CSV (`is_frozen_operating_point`), not
picked out because it performs best on this sweep.

**Scope limitation, stated plainly:** only the *instability* cutoff is
swept across percentiles (50th–99.99th) of the same clean-training
distribution the frozen 99th-percentile cutoff was drawn from. The
missingness (0.20) and disagreement (3.0) cutoffs stay fixed at their
frozen values throughout — this sweep says nothing about sensitivity to
those two gates.

That scope limitation is not academic: for **missing_sensor_subset**,
coverage is 0.0 and risk is undefined at *every single percentile
tested, including the most lenient (99.99th)* — because that condition's
30%-of-sensors-missing state trips the fixed missingness cutoff on every
row regardless of how the instability cutoff is set. The sweep is flat
and uninformative for this condition by construction, and that flatness
is itself the honest finding, not a gap in the analysis to paper over.

For the other four conditions (clean, additive_noise, conflicting_evidence,
sensor_dropout), the pattern is consistent: coverage rises from roughly
0.18–0.42 at the strictest tested cutoff (50th percentile) to a plateau
around 0.84–0.87 by the 99.5th percentile, and instability alone stops
having further effect beyond that point (any remaining abstention/escalation
there comes from the missingness/disagreement gates, unaffected by this
sweep). Selective risk is **not** monotonic with coverage in these four
conditions — it decreases from the strictest setting to a minimum
somewhere around the 95th–99th percentile, then edges back up slightly as
coverage plateaus. This is reported as an observed pattern in this
specific sweep, on this specific validation set, with 5 onset events —
not as evidence that any particular point on the curve is preferable.

## Caveats

- All figures are on 2018 validation only, with the same 5 onset events
  used throughout this project — small-sample variation applies to every
  number here exactly as it did in earlier gates.
- "Selective risk" and "coverage" must be read together. A lower risk
  figure earned by a smaller autonomous denominator is not a safety
  improvement by itself — the missing_sensor_subset case is the sharpest
  illustration, but the general point holds throughout: C's numbers
  everywhere are conditioned on the rows it agreed to decide.
- The sensitivity sweep varies one of three gating signals; conclusions
  from it do not extend to the missingness or disagreement gates without
  a separate sweep over those.
