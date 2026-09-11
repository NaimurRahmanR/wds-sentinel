"""Run five integrated scenarios using REAL evidence on both sides.

WDS evidence comes from the actual 2018 BattLeDIM validation stream and
its existing controlled-degradation pipeline. Hazard evidence comes from
real CWA/CODIS precipitation observations (via the bundled third-party
rebuild). The two domains are intentionally independent; pairing them is
an interoperability/context experiment, not a geographic or causal claim.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wds_sentinel.agents.hazard_agent import HazardAgent  # noqa: E402
from wds_sentinel.agents.integrated_supervisory_agent import IntegratedSupervisoryAgent  # noqa: E402
from wds_sentinel.evidence.adapters import HazardEvidenceAdapter  # noqa: E402
from wds_sentinel.experiments.integrated import build_supervisor_from_real_run, first_real_case  # noqa: E402
from wds_sentinel.hazard.data import load_daily_precipitation  # noqa: E402
from wds_sentinel.hazard.detector import antecedent_precipitation, fit_hazard_thresholds, hazard_elevated_flag  # noqa: E402
from wds_sentinel.prediction.degradation import DegradedStreams, apply_condition  # noqa: E402
from wds_sentinel.prediction.frozen_predictor import build_raw_features, train_frozen_predictor  # noqa: E402
from wds_sentinel.reasoning.rules import corroboration_count  # noqa: E402
from wds_sentinel.reliability.control import fit_instability_cutoff  # noqa: E402
from wds_sentinel.reliability.systems import run_systems  # noqa: E402

DATA = ROOT / "data" / "raw"
HAZARD_DIR = ROOT / "data" / "external" / "hazard_taiwan_precip"
OUT_JSON = ROOT / "results" / "derived" / "hazard_integration_scenarios.json"
OUT_CSV = ROOT / "results" / "derived" / "hazard_integration_scenarios_summary.csv"
OUT_HAZARD = ROOT / "results" / "derived" / "hazard_validation_summary.csv"

# ---- real hazard calibration: 2013-2017 train, 2018 validation ----
train_precip = load_daily_precipitation([HAZARD_DIR / f"466920_{y}_daily.csv" for y in range(2013, 2018)])
val_precip = load_daily_precipitation([HAZARD_DIR / "466920_2018_daily.csv"])
train_ante = antecedent_precipitation(train_precip)
# prepend enough history to make Jan 2018 windows causal and fully defined
val_ante = antecedent_precipitation(pd.concat([train_precip.tail(7), val_precip])).loc[val_precip.index]
thresholds = fit_hazard_thresholds(train_ante)
val_flags = hazard_elevated_flag(val_ante, thresholds)

normal_dates = val_flags.index[val_flags.eq(False).fillna(False)]
elevated_dates = val_flags.index[val_flags.eq(True).fillna(False)]
if len(normal_dates) == 0 or len(elevated_dates) == 0:
    raise RuntimeError("hazard validation set must contain both NORMAL and ELEVATED days")
NORMAL_HAZARD_DATE = pd.Timestamp(normal_dates[0])
ELEVATED_HAZARD_DATE = pd.Timestamp(elevated_dates[0])

hazard_adapter = HazardEvidenceAdapter(
    name="cwa_codis_precip_466920",
    elevated_flag=val_flags,
    raw_value=val_precip,
    provenance_note="CWA/CODIS station 466920 observations via Raingel/historical_weather rebuild",
)
hazard_agent = HazardAgent(hazard_adapter)

hazard_summary = pd.DataFrame(
    [{
        "train_start": train_precip.index.min(),
        "train_end": train_precip.index.max(),
        "validation_start": val_precip.index.min(),
        "validation_end": val_precip.index.max(),
        "percentile": thresholds.percentile,
        "cutoff_1d_mm": thresholds.cutoffs["precip_sum_1d"],
        "cutoff_3d_mm": thresholds.cutoffs["precip_sum_3d"],
        "cutoff_7d_mm": thresholds.cutoffs["precip_sum_7d"],
        "validation_elevated_days": int(val_flags.eq(True).sum()),
        "validation_defined_days": int(val_flags.notna().sum()),
        "validation_total_days": int(len(val_flags)),
        "selected_normal_date": NORMAL_HAZARD_DATE,
        "selected_elevated_date": ELEVATED_HAZARD_DATE,
    }]
)
hazard_summary.to_csv(OUT_HAZARD, index=False)

# ---- real WDS pipeline: frozen model and real 2018 validation evidence ----
predictor, ctx = train_frozen_predictor(DATA)
pressure, flow, level = ctx["pressure"], ctx["flow"], ctx["level"]
label, masks = ctx["label"], ctx["masks"]
train_mask = masks["train"] & ctx["valid_mask"]
val_mask = masks["validation"]
train_stds = {
    "pressure": pressure[train_mask].std(),
    "flow": flow[train_mask].std(),
    "level": level[train_mask].std(),
}
instability_cutoff = fit_instability_cutoff(
    build_raw_features(pressure[train_mask], flow[train_mask], level[train_mask])
)

clean_streams = DegradedStreams(pressure.copy(), flow.copy(), level.copy(), "clean")
clean_run = run_systems(predictor, clean_streams, instability_cutoff)

# Apply the existing fixed missing-sensor degradation ONLY to validation rows.
missing_val = apply_condition(
    "missing_sensor_subset", pressure[val_mask], flow[val_mask], level[val_mask], train_stds
)
missing_pressure, missing_flow, missing_level = pressure.copy(), flow.copy(), level.copy()
missing_pressure.loc[val_mask] = missing_val.pressure
missing_flow.loc[val_mask] = missing_val.flow
missing_level.loc[val_mask] = missing_val.level
missing_streams = DegradedStreams(
    missing_pressure, missing_flow, missing_level, "missing_sensor_subset"
)
missing_run = run_systems(predictor, missing_streams, instability_cutoff)

val_index = clean_run.decision_C.index[clean_run.decision_C.index.isin(pressure.index[val_mask])]
clean_C = clean_run.decision_C.loc[val_index]
missing_C = missing_run.decision_C.reindex(val_index)
val_label = label.reindex(val_index).astype(int)

# Deterministic real-case selection. The ALERT case must be a true onset-window
# row, not a false alert. The degraded case must also be a true onset-window
# row and must trigger the reliability gate.
normal_wds_t = first_real_case(clean_C, val_label, decision="NO_ALERT", label_value=0)
alert_wds_t = first_real_case(clean_C, val_label, decision="ALERT", label_value=1)

degraded_candidates = val_index[(val_label.eq(1)) & missing_C.isin(["ABSTAIN", "ESCALATE"])]
if len(degraded_candidates) == 0:
    raise RuntimeError("no real degraded onset-window row found for missing_sensor_subset")
degraded_wds_t = pd.Timestamp(degraded_candidates[0])

supervisors = {
    "clean": build_supervisor_from_real_run(
        run=clean_run,
        raw_streams=clean_streams,
        predictor_threshold=predictor.threshold,
        instability_cutoff=instability_cutoff,
    ),
    "missing_sensor_subset": build_supervisor_from_real_run(
        run=missing_run,
        raw_streams=missing_streams,
        predictor_threshold=predictor.threshold,
        instability_cutoff=instability_cutoff,
    ),
}

scenario_specs = [
    ("1_normal_wds_normal_hazard", "clean", normal_wds_t, NORMAL_HAZARD_DATE),
    ("2_real_onset_alert_normal_hazard", "clean", alert_wds_t, NORMAL_HAZARD_DATE),
    ("3_elevated_hazard_normal_wds", "clean", normal_wds_t, ELEVATED_HAZARD_DATE),
    ("4_elevated_hazard_real_onset_alert", "clean", alert_wds_t, ELEVATED_HAZARD_DATE),
    ("5_elevated_hazard_degraded_real_onset", "missing_sensor_subset", degraded_wds_t, ELEVATED_HAZARD_DATE),
]

export: dict[str, dict] = {}
summary_rows = []
for name, condition, wds_t, hazard_t in scenario_specs:
    integrated = IntegratedSupervisoryAgent(supervisors[condition], hazard_agent)
    rec = integrated.decide(wds_t, hazard_t)
    wx = rec.wds_execution

    export[name] = {
        "wds_condition": condition,
        "wds_timestamp": str(wds_t),
        "wds_ground_truth_onset_window": int(label.loc[wds_t]),
        "hazard_timestamp": str(hazard_t),
        "evidence_message": {
            "missingness": wx.evidence_message.missingness,
            "availability": wx.evidence_message.availability,
            "disagreement": wx.evidence_message.disagreement,
            "instability": wx.evidence_message.instability,
            "n_evidence_items": len(wx.evidence_message.bundle.items),
        },
        "prediction_message": {
            "score": wx.prediction_message.score,
            "threshold": wx.prediction_message.threshold,
            "predictor_version": wx.prediction_message.predictor_version,
        },
        "reasoning_message": {
            "decision": wx.reasoning_message.decision,
            "facts": wx.reasoning_message.trace.facts,
            "fired_rules": wx.reasoning_message.trace.fired_rules,
            "contradictions": wx.reasoning_message.trace.contradictions,
            "corroboration_count": float(corroboration_count(
                (clean_run if condition == "clean" else missing_run).features
            ).loc[wds_t]),
        },
        "reliability_message": {
            "unreliable": wx.reliability_message.unreliable,
            "reasons": wx.reliability_message.reasons,
            "final_decision": wx.reliability_message.final_decision,
        },
        "hazard_message": {
            "state": rec.hazard_state,
            "available": rec.hazard_message.available,
            "elevated": rec.hazard_message.elevated,
            "daily_precip_mm": float(val_precip.loc[hazard_t]),
            "antecedent": {k: float(v) for k, v in val_ante.loc[hazard_t].to_dict().items()},
            "fired_rules": rec.hazard_trace.fired_rules,
        },
        "wds_operational_state": rec.wds_operational_state,
        "evidence_reliability_state": rec.evidence_reliability_state,
        "integrated_supervisory_action": rec.integrated_action,
        "combined_explanation": rec.combined_explanation,
        "wds_evidence": [
            {
                "source": e.source,
                "modality": e.modality,
                "value": e.value,
                "availability": e.availability,
                "quality": e.quality,
                "provenance": e.provenance,
            }
            for e in wx.record.evidence_refs
        ],
    }
    summary_rows.append({
        "scenario": name,
        "condition": condition,
        "wds_timestamp": wds_t,
        "ground_truth_onset_window": int(label.loc[wds_t]),
        "predictor_score": wx.prediction_message.score,
        "corroboration_count": export[name]["reasoning_message"]["corroboration_count"],
        "missingness": wx.evidence_message.missingness,
        "wds_state": rec.wds_operational_state,
        "hazard_timestamp": hazard_t,
        "hazard_state": rec.hazard_state,
        "integrated_action": rec.integrated_action,
    })

OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
OUT_JSON.write_text(json.dumps(export, indent=2, default=str))
pd.DataFrame(summary_rows).to_csv(OUT_CSV, index=False)

print(hazard_summary.to_string(index=False))
print(pd.DataFrame(summary_rows).to_string(index=False))
print(f"Wrote {OUT_JSON}")
print(f"Wrote {OUT_CSV}")
