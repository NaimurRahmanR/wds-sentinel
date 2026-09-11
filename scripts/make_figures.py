"""Generate publication-style figures from already-derived result tables."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "results" / "derived"
FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# 1) A/B/C coverage under each degradation condition.
df = pd.read_csv(DERIVED / "coverage_aware_evaluation_summary.csv")
pivot = df.pivot(index="condition", columns="system", values="autonomous_coverage")
ax = pivot.plot(kind="bar", figsize=(10, 5))
ax.set_ylabel("Autonomous coverage")
ax.set_xlabel("Evidence condition")
ax.set_ylim(0, 1.05)
ax.set_title("Autonomous coverage of A/B/C across evidence conditions")
ax.legend(title="System", fontsize=8)
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig(FIG / "abc_coverage_by_condition.png", dpi=180)
plt.close()

# 2) Event detection vs false-alert burden for A/B/C.
fig, ax = plt.subplots(figsize=(8, 6))
for system, sub in df.groupby("system"):
    ax.scatter(sub["false_alerts_per_day"], sub["event_detection_rate"], label=system, s=55)
    for _, row in sub.iterrows():
        ax.annotate(row["condition"], (row["false_alerts_per_day"], row["event_detection_rate"]), fontsize=6, alpha=0.8)
ax.set_xlabel("False alert episodes / day")
ax.set_ylabel("Event detection rate")
ax.set_title("Detection–false-alert trade-off on 2018 validation")
ax.legend(fontsize=8)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(FIG / "detection_false_alert_tradeoff.png", dpi=180)
plt.close(fig)

# 3) Integrated real-evidence scenario actions as a compact table figure.
sc = pd.read_csv(DERIVED / "hazard_integration_scenarios_summary.csv")
cols = ["scenario", "wds_state", "hazard_state", "integrated_action"]
show = sc[cols].copy()
show["scenario"] = show["scenario"].str.replace(r"^\d+_", "", regex=True).str.replace("_", " ")
fig, ax = plt.subplots(figsize=(12, 3.7))
ax.axis("off")
table = ax.table(cellText=show.values, colLabels=["Scenario", "WDS state", "Hazard state", "Integrated action"], loc="center", cellLoc="left")
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.45)
ax.set_title("Integrated scenarios use real WDS and real precipitation evidence", pad=15)
fig.tight_layout()
fig.savefig(FIG / "integrated_real_evidence_scenarios.png", dpi=180, bbox_inches="tight")
plt.close(fig)

print(f"Wrote figures to {FIG}")

# 4) Clean-condition temporal comparison: 2018 validation vs final 2019.
if (DERIVED / "final_2019_temporal_evaluation.csv").exists():
    y19 = pd.read_csv(DERIVED / "final_2019_temporal_evaluation.csv")
    d18 = df[df["condition"] == "clean"][["system", "autonomous_coverage", "event_detection_rate"]].copy()
    d18["period"] = "2018 validation"
    d19 = y19[y19["condition"] == "clean"][["system", "autonomous_coverage", "event_detection_rate"]].copy()
    d19["period"] = "2019 final"
    combo = pd.concat([d18, d19], ignore_index=True)
    # Long-form two-metric grouped bars on a single axes.
    rows = []
    for _, r in combo.iterrows():
        rows.append({"label": f"{r['system']}\n{r['period']}", "metric": "Event detection", "value": r["event_detection_rate"]})
        rows.append({"label": f"{r['system']}\n{r['period']}", "metric": "Autonomous coverage", "value": r["autonomous_coverage"]})
    long = pd.DataFrame(rows)
    labels = list(dict.fromkeys(long["label"]))
    x = range(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(12, 5.5))
    det = [long[(long.label == lab) & (long.metric == "Event detection")].value.iloc[0] for lab in labels]
    cov = [long[(long.label == lab) & (long.metric == "Autonomous coverage")].value.iloc[0] for lab in labels]
    ax.bar([i - width/2 for i in x], det, width=width, label="Event detection rate")
    ax.bar([i + width/2 for i in x], cov, width=width, label="Autonomous coverage")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate")
    ax.set_title("Clean-condition temporal comparison: 2018 validation vs frozen 2019")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "final_2019_clean_temporal_comparison.png", dpi=180)
    plt.close(fig)
