"""
Machine-checked audit of the officially published BattLeDIM leak-configuration
and sensor-topology specification files.

Source: https://github.com/KIOS-Research/BattLeDIM (master branch).

This script performs NO training, NO labeling decisions, and NO threshold
invention. It only parses and cross-checks facts that are explicitly present
in the official configuration files, and is meant to be re-run by anyone
verifying the claims in docs/data_audit.md.

Usage:
    python scripts/audit/audit_battledim_config.py /path/to/BattLeDIM-master

where /path/to/BattLeDIM-master is a checkout (or extracted archive) of
https://github.com/KIOS-Research/BattLeDIM — this repository does not vendor
that checkout; see data/PROVENANCE.md for the licence/acquisition situation.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import yaml


def load_yalm(path: Path) -> dict:
    with open(path, "r", encoding="latin1") as f:
        return yaml.safe_load(f.read())


def parse_leak_lines(raw_lines: list[str]) -> list[dict]:
    """raw_lines: list of comma-joined strings; first entry is a header comment."""
    out = []
    for line in raw_lines[1:]:
        parts = [p.strip() for p in line.split(",")]
        link_id, start, end, diam, ltype, peak = parts
        out.append(
            {
                "link_id": link_id,
                "start": datetime.strptime(start, "%Y-%m-%d %H:%M"),
                "end": datetime.strptime(end, "%Y-%m-%d %H:%M"),
                "diameter_m": float(diam),
                "type": ltype,
                "peak": datetime.strptime(peak, "%Y-%m-%d %H:%M"),
            }
        )
    return out


def parse_plain_leak_file(path: Path) -> list[dict]:
    """For files like leakages_info.yalm: a flat comment + CSV lines, not a
    nested YAML 'leakages:' key."""
    out = []
    with open(path, "r", encoding="latin1") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            link_id, start, end, diam, ltype, peak = parts
            out.append(
                {
                    "link_id": link_id,
                    "start": datetime.strptime(start, "%Y-%m-%d %H:%M"),
                    "end": datetime.strptime(end, "%Y-%m-%d %H:%M"),
                    "diameter_m": float(diam),
                    "type": ltype,
                    "peak": datetime.strptime(peak, "%Y-%m-%d %H:%M"),
                }
            )
    return out


def overlaps(leaks: list[dict]) -> list[tuple[str, str]]:
    pairs = []
    for i in range(len(leaks)):
        for j in range(i + 1, len(leaks)):
            a, b = leaks[i], leaks[j]
            if a["start"] < b["end"] and b["start"] < a["end"]:
                pairs.append((a["link_id"], b["link_id"]))
    return pairs


def report(name: str, leaks: list[dict], boundary_end: datetime) -> set[str]:
    print(f"\n--- {name} ---")
    print(f"Total leak events: {len(leaks)}")
    ids = set(l["link_id"] for l in leaks)
    print(f"Unique pipes: {len(ids)}")
    by_type: dict[str, int] = {}
    for l in leaks:
        by_type[l["type"]] = by_type.get(l["type"], 0) + 1
    print(f"By type: {by_type}")
    censored = [l for l in leaks if l["end"] == boundary_end]
    print(
        "Leaks whose endTime == file's EndTime boundary "
        f"(right-censored / never repaired within window): {len(censored)}"
    )
    print(f"  -> pipes: {[l['link_id'] for l in censored]}")
    ov = overlaps(leaks)
    print(f"Number of overlapping (concurrent) leak-interval pairs: {len(ov)}")
    diam = [l["diameter_m"] for l in leaks]
    print(f"Leak diameter range (m): {min(diam):.6f} - {max(diam):.6f}")
    incipient = [l for l in leaks if l["type"] == "incipient"]
    if incipient:
        ramp_days = [(l["peak"] - l["start"]).total_seconds() / 86400 for l in incipient]
        print(f"Incipient ramp-up (start->peak) range: {min(ramp_days):.2f} - {max(ramp_days):.2f} days")
    abrupt = [l for l in leaks if l["type"] == "abrupt"]
    non_instant = [l for l in abrupt if l["peak"] != l["start"]]
    print(f"Abrupt leaks where peakTime != startTime (expected 0): {len(non_instant)}")
    return ids


def main(repo_root: Path) -> None:
    gen = repo_root / "Dataset Generator"
    score = repo_root / "Scoring Algorithm" / "competition_data"

    full = load_yalm(gen / "dataset_configuration.yalm")
    full_leaks = parse_leak_lines(full["leakages"])
    full_ids = report("FULL (2018-01-01 to 2019-12-31)", full_leaks, datetime(2019, 12, 31, 23, 55))
    print(
        f"\nSensor counts (full config): pressure={len(full['pressure_sensors'])}, "
        f"flow={len(full.get('flow_sensors', []))}, level={len(full.get('level_sensors', []))}, "
        f"amr={len(full.get('amrs', []))}"
    )

    hist = load_yalm(gen / "dataset_configuration_historical.yalm")
    hist_leaks = parse_leak_lines(hist["leakages"])
    hist_ids = report("HISTORICAL / 2018", hist_leaks, datetime(2018, 12, 31, 23, 55))

    ev = load_yalm(gen / "dataset_configuration_evaluation.yalm")
    ev_leaks = parse_leak_lines(ev["leakages"])
    ev_ids = report("EVALUATION / 2019", ev_leaks, datetime(2019, 12, 31, 23, 55))

    print("\n--- Cross-year overlap ---")
    carried_over = hist_ids & ev_ids
    print(
        "Pipes with a leak event present in BOTH the 2018 and 2019 configs "
        f"(same/continuing leak straddling the year boundary): {len(carried_over)} "
        f"-> {sorted(carried_over)}"
    )
    print(
        f"{len(hist_ids)}+{len(ev_ids)}-overlap check: "
        f"{len(hist_ids)} + {len(ev_ids)} - {len(carried_over)} = "
        f"{len(hist_ids) + len(ev_ids) - len(carried_over)} (expect {len(full_ids)})"
    )

    score_leaks = parse_plain_leak_file(score / "leakages_info.yalm")
    score_ids = set(l["link_id"] for l in score_leaks)
    print("\n--- Cross-check: Scoring Algorithm/competition_data/leakages_info.yalm ---")
    print(f"Entries: {len(score_leaks)}")
    print(f"Identical pipe ID set to dataset_configuration_evaluation.yalm? {score_ids == ev_ids}")
    score_by_id = {l["link_id"]: l for l in score_leaks}
    ev_by_id = {l["link_id"]: l for l in ev_leaks}
    mismatches = [pid for pid in score_ids & ev_ids if score_by_id[pid] != ev_by_id[pid]]
    print(f"Field-level mismatches between the two independently-located files: {len(mismatches)}")
    for pid in mismatches:
        print(f"  {pid}:")
        print(f"    dataset_configuration_evaluation.yalm: {ev_by_id[pid]}")
        print(f"    competition leakages_info.yalm       : {score_by_id[pid]}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(Path(sys.argv[1]))
