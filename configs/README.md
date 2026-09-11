# Experiment configuration

The codebase predates a single universal YAML runner, so the frozen scientific constants remain defined in the tested Python modules that execute them. The YAML files in this directory are **protocol manifests**: human- and machine-readable records of the settings used by the scripts, not a second source of decision logic.

- `wds_reliability.yaml` — WDS split, target, predictor, systems and degradation settings.
- `hazard_context.yaml` — precipitation calibration/validation and integration semantics.

Executable entry points are listed in the repository README. A reproducibility wrapper captures the main script outputs under `results/raw/`.
