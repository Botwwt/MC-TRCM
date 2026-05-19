# Scripts

Paper-level utilities and experiment orchestration.

- `run_core_mctrcm_protocol.py`: core DepreST-CAT and PSYCHE-D protocol driver for search, final runs, ablations, feature-source controls, ensembles, and summaries.
- `run_revised_fairness_v1.py`: matched feature-source and calibration analysis.
- `calibrate_baseline_predictions.py`: validation-only calibration diagnostics for saved baseline predictions.
- `compute_seed_ensemble.py`, `compute_uncertainty.py`, and `compute_deprest_bootstrap_ci.py`: seed aggregation and uncertainty helpers.
- `generate_core_protocol_tables.py`: LaTeX table generation from core protocol outputs.
- `check_metric_integrity.py` and `validate_manuscript.py`: consistency checks.

Keep shell-specific orchestration here; reusable modeling or metric logic should stay under `src/`.
