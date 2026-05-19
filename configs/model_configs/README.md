# Model Configurations

Model JSON files define the MC-TRCM architecture used by training scripts.

- `mctrcm_final.json`: paper-facing full model configuration.
- `mctrcm_revised_k1_plain.json`: K=1 recursive-depth sensitivity configuration.
- `mctrcm_ablation_*.json`: single-component ablations for FiLM, missingness, task conditioning, and related checks.
- `baselines_default.json`: shared baseline defaults.

Use validation-locked selection before reading test metrics.
