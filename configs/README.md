# Configurations

Configuration files are grouped by purpose:

- `dataset_configs/`: dataset paths, download targets, labels, and feature-view metadata.
- `model_configs/`: MC-TRCM and ablation model structures.
- `train_configs/`: optimizer, sampling, calibration, and early-stopping settings.
- `split_configs/`: participant-level split defaults.
- `protocols/`: validation-locked comparison protocol definitions.

Prefer changing configuration files over hard-coding experiment choices in training scripts.
