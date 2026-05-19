# Models

This directory contains MC-TRCM and the reference models shown or audited for the paper.

- `mctrcm_v2.py`: current MC-TRCM architecture with modality tokens, missingness modeling, FiLM conditioning, and recursive decoding.
- `mctrcm_data.py` and `mctrcm_v2_loader.py`: canonical multitask data loading.
- `train_mctrcm_v2.py`: primary MC-TRCM training entry point.
- `baselines.py`, `run_baselines.py`, and `capacity_aligned_baselines.py`: tabular and MLP baselines.
- `sequence_baselines.py` and `paper_sequence_baselines.py`: GRU, LSTM, and Transformer references.
- `run_deprest_cat_public_baseline.py`, `run_psyche_d_public_baseline.py`, and `run_depresjon_public_baseline.py`: dataset-paper sidecar references.

The main ICONIP benchmark uses the DepreST-CAT and PSYCHE-D tasks, while the other dataset parsers remain available for benchmark framing and descriptive checks.
