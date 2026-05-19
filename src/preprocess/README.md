# Preprocessing

This directory converts public dataset releases into the canonical feature-view tables expected by MC-TRCM and the baseline runners.

Main entry points:

- `download_datasets.py` and `extract_archives.py`: optional helpers for public source artifacts.
- `preprocess_studentlife.py`, `preprocess_deprest_cat.py`, `preprocess_psyche_d.py`, `preprocess_depresjon.py`, and `preprocess_obf.py`: dataset-specific parsers.
- `run_available_parsers.py`: runs the available dataset parsers in sequence.
- `generate_splits.py`: writes participant-level train/validation/test split manifests.
- `audit_temporal_leakage.py`: checks split and timing risks.
- `schema.py` and `base.py`: shared canonical table definitions.

Generated outputs normally go under `data_interim/`; raw source files should stay under the configured raw-data locations.
