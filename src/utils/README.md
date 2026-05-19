# Utilities

Small helpers shared across preprocessing, model training, and evaluation.

- `constants.py`: project paths, dataset identifiers, modality names, window definitions, and concept labels.
- `io.py`: directory creation plus JSON, CSV, and parquet-with-fallback writers.

Dataset-specific or experiment-specific helpers should live beside the code that uses them.
