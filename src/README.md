# Source Package

This package contains the paper-facing implementation.

- `preprocess/`: public dataset parsers, canonical table builders, split generation, and leakage checks.
- `models/`: MC-TRCM, baselines, data loaders, and training entry points.
- `evaluation/`: shared metrics and validation-selection comparison helpers.
- `utils/`: project paths, constants, and small IO helpers.

Keep new code in the narrowest relevant subpackage instead of adding top-level modules.
