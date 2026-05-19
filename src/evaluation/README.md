# Evaluation

Evaluation code shared by the paper-facing workflows.

- `metrics.py`: regression, binary, ordinal, and multiclass metrics.
- `protocol_alignment.py`: validation-selected baseline helpers and protocol constants.
- `compare_mctrcm_to_baselines.py`: joins MC-TRCM results against validation-selected baseline rows.
- `run_monitored_command.py`: lightweight helper for long-running commands.

Paper-level orchestration lives in `scripts/`, especially `run_core_mctrcm_protocol.py`.
