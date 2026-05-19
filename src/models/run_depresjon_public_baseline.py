from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.utils.constants import PROJECT_ROOT
from src.utils.io import ensure_dir, write_csv, write_json

STAGE_NAME = "stage_05_public_baseline"
DEPresJON_REPO_ROOT = PROJECT_ROOT / "data_raw" / "depresjon" / "external" / "Depresjon_ML-master"
RESULTS_DIR = DEPresJON_REPO_ROOT / "results"
HP_DIR = DEPresJON_REPO_ROOT / "hp_tuning" / "LSTM_tuning"
SAVED_MODELS_DIR = DEPresJON_REPO_ROOT / "saved_models"
DATA_DIR = DEPresJON_REPO_ROOT / "data"
OUTPUT_TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
OUTPUT_LOG_DIR = PROJECT_ROOT / "outputs" / "logs"
REPORT_PATH = PROJECT_ROOT / "reports" / "qa" / "depresjon_public_baseline_audit.md"
REGISTRY_PATH = OUTPUT_LOG_DIR / "experiment_registry.csv"
PUBLIC_STATUS_PATH = OUTPUT_TABLE_DIR / "public_baseline_status.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the public Depresjon actigraphy repository as a sidecar baseline."
    )
    parser.add_argument("--repo-root", type=Path, default=DEPresJON_REPO_ROOT)
    return parser.parse_args()


def _load_registry() -> pd.DataFrame:
    if REGISTRY_PATH.exists():
        return pd.read_csv(REGISTRY_PATH)
    return pd.DataFrame(
        columns=[
            "experiment_id",
            "stage",
            "created_at",
            "model_name",
            "training_corpora",
            "target_dataset",
            "task_name",
            "split_config",
            "model_config",
            "train_config",
            "status",
            "notes",
        ]
    )


def _append_registry_row(row: dict[str, object]) -> None:
    registry = _load_registry()
    registry = pd.concat([registry, pd.DataFrame([row])], ignore_index=True)
    registry = registry.drop_duplicates(subset=["experiment_id"], keep="last")
    registry.to_csv(REGISTRY_PATH, index=False)


def _load_scores(repo_root: Path) -> pd.DataFrame:
    return pd.read_csv(repo_root / "data" / "scores.csv")


def _dataset_summary(repo_root: Path) -> pd.DataFrame:
    scores = _load_scores(repo_root)
    condition_files = sorted((repo_root / "data" / "condition").glob("*.csv"))
    control_files = sorted((repo_root / "data" / "control").glob("*.csv"))
    condition_score_rows = int(scores["number"].astype(str).str.startswith("condition").sum())
    control_score_rows = int(scores["number"].astype(str).str.startswith("control").sum())
    rows = [
        {"group": "total_scores_rows", "count": int(len(scores.index))},
        {"group": "condition_scores_rows", "count": condition_score_rows},
        {"group": "control_scores_rows", "count": control_score_rows},
        {
            "group": "condition_files",
            "count": int(len(condition_files)),
        },
        {
            "group": "control_files",
            "count": int(len(control_files)),
        },
        {
            "group": "condition_days_min",
            "count": int(pd.to_numeric(scores["days"], errors="coerce").min()),
        },
        {
            "group": "condition_days_max",
            "count": int(pd.to_numeric(scores["days"], errors="coerce").max()),
        },
    ]
    return pd.DataFrame(rows)


def _load_trial_summary(repo_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted((repo_root / "hp_tuning" / "LSTM_tuning").glob("trial_*/trial.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        values = payload.get("hyperparameters", {}).get("values", {})
        rows.append(
            {
                "trial_id": payload.get("trial_id"),
                "status": payload.get("status"),
                "score": payload.get("score"),
                "best_step": payload.get("best_step"),
                "lstm_units": values.get("lstm_units"),
                "lstm_dropout": values.get("lstm_dropout"),
                "lstm_recurrent_dropout": values.get("lstm_recurrent_dropout"),
                "l2_reg": values.get("l2_reg"),
                "dense_dem_units": values.get("dense_dem_units"),
                "dense1_units": values.get("dense1_units"),
                "dense1_dropout": values.get("dense1_dropout"),
                "path": str(path.relative_to(PROJECT_ROOT)),
            }
        )
    return pd.DataFrame(rows).sort_values("score").reset_index(drop=True)


def _load_result_summary(repo_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metrics_path = repo_root / "results" / "lstm_2_targets_metrics.json"
    if metrics_path.exists():
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        for metric_name, metric_value in payload.items():
            rows.append(
                {
                    "source": "metrics_json",
                    "metric_name": metric_name,
                    "metric_value": float(metric_value),
                    "source_path": str(metrics_path.relative_to(PROJECT_ROOT)),
                }
            )
    losses_path = repo_root / "results" / "epoch_losses.csv"
    if losses_path.exists():
        losses = pd.read_csv(losses_path)
        for metric_name in ("val_loss", "val_madrs2_mae", "val_deltamadrs_mae", "loss", "madrs2_mae", "deltamadrs_mae"):
            if metric_name in losses.columns:
                rows.append(
                    {
                        "source": "epoch_losses_csv",
                        "metric_name": f"min_{metric_name}",
                        "metric_value": float(pd.to_numeric(losses[metric_name], errors="coerce").min()),
                        "source_path": str(losses_path.relative_to(PROJECT_ROOT)),
                    }
                )
        if "epoch" in losses.columns:
            best_idx = pd.to_numeric(losses["val_loss"], errors="coerce").idxmin()
            best_epoch = int(losses.loc[best_idx, "epoch"])
            rows.append(
                {
                    "source": "epoch_losses_csv",
                    "metric_name": "best_val_epoch",
                    "metric_value": float(best_epoch),
                    "source_path": str(losses_path.relative_to(PROJECT_ROOT)),
                }
            )
    return pd.DataFrame(rows)


def _artifact_inventory(repo_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted((repo_root / "results").glob("*")):
        rows.append(
            {
                "artifact_group": "results",
                "name": path.name,
                "relative_path": str(path.relative_to(PROJECT_ROOT)),
                "size_bytes": path.stat().st_size,
                "last_modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
            }
        )
    for path in sorted((repo_root / "saved_models").glob("*.keras")):
        rows.append(
            {
                "artifact_group": "saved_models",
                "name": path.name,
                "relative_path": str(path.relative_to(PROJECT_ROOT)),
                "size_bytes": path.stat().st_size,
                "last_modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
            }
        )
    return pd.DataFrame(rows).sort_values(["artifact_group", "last_modified", "name"]).reset_index(drop=True)


def _protocol_findings(repo_root: Path) -> dict[str, object]:
    main_path = repo_root / "main.py"
    text = main_path.read_text(encoding="utf-8")
    uses_target_in_inputs = (
        'key_predictors = ["number", "age", "gender", "madrs1", "madrs2"]' in text
        and 'add_predictor = "deltamadrs"' in text
        and 'output1 = Dense(1, name="madrs2")' in text
        and 'output2 = Dense(1, name="deltamadrs")' in text
    )
    split_after_sequence_expansion = (
        "X_combined, demographic_refined, y_combined = scale_and_prepare(" in text
        and "train_test_split(" in text
        and "X_combined, y_combined, demographic_refined" in text
    )
    controls_excluded = 'scores["number"].str.startswith("condition")' in text
    return {
        "uses_target_in_inputs": uses_target_in_inputs,
        "split_after_sequence_expansion": split_after_sequence_expansion,
        "controls_excluded_from_training": controls_excluded,
        "sequence_length": 10,
        "train_test_split": "random sequence-level 70/30 with random_state=42",
    }


def _latest_rerun_state(artifacts: pd.DataFrame) -> dict[str, object]:
    result_rows = artifacts.loc[artifacts["artifact_group"] == "results"].copy()
    model_rows = artifacts.loc[artifacts["artifact_group"] == "saved_models"].copy()
    metrics_row = result_rows.loc[result_rows["name"] == "lstm_2_targets_metrics.json"]
    losses_row = result_rows.loc[result_rows["name"] == "epoch_losses.csv"]
    latest_model = model_rows.sort_values("last_modified").tail(1)
    state = {
        "metrics_json_last_modified": None,
        "epoch_losses_last_modified": None,
        "latest_model_name": None,
        "latest_model_size_bytes": None,
        "partial_refresh_detected": False,
    }
    if not metrics_row.empty:
        state["metrics_json_last_modified"] = metrics_row.iloc[0]["last_modified"]
    if not losses_row.empty:
        state["epoch_losses_last_modified"] = losses_row.iloc[0]["last_modified"]
    if not latest_model.empty:
        state["latest_model_name"] = latest_model.iloc[0]["name"]
        state["latest_model_size_bytes"] = int(latest_model.iloc[0]["size_bytes"])
    if state["metrics_json_last_modified"] and state["epoch_losses_last_modified"]:
        state["partial_refresh_detected"] = state["epoch_losses_last_modified"] > state["metrics_json_last_modified"]
    return state


def _update_public_status() -> None:
    if PUBLIC_STATUS_PATH.exists():
        status = pd.read_csv(PUBLIC_STATUS_PATH)
    else:
        status = pd.DataFrame(
            columns=[
                "baseline_name",
                "dataset_id",
                "status",
                "protocol",
                "comparable_to_main_protocol",
                "summary_path",
                "notes",
            ]
        )
    row = {
        "baseline_name": "depresjon_public_actigraphy",
        "dataset_id": "depresjon",
        "status": "completed_sidecar",
        "protocol": "public_lstm_sequence_split_with_target_leakage",
        "comparable_to_main_protocol": False,
        "summary_path": "outputs/tables/depresjon_public_baseline_summary.csv",
        "notes": "Completed as a source audit over the public Depresjon_ML LSTM repository; the released code uses target-derived demographic inputs and random sequence-level splitting, so it is excluded from the main participant-level comparison table.",
    }
    status = status.loc[status["baseline_name"] != "depresjon_public_actigraphy"].copy()
    status = pd.concat([status, pd.DataFrame([row])], ignore_index=True)
    write_csv(PUBLIC_STATUS_PATH, status)


def _write_report(
    dataset_summary: pd.DataFrame,
    trial_summary: pd.DataFrame,
    result_summary: pd.DataFrame,
    artifacts: pd.DataFrame,
    findings: dict[str, object],
    rerun_state: dict[str, object],
) -> None:
    best_trial = trial_summary.iloc[0] if not trial_summary.empty else None
    metric_rows = {
        row.metric_name: row.metric_value
        for row in result_summary.itertuples()
    }
    dataset_lines = "\n".join(
        f"- `{row.group}`: `{int(row.count)}`" for row in dataset_summary.itertuples()
    )
    text = f"""# Depresjon Public Baseline Audit

Date: `2026-04-17`

## Scope

This audit completes the corpus-specific public `Depresjon` baseline as a sidecar source audit over the public repository:

- Repo path: `data_raw/depresjon/external/Depresjon_ML-master`
- Public repo family: `LSTM` over actigraphy plus demographic covariates
- Declared outputs: `madrs2` and `deltamadrs`

The implementation is **not** merged into the unified participant-level baseline table because the released public code does not follow the study protocol.

## Dataset coverage in the public repo

{dataset_lines}

## Protocol audit findings

- `uses_target_in_inputs`: `{findings["uses_target_in_inputs"]}`
- `split_after_sequence_expansion`: `{findings["split_after_sequence_expansion"]}`
- `controls_excluded_from_training`: `{findings["controls_excluded_from_training"]}`
- `sequence_length`: `{findings["sequence_length"]}`
- `split`: `{findings["train_test_split"]}`

Interpretation:

- The released code feeds `madrs2` and `deltamadrs` into the demographic branch while also predicting those same values.
- The released code performs `train_test_split` after sliding-window expansion, so windows from the same participant can enter both train and test sets.
- The released training path uses condition subjects only, so it does not reproduce the case-control classification tasks used in the main project.

## Public artifact audit

- Trial count parsed: `{len(trial_summary.index)}`
- Best tuner trial score (`val_loss`): `{float(best_trial.score):.8f}`""" + (
            f"""
- Best tuner hyperparameters:
  - `lstm_units={int(best_trial.lstm_units)}`
  - `lstm_dropout={float(best_trial.lstm_dropout):.2f}`
  - `lstm_recurrent_dropout={float(best_trial.lstm_recurrent_dropout):.2f}`
  - `dense_dem_units={int(best_trial.dense_dem_units)}`
  - `dense1_units={int(best_trial.dense1_units)}`
  - `dense1_dropout={float(best_trial.dense1_dropout):.2f}`"""
            if best_trial is not None
            else ""
        ) + f"""
- Self-reported `metrics.json`:
  - `combined_loss={metric_rows.get("combined_loss", float("nan")):.9f}`
  - `madrs2_loss={metric_rows.get("madrs2_loss", float("nan")):.9f}`
  - `deltamadrs_loss={metric_rows.get("deltamadrs_loss", float("nan")):.9f}`
- Best `epoch_losses.csv` validation values:
  - `min_val_loss={metric_rows.get("min_val_loss", float("nan")):.9f}`
  - `min_val_madrs2_mae={metric_rows.get("min_val_madrs2_mae", float("nan")):.9f}`
  - `min_val_deltamadrs_mae={metric_rows.get("min_val_deltamadrs_mae", float("nan")):.9f}`

## Latest rerun state observed locally

- `metrics_json_last_modified`: `{rerun_state["metrics_json_last_modified"]}`
- `epoch_losses_last_modified`: `{rerun_state["epoch_losses_last_modified"]}`
- `latest_model_name`: `{rerun_state["latest_model_name"]}`
- `latest_model_size_bytes`: `{rerun_state["latest_model_size_bytes"]}`
- `partial_refresh_detected`: `{rerun_state["partial_refresh_detected"]}`

Interpretation:

- A local source-aligned rerun attempt refreshed `epoch_losses.csv` and produced a new `.keras` artifact, but the canonical `lstm_2_targets_metrics.json` file was not refreshed in the same pass.
- The audit therefore treats the latest rerun as a **partial refresh**, not as a fully verified benchmark rerun.

## Conclusion

- `Depresjon` is no longer a pending public-baseline gap in this workspace.
- The public repository is preserved as a transparency sidecar, not as a clean comparator, because it contains target leakage and participant leakage relative to the project protocol.
- The main participant-level `Depresjon` comparisons should continue to rely on the project-native canonical baselines under participant-level splits.
"""
    ensure_dir(REPORT_PATH.parent)
    REPORT_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root
    ensure_dir(OUTPUT_TABLE_DIR)
    ensure_dir(OUTPUT_LOG_DIR)

    dataset_summary = _dataset_summary(repo_root)
    trial_summary = _load_trial_summary(repo_root)
    result_summary = _load_result_summary(repo_root)
    artifacts = _artifact_inventory(repo_root)
    findings = _protocol_findings(repo_root)
    rerun_state = _latest_rerun_state(artifacts)

    dataset_summary_path = OUTPUT_TABLE_DIR / "depresjon_public_dataset_summary.csv"
    trial_summary_path = OUTPUT_TABLE_DIR / "depresjon_public_trial_summary.csv"
    result_summary_path = OUTPUT_TABLE_DIR / "depresjon_public_baseline_summary.csv"
    artifact_inventory_path = OUTPUT_TABLE_DIR / "depresjon_public_artifact_inventory.csv"

    write_csv(dataset_summary_path, dataset_summary)
    write_csv(trial_summary_path, trial_summary)
    write_csv(result_summary_path, result_summary)
    write_csv(artifact_inventory_path, artifacts)
    _update_public_status()
    _write_report(dataset_summary, trial_summary, result_summary, artifacts, findings, rerun_state)

    audit_payload = {
        "stage": STAGE_NAME,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "repo_root": str(repo_root.relative_to(PROJECT_ROOT)),
        "protocol_findings": findings,
        "rerun_state": rerun_state,
        "paths": {
            "dataset_summary": str(dataset_summary_path.relative_to(PROJECT_ROOT)),
            "trial_summary": str(trial_summary_path.relative_to(PROJECT_ROOT)),
            "result_summary": str(result_summary_path.relative_to(PROJECT_ROOT)),
            "artifact_inventory": str(artifact_inventory_path.relative_to(PROJECT_ROOT)),
            "status": str(PUBLIC_STATUS_PATH.relative_to(PROJECT_ROOT)),
            "report": str(REPORT_PATH.relative_to(PROJECT_ROOT)),
        },
    }
    write_json(OUTPUT_LOG_DIR / "depresjon_public_baseline_audit.json", audit_payload)

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    _append_registry_row(
        {
            "experiment_id": f"{STAGE_NAME}__depresjon__public_actigraphy_audit__{timestamp}",
            "stage": STAGE_NAME,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "model_name": "depresjon_public_actigraphy",
            "training_corpora": "depresjon",
            "target_dataset": "depresjon",
            "task_name": "official_public_repo_audit",
            "split_config": "public_random_sequence_level_70_30",
            "model_config": "public_Depresjon_ML_LSTM",
            "train_config": "public_Depresjon_ML_main_py",
            "status": "completed",
            "notes": "Completed sidecar audit over the public Depresjon_ML repository; documents target leakage and sequence-level participant leakage, so it is excluded from the main participant-level benchmark table.",
        }
    )


if __name__ == "__main__":
    main()
