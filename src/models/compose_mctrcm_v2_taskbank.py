from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

from src.models.mctrcm_data import TaskMetadata
from src.models.train_mctrcm_v2 import (
    CHECKPOINT_DIR,
    CONCEPT_DIR,
    DEFAULT_MODEL_CONFIG_PATH,
    DEFAULT_TRAIN_CONFIG_PATH,
    EXPERIMENT_REGISTRY_PATH,
    LOG_DIR,
    PREDICTION_DIR,
    RESULTS_TABLE_PATH,
    TABLE_DIR,
    _append_registry_row,
    _append_result_rows,
    _dataset_balanced_score,
    _load_json,
    _metric_frame_to_records,
    _write_concept_summary,
    _write_vs_baseline_summary,
)
from src.utils.constants import PROJECT_ROOT
from src.utils.io import ensure_dir, write_json

MODEL_CONFIG_PATH = DEFAULT_MODEL_CONFIG_PATH
TRAIN_CONFIG_PATH = DEFAULT_TRAIN_CONFIG_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compose a task-bank MC-TRCM-v2 run from existing runs.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--base-run-name", required=True)
    parser.add_argument("--donor-run-name", required=True)
    parser.add_argument("--task-keys", nargs="+", required=True)
    parser.add_argument("--datasets", nargs="*", default=["deprest_cat", "psyche_d"])
    parser.add_argument("--copy-task-embedding", action="store_true")
    parser.add_argument("--copy-psyche-shared", action="store_true")
    parser.add_argument("--copy-alpha", type=float, default=1.0)
    parser.add_argument("--notes", type=str, default="")
    return parser.parse_args()


def _prediction_path(run_name: str, split: str) -> Path:
    return PREDICTION_DIR / f"{run_name}__{split}.csv"


def _metric_path(run_name: str) -> Path:
    return PREDICTION_DIR / f"{run_name}__metrics.csv"


def _summary_path(run_name: str) -> Path:
    return LOG_DIR / f"{run_name}__summary.json"


def _config_path(run_name: str) -> Path:
    return LOG_DIR / f"{run_name}__config.json"


def _calibration_path(run_name: str) -> Path:
    return TABLE_DIR / f"{run_name}__calibration_summary.csv"


def _load_summary(run_name: str) -> dict[str, object]:
    return json.loads(_summary_path(run_name).read_text(encoding="utf-8"))


def _load_config(run_name: str) -> dict[str, object]:
    return json.loads(_config_path(run_name).read_text(encoding="utf-8"))


def _load_task_metadata_from_checkpoint(summary: dict[str, object]) -> dict[int, TaskMetadata]:
    checkpoint_path = Path(str(summary["checkpoint_path"]))
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return {
        int(item["task_index"]): TaskMetadata(**item)
        for item in checkpoint["task_metadata"]
    }


def _checkpoint_path_from_summary(summary: dict[str, object]) -> Path:
    checkpoint_path = Path(str(summary["checkpoint_path"]))
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path
    if checkpoint_path.exists():
        return checkpoint_path
    fallback_path = CHECKPOINT_DIR / checkpoint_path.name
    if fallback_path.exists():
        return fallback_path
    return checkpoint_path


def _compose_prediction_frame(
    *,
    base_frame: pd.DataFrame,
    donor_frame: pd.DataFrame,
    task_keys: set[str],
) -> pd.DataFrame:
    join_columns = ["split", "dataset_id", "subject_id", "anchor_id", "task_key"]
    donor_subset = donor_frame.loc[donor_frame["task_key"].isin(task_keys)].copy()
    base_subset = base_frame.loc[base_frame["task_key"].isin(task_keys), join_columns].copy()
    if donor_subset.empty:
        raise ValueError(f"Donor frame does not contain any requested task_keys={sorted(task_keys)}")
    merged = base_subset.merge(
        donor_subset[join_columns],
        on=join_columns,
        how="outer",
        indicator=True,
    )
    if (merged["_merge"] != "both").any():
        missing = merged.loc[merged["_merge"] != "both", join_columns + ["_merge"]]
        raise ValueError(
            "Base and donor prediction rows do not align for the selected tasks.\n"
            f"{missing.head(10).to_string(index=False)}"
        )
    kept = base_frame.loc[~base_frame["task_key"].isin(task_keys)].copy()
    combined = pd.concat([kept, donor_subset], ignore_index=True)
    return combined.sort_values(["split", "dataset_id", "task_name", "subject_id", "anchor_id"]).reset_index(drop=True)


def _compose_metric_frame(
    *,
    base_frame: pd.DataFrame,
    donor_frame: pd.DataFrame,
    task_names: set[str],
) -> pd.DataFrame:
    kept = base_frame.loc[~base_frame["task_name"].isin(task_names)].copy()
    donor_subset = donor_frame.loc[donor_frame["task_name"].isin(task_names)].copy()
    combined = pd.concat([kept, donor_subset], ignore_index=True)
    return combined.sort_values(["split", "dataset_id", "task_name"]).reset_index(drop=True)


def _compose_calibration_frame(
    *,
    base_run_name: str,
    donor_run_name: str,
    run_name: str,
    task_names: set[str],
) -> pd.DataFrame:
    base_path = _calibration_path(base_run_name)
    donor_path = _calibration_path(donor_run_name)
    if not base_path.exists() or not donor_path.exists():
        return pd.DataFrame()
    base_frame = pd.read_csv(base_path)
    donor_frame = pd.read_csv(donor_path)
    kept = base_frame.loc[~base_frame["task_name"].isin(task_names)].copy()
    donor_subset = donor_frame.loc[donor_frame["task_name"].isin(task_names)].copy()
    combined = pd.concat([kept, donor_subset], ignore_index=True)
    if "run_name" in combined.columns:
        combined["run_name"] = run_name
    combined = combined.sort_values(["dataset_id", "task_name"]).reset_index(drop=True)
    combined.to_csv(_calibration_path(run_name), index=False)
    return combined


def _save_transplanted_checkpoint(
    *,
    run_name: str,
    base_summary: dict[str, object],
    donor_summary: dict[str, object],
    task_metadata: dict[int, TaskMetadata],
    task_keys: set[str],
    copy_task_embedding: bool = False,
    copy_psyche_shared: bool = False,
    copy_alpha: float = 1.0,
) -> Path:
    base_checkpoint = torch.load(_checkpoint_path_from_summary(base_summary), map_location="cpu")
    donor_checkpoint = torch.load(_checkpoint_path_from_summary(donor_summary), map_location="cpu")
    base_state = dict(base_checkpoint["model_state_dict"])
    donor_state = donor_checkpoint["model_state_dict"]
    alpha = float(copy_alpha)

    def _mix_value(base_value: torch.Tensor, donor_value: torch.Tensor) -> torch.Tensor:
        if alpha >= 1.0:
            return donor_value.clone()
        if alpha <= 0.0:
            return base_value.clone()
        return (1.0 - alpha) * base_value + alpha * donor_value

    for task_key in sorted(task_keys):
        task_index = next(
            int(index)
            for index, meta in task_metadata.items()
            if str(meta.task_key) == task_key
        )
        prefixes = [
            f"task_heads.{task_index}.",
            f"output_refiners.{task_index}.",
            f"deprest_severity_heads.{task_index}.",
            f"deprest_pair_heads.{task_index}.",
        ]
        for key, value in donor_state.items():
            if any(key.startswith(prefix) for prefix in prefixes):
                base_value = base_state.get(key)
                if base_value is None or tuple(base_value.shape) != tuple(value.shape):
                    continue
                base_state[key] = _mix_value(base_value, value)
        if copy_task_embedding:
            embedding_key = "task_embedding.weight"
            if embedding_key in base_state and embedding_key in donor_state:
                base_state[embedding_key][task_index] = _mix_value(
                    base_state[embedding_key][task_index],
                    donor_state[embedding_key][task_index],
                )
        if "task_log_vars" in base_checkpoint and "task_log_vars" in donor_checkpoint:
            base_checkpoint["task_log_vars"][task_index] = donor_checkpoint["task_log_vars"][task_index]
        if "calibration_params" in base_checkpoint and "calibration_params" in donor_checkpoint:
            donor_calibration = donor_checkpoint["calibration_params"].get(task_index)
            if donor_calibration is not None:
                base_checkpoint["calibration_params"][task_index] = donor_calibration

    if copy_psyche_shared:
        psyche_prefixes = [
            "psyche_end_head.",
            "psyche_end_adapter.",
            "psyche_start_embedding.",
            "psyche_start_score_encoder.",
            "psyche_two_stage_adapter.",
            "psyche_delta_bridge.",
        ]
        for key, value in donor_state.items():
            if any(key.startswith(prefix) for prefix in psyche_prefixes):
                base_state[key] = _mix_value(base_state[key], value)

    base_checkpoint["model_state_dict"] = base_state
    checkpoint_path = ensure_dir(CHECKPOINT_DIR) / f"{run_name}.pt"
    torch.save(base_checkpoint, checkpoint_path)
    return checkpoint_path


def main() -> None:
    args = parse_args()
    task_keys = {str(task_key).strip() for task_key in args.task_keys if str(task_key).strip()}
    task_names = {task_key.split("::", 1)[1] for task_key in task_keys}

    base_summary = _load_summary(args.base_run_name)
    donor_summary = _load_summary(args.donor_run_name)
    base_config = _load_config(args.base_run_name)
    donor_config = _load_config(args.donor_run_name)
    task_metadata = _load_task_metadata_from_checkpoint(base_summary)

    valid_base = pd.read_csv(_prediction_path(args.base_run_name, "valid"))
    valid_donor = pd.read_csv(_prediction_path(args.donor_run_name, "valid"))
    test_base = pd.read_csv(_prediction_path(args.base_run_name, "test"))
    test_donor = pd.read_csv(_prediction_path(args.donor_run_name, "test"))
    valid_predictions = _compose_prediction_frame(base_frame=valid_base, donor_frame=valid_donor, task_keys=task_keys)
    test_predictions = _compose_prediction_frame(base_frame=test_base, donor_frame=test_donor, task_keys=task_keys)

    metrics_base = pd.read_csv(_metric_path(args.base_run_name))
    metrics_donor = pd.read_csv(_metric_path(args.donor_run_name))
    metric_frame = _compose_metric_frame(base_frame=metrics_base, donor_frame=metrics_donor, task_names=task_names)
    valid_metrics = metric_frame.loc[metric_frame["split"] == "valid"].reset_index(drop=True)
    test_metrics = metric_frame.loc[metric_frame["split"] == "test"].reset_index(drop=True)

    ensure_dir(PREDICTION_DIR)
    ensure_dir(CONCEPT_DIR)
    ensure_dir(LOG_DIR)
    ensure_dir(TABLE_DIR)

    valid_predictions.to_csv(_prediction_path(args.run_name, "valid"), index=False)
    test_predictions.to_csv(_prediction_path(args.run_name, "test"), index=False)
    pd.concat([valid_predictions, test_predictions], ignore_index=True).to_csv(
        CONCEPT_DIR / f"{args.run_name}__concepts.csv",
        index=False,
    )
    metric_frame.to_csv(_metric_path(args.run_name), index=False)

    _compose_calibration_frame(
        base_run_name=args.base_run_name,
        donor_run_name=args.donor_run_name,
        run_name=args.run_name,
        task_names=task_names,
    )
    transplanted_checkpoint_path = _save_transplanted_checkpoint(
        run_name=args.run_name,
        base_summary=base_summary,
        donor_summary=donor_summary,
        task_metadata=task_metadata,
        task_keys=task_keys,
        copy_task_embedding=bool(args.copy_task_embedding),
        copy_psyche_shared=bool(args.copy_psyche_shared),
        copy_alpha=float(args.copy_alpha),
    )
    model_config = _load_json(MODEL_CONFIG_PATH)
    train_config = _load_json(TRAIN_CONFIG_PATH)
    model_name = str(model_config.get("model_name", "mctrcm_v2_taskbank"))
    concept_summary = _write_concept_summary(test_predictions, args.run_name)
    vs_baseline = _write_vs_baseline_summary(test_metrics, args.run_name, model_name)
    _append_result_rows(
        _metric_frame_to_records(
            metric_frame,
            run_name=args.run_name,
            datasets=args.datasets,
            model_config=model_config,
            train_config=train_config,
            parameter_count=int(base_summary["parameter_count"]),
            model_name=model_name,
        )
    )

    summary = {
        "run_name": args.run_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "composition_mode": "task_bank",
        "datasets": list(args.datasets),
        "base_run_name": args.base_run_name,
        "donor_run_name": args.donor_run_name,
        "task_keys": sorted(task_keys),
        "task_names": sorted(task_names),
        "copy_task_embedding": bool(args.copy_task_embedding),
        "copy_psyche_shared": bool(args.copy_psyche_shared),
        "copy_alpha": float(args.copy_alpha),
        "parameter_count": int(base_summary["parameter_count"]),
        "base_checkpoint_path": base_summary["checkpoint_path"],
        "donor_checkpoint_path": donor_summary["checkpoint_path"],
        "checkpoint_path": str(transplanted_checkpoint_path),
        "base_final_test_score": float(base_summary["final_test_score"]),
        "donor_final_test_score": float(donor_summary["final_test_score"]),
        "final_valid_score": _dataset_balanced_score(valid_metrics, task_metadata),
        "final_test_score": _dataset_balanced_score(test_metrics, task_metadata),
        "metric_path": str(_metric_path(args.run_name)),
        "vs_baseline_path": str((TABLE_DIR / f"{args.run_name}__vs_baseline.csv").relative_to(PROJECT_ROOT)),
        "concept_summary_path": str((CONCEPT_DIR / f"{args.run_name}__concept_summary.csv").relative_to(PROJECT_ROOT)),
        "notes": args.notes,
    }
    write_json(_summary_path(args.run_name), summary)
    write_json(
        _config_path(args.run_name),
        {
            "composition_mode": "task_bank",
            "base_run_name": args.base_run_name,
            "donor_run_name": args.donor_run_name,
            "task_keys": sorted(task_keys),
            "datasets": list(args.datasets),
            "copy_task_embedding": bool(args.copy_task_embedding),
            "copy_psyche_shared": bool(args.copy_psyche_shared),
            "copy_alpha": float(args.copy_alpha),
            "notes": args.notes,
            "base_run_config": base_config,
            "donor_run_config": donor_config,
        },
    )

    _append_registry_row(
        {
            "experiment_id": f"stage_13_taskbank__{args.run_name}__{datetime.now().strftime('%Y%m%dT%H%M%S')}",
            "stage": "stage_13_model_upgrade",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "model_name": "mctrcm_v2_task_bank",
            "training_corpora": "|".join(args.datasets),
            "target_dataset": "multi_corpus_bundle",
            "task_name": "task_bank_composition",
            "split_config": "participant_level_dataset_specific_splits",
            "model_config": str(MODEL_CONFIG_PATH.relative_to(PROJECT_ROOT)),
            "train_config": str(TRAIN_CONFIG_PATH.relative_to(PROJECT_ROOT)),
            "status": "completed",
            "notes": (
                f"base_run={args.base_run_name}; donor_run={args.donor_run_name}; "
                f"task_keys={'|'.join(sorted(task_keys))}; "
                f"copy_task_embedding={bool(args.copy_task_embedding)}; "
                f"copy_psyche_shared={bool(args.copy_psyche_shared)}; "
                f"copy_alpha={float(args.copy_alpha):.3f}; "
                f"final_test_score={summary['final_test_score']:.6f}; "
                f"registry_path={EXPERIMENT_REGISTRY_PATH.relative_to(PROJECT_ROOT)}; "
                f"results_path={RESULTS_TABLE_PATH.relative_to(PROJECT_ROOT)}"
            ),
        }
    )


if __name__ == "__main__":
    main()
