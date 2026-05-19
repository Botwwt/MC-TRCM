from __future__ import annotations

import argparse
import json
import math
import os
import random
import threading
import time
from collections import defaultdict
from dataclasses import asdict, replace
from datetime import datetime
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.isotonic import IsotonicRegression
from torch import nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from src.evaluation.metrics import compute_metrics
from src.evaluation.protocol_alignment import select_validation_best_baselines
from src.models.baseline_data import TaskBundle, load_task_bundle
from src.models.baselines import build_estimator
from src.models.capacity_aligned_baselines import fit_capacity_aligned_baseline
from src.models.mctrcm_data import PreparedMultiCorpusData, TaskMetadata, prepare_multicorpus_data
from src.models.mctrcm_v2 import MCTRCMV2, TaskHeadSpecV2
from src.models.sequence_baselines import fit_sequence_baseline, prepare_sequence_arrays
from src.utils.constants import PROJECT_ROOT, concept_keys_for_dim
from src.utils.io import ensure_dir, write_json

CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"
PREDICTION_DIR = PROJECT_ROOT / "outputs" / "predictions" / "mctrcm_v2"
CONCEPT_DIR = PROJECT_ROOT / "outputs" / "concept_exports"
LOG_DIR = PROJECT_ROOT / "outputs" / "logs"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
DEFAULT_MODEL_CONFIG_PATH = PROJECT_ROOT / "configs" / "model_configs" / "mctrcm_v2_family_a.json"
DEFAULT_TRAIN_CONFIG_PATH = PROJECT_ROOT / "configs" / "train_configs" / "mctrcm_v2_family_a.json"
EXPERIMENT_REGISTRY_PATH = LOG_DIR / "experiment_registry.csv"
RESULTS_TABLE_PATH = TABLE_DIR / "mctrcm_v2_results.csv"
BASELINE_RESULTS_PATH = TABLE_DIR / "baseline_results.csv"
TEACHER_ROOT = PROJECT_ROOT / "outputs" / "predictions" / "teacher_cache"
_PROGRESS_LOG_PATH: Path | None = None
_STALL_SECONDS: float | None = None
_LAST_PROGRESS_TS: float = time.time()
_LAST_PROGRESS_MESSAGE: str = "process initialized"
_STALL_THREAD: threading.Thread | None = None


def _progress(message: str) -> None:
    global _LAST_PROGRESS_TS
    global _LAST_PROGRESS_MESSAGE
    timestamp = datetime.now().strftime("%H:%M:%S")
    line = f"[train_mctrcm_v2 {timestamp}] {message}"
    _LAST_PROGRESS_TS = time.time()
    _LAST_PROGRESS_MESSAGE = line
    print(line, flush=True)
    if _PROGRESS_LOG_PATH is not None:
        with _PROGRESS_LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def _set_progress_log_path(path: str | None) -> None:
    global _PROGRESS_LOG_PATH
    if path is None or not str(path).strip():
        _PROGRESS_LOG_PATH = None
        return
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    ensure_dir(resolved.parent)
    resolved.write_text("", encoding="utf-8")
    _PROGRESS_LOG_PATH = resolved


def _stall_watchdog_loop() -> None:
    while True:
        time.sleep(1.0)
        if _STALL_SECONDS is None or _STALL_SECONDS <= 0:
            return
        elapsed = time.time() - _LAST_PROGRESS_TS
        if elapsed <= _STALL_SECONDS:
            continue
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = (
            f"[train_mctrcm_v2 {timestamp}] "
            f"Stall watchdog triggered after {int(_STALL_SECONDS)}s without progress. "
            f"Last progress: {_LAST_PROGRESS_MESSAGE}"
        )
        try:
            print(line, flush=True)
            if _PROGRESS_LOG_PATH is not None:
                with _PROGRESS_LOG_PATH.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
        finally:
            os._exit(124)


def _set_stall_watchdog(stall_seconds: int | None) -> None:
    global _STALL_SECONDS
    global _STALL_THREAD
    if stall_seconds is None or int(stall_seconds) <= 0:
        _STALL_SECONDS = None
        _STALL_THREAD = None
        return
    _STALL_SECONDS = float(int(stall_seconds))
    if _STALL_THREAD is not None and _STALL_THREAD.is_alive():
        return
    _STALL_THREAD = threading.Thread(
        target=_stall_watchdog_loop,
        name="mctrcm_v2_stall_watchdog",
        daemon=True,
    )
    _STALL_THREAD.start()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MC-TRCM-v2 Family A.")
    parser.add_argument("--datasets", nargs="*", default=["deprest_cat", "psyche_d"])
    parser.add_argument("--model-config-path", type=str, default=None)
    parser.add_argument("--train-config-path", type=str, default=None)
    parser.add_argument("--model-family-name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--token-dim", type=int, default=None)
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--encoder-hidden-dim", type=int, default=None)
    parser.add_argument("--transformer-ff-dim", type=int, default=None)
    parser.add_argument("--output-refine-dim", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260417)
    parser.add_argument("--run-name", type=str, default="mctrcm_v2_family_a_round1")
    parser.add_argument("--progress-log-path", type=str, default=None)
    parser.add_argument("--stall-seconds", type=int, default=None)
    parser.add_argument("--init-checkpoint", type=str, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--use-checkpoint-calibration", action="store_true")
    parser.add_argument(
        "--skip-test-eval",
        action="store_true",
        help="Write validation predictions/metrics only. Use for validation-only hyperparameter search.",
    )
    parser.add_argument("--recursion-steps", type=int, default=None)
    parser.add_argument("--concept-dim", type=int, default=None)
    parser.add_argument("--disable-concept-bottleneck", action="store_true")
    parser.add_argument("--temporal-frontend-mode", choices=["none", "gru", "recency_pool"], default=None)
    parser.add_argument("--enable-missingness-tokens", action="store_true")
    parser.add_argument("--enable-missingness-embedding", action="store_true")
    parser.add_argument("--disable-missingness-tokens", action="store_true")
    parser.add_argument("--disable-missingness-embedding", action="store_true")
    parser.add_argument("--missingness-embedding-norm", action="store_true")
    parser.add_argument("--modality-dropout", type=float, default=None)
    parser.add_argument("--conditioning-mode", choices=["dataset_task", "dataset_only", "task_only", "none"], default=None)
    parser.add_argument("--film-conditioning-mode", choices=["both", "pre", "post", "none"], default=None)
    parser.add_argument("--disable-film-conditioning", action="store_true")
    parser.add_argument("--enable-tree-gated-head", action="store_true")
    parser.add_argument("--tree-head-task-names", nargs="*", default=None)
    parser.add_argument("--tree-head-depth", type=int, default=None)
    parser.add_argument("--enable-ordered-threshold-ordinal-head", action="store_true")
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--min-epochs", type=int, default=None)
    parser.add_argument("--disable-pcgrad", action="store_true")
    parser.add_argument("--use-gradnorm", action="store_true")
    parser.add_argument("--disable-uncertainty-weighting", action="store_true")
    parser.add_argument("--task-sampling-power", type=float, default=None)
    parser.add_argument("--classification-sampling-power", type=float, default=None)
    parser.add_argument("--regression-loss", choices=["huber", "mse", "mae", "mae_huber"], default=None)
    parser.add_argument("--distill-mode", choices=["none", "strongest_clean"], default=None)
    parser.add_argument("--train-task-keys", nargs="*", default=None)
    parser.add_argument("--distill-task-keys", nargs="*", default=None)
    parser.add_argument("--distill-task-weights", nargs="*", default=None)
    parser.add_argument("--task-loss-weight-specs", nargs="*", default=None)
    parser.add_argument("--distill-classification-weight", type=float, default=None)
    parser.add_argument("--distill-regression-weight", type=float, default=None)
    parser.add_argument("--l1-penalty", type=float, default=None)
    parser.add_argument("--group-penalty", type=float, default=None)
    parser.add_argument("--disable-sparse-penalty", action="store_true")
    parser.add_argument("--sparse-warmup-epochs", type=int, default=None)
    parser.add_argument("--psyche-hierarchical-weight", type=float, default=None)
    parser.add_argument("--disable-native-branch", action="store_true")
    parser.add_argument(
        "--feature-source-condition",
        choices=[
            "FULL",
            "SENSOR_VALUES_ONLY",
            "SENSOR_PLUS_MISSINGNESS",
            "MISSINGNESS_ONLY",
            "SYMPTOM_STATIC_CLINICAL",
        ],
        default="FULL",
        help=(
            "Restrict MC-TRCM input sources for matched feature-source controls. "
            "Selection, calibration, and final evaluation remain governed by the caller's split protocol."
        ),
    )
    parser.add_argument("--base-communication-only", action="store_true")
    parser.add_argument("--disable-psyche-hierarchical", action="store_true")
    parser.add_argument("--enable-psyche-two-stage", action="store_true")
    parser.add_argument("--enable-psyche-binary-correction", action="store_true")
    parser.add_argument("--enable-psyche-native-adapter", action="store_true")
    parser.add_argument("--enable-psyche-taskwise-native-adapter", action="store_true")
    parser.add_argument("--psyche-native-task-names", nargs="*", default=None)
    parser.add_argument("--disable-deprest-adapter", action="store_true")
    parser.add_argument("--enable-deprest-category-adapter", action="store_true")
    parser.add_argument("--enable-deprest-coverage-routing", action="store_true")
    parser.add_argument("--deprest-coverage-task-names", nargs="*", default=None)
    parser.add_argument("--enable-deprest-concept-compatibility", action="store_true")
    parser.add_argument("--deprest-concept-compat-task-names", nargs="*", default=None)
    parser.add_argument("--enable-deprest-concept-contrast", action="store_true")
    parser.add_argument("--deprest-concept-contrast-task-names", nargs="*", default=None)
    parser.add_argument("--enable-deprest-gad7-reg-bridge", action="store_true")
    parser.add_argument("--enable-deprest-severity-head", action="store_true")
    parser.add_argument("--deprest-severity-task-names", nargs="*", default=None)
    parser.add_argument("--enable-deprest-ordinal-hybrid-loss", action="store_true")
    parser.add_argument("--deprest-ordinal-hybrid-weight", type=float, default=None)
    parser.add_argument("--ordinal-hybrid-task-keys", nargs="*", default=None)
    parser.add_argument("--ordinal-hybrid-weight", type=float, default=None)
    parser.add_argument("--enable-ordinal-aux-class-head", action="store_true")
    parser.add_argument("--ordinal-aux-class-task-keys", nargs="*", default=None)
    parser.add_argument("--ordinal-aux-class-weight", type=float, default=None)
    parser.add_argument("--ordinal-labeldist-task-keys", nargs="*", default=None)
    parser.add_argument("--ordinal-labeldist-weight", type=float, default=None)
    parser.add_argument("--ordinal-labeldist-sigma", type=float, default=None)
    parser.add_argument("--ordinal-labeldist-edge-sigma", type=float, default=None)
    parser.add_argument("--ordinal-edge-boost-task-keys", nargs="*", default=None)
    parser.add_argument("--ordinal-edge-boost-factor", type=float, default=None)
    parser.add_argument("--binary-focal-task-keys", nargs="*", default=None)
    parser.add_argument("--binary-focal-gamma", type=float, default=None)
    parser.add_argument("--ordinal-focal-task-keys", nargs="*", default=None)
    parser.add_argument("--ordinal-focal-gamma", type=float, default=None)
    parser.add_argument("--multiclass-focal-task-keys", nargs="*", default=None)
    parser.add_argument("--multiclass-focal-gamma", type=float, default=None)
    parser.add_argument("--multiclass-label-smoothing-task-keys", nargs="*", default=None)
    parser.add_argument("--multiclass-label-smoothing", type=float, default=None)
    parser.add_argument("--class-balance-mode", choices=["inverse", "effective_number", "none"], default=None)
    parser.add_argument("--class-balance-beta", type=float, default=None)
    parser.add_argument("--max-class-weight", type=float, default=None)
    parser.add_argument("--enable-deprest-gad7-pair-aux", action="store_true")
    parser.add_argument("--deprest-gad7-pair-weight", type=float, default=None)
    parser.add_argument("--freeze-shared", action="store_true")
    parser.add_argument("--freeze-nontarget-heads", action="store_true")
    parser.add_argument("--enable-concept-residual", action="store_true")
    parser.add_argument("--enable-deprest-edge-ovr", action="store_true")
    parser.add_argument("--deprest-edge-ovr-task-names", nargs="*", default=None)
    parser.add_argument("--deprest-edge-ovr-weight", type=float, default=None)
    parser.add_argument("--shared-token-refiner-type", choices=["none", "mlp", "attention"], default=None)
    parser.add_argument("--shared-token-refiner-steps", type=int, default=None)
    parser.add_argument("--shared-token-refiner-nograd-steps", type=int, default=None)
    parser.add_argument("--task-local-trm-mode", choices=["none", "mlp", "attention"], default=None)
    parser.add_argument("--task-local-trm-reasoning-dim", type=int, default=None)
    parser.add_argument("--task-local-trm-h-cycles", type=int, default=None)
    parser.add_argument("--task-local-trm-l-cycles", type=int, default=None)
    parser.add_argument("--task-local-trm-task-names", nargs="*", default=None)
    parser.add_argument("--enable-deprest-construct-bridge", action="store_true")
    parser.add_argument("--deprest-construct-task-names", nargs="*", default=None)
    parser.add_argument("--deprest-construct-weight", type=float, default=None)
    parser.add_argument("--enable-psyche-delta-bridge", action="store_true")
    parser.add_argument("--psyche-delta-weight", type=float, default=None)
    parser.add_argument("--enable-deprest-edge-specialist", action="store_true")
    parser.add_argument("--deprest-edge-task-names", nargs="*", default=None)
    parser.add_argument("--ordinal-calibration-mode", choices=["scalar", "vector"], default=None)
    parser.add_argument("--auto-ordinal-bias-calibration", action="store_true")
    parser.add_argument("--ordinal-bias-task-keys", nargs="*", default=None)
    parser.add_argument("--fixed-ordinal-threshold-specs", nargs="*", default=None)
    parser.add_argument("--fixed-binary-threshold-specs", nargs="*", default=None)
    parser.add_argument("--binary-threshold-prevalence-tolerance", type=float, default=None)
    parser.add_argument("--fixed-multiclass-bias-specs", nargs="*", default=None)
    parser.add_argument("--regression-calibration-task-keys", nargs="*", default=None)
    parser.add_argument("--regression-calibration-mode", choices=["affine", "isotonic"], default=None)
    parser.add_argument("--paired-regression-bridge-specs", nargs="*", default=None)
    parser.add_argument("--multiclass-bias-task-keys", nargs="*", default=None)
    parser.add_argument("--multiclass-task-keys", nargs="*", default=None)
    return parser.parse_args()


def _load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_project_path(path_value: str | None, default_path: Path) -> Path:
    if not path_value:
        return default_path
    resolved = Path(path_value)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _move_batch(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    moved: dict[str, object] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _load_registry() -> pd.DataFrame:
    if EXPERIMENT_REGISTRY_PATH.exists():
        return pd.read_csv(EXPERIMENT_REGISTRY_PATH)
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
    registry.to_csv(EXPERIMENT_REGISTRY_PATH, index=False)


def _load_results_table() -> pd.DataFrame:
    if RESULTS_TABLE_PATH.exists():
        return pd.read_csv(RESULTS_TABLE_PATH)
    return pd.DataFrame()


def _append_result_rows(rows: list[dict[str, object]]) -> None:
    results = _load_results_table()
    results = pd.concat([results, pd.DataFrame(rows)], ignore_index=True)
    results = results.drop_duplicates(subset=["run_name", "split", "dataset_id", "task_name"], keep="last")
    ensure_dir(RESULTS_TABLE_PATH.parent)
    results.to_csv(RESULTS_TABLE_PATH, index=False)


def _sanitize_name(value: object) -> str:
    sanitized = "".join(char if str(char).isalnum() or str(char) == "-" else "_" for char in str(value))
    return sanitized.strip("_") or "value"


def _count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _derive_deprest_comm_groups(columns: list[str]) -> dict[str, list[int]]:
    call_indices: list[int] = []
    text_indices: list[int] = []
    meta_indices: list[int] = []
    for index, column in enumerate(columns):
        if "_call_" in column and "_coverage_" not in column and "_share_" not in column:
            call_indices.append(index)
            continue
        if "_text_" in column and "_coverage_" not in column and "_share_" not in column:
            text_indices.append(index)
            continue
        if any(token in column for token in ("_coverage_", "_share_", "_outgoing_incoming_", "_duration_", "_contacts_")):
            meta_indices.append(index)
            continue
    return {
        "call": call_indices,
        "text": text_indices,
        "meta": meta_indices,
    }


def _primary_metric_name(label_type: str) -> str:
    return "r2" if label_type == "continuous" else "balanced_accuracy"


def _task_metrics_score(label_type: str, metrics: dict[str, float]) -> float:
    if label_type == "continuous":
        r2 = metrics.get("r2")
        if r2 is not None and not np.isnan(r2):
            return float(r2)
        rmse = metrics.get("rmse")
        return float(-rmse) if rmse is not None and not np.isnan(rmse) else -1e9
    balanced_accuracy = metrics.get("balanced_accuracy")
    return float(balanced_accuracy) if balanced_accuracy is not None and not np.isnan(balanced_accuracy) else -1e9


def _build_task_head_specs(task_metadata: list[TaskMetadata]) -> list[TaskHeadSpecV2]:
    specs: list[TaskHeadSpecV2] = []
    for meta in task_metadata:
        if meta.label_type == "continuous":
            output_dim = 1
            num_classes = None
        elif meta.label_type == "binary":
            output_dim = 1
            num_classes = len(meta.class_space or [0, 1])
        elif meta.label_type == "ordinal":
            num_classes = len(meta.class_space or [])
            output_dim = max(num_classes - 1, 1)
        else:
            num_classes = len(meta.class_space or [])
            output_dim = num_classes
        specs.append(
            TaskHeadSpecV2(
                task_index=meta.task_index,
                label_type=meta.label_type,
                num_classes=num_classes,
                output_dim=output_dim,
            )
        )
    return specs


def _task_output_dim(meta: TaskMetadata) -> int:
    if meta.label_type == "continuous":
        return 1
    if meta.label_type == "binary":
        return 1
    if meta.label_type == "ordinal":
        return max(len(meta.class_space or []) - 1, 1)
    return len(meta.class_space or [])


def _parse_task_float_list_specs(items: list[str] | None) -> dict[str, list[float]]:
    specs: dict[str, list[float]] = {}
    for item in items or []:
        raw_item = str(item).strip()
        if not raw_item or "=" not in raw_item:
            continue
        task_name, values = raw_item.split("=", 1)
        parsed_values = [
            float(value)
            for value in str(values).split(",")
            if str(value).strip()
        ]
        if parsed_values:
            specs[str(task_name).strip()] = parsed_values
    return specs


def _parse_task_scalar_specs(items: list[str] | None) -> dict[str, float]:
    specs: dict[str, float] = {}
    for item in items or []:
        raw_item = str(item).strip()
        if not raw_item or "=" not in raw_item:
            continue
        task_name, value = raw_item.split("=", 1)
        task_name = str(task_name).strip()
        if not task_name:
            continue
        specs[task_name] = float(value)
    return specs


def _parse_task_pair_specs(items: list[str] | None) -> dict[str, str]:
    specs: dict[str, str] = {}
    for item in items or []:
        raw_item = str(item).strip()
        if not raw_item or "=" not in raw_item:
            continue
        target_task_key, source_task_key = raw_item.split("=", 1)
        target_task_key = str(target_task_key).strip()
        source_task_key = str(source_task_key).strip()
        if not target_task_key or not source_task_key:
            continue
        specs[target_task_key] = source_task_key
    return specs


def _resolve_task_override(meta: TaskMetadata, overrides: dict[str, list[float]] | None) -> list[float] | None:
    if not overrides:
        return None
    if meta.task_key in overrides:
        return overrides[meta.task_key]
    if meta.task_name in overrides:
        return overrides[meta.task_name]
    return None


def _apply_task_label_overrides(
    prepared: PreparedMultiCorpusData,
    *,
    multiclass_task_keys: list[str] | None = None,
) -> dict[str, str]:
    overrides: dict[str, str] = {}
    multiclass_set = {
        str(task_key).strip()
        for task_key in (multiclass_task_keys or [])
        if str(task_key).strip()
    }
    if not multiclass_set:
        return overrides

    matched_keys = {meta.task_key for meta in prepared.task_metadata if meta.task_key in multiclass_set}
    missing_keys = sorted(multiclass_set - matched_keys)
    if missing_keys:
        raise ValueError(f"No matching task keys found for multiclass-task-keys={missing_keys}")

    updated_metadata: list[TaskMetadata] = []
    for meta in prepared.task_metadata:
        label_type = "multiclass" if meta.task_key in multiclass_set else meta.label_type
        updated_meta = replace(
            meta,
            label_type=label_type,
            output_dim=_task_output_dim(replace(meta, label_type=label_type)),
        )
        updated_metadata.append(updated_meta)
        if label_type != meta.label_type:
            overrides[meta.task_key] = label_type

    prepared.task_metadata = updated_metadata
    for dataset in (prepared.train, prepared.valid, prepared.test):
        dataset.task_metadata = {meta.task_index: meta for meta in updated_metadata}
        if overrides:
            dataset.frame.loc[dataset.frame["task_key"].isin(overrides.keys()), "label_type"] = dataset.frame.loc[
                dataset.frame["task_key"].isin(overrides.keys()),
                "task_key",
            ].map(overrides)
    return overrides


def _selected_task_index_set(task_metadata: list[TaskMetadata], selected_task_keys: list[str] | None) -> set[int]:
    if not selected_task_keys:
        return set()
    wanted = {str(task_key).strip() for task_key in selected_task_keys if str(task_key).strip()}
    return {
        int(meta.task_index)
        for meta in task_metadata
        if str(meta.task_key) in wanted
    }


def _task_subset_indices(dataset, selected_task_indices: set[int]) -> list[int]:
    if not selected_task_indices:
        return list(range(len(dataset)))
    return [
        int(index)
        for index, task_index in enumerate(dataset.task_index.tolist())
        if int(task_index) in selected_task_indices
    ]


def _freeze_for_target_task_refinement(
    model: MCTRCMV2,
    *,
    target_task_indices: set[int],
    freeze_shared: bool,
    freeze_nontarget_heads: bool,
) -> None:
    if freeze_shared:
        for parameter in model.shared_parameters():
            parameter.requires_grad = False

    if freeze_nontarget_heads:
        for task_id_str, module in model.task_heads.items():
            requires_grad = int(task_id_str) in target_task_indices
            for parameter in module.parameters():
                parameter.requires_grad = requires_grad
        for task_id_str, module in model.output_refiners.items():
            requires_grad = int(task_id_str) in target_task_indices
            for parameter in module.parameters():
                parameter.requires_grad = requires_grad
        for task_id_str, module in model.deprest_severity_score_heads.items():
            requires_grad = int(task_id_str) in target_task_indices
            for parameter in module.parameters():
                parameter.requires_grad = requires_grad
        for task_id_str, module in model.deprest_severity_residual_heads.items():
            requires_grad = int(task_id_str) in target_task_indices
            for parameter in module.parameters():
                parameter.requires_grad = requires_grad
        for task_id_str, parameter in model.deprest_severity_log_scales.items():
            parameter.requires_grad = int(task_id_str) in target_task_indices
        for task_id_str, module in model.deprest_concept_compat_modules.items():
            requires_grad = int(task_id_str) in target_task_indices
            for parameter in module.parameters():
                parameter.requires_grad = requires_grad
        for task_id_str, module in model.deprest_concept_contrast_modules.items():
            requires_grad = int(task_id_str) in target_task_indices
            for parameter in module.parameters():
                parameter.requires_grad = requires_grad
        for task_id_str, module in model.deprest_coverage_routing_modules.items():
            requires_grad = int(task_id_str) in target_task_indices
            for parameter in module.parameters():
                parameter.requires_grad = requires_grad
        for task_id_str, module in model.deprest_pair_heads.items():
            requires_grad = int(task_id_str) in target_task_indices
            for parameter in module.parameters():
                parameter.requires_grad = requires_grad
        for task_id_str, module in model.deprest_task_bridge_modules.items():
            requires_grad = int(task_id_str) in target_task_indices
            for parameter in module.parameters():
                parameter.requires_grad = requires_grad
        for task_id_str, module in model.psyche_binary_correction_heads.items():
            requires_grad = int(task_id_str) in target_task_indices
            for parameter in module.parameters():
                parameter.requires_grad = requires_grad
        for task_id_str, module in model.deprest_edge_specialist_heads.items():
            requires_grad = int(task_id_str) in target_task_indices
            for parameter in module.parameters():
                parameter.requires_grad = requires_grad
        for task_id_str, module in model.deprest_edge_ovr_heads.items():
            requires_grad = int(task_id_str) in target_task_indices
            for parameter in module.parameters():
                parameter.requires_grad = requires_grad
        for task_id_str, parameter in model.deprest_edge_ovr_mix_logits.items():
            parameter.requires_grad = int(task_id_str) in target_task_indices


def _load_compatible_model_state(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> tuple[list[str], list[str], list[str]]:
    def _adapt_recursive_input_weight(
        loaded_value: torch.Tensor,
        current_value: torch.Tensor,
    ) -> torch.Tensor | None:
        if not isinstance(model, MCTRCMV2):
            return None
        if model.use_concept_bottleneck:
            return None
        if loaded_value.ndim != 2 or current_value.ndim != 2:
            return None
        if loaded_value.shape[0] != current_value.shape[0]:
            return None
        if loaded_value.shape[1] <= current_value.shape[1]:
            return None
        latent_dim = int(getattr(model, "latent_dim", 0))
        output_refine_dim = int(getattr(model, "output_refine_dim", 0))
        conditioning_dim = int(model.dataset_embedding.embedding_dim + model.task_embedding.embedding_dim)
        retained_tail_dim = output_refine_dim + conditioning_dim
        removed_concept_dim = loaded_value.shape[1] - current_value.shape[1]
        if latent_dim <= 0 or removed_concept_dim <= 0:
            return None
        if current_value.shape[1] != latent_dim + retained_tail_dim:
            return None
        if loaded_value.shape[1] != latent_dim + removed_concept_dim + retained_tail_dim:
            return None
        return torch.cat(
            [
                loaded_value[:, :latent_dim],
                loaded_value[:, latent_dim + removed_concept_dim :],
            ],
            dim=1,
        )

    def _adapt_latent_task_adapter_param(
        key: str,
        current_value: torch.Tensor,
    ) -> torch.Tensor | None:
        if not isinstance(model, MCTRCMV2):
            return None
        if model.use_concept_bottleneck:
            return None
        concept_head_weight = state_dict.get("concept_head.weight")
        concept_head_bias = state_dict.get("concept_head.bias")
        concept_post_linear_weight = state_dict.get("concept_post.1.weight")
        concept_post_linear_bias = state_dict.get("concept_post.1.bias")
        concept_post_output_norm_weight = state_dict.get("concept_post.4.weight")
        concept_post_output_norm_bias = state_dict.get("concept_post.4.bias")
        if key == "latent_task_adapter.1.weight":
            if (
                concept_head_weight is None
                or concept_post_linear_weight is None
                or concept_head_weight.ndim != 2
                or concept_post_linear_weight.ndim != 2
                or current_value.ndim != 2
            ):
                return None
            latent_dim = int(getattr(model, "latent_dim", 0))
            conditioning_dim = int(model.dataset_embedding.embedding_dim + model.task_embedding.embedding_dim)
            if current_value.shape[1] != latent_dim + conditioning_dim:
                return None
            if concept_head_weight.shape[1] != latent_dim:
                return None
            if concept_post_linear_weight.shape[1] != concept_head_weight.shape[0]:
                return None
            adapted = current_value.clone()
            adapted.zero_()
            composed = concept_post_linear_weight @ concept_head_weight
            if composed.shape != (current_value.shape[0], latent_dim):
                return None
            adapted[:, :latent_dim] = composed.to(dtype=current_value.dtype, device=current_value.device)
            return adapted
        if key == "latent_task_adapter.1.bias":
            if (
                concept_head_weight is None
                or concept_head_bias is None
                or concept_post_linear_weight is None
                or concept_post_linear_bias is None
                or current_value.ndim != 1
            ):
                return None
            if concept_head_bias.ndim != 1 or concept_post_linear_bias.ndim != 1:
                return None
            if concept_post_linear_bias.shape[0] != current_value.shape[0]:
                return None
            if concept_post_linear_weight.shape[1] != concept_head_bias.shape[0]:
                return None
            adapted = concept_post_linear_bias + concept_post_linear_weight @ concept_head_bias
            return adapted.to(dtype=current_value.dtype, device=current_value.device)
        if key == "latent_task_adapter.4.weight":
            if (
                concept_post_output_norm_weight is None
                or concept_post_output_norm_weight.shape != current_value.shape
            ):
                return None
            return concept_post_output_norm_weight.to(dtype=current_value.dtype, device=current_value.device)
        if key == "latent_task_adapter.4.bias":
            if (
                concept_post_output_norm_bias is None
                or concept_post_output_norm_bias.shape != current_value.shape
            ):
                return None
            return concept_post_output_norm_bias.to(dtype=current_value.dtype, device=current_value.device)
        return None

    current_state = model.state_dict()
    compatible_state: dict[str, torch.Tensor] = {}
    skipped_shape_mismatch: list[str] = []
    skipped_missing_in_model: list[str] = []
    for key, value in state_dict.items():
        if key not in current_state:
            skipped_missing_in_model.append(key)
            continue
        if current_state[key].shape != value.shape:
            adapted_value = None
            if key == "recursive_cell.weight_ih":
                adapted_value = _adapt_recursive_input_weight(value, current_state[key])
            elif key.startswith("latent_task_adapter."):
                adapted_value = _adapt_latent_task_adapter_param(key, current_state[key])
            if adapted_value is None or adapted_value.shape != current_state[key].shape:
                skipped_shape_mismatch.append(key)
                continue
            compatible_state[key] = adapted_value
            continue
        compatible_state[key] = value
    for key in (
        "latent_task_adapter.1.weight",
        "latent_task_adapter.1.bias",
        "latent_task_adapter.4.weight",
        "latent_task_adapter.4.bias",
    ):
        if key in current_state and key not in compatible_state:
            adapted_value = _adapt_latent_task_adapter_param(key, current_state[key])
            if adapted_value is not None and adapted_value.shape == current_state[key].shape:
                compatible_state[key] = adapted_value
    missing_keys, unexpected_keys = model.load_state_dict(compatible_state, strict=False)
    return list(missing_keys), list(unexpected_keys), skipped_shape_mismatch + skipped_missing_in_model


def _fit_class_space(bundle: TaskBundle) -> tuple[np.ndarray, dict[object, int], bool]:
    combined = pd.concat(
        [
            bundle.train_frame[["y_raw"]],
            bundle.valid_frame[["y_raw"]],
            bundle.test_frame[["y_raw"]],
        ],
        ignore_index=True,
    )
    numeric_values = pd.to_numeric(combined["y_raw"], errors="coerce")
    if numeric_values.notna().all():
        class_space = np.sort(numeric_values.astype(int).unique())
        mapping = {value: index for index, value in enumerate(class_space.tolist())}
        return class_space, mapping, True

    class_space = np.sort(combined["y_raw"].astype(str).unique())
    mapping = {value: index for index, value in enumerate(class_space.tolist())}
    return class_space, mapping, False


def _encode_class_targets(
    frame: pd.DataFrame,
    mapping: dict[object, int],
    numeric_labels: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if numeric_labels:
        raw_values = pd.to_numeric(frame["y_raw"], errors="coerce").astype(int)
    else:
        raw_values = frame["y_raw"].astype(str)
    encoded = raw_values.map(mapping).to_numpy(dtype=int)
    return encoded, raw_values.to_numpy()


def _align_probabilities(probability_matrix: np.ndarray, trained_classes: np.ndarray, n_classes: int) -> np.ndarray:
    if probability_matrix.ndim == 1:
        probability_matrix = np.column_stack([1.0 - probability_matrix, probability_matrix])

    aligned = np.zeros((probability_matrix.shape[0], n_classes), dtype=float)
    for trained_index, trained_class in enumerate(trained_classes):
        if 0 <= int(trained_class) < n_classes:
            aligned[:, int(trained_class)] = probability_matrix[:, trained_index]
    zero_rows = aligned.sum(axis=1) <= 0.0
    if zero_rows.any():
        aligned[zero_rows] = 1.0 / n_classes
    aligned = aligned / aligned.sum(axis=1, keepdims=True)
    return aligned


def _dataset_balanced_sample_weights(
    train_frame: pd.DataFrame,
    dataset_priority: dict[str, float],
    *,
    task_sampling_power: float = 0.0,
    classification_sampling_power: float = 0.0,
    max_weight_multiplier: float | None = None,
) -> np.ndarray:
    dataset_counts = train_frame["dataset_id"].value_counts().to_dict()
    task_counts = train_frame["task_key"].value_counts().to_dict() if "task_key" in train_frame.columns else {}
    class_counts: dict[tuple[str, str], int] = {}
    if classification_sampling_power > 0.0 and {"task_key", "label_type", "y_raw"}.issubset(train_frame.columns):
        classification_frame = train_frame.loc[train_frame["label_type"] != "continuous"].copy()
        if not classification_frame.empty:
            class_counts = {
                (str(task_key), str(y_raw)): int(count)
                for (task_key, y_raw), count in classification_frame.groupby(["task_key", "y_raw"]).size().items()
            }
    weights = []
    for row in train_frame.to_dict(orient="records"):
        dataset_id = str(row["dataset_id"])
        priority = float(dataset_priority.get(dataset_id, 1.0))
        weight = priority / float(dataset_counts[dataset_id])
        task_key = str(row.get("task_key", ""))
        if task_sampling_power > 0.0 and task_key in task_counts:
            weight *= float(task_counts[task_key]) ** (-float(task_sampling_power))
        if (
            classification_sampling_power > 0.0
            and str(row.get("label_type")) != "continuous"
            and task_key
        ):
            class_count = class_counts.get((task_key, str(row.get("y_raw"))))
            if class_count is not None and class_count > 0:
                weight *= float(class_count) ** (-float(classification_sampling_power))
        weights.append(weight)
    weights_array = np.asarray(weights, dtype=np.float64)
    if weights_array.size == 0:
        return weights_array.astype(np.float32)
    weights_array = weights_array / max(float(np.mean(weights_array)), 1e-12)
    if max_weight_multiplier is not None and float(max_weight_multiplier) > 0.0:
        median_weight = max(float(np.median(weights_array)), 1e-12)
        weights_array = np.minimum(weights_array, median_weight * float(max_weight_multiplier))
    return weights_array.astype(np.float32)


def _class_weight_vector(
    counts: np.ndarray,
    *,
    mode: str = "inverse",
    beta: float = 0.999,
    max_weight: float | None = None,
) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float64)
    counts = np.where(counts > 0.0, counts, 1.0)
    normalized_mode = str(mode or "inverse").lower()
    if normalized_mode == "none":
        weights = np.ones_like(counts, dtype=np.float64)
    elif normalized_mode == "effective_number":
        beta = float(np.clip(beta, 0.0, 0.999999))
        if beta <= 0.0:
            weights = np.ones_like(counts, dtype=np.float64)
        else:
            effective_num = 1.0 - np.power(beta, counts)
            weights = (1.0 - beta) / np.clip(effective_num, a_min=1e-12, a_max=None)
    else:
        weights = float(np.sum(counts)) / (float(len(counts)) * counts)
    weights = weights / max(float(np.mean(weights)), 1e-12)
    if max_weight is not None and float(max_weight) > 0.0:
        weights = np.minimum(weights, float(max_weight))
        weights = weights / max(float(np.mean(weights)), 1e-12)
    return weights.astype(np.float32)


def _build_task_training_info(
    prepared: PreparedMultiCorpusData,
    *,
    class_balance_mode: str = "inverse",
    class_balance_beta: float = 0.999,
    max_class_weight: float | None = None,
) -> dict[int, dict[str, object]]:
    info: dict[int, dict[str, object]] = {}
    train_frame = prepared.train.frame.copy()
    for meta in prepared.task_metadata:
        task_frame = train_frame.loc[train_frame["task_key"] == meta.task_key].copy()
        task_info: dict[str, object] = {
            "task_key": meta.task_key,
            "label_type": meta.label_type,
            "task_name": meta.task_name,
            "dataset_id": meta.dataset_id,
            "class_space": meta.class_space,
        }
        if meta.label_type == "continuous":
            raw_targets = pd.to_numeric(task_frame["y_raw"], errors="coerce").to_numpy(dtype=np.float32)
            mean = float(np.nanmean(raw_targets)) if len(raw_targets) else 0.0
            std = float(np.nanstd(raw_targets)) if len(raw_targets) else 1.0
            if not np.isfinite(std) or std <= 0.0:
                std = 1.0
            task_info["target_mean"] = mean
            task_info["target_std"] = std
            info[meta.task_index] = task_info
            continue

        target_index = prepared.train.target_index[train_frame["task_key"] == meta.task_key]
        if meta.label_type == "binary":
            counts = np.bincount(target_index, minlength=2).astype(np.float32)
            weights = _class_weight_vector(
                counts,
                mode=class_balance_mode,
                beta=class_balance_beta,
                max_weight=max_class_weight,
            )
            task_info["pos_weight"] = float(weights[1] / max(float(weights[0]), 1e-8))
            task_info["class_weights"] = weights.tolist()
        elif meta.label_type == "ordinal":
            num_classes = len(meta.class_space or [])
            pos_weights = []
            for threshold in range(max(num_classes - 1, 1)):
                positives = max(int(np.sum(target_index > threshold)), 1)
                negatives = max(int(np.sum(target_index <= threshold)), 1)
                threshold_weights = _class_weight_vector(
                    np.asarray([negatives, positives], dtype=np.float32),
                    mode=class_balance_mode,
                    beta=class_balance_beta,
                    max_weight=max_class_weight,
                )
                pos_weights.append(float(threshold_weights[1] / max(float(threshold_weights[0]), 1e-8)))
            task_info["pos_weights"] = pos_weights
            counts = np.bincount(target_index, minlength=num_classes).astype(np.float32)
            weights = _class_weight_vector(
                counts,
                mode=class_balance_mode,
                beta=class_balance_beta,
                max_weight=max_class_weight,
            )
            task_info["ordinal_class_weights"] = weights.tolist()
            low_positives = max(int(np.sum(target_index == 0)), 1)
            low_negatives = max(int(len(target_index) - np.sum(target_index == 0)), 1)
            high_positives = max(int(np.sum(target_index == max(num_classes - 1, 0))), 1)
            high_negatives = max(int(len(target_index) - np.sum(target_index == max(num_classes - 1, 0))), 1)
            edge_low_weights = _class_weight_vector(
                np.asarray([low_negatives, low_positives], dtype=np.float32),
                mode=class_balance_mode,
                beta=class_balance_beta,
                max_weight=max_class_weight,
            )
            edge_high_weights = _class_weight_vector(
                np.asarray([high_negatives, high_positives], dtype=np.float32),
                mode=class_balance_mode,
                beta=class_balance_beta,
                max_weight=max_class_weight,
            )
            task_info["edge_low_pos_weight"] = float(edge_low_weights[1] / max(float(edge_low_weights[0]), 1e-8))
            task_info["edge_high_pos_weight"] = float(edge_high_weights[1] / max(float(edge_high_weights[0]), 1e-8))
        else:
            num_classes = len(meta.class_space or [])
            counts = np.bincount(target_index, minlength=num_classes).astype(np.float32)
            weights = _class_weight_vector(
                counts,
                mode=class_balance_mode,
                beta=class_balance_beta,
                max_weight=max_class_weight,
            )
            task_info["class_weights"] = weights.tolist()
        info[meta.task_index] = task_info
    return info


def _strongest_clean_teacher_map(dataset_ids: list[str]) -> dict[tuple[str, str], str]:
    frame = select_validation_best_baselines()
    frame = frame.loc[frame["dataset_id"].isin(dataset_ids)].copy()
    mapping = {}
    for row in frame.to_dict(orient="records"):
        mapping[(str(row["dataset_id"]), str(row["task_name"]))] = str(row["baseline_model_name"])
    return mapping


def _teacher_prediction_frame(
    base_frame: pd.DataFrame,
    *,
    dataset_id: str,
    task_name: str,
    split: str,
    label_type: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray | None,
    class_space: np.ndarray | None,
    teacher_model_name: str,
) -> pd.DataFrame:
    output = base_frame[["dataset_id", "subject_id", "anchor_id", "task_name", "split"]].copy()
    output["dataset_id"] = dataset_id
    output["task_name"] = task_name
    output["split"] = split
    output["label_type"] = label_type
    output["teacher_model_name"] = teacher_model_name
    output["y_true"] = y_true
    output["y_pred"] = y_pred
    if probabilities is not None and class_space is not None:
        for class_index, class_value in enumerate(class_space):
            output[f"proba_{_sanitize_name(class_value)}"] = probabilities[:, class_index]
    return output


def _fit_teacher_predictions(
    bundle: TaskBundle,
    model_name: str,
    seed: int,
) -> pd.DataFrame:
    if model_name in {"gru_small", "lstm_small", "transformer_small", "gru", "lstm", "transformer"}:
        sequence_arrays = prepare_sequence_arrays(bundle)
        fit_fn = fit_sequence_baseline if model_name in {"gru_small", "lstm_small", "transformer_small"} else fit_capacity_aligned_baseline
        fit_kwargs = {}
        if fit_fn is fit_capacity_aligned_baseline:
            fit_kwargs["sequence_length"] = sequence_arrays.train.shape[1]
        if bundle.label_type == "continuous":
            y_train = pd.to_numeric(bundle.train_frame["y_raw"], errors="coerce").to_numpy(dtype=float)
            y_valid = pd.to_numeric(bundle.valid_frame["y_raw"], errors="coerce").to_numpy(dtype=float)
            y_test = pd.to_numeric(bundle.test_frame["y_raw"], errors="coerce").to_numpy(dtype=float)
            outputs = fit_fn(
                model_name=model_name,
                label_type=bundle.label_type,
                x_train=sequence_arrays.train,
                x_valid=sequence_arrays.valid,
                x_test=sequence_arrays.test,
                y_train=y_train,
                y_valid=y_valid,
                y_test=y_test,
                n_classes=1,
                seed=seed,
                **fit_kwargs,
            )
            frames = [
                _teacher_prediction_frame(
                    bundle.train_frame,
                    dataset_id=bundle.dataset_id,
                    task_name=bundle.task_name,
                    split="train",
                    label_type=bundle.label_type,
                    y_true=y_train,
                    y_pred=outputs["train_pred"],
                    probabilities=None,
                    class_space=None,
                    teacher_model_name=model_name,
                ),
                _teacher_prediction_frame(
                    bundle.valid_frame,
                    dataset_id=bundle.dataset_id,
                    task_name=bundle.task_name,
                    split="valid",
                    label_type=bundle.label_type,
                    y_true=y_valid,
                    y_pred=outputs["valid_pred"],
                    probabilities=None,
                    class_space=None,
                    teacher_model_name=model_name,
                ),
                _teacher_prediction_frame(
                    bundle.test_frame,
                    dataset_id=bundle.dataset_id,
                    task_name=bundle.task_name,
                    split="test",
                    label_type=bundle.label_type,
                    y_true=y_test,
                    y_pred=outputs["test_pred"],
                    probabilities=None,
                    class_space=None,
                    teacher_model_name=model_name,
                ),
            ]
            return pd.concat(frames, ignore_index=True)

        class_space, class_mapping, numeric_labels = _fit_class_space(bundle)
        y_train, y_train_raw = _encode_class_targets(bundle.train_frame, class_mapping, numeric_labels)
        y_valid, y_valid_raw = _encode_class_targets(bundle.valid_frame, class_mapping, numeric_labels)
        y_test, y_test_raw = _encode_class_targets(bundle.test_frame, class_mapping, numeric_labels)
        outputs = fit_fn(
            model_name=model_name,
            label_type=bundle.label_type,
            x_train=sequence_arrays.train,
            x_valid=sequence_arrays.valid,
            x_test=sequence_arrays.test,
            y_train=y_train,
            y_valid=y_valid,
            y_test=y_test,
            n_classes=len(class_space),
            seed=seed,
            **fit_kwargs,
        )
        train_pred = outputs["train_pred"].astype(int)
        valid_pred = outputs["valid_pred"].astype(int)
        test_pred = outputs["test_pred"].astype(int)
        frames = [
            _teacher_prediction_frame(
                bundle.train_frame,
                dataset_id=bundle.dataset_id,
                task_name=bundle.task_name,
                split="train",
                label_type=bundle.label_type,
                y_true=y_train_raw,
                y_pred=class_space[train_pred],
                probabilities=outputs["train_proba"],
                class_space=class_space,
                teacher_model_name=model_name,
            ),
            _teacher_prediction_frame(
                bundle.valid_frame,
                dataset_id=bundle.dataset_id,
                task_name=bundle.task_name,
                split="valid",
                label_type=bundle.label_type,
                y_true=y_valid_raw,
                y_pred=class_space[valid_pred],
                probabilities=outputs["valid_proba"],
                class_space=class_space,
                teacher_model_name=model_name,
            ),
            _teacher_prediction_frame(
                bundle.test_frame,
                dataset_id=bundle.dataset_id,
                task_name=bundle.task_name,
                split="test",
                label_type=bundle.label_type,
                y_true=y_test_raw,
                y_pred=class_space[test_pred],
                probabilities=outputs["test_proba"],
                class_space=class_space,
                teacher_model_name=model_name,
            ),
        ]
        return pd.concat(frames, ignore_index=True)

    x_train = bundle.train_frame[bundle.feature_columns]
    x_valid = bundle.valid_frame[bundle.feature_columns]
    x_test = bundle.test_frame[bundle.feature_columns]
    if model_name == "mlp":
        x_train_array = x_train.to_numpy(dtype=np.float32)
        x_valid_array = x_valid.to_numpy(dtype=np.float32)
        x_test_array = x_test.to_numpy(dtype=np.float32)
        means = np.nanmean(x_train_array, axis=0)
        means = np.nan_to_num(means, nan=0.0)
        stds = np.nanstd(x_train_array, axis=0)
        stds = np.nan_to_num(stds, nan=1.0)
        stds[stds == 0.0] = 1.0

        def _normalize(array: np.ndarray) -> np.ndarray:
            normalized = (array - means) / stds
            return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        x_train_array = _normalize(x_train_array)
        x_valid_array = _normalize(x_valid_array)
        x_test_array = _normalize(x_test_array)
        if bundle.label_type == "continuous":
            y_train = pd.to_numeric(bundle.train_frame["y_raw"], errors="coerce").to_numpy(dtype=float)
            y_valid = pd.to_numeric(bundle.valid_frame["y_raw"], errors="coerce").to_numpy(dtype=float)
            y_test = pd.to_numeric(bundle.test_frame["y_raw"], errors="coerce").to_numpy(dtype=float)
            outputs = fit_capacity_aligned_baseline(
                model_name=model_name,
                label_type=bundle.label_type,
                x_train=x_train_array,
                x_valid=x_valid_array,
                x_test=x_test_array,
                y_train=y_train,
                y_valid=y_valid,
                y_test=y_test,
                n_classes=1,
                seed=seed,
            )
            frames = [
                _teacher_prediction_frame(bundle.train_frame, dataset_id=bundle.dataset_id, task_name=bundle.task_name, split="train", label_type=bundle.label_type, y_true=y_train, y_pred=outputs["train_pred"], probabilities=None, class_space=None, teacher_model_name=model_name),
                _teacher_prediction_frame(bundle.valid_frame, dataset_id=bundle.dataset_id, task_name=bundle.task_name, split="valid", label_type=bundle.label_type, y_true=y_valid, y_pred=outputs["valid_pred"], probabilities=None, class_space=None, teacher_model_name=model_name),
                _teacher_prediction_frame(bundle.test_frame, dataset_id=bundle.dataset_id, task_name=bundle.task_name, split="test", label_type=bundle.label_type, y_true=y_test, y_pred=outputs["test_pred"], probabilities=None, class_space=None, teacher_model_name=model_name),
            ]
            return pd.concat(frames, ignore_index=True)

        class_space, class_mapping, numeric_labels = _fit_class_space(bundle)
        y_train, y_train_raw = _encode_class_targets(bundle.train_frame, class_mapping, numeric_labels)
        y_valid, y_valid_raw = _encode_class_targets(bundle.valid_frame, class_mapping, numeric_labels)
        y_test, y_test_raw = _encode_class_targets(bundle.test_frame, class_mapping, numeric_labels)
        outputs = fit_capacity_aligned_baseline(
            model_name=model_name,
            label_type=bundle.label_type,
            x_train=x_train_array,
            x_valid=x_valid_array,
            x_test=x_test_array,
            y_train=y_train,
            y_valid=y_valid,
            y_test=y_test,
            n_classes=len(class_space),
            seed=seed,
        )
        train_pred = outputs["train_pred"].astype(int)
        valid_pred = outputs["valid_pred"].astype(int)
        test_pred = outputs["test_pred"].astype(int)
        frames = [
            _teacher_prediction_frame(bundle.train_frame, dataset_id=bundle.dataset_id, task_name=bundle.task_name, split="train", label_type=bundle.label_type, y_true=y_train_raw, y_pred=class_space[train_pred], probabilities=outputs["train_proba"], class_space=class_space, teacher_model_name=model_name),
            _teacher_prediction_frame(bundle.valid_frame, dataset_id=bundle.dataset_id, task_name=bundle.task_name, split="valid", label_type=bundle.label_type, y_true=y_valid_raw, y_pred=class_space[valid_pred], probabilities=outputs["valid_proba"], class_space=class_space, teacher_model_name=model_name),
            _teacher_prediction_frame(bundle.test_frame, dataset_id=bundle.dataset_id, task_name=bundle.task_name, split="test", label_type=bundle.label_type, y_true=y_test_raw, y_pred=class_space[test_pred], probabilities=outputs["test_proba"], class_space=class_space, teacher_model_name=model_name),
        ]
        return pd.concat(frames, ignore_index=True)

    estimator = build_estimator(
        model_name=model_name,
        label_type=bundle.label_type,
        seed=seed,
        n_classes=1 if bundle.label_type == "continuous" else len(_fit_class_space(bundle)[0]),
    )
    if bundle.label_type == "continuous":
        y_train = pd.to_numeric(bundle.train_frame["y_raw"], errors="coerce").to_numpy(dtype=float)
        y_valid = pd.to_numeric(bundle.valid_frame["y_raw"], errors="coerce").to_numpy(dtype=float)
        y_test = pd.to_numeric(bundle.test_frame["y_raw"], errors="coerce").to_numpy(dtype=float)
        estimator.fit(x_train, y_train)
        frames = [
            _teacher_prediction_frame(bundle.train_frame, dataset_id=bundle.dataset_id, task_name=bundle.task_name, split="train", label_type=bundle.label_type, y_true=y_train, y_pred=estimator.predict(x_train), probabilities=None, class_space=None, teacher_model_name=model_name),
            _teacher_prediction_frame(bundle.valid_frame, dataset_id=bundle.dataset_id, task_name=bundle.task_name, split="valid", label_type=bundle.label_type, y_true=y_valid, y_pred=estimator.predict(x_valid), probabilities=None, class_space=None, teacher_model_name=model_name),
            _teacher_prediction_frame(bundle.test_frame, dataset_id=bundle.dataset_id, task_name=bundle.task_name, split="test", label_type=bundle.label_type, y_true=y_test, y_pred=estimator.predict(x_test), probabilities=None, class_space=None, teacher_model_name=model_name),
        ]
        return pd.concat(frames, ignore_index=True)

    class_space, class_mapping, numeric_labels = _fit_class_space(bundle)
    y_train, y_train_raw = _encode_class_targets(bundle.train_frame, class_mapping, numeric_labels)
    y_valid, y_valid_raw = _encode_class_targets(bundle.valid_frame, class_mapping, numeric_labels)
    y_test, y_test_raw = _encode_class_targets(bundle.test_frame, class_mapping, numeric_labels)
    estimator = build_estimator(model_name=model_name, label_type=bundle.label_type, seed=seed, n_classes=len(class_space))
    estimator.fit(x_train, y_train)

    estimator_core = estimator.named_steps["model"] if hasattr(estimator, "named_steps") else estimator
    trained_classes = np.asarray(getattr(estimator_core, "classes_", np.arange(len(class_space))), dtype=int)
    train_proba = _align_probabilities(estimator.predict_proba(x_train), trained_classes, len(class_space))
    valid_proba = _align_probabilities(estimator.predict_proba(x_valid), trained_classes, len(class_space))
    test_proba = _align_probabilities(estimator.predict_proba(x_test), trained_classes, len(class_space))
    train_pred = train_proba.argmax(axis=1)
    valid_pred = valid_proba.argmax(axis=1)
    test_pred = test_proba.argmax(axis=1)
    frames = [
        _teacher_prediction_frame(bundle.train_frame, dataset_id=bundle.dataset_id, task_name=bundle.task_name, split="train", label_type=bundle.label_type, y_true=y_train_raw, y_pred=class_space[train_pred], probabilities=train_proba, class_space=class_space, teacher_model_name=model_name),
        _teacher_prediction_frame(bundle.valid_frame, dataset_id=bundle.dataset_id, task_name=bundle.task_name, split="valid", label_type=bundle.label_type, y_true=y_valid_raw, y_pred=class_space[valid_pred], probabilities=valid_proba, class_space=class_space, teacher_model_name=model_name),
        _teacher_prediction_frame(bundle.test_frame, dataset_id=bundle.dataset_id, task_name=bundle.task_name, split="test", label_type=bundle.label_type, y_true=y_test_raw, y_pred=class_space[test_pred], probabilities=test_proba, class_space=class_space, teacher_model_name=model_name),
    ]
    return pd.concat(frames, ignore_index=True)


def _build_teacher_cache(
    prepared: PreparedMultiCorpusData,
    task_metadata: dict[int, TaskMetadata],
    seed: int,
    cache_tag: str,
    enabled_task_keys: set[str] | None = None,
) -> tuple[dict[int, dict[str, object]], pd.DataFrame]:
    teacher_map = _strongest_clean_teacher_map(sorted(prepared.dataset_to_index.keys()))
    cache_dir = ensure_dir(TEACHER_ROOT / cache_tag)
    cache: dict[int, dict[str, object]] = {}
    summary_rows = []
    for task_index, meta in task_metadata.items():
        if enabled_task_keys is not None and str(meta.task_key) not in enabled_task_keys:
            continue
        teacher_model_name = teacher_map.get((meta.dataset_id, meta.task_name))
        if teacher_model_name is None:
            continue
        cache_path = cache_dir / f"{meta.dataset_id}__{meta.task_name}.csv"
        if cache_path.exists():
            teacher_frame = pd.read_csv(cache_path)
        else:
            bundle = load_task_bundle(meta.dataset_id, meta.task_name)
            teacher_frame = _fit_teacher_predictions(bundle, teacher_model_name, seed)
            teacher_frame.to_csv(cache_path, index=False)
        split_maps: dict[str, dict[str, object]] = {}
        probability_columns = [column for column in teacher_frame.columns if column.startswith("proba_")]
        for split_name, split_frame in teacher_frame.groupby("split", dropna=False):
            anchor_map = {}
            for row in split_frame.to_dict(orient="records"):
                entry = {
                    "prediction": row["y_pred"],
                    "y_true": row["y_true"],
                }
                if probability_columns:
                    entry["probabilities"] = np.asarray([row[column] for column in probability_columns], dtype=np.float32)
                anchor_map[str(row["anchor_id"])] = entry
            split_maps[str(split_name)] = anchor_map
        cache[task_index] = {
            "teacher_model_name": teacher_model_name,
            "cache_path": str(cache_path),
            "splits": split_maps,
        }
        summary_rows.append(
            {
                "task_index": task_index,
                "dataset_id": meta.dataset_id,
                "task_name": meta.task_name,
                "teacher_model_name": teacher_model_name,
                "cache_path": str(cache_path.relative_to(PROJECT_ROOT)),
                "n_train": len(split_maps.get("train", {})),
                "n_valid": len(split_maps.get("valid", {})),
                "n_test": len(split_maps.get("test", {})),
                "task_key": meta.task_key,
            }
        )
    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_csv(cache_dir / "teacher_summary.csv", index=False)
    return cache, summary_frame


def _ordinal_targets(target_index: torch.Tensor, num_classes: int) -> torch.Tensor:
    thresholds = torch.arange(num_classes - 1, device=target_index.device).unsqueeze(0)
    return (target_index.unsqueeze(1) > thresholds).float()


def _ordinal_label_distribution(
    target_index: torch.Tensor,
    num_classes: int,
    *,
    sigma: float,
    edge_sigma: float | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if dtype is None:
        dtype = torch.float32
    class_positions = torch.arange(num_classes, device=target_index.device, dtype=dtype).unsqueeze(0)
    target_positions = target_index.to(dtype=dtype).unsqueeze(1)
    sigma_tensor = torch.full(
        (target_index.size(0), 1),
        float(max(sigma, 1e-3)),
        device=target_index.device,
        dtype=dtype,
    )
    if edge_sigma is not None and num_classes > 1:
        edge_mask = (target_index == 0) | (target_index == num_classes - 1)
        edge_tensor = torch.full_like(sigma_tensor, float(max(edge_sigma, 1e-3)))
        sigma_tensor = torch.where(edge_mask.unsqueeze(1), edge_tensor, sigma_tensor)
    squared_distance = torch.square((class_positions - target_positions) / sigma_tensor.clamp_min(1e-3))
    distribution = torch.exp(-0.5 * squared_distance)
    return distribution / distribution.sum(dim=1, keepdim=True).clamp_min(1e-8)


def _ordinal_probabilities_torch(logits: torch.Tensor) -> torch.Tensor:
    cumulative = torch.sigmoid(logits)
    cumulative = torch.cummin(cumulative, dim=1).values
    num_classes = cumulative.size(1) + 1
    probabilities = logits.new_zeros((logits.size(0), num_classes))
    probabilities[:, 0] = 1.0 - cumulative[:, 0]
    for class_index in range(1, num_classes - 1):
        probabilities[:, class_index] = torch.clamp(cumulative[:, class_index - 1] - cumulative[:, class_index], min=0.0)
    probabilities[:, -1] = torch.clamp(cumulative[:, -1], min=0.0)
    row_sums = probabilities.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return probabilities / row_sums


def _student_probabilities(meta: TaskMetadata, logits: torch.Tensor) -> torch.Tensor:
    if meta.label_type == "binary":
        positive = torch.sigmoid(logits.squeeze(-1))
        return torch.stack([1.0 - positive, positive], dim=-1)
    if meta.label_type == "ordinal":
        return _ordinal_probabilities_torch(logits)
    return torch.softmax(logits, dim=-1)


def _regression_training_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    training_info: dict[str, object],
) -> torch.Tensor:
    mode = str(training_info.get("regression_loss", "huber") or "huber").lower()
    if mode == "mse":
        return F.mse_loss(prediction, target)
    if mode == "mae":
        return F.l1_loss(prediction, target)
    if mode == "mae_huber":
        return 0.5 * F.huber_loss(prediction, target) + 0.5 * F.l1_loss(prediction, target)
    return F.huber_loss(prediction, target)


def _teacher_distillation_loss(
    meta: TaskMetadata,
    logits: torch.Tensor,
    batch_anchor_ids: list[str],
    teacher_entries: dict[str, object],
    training_info: dict[str, object],
    distillation_config: dict[str, float],
    task_key: str,
) -> torch.Tensor | None:
    available = [teacher_entries.get(str(anchor_id)) for anchor_id in batch_anchor_ids]
    if not available or any(entry is None for entry in available):
        return None
    if meta.label_type == "continuous":
        teacher_values = torch.as_tensor(
            [float(entry["prediction"]) for entry in available],
            dtype=logits.dtype,
            device=logits.device,
        )
        target_mean = float(training_info["target_mean"])
        target_std = float(training_info["target_std"])
        student_values = logits.squeeze(-1) * target_std + target_mean
        task_multiplier = float(distillation_config.get("task_weights", {}).get(task_key, 1.0))
        return float(distillation_config["regression_weight"]) * task_multiplier * F.huber_loss(student_values, teacher_values)

    teacher_probabilities = torch.as_tensor(
        np.stack([entry["probabilities"] for entry in available], axis=0),
        dtype=logits.dtype,
        device=logits.device,
    )
    student_probabilities = _student_probabilities(meta, logits).clamp_min(1e-8)
    task_multiplier = float(distillation_config.get("task_weights", {}).get(task_key, 1.0))
    return float(distillation_config["classification_weight"]) * F.kl_div(
        student_probabilities.log(),
        teacher_probabilities,
        reduction="batchmean",
    ) * task_multiplier


def _task_loss(
    logits: torch.Tensor,
    batch: dict[str, object],
    mask: torch.Tensor,
    meta: TaskMetadata,
    training_info: dict[str, object],
) -> torch.Tensor:
    if meta.label_type == "continuous":
        target_mean = float(training_info["target_mean"])
        target_std = float(training_info["target_std"])
        standardized_target = (batch["target_float"][mask] - target_mean) / target_std
        return _regression_training_loss(logits.squeeze(-1), standardized_target, training_info)
    if meta.label_type == "binary":
        pos_weight = torch.tensor(
            float(training_info["pos_weight"]),
            dtype=logits.dtype,
            device=logits.device,
        )
        focal_gamma = float(training_info.get("binary_focal_gamma", 0.0) or 0.0)
        target = batch["target_index"][mask].float()
        binary_logits = logits.squeeze(-1)
        if focal_gamma > 0.0:
            per_entry_loss = F.binary_cross_entropy_with_logits(
                binary_logits,
                target,
                pos_weight=pos_weight,
                reduction="none",
            )
            probabilities = torch.sigmoid(binary_logits)
            p_t = target * probabilities + (1.0 - target) * (1.0 - probabilities)
            focal_weight = torch.pow((1.0 - p_t).clamp_min(1e-6), focal_gamma)
            return (focal_weight * per_entry_loss).mean()
        return F.binary_cross_entropy_with_logits(
            binary_logits,
            target,
            pos_weight=pos_weight,
        )
    if meta.label_type == "ordinal":
        ordinal_target = _ordinal_targets(batch["target_index"][mask], len(meta.class_space or []))
        pos_weights = torch.as_tensor(
            training_info["pos_weights"],
            dtype=logits.dtype,
            device=logits.device,
        )
        focal_gamma = float(training_info.get("ordinal_focal_gamma", 0.0) or 0.0)
        if focal_gamma > 0.0:
            per_entry_loss = F.binary_cross_entropy_with_logits(
                logits,
                ordinal_target,
                pos_weight=pos_weights,
                reduction="none",
            )
            probabilities = torch.sigmoid(logits)
            p_t = ordinal_target * probabilities + (1.0 - ordinal_target) * (1.0 - probabilities)
            focal_weight = torch.pow((1.0 - p_t).clamp_min(1e-6), focal_gamma)
            ordinal_loss = (focal_weight * per_entry_loss).mean()
        else:
            ordinal_loss = F.binary_cross_entropy_with_logits(logits, ordinal_target, pos_weight=pos_weights)
        total_loss = ordinal_loss
        class_probabilities: torch.Tensor | None = None
        hybrid_weight = float(training_info.get("ordinal_hybrid_weight", 0.0) or 0.0)
        if hybrid_weight > 0.0 and "ordinal_class_weights" in training_info:
            class_probabilities = _ordinal_probabilities_torch(logits).clamp_min(1e-8)
            class_weights = torch.as_tensor(
                training_info["ordinal_class_weights"],
                dtype=logits.dtype,
                device=logits.device,
            )
            class_loss = F.nll_loss(
                torch.log(class_probabilities),
                batch["target_index"][mask],
                weight=class_weights,
            )
            total_loss = total_loss + hybrid_weight * class_loss
        labeldist_weight = float(training_info.get("ordinal_labeldist_weight", 0.0) or 0.0)
        if labeldist_weight > 0.0:
            if class_probabilities is None:
                class_probabilities = _ordinal_probabilities_torch(logits).clamp_min(1e-8)
            sigma = float(training_info.get("ordinal_labeldist_sigma", 0.9) or 0.9)
            edge_sigma_value = training_info.get("ordinal_labeldist_edge_sigma")
            edge_sigma = float(edge_sigma_value) if edge_sigma_value is not None else None
            target_distribution = _ordinal_label_distribution(
                batch["target_index"][mask],
                len(meta.class_space or []),
                sigma=sigma,
                edge_sigma=edge_sigma,
                dtype=logits.dtype,
            )
            per_entry_distribution_loss = -(target_distribution * torch.log(class_probabilities)).sum(dim=1)
            if "ordinal_class_weights" in training_info:
                class_weights = torch.as_tensor(
                    training_info["ordinal_class_weights"],
                    dtype=logits.dtype,
                    device=logits.device,
                )
                sample_weights = class_weights[batch["target_index"][mask]]
                distribution_loss = (per_entry_distribution_loss * sample_weights).mean()
            else:
                distribution_loss = per_entry_distribution_loss.mean()
            total_loss = total_loss + labeldist_weight * distribution_loss
        return total_loss
    class_weights = torch.as_tensor(
        training_info["class_weights"],
        dtype=logits.dtype,
        device=logits.device,
    )
    target_index = batch["target_index"][mask]
    focal_gamma = float(training_info.get("multiclass_focal_gamma", 0.0) or 0.0)
    label_smoothing = float(training_info.get("multiclass_label_smoothing", 0.0) or 0.0)
    if focal_gamma > 0.0:
        per_entry_loss = F.cross_entropy(
            logits,
            target_index,
            weight=class_weights,
            reduction="none",
            label_smoothing=label_smoothing,
        )
        probabilities = torch.softmax(logits, dim=-1)
        p_t = probabilities.gather(1, target_index.unsqueeze(1)).squeeze(1)
        focal_weight = torch.pow((1.0 - p_t).clamp_min(1e-6), focal_gamma)
        return (focal_weight * per_entry_loss).mean()
    return F.cross_entropy(
        logits,
        target_index,
        weight=class_weights,
        label_smoothing=label_smoothing,
    )


def _paired_task_loss(
    logits: torch.Tensor,
    target_float: torch.Tensor,
    target_index: torch.Tensor,
    meta: TaskMetadata,
    training_info: dict[str, object],
) -> torch.Tensor:
    if meta.label_type == "continuous":
        target_mean = float(training_info["target_mean"])
        target_std = float(training_info["target_std"])
        standardized_target = (target_float - target_mean) / target_std
        return _regression_training_loss(logits.squeeze(-1), standardized_target, training_info)
    if meta.label_type == "binary":
        pos_weight = torch.tensor(
            float(training_info["pos_weight"]),
            dtype=logits.dtype,
            device=logits.device,
        )
        return F.binary_cross_entropy_with_logits(
            logits.squeeze(-1),
            target_index.float(),
            pos_weight=pos_weight,
        )
    if meta.label_type == "ordinal":
        ordinal_target = _ordinal_targets(target_index, len(meta.class_space or []))
        pos_weights = torch.as_tensor(
            training_info["pos_weights"],
            dtype=logits.dtype,
            device=logits.device,
        )
        return F.binary_cross_entropy_with_logits(logits, ordinal_target, pos_weight=pos_weights)
    class_weights = torch.as_tensor(
        training_info["class_weights"],
        dtype=logits.dtype,
        device=logits.device,
    )
    return F.cross_entropy(logits, target_index, weight=class_weights)


def _step_weights(num_steps: int) -> list[float]:
    weights = np.linspace(1.0, float(num_steps), num_steps, dtype=np.float64)
    weights = weights / weights.sum()
    return weights.tolist()


def _grad_dot(first: list[torch.Tensor | None], second: list[torch.Tensor | None]) -> torch.Tensor:
    dot = None
    for left, right in zip(first, second):
        if left is None or right is None:
            continue
        value = torch.sum(left * right)
        dot = value if dot is None else dot + value
    if dot is None:
        dot = torch.zeros((), device=first[0].device if first and first[0] is not None else "cpu")
    return dot


def _grad_norm_sq(gradient_set: list[torch.Tensor | None]) -> torch.Tensor:
    total = None
    for gradient in gradient_set:
        if gradient is None:
            continue
        value = torch.sum(gradient * gradient)
        total = value if total is None else total + value
    if total is None:
        total = torch.zeros((), device=gradient_set[0].device if gradient_set and gradient_set[0] is not None else "cpu")
    return total


def _clone_gradients(gradient_set: list[torch.Tensor | None]) -> list[torch.Tensor | None]:
    return [gradient.clone() if gradient is not None else None for gradient in gradient_set]


def _offload_gradients_to_cpu(gradient_set: tuple[torch.Tensor | None, ...]) -> list[torch.Tensor | None]:
    return [
        gradient.detach().to(device="cpu", copy=True) if gradient is not None else None
        for gradient in gradient_set
    ]


def _pcgrad_combine(gradient_sets: list[list[torch.Tensor | None]]) -> list[torch.Tensor | None]:
    if not gradient_sets:
        return []
    projected_sets: list[list[torch.Tensor | None]] = []
    for gradient_set in gradient_sets:
        projected = _clone_gradients(gradient_set)
        shuffled = list(range(len(gradient_sets)))
        random.shuffle(shuffled)
        for other_index in shuffled:
            other = gradient_sets[other_index]
            dot = _grad_dot(projected, other)
            if float(dot.detach().cpu()) >= 0.0:
                continue
            denom = _grad_norm_sq(other)
            denom_value = float(denom.detach().cpu())
            if denom_value <= 0.0:
                continue
            coefficient = dot / denom
            updated: list[torch.Tensor | None] = []
            for current, other_gradient in zip(projected, other):
                if current is None or other_gradient is None:
                    updated.append(current)
                else:
                    updated.append(current - coefficient * other_gradient)
            projected = updated
        projected_sets.append(projected)

    combined: list[torch.Tensor | None] = []
    for parameter_position in range(len(projected_sets[0])):
        gradients = [gradient_set[parameter_position] for gradient_set in projected_sets if gradient_set[parameter_position] is not None]
        if not gradients:
            combined.append(None)
        else:
            combined.append(torch.stack(gradients, dim=0).mean(dim=0))
    return combined


def _apply_pcgrad(
    model: MCTRCMV2,
    total_loss: torch.Tensor,
    weighted_task_losses: list[torch.Tensor],
    regularization_loss: torch.Tensor,
) -> None:
    shared_parameters = [parameter for parameter in model.shared_parameters() if parameter.requires_grad and parameter.numel() > 0]
    if len(shared_parameters) == 0 or len(weighted_task_losses) <= 1:
        total_loss.backward()
        return

    gradient_sets = []
    for loss in weighted_task_losses:
        task_gradients = torch.autograd.grad(loss, shared_parameters, retain_graph=True, allow_unused=True)
        gradient_sets.append(_offload_gradients_to_cpu(task_gradients))
        del task_gradients
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    projected_gradients = _pcgrad_combine(gradient_sets)

    regularization_gradients = None
    if regularization_loss.requires_grad and float(regularization_loss.detach().cpu()) != 0.0:
        raw_regularization_gradients = torch.autograd.grad(
            regularization_loss,
            shared_parameters,
            retain_graph=True,
            allow_unused=True,
        )
        regularization_gradients = _offload_gradients_to_cpu(raw_regularization_gradients)
        del raw_regularization_gradients
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_loss.backward()
    for parameter, projected, reg_gradient in zip(
        shared_parameters,
        projected_gradients,
        regularization_gradients if regularization_gradients is not None else [None] * len(shared_parameters),
    ):
        combined = None
        if projected is not None and reg_gradient is not None:
            combined = projected + reg_gradient
        elif projected is not None:
            combined = projected
        elif reg_gradient is not None:
            combined = reg_gradient
        if combined is not None:
            parameter.grad = combined.to(device=parameter.device, dtype=parameter.dtype)


def _apply_gradnorm_balancing(
    model: MCTRCMV2,
    total_loss: torch.Tensor,
    weighted_task_losses: list[torch.Tensor],
    regularization_loss: torch.Tensor,
) -> None:
    shared_parameters = [parameter for parameter in model.shared_parameters() if parameter.requires_grad and parameter.numel() > 0]
    if len(shared_parameters) == 0 or len(weighted_task_losses) <= 1:
        total_loss.backward()
        return

    grad_norms = []
    for loss in weighted_task_losses:
        gradients = torch.autograd.grad(loss, shared_parameters, retain_graph=True, allow_unused=True)
        norm_sq = None
        for gradient in gradients:
            if gradient is None:
                continue
            value = torch.sum(gradient.detach() * gradient.detach())
            norm_sq = value if norm_sq is None else norm_sq + value
        if norm_sq is None:
            norm_sq = torch.zeros((), device=total_loss.device, dtype=total_loss.dtype)
        grad_norms.append(torch.sqrt(norm_sq + 1e-8))
    norm_tensor = torch.stack(grad_norms).to(device=total_loss.device, dtype=total_loss.dtype)
    target_norm = norm_tensor.mean().detach()
    balancing_weights = target_norm / norm_tensor.clamp_min(1e-6)
    balancing_weights = balancing_weights / balancing_weights.mean().clamp_min(1e-6)

    task_term = torch.stack(weighted_task_losses).mean()
    auxiliary_term = total_loss - task_term - regularization_loss
    balanced_task_term = torch.stack(
        [
            loss * balancing_weights[index].detach()
            for index, loss in enumerate(weighted_task_losses)
        ]
    ).mean()
    (balanced_task_term + regularization_loss + auxiliary_term).backward()


def _compute_train_losses(
    model: MCTRCMV2,
    batch: dict[str, object],
    task_metadata: dict[int, TaskMetadata],
    task_training_info: dict[int, dict[str, object]],
    task_log_vars: nn.Parameter,
    config: dict[str, object],
    epoch_index: int,
    teacher_cache: dict[int, dict[str, object]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], dict[str, float], dict[str, object]]:
    outputs = model(batch)
    step_weights = _step_weights(len(outputs["steps"]))
    raw_task_losses: dict[int, torch.Tensor] = {}
    distill_loss_map: dict[int, torch.Tensor] = {}
    pair_loss_map: dict[int, torch.Tensor] = {}
    edge_ovr_loss_map: dict[int, torch.Tensor] = {}
    construct_loss_map: dict[int, torch.Tensor] = {}
    auxiliary_losses: list[torch.Tensor] = []
    psyche_end_aux_total = 0.0
    psyche_delta_aux_total = 0.0
    final_step = outputs["steps"][-1]
    for step_weight, step_output in zip(step_weights, outputs["steps"]):
        for task_index, meta in task_metadata.items():
            mask = batch["task_index"] == task_index
            if not torch.any(mask):
                continue
            logits = step_output["logits_by_task"][task_index]
            task_loss = _task_loss(logits, batch, mask, meta, task_training_info[task_index])
            ordinal_aux_logits = step_output.get("ordinal_aux_logits_by_task", {}).get(task_index)
            ordinal_aux_weight = float(config.get("ordinal_aux_class_weight", 0.0) or 0.0)
            if ordinal_aux_logits is not None and meta.label_type == "ordinal" and ordinal_aux_weight > 0.0:
                class_weights = torch.as_tensor(
                    task_training_info[task_index]["ordinal_class_weights"],
                    dtype=ordinal_aux_logits.dtype,
                    device=ordinal_aux_logits.device,
                )
                aux_class_loss = F.cross_entropy(
                    ordinal_aux_logits,
                    batch["target_index"][mask],
                    weight=class_weights,
                )
                task_loss = task_loss + ordinal_aux_weight * aux_class_loss
            if task_index not in raw_task_losses:
                raw_task_losses[task_index] = task_loss * step_weight
            else:
                raw_task_losses[task_index] = raw_task_losses[task_index] + task_loss * step_weight
            paired_logits = step_output.get("paired_logits_by_task", {}).get(task_index)
            if paired_logits is not None:
                local_pair_mask = batch["paired_task_mask"][mask] > 0.5
                if torch.any(local_pair_mask):
                    paired_target_task_indices = batch["paired_task_index"][mask][local_pair_mask]
                    unique_target_task_indices = torch.unique(paired_target_task_indices)
                    if len(unique_target_task_indices) == 1:
                        paired_target_task_index = int(unique_target_task_indices.item())
                        paired_target_meta = task_metadata[paired_target_task_index]
                        auxiliary_weight = float(config.get("deprest_gad7_pair_weight", 0.0) or 0.0)
                        if auxiliary_weight > 0.0:
                            paired_logits_local = paired_logits[local_pair_mask]
                            paired_target_float = batch["paired_target_float"][mask][local_pair_mask]
                            paired_target_index = batch["paired_target_index"][mask][local_pair_mask]
                            if paired_target_meta.label_type == "continuous":
                                valid_mask = torch.isfinite(paired_target_float)
                            else:
                                valid_mask = paired_target_index >= 0
                            if torch.any(valid_mask):
                                pair_loss = _paired_task_loss(
                                    paired_logits_local[valid_mask],
                                    paired_target_float[valid_mask],
                                    paired_target_index[valid_mask],
                                    paired_target_meta,
                                    task_training_info[paired_target_task_index],
                                )
                                raw_task_losses[task_index] = raw_task_losses[task_index] + auxiliary_weight * pair_loss * step_weight
                                weighted_pair_loss = auxiliary_weight * pair_loss
                                if task_index not in pair_loss_map:
                                    pair_loss_map[task_index] = weighted_pair_loss.detach()
                                else:
                                    pair_loss_map[task_index] = pair_loss_map[task_index] + weighted_pair_loss.detach()
            edge_ovr_logits = step_output.get("edge_ovr_logits_by_task", {}).get(task_index)
            edge_ovr_probabilities = step_output.get("edge_ovr_probs_by_task", {}).get(task_index)
            edge_ovr_weight = float(config.get("deprest_edge_ovr_weight", 0.0) or 0.0)
            if (
                edge_ovr_logits is not None
                and edge_ovr_probabilities is not None
                and meta.label_type == "ordinal"
                and edge_ovr_weight > 0.0
            ):
                low_pos_weight = torch.tensor(
                    float(task_training_info[task_index].get("edge_low_pos_weight", 1.0)),
                    dtype=edge_ovr_logits.dtype,
                    device=edge_ovr_logits.device,
                )
                high_pos_weight = torch.tensor(
                    float(task_training_info[task_index].get("edge_high_pos_weight", 1.0)),
                    dtype=edge_ovr_logits.dtype,
                    device=edge_ovr_logits.device,
                )
                target_index = batch["target_index"][mask]
                low_target = (target_index == 0).float()
                high_target = (target_index == max(len(meta.class_space or []) - 1, 0)).float()
                low_loss = F.binary_cross_entropy_with_logits(
                    edge_ovr_logits[:, 0],
                    low_target,
                    pos_weight=low_pos_weight,
                )
                high_loss = F.binary_cross_entropy_with_logits(
                    edge_ovr_logits[:, 1],
                    high_target,
                    pos_weight=high_pos_weight,
                )
                class_weights = torch.as_tensor(
                    task_training_info[task_index]["ordinal_class_weights"],
                    dtype=edge_ovr_probabilities.dtype,
                    device=edge_ovr_probabilities.device,
                )
                decision_loss = F.nll_loss(
                    torch.log(edge_ovr_probabilities.clamp_min(1e-8)),
                    target_index,
                    weight=class_weights,
                )
                edge_loss = 0.5 * (low_loss + high_loss) + 0.5 * decision_loss
                raw_task_losses[task_index] = raw_task_losses[task_index] + edge_ovr_weight * edge_loss * step_weight
                weighted_edge_loss = edge_ovr_weight * edge_loss
                if task_index not in edge_ovr_loss_map:
                    edge_ovr_loss_map[task_index] = weighted_edge_loss.detach()
                else:
                    edge_ovr_loss_map[task_index] = edge_ovr_loss_map[task_index] + weighted_edge_loss.detach()
            construct_score = step_output.get("deprest_construct_scores_by_task", {}).get(task_index)
            construct_weight = float(config.get("deprest_construct_weight", 0.0) or 0.0)
            if construct_score is not None and construct_weight > 0.0 and str(meta.dataset_id) == "deprest_cat":
                if meta.label_type == "continuous":
                    target_score = batch["target_float"][mask]
                    valid_mask = torch.isfinite(target_score)
                else:
                    target_score = batch["paired_target_float"][mask]
                    valid_mask = (batch["paired_task_mask"][mask] > 0.5) & torch.isfinite(target_score)
                if torch.any(valid_mask):
                    construct_loss = F.huber_loss(
                        construct_score.squeeze(-1)[valid_mask],
                        target_score[valid_mask],
                    )
                    raw_task_losses[task_index] = raw_task_losses[task_index] + construct_weight * construct_loss * step_weight
                    weighted_construct_loss = construct_weight * construct_loss
                    if task_index not in construct_loss_map:
                        construct_loss_map[task_index] = weighted_construct_loss.detach()
                    else:
                        construct_loss_map[task_index] = construct_loss_map[task_index] + weighted_construct_loss.detach()
        if step_output.get("psyche_end_logits") is not None:
            psyche_mask = batch["psyche_end_mask"] > 0.5
            if torch.any(psyche_mask):
                aux_loss = F.cross_entropy(
                    step_output["psyche_end_logits"][psyche_mask],
                    batch["psyche_end_target"][psyche_mask],
                )
                weighted_aux_loss = aux_loss * step_weight * float(config.get("psyche_hierarchical_weight", 0.2))
                auxiliary_losses.append(weighted_aux_loss)
                psyche_end_aux_total += float(weighted_aux_loss.detach().cpu())
        psyche_delta_state = step_output.get("psyche_delta_bridge_state")
        psyche_delta_weight = float(config.get("psyche_delta_weight", 0.0) or 0.0)
        if psyche_delta_state is not None and psyche_delta_weight > 0.0:
            psyche_delta_mask = (
                psyche_delta_state["mask"]
                & (batch["psyche_start_score_mask"] > 0.5)
                & (batch["psyche_end_score_mask"] > 0.5)
            )
            if torch.any(psyche_delta_mask):
                target_delta = batch["psyche_end_score"][psyche_delta_mask] - batch["psyche_start_score"][psyche_delta_mask]
                predicted_delta = psyche_delta_state["predicted_delta"].squeeze(-1)[psyche_delta_mask]
                predicted_end_score = psyche_delta_state["predicted_end_score"].squeeze(-1)[psyche_delta_mask]
                delta_loss = F.huber_loss(predicted_delta, target_delta)
                end_score_loss = F.huber_loss(predicted_end_score, batch["psyche_end_score"][psyche_delta_mask])
                weighted_aux_loss = (delta_loss + 0.25 * end_score_loss) * step_weight * psyche_delta_weight
                auxiliary_losses.append(weighted_aux_loss)
                psyche_delta_aux_total += float(weighted_aux_loss.detach().cpu())

    if teacher_cache:
        for task_index, meta in task_metadata.items():
            teacher_info = teacher_cache.get(task_index)
            if teacher_info is None:
                continue
            mask = batch["task_index"] == task_index
            if not torch.any(mask):
                continue
            logits = final_step["logits_by_task"][task_index]
            sample_indices = mask.nonzero(as_tuple=False).squeeze(-1).detach().cpu().tolist()
            anchor_ids = [batch["anchor_id"][sample_index] for sample_index in sample_indices]
            teacher_entries = teacher_info["splits"].get("train", {})
            distill_loss = _teacher_distillation_loss(
                meta,
                logits,
                anchor_ids,
                teacher_entries,
                task_training_info[task_index],
                config["distillation"],
                meta.task_key,
            )
            if distill_loss is not None:
                if task_index in raw_task_losses:
                    raw_task_losses[task_index] = raw_task_losses[task_index] + distill_loss
                else:
                    raw_task_losses[task_index] = distill_loss
                distill_loss_map[task_index] = distill_loss

    weighted_task_losses: list[torch.Tensor] = []
    detached_loss_map: dict[str, float] = {}
    for task_index, task_loss in raw_task_losses.items():
        task_loss_weight = float(task_training_info[task_index].get("task_loss_weight", 1.0) or 1.0)
        if task_loss_weight != 1.0:
            task_loss = task_loss * task_loss_weight
        if bool(config["use_uncertainty_weighting"]):
            log_var = task_log_vars[task_index]
            weighted = torch.exp(-log_var) * task_loss + log_var
        else:
            weighted = task_loss
        weighted_task_losses.append(weighted)
        task_key = task_metadata[task_index].task_key
        detached_loss_map[task_key] = float(task_loss.detach().cpu())
        if task_index in distill_loss_map:
            detached_loss_map[f"{task_key}::distill"] = float(distill_loss_map[task_index].detach().cpu())
        if task_index in pair_loss_map:
            detached_loss_map[f"{task_key}::pair_aux"] = float(pair_loss_map[task_index].detach().cpu())
        if task_index in edge_ovr_loss_map:
            detached_loss_map[f"{task_key}::edge_ovr_aux"] = float(edge_ovr_loss_map[task_index].detach().cpu())
        if task_index in construct_loss_map:
            detached_loss_map[f"{task_key}::construct_aux"] = float(construct_loss_map[task_index].detach().cpu())

    sparse_warmup_epochs = int(config.get("sparse_warmup_epochs", 0) or 0)
    sparse_scale = 0.0 if epoch_index <= sparse_warmup_epochs else 1.0
    if final_step["concepts"].numel() > 0:
        concept_l1 = final_step["concepts"].abs().mean() * float(config["sparse_penalty"]["l1"]) * sparse_scale
    else:
        concept_l1 = torch.zeros((), device=final_step["task_features"].device)
    group_penalty = model.sparse_penalty() * float(config["sparse_penalty"]["group"]) * sparse_scale
    regularization_loss = concept_l1 + group_penalty
    auxiliary_loss = (
        torch.stack(auxiliary_losses).sum()
        if auxiliary_losses
        else torch.zeros((), device=final_step["concepts"].device)
    )
    task_term = (
        torch.stack(weighted_task_losses).mean()
        if weighted_task_losses
        else torch.zeros((), device=final_step["concepts"].device)
    )
    total_loss = task_term + regularization_loss + auxiliary_loss
    aux = {
        "outputs": outputs,
        "raw_task_losses": detached_loss_map,
        "concept_l1": float(concept_l1.detach().cpu()),
        "group_penalty": float(group_penalty.detach().cpu()),
        "psyche_auxiliary_loss": float(auxiliary_loss.detach().cpu()),
    }
    if psyche_end_aux_total > 0.0:
        detached_loss_map["psyche_d::end_category_aux"] = psyche_end_aux_total
    if psyche_delta_aux_total > 0.0:
        detached_loss_map["psyche_d::delta_bridge_aux"] = psyche_delta_aux_total
    return total_loss, regularization_loss, weighted_task_losses, detached_loss_map, aux


def _build_split_records(
    model: MCTRCMV2,
    loader: DataLoader,
    task_metadata: dict[int, TaskMetadata],
    device: torch.device,
    split_name: str,
) -> dict[int, dict[str, object]]:
    records: dict[int, dict[str, object]] = {}
    model.eval()
    batch_count = len(loader)
    with torch.no_grad():
        for batch_index, raw_batch in enumerate(loader, start=1):
            if batch_count <= 4 or batch_index == 1 or batch_index == batch_count or batch_index % 50 == 0:
                _progress(f"{split_name}: fetched batch {batch_index}/{batch_count}")
            batch = _move_batch(raw_batch, device)
            if batch_count <= 4 or batch_index == 1 or batch_index == batch_count or batch_index % 50 == 0:
                _progress(f"{split_name}: moved batch {batch_index}/{batch_count} to {device.type}")
            outputs = model(batch)
            if batch_count <= 4 or batch_index == 1 or batch_index == batch_count or batch_index % 50 == 0:
                _progress(f"{split_name}: model forward finished for batch {batch_index}/{batch_count}")
            final_step = outputs["steps"][-1]
            for task_index, meta in task_metadata.items():
                mask = batch["task_index"] == task_index
                if not torch.any(mask):
                    continue
                indices = mask.nonzero(as_tuple=False).squeeze(-1).detach().cpu().tolist()
                holder = records.setdefault(
                    task_index,
                    {
                        "split": split_name,
                        "dataset_id": meta.dataset_id,
                        "task_name": meta.task_name,
                        "task_key": meta.task_key,
                        "label_type": meta.label_type,
                        "subject_id": [],
                        "anchor_id": [],
                        "y_true_raw": [],
                        "y_true_index": [],
                        "target_float": [],
                        "logits": [],
                        "ordinal_aux_logits": [],
                        "concepts": [],
                        "edge_ovr_logits": [],
                        "edge_ovr_mix_scale": [],
                    },
                )
                logits = final_step["logits_by_task"][task_index].detach().cpu().numpy()
                concepts = final_step["concepts"][mask].detach().cpu().numpy()
                holder["logits"].append(logits)
                holder["concepts"].append(concepts)
                ordinal_aux_logits = final_step.get("ordinal_aux_logits_by_task", {}).get(task_index)
                if ordinal_aux_logits is not None:
                    holder["ordinal_aux_logits"].append(ordinal_aux_logits.detach().cpu().numpy())
                edge_ovr_logits = final_step.get("edge_ovr_logits_by_task", {}).get(task_index)
                edge_ovr_mix_scale = final_step.get("edge_ovr_mix_scale_by_task", {}).get(task_index)
                if edge_ovr_logits is not None and edge_ovr_mix_scale is not None:
                    holder["edge_ovr_logits"].append(edge_ovr_logits.detach().cpu().numpy())
                    holder["edge_ovr_mix_scale"].append(edge_ovr_mix_scale.detach().cpu().numpy())

                if meta.label_type == "continuous":
                    target_values = batch["target_float"][mask].detach().cpu().numpy()
                    holder["target_float"].extend(target_values.tolist())
                    holder["y_true_raw"].extend(target_values.tolist())
                else:
                    target_index = batch["target_index"][mask].detach().cpu().numpy()
                    holder["y_true_index"].extend(target_index.tolist())
                    class_space = np.asarray(meta.class_space, dtype=object)
                    holder["y_true_raw"].extend(class_space[target_index].tolist())

                for local_index, global_index in enumerate(indices):
                    holder["subject_id"].append(raw_batch["subject_id"][global_index])
                    holder["anchor_id"].append(raw_batch["anchor_id"][global_index])
            if batch_count <= 4 or batch_index == 1 or batch_index == batch_count or batch_index % 50 == 0:
                _progress(f"{split_name}: collated batch {batch_index}/{batch_count}")
    for holder in records.values():
        holder["logits"] = np.concatenate(holder["logits"], axis=0)
        holder["ordinal_aux_logits"] = (
            np.concatenate(holder["ordinal_aux_logits"], axis=0)
            if holder["ordinal_aux_logits"]
            else None
        )
        holder["concepts"] = np.concatenate(holder["concepts"], axis=0)
        holder["edge_ovr_logits"] = (
            np.concatenate(holder["edge_ovr_logits"], axis=0)
            if holder["edge_ovr_logits"]
            else None
        )
        holder["edge_ovr_mix_scale"] = (
            np.concatenate(holder["edge_ovr_mix_scale"], axis=0)
            if holder["edge_ovr_mix_scale"]
            else None
        )
    return records


def _sigmoid_binary_probabilities(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = logits.reshape(-1) / max(temperature, 1e-3)
    positive = 1.0 / (1.0 + np.exp(-scaled))
    return np.column_stack([1.0 - positive, positive])


def _ordinal_probabilities(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = logits / max(temperature, 1e-3)
    cumulative = 1.0 / (1.0 + np.exp(-scaled))
    cumulative = np.minimum.accumulate(cumulative, axis=1)
    num_classes = cumulative.shape[1] + 1
    probabilities = np.zeros((logits.shape[0], num_classes), dtype=np.float64)
    probabilities[:, 0] = 1.0 - cumulative[:, 0]
    for class_index in range(1, num_classes - 1):
        probabilities[:, class_index] = np.clip(cumulative[:, class_index - 1] - cumulative[:, class_index], 0.0, 1.0)
    probabilities[:, -1] = np.clip(cumulative[:, -1], 0.0, 1.0)
    row_sums = probabilities.sum(axis=1, keepdims=True)
    row_sums[row_sums <= 0.0] = 1.0
    return probabilities / row_sums


def _edge_ovr_probabilities(
    base_probabilities: np.ndarray,
    edge_logits: np.ndarray,
    mix_scale: np.ndarray,
) -> np.ndarray:
    edge_mass = (1.0 / (1.0 + np.exp(-edge_logits))) * mix_scale
    edge_total = edge_mass.sum(axis=1, keepdims=True)
    downscale = np.minimum(1.0, 0.95 / np.clip(edge_total, a_min=1e-6, a_max=None))
    edge_mass = edge_mass * downscale
    remaining = np.clip(1.0 - edge_mass.sum(axis=1, keepdims=True), a_min=1e-6, a_max=None)
    probabilities = base_probabilities * remaining
    probabilities[:, 0] = probabilities[:, 0] + edge_mass[:, 0]
    probabilities[:, -1] = probabilities[:, -1] + edge_mass[:, 1]
    row_sums = probabilities.sum(axis=1, keepdims=True)
    row_sums[row_sums <= 0.0] = 1.0
    return probabilities / row_sums


def _apply_multiclass_biases(probabilities: np.ndarray, class_biases: list[float] | np.ndarray) -> np.ndarray:
    bias_array = np.asarray(class_biases, dtype=np.float64).reshape(1, -1)
    biased = np.asarray(probabilities, dtype=np.float64) * bias_array
    row_sums = biased.sum(axis=1, keepdims=True)
    row_sums[row_sums <= 0.0] = 1.0
    return biased / row_sums


def _classification_objective(
    y_true_index: np.ndarray,
    y_pred_index: np.ndarray,
    probabilities: np.ndarray | None = None,
) -> tuple[float, float, float]:
    y_true_index = np.asarray(y_true_index, dtype=np.int64).reshape(-1)
    y_pred_index = np.asarray(y_pred_index, dtype=np.int64).reshape(-1)
    if y_true_index.size == 0 or y_pred_index.size == 0:
        return -1e9, -1e9, -1e9

    num_classes = int(max(np.max(y_true_index), np.max(y_pred_index)) + 1)
    encoded = y_true_index * num_classes + y_pred_index
    confusion = np.bincount(encoded, minlength=num_classes * num_classes).reshape(num_classes, num_classes).astype(np.float64)
    true_support = confusion.sum(axis=1)
    pred_support = confusion.sum(axis=0)
    true_positives = np.diag(confusion)

    balanced_accuracy_labels = true_support > 0.0
    recalls = np.divide(
        true_positives,
        true_support,
        out=np.zeros_like(true_positives),
        where=true_support > 0.0,
    )
    balanced_accuracy = (
        float(recalls[balanced_accuracy_labels].mean())
        if np.any(balanced_accuracy_labels)
        else -1e9
    )

    macro_f1_labels = (true_support + pred_support) > 0.0
    f1_denominator = (2.0 * true_positives) + (pred_support - true_positives) + (true_support - true_positives)
    per_class_f1 = np.divide(
        2.0 * true_positives,
        f1_denominator,
        out=np.zeros_like(true_positives),
        where=f1_denominator > 0.0,
    )
    macro_f1 = (
        float(per_class_f1[macro_f1_labels].mean())
        if np.any(macro_f1_labels)
        else -1e9
    )

    if probabilities is not None:
        normalized = np.asarray(probabilities, dtype=np.float64)
        if normalized.ndim == 1:
            normalized = np.column_stack([1.0 - normalized, normalized])
        if normalized.shape[1] < num_classes:
            padded = np.zeros((normalized.shape[0], num_classes), dtype=np.float64)
            padded[:, : normalized.shape[1]] = normalized
            normalized = padded
        normalized = np.clip(normalized, 1e-8, 1.0)
        normalized = normalized / np.clip(normalized.sum(axis=1, keepdims=True), a_min=1e-8, a_max=None)
        nll = -float(np.mean(np.log(normalized[np.arange(len(y_true_index)), y_true_index])))
        tertiary = -nll
    else:
        tertiary = -float(np.mean(np.abs(np.asarray(y_pred_index, dtype=np.int64) - np.asarray(y_true_index, dtype=np.int64))))
    return balanced_accuracy, macro_f1, tertiary


def _objective_improves(
    candidate: tuple[float, float, float],
    incumbent: tuple[float, float, float],
    *,
    primary_atol: float = 1e-12,
    secondary_atol: float = 1e-12,
    tertiary_atol: float = 1e-10,
) -> bool:
    for candidate_value, incumbent_value, atol in zip(
        candidate,
        incumbent,
        (primary_atol, secondary_atol, tertiary_atol),
    ):
        delta = float(candidate_value) - float(incumbent_value)
        if delta > atol:
            return True
        if delta < -atol:
            return False
    return False


def _fit_multiclass_biases(
    probabilities: np.ndarray,
    y_true_index: np.ndarray,
) -> list[float]:
    num_classes = int(probabilities.shape[1])
    if num_classes <= 1:
        return [1.0]

    objective_cache: dict[tuple[float, ...], tuple[float, float, float]] = {}

    def _objective(class_biases: list[float] | np.ndarray) -> tuple[float, float, float]:
        cache_key = tuple(round(float(value), 12) for value in class_biases)
        cached = objective_cache.get(cache_key)
        if cached is not None:
            return cached
        biased = _apply_multiclass_biases(probabilities, class_biases)
        y_pred = biased.argmax(axis=1)
        objective = _classification_objective(y_true_index, y_pred, biased)
        objective_cache[cache_key] = objective
        return objective

    class_counts = np.bincount(y_true_index, minlength=num_classes).astype(np.float64)
    target_priors = class_counts / max(class_counts.sum(), 1.0)
    predicted_priors = np.clip(probabilities.mean(axis=0), a_min=1e-8, a_max=None)
    prior_correction = np.clip(target_priors / predicted_priors, 1e-3, 1e3)
    starting_points = [
        [1.0] * num_classes,
        prior_correction.astype(np.float64).tolist(),
    ]
    # Exhaustively trying every pairwise decision boundary can dominate runtime
    # for multiclass calibration. A bounded quantile search keeps the same
    # validation-safe objective while avoiding minute-long calibration stalls.
    max_boundary_candidates = 96
    max_coordinate_rounds = 8
    epsilon = 1e-5
    best_biases = starting_points[0]
    best_objective = _objective(best_biases)
    last_heartbeat = time.time()
    for start_biases in starting_points:
        current_biases = [float(max(value, 1e-6)) for value in start_biases]
        current_objective = _objective(current_biases)
        improved = True
        coordinate_round = 0
        while improved and coordinate_round < max_coordinate_rounds:
            coordinate_round += 1
            improved = False
            for class_index in range(num_classes):
                if time.time() - last_heartbeat >= 30.0:
                    _progress(f"calibration: multiclass bias search heartbeat (class={class_index + 1}/{num_classes})")
                    last_heartbeat = time.time()
                candidates = {
                    float(current_biases[class_index]),
                    1.0,
                    float(prior_correction[class_index]),
                }
                boundaries: list[float] = []
                for row in probabilities:
                    current_probability = float(row[class_index])
                    if current_probability <= 0.0:
                        continue
                    for other_index, other_probability in enumerate(row):
                        if other_index == class_index or other_probability <= 0.0:
                            continue
                        boundary = float(current_biases[other_index] * other_probability / current_probability)
                        if boundary <= 0.0 or not math.isfinite(boundary):
                            continue
                        boundaries.extend([boundary * (1.0 - epsilon), boundary * (1.0 + epsilon)])
                if boundaries:
                    finite_boundaries = np.asarray(
                        [value for value in boundaries if value > 0.0 and math.isfinite(float(value))],
                        dtype=np.float64,
                    )
                    if finite_boundaries.size:
                        finite_boundaries = np.clip(finite_boundaries, 1e-6, 1e6)
                        if finite_boundaries.size > max_boundary_candidates:
                            quantiles = np.linspace(0.0, 1.0, max_boundary_candidates)
                            finite_boundaries = np.quantile(finite_boundaries, quantiles)
                        candidates.update(float(value) for value in finite_boundaries.tolist())
                local_best_value = current_biases[class_index]
                local_best_objective = current_objective
                for candidate in sorted(candidates):
                    if candidate <= 0.0 or not math.isfinite(candidate):
                        continue
                    candidate_biases = current_biases.copy()
                    candidate_biases[class_index] = float(candidate)
                    candidate_objective = _objective(candidate_biases)
                    if _objective_improves(candidate_objective, local_best_objective):
                        local_best_value = float(candidate)
                        local_best_objective = candidate_objective
                if _objective_improves(local_best_objective, current_objective):
                    current_biases[class_index] = local_best_value
                    current_objective = local_best_objective
                    improved = True
        if _objective_improves(current_objective, best_objective):
            best_biases = current_biases
            best_objective = current_objective
    return [float(value) for value in best_biases]


def _score_threshold_candidates(source_scores: np.ndarray, *, max_candidates: int = 31) -> list[float]:
    unique_scores = np.unique(np.round(np.asarray(source_scores, dtype=np.float64), decimals=12))
    if unique_scores.size == 0:
        return [0.0]
    candidates = unique_scores.tolist()
    if unique_scores.size >= 2:
        candidates.extend(((unique_scores[:-1] + unique_scores[1:]) / 2.0).tolist())
        candidates.extend([float(unique_scores[0] - 1e-6), float(unique_scores[-1] + 1e-6)])
    if len(candidates) > max_candidates:
        ranked = np.linspace(0, len(candidates) - 1, max_candidates).round().astype(int)
        candidates = [sorted(candidates)[int(index)] for index in ranked.tolist()]
    return sorted({float(candidate) for candidate in candidates if math.isfinite(float(candidate))})


def _score_to_label_indices(source_scores: np.ndarray, thresholds: list[float]) -> np.ndarray:
    source_scores = np.asarray(source_scores, dtype=np.float64).reshape(-1)
    predictions = np.zeros(source_scores.shape[0], dtype=np.int64)
    for threshold in thresholds:
        predictions = predictions + (source_scores >= float(threshold)).astype(np.int64)
    return predictions


def _bridge_match_key(subject_id: object, anchor_id: object) -> str:
    subject_text = str(subject_id)
    anchor_text = str(anchor_id)
    parts = anchor_text.split("__")
    if len(parts) >= 2:
        anchor_text = "__".join(parts[:2])
    return f"{subject_text}::{anchor_text}"


def _fit_score_bridge_thresholds(
    y_true_index: np.ndarray,
    source_scores: np.ndarray,
    num_classes: int,
) -> list[float]:
    num_thresholds = max(int(num_classes) - 1, 1)
    candidates = _score_threshold_candidates(source_scores)
    best_thresholds = [0.0] * num_thresholds
    best_objective = (-1e9, -1e9, -1e9)
    for threshold_tuple in combinations_with_replacement(candidates, num_thresholds):
        threshold_list = [float(value) for value in threshold_tuple]
        y_pred_index = _score_to_label_indices(source_scores, threshold_list)
        objective = _classification_objective(y_true_index, y_pred_index)
        if _objective_improves(objective, best_objective):
            best_thresholds = threshold_list
            best_objective = objective
    return best_thresholds


def _ordinal_expected_scores(logits: np.ndarray, temperature: float) -> tuple[np.ndarray, np.ndarray]:
    probabilities = _ordinal_probabilities(np.asarray(logits, dtype=np.float64), temperature)
    class_scores = np.arange(probabilities.shape[1], dtype=np.float64)
    expected_scores = probabilities @ class_scores
    return expected_scores, probabilities


def _classification_holder_objective(
    holder: dict[str, object],
    predictions: np.ndarray,
    probabilities: np.ndarray | None = None,
) -> tuple[float, float, float]:
    return _classification_objective(
        np.asarray(holder["y_true_index"], dtype=np.int64),
        np.asarray(predictions, dtype=np.int64),
        probabilities,
    )


def _teacher_probabilities_for_holder(
    holder: dict[str, object],
    teacher_entry: dict[str, object] | None,
) -> np.ndarray | None:
    if not teacher_entry:
        return None
    split_name = str(holder.get("split", ""))
    split_map = (teacher_entry.get("splits") or {}).get(split_name)
    if not split_map:
        return None
    probabilities: list[np.ndarray] = []
    for anchor_id in holder.get("anchor_id", []):
        entry = split_map.get(str(anchor_id))
        if entry is None or "probabilities" not in entry:
            return None
        probabilities.append(np.asarray(entry["probabilities"], dtype=np.float64))
    if not probabilities:
        return None
    matrix = np.stack(probabilities, axis=0)
    matrix = np.clip(matrix, 1e-8, 1.0)
    matrix = matrix / np.clip(matrix.sum(axis=1, keepdims=True), a_min=1e-8, a_max=None)
    return matrix


def _fit_probability_threshold(
    y_true_index: np.ndarray,
    positive_probabilities: np.ndarray,
    full_probabilities: np.ndarray,
) -> float:
    unique_probabilities = np.unique(np.round(np.asarray(positive_probabilities, dtype=np.float64), decimals=12))
    if unique_probabilities.size <= 1:
        return 0.5
    candidates = np.concatenate(
        [
            np.asarray([max(float(unique_probabilities[0]) - 1e-6, 0.0)]),
            unique_probabilities,
            (unique_probabilities[:-1] + unique_probabilities[1:]) / 2.0,
            np.asarray([min(float(unique_probabilities[-1]) + 1e-6, 1.0)]),
        ]
    )
    best_threshold = 0.5
    best_objective = (-1e9, -1e9, -1e9)
    for threshold in candidates:
        predictions = (positive_probabilities >= float(threshold)).astype(np.int64)
        objective = _classification_objective(y_true_index, predictions, full_probabilities)
        if _objective_improves(objective, best_objective):
            best_objective = objective
            best_threshold = float(threshold)
    return best_threshold


def _infer_auto_paired_regression_bridges(task_metadata: dict[int, TaskMetadata]) -> dict[int, int]:
    by_dataset_name = {
        (meta.dataset_id, meta.task_name): meta
        for meta in task_metadata.values()
    }
    bridges: dict[int, int] = {}
    for meta in task_metadata.values():
        if meta.label_type == "continuous":
            continue
        candidate_names: list[str] = []
        if meta.task_name.endswith("_cat"):
            candidate_names.append(meta.task_name[: -len("_cat")] + "_reg")
        if meta.task_name.endswith("_category"):
            candidate_names.append(meta.task_name[: -len("_category")] + "_reg")
        if meta.task_name.endswith("_ordinal"):
            candidate_names.append(meta.task_name[: -len("_ordinal")] + "_reg")
        for candidate_name in candidate_names:
            source_meta = by_dataset_name.get((meta.dataset_id, candidate_name))
            if source_meta is not None and source_meta.label_type == "continuous":
                bridges[int(meta.task_index)] = int(source_meta.task_index)
                break
    return bridges


def _fit_temperature(meta: TaskMetadata, holder: dict[str, object]) -> float:
    if meta.label_type == "continuous":
        return 1.0
    logits = np.asarray(holder["logits"], dtype=np.float64)
    y_index = np.asarray(holder["y_true_index"], dtype=np.int64)
    candidates = np.unique(np.concatenate([np.linspace(0.5, 3.0, 21), np.linspace(3.25, 8.0, 20)]))
    best_temperature = 1.0
    best_objective = math.inf
    for temperature in candidates:
        if meta.label_type == "binary":
            scaled = logits.reshape(-1) / float(temperature)
            probabilities = 1.0 / (1.0 + np.exp(-np.clip(scaled, -40.0, 40.0)))
            targets = y_index.astype(np.float64)
            brier = float(np.mean((probabilities - targets) ** 2))
            nll = F.binary_cross_entropy_with_logits(
                torch.as_tensor(scaled, dtype=torch.float32),
                torch.as_tensor(targets, dtype=torch.float32),
            ).item()
            objective = brier + 0.05 * float(nll)
        elif meta.label_type == "ordinal":
            scaled_logits = logits / float(temperature)
            probabilities = _ordinal_probabilities(scaled_logits, 1.0)
            targets_one_hot = np.eye(probabilities.shape[1], dtype=np.float64)[y_index]
            brier = float(np.mean(np.sum((probabilities - targets_one_hot) ** 2, axis=1)))
            targets = torch.as_tensor(_ordinal_targets(torch.as_tensor(y_index), len(meta.class_space or [])), dtype=torch.float32)
            nll = F.binary_cross_entropy_with_logits(
                torch.as_tensor(scaled_logits, dtype=torch.float32),
                targets,
            ).item()
            objective = brier + 0.05 * float(nll)
        else:
            scaled_logits = logits / float(temperature)
            scaled_logits = scaled_logits - scaled_logits.max(axis=1, keepdims=True)
            probabilities = np.exp(scaled_logits)
            probabilities = probabilities / np.clip(probabilities.sum(axis=1, keepdims=True), a_min=1e-8, a_max=None)
            targets_one_hot = np.eye(probabilities.shape[1], dtype=np.float64)[y_index]
            brier = float(np.mean(np.sum((probabilities - targets_one_hot) ** 2, axis=1)))
            nll = F.cross_entropy(
                torch.as_tensor(logits / float(temperature), dtype=torch.float32),
                torch.as_tensor(y_index, dtype=torch.long),
            ).item()
            objective = brier + 0.05 * float(nll)
        if objective < best_objective:
            best_objective = objective
            best_temperature = float(temperature)
    return best_temperature


def _fit_multiclass_temperature(logits: np.ndarray, y_index: np.ndarray) -> float:
    logits = np.asarray(logits, dtype=np.float64)
    y_index = np.asarray(y_index, dtype=np.int64)
    candidates = np.unique(np.concatenate([np.linspace(0.5, 3.0, 21), np.linspace(3.25, 8.0, 20)]))
    best_temperature = 1.0
    best_objective = math.inf
    for temperature in candidates:
        scaled_logits = logits / float(temperature)
        shifted = scaled_logits - scaled_logits.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities = probabilities / np.clip(probabilities.sum(axis=1, keepdims=True), a_min=1e-8, a_max=None)
        targets_one_hot = np.eye(probabilities.shape[1], dtype=np.float64)[y_index]
        brier = float(np.mean(np.sum((probabilities - targets_one_hot) ** 2, axis=1)))
        nll = F.cross_entropy(
            torch.as_tensor(scaled_logits, dtype=torch.float32),
            torch.as_tensor(y_index, dtype=torch.long),
        ).item()
        objective = brier + 0.05 * float(nll)
        if objective < best_objective:
            best_objective = objective
            best_temperature = float(temperature)
    return best_temperature


def _fit_threshold(
    meta: TaskMetadata,
    holder: dict[str, object],
    temperature: float,
    *,
    prevalence_tolerance: float = 0.0,
) -> float:
    if meta.label_type != "binary":
        return 0.5
    probabilities = _sigmoid_binary_probabilities(np.asarray(holder["logits"]), temperature)
    y_true = np.asarray(holder["y_true_index"], dtype=np.int64)
    positive_probabilities = probabilities[:, 1].astype(np.float64)
    unique_probabilities = np.unique(np.round(positive_probabilities, decimals=12))
    if unique_probabilities.size <= 1:
        return 0.5
    midpoint_candidates = (unique_probabilities[:-1] + unique_probabilities[1:]) / 2.0
    candidates = np.concatenate(
        [
            np.asarray([max(float(unique_probabilities[0]) - 1e-6, 0.0)]),
            unique_probabilities,
            midpoint_candidates,
            np.asarray([min(float(unique_probabilities[-1]) + 1e-6, 1.0)]),
        ]
    )
    best_threshold = 0.5
    best_objective = (-1e9, -1e9, -1e9)
    candidate_records: list[tuple[float, float, float, float]] = []
    target_prevalence = float(np.mean(y_true)) if y_true.size else 0.0
    candidate_array = np.asarray(candidates, dtype=np.float64)
    order = np.argsort(positive_probabilities)
    sorted_probabilities = positive_probabilities[order]
    sorted_targets = y_true[order].astype(np.int64)
    positive_prefix = np.concatenate(
        [np.asarray([0], dtype=np.int64), np.cumsum(sorted_targets == 1, dtype=np.int64)]
    )
    negative_prefix = np.concatenate(
        [np.asarray([0], dtype=np.int64), np.cumsum(sorted_targets == 0, dtype=np.int64)]
    )
    split_indices = np.searchsorted(sorted_probabilities, candidate_array, side="left")
    total_positive = int(positive_prefix[-1])
    total_negative = int(negative_prefix[-1])
    true_positive = total_positive - positive_prefix[split_indices]
    false_positive = total_negative - negative_prefix[split_indices]
    false_negative = total_positive - true_positive
    true_negative = total_negative - false_positive

    positive_recall = np.divide(
        true_positive,
        total_positive,
        out=np.zeros_like(candidate_array, dtype=np.float64),
        where=total_positive > 0,
    )
    negative_recall = np.divide(
        true_negative,
        total_negative,
        out=np.zeros_like(candidate_array, dtype=np.float64),
        where=total_negative > 0,
    )
    if total_positive > 0 and total_negative > 0:
        balanced_accuracy_values = 0.5 * (positive_recall + negative_recall)
    elif total_positive > 0:
        balanced_accuracy_values = positive_recall
    elif total_negative > 0:
        balanced_accuracy_values = negative_recall
    else:
        balanced_accuracy_values = np.full_like(candidate_array, -1e9, dtype=np.float64)

    positive_f1_denominator = (2.0 * true_positive) + false_positive + false_negative
    positive_f1 = np.divide(
        2.0 * true_positive,
        positive_f1_denominator,
        out=np.zeros_like(candidate_array, dtype=np.float64),
        where=positive_f1_denominator > 0.0,
    )
    negative_f1_denominator = (2.0 * true_negative) + false_positive + false_negative
    negative_f1 = np.divide(
        2.0 * true_negative,
        negative_f1_denominator,
        out=np.zeros_like(candidate_array, dtype=np.float64),
        where=negative_f1_denominator > 0.0,
    )
    predicted_positive = true_positive + false_positive
    predicted_negative = true_negative + false_negative
    f1_sum = np.zeros_like(candidate_array, dtype=np.float64)
    f1_count = np.zeros_like(candidate_array, dtype=np.float64)
    negative_label_present = (total_negative + predicted_negative) > 0
    positive_label_present = (total_positive + predicted_positive) > 0
    f1_sum = f1_sum + np.where(negative_label_present, negative_f1, 0.0)
    f1_count = f1_count + np.where(negative_label_present, 1.0, 0.0)
    f1_sum = f1_sum + np.where(positive_label_present, positive_f1, 0.0)
    f1_count = f1_count + np.where(positive_label_present, 1.0, 0.0)
    macro_f1_values = np.divide(
        f1_sum,
        f1_count,
        out=np.full_like(candidate_array, -1e9, dtype=np.float64),
        where=f1_count > 0.0,
    )
    predicted_prevalence_values = np.divide(
        predicted_positive,
        y_true.size,
        out=np.zeros_like(candidate_array, dtype=np.float64),
        where=y_true.size > 0,
    )
    tertiary_values = -np.abs(candidate_array - 0.5)
    objective_matrix = np.column_stack([balanced_accuracy_values, macro_f1_values, tertiary_values])
    best_index = 0
    for candidate_index, objective_values in enumerate(objective_matrix):
        objective = (float(objective_values[0]), float(objective_values[1]), float(objective_values[2]))
        threshold_value = float(candidate_array[candidate_index])
        candidate_records.append(
            (
                objective[0],
                objective[1],
                threshold_value,
                float(predicted_prevalence_values[candidate_index]),
            )
        )
        if _objective_improves(objective, best_objective):
            best_objective = objective
            best_threshold = threshold_value
            best_index = candidate_index
    tolerance = max(float(prevalence_tolerance), 0.0)
    if tolerance > 0.0 and candidate_records:
        best_score = float(best_objective[0])
        eligible = [record for record in candidate_records if record[0] >= best_score - tolerance]
        if eligible:
            eligible.sort(
                key=lambda record: (
                    abs(record[3] - target_prevalence),
                    -record[0],
                    -record[1],
                    abs(record[2] - 0.5),
                )
            )
            best_threshold = float(eligible[0][2])
    else:
        best_threshold = float(candidate_array[best_index])
    return best_threshold


def _fit_ordinal_threshold(meta: TaskMetadata, holder: dict[str, object], temperature: float) -> float:
    if meta.label_type != "ordinal":
        return 0.5
    logits = np.asarray(holder["logits"])
    y_true = np.asarray(holder["y_true_index"], dtype=np.int64)
    candidates = np.linspace(0.35, 0.65, 13)
    best_threshold = 0.5
    best_score = -1e9
    cumulative = 1.0 / (1.0 + np.exp(-(logits / max(temperature, 1e-3))))
    cumulative = np.minimum.accumulate(cumulative, axis=1)
    class_probabilities = _ordinal_probabilities(logits, temperature)
    for threshold in candidates:
        y_pred = (cumulative > threshold).sum(axis=1)
        metrics = compute_metrics("multiclass", y_true, y_pred, class_probabilities)
        score = metrics.get("balanced_accuracy")
        if score is not None and not np.isnan(score) and score > best_score:
            best_score = float(score)
            best_threshold = float(threshold)
    return best_threshold


def _fit_ordinal_thresholds(meta: TaskMetadata, holder: dict[str, object], temperature: float) -> list[float]:
    if meta.label_type != "ordinal":
        return [0.5]
    logits = np.asarray(holder["logits"])
    y_true = np.asarray(holder["y_true_index"], dtype=np.int64)
    num_thresholds = logits.shape[1]
    candidates = np.linspace(0.25, 0.75, 11)
    cumulative = 1.0 / (1.0 + np.exp(-(logits / max(temperature, 1e-3))))
    cumulative = np.minimum.accumulate(cumulative, axis=1)
    class_probabilities = _ordinal_probabilities(logits, temperature)
    thresholds = np.full(num_thresholds, 0.5, dtype=np.float64)
    best_global_score = -1e9
    for _ in range(3):
        improved = False
        for threshold_index in range(num_thresholds):
            lower_bound = thresholds[threshold_index - 1] if threshold_index > 0 else candidates[0]
            upper_bound = thresholds[threshold_index + 1] if threshold_index < num_thresholds - 1 else candidates[-1]
            best_threshold = thresholds[threshold_index]
            best_score = -1e9
            for threshold in candidates:
                if threshold < lower_bound or threshold > upper_bound:
                    continue
                candidate_thresholds = thresholds.copy()
                candidate_thresholds[threshold_index] = threshold
                y_pred = (cumulative > candidate_thresholds.reshape(1, -1)).sum(axis=1)
                metrics = compute_metrics("multiclass", y_true, y_pred, class_probabilities)
                score = metrics.get("balanced_accuracy")
                if score is not None and not np.isnan(score) and score > best_score:
                    best_score = float(score)
                    best_threshold = float(threshold)
            thresholds[threshold_index] = best_threshold
            if best_score > best_global_score:
                best_global_score = best_score
                improved = True
        if not improved:
            break
    return thresholds.astype(float).tolist()


def _fit_calibration(
    valid_records: dict[int, dict[str, object]],
    task_metadata: dict[int, TaskMetadata],
    task_training_info: dict[int, dict[str, object]],
    ordinal_calibration_mode: str = "vector",
    fixed_ordinal_threshold_specs: dict[str, list[float]] | None = None,
    fixed_binary_threshold_specs: dict[str, list[float]] | None = None,
    fixed_multiclass_bias_specs: dict[str, list[float]] | None = None,
    regression_calibration_task_keys: set[str] | None = None,
    regression_calibration_mode: str = "affine",
    paired_regression_bridge_task_indices: dict[int, int] | None = None,
    multiclass_bias_task_indices: set[int] | None = None,
    ordinal_bias_task_indices: set[int] | None = None,
    binary_threshold_prevalence_tolerance: float = 0.0,
) -> dict[int, dict[str, float]]:
    params: dict[int, dict[str, object]] = {}
    _progress(f"calibration: fitting parameters for {len(valid_records)} task(s)")
    for task_index, holder in valid_records.items():
        meta = task_metadata[task_index]
        _progress(f"calibration: base fitting for {meta.task_key} ({meta.label_type})")
        if meta.label_type == "continuous":
            calibration = {
                "temperature": 1.0,
                "threshold": 0.5,
                "ordinal_threshold": 0.5,
                "ordinal_thresholds": [0.5],
            }
            if regression_calibration_task_keys and meta.task_key in regression_calibration_task_keys:
                target_mean = float(task_training_info[task_index]["target_mean"])
                target_std = float(task_training_info[task_index]["target_std"])
                raw_predictions = np.asarray(holder["logits"], dtype=np.float64).reshape(-1) * target_std + target_mean
                y_true = np.asarray(holder["target_float"], dtype=np.float64)
                if regression_calibration_mode == "isotonic":
                    if raw_predictions.size >= 2 and np.nanstd(raw_predictions) > 1e-8:
                        iso = IsotonicRegression(out_of_bounds="clip")
                        iso.fit(raw_predictions, y_true)
                        calibration["regression_isotonic_x"] = np.asarray(iso.X_thresholds_, dtype=float).tolist()
                        calibration["regression_isotonic_y"] = np.asarray(iso.y_thresholds_, dtype=float).tolist()
                    else:
                        calibration["regression_isotonic_x"] = [0.0, 1.0]
                        calibration["regression_isotonic_y"] = [0.0, 1.0]
                else:
                    if raw_predictions.size >= 2 and np.nanstd(raw_predictions) > 1e-8:
                        design = np.column_stack([raw_predictions, np.ones_like(raw_predictions)])
                        slope, intercept = np.linalg.lstsq(design, y_true, rcond=None)[0].tolist()
                    else:
                        slope, intercept = 1.0, 0.0
                    calibration["regression_affine_slope"] = float(slope)
                    calibration["regression_affine_intercept"] = float(intercept)
            params[task_index] = calibration
            continue
        temperature = _fit_temperature(meta, holder)
        fixed_binary_threshold = _resolve_task_override(meta, fixed_binary_threshold_specs)
        if meta.label_type == "binary" and fixed_binary_threshold:
            threshold = float(fixed_binary_threshold[0])
        else:
            threshold = _fit_threshold(
                meta,
                holder,
                temperature,
                prevalence_tolerance=float(binary_threshold_prevalence_tolerance),
            )
        fixed_ordinal_thresholds = _resolve_task_override(meta, fixed_ordinal_threshold_specs)
        if meta.label_type == "ordinal" and fixed_ordinal_thresholds:
            num_thresholds = max(len(meta.class_space or []) - 1, 1)
            if len(fixed_ordinal_thresholds) == 1:
                ordinal_thresholds = [float(fixed_ordinal_thresholds[0])] * num_thresholds
            else:
                ordinal_thresholds = [float(value) for value in fixed_ordinal_thresholds[:num_thresholds]]
                if len(ordinal_thresholds) < num_thresholds:
                    ordinal_thresholds.extend([float(ordinal_thresholds[-1])] * (num_thresholds - len(ordinal_thresholds)))
        elif meta.label_type == "ordinal" and ordinal_calibration_mode == "scalar":
            scalar_threshold = _fit_ordinal_threshold(meta, holder, temperature)
            ordinal_thresholds = [float(scalar_threshold)] * max(len(meta.class_space or []) - 1, 1)
        else:
            ordinal_thresholds = _fit_ordinal_thresholds(meta, holder, temperature)
        params[task_index] = {
            "temperature": float(temperature),
            "threshold": float(threshold),
            "ordinal_threshold": float(np.mean(ordinal_thresholds)),
            "ordinal_thresholds": ordinal_thresholds,
        }
        if meta.label_type == "ordinal":
            base_predictions, base_probabilities = _apply_postprocessing(
                meta,
                holder,
                params[task_index],
                task_training_info[task_index],
            )
            best_objective = _classification_holder_objective(holder, base_predictions, base_probabilities)
            expected_scores, expected_probabilities = _ordinal_expected_scores(
                np.asarray(holder["logits"], dtype=np.float64),
                temperature,
            )
            expected_thresholds = _fit_score_bridge_thresholds(
                np.asarray(holder["y_true_index"], dtype=np.int64),
                expected_scores,
                len(meta.class_space or []),
            )
            expected_predictions = _score_to_label_indices(expected_scores, expected_thresholds)
            expected_objective = _classification_holder_objective(
                holder,
                expected_predictions,
                expected_probabilities,
            )
            if _objective_improves(expected_objective, best_objective):
                params[task_index] = {
                    **params[task_index],
                    "prediction_source": "ordinal_expected_score",
                    "ordinal_score_thresholds": [float(value) for value in expected_thresholds],
                }
                best_objective = expected_objective
                _progress(f"calibration: selected expected-score ordinal thresholds for {meta.task_key}")
            if int(task_index) in (ordinal_bias_task_indices or set()):
                biased_probabilities = _ordinal_probabilities(
                    np.asarray(holder["logits"], dtype=np.float64),
                    temperature,
                )
                class_biases = _fit_multiclass_biases(
                    biased_probabilities,
                    np.asarray(holder["y_true_index"], dtype=np.int64),
                )
                biased_probabilities = _apply_multiclass_biases(biased_probabilities, class_biases)
                biased_predictions = biased_probabilities.argmax(axis=1)
                biased_objective = _classification_holder_objective(
                    holder,
                    biased_predictions,
                    biased_probabilities,
                )
                if _objective_improves(biased_objective, best_objective):
                    params[task_index] = {
                        **params[task_index],
                        "prediction_source": "ordinal_class_bias",
                        "ordinal_class_biases": [float(value) for value in class_biases],
                    }
                    best_objective = biased_objective
                    _progress(f"calibration: selected ordinal class-bias decoding for {meta.task_key}")
            aux_logits = holder.get("ordinal_aux_logits")
            if aux_logits is not None:
                aux_logits = np.asarray(aux_logits, dtype=np.float64)
                aux_temperature = _fit_multiclass_temperature(
                    aux_logits,
                    np.asarray(holder["y_true_index"], dtype=np.int64),
                )
                scaled_aux_logits = aux_logits / max(aux_temperature, 1e-3)
                scaled_aux_logits = scaled_aux_logits - scaled_aux_logits.max(axis=1, keepdims=True)
                aux_probabilities = np.exp(scaled_aux_logits)
                aux_probabilities = aux_probabilities / np.clip(
                    aux_probabilities.sum(axis=1, keepdims=True),
                    a_min=1e-8,
                    a_max=None,
                )
                aux_biases = _fit_multiclass_biases(
                    aux_probabilities,
                    np.asarray(holder["y_true_index"], dtype=np.int64),
                )
                aux_probabilities = _apply_multiclass_biases(aux_probabilities, aux_biases)
                aux_predictions = aux_probabilities.argmax(axis=1)
                aux_objective = _classification_holder_objective(
                    holder,
                    aux_predictions,
                    aux_probabilities,
                )
                if _objective_improves(aux_objective, best_objective):
                    params[task_index] = {
                        **params[task_index],
                        "prediction_source": "ordinal_aux_class",
                        "ordinal_aux_temperature": float(aux_temperature),
                        "ordinal_aux_class_biases": [float(value) for value in aux_biases],
                    }
                    _progress(f"calibration: selected auxiliary class head for {meta.task_key}")
        _progress(f"calibration: base fitting complete for {meta.task_key}")

    for task_index in sorted(multiclass_bias_task_indices or set()):
        meta = task_metadata.get(task_index)
        holder = valid_records.get(task_index)
        if meta is None or holder is None or meta.label_type != "multiclass":
            continue
        _progress(f"calibration: fitting multiclass biases for {meta.task_key}")
        fixed_multiclass_biases = _resolve_task_override(meta, fixed_multiclass_bias_specs)
        if fixed_multiclass_biases:
            params.setdefault(task_index, {})["multiclass_class_biases"] = [float(value) for value in fixed_multiclass_biases]
            _progress(f"calibration: reused fixed multiclass biases for {meta.task_key}")
            continue
        temperature = float(params.get(task_index, {}).get("temperature", 1.0))
        logits = np.asarray(holder["logits"], dtype=np.float64) / max(temperature, 1e-3)
        logits = logits - logits.max(axis=1, keepdims=True)
        base_probabilities = np.exp(logits)
        base_probabilities = base_probabilities / np.clip(base_probabilities.sum(axis=1, keepdims=True), a_min=1e-8, a_max=None)
        class_biases = _fit_multiclass_biases(
            base_probabilities,
            np.asarray(holder["y_true_index"], dtype=np.int64),
        )
        params.setdefault(task_index, {})["multiclass_class_biases"] = class_biases
        _progress(f"calibration: multiclass biases complete for {meta.task_key}")

    for target_task_index, source_task_index in sorted((paired_regression_bridge_task_indices or {}).items()):
        target_meta = task_metadata.get(target_task_index)
        source_meta = task_metadata.get(source_task_index)
        target_holder = valid_records.get(target_task_index)
        source_holder = valid_records.get(source_task_index)
        if (
            target_meta is None
            or source_meta is None
            or target_holder is None
            or source_holder is None
            or source_meta.label_type != "continuous"
            or target_meta.label_type == "continuous"
        ):
            continue
        source_predictions, _ = _apply_postprocessing(
            source_meta,
            source_holder,
            params.get(source_task_index, {}),
            task_training_info[source_task_index],
        )
        source_prediction_map = {
            _bridge_match_key(subject_id, anchor_id): float(np.asarray(source_predictions, dtype=np.float64)[row_index])
            for row_index, (subject_id, anchor_id) in enumerate(zip(source_holder["subject_id"], source_holder["anchor_id"]))
        }
        aligned_scores: list[float] = []
        aligned_targets: list[int] = []
        for row_index, (subject_id, anchor_id) in enumerate(zip(target_holder["subject_id"], target_holder["anchor_id"])):
            match_key = _bridge_match_key(subject_id, anchor_id)
            if match_key not in source_prediction_map:
                continue
            aligned_scores.append(float(source_prediction_map[match_key]))
            aligned_targets.append(int(target_holder["y_true_index"][row_index]))
        if not aligned_scores or len(aligned_scores) != len(target_holder["subject_id"]):
            continue
        thresholds = _fit_score_bridge_thresholds(
            np.asarray(aligned_targets, dtype=np.int64),
            np.asarray(aligned_scores, dtype=np.float64),
            len(target_meta.class_space or []),
        )
        current_predictions, current_probabilities = _apply_postprocessing(
            target_meta,
            target_holder,
            params.get(target_task_index, {}),
            task_training_info[target_task_index],
        )
        current_objective = _classification_holder_objective(
            target_holder,
            current_predictions,
            current_probabilities,
        )
        bridge_predictions = _score_to_label_indices(
            np.asarray(aligned_scores, dtype=np.float64),
            thresholds,
        )
        bridge_objective = _classification_holder_objective(
            target_holder,
            bridge_predictions,
            current_probabilities,
        )
        if not _objective_improves(bridge_objective, current_objective):
            _progress(f"calibration: paired regression bridge skipped for {target_meta.task_key} (no validation gain)")
            continue
        target_params = params.setdefault(target_task_index, {})
        target_params["paired_regression_source_task_index"] = int(source_task_index)
        target_params["paired_regression_thresholds"] = [float(value) for value in thresholds]
        target_params["prediction_source"] = "paired_regression_bridge"
        _progress(f"calibration: paired regression bridge ready for {target_meta.task_key}")
    _progress("calibration: all parameters ready")
    return params


def _apply_fixed_calibration_overrides(
    calibration_params: dict[int, dict[str, object]],
    task_metadata: dict[int, TaskMetadata],
    fixed_ordinal_threshold_specs: dict[str, list[float]] | None = None,
    fixed_binary_threshold_specs: dict[str, list[float]] | None = None,
) -> dict[int, dict[str, object]]:
    updated: dict[int, dict[str, object]] = {
        int(task_index): dict(params)
        for task_index, params in calibration_params.items()
    }
    for task_index, meta in task_metadata.items():
        calibration = dict(
            updated.get(
                int(task_index),
                {
                    "temperature": 1.0,
                    "threshold": 0.5,
                    "ordinal_threshold": 0.5,
                    "ordinal_thresholds": [0.5],
                },
            )
        )
        fixed_binary_threshold = _resolve_task_override(meta, fixed_binary_threshold_specs)
        if meta.label_type == "binary" and fixed_binary_threshold:
            calibration["threshold"] = float(fixed_binary_threshold[0])
        fixed_ordinal_thresholds = _resolve_task_override(meta, fixed_ordinal_threshold_specs)
        if meta.label_type == "ordinal" and fixed_ordinal_thresholds:
            num_thresholds = max(len(meta.class_space or []) - 1, 1)
            if len(fixed_ordinal_thresholds) == 1:
                ordinal_thresholds = [float(fixed_ordinal_thresholds[0])] * num_thresholds
            else:
                ordinal_thresholds = [float(value) for value in fixed_ordinal_thresholds[:num_thresholds]]
                if len(ordinal_thresholds) < num_thresholds:
                    ordinal_thresholds.extend([float(ordinal_thresholds[-1])] * (num_thresholds - len(ordinal_thresholds)))
            calibration["ordinal_thresholds"] = ordinal_thresholds
            calibration["ordinal_threshold"] = float(np.mean(ordinal_thresholds))
        updated[int(task_index)] = calibration
    return updated


def _merge_preserved_calibration(
    fitted_calibration: dict[int, dict[str, object]],
    preserved_calibration: dict[int, dict[str, object]] | None,
    preserved_task_indices: set[int] | None,
) -> dict[int, dict[str, object]]:
    merged: dict[int, dict[str, object]] = {
        int(task_index): dict(params)
        for task_index, params in fitted_calibration.items()
    }
    if not preserved_calibration or not preserved_task_indices:
        return merged
    normalized_preserved = {
        int(task_index): dict(params)
        for task_index, params in preserved_calibration.items()
    }
    for task_index in preserved_task_indices:
        if int(task_index) in normalized_preserved:
            merged[int(task_index)] = normalized_preserved[int(task_index)]
    return merged


def _apply_postprocessing(
    meta: TaskMetadata,
    holder: dict[str, object],
    calibration: dict[str, float],
    training_info: dict[str, object],
) -> tuple[np.ndarray, np.ndarray | None]:
    if meta.label_type == "continuous":
        logits = np.asarray(holder["logits"], dtype=np.float64).reshape(-1)
        target_mean = float(training_info["target_mean"])
        target_std = float(training_info["target_std"])
        predictions = logits * target_std + target_mean
        iso_x = calibration.get("regression_isotonic_x")
        iso_y = calibration.get("regression_isotonic_y")
        if iso_x is not None and iso_y is not None:
            iso_x_array = np.asarray(iso_x, dtype=np.float64)
            iso_y_array = np.asarray(iso_y, dtype=np.float64)
            predictions = np.interp(
                predictions,
                iso_x_array,
                iso_y_array,
                left=float(iso_y_array[0]),
                right=float(iso_y_array[-1]),
            )
            return predictions, None
        slope = calibration.get("regression_affine_slope")
        intercept = calibration.get("regression_affine_intercept")
        if slope is not None and intercept is not None:
            predictions = float(slope) * predictions + float(intercept)
        return predictions, None
    temperature = float(calibration.get("temperature", 1.0))
    if meta.label_type == "binary":
        probabilities = _sigmoid_binary_probabilities(np.asarray(holder["logits"]), temperature)
        threshold = float(calibration.get("threshold", 0.5))
        predictions = (probabilities[:, 1] >= threshold).astype(int)
        return predictions, probabilities
    if meta.label_type == "ordinal":
        probabilities = _ordinal_probabilities(np.asarray(holder["logits"]), temperature)
        if calibration.get("prediction_source") == "ordinal_aux_class":
            aux_logits = holder.get("ordinal_aux_logits")
            if aux_logits is not None:
                aux_temperature = float(calibration.get("ordinal_aux_temperature", 1.0))
                logits = np.asarray(aux_logits, dtype=np.float64) / max(aux_temperature, 1e-3)
                logits = logits - logits.max(axis=1, keepdims=True)
                exp_logits = np.exp(logits)
                probabilities = exp_logits / np.clip(exp_logits.sum(axis=1, keepdims=True), a_min=1e-8, a_max=None)
                class_biases = calibration.get("ordinal_aux_class_biases")
                if isinstance(class_biases, list) and class_biases:
                    probabilities = _apply_multiclass_biases(probabilities, class_biases)
                predictions = probabilities.argmax(axis=1)
                return predictions, probabilities
        if calibration.get("prediction_source") == "ordinal_class_bias":
            class_biases = calibration.get("ordinal_class_biases")
            if isinstance(class_biases, list) and class_biases:
                probabilities = _apply_multiclass_biases(probabilities, class_biases)
                predictions = probabilities.argmax(axis=1)
                return predictions, probabilities
        if calibration.get("prediction_source") == "ordinal_expected_score":
            thresholds = calibration.get("ordinal_score_thresholds")
            if isinstance(thresholds, list) and thresholds:
                class_scores = np.arange(probabilities.shape[1], dtype=np.float64)
                expected_scores = probabilities @ class_scores
                predictions = _score_to_label_indices(
                    expected_scores,
                    [float(value) for value in thresholds],
                )
                return predictions, probabilities
        edge_ovr_logits = holder.get("edge_ovr_logits")
        edge_ovr_mix_scale = holder.get("edge_ovr_mix_scale")
        if edge_ovr_logits is not None and edge_ovr_mix_scale is not None:
            probabilities = _edge_ovr_probabilities(
                probabilities,
                np.asarray(edge_ovr_logits, dtype=np.float64),
                np.asarray(edge_ovr_mix_scale, dtype=np.float64).reshape(-1, 1),
            )
            predictions = probabilities.argmax(axis=1)
            return predictions, probabilities
        scaled = np.asarray(holder["logits"], dtype=np.float64) / max(temperature, 1e-3)
        cumulative = 1.0 / (1.0 + np.exp(-scaled))
        cumulative = np.minimum.accumulate(cumulative, axis=1)
        thresholds = calibration.get("ordinal_thresholds")
        if isinstance(thresholds, list) and thresholds:
            threshold_array = np.asarray(thresholds, dtype=np.float64).reshape(1, -1)
            predictions = (cumulative > threshold_array).sum(axis=1)
        else:
            threshold = float(calibration.get("ordinal_threshold", 0.5))
            predictions = (cumulative > threshold).sum(axis=1)
        return predictions, probabilities
    logits = np.asarray(holder["logits"], dtype=np.float64) / max(temperature, 1e-3)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    probabilities = exp_logits / np.clip(exp_logits.sum(axis=1, keepdims=True), a_min=1e-8, a_max=None)
    class_biases = calibration.get("multiclass_class_biases")
    if isinstance(class_biases, list) and class_biases:
        probabilities = _apply_multiclass_biases(probabilities, class_biases)
    predictions = probabilities.argmax(axis=1)
    return predictions, probabilities


def _records_to_outputs(
    split_records: dict[int, dict[str, object]],
    task_metadata: dict[int, TaskMetadata],
    calibration_params: dict[int, dict[str, float]],
    task_training_info: dict[int, dict[str, object]],
    split_name: str,
    concept_names: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    postprocessed_outputs: dict[int, dict[str, object]] = {}

    for task_index, holder in split_records.items():
        meta = task_metadata[task_index]
        calibration = calibration_params.get(task_index, {})
        y_pred_index_or_value, probabilities = _apply_postprocessing(
            meta,
            holder,
            calibration,
            task_training_info[task_index],
        )
        postprocessed_outputs[task_index] = {
            "y_pred": np.asarray(y_pred_index_or_value),
            "probabilities": probabilities,
        }

    for task_index, holder in split_records.items():
        meta = task_metadata[task_index]
        calibration = calibration_params.get(task_index, {})
        output_bundle = postprocessed_outputs[task_index]
        y_pred_index_or_value = output_bundle["y_pred"]
        probabilities = output_bundle["probabilities"]

        source_task_index = calibration.get("paired_regression_source_task_index")
        bridge_thresholds = calibration.get("paired_regression_thresholds")
        if (
            meta.label_type != "continuous"
            and source_task_index is not None
            and isinstance(bridge_thresholds, list)
            and bridge_thresholds
            and int(source_task_index) in postprocessed_outputs
        ):
            source_holder = split_records[int(source_task_index)]
            source_predictions = np.asarray(postprocessed_outputs[int(source_task_index)]["y_pred"], dtype=np.float64)
            source_prediction_map = {
                _bridge_match_key(subject_id, anchor_id): float(source_predictions[row_index])
                for row_index, (subject_id, anchor_id) in enumerate(zip(source_holder["subject_id"], source_holder["anchor_id"]))
            }
            aligned_source_predictions = [
                source_prediction_map[_bridge_match_key(subject_id, anchor_id)]
                for subject_id, anchor_id in zip(holder["subject_id"], holder["anchor_id"])
            ]
            y_pred_index_or_value = _score_to_label_indices(
                np.asarray(aligned_source_predictions, dtype=np.float64),
                [float(value) for value in bridge_thresholds],
            )

        if meta.label_type == "continuous":
            y_true = np.asarray(holder["target_float"], dtype=np.float64)
            metrics = compute_metrics("continuous", y_true, np.asarray(y_pred_index_or_value, dtype=np.float64))
            y_pred_raw = np.asarray(y_pred_index_or_value, dtype=np.float64).tolist()
            y_true_raw = holder["y_true_raw"]
        else:
            y_true_index = np.asarray(holder["y_true_index"], dtype=np.int64)
            metric_label_type = "binary" if meta.label_type == "binary" else "multiclass"
            metrics = compute_metrics(metric_label_type, y_true_index, y_pred_index_or_value, probabilities)
            class_space = np.asarray(meta.class_space, dtype=object)
            y_pred_raw = class_space[np.asarray(y_pred_index_or_value, dtype=np.int64)].tolist()
            y_true_raw = holder["y_true_raw"]

        metrics["split"] = split_name
        metrics["dataset_id"] = meta.dataset_id
        metrics["task_name"] = meta.task_name
        metrics["task_key"] = meta.task_key
        metric_rows.append(metrics)

        for row_index, subject_id in enumerate(holder["subject_id"]):
            row = {
                "split": split_name,
                "dataset_id": meta.dataset_id,
                "subject_id": subject_id,
                "anchor_id": holder["anchor_id"][row_index],
                "task_name": meta.task_name,
                "task_key": meta.task_key,
                "label_type": meta.label_type,
                "y_true": y_true_raw[row_index],
                "y_pred": y_pred_raw[row_index],
                "temperature": calibration.get("temperature", 1.0),
                "threshold": calibration.get("threshold", calibration.get("ordinal_threshold", 0.5)),
            }
            if meta.label_type != "continuous":
                row["y_true_index"] = int(holder["y_true_index"][row_index])
                row["y_pred_index"] = int(np.asarray(y_pred_index_or_value, dtype=np.int64)[row_index])
                if probabilities is not None:
                    for class_position, class_value in enumerate(meta.class_space or []):
                        row[f"proba_{class_value}"] = float(probabilities[row_index, class_position])
            active_concept_names = concept_names or concept_keys_for_dim(holder["concepts"].shape[1])
            for concept_position, concept_name in enumerate(active_concept_names):
                row[f"concept_{concept_name}"] = float(holder["concepts"][row_index, concept_position])
            prediction_rows.append(row)
    return pd.DataFrame(prediction_rows), pd.DataFrame(metric_rows)


def _dataset_balanced_score(metric_frame: pd.DataFrame, task_metadata: dict[int, TaskMetadata]) -> float:
    if metric_frame.empty:
        return -1e9
    task_scores = []
    for row in metric_frame.to_dict(orient="records"):
        label_type = task_metadata[
            next(task_index for task_index, meta in task_metadata.items() if meta.task_key == row["task_key"])
        ].label_type
        task_scores.append(
            {
                "dataset_id": row["dataset_id"],
                "score": _task_metrics_score(label_type, row),
            }
        )
    score_frame = pd.DataFrame(task_scores)
    return float(score_frame.groupby("dataset_id")["score"].mean().mean())


def _selection_score(
    metric_frame: pd.DataFrame,
    task_metadata: dict[int, TaskMetadata],
    train_config: dict[str, object],
) -> float:
    if metric_frame.empty:
        return -1e9
    classification_weight = float(train_config.get("selection_classification_weight", 1.0) or 1.0)
    regression_weight = float(train_config.get("selection_regression_weight", 1.0) or 1.0)
    task_rows = []
    task_by_key = {meta.task_key: meta for meta in task_metadata.values()}
    for row in metric_frame.to_dict(orient="records"):
        meta = task_by_key[str(row["task_key"])]
        metric_score = _task_metrics_score(meta.label_type, row)
        weight = regression_weight if meta.label_type == "continuous" else classification_weight
        task_rows.append(
            {
                "dataset_id": row["dataset_id"],
                "weighted_score": float(metric_score) * float(weight),
                "weight": float(weight),
            }
        )
    score_frame = pd.DataFrame(task_rows)
    dataset_scores = []
    for _, dataset_frame in score_frame.groupby("dataset_id"):
        denominator = max(float(dataset_frame["weight"].sum()), 1e-12)
        dataset_scores.append(float(dataset_frame["weighted_score"].sum() / denominator))
    return float(np.mean(dataset_scores)) if dataset_scores else -1e9


def _metric_frame_to_records(
    metric_frame: pd.DataFrame,
    run_name: str,
    datasets: list[str],
    model_config: dict[str, object],
    train_config: dict[str, object],
    parameter_count: int,
    model_name: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in metric_frame.to_dict(orient="records"):
        record = {
            "run_name": run_name,
            "model_name": model_name,
            "datasets": "|".join(datasets),
            "recursion_steps": model_config["recursion_steps"],
            "parameter_count": parameter_count,
            "use_pcgrad": bool(train_config["use_pcgrad"]),
            "use_gradnorm": bool(train_config.get("use_gradnorm", False)),
            "use_uncertainty_weighting": bool(train_config["use_uncertainty_weighting"]),
            "token_dim": int(model_config["token_dim"]),
            "latent_dim": int(model_config["latent_dim"]),
            "output_refine_dim": int(model_config.get("output_refine_dim", 0)),
            "conditioning_mode": str(model_config.get("conditioning_mode", "dataset_task")),
            "film_conditioning_mode": str(model_config.get("film_conditioning_mode", "both")),
            "enable_missingness_tokens": bool(model_config.get("enable_missingness_tokens", False)),
            "enable_missingness_embedding": bool(model_config.get("enable_missingness_embedding", False)),
            "modality_dropout": float(model_config.get("modality_dropout", 0.0) or 0.0),
            "regression_loss": str(train_config.get("regression_loss", "huber")),
        }
        record.update(row)
        rows.append(record)
    return rows


def _write_calibration_summary(
    calibration_params: dict[int, dict[str, object]],
    task_metadata: dict[int, TaskMetadata],
    run_name: str,
) -> pd.DataFrame:
    rows = []
    for task_index, params in calibration_params.items():
        meta = task_metadata[task_index]
        rows.append(
            {
                "run_name": run_name,
                "dataset_id": meta.dataset_id,
                "task_name": meta.task_name,
                "label_type": meta.label_type,
                "temperature": params.get("temperature", 1.0),
                "threshold": params.get("threshold"),
                "ordinal_threshold": params.get("ordinal_threshold"),
                "ordinal_thresholds": json.dumps(params.get("ordinal_thresholds", [])),
                "regression_affine_slope": params.get("regression_affine_slope"),
                "regression_affine_intercept": params.get("regression_affine_intercept"),
                "regression_isotonic_x": json.dumps(params.get("regression_isotonic_x", [])),
                "regression_isotonic_y": json.dumps(params.get("regression_isotonic_y", [])),
                "multiclass_class_biases": json.dumps(params.get("multiclass_class_biases", [])),
                "ordinal_class_biases": json.dumps(params.get("ordinal_class_biases", [])),
                "ordinal_aux_temperature": params.get("ordinal_aux_temperature"),
                "ordinal_aux_class_biases": json.dumps(params.get("ordinal_aux_class_biases", [])),
                "paired_regression_source_task_key": (
                    task_metadata[int(params["paired_regression_source_task_index"])].task_key
                    if params.get("paired_regression_source_task_index") is not None
                    and int(params["paired_regression_source_task_index"]) in task_metadata
                    else None
                ),
                "paired_regression_thresholds": json.dumps(params.get("paired_regression_thresholds", [])),
                "prediction_source": params.get("prediction_source"),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(TABLE_DIR / f"{run_name}__calibration_summary.csv", index=False)
    return frame


def _write_concept_summary(prediction_frame: pd.DataFrame, run_name: str) -> pd.DataFrame:
    concept_columns = [column for column in prediction_frame.columns if column.startswith("concept_")]
    rows = []
    for (dataset_id, task_name), frame in prediction_frame.groupby(["dataset_id", "task_name"], dropna=False):
        if concept_columns:
            mean_abs = frame[concept_columns].abs().mean(axis=0)
            top_concept = mean_abs.sort_values(ascending=False).index[0]
            top_concept_mean_abs = float(mean_abs[top_concept])
        else:
            top_concept = "none"
            top_concept_mean_abs = 0.0
        rows.append(
            {
                "run_name": run_name,
                "dataset_id": dataset_id,
                "task_name": task_name,
                "top_concept": top_concept,
                "top_concept_mean_abs": top_concept_mean_abs,
            }
        )
    output = pd.DataFrame(rows)
    output.to_csv(CONCEPT_DIR / f"{run_name}__concept_summary.csv", index=False)
    return output


def _write_vs_baseline_summary(metric_frame: pd.DataFrame, run_name: str, model_name: str) -> pd.DataFrame:
    if not BASELINE_RESULTS_PATH.exists():
        output = pd.DataFrame()
        output.to_csv(TABLE_DIR / f"{run_name}__vs_baseline.csv", index=False)
        return output
    baseline = select_validation_best_baselines()
    comparison_rows = []
    for row in metric_frame.to_dict(orient="records"):
        baseline_row = baseline.loc[
            (baseline["dataset_id"] == row["dataset_id"]) & (baseline["task_name"] == row["task_name"])
        ]
        if baseline_row.empty:
            continue
        baseline_row = baseline_row.iloc[0]
        label_type = "continuous" if not pd.isna(row.get("r2")) else "categorical"
        primary_metric = "r2" if label_type == "continuous" else "balanced_accuracy"
        score = row.get(primary_metric)
        comparison_rows.append(
            {
                "run_name": run_name,
                "reference_model_name": model_name,
                "dataset_id": row["dataset_id"],
                "task_name": row["task_name"],
                "primary_metric": primary_metric,
                "model_score": score,
                "baseline_model_name": baseline_row["baseline_model_name"],
                "baseline_experiment_id": baseline_row["baseline_experiment_id"],
                "baseline_selection_metric": baseline_row["baseline_selection_metric"],
                "baseline_selection_score": baseline_row["baseline_selection_score"],
                "baseline_score": baseline_row["baseline_test_score"],
                "delta_model_minus_baseline": float(score - baseline_row["baseline_test_score"]),
                "winner": "model" if score > baseline_row["baseline_test_score"] else ("tie" if score == baseline_row["baseline_test_score"] else "baseline"),
            }
        )
    output = pd.DataFrame(comparison_rows)
    output.to_csv(TABLE_DIR / f"{run_name}__vs_baseline.csv", index=False)
    return output


def main() -> None:
    args = parse_args()
    _set_progress_log_path(args.progress_log_path)
    _set_stall_watchdog(args.stall_seconds)
    if args.stall_seconds is not None and int(args.stall_seconds) > 0:
        _progress(f"Internal stall watchdog enabled ({int(args.stall_seconds)}s)")
    _set_seed(args.seed)
    model_config_path = _resolve_project_path(args.model_config_path, DEFAULT_MODEL_CONFIG_PATH)
    train_config_path = _resolve_project_path(args.train_config_path, DEFAULT_TRAIN_CONFIG_PATH)
    model_config = _load_json(model_config_path)
    train_config = _load_json(train_config_path)
    if args.model_family_name is not None:
        model_config["model_name"] = str(args.model_family_name)
    elif "model_name" not in model_config:
        model_config["model_name"] = "mctrcm_v2_family_a"
    fixed_ordinal_threshold_specs = _parse_task_float_list_specs(args.fixed_ordinal_threshold_specs)
    fixed_binary_threshold_specs = _parse_task_float_list_specs(args.fixed_binary_threshold_specs)
    fixed_multiclass_bias_specs = _parse_task_float_list_specs(args.fixed_multiclass_bias_specs)
    paired_regression_bridge_specs = _parse_task_pair_specs(args.paired_regression_bridge_specs)

    if args.recursion_steps is not None:
        model_config["recursion_steps"] = args.recursion_steps
    if args.disable_concept_bottleneck:
        model_config["concept_dim"] = 0
    elif args.concept_dim is not None:
        model_config["concept_dim"] = int(args.concept_dim)
    elif "concept_dim" not in model_config:
        model_config["concept_dim"] = 8
    model_config["use_concept_bottleneck"] = bool(model_config["concept_dim"] > 0)
    if "task_feature_dim" not in model_config:
        model_config["task_feature_dim"] = int(model_config.get("concept_hidden_dim", 48))
    if args.temporal_frontend_mode is not None:
        model_config["temporal_frontend_mode"] = str(args.temporal_frontend_mode)
    elif "temporal_frontend_mode" not in model_config:
        model_config["temporal_frontend_mode"] = "none"
    model_config["enable_missingness_tokens"] = bool(
        args.enable_missingness_tokens or model_config.get("enable_missingness_tokens", False)
    )
    model_config["enable_missingness_embedding"] = bool(
        args.enable_missingness_embedding or model_config.get("enable_missingness_embedding", False)
    )
    if args.disable_missingness_tokens:
        model_config["enable_missingness_tokens"] = False
    if args.disable_missingness_embedding:
        model_config["enable_missingness_embedding"] = False
    if args.missingness_embedding_norm:
        model_config["missingness_embedding_norm"] = True
    elif "missingness_embedding_norm" not in model_config:
        model_config["missingness_embedding_norm"] = False
    if args.modality_dropout is not None:
        model_config["modality_dropout"] = float(args.modality_dropout)
    elif "modality_dropout" not in model_config:
        model_config["modality_dropout"] = 0.0
    if args.conditioning_mode is not None:
        model_config["conditioning_mode"] = str(args.conditioning_mode)
    elif "conditioning_mode" not in model_config:
        model_config["conditioning_mode"] = "dataset_task"
    if args.disable_film_conditioning:
        model_config["film_conditioning_mode"] = "none"
    elif args.film_conditioning_mode is not None:
        model_config["film_conditioning_mode"] = str(args.film_conditioning_mode)
    elif "film_conditioning_mode" not in model_config:
        model_config["film_conditioning_mode"] = "both"
    model_config["enable_tree_gated_head"] = bool(
        args.enable_tree_gated_head or model_config.get("enable_tree_gated_head", False)
    )
    model_config["use_ordered_threshold_ordinal_head"] = bool(
        args.enable_ordered_threshold_ordinal_head
        or model_config.get("use_ordered_threshold_ordinal_head", False)
    )
    if args.tree_head_depth is not None:
        model_config["tree_head_depth"] = int(args.tree_head_depth)
    elif "tree_head_depth" not in model_config:
        model_config["tree_head_depth"] = 2
    if args.epochs is not None:
        train_config["max_epochs"] = args.epochs
    if args.batch_size is not None:
        train_config["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        train_config["learning_rate"] = args.learning_rate
    if args.weight_decay is not None:
        train_config["weight_decay"] = args.weight_decay
    if args.dropout is not None:
        model_config["dropout"] = float(args.dropout)
    if args.token_dim is not None:
        model_config["token_dim"] = int(args.token_dim)
    if args.latent_dim is not None:
        model_config["latent_dim"] = int(args.latent_dim)
    if args.encoder_hidden_dim is not None:
        model_config["encoder_hidden_dim"] = int(args.encoder_hidden_dim)
    if args.transformer_ff_dim is not None:
        model_config["transformer_ff_dim"] = int(args.transformer_ff_dim)
    if args.output_refine_dim is not None:
        model_config["output_refine_dim"] = int(args.output_refine_dim)
    if args.patience is not None:
        train_config["patience"] = args.patience
    if args.min_epochs is not None:
        train_config["min_epochs"] = args.min_epochs
    if args.disable_pcgrad:
        train_config["use_pcgrad"] = False
    if args.use_gradnorm:
        train_config["use_gradnorm"] = True
        train_config["use_pcgrad"] = False
    elif "use_gradnorm" not in train_config:
        train_config["use_gradnorm"] = False
    if args.disable_uncertainty_weighting:
        train_config["use_uncertainty_weighting"] = False
    if args.regression_loss is not None:
        train_config["regression_loss"] = str(args.regression_loss)
    if args.distill_mode is not None:
        train_config["distillation"]["mode"] = args.distill_mode
    if args.distill_task_keys is not None:
        train_config["distillation"]["task_subset"] = [str(task_key) for task_key in args.distill_task_keys]
    elif "task_subset" not in train_config["distillation"]:
        train_config["distillation"]["task_subset"] = []
    if args.distill_task_weights is not None:
        task_weights = {}
        for item in args.distill_task_weights:
            if "=" not in str(item):
                continue
            task_key, weight = str(item).split("=", 1)
            task_weights[str(task_key)] = float(weight)
        train_config["distillation"]["task_weights"] = task_weights
    elif "task_weights" not in train_config["distillation"]:
        train_config["distillation"]["task_weights"] = {}
    task_loss_weight_specs = _parse_task_scalar_specs(args.task_loss_weight_specs)
    train_config["task_loss_weights"] = task_loss_weight_specs
    if args.distill_classification_weight is not None:
        train_config["distillation"]["classification_weight"] = float(args.distill_classification_weight)
    if args.distill_regression_weight is not None:
        train_config["distillation"]["regression_weight"] = float(args.distill_regression_weight)
    if args.l1_penalty is not None:
        train_config["sparse_penalty"]["l1"] = float(args.l1_penalty)
    if args.group_penalty is not None:
        train_config["sparse_penalty"]["group"] = float(args.group_penalty)
    if args.disable_sparse_penalty:
        train_config["sparse_penalty"]["l1"] = 0.0
        train_config["sparse_penalty"]["group"] = 0.0
    if args.sparse_warmup_epochs is not None:
        train_config["sparse_warmup_epochs"] = int(args.sparse_warmup_epochs)
    elif "sparse_warmup_epochs" not in train_config:
        train_config["sparse_warmup_epochs"] = 0
    if args.shared_token_refiner_type is not None:
        model_config["shared_token_refiner_type"] = str(args.shared_token_refiner_type)
    elif "shared_token_refiner_type" not in model_config:
        model_config["shared_token_refiner_type"] = "none"
    if args.shared_token_refiner_steps is not None:
        model_config["shared_token_refiner_steps"] = int(args.shared_token_refiner_steps)
    elif "shared_token_refiner_steps" not in model_config:
        model_config["shared_token_refiner_steps"] = 0
    if args.shared_token_refiner_nograd_steps is not None:
        model_config["shared_token_refiner_nograd_steps"] = int(args.shared_token_refiner_nograd_steps)
    elif "shared_token_refiner_nograd_steps" not in model_config:
        model_config["shared_token_refiner_nograd_steps"] = 0
    if args.task_local_trm_mode is not None:
        model_config["task_local_trm_mode"] = str(args.task_local_trm_mode)
    elif "task_local_trm_mode" not in model_config:
        model_config["task_local_trm_mode"] = "none"
    if args.task_local_trm_reasoning_dim is not None:
        model_config["task_local_trm_reasoning_dim"] = int(args.task_local_trm_reasoning_dim)
    elif "task_local_trm_reasoning_dim" not in model_config:
        model_config["task_local_trm_reasoning_dim"] = int(model_config["token_dim"])
    if args.task_local_trm_h_cycles is not None:
        model_config["task_local_trm_h_cycles"] = int(args.task_local_trm_h_cycles)
    elif "task_local_trm_h_cycles" not in model_config:
        model_config["task_local_trm_h_cycles"] = 0
    if args.task_local_trm_l_cycles is not None:
        model_config["task_local_trm_l_cycles"] = int(args.task_local_trm_l_cycles)
    elif "task_local_trm_l_cycles" not in model_config:
        model_config["task_local_trm_l_cycles"] = 0
    if args.psyche_hierarchical_weight is not None:
        train_config["psyche_hierarchical_weight"] = float(args.psyche_hierarchical_weight)
    elif "psyche_hierarchical_weight" not in train_config:
        train_config["psyche_hierarchical_weight"] = 0.0
    if args.ordinal_calibration_mode is not None:
        train_config["ordinal_calibration_mode"] = str(args.ordinal_calibration_mode)
    elif "ordinal_calibration_mode" not in train_config:
        train_config["ordinal_calibration_mode"] = "vector"
    if args.regression_calibration_mode is not None:
        train_config["regression_calibration_mode"] = str(args.regression_calibration_mode)
    elif "regression_calibration_mode" not in train_config:
        train_config["regression_calibration_mode"] = "affine"
    if args.deprest_gad7_pair_weight is not None:
        train_config["deprest_gad7_pair_weight"] = float(args.deprest_gad7_pair_weight)
    elif "deprest_gad7_pair_weight" not in train_config:
        train_config["deprest_gad7_pair_weight"] = 0.0
    if args.deprest_edge_ovr_weight is not None:
        train_config["deprest_edge_ovr_weight"] = float(args.deprest_edge_ovr_weight)
    elif "deprest_edge_ovr_weight" not in train_config:
        train_config["deprest_edge_ovr_weight"] = 0.0
    if args.deprest_construct_weight is not None:
        train_config["deprest_construct_weight"] = float(args.deprest_construct_weight)
    elif "deprest_construct_weight" not in train_config:
        train_config["deprest_construct_weight"] = 0.0
    if args.psyche_delta_weight is not None:
        train_config["psyche_delta_weight"] = float(args.psyche_delta_weight)
    elif "psyche_delta_weight" not in train_config:
        train_config["psyche_delta_weight"] = 0.0
    train_config.setdefault("task_sampling_power", 0.0)
    train_config.setdefault("classification_sampling_power", 0.0)
    if args.task_sampling_power is not None:
        train_config["task_sampling_power"] = float(args.task_sampling_power)
    if args.classification_sampling_power is not None:
        train_config["classification_sampling_power"] = float(args.classification_sampling_power)
    train_config.setdefault("max_sample_weight_multiplier", 25.0)
    train_config.setdefault("selection_classification_weight", 1.0)
    train_config.setdefault("selection_regression_weight", 1.0)
    train_config.setdefault("auto_paired_regression_bridge", False)
    train_config.setdefault("auto_multiclass_bias_calibration", False)
    train_config.setdefault("auto_ordinal_bias_calibration", False)
    train_config.setdefault("binary_threshold_prevalence_tolerance", 0.0)
    train_config.setdefault("regression_loss", "huber")
    if args.auto_ordinal_bias_calibration:
        train_config["auto_ordinal_bias_calibration"] = True
    train_config.setdefault("class_balance_mode", "inverse")
    train_config.setdefault("class_balance_beta", 0.999)
    train_config.setdefault("max_class_weight", None)
    train_config.setdefault("ordinal_aux_class_weight", 0.0)
    if args.class_balance_mode is not None:
        train_config["class_balance_mode"] = str(args.class_balance_mode)
    if args.class_balance_beta is not None:
        train_config["class_balance_beta"] = float(args.class_balance_beta)
    if args.max_class_weight is not None:
        train_config["max_class_weight"] = float(args.max_class_weight)
    if args.ordinal_aux_class_weight is not None:
        train_config["ordinal_aux_class_weight"] = float(args.ordinal_aux_class_weight)
    if args.binary_threshold_prevalence_tolerance is not None:
        train_config["binary_threshold_prevalence_tolerance"] = float(args.binary_threshold_prevalence_tolerance)

    _progress(
        "Preparing data for "
        + ",".join(args.datasets)
        + f" (native_branch={'off' if args.disable_native_branch else 'on'}, "
        + f"augmented_comm={'off' if args.base_communication_only else 'on'}, "
        + f"feature_source={args.feature_source_condition})"
    )
    prepared = prepare_multicorpus_data(
        args.datasets,
        include_native_features=(not args.disable_native_branch) or bool(args.enable_psyche_native_adapter),
        include_augmented_communication_features=not args.base_communication_only,
        feature_source_condition=args.feature_source_condition,
    )
    _progress(
        "Prepared splits: "
        + f"train={len(prepared.train.frame)} "
        + f"valid={len(prepared.valid.frame)} "
        + f"test={len(prepared.test.frame)} "
        + f"tasks={len(prepared.task_metadata)} "
        + f"datasets={len(prepared.dataset_to_index)}"
    )
    task_label_overrides = _apply_task_label_overrides(
        prepared,
        multiclass_task_keys=args.multiclass_task_keys,
    )
    task_metadata = {meta.task_index: meta for meta in prepared.task_metadata}
    task_key_to_index = {
        str(meta.task_key): int(meta.task_index)
        for meta in prepared.task_metadata
    }
    task_name_to_indices: dict[str, list[int]] = defaultdict(list)
    for meta in prepared.task_metadata:
        task_name_to_indices[str(meta.task_name)].append(int(meta.task_index))

    def _resolve_requested_task_index(requested_task_key: str) -> int:
        requested_task_key = str(requested_task_key).strip()
        if requested_task_key in task_key_to_index:
            return int(task_key_to_index[requested_task_key])
        candidate_indices = task_name_to_indices.get(requested_task_key, [])
        if len(candidate_indices) == 1:
            return int(candidate_indices[0])
        raise ValueError(f"No unique task match found for calibration task spec {requested_task_key!r}")

    def _resolve_optional_task_index(requested_task_key: str) -> int | None:
        requested_task_key = str(requested_task_key).strip()
        if requested_task_key in task_key_to_index:
            return int(task_key_to_index[requested_task_key])
        candidate_indices = task_name_to_indices.get(requested_task_key, [])
        if len(candidate_indices) == 1:
            return int(candidate_indices[0])
        return None

    paired_regression_bridge_task_indices = {
        _resolve_requested_task_index(target_task_key): _resolve_requested_task_index(source_task_key)
        for target_task_key, source_task_key in paired_regression_bridge_specs.items()
    }
    if bool(train_config.get("auto_paired_regression_bridge", False)):
        inferred_bridges = _infer_auto_paired_regression_bridges(task_metadata)
        for target_task_index, source_task_index in inferred_bridges.items():
            paired_regression_bridge_task_indices.setdefault(int(target_task_index), int(source_task_index))
        if inferred_bridges:
            _progress(
                "Auto paired regression bridges: "
                + ", ".join(
                    f"{task_metadata[target].task_key}->{task_metadata[source].task_key}"
                    for target, source in sorted(inferred_bridges.items())
                )
            )
    multiclass_bias_task_indices = {
        _resolve_requested_task_index(task_key)
        for task_key in (args.multiclass_bias_task_keys or [])
        if str(task_key).strip()
    }
    multiclass_bias_task_indices.update(
        _resolve_requested_task_index(task_key)
        for task_key in fixed_multiclass_bias_specs.keys()
    )
    if bool(train_config.get("auto_multiclass_bias_calibration", False)):
        multiclass_bias_task_indices.update(
            int(meta.task_index)
            for meta in prepared.task_metadata
            if meta.label_type == "multiclass"
        )
    ordinal_bias_task_keys = {
        str(task_key).strip()
        for task_key in (args.ordinal_bias_task_keys or train_config.get("ordinal_bias_task_keys") or [])
        if str(task_key).strip()
    }
    ordinal_bias_task_indices = {
        int(task_index)
        for task_index in (_resolve_optional_task_index(task_key) for task_key in ordinal_bias_task_keys)
        if task_index is not None
    }
    if bool(train_config.get("auto_ordinal_bias_calibration", False)):
        ordinal_bias_task_indices.update(
            int(meta.task_index)
            for meta in prepared.task_metadata
            if meta.label_type == "ordinal"
        )
    task_head_specs = _build_task_head_specs(prepared.task_metadata)
    modality_input_dims = {
        modality: len(columns)
        for modality, columns in prepared.modality_feature_columns.items()
    }
    model_config["native_input_dim"] = 0 if args.disable_native_branch else len(prepared.native_feature_columns)
    model_config["psyche_native_input_dim"] = len(prepared.native_feature_columns) if args.enable_psyche_native_adapter else 0
    model_config["use_psyche_taskwise_native_adapter"] = bool(args.enable_psyche_taskwise_native_adapter)
    model_config["deprest_comm_input_dim"] = len(prepared.deprest_comm_feature_columns)
    model_config["temporal_slice_dims"] = dict(prepared.temporal_slice_dims)
    model_config["temporal_num_slices"] = int(prepared.temporal_num_slices)
    model_config["missing_signal_dim"] = int(prepared.missing_signal_dim)
    model_config["native_missing_signal_dim"] = int(prepared.native_missing_signal_dim)
    model_config["concept_names"] = concept_keys_for_dim(int(model_config["concept_dim"]))
    model_config["deprest_comm_group_sizes"] = {
        key: len(value)
        for key, value in _derive_deprest_comm_groups(prepared.deprest_comm_feature_columns).items()
    }
    deprest_comm_groups = _derive_deprest_comm_groups(prepared.deprest_comm_feature_columns)
    use_deprest_comm_features = (
        (not args.disable_deprest_adapter)
        or bool(args.enable_deprest_category_adapter)
        or bool(args.enable_deprest_coverage_routing)
        or bool(args.enable_deprest_concept_compatibility)
        or bool(args.enable_deprest_concept_contrast)
        or bool(args.enable_deprest_edge_specialist)
        or bool(args.enable_deprest_edge_ovr)
    )
    deprest_category_task_indices = sorted(
        meta.task_index
        for meta in prepared.task_metadata
        if meta.dataset_id == "deprest_cat" and meta.task_name in {"gad7_cat", "phq9_cat"}
    )
    deprest_coverage_task_names = {
        str(task_name).strip()
        for task_name in (args.deprest_coverage_task_names or ["gad7_cat"])
        if str(task_name).strip()
    }
    deprest_coverage_task_indices = sorted(
        meta.task_index
        for meta in prepared.task_metadata
        if bool(args.enable_deprest_coverage_routing)
        and meta.dataset_id == "deprest_cat"
        and meta.task_name in deprest_coverage_task_names
    )
    deprest_concept_compat_task_names = {
        str(task_name).strip()
        for task_name in (args.deprest_concept_compat_task_names or ["gad7_cat"])
        if str(task_name).strip()
    }
    deprest_concept_compat_task_indices = sorted(
        meta.task_index
        for meta in prepared.task_metadata
        if bool(args.enable_deprest_concept_compatibility)
        and meta.dataset_id == "deprest_cat"
        and meta.task_name in deprest_concept_compat_task_names
    )
    deprest_concept_compat_feature_indices = [
        index
        for index, column in enumerate(prepared.deprest_comm_feature_columns)
        if any(token in column for token in ("_coverage_", "_share_", "_duration_", "_contacts_", "_outgoing_incoming_"))
    ]
    deprest_concept_contrast_task_names = {
        str(task_name).strip()
        for task_name in (args.deprest_concept_contrast_task_names or ["gad7_cat"])
        if str(task_name).strip()
    }
    deprest_concept_contrast_task_indices = sorted(
        meta.task_index
        for meta in prepared.task_metadata
        if bool(args.enable_deprest_concept_contrast)
        and meta.dataset_id == "deprest_cat"
        and meta.task_name in deprest_concept_contrast_task_names
    )
    deprest_concept_contrast_feature_indices = [
        index
        for index, column in enumerate(prepared.deprest_comm_feature_columns)
        if any(token in column for token in ("_coverage_", "_share_", "_duration_", "_contacts_", "_outgoing_incoming_"))
    ]
    deprest_concept_contrast_pair_indices = [5, 6]
    deprest_edge_task_names = {
        str(task_name).strip()
        for task_name in (args.deprest_edge_task_names or ["gad7_cat"])
        if str(task_name).strip()
    }
    deprest_edge_task_indices = sorted(
        meta.task_index
        for meta in prepared.task_metadata
        if bool(args.enable_deprest_edge_specialist)
        and meta.dataset_id == "deprest_cat"
        and meta.task_name in deprest_edge_task_names
    )
    deprest_edge_ovr_task_names = {
        str(task_name).strip()
        for task_name in (args.deprest_edge_ovr_task_names or ["gad7_cat"])
        if str(task_name).strip()
    }
    deprest_edge_ovr_task_indices = sorted(
        meta.task_index
        for meta in prepared.task_metadata
        if bool(args.enable_deprest_edge_ovr)
        and meta.dataset_id == "deprest_cat"
        and meta.task_name in deprest_edge_ovr_task_names
    )
    use_psyche_conditioning = (not args.disable_psyche_hierarchical) or bool(args.enable_psyche_two_stage)
    psyche_change_task_indices = sorted(
        meta.task_index
        for meta in prepared.task_metadata
        if meta.dataset_id == "psyche_d" and meta.task_name in {"phq_change_binary", "phq_change_multiclass"}
    )
    psyche_binary_correction_task_indices = sorted(
        meta.task_index
        for meta in prepared.task_metadata
        if bool(args.enable_psyche_binary_correction)
        and meta.dataset_id == "psyche_d"
        and meta.task_name == "phq_change_binary"
    )
    psyche_native_task_names = {
        str(task_name).strip()
        for task_name in (args.psyche_native_task_names or ["phq_change_binary", "phq_change_multiclass"])
        if str(task_name).strip()
    }
    psyche_native_task_indices = sorted(
        meta.task_index
        for meta in prepared.task_metadata
        if bool(args.enable_psyche_native_adapter)
        and meta.dataset_id == "psyche_d"
        and meta.task_name in psyche_native_task_names
    )
    task_local_trm_task_names = {
        str(task_name).strip()
        for task_name in (args.task_local_trm_task_names or [])
        if str(task_name).strip()
    }
    task_local_trm_task_indices = sorted(
        meta.task_index
        for meta in prepared.task_metadata
        if task_local_trm_task_names and meta.task_name in task_local_trm_task_names
    )
    tree_head_task_names = {
        str(task_name).strip()
        for task_name in (args.tree_head_task_names or [])
        if str(task_name).strip()
    }
    tree_head_task_indices = sorted(
        meta.task_index
        for meta in prepared.task_metadata
        if bool(model_config.get("enable_tree_gated_head", False)) and meta.task_name in tree_head_task_names
    )
    auto_tree_head_label_types = {
        str(label_type).strip()
        for label_type in model_config.get("auto_tree_head_label_types", [])
        if str(label_type).strip()
    }
    if bool(model_config.get("enable_tree_gated_head", False)) and auto_tree_head_label_types:
        tree_head_task_indices = sorted(
            set(tree_head_task_indices)
            | {
                int(meta.task_index)
                for meta in prepared.task_metadata
                if meta.label_type in auto_tree_head_label_types
            }
        )
    ordinal_aux_class_task_keys = {
        str(task_key).strip()
        for task_key in (args.ordinal_aux_class_task_keys or train_config.get("ordinal_aux_class_task_keys") or [])
        if str(task_key).strip()
    }
    ordinal_aux_class_task_indices = sorted(
        int(task_index)
        for task_index in (_resolve_optional_task_index(task_key) for task_key in ordinal_aux_class_task_keys)
        if task_index is not None
    )
    if bool(args.enable_ordinal_aux_class_head or train_config.get("enable_ordinal_aux_class_head", False)):
        if not ordinal_aux_class_task_indices:
            ordinal_aux_class_task_indices = sorted(
                int(meta.task_index)
                for meta in prepared.task_metadata
                if meta.label_type == "ordinal"
            )
    deprest_construct_bridge_specs: dict[str, dict[str, object]] = {}
    if args.enable_deprest_construct_bridge:
        selected_construct_tasks = {
            str(task_name).strip()
            for task_name in (args.deprest_construct_task_names or ["gad7_cat", "phq9_cat"])
            if str(task_name).strip()
        }
        deprest_meta_by_name = {
            str(meta.task_name): meta
            for meta in prepared.task_metadata
            if meta.dataset_id == "deprest_cat"
        }
        construct_definitions = [
            ("gad7", "gad7_cat", "gad7_reg", [5.0, 10.0, 15.0], 21.0),
            ("phq9", "phq9_cat", "phq9_reg", [5.0, 10.0, 15.0, 20.0], 27.0),
        ]
        for construct_key, categorical_task_name, regression_task_name, thresholds, score_max in construct_definitions:
            if categorical_task_name not in selected_construct_tasks:
                continue
            categorical_meta = deprest_meta_by_name.get(categorical_task_name)
            regression_meta = deprest_meta_by_name.get(regression_task_name)
            if categorical_meta is None:
                continue
            task_indices = [int(categorical_meta.task_index)]
            if regression_meta is not None:
                task_indices.append(int(regression_meta.task_index))
            deprest_construct_bridge_specs[construct_key] = {
                "task_indices": task_indices,
                "categorical_task_indices": [int(categorical_meta.task_index)],
                "thresholds": thresholds,
                "score_max": score_max,
            }
    selected_train_task_indices = _selected_task_index_set(prepared.task_metadata, args.train_task_keys)
    if args.train_task_keys and not selected_train_task_indices:
        raise ValueError(f"No matching task keys found for train-task-keys={args.train_task_keys}")
    deprest_severity_specs: dict[int, dict[str, object]] = {}
    if args.enable_deprest_severity_head:
        selected_severity_tasks = {
            str(task_name).strip()
            for task_name in (args.deprest_severity_task_names or ["phq9_cat", "gad7_cat"])
            if str(task_name).strip()
        }
        for meta in prepared.task_metadata:
            if meta.dataset_id != "deprest_cat":
                continue
            if meta.task_name == "phq9_cat" and meta.task_name in selected_severity_tasks:
                deprest_severity_specs[int(meta.task_index)] = {
                    "thresholds": [5.0, 10.0, 15.0, 20.0],
                    "score_max": 27.0,
                }
            elif meta.task_name == "gad7_cat" and meta.task_name in selected_severity_tasks:
                deprest_severity_specs[int(meta.task_index)] = {
                    "thresholds": [5.0, 10.0, 15.0],
                    "score_max": 21.0,
                }
    deprest_pair_specs: dict[int, dict[str, object]] = {}
    if args.enable_deprest_gad7_pair_aux:
        task_meta_by_name = {
            str(meta.task_name): meta
            for meta in prepared.task_metadata
            if meta.dataset_id == "deprest_cat"
        }
        gad7_cat_meta = task_meta_by_name.get("gad7_cat")
        gad7_reg_meta = task_meta_by_name.get("gad7_reg")
        if gad7_cat_meta is not None and gad7_reg_meta is not None:
            deprest_pair_specs[int(gad7_cat_meta.task_index)] = {
                "target_task_index": int(gad7_reg_meta.task_index),
                "output_dim": 1,
            }
            deprest_pair_specs[int(gad7_reg_meta.task_index)] = {
                "target_task_index": int(gad7_cat_meta.task_index),
                "output_dim": max(len(gad7_cat_meta.class_space or []) - 1, 1) if gad7_cat_meta.label_type == "ordinal" else gad7_cat_meta.output_dim,
            }
    deprest_task_bridge_specs: dict[int, dict[str, object]] = {}
    if args.enable_deprest_gad7_reg_bridge:
        task_meta_by_name = {
            str(meta.task_name): meta
            for meta in prepared.task_metadata
            if meta.dataset_id == "deprest_cat"
        }
        gad7_cat_meta = task_meta_by_name.get("gad7_cat")
        gad7_reg_meta = task_meta_by_name.get("gad7_reg")
        if gad7_cat_meta is not None and gad7_reg_meta is not None:
            deprest_task_bridge_specs[int(gad7_cat_meta.task_index)] = {
                "source_task_index": int(gad7_reg_meta.task_index),
            }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        _progress(f"Using device cuda:{torch.cuda.current_device()} ({torch.cuda.get_device_name(0)})")
    else:
        _progress("Using device cpu")
    model = MCTRCMV2(
        modality_input_dims=modality_input_dims,
        num_datasets=len(prepared.dataset_to_index),
        task_head_specs=task_head_specs,
        native_input_dim=0 if args.disable_native_branch else len(prepared.native_feature_columns),
        temporal_slice_dims=prepared.temporal_slice_dims,
        temporal_num_slices=prepared.temporal_num_slices,
        temporal_frontend_mode=str(model_config.get("temporal_frontend_mode", "none")),
        enable_missingness_tokens=bool(model_config.get("enable_missingness_tokens", False)),
        enable_missingness_embedding=bool(model_config.get("enable_missingness_embedding", False)),
        missing_signal_dim=int(prepared.missing_signal_dim),
        native_missing_signal_dim=int(prepared.native_missing_signal_dim),
        psyche_native_input_dim=len(prepared.native_feature_columns) if args.enable_psyche_native_adapter else 0,
        psyche_dataset_index=None if not use_psyche_conditioning else prepared.dataset_to_index.get("psyche_d"),
        use_psyche_two_stage=bool(args.enable_psyche_two_stage),
        psyche_change_task_indices=psyche_change_task_indices if args.enable_psyche_two_stage else [],
        psyche_binary_correction_task_indices=psyche_binary_correction_task_indices if args.enable_psyche_two_stage else [],
        use_psyche_native_adapter=bool(args.enable_psyche_native_adapter),
        use_psyche_taskwise_native_adapter=bool(args.enable_psyche_taskwise_native_adapter),
        psyche_native_task_indices=psyche_native_task_indices,
        deprest_dataset_index=None if not use_deprest_comm_features else prepared.dataset_to_index.get("deprest_cat"),
        deprest_comm_input_dim=0 if not use_deprest_comm_features else len(prepared.deprest_comm_feature_columns),
        deprest_comm_group_indices={} if not use_deprest_comm_features else deprest_comm_groups,
        use_deprest_global_adapter=not args.disable_deprest_adapter,
        use_deprest_category_adapter=bool(args.enable_deprest_category_adapter),
        deprest_category_task_indices=deprest_category_task_indices if args.enable_deprest_category_adapter else [],
        use_deprest_coverage_routing=bool(args.enable_deprest_coverage_routing),
        deprest_coverage_task_indices=deprest_coverage_task_indices,
        use_deprest_concept_compatibility=bool(args.enable_deprest_concept_compatibility),
        deprest_concept_compat_task_indices=deprest_concept_compat_task_indices,
        deprest_concept_compat_feature_indices=deprest_concept_compat_feature_indices,
        use_deprest_concept_contrast=bool(args.enable_deprest_concept_contrast),
        deprest_concept_contrast_task_indices=deprest_concept_contrast_task_indices,
        deprest_concept_contrast_feature_indices=deprest_concept_contrast_feature_indices,
        deprest_concept_contrast_pair_indices=deprest_concept_contrast_pair_indices,
        deprest_task_bridge_specs=deprest_task_bridge_specs,
        deprest_severity_specs=deprest_severity_specs,
        deprest_pair_specs=deprest_pair_specs,
        deprest_construct_bridge_specs=deprest_construct_bridge_specs,
        use_concept_residual=bool(args.enable_concept_residual),
        shared_token_refiner_type=str(model_config.get("shared_token_refiner_type", "none")),
        shared_token_refiner_steps=int(model_config.get("shared_token_refiner_steps", 0)),
        shared_token_refiner_nograd_steps=int(model_config.get("shared_token_refiner_nograd_steps", 0)),
        task_local_trm_mode=str(model_config.get("task_local_trm_mode", "none")),
        task_local_trm_reasoning_dim=int(model_config.get("task_local_trm_reasoning_dim", model_config["token_dim"])),
        task_local_trm_h_cycles=int(model_config.get("task_local_trm_h_cycles", 0)),
        task_local_trm_l_cycles=int(model_config.get("task_local_trm_l_cycles", 0)),
        task_local_trm_task_indices=task_local_trm_task_indices,
        use_psyche_delta_bridge=bool(args.enable_psyche_delta_bridge),
        psyche_delta_bridge_task_indices=psyche_change_task_indices if args.enable_psyche_delta_bridge else [],
        use_deprest_edge_specialist=bool(args.enable_deprest_edge_specialist),
        deprest_edge_task_indices=deprest_edge_task_indices,
        use_deprest_edge_ovr=bool(args.enable_deprest_edge_ovr),
        deprest_edge_ovr_task_indices=deprest_edge_ovr_task_indices,
        enable_tree_gated_head=bool(model_config.get("enable_tree_gated_head", False)),
        tree_head_task_indices=tree_head_task_indices,
        tree_head_depth=int(model_config.get("tree_head_depth", 2)),
        use_ordered_threshold_ordinal_head=bool(model_config.get("use_ordered_threshold_ordinal_head", False)),
        ordinal_aux_class_task_indices=ordinal_aux_class_task_indices,
        token_dim=model_config["token_dim"],
        encoder_hidden_dim=model_config["encoder_hidden_dim"],
        transformer_layers=model_config["transformer_layers"],
        transformer_heads=model_config["transformer_heads"],
        transformer_ff_dim=model_config["transformer_ff_dim"],
        latent_dim=model_config["latent_dim"],
        dataset_embedding_dim=model_config["dataset_embedding_dim"],
        task_embedding_dim=model_config["task_embedding_dim"],
        disable_task_conditioning=bool(model_config.get("disable_task_conditioning", False)),
        conditioning_mode=str(model_config.get("conditioning_mode", "dataset_task")),
        film_conditioning_mode=str(model_config.get("film_conditioning_mode", "both")),
        concept_dim=model_config["concept_dim"],
        concept_hidden_dim=model_config["concept_hidden_dim"],
        task_feature_dim=int(model_config.get("task_feature_dim", model_config["concept_hidden_dim"])),
        output_refine_dim=model_config["output_refine_dim"],
        recursion_steps=model_config["recursion_steps"],
        modality_dropout=float(model_config.get("modality_dropout", 0.0) or 0.0),
        missingness_embedding_norm=bool(model_config.get("missingness_embedding_norm", False)),
        dropout=model_config["dropout"],
        activation=model_config["activation"],
    ).to(device)
    concept_names = list(model_config.get("concept_names", concept_keys_for_dim(int(model_config["concept_dim"]))))

    task_log_vars = nn.Parameter(torch.zeros(len(prepared.task_metadata), device=device))
    init_checkpoint_missing_keys: list[str] = []
    init_checkpoint_unexpected_keys: list[str] = []
    init_checkpoint_skipped_keys: list[str] = []
    loaded_checkpoint_calibration_params: dict[int, dict[str, object]] | None = None
    if args.init_checkpoint:
        checkpoint_path = Path(args.init_checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = PROJECT_ROOT / checkpoint_path
        _progress(f"Loading init checkpoint from {checkpoint_path}")
        init_checkpoint = torch.load(checkpoint_path, map_location=device)
        (
            init_checkpoint_missing_keys,
            init_checkpoint_unexpected_keys,
            init_checkpoint_skipped_keys,
        ) = _load_compatible_model_state(
            model,
            init_checkpoint["model_state_dict"],
        )
        if "task_log_vars" in init_checkpoint:
            saved_task_log_vars = init_checkpoint["task_log_vars"].to(device)
            if saved_task_log_vars.shape == task_log_vars.shape:
                task_log_vars.data.copy_(saved_task_log_vars)
        if "calibration_params" in init_checkpoint:
            loaded_checkpoint_calibration_params = init_checkpoint["calibration_params"]
        _progress(
            "Loaded init checkpoint with "
            + f"missing={len(init_checkpoint_missing_keys)} "
            + f"unexpected={len(init_checkpoint_unexpected_keys)} "
            + f"skipped={len(init_checkpoint_skipped_keys)}"
        )
    if selected_train_task_indices and (args.freeze_shared or args.freeze_nontarget_heads):
        _freeze_for_target_task_refinement(
            model,
            target_task_indices=selected_train_task_indices,
            freeze_shared=bool(args.freeze_shared),
            freeze_nontarget_heads=bool(args.freeze_nontarget_heads),
        )
    preserved_calibration_task_indices = (
        {int(task_index) for task_index in task_metadata.keys() if int(task_index) not in selected_train_task_indices}
        if loaded_checkpoint_calibration_params is not None and bool(args.freeze_nontarget_heads)
        else set()
    )
    parameter_count = _count_parameters(model) + int(task_log_vars.numel())
    _progress(
        f"Model ready: parameters={parameter_count} "
        + f"representation={'concept_bottleneck' if bool(model_config.get('use_concept_bottleneck', True)) else 'predictive_latent'}"
    )
    task_training_info = _build_task_training_info(
        prepared,
        class_balance_mode=str(train_config.get("class_balance_mode", "inverse")),
        class_balance_beta=float(train_config.get("class_balance_beta", 0.999) or 0.999),
        max_class_weight=(
            float(train_config["max_class_weight"])
            if train_config.get("max_class_weight") is not None
            else None
        ),
    )
    regression_loss_name = str(train_config.get("regression_loss", "huber") or "huber")
    for task_info in task_training_info.values():
        if str(task_info.get("label_type")) == "continuous":
            task_info["regression_loss"] = regression_loss_name
    if task_loss_weight_specs:
        for task_index, task_info in task_training_info.items():
            task_meta = task_metadata[task_index]
            task_weight = (
                task_loss_weight_specs.get(str(task_info.get("task_key")))
                or task_loss_weight_specs.get(str(task_info.get("task_name")))
                or task_loss_weight_specs.get(str(task_meta.task_key))
                or task_loss_weight_specs.get(str(task_meta.task_name))
            )
            if task_weight is not None:
                task_info["task_loss_weight"] = float(task_weight)
    if args.enable_deprest_ordinal_hybrid_loss:
        hybrid_weight = float(args.deprest_ordinal_hybrid_weight if args.deprest_ordinal_hybrid_weight is not None else 0.5)
        for task_index, task_info in task_training_info.items():
            if (
                str(task_info.get("dataset_id")) == "deprest_cat"
                and str(task_info.get("task_name")) in {"gad7_cat", "phq9_cat"}
                and str(task_info.get("label_type")) == "ordinal"
            ):
                task_info["ordinal_hybrid_weight"] = hybrid_weight
    ordinal_hybrid_task_keys = {
        str(task_key).strip()
        for task_key in (args.ordinal_hybrid_task_keys or train_config.get("ordinal_hybrid_task_keys") or [])
        if str(task_key).strip()
    }
    if ordinal_hybrid_task_keys:
        hybrid_weight = float(args.ordinal_hybrid_weight if args.ordinal_hybrid_weight is not None else train_config.get("ordinal_hybrid_weight", 0.35))
        for task_index, task_info in task_training_info.items():
            if str(task_info.get("task_key")) in ordinal_hybrid_task_keys and str(task_info.get("label_type")) == "ordinal":
                task_info["ordinal_hybrid_weight"] = hybrid_weight
    ordinal_labeldist_task_keys = {
        str(task_key).strip()
        for task_key in (args.ordinal_labeldist_task_keys or [])
        if str(task_key).strip()
    }
    if ordinal_labeldist_task_keys:
        labeldist_weight = float(args.ordinal_labeldist_weight if args.ordinal_labeldist_weight is not None else 0.35)
        labeldist_sigma = float(args.ordinal_labeldist_sigma if args.ordinal_labeldist_sigma is not None else 0.9)
        labeldist_edge_sigma = float(
            args.ordinal_labeldist_edge_sigma if args.ordinal_labeldist_edge_sigma is not None else 0.55
        )
        for task_index, task_info in task_training_info.items():
            if (
                str(task_info.get("task_key")) in ordinal_labeldist_task_keys
                and str(task_info.get("label_type")) == "ordinal"
            ):
                task_info["ordinal_labeldist_weight"] = labeldist_weight
                task_info["ordinal_labeldist_sigma"] = labeldist_sigma
                task_info["ordinal_labeldist_edge_sigma"] = labeldist_edge_sigma
    ordinal_edge_boost_task_keys = {
        str(task_key).strip()
        for task_key in (args.ordinal_edge_boost_task_keys or [])
        if str(task_key).strip()
    }
    if ordinal_edge_boost_task_keys:
        edge_boost_factor = float(args.ordinal_edge_boost_factor if args.ordinal_edge_boost_factor is not None else 2.0)
        for task_index, task_info in task_training_info.items():
            if (
                str(task_info.get("task_key")) in ordinal_edge_boost_task_keys
                and str(task_info.get("label_type")) == "ordinal"
                and "ordinal_class_weights" in task_info
            ):
                weights = np.asarray(task_info["ordinal_class_weights"], dtype=np.float32)
                if weights.size >= 2:
                    weights[0] *= edge_boost_factor
                    weights[-1] *= edge_boost_factor
                    weights = weights * (weights.size / max(float(np.sum(weights)), 1e-8))
                task_info["ordinal_class_weights"] = weights.tolist()
                task_info["ordinal_edge_boost_factor"] = edge_boost_factor
    ordinal_focal_task_keys = {
        str(task_key).strip()
        for task_key in (args.ordinal_focal_task_keys or [])
        if str(task_key).strip()
    }
    if ordinal_focal_task_keys:
        focal_gamma = float(args.ordinal_focal_gamma if args.ordinal_focal_gamma is not None else 1.5)
        for task_index, task_info in task_training_info.items():
            if str(task_info.get("task_key")) in ordinal_focal_task_keys and str(task_info.get("label_type")) == "ordinal":
                task_info["ordinal_focal_gamma"] = focal_gamma
    binary_focal_task_keys = {
        str(task_key).strip()
        for task_key in (args.binary_focal_task_keys or [])
        if str(task_key).strip()
    }
    if binary_focal_task_keys:
        focal_gamma = float(args.binary_focal_gamma if args.binary_focal_gamma is not None else 1.5)
        for task_index, task_info in task_training_info.items():
            if str(task_info.get("task_key")) in binary_focal_task_keys and str(task_info.get("label_type")) == "binary":
                task_info["binary_focal_gamma"] = focal_gamma
    multiclass_focal_task_keys = {
        str(task_key).strip()
        for task_key in (args.multiclass_focal_task_keys or [])
        if str(task_key).strip()
    }
    if multiclass_focal_task_keys:
        focal_gamma = float(args.multiclass_focal_gamma if args.multiclass_focal_gamma is not None else 1.5)
        for task_index, task_info in task_training_info.items():
            if str(task_info.get("task_key")) in multiclass_focal_task_keys and str(task_info.get("label_type")) == "multiclass":
                task_info["multiclass_focal_gamma"] = focal_gamma
    multiclass_label_smoothing_task_keys = {
        str(task_key).strip()
        for task_key in (args.multiclass_label_smoothing_task_keys or [])
        if str(task_key).strip()
    }
    if multiclass_label_smoothing_task_keys:
        label_smoothing = float(args.multiclass_label_smoothing if args.multiclass_label_smoothing is not None else 0.05)
        for task_index, task_info in task_training_info.items():
            if (
                str(task_info.get("task_key")) in multiclass_label_smoothing_task_keys
                and str(task_info.get("label_type")) == "multiclass"
            ):
                task_info["multiclass_label_smoothing"] = label_smoothing
    teacher_cache = None
    teacher_summary = pd.DataFrame()
    teacher_summary_path = None
    distill_task_subset = {
        str(task_key)
        for task_key in train_config["distillation"].get("task_subset", [])
        if str(task_key).strip()
    }
    if str(train_config["distillation"]["mode"]) != "none":
        _progress(f"Building teacher cache (mode={train_config['distillation']['mode']})")
        teacher_cache, teacher_summary = _build_teacher_cache(
            prepared,
            task_metadata,
            seed=args.seed,
            cache_tag=f"{args.run_name}__{train_config['distillation']['mode']}",
            enabled_task_keys=distill_task_subset or None,
        )
        teacher_summary_path = TEACHER_ROOT / f"{args.run_name}__{train_config['distillation']['mode']}" / "teacher_summary.csv"
        _progress(f"Teacher cache ready at {teacher_summary_path}")

    train_subset_indices = _task_subset_indices(prepared.train, selected_train_task_indices)
    train_dataset = Subset(prepared.train, train_subset_indices) if selected_train_task_indices else prepared.train
    train_sampling_frame = prepared.train.frame.iloc[train_subset_indices].reset_index(drop=True) if selected_train_task_indices else prepared.train.frame
    sample_weights = _dataset_balanced_sample_weights(
        train_sampling_frame,
        dataset_priority=train_config["dataset_priority"],
        task_sampling_power=float(train_config.get("task_sampling_power", 0.0) or 0.0),
        classification_sampling_power=float(train_config.get("classification_sampling_power", 0.0) or 0.0),
        max_weight_multiplier=float(train_config.get("max_sample_weight_multiplier", 25.0) or 25.0),
    )
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(train_dataset, batch_size=int(train_config["batch_size"]), sampler=sampler)
    valid_loader = DataLoader(prepared.valid, batch_size=int(train_config["batch_size"]), shuffle=False)
    test_loader = DataLoader(prepared.test, batch_size=int(train_config["batch_size"]), shuffle=False)
    train_batch_count = len(train_loader)
    _progress(
        "DataLoaders ready: "
        + f"train_batches={train_batch_count} "
        + f"valid_batches={len(valid_loader)} "
        + f"test_batches={len(test_loader)} "
        + f"batch_size={int(train_config['batch_size'])}"
    )

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if task_log_vars.requires_grad:
        trainable_parameters = trainable_parameters + [task_log_vars]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(train_config["learning_rate"]),
        weight_decay=float(train_config["weight_decay"]),
    )

    best_score = -1e9
    best_epoch = 0
    stale_epochs = 0
    best_checkpoint = ensure_dir(CHECKPOINT_DIR) / f"{args.run_name}.pt"
    history: list[dict[str, object]] = []

    if bool(args.eval_only):
        if args.init_checkpoint is None:
            raise ValueError("--eval-only requires --init-checkpoint")
        _progress("Running eval-only pass")
        regression_calibration_task_keys = {
            str(task_key)
            for task_key in (args.regression_calibration_task_keys or [])
            if str(task_key).strip()
        }
        valid_records = _build_split_records(model, valid_loader, task_metadata, device, "valid")
        if bool(args.use_checkpoint_calibration) and loaded_checkpoint_calibration_params is not None:
            calibration_params = loaded_checkpoint_calibration_params
        else:
            calibration_params = _fit_calibration(
                valid_records,
                task_metadata,
                task_training_info,
                ordinal_calibration_mode=str(train_config["ordinal_calibration_mode"]),
                fixed_ordinal_threshold_specs=fixed_ordinal_threshold_specs,
                fixed_binary_threshold_specs=fixed_binary_threshold_specs,
                fixed_multiclass_bias_specs=fixed_multiclass_bias_specs,
                regression_calibration_task_keys=regression_calibration_task_keys,
                regression_calibration_mode=str(train_config["regression_calibration_mode"]),
                paired_regression_bridge_task_indices=paired_regression_bridge_task_indices,
                multiclass_bias_task_indices=multiclass_bias_task_indices,
                ordinal_bias_task_indices=ordinal_bias_task_indices,
                binary_threshold_prevalence_tolerance=float(train_config.get("binary_threshold_prevalence_tolerance", 0.0) or 0.0),
            )
        calibration_params = _merge_preserved_calibration(
            calibration_params,
            loaded_checkpoint_calibration_params,
            preserved_calibration_task_indices,
        )
        calibration_params = _apply_fixed_calibration_overrides(
            calibration_params,
            task_metadata,
            fixed_ordinal_threshold_specs=fixed_ordinal_threshold_specs,
            fixed_binary_threshold_specs=fixed_binary_threshold_specs,
        )
        _, valid_metrics = _records_to_outputs(
            valid_records,
            task_metadata,
            calibration_params,
            task_training_info,
            "valid",
            concept_names=concept_names,
        )
        best_score = _selection_score(valid_metrics, task_metadata, train_config)
        _progress(f"Eval-only validation score={best_score:.6f}; saving checkpoint snapshot")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "task_log_vars": task_log_vars.detach().cpu(),
                "model_config": model_config,
                "train_config": train_config,
                "dataset_ids": args.datasets,
                "task_metadata": [asdict(meta) for meta in prepared.task_metadata],
                "feature_stats": prepared.feature_stats,
                "parameter_count": parameter_count,
                "calibration_params": calibration_params,
            },
            best_checkpoint,
        )
    else:
        for epoch in range(1, int(train_config["max_epochs"]) + 1):
            _progress(
                f"Epoch {epoch}/{int(train_config['max_epochs'])} started "
                + f"(best_valid={best_score:.6f}, stale_epochs={stale_epochs})"
            )
            model.train()
            epoch_losses = []
            epoch_task_losses: dict[str, list[float]] = defaultdict(list)
            for batch_index, raw_batch in enumerate(train_loader, start=1):
                if train_batch_count <= 4 or batch_index == 1 or batch_index == train_batch_count or batch_index % 50 == 0:
                    _progress(f"Epoch {epoch}: fetched train batch {batch_index}/{train_batch_count}")
                batch = _move_batch(raw_batch, device)
                optimizer.zero_grad(set_to_none=True)
                total_loss, regularization_loss, weighted_task_losses, detached_task_losses, aux = _compute_train_losses(
                    model,
                    batch,
                    task_metadata,
                    task_training_info,
                    task_log_vars,
                    train_config,
                    epoch,
                    teacher_cache=teacher_cache,
                )
                if bool(train_config.get("use_gradnorm", False)):
                    _apply_gradnorm_balancing(model, total_loss, weighted_task_losses, regularization_loss)
                elif bool(train_config["use_pcgrad"]):
                    _apply_pcgrad(model, total_loss, weighted_task_losses, regularization_loss)
                else:
                    total_loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=float(train_config["grad_clip"]))
                optimizer.step()
                if train_batch_count <= 4 or batch_index == 1 or batch_index == train_batch_count or batch_index % 50 == 0:
                    _progress(f"Epoch {epoch}: optimized train batch {batch_index}/{train_batch_count}")
                epoch_losses.append(float(total_loss.detach().cpu()))
                for task_key, value in detached_task_losses.items():
                    epoch_task_losses[task_key].append(float(value))

            _progress(f"Epoch {epoch}: building validation records")
            valid_records = _build_split_records(model, valid_loader, task_metadata, device, "valid")
            regression_calibration_task_keys = {
                str(task_key)
                for task_key in (args.regression_calibration_task_keys or [])
                if str(task_key).strip()
            }
            calibration_params = _fit_calibration(
                valid_records,
                task_metadata,
                task_training_info,
                ordinal_calibration_mode=str(train_config["ordinal_calibration_mode"]),
                fixed_ordinal_threshold_specs=fixed_ordinal_threshold_specs,
                fixed_binary_threshold_specs=fixed_binary_threshold_specs,
                fixed_multiclass_bias_specs=fixed_multiclass_bias_specs,
                regression_calibration_task_keys=regression_calibration_task_keys,
                regression_calibration_mode=str(train_config["regression_calibration_mode"]),
                paired_regression_bridge_task_indices=paired_regression_bridge_task_indices,
                multiclass_bias_task_indices=multiclass_bias_task_indices,
                ordinal_bias_task_indices=ordinal_bias_task_indices,
                binary_threshold_prevalence_tolerance=float(train_config.get("binary_threshold_prevalence_tolerance", 0.0) or 0.0),
            )
            calibration_params = _merge_preserved_calibration(
                calibration_params,
                loaded_checkpoint_calibration_params,
                preserved_calibration_task_indices,
            )
            calibration_params = _apply_fixed_calibration_overrides(
                calibration_params,
                task_metadata,
                fixed_ordinal_threshold_specs=fixed_ordinal_threshold_specs,
                fixed_binary_threshold_specs=fixed_binary_threshold_specs,
            )
            _, valid_metrics = _records_to_outputs(
                valid_records,
                task_metadata,
                calibration_params,
                task_training_info,
                "valid",
                concept_names=concept_names,
            )
            _progress(f"Epoch {epoch}: validation records converted to metrics")
            valid_score = _selection_score(valid_metrics, task_metadata, train_config)
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": float(np.mean(epoch_losses)) if epoch_losses else None,
                    "valid_score": valid_score,
                    "task_losses": {task_key: float(np.mean(values)) for task_key, values in epoch_task_losses.items()},
                }
            )
            mean_epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
            if valid_score > best_score:
                best_score = valid_score
                best_epoch = epoch
                stale_epochs = 0
                _progress(
                    f"Epoch {epoch} improved validation: "
                    + f"train_loss={mean_epoch_loss:.6f} valid_score={valid_score:.6f}"
                )
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "task_log_vars": task_log_vars.detach().cpu(),
                        "model_config": model_config,
                        "train_config": train_config,
                        "dataset_ids": args.datasets,
                        "task_metadata": [asdict(meta) for meta in prepared.task_metadata],
                        "feature_stats": prepared.feature_stats,
                        "parameter_count": parameter_count,
                        "calibration_params": calibration_params,
                    },
                    best_checkpoint,
                )
            else:
                stale_epochs += 1
                _progress(
                    f"Epoch {epoch} completed without improvement: "
                    + f"train_loss={mean_epoch_loss:.6f} valid_score={valid_score:.6f} "
                    + f"best_valid={best_score:.6f} stale_epochs={stale_epochs}"
                )

            if epoch >= int(train_config["min_epochs"]) and stale_epochs >= int(train_config["patience"]):
                _progress(
                    f"Early stopping triggered at epoch {epoch} "
                    + f"(best_epoch={best_epoch}, best_valid={best_score:.6f})"
                )
                break

    _progress(f"Reloading best checkpoint from {best_checkpoint}")
    checkpoint = torch.load(best_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    task_log_vars.data.copy_(checkpoint["task_log_vars"].to(device))

    evaluation_label = "validation-only evaluation" if bool(args.skip_test_eval) else "validation/test evaluation"
    _progress(f"Running final {evaluation_label}")
    valid_records = _build_split_records(model, valid_loader, task_metadata, device, "valid")
    regression_calibration_task_keys = {
        str(task_key)
        for task_key in (args.regression_calibration_task_keys or [])
        if str(task_key).strip()
    }
    if bool(args.eval_only) and bool(args.use_checkpoint_calibration) and "calibration_params" in checkpoint:
        calibration_params = checkpoint["calibration_params"]
    else:
        calibration_params = _fit_calibration(
            valid_records,
            task_metadata,
            task_training_info,
            ordinal_calibration_mode=str(train_config["ordinal_calibration_mode"]),
            fixed_ordinal_threshold_specs=fixed_ordinal_threshold_specs,
            fixed_binary_threshold_specs=fixed_binary_threshold_specs,
            fixed_multiclass_bias_specs=fixed_multiclass_bias_specs,
            regression_calibration_task_keys=regression_calibration_task_keys,
            regression_calibration_mode=str(train_config["regression_calibration_mode"]),
            paired_regression_bridge_task_indices=paired_regression_bridge_task_indices,
            multiclass_bias_task_indices=multiclass_bias_task_indices,
            ordinal_bias_task_indices=ordinal_bias_task_indices,
            binary_threshold_prevalence_tolerance=float(train_config.get("binary_threshold_prevalence_tolerance", 0.0) or 0.0),
        )
    calibration_params = _merge_preserved_calibration(
        calibration_params,
        loaded_checkpoint_calibration_params,
        preserved_calibration_task_indices,
    )
    calibration_params = _apply_fixed_calibration_overrides(
        calibration_params,
        task_metadata,
        fixed_ordinal_threshold_specs=fixed_ordinal_threshold_specs,
        fixed_binary_threshold_specs=fixed_binary_threshold_specs,
    )
    valid_predictions, valid_metrics = _records_to_outputs(
        valid_records,
        task_metadata,
        calibration_params,
        task_training_info,
        "valid",
        concept_names=concept_names,
    )
    if bool(args.skip_test_eval):
        _progress("Final validation calibration ready; skipping test evaluation by request")
        test_predictions = pd.DataFrame()
        test_metrics = pd.DataFrame()
    else:
        _progress("Final validation calibration ready; building test records")
        test_records = _build_split_records(model, test_loader, task_metadata, device, "test")
        test_predictions, test_metrics = _records_to_outputs(
            test_records,
            task_metadata,
            calibration_params,
            task_training_info,
            "test",
            concept_names=concept_names,
        )

    ensure_dir(PREDICTION_DIR)
    ensure_dir(CONCEPT_DIR)
    ensure_dir(LOG_DIR)
    ensure_dir(TABLE_DIR)

    valid_predictions.to_csv(PREDICTION_DIR / f"{args.run_name}__valid.csv", index=False)
    if not bool(args.skip_test_eval):
        test_predictions.to_csv(PREDICTION_DIR / f"{args.run_name}__test.csv", index=False)
    combined_frames = [valid_predictions]
    if not test_predictions.empty:
        combined_frames.append(test_predictions)
    combined_predictions = pd.concat(combined_frames, ignore_index=True)
    export_concept_artifacts = bool(model_config.get("use_concept_bottleneck", True))
    if export_concept_artifacts:
        combined_predictions.to_csv(CONCEPT_DIR / f"{args.run_name}__concepts.csv", index=False)

    metric_frames = [valid_metrics]
    if not test_metrics.empty:
        metric_frames.append(test_metrics)
    metric_table = pd.concat(metric_frames, ignore_index=True)
    metric_table.to_csv(PREDICTION_DIR / f"{args.run_name}__metrics.csv", index=False)
    _append_result_rows(
        _metric_frame_to_records(
            metric_table,
            run_name=args.run_name,
            datasets=args.datasets,
            model_config=model_config,
            train_config=train_config,
            parameter_count=parameter_count,
            model_name=str(model_config["model_name"]),
        )
    )
    calibration_frame = _write_calibration_summary(calibration_params, task_metadata, args.run_name)
    if export_concept_artifacts and not test_predictions.empty:
        concept_summary = _write_concept_summary(test_predictions, args.run_name)
    else:
        concept_source = test_predictions if not test_predictions.empty else valid_predictions
        concept_summary = pd.DataFrame(
            [
                {
                    "run_name": args.run_name,
                    "dataset_id": dataset_id,
                    "task_name": task_name,
                    "top_concept": "not_applicable",
                    "top_concept_mean_abs": 0.0,
                }
                for dataset_id, task_name in sorted(set(zip(concept_source["dataset_id"], concept_source["task_name"])))
            ]
        )
    if not test_metrics.empty:
        vs_baseline = _write_vs_baseline_summary(test_metrics, args.run_name, str(model_config["model_name"]))
    else:
        vs_baseline = pd.DataFrame()

    lagging = vs_baseline.loc[vs_baseline["winner"] == "baseline"].copy() if not vs_baseline.empty else pd.DataFrame()
    failure_notes = []
    for row in lagging.to_dict(orient="records"):
        failure_notes.append(
            f"{row['dataset_id']}::{row['task_name']} delta={row['delta_model_minus_baseline']:.4f} vs {row['baseline_model_name']}"
        )

    summary = {
        "run_name": args.run_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_name": str(model_config["model_name"]),
        "model_config_path": str(model_config_path.relative_to(PROJECT_ROOT)),
        "train_config_path": str(train_config_path.relative_to(PROJECT_ROOT)),
        "datasets": args.datasets,
        "seed": args.seed,
        "parameter_count": parameter_count,
        "representation_kind": "concept_bottleneck" if bool(model_config.get("use_concept_bottleneck", True)) else "predictive_latent",
        "distillation_mode": train_config["distillation"]["mode"],
        "distillation_task_subset": sorted(distill_task_subset),
        "disable_native_branch": bool(args.disable_native_branch),
        "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint is not None else None,
        "eval_only": bool(args.eval_only),
        "skip_test_eval": bool(args.skip_test_eval),
        "use_checkpoint_calibration": bool(args.use_checkpoint_calibration),
        "use_pcgrad": bool(train_config["use_pcgrad"]),
        "use_gradnorm": bool(train_config.get("use_gradnorm", False)),
        "use_uncertainty_weighting": bool(train_config["use_uncertainty_weighting"]),
        "task_sampling_power": float(train_config.get("task_sampling_power", 0.0) or 0.0),
        "classification_sampling_power": float(train_config.get("classification_sampling_power", 0.0) or 0.0),
        "train_task_keys": list(args.train_task_keys or []),
        "freeze_shared": bool(args.freeze_shared),
        "freeze_nontarget_heads": bool(args.freeze_nontarget_heads),
        "base_communication_only": bool(args.base_communication_only),
        "concept_dim": int(model_config["concept_dim"]),
        "use_concept_bottleneck": bool(model_config.get("use_concept_bottleneck", True)),
        "concept_names": concept_names,
        "temporal_frontend_mode": str(model_config.get("temporal_frontend_mode", "none")),
        "enable_missingness_tokens": bool(model_config.get("enable_missingness_tokens", False)),
        "enable_missingness_embedding": bool(model_config.get("enable_missingness_embedding", False)),
        "missingness_embedding_norm": bool(model_config.get("missingness_embedding_norm", False)),
        "modality_dropout": float(model_config.get("modality_dropout", 0.0) or 0.0),
        "enable_tree_gated_head": bool(model_config.get("enable_tree_gated_head", False)),
        "disable_task_conditioning": bool(model_config.get("disable_task_conditioning", False)),
        "conditioning_mode": str(model_config.get("conditioning_mode", "dataset_task")),
        "film_conditioning_mode": str(model_config.get("film_conditioning_mode", "both")),
        "token_dim": int(model_config["token_dim"]),
        "latent_dim": int(model_config["latent_dim"]),
        "output_refine_dim": int(model_config["output_refine_dim"]),
        "regression_loss": str(train_config.get("regression_loss", "huber")),
        "tree_head_task_names": list(args.tree_head_task_names or []),
        "tree_head_depth": int(model_config.get("tree_head_depth", 2)),
        "disable_psyche_hierarchical": bool(args.disable_psyche_hierarchical),
        "enable_psyche_two_stage": bool(args.enable_psyche_two_stage),
        "enable_psyche_binary_correction": bool(args.enable_psyche_binary_correction),
        "enable_psyche_native_adapter": bool(args.enable_psyche_native_adapter),
        "enable_psyche_taskwise_native_adapter": bool(args.enable_psyche_taskwise_native_adapter),
        "psyche_native_task_names": list(args.psyche_native_task_names or []),
        "disable_deprest_adapter": bool(args.disable_deprest_adapter),
        "enable_deprest_category_adapter": bool(args.enable_deprest_category_adapter),
        "enable_deprest_coverage_routing": bool(args.enable_deprest_coverage_routing),
        "deprest_coverage_task_names": list(args.deprest_coverage_task_names or []),
        "enable_deprest_concept_compatibility": bool(args.enable_deprest_concept_compatibility),
        "deprest_concept_compat_task_names": list(args.deprest_concept_compat_task_names or []),
        "enable_deprest_concept_contrast": bool(args.enable_deprest_concept_contrast),
        "deprest_concept_contrast_task_names": list(args.deprest_concept_contrast_task_names or []),
        "enable_deprest_gad7_reg_bridge": bool(args.enable_deprest_gad7_reg_bridge),
        "enable_deprest_severity_head": bool(args.enable_deprest_severity_head),
        "deprest_severity_task_names": list(args.deprest_severity_task_names or []),
        "enable_deprest_ordinal_hybrid_loss": bool(args.enable_deprest_ordinal_hybrid_loss),
        "deprest_ordinal_hybrid_weight": float(args.deprest_ordinal_hybrid_weight) if args.deprest_ordinal_hybrid_weight is not None else None,
        "ordinal_hybrid_task_keys": list(args.ordinal_hybrid_task_keys or train_config.get("ordinal_hybrid_task_keys") or []),
        "ordinal_hybrid_weight": float(args.ordinal_hybrid_weight if args.ordinal_hybrid_weight is not None else train_config.get("ordinal_hybrid_weight", 0.0) or 0.0),
        "enable_ordinal_aux_class_head": bool(args.enable_ordinal_aux_class_head or train_config.get("enable_ordinal_aux_class_head", False)),
        "ordinal_aux_class_task_keys": list(args.ordinal_aux_class_task_keys or train_config.get("ordinal_aux_class_task_keys") or []),
        "ordinal_aux_class_weight": float(train_config.get("ordinal_aux_class_weight", 0.0) or 0.0),
        "ordinal_labeldist_task_keys": list(args.ordinal_labeldist_task_keys or []),
        "ordinal_labeldist_weight": float(args.ordinal_labeldist_weight) if args.ordinal_labeldist_weight is not None else None,
        "ordinal_labeldist_sigma": float(args.ordinal_labeldist_sigma) if args.ordinal_labeldist_sigma is not None else None,
        "ordinal_labeldist_edge_sigma": float(args.ordinal_labeldist_edge_sigma) if args.ordinal_labeldist_edge_sigma is not None else None,
        "ordinal_edge_boost_task_keys": list(args.ordinal_edge_boost_task_keys or []),
        "ordinal_edge_boost_factor": float(args.ordinal_edge_boost_factor) if args.ordinal_edge_boost_factor is not None else None,
        "binary_focal_task_keys": list(args.binary_focal_task_keys or []),
        "binary_focal_gamma": float(args.binary_focal_gamma) if args.binary_focal_gamma is not None else None,
        "ordinal_focal_task_keys": list(args.ordinal_focal_task_keys or []),
        "ordinal_focal_gamma": float(args.ordinal_focal_gamma) if args.ordinal_focal_gamma is not None else None,
        "multiclass_focal_task_keys": list(args.multiclass_focal_task_keys or []),
        "multiclass_focal_gamma": float(args.multiclass_focal_gamma) if args.multiclass_focal_gamma is not None else None,
        "multiclass_label_smoothing_task_keys": list(args.multiclass_label_smoothing_task_keys or []),
        "multiclass_label_smoothing": float(args.multiclass_label_smoothing) if args.multiclass_label_smoothing is not None else None,
        "task_loss_weight_specs": task_loss_weight_specs,
        "enable_deprest_gad7_pair_aux": bool(args.enable_deprest_gad7_pair_aux),
        "deprest_gad7_pair_weight": float(train_config.get("deprest_gad7_pair_weight", 0.0)),
        "enable_deprest_edge_ovr": bool(args.enable_deprest_edge_ovr),
        "deprest_edge_ovr_task_names": list(args.deprest_edge_ovr_task_names or []),
        "deprest_edge_ovr_weight": float(train_config.get("deprest_edge_ovr_weight", 0.0)),
        "enable_concept_residual": bool(args.enable_concept_residual),
        "shared_token_refiner_type": str(model_config.get("shared_token_refiner_type", "none")),
        "shared_token_refiner_steps": int(model_config.get("shared_token_refiner_steps", 0)),
        "shared_token_refiner_nograd_steps": int(model_config.get("shared_token_refiner_nograd_steps", 0)),
        "task_local_trm_mode": str(model_config.get("task_local_trm_mode", "none")),
        "task_local_trm_reasoning_dim": int(model_config.get("task_local_trm_reasoning_dim", model_config["token_dim"])),
        "task_local_trm_h_cycles": int(model_config.get("task_local_trm_h_cycles", 0)),
        "task_local_trm_l_cycles": int(model_config.get("task_local_trm_l_cycles", 0)),
        "task_local_trm_task_names": list(args.task_local_trm_task_names or []),
        "enable_deprest_construct_bridge": bool(args.enable_deprest_construct_bridge),
        "deprest_construct_task_names": list(args.deprest_construct_task_names or []),
        "deprest_construct_weight": float(train_config.get("deprest_construct_weight", 0.0)),
        "enable_psyche_delta_bridge": bool(args.enable_psyche_delta_bridge),
        "psyche_delta_weight": float(train_config.get("psyche_delta_weight", 0.0)),
        "enable_deprest_edge_specialist": bool(args.enable_deprest_edge_specialist),
        "deprest_edge_task_names": list(args.deprest_edge_task_names or []),
        "multiclass_task_keys": list(args.multiclass_task_keys or []),
        "task_label_overrides": task_label_overrides,
        "fixed_ordinal_threshold_specs": fixed_ordinal_threshold_specs,
        "fixed_binary_threshold_specs": fixed_binary_threshold_specs,
        "fixed_multiclass_bias_specs": fixed_multiclass_bias_specs,
        "regression_calibration_task_keys": list(args.regression_calibration_task_keys or []),
        "regression_calibration_mode": str(train_config["regression_calibration_mode"]),
        "paired_regression_bridge_specs": paired_regression_bridge_specs,
        "multiclass_bias_task_keys": list(args.multiclass_bias_task_keys or []),
        "ordinal_bias_task_keys": list(args.ordinal_bias_task_keys or train_config.get("ordinal_bias_task_keys") or []),
        "auto_ordinal_bias_calibration": bool(train_config.get("auto_ordinal_bias_calibration", False)),
        "binary_threshold_prevalence_tolerance": float(train_config.get("binary_threshold_prevalence_tolerance", 0.0) or 0.0),
        "class_balance_mode": str(train_config.get("class_balance_mode", "inverse")),
        "class_balance_beta": float(train_config.get("class_balance_beta", 0.999) or 0.999),
        "max_class_weight": train_config.get("max_class_weight"),
        "init_checkpoint_missing_keys": init_checkpoint_missing_keys,
        "init_checkpoint_unexpected_keys": init_checkpoint_unexpected_keys,
        "init_checkpoint_skipped_keys": init_checkpoint_skipped_keys,
        "best_epoch": best_epoch,
        "best_valid_score": best_score,
        "final_valid_score": _selection_score(valid_metrics, task_metadata, train_config),
        "final_test_score": (
            _selection_score(test_metrics, task_metadata, train_config)
            if not test_metrics.empty
            else None
        ),
        "checkpoint_path": str(best_checkpoint),
        "metric_path": str(PREDICTION_DIR / f"{args.run_name}__metrics.csv"),
        "calibration_path": str((TABLE_DIR / f"{args.run_name}__calibration_summary.csv").relative_to(PROJECT_ROOT)),
        "vs_baseline_path": (
            str((TABLE_DIR / f"{args.run_name}__vs_baseline.csv").relative_to(PROJECT_ROOT))
            if not test_metrics.empty
            else None
        ),
        "concept_summary_path": (
            str((CONCEPT_DIR / f"{args.run_name}__concept_summary.csv").relative_to(PROJECT_ROOT))
            if export_concept_artifacts
            else None
        ),
        "concept_artifacts_exported": export_concept_artifacts,
        "teacher_summary_path": str(teacher_summary_path.relative_to(PROJECT_ROOT)) if teacher_summary_path is not None else None,
        "history": history,
        "failure_notes": failure_notes,
    }
    write_json(LOG_DIR / f"{args.run_name}__summary.json", summary)
    write_json(
        LOG_DIR / f"{args.run_name}__config.json",
        {
            "model_config": model_config,
            "train_config": train_config,
            "seed": args.seed,
            "model_name": str(model_config["model_name"]),
            "model_config_path": str(model_config_path.relative_to(PROJECT_ROOT)),
            "train_config_path": str(train_config_path.relative_to(PROJECT_ROOT)),
            "datasets": args.datasets,
            "parameter_count": parameter_count,
            "representation_kind": "concept_bottleneck" if bool(model_config.get("use_concept_bottleneck", True)) else "predictive_latent",
            "disable_native_branch": bool(args.disable_native_branch),
            "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint is not None else None,
            "eval_only": bool(args.eval_only),
            "use_checkpoint_calibration": bool(args.use_checkpoint_calibration),
            "use_pcgrad": bool(train_config["use_pcgrad"]),
            "use_gradnorm": bool(train_config.get("use_gradnorm", False)),
            "use_uncertainty_weighting": bool(train_config["use_uncertainty_weighting"]),
            "task_sampling_power": float(train_config.get("task_sampling_power", 0.0) or 0.0),
            "classification_sampling_power": float(train_config.get("classification_sampling_power", 0.0) or 0.0),
            "skip_test_eval": bool(args.skip_test_eval),
            "train_task_keys": list(args.train_task_keys or []),
            "freeze_shared": bool(args.freeze_shared),
            "freeze_nontarget_heads": bool(args.freeze_nontarget_heads),
            "base_communication_only": bool(args.base_communication_only),
            "concept_dim": int(model_config["concept_dim"]),
            "use_concept_bottleneck": bool(model_config.get("use_concept_bottleneck", True)),
            "concept_names": concept_names,
            "temporal_frontend_mode": str(model_config.get("temporal_frontend_mode", "none")),
            "enable_missingness_tokens": bool(model_config.get("enable_missingness_tokens", False)),
            "enable_missingness_embedding": bool(model_config.get("enable_missingness_embedding", False)),
            "missingness_embedding_norm": bool(model_config.get("missingness_embedding_norm", False)),
            "modality_dropout": float(model_config.get("modality_dropout", 0.0) or 0.0),
            "enable_tree_gated_head": bool(model_config.get("enable_tree_gated_head", False)),
            "disable_task_conditioning": bool(model_config.get("disable_task_conditioning", False)),
            "conditioning_mode": str(model_config.get("conditioning_mode", "dataset_task")),
            "film_conditioning_mode": str(model_config.get("film_conditioning_mode", "both")),
            "token_dim": int(model_config["token_dim"]),
            "latent_dim": int(model_config["latent_dim"]),
            "output_refine_dim": int(model_config["output_refine_dim"]),
            "regression_loss": str(train_config.get("regression_loss", "huber")),
            "tree_head_task_names": list(args.tree_head_task_names or []),
            "tree_head_depth": int(model_config.get("tree_head_depth", 2)),
            "disable_psyche_hierarchical": bool(args.disable_psyche_hierarchical),
            "enable_psyche_two_stage": bool(args.enable_psyche_two_stage),
            "enable_psyche_binary_correction": bool(args.enable_psyche_binary_correction),
            "enable_psyche_native_adapter": bool(args.enable_psyche_native_adapter),
            "enable_psyche_taskwise_native_adapter": bool(args.enable_psyche_taskwise_native_adapter),
            "psyche_native_task_names": list(args.psyche_native_task_names or []),
            "disable_deprest_adapter": bool(args.disable_deprest_adapter),
            "enable_deprest_category_adapter": bool(args.enable_deprest_category_adapter),
            "enable_deprest_coverage_routing": bool(args.enable_deprest_coverage_routing),
            "deprest_coverage_task_names": list(args.deprest_coverage_task_names or []),
            "enable_deprest_concept_compatibility": bool(args.enable_deprest_concept_compatibility),
            "deprest_concept_compat_task_names": list(args.deprest_concept_compat_task_names or []),
            "enable_deprest_concept_contrast": bool(args.enable_deprest_concept_contrast),
            "deprest_concept_contrast_task_names": list(args.deprest_concept_contrast_task_names or []),
            "enable_deprest_gad7_reg_bridge": bool(args.enable_deprest_gad7_reg_bridge),
            "enable_deprest_severity_head": bool(args.enable_deprest_severity_head),
            "deprest_severity_task_names": list(args.deprest_severity_task_names or []),
            "enable_deprest_ordinal_hybrid_loss": bool(args.enable_deprest_ordinal_hybrid_loss),
            "deprest_ordinal_hybrid_weight": float(args.deprest_ordinal_hybrid_weight) if args.deprest_ordinal_hybrid_weight is not None else None,
            "ordinal_hybrid_task_keys": list(args.ordinal_hybrid_task_keys or train_config.get("ordinal_hybrid_task_keys") or []),
            "ordinal_hybrid_weight": float(args.ordinal_hybrid_weight if args.ordinal_hybrid_weight is not None else train_config.get("ordinal_hybrid_weight", 0.0) or 0.0),
            "enable_ordinal_aux_class_head": bool(args.enable_ordinal_aux_class_head or train_config.get("enable_ordinal_aux_class_head", False)),
            "ordinal_aux_class_task_keys": list(args.ordinal_aux_class_task_keys or train_config.get("ordinal_aux_class_task_keys") or []),
            "ordinal_aux_class_weight": float(train_config.get("ordinal_aux_class_weight", 0.0) or 0.0),
            "ordinal_labeldist_task_keys": list(args.ordinal_labeldist_task_keys or []),
            "ordinal_labeldist_weight": float(args.ordinal_labeldist_weight) if args.ordinal_labeldist_weight is not None else None,
            "ordinal_labeldist_sigma": float(args.ordinal_labeldist_sigma) if args.ordinal_labeldist_sigma is not None else None,
            "ordinal_labeldist_edge_sigma": float(args.ordinal_labeldist_edge_sigma) if args.ordinal_labeldist_edge_sigma is not None else None,
            "ordinal_edge_boost_task_keys": list(args.ordinal_edge_boost_task_keys or []),
            "ordinal_edge_boost_factor": float(args.ordinal_edge_boost_factor) if args.ordinal_edge_boost_factor is not None else None,
            "binary_focal_task_keys": list(args.binary_focal_task_keys or []),
            "binary_focal_gamma": float(args.binary_focal_gamma) if args.binary_focal_gamma is not None else None,
            "ordinal_focal_task_keys": list(args.ordinal_focal_task_keys or []),
            "ordinal_focal_gamma": float(args.ordinal_focal_gamma) if args.ordinal_focal_gamma is not None else None,
            "multiclass_focal_task_keys": list(args.multiclass_focal_task_keys or []),
            "multiclass_focal_gamma": float(args.multiclass_focal_gamma) if args.multiclass_focal_gamma is not None else None,
            "multiclass_label_smoothing_task_keys": list(args.multiclass_label_smoothing_task_keys or []),
            "multiclass_label_smoothing": float(args.multiclass_label_smoothing) if args.multiclass_label_smoothing is not None else None,
            "task_loss_weight_specs": task_loss_weight_specs,
            "enable_deprest_gad7_pair_aux": bool(args.enable_deprest_gad7_pair_aux),
            "deprest_gad7_pair_weight": float(train_config.get("deprest_gad7_pair_weight", 0.0)),
            "enable_deprest_edge_ovr": bool(args.enable_deprest_edge_ovr),
            "deprest_edge_ovr_task_names": list(args.deprest_edge_ovr_task_names or []),
            "deprest_edge_ovr_weight": float(train_config.get("deprest_edge_ovr_weight", 0.0)),
            "enable_concept_residual": bool(args.enable_concept_residual),
            "shared_token_refiner_type": str(model_config.get("shared_token_refiner_type", "none")),
            "shared_token_refiner_steps": int(model_config.get("shared_token_refiner_steps", 0)),
            "shared_token_refiner_nograd_steps": int(model_config.get("shared_token_refiner_nograd_steps", 0)),
            "task_local_trm_mode": str(model_config.get("task_local_trm_mode", "none")),
            "task_local_trm_reasoning_dim": int(model_config.get("task_local_trm_reasoning_dim", model_config["token_dim"])),
            "task_local_trm_h_cycles": int(model_config.get("task_local_trm_h_cycles", 0)),
            "task_local_trm_l_cycles": int(model_config.get("task_local_trm_l_cycles", 0)),
            "task_local_trm_task_names": list(args.task_local_trm_task_names or []),
            "enable_deprest_construct_bridge": bool(args.enable_deprest_construct_bridge),
            "deprest_construct_task_names": list(args.deprest_construct_task_names or []),
            "deprest_construct_weight": float(train_config.get("deprest_construct_weight", 0.0)),
            "enable_psyche_delta_bridge": bool(args.enable_psyche_delta_bridge),
            "psyche_delta_weight": float(train_config.get("psyche_delta_weight", 0.0)),
            "enable_deprest_edge_specialist": bool(args.enable_deprest_edge_specialist),
            "deprest_edge_task_names": list(args.deprest_edge_task_names or []),
            "multiclass_task_keys": list(args.multiclass_task_keys or []),
            "task_label_overrides": task_label_overrides,
            "fixed_ordinal_threshold_specs": fixed_ordinal_threshold_specs,
            "fixed_binary_threshold_specs": fixed_binary_threshold_specs,
            "fixed_multiclass_bias_specs": fixed_multiclass_bias_specs,
            "regression_calibration_task_keys": list(args.regression_calibration_task_keys or []),
            "regression_calibration_mode": str(train_config["regression_calibration_mode"]),
            "paired_regression_bridge_specs": paired_regression_bridge_specs,
            "multiclass_bias_task_keys": list(args.multiclass_bias_task_keys or []),
            "ordinal_bias_task_keys": list(args.ordinal_bias_task_keys or train_config.get("ordinal_bias_task_keys") or []),
            "auto_ordinal_bias_calibration": bool(train_config.get("auto_ordinal_bias_calibration", False)),
            "binary_threshold_prevalence_tolerance": float(train_config.get("binary_threshold_prevalence_tolerance", 0.0) or 0.0),
            "class_balance_mode": str(train_config.get("class_balance_mode", "inverse")),
            "class_balance_beta": float(train_config.get("class_balance_beta", 0.999) or 0.999),
            "max_class_weight": train_config.get("max_class_weight"),
            "init_checkpoint_missing_keys": init_checkpoint_missing_keys,
            "init_checkpoint_unexpected_keys": init_checkpoint_unexpected_keys,
            "init_checkpoint_skipped_keys": init_checkpoint_skipped_keys,
            "concept_artifacts_exported": export_concept_artifacts,
        },
    )
    final_test_score = summary["final_test_score"]
    final_test_text = f"{final_test_score:.6f}" if final_test_score is not None else "not_evaluated"
    _progress(
        f"Completed run {args.run_name}: "
        + f"best_epoch={best_epoch} best_valid={best_score:.6f} "
        + f"final_test={final_test_text}"
    )

    experiment_id = f"stage_13_mctrcm_v2__{args.run_name}__{datetime.now().strftime('%Y%m%dT%H%M%S')}"
    _append_registry_row(
        {
            "experiment_id": experiment_id,
            "stage": "stage_13_model_upgrade",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "model_name": str(model_config["model_name"]),
            "training_corpora": "|".join(args.datasets),
            "target_dataset": "multi_corpus_bundle",
            "task_name": "multi_task_bundle",
            "split_config": "participant_level_dataset_specific_splits",
            "model_config": str(model_config_path.relative_to(PROJECT_ROOT)),
            "train_config": str(train_config_path.relative_to(PROJECT_ROOT)),
            "status": "completed",
            "notes": (
                f"model_name={str(model_config['model_name'])}; "
                f"representation_kind={'concept_bottleneck' if bool(model_config.get('use_concept_bottleneck', True)) else 'predictive_latent'}; "
                f"run_name={args.run_name}; parameter_count={parameter_count}; "
                f"best_valid_score={best_score:.6f}; best_epoch={best_epoch}; "
                f"eval_only={bool(args.eval_only)}; "
                f"use_checkpoint_calibration={bool(args.use_checkpoint_calibration)}; "
                f"use_pcgrad={train_config['use_pcgrad']}; "
                f"use_gradnorm={bool(train_config.get('use_gradnorm', False))}; "
                f"use_uncertainty_weighting={train_config['use_uncertainty_weighting']}; "
                f"task_sampling_power={float(train_config.get('task_sampling_power', 0.0) or 0.0):g}; "
                f"classification_sampling_power={float(train_config.get('classification_sampling_power', 0.0) or 0.0):g}; "
                f"distillation_mode={train_config['distillation']['mode']}; "
                f"disable_native_branch={bool(args.disable_native_branch)}; "
                f"base_communication_only={bool(args.base_communication_only)}; "
                f"disable_psyche_hierarchical={bool(args.disable_psyche_hierarchical)}; "
                f"enable_psyche_binary_correction={bool(args.enable_psyche_binary_correction)}; "
                f"disable_deprest_adapter={bool(args.disable_deprest_adapter)}; "
                f"enable_deprest_coverage_routing={bool(args.enable_deprest_coverage_routing)}; "
                f"deprest_coverage_task_names={'|'.join(args.deprest_coverage_task_names) if args.deprest_coverage_task_names else 'none'}; "
                f"enable_deprest_concept_compatibility={bool(args.enable_deprest_concept_compatibility)}; "
                f"deprest_concept_compat_task_names={'|'.join(args.deprest_concept_compat_task_names) if args.deprest_concept_compat_task_names else 'none'}; "
                f"enable_deprest_concept_contrast={bool(args.enable_deprest_concept_contrast)}; "
                f"deprest_concept_contrast_task_names={'|'.join(args.deprest_concept_contrast_task_names) if args.deprest_concept_contrast_task_names else 'none'}; "
                f"enable_deprest_gad7_reg_bridge={bool(args.enable_deprest_gad7_reg_bridge)}; "
                f"ordinal_hybrid_task_keys={'|'.join(args.ordinal_hybrid_task_keys or train_config.get('ordinal_hybrid_task_keys') or []) or 'none'}; "
                f"ordinal_hybrid_weight={float(args.ordinal_hybrid_weight if args.ordinal_hybrid_weight is not None else train_config.get('ordinal_hybrid_weight', 0.0) or 0.0)}; "
                f"enable_ordinal_aux_class_head={bool(args.enable_ordinal_aux_class_head or train_config.get('enable_ordinal_aux_class_head', False))}; "
                f"ordinal_aux_class_task_keys={'|'.join(args.ordinal_aux_class_task_keys or train_config.get('ordinal_aux_class_task_keys') or []) or 'none'}; "
                f"ordinal_aux_class_weight={float(train_config.get('ordinal_aux_class_weight', 0.0) or 0.0)}; "
                f"ordinal_labeldist_task_keys={'|'.join(args.ordinal_labeldist_task_keys) if args.ordinal_labeldist_task_keys else 'none'}; "
                f"ordinal_labeldist_weight={float(args.ordinal_labeldist_weight) if args.ordinal_labeldist_weight is not None else 'none'}; "
                f"ordinal_labeldist_sigma={float(args.ordinal_labeldist_sigma) if args.ordinal_labeldist_sigma is not None else 'none'}; "
                f"ordinal_labeldist_edge_sigma={float(args.ordinal_labeldist_edge_sigma) if args.ordinal_labeldist_edge_sigma is not None else 'none'}; "
                f"task_loss_weights={'|'.join(f'{key}:{value:g}' for key, value in sorted(task_loss_weight_specs.items())) if task_loss_weight_specs else 'none'}; "
                f"multiclass_focal_task_keys={'|'.join(args.multiclass_focal_task_keys) if args.multiclass_focal_task_keys else 'none'}; "
                f"multiclass_focal_gamma={float(args.multiclass_focal_gamma) if args.multiclass_focal_gamma is not None else 'none'}; "
                f"multiclass_label_smoothing_task_keys={'|'.join(args.multiclass_label_smoothing_task_keys) if args.multiclass_label_smoothing_task_keys else 'none'}; "
                f"multiclass_label_smoothing={float(args.multiclass_label_smoothing) if args.multiclass_label_smoothing is not None else 'none'}; "
                f"enable_deprest_edge_ovr={bool(args.enable_deprest_edge_ovr)}; "
                f"shared_token_refiner_type={str(model_config.get('shared_token_refiner_type', 'none'))}; "
                f"shared_token_refiner_steps={int(model_config.get('shared_token_refiner_steps', 0))}; "
                f"task_local_trm_mode={str(model_config.get('task_local_trm_mode', 'none'))}; "
                f"task_local_trm_reasoning_dim={int(model_config.get('task_local_trm_reasoning_dim', model_config['token_dim']))}; "
                f"task_local_trm_h_cycles={int(model_config.get('task_local_trm_h_cycles', 0))}; "
                f"task_local_trm_l_cycles={int(model_config.get('task_local_trm_l_cycles', 0))}; "
                f"task_local_trm_task_names={'|'.join(args.task_local_trm_task_names) if args.task_local_trm_task_names else 'none'}; "
                f"enable_deprest_edge_specialist={bool(args.enable_deprest_edge_specialist)}; "
                f"fixed_multiclass_bias_specs={json.dumps(fixed_multiclass_bias_specs, sort_keys=True)}; "
                f"paired_regression_bridge_specs={json.dumps(paired_regression_bridge_specs, sort_keys=True)}; "
                f"multiclass_bias_task_keys={'|'.join(args.multiclass_bias_task_keys) if args.multiclass_bias_task_keys else 'none'}; "
                f"ordinal_bias_task_keys={'|'.join(args.ordinal_bias_task_keys or train_config.get('ordinal_bias_task_keys') or []) or 'none'}; "
                f"auto_ordinal_bias_calibration={bool(train_config.get('auto_ordinal_bias_calibration', False))}; "
                f"binary_threshold_prevalence_tolerance={float(train_config.get('binary_threshold_prevalence_tolerance', 0.0) or 0.0)}; "
                f"class_balance_mode={str(train_config.get('class_balance_mode', 'inverse'))}; "
                f"class_balance_beta={float(train_config.get('class_balance_beta', 0.999) or 0.999)}; "
                f"distillation_task_subset={'|'.join(sorted(distill_task_subset)) if distill_task_subset else 'all'}; "
                f"multiclass_task_keys={'|'.join(sorted(task_label_overrides)) if task_label_overrides else 'none'}"
            ),
        }
    )


if __name__ == "__main__":
    main()
