from __future__ import annotations

import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_metrics
from src.models.baseline_data import TaskBundle, list_dataset_tasks, load_task_bundle
from src.models.baselines import FIRST_WAVE_MODELS, build_estimator, get_baseline_spec, probe_dependencies
from src.models.capacity_aligned_baselines import fit_capacity_aligned_baseline
from src.models.sequence_baselines import fit_sequence_baseline, prepare_sequence_arrays
from src.utils.constants import DATASET_IDS, PROJECT_ROOT
from src.utils.io import ensure_dir, write_json

STAGE_NAME = "stage_05_baselines"
MODEL_CONFIG_PATH = "configs/model_configs/baselines_default.json"
TRAIN_CONFIG_PATH = "configs/train_configs/default_train.json"
EXPERIMENT_REGISTRY_PATH = PROJECT_ROOT / "outputs" / "logs" / "experiment_registry.csv"
BASELINE_RESULTS_PATH = PROJECT_ROOT / "outputs" / "tables" / "baseline_results.csv"
BASELINE_AUDIT_PATH = PROJECT_ROOT / "outputs" / "logs" / "baseline_run_audit.json"
PREDICTION_ROOT = PROJECT_ROOT / "outputs" / "predictions" / "baselines"
TEMP_ROOT = PROJECT_ROOT / "outputs" / "tmp"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run unified baseline models on canonical window tables.")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=["deprest_cat", "depresjon", "obf"],
        choices=DATASET_IDS,
        help="Datasets to run in this baseline wave.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=list(FIRST_WAVE_MODELS),
        help="Baseline model names to run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260417,
        help="Random seed for first-wave baseline runs.",
    )
    return parser.parse_args()


def _sanitize_name(value: object) -> str:
    sanitized = "".join(char if str(char).isalnum() or str(char) == "-" else "_" for char in str(value))
    return sanitized.strip("_") or "value"


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
    registry = registry.drop_duplicates(subset=["experiment_id"], keep="last")
    registry.to_csv(EXPERIMENT_REGISTRY_PATH, index=False)


def _load_results_table() -> pd.DataFrame:
    if BASELINE_RESULTS_PATH.exists():
        return pd.read_csv(BASELINE_RESULTS_PATH)
    return pd.DataFrame()


def _append_result_row(row: dict[str, object]) -> None:
    results = _load_results_table()
    results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
    subset = ["dataset_id", "task_name", "model_name"]
    if "seed" in results.columns:
        subset.append("seed")
    results = results.drop_duplicates(
        subset=subset,
        keep="last",
    )
    ensure_dir(BASELINE_RESULTS_PATH.parent)
    results.to_csv(BASELINE_RESULTS_PATH, index=False)


def _prepare_targets(frame: pd.DataFrame, label_type: str) -> np.ndarray:
    if label_type == "continuous":
        return pd.to_numeric(frame["y_raw"], errors="coerce").to_numpy(dtype=float)
    raise ValueError("_prepare_targets is reserved for continuous targets only.")


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


def _align_probabilities(model, probability_matrix: np.ndarray, n_classes: int) -> np.ndarray:
    if probability_matrix.ndim == 1:
        probability_matrix = np.column_stack([1.0 - probability_matrix, probability_matrix])

    estimator = model.named_steps["model"] if hasattr(model, "named_steps") else model
    trained_classes = np.asarray(getattr(estimator, "classes_", np.arange(probability_matrix.shape[1])), dtype=int)
    aligned = np.zeros((probability_matrix.shape[0], n_classes), dtype=float)
    for trained_index, trained_class in enumerate(trained_classes):
        if 0 <= int(trained_class) < n_classes:
            aligned[:, int(trained_class)] = probability_matrix[:, trained_index]
    zero_rows = aligned.sum(axis=1) <= 0.0
    if zero_rows.any():
        aligned[zero_rows] = 1.0 / n_classes
    aligned = aligned / aligned.sum(axis=1, keepdims=True)
    return aligned


def _build_prediction_frame(
    frame: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray | None,
    class_space: np.ndarray | None,
    label_type: str | None = None,
    y_true_index: np.ndarray | None = None,
    y_pred_index: np.ndarray | None = None,
) -> pd.DataFrame:
    output = frame[["dataset_id", "subject_id", "anchor_id", "task_name", "split"]].copy()
    if label_type is not None:
        output["label_type"] = label_type
    output["y_true"] = y_true
    output["y_pred"] = y_pred
    if y_true_index is not None:
        output["y_true_index"] = y_true_index
    if y_pred_index is not None:
        output["y_pred_index"] = y_pred_index
    if probabilities is not None and class_space is not None:
        for class_index, class_value in enumerate(class_space):
            output[f"proba_{_sanitize_name(class_value)}"] = probabilities[:, class_index]
    return output


def _collect_metric_row(
    experiment_id: str,
    dataset_id: str,
    task_name: str,
    model_name: str,
    label_type: str,
    status: str,
    notes: str,
    bundle: TaskBundle | None = None,
    valid_metrics: dict[str, float] | None = None,
    test_metrics: dict[str, float] | None = None,
    parameter_count: int | None = None,
) -> dict[str, object]:
    row = {
        "experiment_id": experiment_id,
        "dataset_id": dataset_id,
        "task_name": task_name,
        "model_name": model_name,
        "label_type": label_type,
        "status": status,
        "notes": notes,
        "n_train": len(bundle.train_frame) if bundle else np.nan,
        "n_valid": len(bundle.valid_frame) if bundle else np.nan,
        "n_test": len(bundle.test_frame) if bundle else np.nan,
        "parameter_count": parameter_count if parameter_count is not None else np.nan,
    }
    for prefix, metrics in (("valid", valid_metrics or {}), ("test", test_metrics or {})):
        for metric_name, metric_value in metrics.items():
            row[f"{prefix}_{metric_name}"] = metric_value
    return row


def _record_registry(
    experiment_id: str,
    model_name: str,
    dataset_id: str,
    task_name: str,
    status: str,
    notes: str,
) -> None:
    _append_registry_row(
        {
            "experiment_id": experiment_id,
            "stage": STAGE_NAME,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "model_name": model_name,
            "training_corpora": dataset_id,
            "target_dataset": dataset_id,
            "task_name": task_name,
            "split_config": f"data_interim/window_tables/{dataset_id}/splits.json",
            "model_config": MODEL_CONFIG_PATH,
            "train_config": TRAIN_CONFIG_PATH,
            "status": status,
            "notes": notes,
        }
    )


def run_single_experiment(
    dataset_id: str,
    task_name: str,
    model_name: str,
    dependencies: dict[str, bool],
    seed: int,
) -> dict[str, object]:
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    experiment_id = f"{STAGE_NAME}__{dataset_id}__{task_name}__{model_name}__{timestamp}"
    spec = get_baseline_spec(model_name)
    bundle = load_task_bundle(dataset_id=dataset_id, task_name=task_name)

    if not spec.implemented:
        notes = spec.note
        _record_registry(experiment_id, model_name, dataset_id, task_name, "skipped", notes)
        return _collect_metric_row(
            experiment_id,
            dataset_id,
            task_name,
            model_name,
            bundle.label_type,
            "skipped",
            notes,
            bundle,
        )

    if spec.dependency and not dependencies.get(spec.dependency, False):
        notes = f"Missing optional dependency: {spec.dependency}"
        _record_registry(experiment_id, model_name, dataset_id, task_name, "skipped", notes)
        return _collect_metric_row(
            experiment_id,
            dataset_id,
            task_name,
            model_name,
            bundle.label_type,
            "skipped",
            notes,
            bundle,
        )

    if bundle.train_frame.empty or bundle.valid_frame.empty or bundle.test_frame.empty:
        notes = "At least one split is empty after task filtering."
        _record_registry(experiment_id, model_name, dataset_id, task_name, "failed", notes)
        return _collect_metric_row(
            experiment_id,
            dataset_id,
            task_name,
            model_name,
            bundle.label_type,
            "failed",
            notes,
            bundle,
        )

    x_train = bundle.train_frame[bundle.feature_columns]
    x_valid = bundle.valid_frame[bundle.feature_columns]
    x_test = bundle.test_frame[bundle.feature_columns]
    notes = []
    outputs = None
    parameter_count = None
    if bundle.dropped_feature_columns:
        notes.append(f"dropped_all_nan_features={len(bundle.dropped_feature_columns)}")

    try:
        if spec.family == "sequence":
            sequence_arrays = prepare_sequence_arrays(bundle)
            notes.append(
                "sequence_view=short_medium_long_static_tokens"
            )
            if bundle.label_type == "continuous":
                y_train = _prepare_targets(bundle.train_frame, bundle.label_type)
                y_valid = _prepare_targets(bundle.valid_frame, bundle.label_type)
                y_test = _prepare_targets(bundle.test_frame, bundle.label_type)
                outputs = fit_sequence_baseline(
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
                )
                valid_pred = outputs["valid_pred"]
                test_pred = outputs["test_pred"]
                valid_metrics = outputs["valid_metrics"]
                test_metrics = outputs["test_metrics"]
                valid_predictions = _build_prediction_frame(
                    bundle.valid_frame,
                    y_valid,
                    valid_pred,
                    None,
                    None,
                    label_type=bundle.label_type,
                )
                test_predictions = _build_prediction_frame(
                    bundle.test_frame,
                    y_test,
                    test_pred,
                    None,
                    None,
                    label_type=bundle.label_type,
                )
            else:
                class_space, class_mapping, numeric_labels = _fit_class_space(bundle)
                y_train, _ = _encode_class_targets(bundle.train_frame, class_mapping, numeric_labels)
                y_valid, y_valid_raw = _encode_class_targets(bundle.valid_frame, class_mapping, numeric_labels)
                y_test, y_test_raw = _encode_class_targets(bundle.test_frame, class_mapping, numeric_labels)
                if np.unique(y_train).size < 2:
                    raise ValueError("Training split contains fewer than two classes.")
                outputs = fit_sequence_baseline(
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
                )
                valid_pred = outputs["valid_pred"].astype(int)
                test_pred = outputs["test_pred"].astype(int)
                valid_proba = outputs["valid_proba"]
                test_proba = outputs["test_proba"]
                valid_pred_raw = class_space[valid_pred]
                test_pred_raw = class_space[test_pred]
                valid_metrics = outputs["valid_metrics"]
                test_metrics = outputs["test_metrics"]
                valid_predictions = _build_prediction_frame(
                    bundle.valid_frame,
                    y_valid_raw,
                    valid_pred_raw,
                    valid_proba,
                    class_space,
                    label_type=bundle.label_type,
                    y_true_index=y_valid,
                    y_pred_index=valid_pred,
                )
                test_predictions = _build_prediction_frame(
                    bundle.test_frame,
                    y_test_raw,
                    test_pred_raw,
                    test_proba,
                    class_space,
                    label_type=bundle.label_type,
                    y_true_index=y_test,
                    y_pred_index=test_pred,
                )
                if bundle.label_type == "ordinal":
                    notes.append("ordinal_handled_as_ordered_classification_baseline")
        elif spec.family == "capacity_sequence":
            sequence_arrays = prepare_sequence_arrays(bundle)
            notes.append("sequence_view=short_medium_long_static_tokens")
            if bundle.label_type == "continuous":
                y_train = _prepare_targets(bundle.train_frame, bundle.label_type)
                y_valid = _prepare_targets(bundle.valid_frame, bundle.label_type)
                y_test = _prepare_targets(bundle.test_frame, bundle.label_type)
                outputs = fit_capacity_aligned_baseline(
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
                    sequence_length=sequence_arrays.train.shape[1],
                )
                valid_pred = outputs["valid_pred"]
                test_pred = outputs["test_pred"]
                valid_metrics = outputs["valid_metrics"]
                test_metrics = outputs["test_metrics"]
                parameter_count = int(outputs["parameter_count"])
                notes.append(f"capacity_aligned={outputs['capacity_note']}")
                valid_predictions = _build_prediction_frame(
                    bundle.valid_frame,
                    y_valid,
                    valid_pred,
                    None,
                    None,
                    label_type=bundle.label_type,
                )
                test_predictions = _build_prediction_frame(
                    bundle.test_frame,
                    y_test,
                    test_pred,
                    None,
                    None,
                    label_type=bundle.label_type,
                )
            else:
                class_space, class_mapping, numeric_labels = _fit_class_space(bundle)
                y_train, _ = _encode_class_targets(bundle.train_frame, class_mapping, numeric_labels)
                y_valid, y_valid_raw = _encode_class_targets(bundle.valid_frame, class_mapping, numeric_labels)
                y_test, y_test_raw = _encode_class_targets(bundle.test_frame, class_mapping, numeric_labels)
                if np.unique(y_train).size < 2:
                    raise ValueError("Training split contains fewer than two classes.")
                outputs = fit_capacity_aligned_baseline(
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
                    sequence_length=sequence_arrays.train.shape[1],
                )
                valid_pred = outputs["valid_pred"].astype(int)
                test_pred = outputs["test_pred"].astype(int)
                valid_proba = outputs["valid_proba"]
                test_proba = outputs["test_proba"]
                valid_pred_raw = class_space[valid_pred]
                test_pred_raw = class_space[test_pred]
                valid_metrics = outputs["valid_metrics"]
                test_metrics = outputs["test_metrics"]
                parameter_count = int(outputs["parameter_count"])
                notes.append(f"capacity_aligned={outputs['capacity_note']}")
                valid_predictions = _build_prediction_frame(
                    bundle.valid_frame,
                    y_valid_raw,
                    valid_pred_raw,
                    valid_proba,
                    class_space,
                    label_type=bundle.label_type,
                    y_true_index=y_valid,
                    y_pred_index=valid_pred,
                )
                test_predictions = _build_prediction_frame(
                    bundle.test_frame,
                    y_test_raw,
                    test_pred_raw,
                    test_proba,
                    class_space,
                    label_type=bundle.label_type,
                    y_true_index=y_test,
                    y_pred_index=test_pred,
                )
                if bundle.label_type == "ordinal":
                    notes.append("ordinal_handled_as_ordered_classification_baseline")
        elif spec.family == "torch_tabular":
            x_train_array = x_train.to_numpy(dtype=np.float32)
            x_valid_array = x_valid.to_numpy(dtype=np.float32)
            x_test_array = x_test.to_numpy(dtype=np.float32)
            means = np.nanmean(x_train_array, axis=0)
            means = np.nan_to_num(means, nan=0.0)
            stds = np.nanstd(x_train_array, axis=0)
            stds = np.nan_to_num(stds, nan=1.0)
            stds[stds == 0.0] = 1.0

            def _normalize_tabular(array: np.ndarray) -> np.ndarray:
                normalized = (array - means) / stds
                return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

            x_train_array = _normalize_tabular(x_train_array)
            x_valid_array = _normalize_tabular(x_valid_array)
            x_test_array = _normalize_tabular(x_test_array)
            if bundle.label_type == "continuous":
                y_train = _prepare_targets(bundle.train_frame, bundle.label_type)
                y_valid = _prepare_targets(bundle.valid_frame, bundle.label_type)
                y_test = _prepare_targets(bundle.test_frame, bundle.label_type)
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
                valid_pred = outputs["valid_pred"]
                test_pred = outputs["test_pred"]
                valid_metrics = outputs["valid_metrics"]
                test_metrics = outputs["test_metrics"]
                parameter_count = int(outputs["parameter_count"])
                notes.append(f"capacity_aligned={outputs['capacity_note']}")
                valid_predictions = _build_prediction_frame(
                    bundle.valid_frame,
                    y_valid,
                    valid_pred,
                    None,
                    None,
                    label_type=bundle.label_type,
                )
                test_predictions = _build_prediction_frame(
                    bundle.test_frame,
                    y_test,
                    test_pred,
                    None,
                    None,
                    label_type=bundle.label_type,
                )
            else:
                class_space, class_mapping, numeric_labels = _fit_class_space(bundle)
                y_train, _ = _encode_class_targets(bundle.train_frame, class_mapping, numeric_labels)
                y_valid, y_valid_raw = _encode_class_targets(bundle.valid_frame, class_mapping, numeric_labels)
                y_test, y_test_raw = _encode_class_targets(bundle.test_frame, class_mapping, numeric_labels)
                if np.unique(y_train).size < 2:
                    raise ValueError("Training split contains fewer than two classes.")
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
                valid_pred = outputs["valid_pred"].astype(int)
                test_pred = outputs["test_pred"].astype(int)
                valid_proba = outputs["valid_proba"]
                test_proba = outputs["test_proba"]
                valid_pred_raw = class_space[valid_pred]
                test_pred_raw = class_space[test_pred]
                valid_metrics = outputs["valid_metrics"]
                test_metrics = outputs["test_metrics"]
                parameter_count = int(outputs["parameter_count"])
                notes.append(f"capacity_aligned={outputs['capacity_note']}")
                valid_predictions = _build_prediction_frame(
                    bundle.valid_frame,
                    y_valid_raw,
                    valid_pred_raw,
                    valid_proba,
                    class_space,
                    label_type=bundle.label_type,
                    y_true_index=y_valid,
                    y_pred_index=valid_pred,
                )
                test_predictions = _build_prediction_frame(
                    bundle.test_frame,
                    y_test_raw,
                    test_pred_raw,
                    test_proba,
                    class_space,
                    label_type=bundle.label_type,
                    y_true_index=y_test,
                    y_pred_index=test_pred,
                )
                if bundle.label_type == "ordinal":
                    notes.append("ordinal_handled_as_ordered_classification_baseline")
        elif bundle.label_type == "continuous":
            y_train = _prepare_targets(bundle.train_frame, bundle.label_type)
            y_valid = _prepare_targets(bundle.valid_frame, bundle.label_type)
            y_test = _prepare_targets(bundle.test_frame, bundle.label_type)

            estimator = build_estimator(model_name=model_name, label_type=bundle.label_type, seed=seed, n_classes=1)
            estimator.fit(x_train, y_train)
            valid_pred = estimator.predict(x_valid)
            test_pred = estimator.predict(x_test)
            valid_metrics = compute_metrics(bundle.label_type, y_valid, valid_pred)
            test_metrics = compute_metrics(bundle.label_type, y_test, test_pred)

            valid_predictions = _build_prediction_frame(
                bundle.valid_frame,
                y_valid,
                valid_pred,
                None,
                None,
                label_type=bundle.label_type,
            )
            test_predictions = _build_prediction_frame(
                bundle.test_frame,
                y_test,
                test_pred,
                None,
                None,
                label_type=bundle.label_type,
            )
        else:
            class_space, class_mapping, numeric_labels = _fit_class_space(bundle)
            y_train, y_train_raw = _encode_class_targets(bundle.train_frame, class_mapping, numeric_labels)
            y_valid, y_valid_raw = _encode_class_targets(bundle.valid_frame, class_mapping, numeric_labels)
            y_test, y_test_raw = _encode_class_targets(bundle.test_frame, class_mapping, numeric_labels)

            if np.unique(y_train).size < 2:
                raise ValueError("Training split contains fewer than two classes.")

            estimator = build_estimator(
                model_name=model_name,
                label_type=bundle.label_type,
                seed=seed,
                n_classes=len(class_space),
            )
            estimator.fit(x_train, y_train)

            valid_proba = _align_probabilities(estimator, estimator.predict_proba(x_valid), len(class_space))
            test_proba = _align_probabilities(estimator, estimator.predict_proba(x_test), len(class_space))
            valid_pred = valid_proba.argmax(axis=1)
            test_pred = test_proba.argmax(axis=1)
            valid_pred_raw = class_space[valid_pred]
            test_pred_raw = class_space[test_pred]

            valid_metrics = compute_metrics(bundle.label_type, y_valid, valid_pred, valid_proba)
            test_metrics = compute_metrics(bundle.label_type, y_test, test_pred, test_proba)
            valid_predictions = _build_prediction_frame(
                bundle.valid_frame,
                y_valid_raw,
                valid_pred_raw,
                valid_proba,
                class_space,
                label_type=bundle.label_type,
                y_true_index=y_valid,
                y_pred_index=valid_pred,
            )
            test_predictions = _build_prediction_frame(
                bundle.test_frame,
                y_test_raw,
                test_pred_raw,
                test_proba,
                class_space,
                label_type=bundle.label_type,
                y_true_index=y_test,
                y_pred_index=test_pred,
            )

            if bundle.label_type == "ordinal":
                notes.append("ordinal_handled_as_ordered_classification_baseline")

        prediction_dir = ensure_dir(PREDICTION_ROOT / model_name)
        prediction_path = prediction_dir / f"{dataset_id}__{task_name}__seed{seed}.csv"
        pd.concat([valid_predictions, test_predictions], ignore_index=True).to_csv(prediction_path, index=False)

        note_text = "; ".join(notes) if notes else "ok"
        _record_registry(experiment_id, model_name, dataset_id, task_name, "completed", note_text)
        return _collect_metric_row(
            experiment_id,
            dataset_id,
            task_name,
            model_name,
            bundle.label_type,
            "completed",
            note_text,
            bundle,
            valid_metrics,
            test_metrics,
            parameter_count=parameter_count,
        )
    except Exception as exc:  # pragma: no cover - backend dependent
        note_text = "; ".join(notes + [f"error={exc}"]) if notes else f"error={exc}"
        _record_registry(experiment_id, model_name, dataset_id, task_name, "failed", note_text)
        return _collect_metric_row(
            experiment_id,
            dataset_id,
            task_name,
            model_name,
            bundle.label_type,
            "failed",
            note_text,
            bundle,
        )


def main() -> None:
    args = parse_args()
    temp_dir = ensure_dir(TEMP_ROOT)
    os.environ["TMP"] = str(temp_dir)
    os.environ["TEMP"] = str(temp_dir)
    os.environ["JOBLIB_TEMP_FOLDER"] = str(temp_dir)
    dependencies = probe_dependencies()
    audit_payload = {
        "stage": STAGE_NAME,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dependencies": dependencies,
        "datasets": args.datasets,
        "models": args.models,
        "results": [],
    }

    for dataset_id in args.datasets:
        for task_spec in list_dataset_tasks(dataset_id):
            task_name = task_spec["task_name"]
            for model_name in args.models:
                result_row = run_single_experiment(
                    dataset_id=dataset_id,
                    task_name=task_name,
                    model_name=model_name,
                    dependencies=dependencies,
                    seed=args.seed,
                )
                result_row["seed"] = args.seed
                _append_result_row(result_row)
                audit_payload["results"].append(result_row)

    write_json(BASELINE_AUDIT_PATH, audit_payload)


if __name__ == "__main__":
    main()
