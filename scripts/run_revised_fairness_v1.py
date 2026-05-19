from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BUNDLE_ROOT = ROOT
if str(BUNDLE_ROOT) not in sys.path:
    sys.path.insert(0, str(BUNDLE_ROOT))

from src.evaluation.metrics import compute_metrics  # noqa: E402
from src.models.baselines import build_estimator, probe_dependencies  # noqa: E402


WINDOW_ROOT = BUNDLE_ROOT / "data_interim" / "window_tables"
MCTRCM_PRED_ROOT = BUNDLE_ROOT / "outputs" / "predictions" / "mctrcm_v2"
MCTRCM_LOG_ROOT = BUNDLE_ROOT / "outputs" / "logs"
OUT_ROOT = ROOT / "results" / "revised_fairness_v1"
TABLE_ROOT = ROOT / "tables" / "final"

SEEDS = [20260417, 20260418, 20260419, 20260420, 20260421]
PRIMARY_TASKS = {
    "deprest_cat": ["phq9_reg", "gad7_reg", "phq9_cat", "gad7_cat"],
    "psyche_d": ["phq_change_binary", "phq_change_multiclass"],
}
TASK_DISPLAY = {
    ("deprest_cat", "phq9_reg"): "DepreST-CAT PHQ-9 severity",
    ("deprest_cat", "gad7_reg"): "DepreST-CAT GAD-7 severity",
    ("deprest_cat", "phq9_cat"): "DepreST-CAT PHQ-9 category",
    ("deprest_cat", "gad7_cat"): "DepreST-CAT GAD-7 category",
    ("psyche_d", "phq_change_binary"): "PSYCHE-D PHQ-change binary",
    ("psyche_d", "phq_change_multiclass"): "PSYCHE-D PHQ-change multiclass",
}
SIMPLE_MODELS = ["null", "elastic_net", "lightgbm", "xgboost", "ebm", "mlp"]
FEATURE_SETS = [
    "FULL",
    "SENSOR_ONLY",
    "SENSOR_VALUES_ONLY",
    "MISSINGNESS_ONLY",
    "SYMPTOM_CONTEXT_ONLY",
    "STATIC_CLINICAL_ONLY",
    "SYMPTOM_STATIC_CLINICAL",
    "SENSOR_PLUS_STATIC",
]
CLASSIFICATION_TYPES = {"binary", "ordinal", "multiclass"}
SENSOR_MODALITIES = ["activity", "sleep", "communication", "phone_use", "mobility"]
SENSOR_PREFIXES = tuple(f"feat_{name}_" for name in SENSOR_MODALITIES)
SENSOR_NATIVE_PREFIXES = ("feat_native_steps_", "feat_native_sleep_")
MISSING_TOKENS = ("missing_ratio",)


@dataclass
class SplitBundle:
    dataset: str
    task: str
    label_type: str
    class_space: np.ndarray | None
    numeric_labels: bool
    frame: pd.DataFrame
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    split_manifest_id: str


def ensure_dirs() -> None:
    for path in [
        OUT_ROOT,
        OUT_ROOT / "model_configs",
        OUT_ROOT / "predictions",
        OUT_ROOT / "calibration",
        OUT_ROOT / "logs",
        TABLE_ROOT,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def safe_float(value: Any) -> float:
    try:
        value = float(value)
        if math.isfinite(value):
            return value
    except (TypeError, ValueError):
        pass
    return math.nan


def standard_error(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if len(values) <= 1:
        return 0.0 if len(values) == 1 else math.nan
    return float(values.std(ddof=1) / math.sqrt(len(values)))


def fmt_metric(mean: float, se: float) -> str:
    if pd.isna(mean):
        return "--"
    if pd.isna(se):
        return f"{mean:.3f}"
    return f"{mean:.3f} $\\pm$ {se:.3f}"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_split_manifest(dataset: str) -> dict[str, Any]:
    return read_json(WINDOW_ROOT / dataset / "splits.json")


def assign_split(subject_id: object, manifest: dict[str, Any]) -> str | None:
    normalized = str(subject_id)
    for split_name, key in (
        ("train", "train_subjects"),
        ("valid", "valid_subjects"),
        ("test", "test_subjects"),
    ):
        if normalized in {str(item) for item in manifest.get(key, [])}:
            return split_name
    return None


def load_task_bundle(dataset: str, task: str) -> SplitBundle:
    dataset_dir = WINDOW_ROOT / dataset
    windows = pd.read_parquet(dataset_dir / "windows_wide.parquet")
    labels = pd.read_csv(dataset_dir / "labels.csv")
    labels = labels.loc[(labels["label_available"] == 1) & (labels["task_name"] == task)].copy()
    manifest = load_split_manifest(dataset)

    label_columns = [
        "anchor_id",
        "task_name",
        "label_type",
        "y_raw",
        "class_label",
        "label_available",
    ]
    for optional in ["phq9_score_start", "phq9_cat_start", "phq9_score_end", "phq9_cat_end"]:
        if optional in labels.columns:
            label_columns.append(optional)

    frame = windows.merge(labels[label_columns], on=["anchor_id", "task_name"], how="inner")
    frame["split"] = frame["subject_id"].map(lambda value: assign_split(value, manifest))
    if frame["split"].isna().any():
        missing = sorted(frame.loc[frame["split"].isna(), "subject_id"].astype(str).unique().tolist())
        raise ValueError(f"Missing split assignment for {dataset}/{task}: {missing[:5]}")

    if "phq9_score_start" in frame.columns:
        frame["prior_phq9_score_start"] = pd.to_numeric(frame["phq9_score_start"], errors="coerce")
    if "phq9_cat_start" in frame.columns:
        frame["prior_phq9_cat_start"] = pd.to_numeric(frame["phq9_cat_start"], errors="coerce")

    label_types = sorted(frame["label_type"].astype(str).unique().tolist())
    if len(label_types) != 1:
        raise ValueError(f"Expected one label type for {dataset}/{task}, got {label_types}")
    label_type = label_types[0]

    if label_type == "continuous":
        class_space = None
        numeric_labels = True
    else:
        raw = pd.to_numeric(frame["y_raw"], errors="coerce")
        numeric_labels = bool(raw.notna().all())
        if numeric_labels:
            class_space = np.sort(raw.astype(int).unique())
        else:
            class_space = np.sort(frame["y_raw"].astype(str).unique())

    split_manifest_id = str((dataset_dir / "splits.json").relative_to(ROOT))
    return SplitBundle(
        dataset=dataset,
        task=task,
        label_type=label_type,
        class_space=class_space,
        numeric_labels=numeric_labels,
        frame=frame.reset_index(drop=True),
        train=frame.loc[frame["split"] == "train"].reset_index(drop=True),
        valid=frame.loc[frame["split"] == "valid"].reset_index(drop=True),
        test=frame.loc[frame["split"] == "test"].reset_index(drop=True),
        split_manifest_id=split_manifest_id,
    )


def all_candidate_feature_columns(frame: pd.DataFrame) -> list[str]:
    prefixes = ("feat_", "modality_mask_", "concept_mask_", "prior_phq9_")
    return sorted([column for column in frame.columns if column.startswith(prefixes)])


def is_missingness_column(column: str) -> bool:
    return (
        column.startswith("modality_mask_")
        or any(token in column for token in MISSING_TOKENS)
        or column == "feat_static_missing_rate"
    )


def is_sensor_value_column(column: str) -> bool:
    if column.startswith(SENSOR_NATIVE_PREFIXES):
        return True
    if not column.startswith(SENSOR_PREFIXES):
        return False
    return not is_missingness_column(column)


def is_sensor_missingness_column(column: str) -> bool:
    if column.startswith("modality_mask_"):
        modality = column.replace("modality_mask_", "", 1)
        return modality in SENSOR_MODALITIES
    if not column.startswith(SENSOR_PREFIXES):
        return False
    return is_missingness_column(column)


def is_symptom_value_column(dataset: str, column: str) -> bool:
    if dataset == "deprest_cat":
        return False
    if column.startswith("prior_phq9_"):
        return dataset == "psyche_d"
    if column.startswith("feat_symptom_context_") and not is_missingness_column(column):
        return True
    if column == "feat_static_baseline_severity":
        return True
    return False


def is_deprest_prior_treatment_column(column: str) -> bool:
    return column.startswith("feat_symptom_context_")


def is_static_clinical_column(dataset: str, column: str) -> bool:
    if column in {"feat_static_age", "feat_static_sex", "feat_static_missing_rate"}:
        return True
    if dataset == "deprest_cat" and is_deprest_prior_treatment_column(column):
        return True
    if column.startswith("feat_native_") and not column.startswith(SENSOR_NATIVE_PREFIXES):
        return True
    if column == "feat_static_baseline_severity":
        return dataset not in {"psyche_d", "studentlife", "depresjon", "obf"}
    return False


def feature_columns_for_set(dataset: str, frame: pd.DataFrame, feature_set: str) -> list[str]:
    candidates = all_candidate_feature_columns(frame)
    sensor_values = [column for column in candidates if is_sensor_value_column(column)]
    sensor_missing = [column for column in candidates if is_sensor_missingness_column(column)]
    missingness = [column for column in candidates if is_missingness_column(column)]
    symptom = [column for column in candidates if is_symptom_value_column(dataset, column)]
    static_clinical = [column for column in candidates if is_static_clinical_column(dataset, column)]

    if feature_set == "FULL":
        return candidates
    if feature_set == "SENSOR_ONLY":
        return sorted(set(sensor_values + sensor_missing))
    if feature_set == "SENSOR_VALUES_ONLY":
        return sorted(set(sensor_values))
    if feature_set == "MISSINGNESS_ONLY":
        return sorted(set(missingness))
    if feature_set == "SYMPTOM_CONTEXT_ONLY":
        return sorted(set(symptom))
    if feature_set == "STATIC_CLINICAL_ONLY":
        return sorted(set(static_clinical))
    if feature_set == "SYMPTOM_STATIC_CLINICAL":
        return sorted(set(symptom + static_clinical))
    if feature_set == "SENSOR_PLUS_STATIC":
        return sorted(set(sensor_values + sensor_missing + static_clinical))
    raise KeyError(f"Unknown feature set: {feature_set}")


def build_feature_set_definitions() -> dict[str, Any]:
    definitions: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sensor_only_contains_missingness_summaries": True,
        "sensor_values_only_contains_missingness_summaries": False,
        "notes": [
            "All splits reuse the repository-local data_interim/window_tables/*/splits.json manifests.",
            "DepreST-CAT prior treatment indicators are stored in feat_symptom_context_* by preprocessing, but this suite classifies them as STATIC_CLINICAL context rather than prior symptom context.",
            "DepreST-CAT SYMPTOM_CONTEXT_ONLY intentionally contains no current PHQ-9 or GAD-7 target-derived feature.",
            "PSYCHE-D prior_phq9_score_start and prior_phq9_cat_start are copied from label metadata as pre-endpoint baseline symptom context and are included only in feature sets that include symptom context or FULL.",
            "MISSINGNESS_ONLY uses observed modality indicators and missing-rate columns; communication coverage columns are treated as behavior summaries, not missingness parameters.",
        ],
        "feature_sets": {},
        "by_dataset_candidates": {},
    }
    for dataset in PRIMARY_TASKS:
        frame = pd.read_parquet(WINDOW_ROOT / dataset / "windows_wide.parquet")
        labels = pd.read_csv(WINDOW_ROOT / dataset / "labels.csv")
        if "phq9_score_start" in labels.columns:
            frame["prior_phq9_score_start"] = np.nan
        if "phq9_cat_start" in labels.columns:
            frame["prior_phq9_cat_start"] = np.nan
        all_cols = all_candidate_feature_columns(frame)
        definitions["by_dataset_candidates"][dataset] = {
            "all_candidate_columns": all_cols,
            "sensor_value_columns": [column for column in all_cols if is_sensor_value_column(column)],
            "sensor_missingness_columns": [column for column in all_cols if is_sensor_missingness_column(column)],
            "missingness_columns": [column for column in all_cols if is_missingness_column(column)],
            "symptom_context_columns": [column for column in all_cols if is_symptom_value_column(dataset, column)],
            "static_clinical_columns": [column for column in all_cols if is_static_clinical_column(dataset, column)],
        }
    for feature_set in FEATURE_SETS:
        definitions["feature_sets"][feature_set] = {
            "included_modality_groups": {
                "FULL": ["sensor", "missingness", "symptom_context", "static_clinical", "native", "modality_masks", "concept_masks"],
                "SENSOR_ONLY": ["activity", "sleep", "communication", "phone_use", "mobility", "sensor_missingness"],
                "SENSOR_VALUES_ONLY": ["activity", "sleep", "communication", "phone_use", "mobility"],
                "MISSINGNESS_ONLY": ["observed_indicators", "missing_ratios", "static_missing_rate"],
                "SYMPTOM_CONTEXT_ONLY": ["prior_or_baseline_symptom"],
                "STATIC_CLINICAL_ONLY": ["age", "sex", "demographic", "clinical_context", "prior_treatment"],
                "SYMPTOM_STATIC_CLINICAL": ["prior_or_baseline_symptom", "static_clinical"],
                "SENSOR_PLUS_STATIC": ["sensor", "sensor_missingness", "static_clinical"],
            }[feature_set],
            "included_columns_by_dataset": {
                dataset: feature_columns_for_set(
                    dataset,
                    pd.read_parquet(WINDOW_ROOT / dataset / "windows_wide.parquet").assign(
                        **(
                            {"prior_phq9_score_start": np.nan, "prior_phq9_cat_start": np.nan}
                            if dataset == "psyche_d"
                            else {}
                        )
                    ),
                    feature_set,
                )
                for dataset in PRIMARY_TASKS
            },
            "excluded_columns_by_dataset": {},
        }
        for dataset in PRIMARY_TASKS:
            all_cols = set(definitions["by_dataset_candidates"][dataset]["all_candidate_columns"])
            included = set(definitions["feature_sets"][feature_set]["included_columns_by_dataset"][dataset])
            definitions["feature_sets"][feature_set]["excluded_columns_by_dataset"][dataset] = sorted(all_cols - included)
    return definitions


def drop_all_nan_train_columns(bundle: SplitBundle, columns: list[str]) -> tuple[list[str], list[str], bool]:
    kept = [column for column in columns if column in bundle.train.columns and bundle.train[column].notna().any()]
    dropped = [column for column in columns if column not in kept]
    used_constant = False
    if not kept:
        for frame in (bundle.train, bundle.valid, bundle.test):
            frame["__constant_feature__"] = 0.0
        kept = ["__constant_feature__"]
        used_constant = True
    return kept, dropped, used_constant


def class_mapping(bundle: SplitBundle) -> dict[Any, int]:
    if bundle.class_space is None:
        return {}
    return {value: index for index, value in enumerate(bundle.class_space.tolist())}


def n_classes_for_bundle(bundle: SplitBundle) -> int:
    return 0 if bundle.class_space is None else int(len(bundle.class_space))


def encode_y(frame: pd.DataFrame, bundle: SplitBundle) -> np.ndarray:
    if bundle.label_type == "continuous":
        return pd.to_numeric(frame["y_raw"], errors="coerce").to_numpy(dtype=float)
    mapping = class_mapping(bundle)
    if bundle.numeric_labels:
        raw = pd.to_numeric(frame["y_raw"], errors="coerce").astype(int)
    else:
        raw = frame["y_raw"].astype(str)
    return raw.map(mapping).to_numpy(dtype=int)


def raw_y(frame: pd.DataFrame, bundle: SplitBundle) -> np.ndarray:
    if bundle.label_type == "continuous":
        return pd.to_numeric(frame["y_raw"], errors="coerce").to_numpy(dtype=float)
    if bundle.numeric_labels:
        return pd.to_numeric(frame["y_raw"], errors="coerce").to_numpy()
    return frame["y_raw"].astype(str).to_numpy()


def normalize_probabilities(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim == 1:
        probabilities = np.column_stack([1.0 - probabilities, probabilities])
    probabilities = np.nan_to_num(probabilities, nan=0.0, posinf=0.0, neginf=0.0)
    probabilities = np.clip(probabilities, 1e-12, 1.0)
    sums = probabilities.sum(axis=1, keepdims=True)
    zero = sums[:, 0] <= 0
    if np.any(zero):
        probabilities[zero] = 1.0 / probabilities.shape[1]
        sums = probabilities.sum(axis=1, keepdims=True)
    return probabilities / sums


def ece_score(y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    probabilities = normalize_probabilities(probabilities)
    y_true = np.asarray(y_true, dtype=int)
    if len(y_true) == 0:
        return math.nan
    pred = probabilities.argmax(axis=1)
    conf = probabilities.max(axis=1)
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        if right == 1.0:
            mask = (conf >= left) & (conf <= right)
        else:
            mask = (conf >= left) & (conf < right)
        if np.any(mask):
            total += float(mask.mean()) * abs(float(correct[mask].mean()) - float(conf[mask].mean()))
    return float(total)


def nll_score(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    probabilities = normalize_probabilities(probabilities)
    indices = np.arange(len(y_true))
    return float(-np.mean(np.log(probabilities[indices, y_true] + 1e-12)))


def brier_multiclass(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    probabilities = normalize_probabilities(probabilities)
    target = np.zeros_like(probabilities)
    target[np.arange(len(y_true)), y_true] = 1.0
    if probabilities.shape[1] == 2:
        return float(np.mean((probabilities[:, 1] - y_true.astype(float)) ** 2))
    return float(np.mean(np.sum((probabilities - target) ** 2, axis=1)))


def temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    probabilities = normalize_probabilities(probabilities)
    logits = np.log(probabilities + 1e-12) / float(temperature)
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def fit_temperature(y_valid: np.ndarray, valid_prob: np.ndarray) -> tuple[float, dict[str, float]]:
    best_temp = 1.0
    best_score = math.inf
    diagnostics: dict[str, float] = {}
    for temp in np.concatenate([np.linspace(0.5, 3.0, 26), np.array([1.0])]):
        scaled = temperature_scale(valid_prob, float(temp))
        score = nll_score(y_valid, scaled) + brier_multiclass(y_valid, scaled)
        if score < best_score:
            best_score = score
            best_temp = float(temp)
            diagnostics = {
                "validation_nll": nll_score(y_valid, scaled),
                "validation_brier": brier_multiclass(y_valid, scaled),
                "validation_ece": ece_score(y_valid, scaled),
            }
    return best_temp, diagnostics


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import f1_score

    return safe_float(f1_score(y_true, y_pred, average="macro"))


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import balanced_accuracy_score

    return safe_float(balanced_accuracy_score(y_true, y_pred))


def fit_binary_threshold(y_valid: np.ndarray, valid_prob: np.ndarray) -> float:
    positive = normalize_probabilities(valid_prob)[:, 1]
    quantiles = np.quantile(positive, np.linspace(0.05, 0.95, 19)) if len(positive) else np.array([0.5])
    candidates = np.unique(np.concatenate([np.linspace(0.05, 0.95, 37), quantiles, np.array([0.5])]))
    best_key = (-math.inf, -math.inf, -math.inf)
    best = 0.5
    for threshold in candidates:
        pred = (positive >= float(threshold)).astype(int)
        key = (
            balanced_accuracy(y_valid, pred),
            macro_f1(y_valid, pred),
            -abs(float(threshold) - 0.5),
        )
        if key > best_key:
            best_key = key
            best = float(threshold)
    return best


def apply_class_bias(probabilities: np.ndarray, biases: np.ndarray) -> np.ndarray:
    probabilities = normalize_probabilities(probabilities)
    logits = np.log(probabilities + 1e-12) + biases.reshape(1, -1)
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def fit_class_bias(y_valid: np.ndarray, valid_prob: np.ndarray) -> np.ndarray:
    probabilities = normalize_probabilities(valid_prob)
    n_classes = probabilities.shape[1]
    biases = np.zeros(n_classes, dtype=float)
    grids = [np.linspace(-2.0, 2.0, 17), np.linspace(-0.6, 0.6, 13)]
    best_global = (-math.inf, -math.inf)
    for grid in grids:
        for _ in range(2):
            for class_index in range(n_classes):
                best_value = biases[class_index]
                best_key = best_global
                for value in grid:
                    trial = biases.copy()
                    trial[class_index] = float(value)
                    trial -= trial.mean()
                    pred = apply_class_bias(probabilities, trial).argmax(axis=1)
                    key = (balanced_accuracy(y_valid, pred), macro_f1(y_valid, pred))
                    if key > best_key:
                        best_key = key
                        best_value = float(value)
                biases[class_index] = best_value
                biases -= biases.mean()
                best_global = best_key
    return biases


def classification_metric_bundle(y_true: np.ndarray, probabilities: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    metrics = compute_metrics("binary" if probabilities.shape[1] == 2 else "multiclass", y_true, y_pred, probabilities)
    return {
        "ba": safe_float(metrics.get("balanced_accuracy")),
        "macro_f1": safe_float(metrics.get("macro_f1")),
        "auroc": safe_float(metrics.get("auroc")),
        "auprc": safe_float(metrics.get("auprc")),
        "brier": safe_float(metrics.get("brier_score")),
        "ece": safe_float(metrics.get("ece")),
    }


def apply_validation_postprocessing(
    label_type: str,
    y_valid: np.ndarray,
    valid_prob: np.ndarray,
    y_test: np.ndarray,
    test_prob: np.ndarray,
) -> dict[str, Any]:
    valid_prob = normalize_probabilities(valid_prob)
    test_prob = normalize_probabilities(test_prob)
    valid_uncal_pred = valid_prob.argmax(axis=1)
    test_uncal_pred = test_prob.argmax(axis=1)
    valid_uncal = classification_metric_bundle(y_valid, valid_prob, valid_uncal_pred)
    test_uncal = classification_metric_bundle(y_test, test_prob, test_uncal_pred)

    temperature, temp_diag = fit_temperature(y_valid, valid_prob)
    valid_temp = temperature_scale(valid_prob, temperature)
    test_temp = temperature_scale(test_prob, temperature)
    payload: dict[str, Any] = {
        "temperature": temperature,
        "temperature_validation_diagnostics": temp_diag,
        "uncalibrated_valid_pred": valid_uncal_pred,
        "uncalibrated_test_pred": test_uncal_pred,
        "uncalibrated_valid_prob": valid_prob,
        "uncalibrated_test_prob": test_prob,
        "uncalibrated_valid_metrics": valid_uncal,
        "uncalibrated_test_metrics": test_uncal,
    }

    if valid_prob.shape[1] == 2 and label_type == "binary":
        threshold = fit_binary_threshold(y_valid, valid_temp)
        valid_pred = (valid_temp[:, 1] >= threshold).astype(int)
        test_pred = (test_temp[:, 1] >= threshold).astype(int)
        valid_cal = classification_metric_bundle(y_valid, valid_temp, valid_pred)
        test_cal = classification_metric_bundle(y_test, test_temp, test_pred)
        payload.update(
            {
                "postprocessing_type": "temperature_plus_binary_threshold",
                "threshold": threshold,
                "biases": None,
                "calibrated_valid_pred": valid_pred,
                "calibrated_test_pred": test_pred,
                "calibrated_valid_prob": valid_temp,
                "calibrated_test_prob": test_temp,
                "calibrated_valid_metrics": valid_cal,
                "calibrated_test_metrics": test_cal,
            }
        )
        return payload

    biases = fit_class_bias(y_valid, valid_temp)
    valid_adjusted = apply_class_bias(valid_temp, biases)
    test_adjusted = apply_class_bias(test_temp, biases)
    valid_pred = valid_adjusted.argmax(axis=1)
    test_pred = test_adjusted.argmax(axis=1)
    valid_cal = classification_metric_bundle(y_valid, valid_adjusted, valid_pred)
    test_cal = classification_metric_bundle(y_test, test_adjusted, test_pred)
    payload.update(
        {
            "postprocessing_type": "temperature_plus_class_bias",
            "threshold": None,
            "biases": biases.tolist(),
            "calibrated_valid_pred": valid_pred,
            "calibrated_test_pred": test_pred,
            "calibrated_valid_prob": valid_adjusted,
            "calibrated_test_prob": test_adjusted,
            "calibrated_valid_metrics": valid_cal,
            "calibrated_test_metrics": test_cal,
        }
    )
    return payload


def align_probabilities(model: Any, probabilities: np.ndarray, n_classes: int) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim == 1:
        probabilities = np.column_stack([1.0 - probabilities, probabilities])
    estimator = model.named_steps["model"] if hasattr(model, "named_steps") else model
    classes = np.asarray(getattr(estimator, "classes_", np.arange(probabilities.shape[1])), dtype=int)
    aligned = np.zeros((probabilities.shape[0], n_classes), dtype=float)
    for trained_index, trained_class in enumerate(classes):
        if 0 <= int(trained_class) < n_classes and trained_index < probabilities.shape[1]:
            aligned[:, int(trained_class)] = probabilities[:, trained_index]
    zero = aligned.sum(axis=1) <= 0
    if np.any(zero):
        aligned[zero] = 1.0 / n_classes
    return normalize_probabilities(aligned)


def build_mlp_estimator(label_type: str, seed: int) -> Any:
    from sklearn.impute import SimpleImputer
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if label_type == "continuous":
        model = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            alpha=1e-4,
            batch_size=128,
            learning_rate_init=1e-3,
            max_iter=500,
            early_stopping=True,
            n_iter_no_change=20,
            random_state=seed,
        )
    else:
        model = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            alpha=1e-4,
            batch_size=128,
            learning_rate_init=1e-3,
            max_iter=500,
            early_stopping=True,
            n_iter_no_change=20,
            random_state=seed,
        )
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", model)])


def run_null(bundle: SplitBundle) -> dict[str, Any]:
    y_train = encode_y(bundle.train, bundle)
    y_valid = encode_y(bundle.valid, bundle)
    y_test = encode_y(bundle.test, bundle)
    if bundle.label_type == "continuous":
        mean_value = float(np.nanmean(y_train))
        return {
            "valid_pred": np.full(len(y_valid), mean_value, dtype=float),
            "test_pred": np.full(len(y_test), mean_value, dtype=float),
            "valid_metrics": compute_metrics(bundle.label_type, y_valid, np.full(len(y_valid), mean_value)),
            "test_metrics": compute_metrics(bundle.label_type, y_test, np.full(len(y_test), mean_value)),
            "model_details": {"train_mean": mean_value},
        }
    n_classes = n_classes_for_bundle(bundle)
    counts = np.bincount(y_train, minlength=n_classes).astype(float)
    priors = counts / counts.sum()
    valid_prob = np.tile(priors.reshape(1, -1), (len(y_valid), 1))
    test_prob = np.tile(priors.reshape(1, -1), (len(y_test), 1))
    return {
        "valid_prob": valid_prob,
        "test_prob": test_prob,
        "model_details": {"train_class_priors": priors.tolist(), "majority_index": int(np.argmax(priors))},
    }


def run_simple_model(bundle: SplitBundle, feature_columns: list[str], model_name: str, seed: int) -> dict[str, Any]:
    if model_name == "null":
        return run_null(bundle)

    x_train = bundle.train[feature_columns]
    x_valid = bundle.valid[feature_columns]
    x_test = bundle.test[feature_columns]
    y_train = encode_y(bundle.train, bundle)
    y_valid = encode_y(bundle.valid, bundle)
    y_test = encode_y(bundle.test, bundle)

    if model_name == "mlp":
        estimator = build_mlp_estimator(bundle.label_type, seed)
    else:
        estimator = build_estimator(
            model_name=model_name,
            label_type=bundle.label_type,
            seed=seed,
            n_classes=1 if bundle.label_type == "continuous" else n_classes_for_bundle(bundle),
        )
    estimator.fit(x_train, y_train)

    if bundle.label_type == "continuous":
        valid_pred = estimator.predict(x_valid)
        test_pred = estimator.predict(x_test)
        return {
            "valid_pred": np.asarray(valid_pred, dtype=float),
            "test_pred": np.asarray(test_pred, dtype=float),
            "valid_metrics": compute_metrics(bundle.label_type, y_valid, valid_pred),
            "test_metrics": compute_metrics(bundle.label_type, y_test, test_pred),
            "model_details": {"estimator": model_name},
        }

    valid_prob = align_probabilities(estimator, estimator.predict_proba(x_valid), n_classes_for_bundle(bundle))
    test_prob = align_probabilities(estimator, estimator.predict_proba(x_test), n_classes_for_bundle(bundle))
    return {
        "valid_prob": valid_prob,
        "test_prob": test_prob,
        "model_details": {"estimator": model_name},
    }


class SimpleMultiTaskMLP:
    def __init__(self, input_dim: int, task_specs: dict[str, dict[str, Any]], seed: int) -> None:
        import torch
        from torch import nn

        self.torch = torch
        self.nn = nn
        self.seed = seed
        self.task_order = sorted(task_specs)
        self.task_specs = task_specs
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # Keep this baseline on CPU. It is small enough, and CPU avoids losing
        # the whole process to device-side CUDA asserts when a target bug occurs.
        self.device = torch.device("cpu")

        class Net(nn.Module):
            def __init__(self, input_dim: int, task_specs: dict[str, dict[str, Any]], task_order: list[str]) -> None:
                super().__init__()
                self.shared = nn.Sequential(
                    nn.Linear(input_dim, 512),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(512, 256),
                    nn.GELU(),
                    nn.Dropout(0.1),
                )
                self.heads = nn.ModuleDict()
                for task_key in task_order:
                    out_dim = 1 if task_specs[task_key]["label_type"] == "continuous" else int(task_specs[task_key]["n_classes"])
                    self.heads[task_key] = nn.Linear(256, out_dim)

            def forward(self, x: Any, task_key: str) -> Any:
                return self.heads[task_key](self.shared(x))

        self.model = Net(input_dim, task_specs, self.task_order).to(self.device)
        self.parameter_count = sum(parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad)

    def fit_predict(self, train_frame: pd.DataFrame, valid_frame: pd.DataFrame, test_frame: pd.DataFrame, feature_columns: list[str]) -> dict[str, Any]:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        train_x_raw = train_frame[feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        valid_x_raw = valid_frame[feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        test_x_raw = test_frame[feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        means = np.nanmean(train_x_raw, axis=0)
        means = np.nan_to_num(means, nan=0.0)
        stds = np.nanstd(train_x_raw, axis=0)
        stds = np.nan_to_num(stds, nan=1.0)
        stds[stds == 0.0] = 1.0

        def transform(array: np.ndarray) -> np.ndarray:
            return np.nan_to_num((array - means) / stds, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        train_x = transform(train_x_raw)
        valid_x = transform(valid_x_raw)
        test_x = transform(test_x_raw)

        y_float_train = train_frame["target_float"].to_numpy(dtype=np.float32)
        y_index_train = train_frame["target_index"].to_numpy(dtype=np.int64)
        task_index_train = train_frame["task_index"].to_numpy(dtype=np.int64)

        dataset = TensorDataset(
            torch.as_tensor(train_x, dtype=torch.float32),
            torch.as_tensor(task_index_train, dtype=torch.long),
            torch.as_tensor(y_float_train, dtype=torch.float32),
            torch.as_tensor(y_index_train, dtype=torch.long),
        )
        loader = DataLoader(dataset, batch_size=128, shuffle=True)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        ce_losses: dict[str, Any] = {}
        for task_key, spec in self.task_specs.items():
            if spec["label_type"] == "continuous":
                continue
            counts = np.asarray(spec["train_counts"], dtype=np.float32)
            counts[counts == 0.0] = 1.0
            weights = counts.sum() / (len(counts) * counts)
            ce_losses[task_key] = self.nn.CrossEntropyLoss(
                weight=torch.as_tensor(weights, dtype=torch.float32, device=self.device)
            )
        huber = self.nn.SmoothL1Loss()

        best_state = None
        best_score = -1e9
        stale = 0
        task_index_to_key = {i: key for i, key in enumerate(self.task_order)}
        for epoch in range(1, 121):
            self.model.train()
            for batch_x, batch_task, batch_y_float, batch_y_index in loader:
                batch_x = batch_x.to(self.device)
                batch_task = batch_task.to(self.device)
                batch_y_float = batch_y_float.to(self.device)
                batch_y_index = batch_y_index.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                losses = []
                for task_idx, task_key in task_index_to_key.items():
                    mask = batch_task == int(task_idx)
                    if not torch.any(mask):
                        continue
                    logits = self.model(batch_x[mask], task_key)
                    spec = self.task_specs[task_key]
                    if spec["label_type"] == "continuous":
                        losses.append(huber(logits.squeeze(-1), batch_y_float[mask]))
                    else:
                        losses.append(ce_losses[task_key](logits, batch_y_index[mask]))
                if not losses:
                    continue
                loss = torch.stack(losses).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer.step()

            valid_outputs = self.predict_frame(valid_x, valid_frame)
            valid_scores = []
            for task_key, spec in self.task_specs.items():
                mask = valid_frame["task_key"].eq(task_key).to_numpy()
                if not np.any(mask):
                    continue
                if spec["label_type"] == "continuous":
                    metrics = compute_metrics("continuous", valid_frame.loc[mask, "y_raw"].to_numpy(dtype=float), valid_outputs[task_key]["pred"])
                    valid_scores.append(safe_float(metrics.get("r2")))
                else:
                    metrics = compute_metrics(spec["label_type"], valid_frame.loc[mask, "target_index"].to_numpy(dtype=int), valid_outputs[task_key]["pred"], valid_outputs[task_key]["prob"])
                    valid_scores.append(safe_float(metrics.get("balanced_accuracy")))
            score = float(np.nanmean(valid_scores)) if valid_scores else -1e9
            if score > best_score:
                best_score = score
                stale = 0
                best_state = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
            else:
                stale += 1
            if epoch >= 20 and stale >= 12:
                break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return {
            "valid_outputs": self.predict_frame(valid_x, valid_frame),
            "test_outputs": self.predict_frame(test_x, test_frame),
            "feature_mean": means.tolist(),
            "feature_std": stds.tolist(),
            "epochs_ran": epoch,
            "best_validation_macro_primary": best_score,
        }

    def predict_frame(self, features: np.ndarray, frame: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
        import torch

        self.model.eval()
        outputs: dict[str, dict[str, np.ndarray]] = {}
        with torch.no_grad():
            for task_key, spec in self.task_specs.items():
                mask = frame["task_key"].eq(task_key).to_numpy()
                if not np.any(mask):
                    continue
                x = torch.as_tensor(features[mask], dtype=torch.float32, device=self.device)
                logits = self.model(x, task_key).detach().cpu().numpy()
                if spec["label_type"] == "continuous":
                    pred = logits.reshape(-1)
                    y_mean = float(spec["y_mean"])
                    y_std = float(spec["y_std"])
                    outputs[task_key] = {"pred": pred * y_std + y_mean}
                else:
                    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
                    prob = exp / exp.sum(axis=1, keepdims=True)
                    outputs[task_key] = {"pred": prob.argmax(axis=1), "prob": prob}
        return outputs


def prepare_multitask_frames(dataset: str, feature_columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]]]:
    frames = []
    task_specs: dict[str, dict[str, Any]] = {}
    task_key_order = sorted([f"{dataset}::{task}" for task in PRIMARY_TASKS[dataset]])
    for task in PRIMARY_TASKS[dataset]:
        task_key = f"{dataset}::{task}"
        task_idx = task_key_order.index(task_key)
        bundle = load_task_bundle(dataset, task)
        frame = bundle.frame.copy()
        frame["task_key"] = task_key
        frame["task_index"] = task_idx
        if bundle.label_type == "continuous":
            y_train = pd.to_numeric(bundle.train["y_raw"], errors="coerce").to_numpy(dtype=float)
            y_mean = float(np.nanmean(y_train))
            y_std = float(np.nanstd(y_train))
            if not math.isfinite(y_std) or y_std == 0.0:
                y_std = 1.0
            frame["target_float"] = (pd.to_numeric(frame["y_raw"], errors="coerce").to_numpy(dtype=float) - y_mean) / y_std
            frame["target_index"] = -1
            task_specs[frame["task_key"].iloc[0]] = {
                "label_type": bundle.label_type,
                "n_classes": 1,
                "y_mean": y_mean,
                "y_std": y_std,
                "class_space": None,
                "train_counts": [],
            }
        else:
            encoded = encode_y(frame, bundle)
            train_encoded = encode_y(bundle.train, bundle)
            n_classes = n_classes_for_bundle(bundle)
            frame["target_float"] = 0.0
            frame["target_index"] = encoded
            task_specs[frame["task_key"].iloc[0]] = {
                "label_type": bundle.label_type,
                "n_classes": n_classes,
                "y_mean": 0.0,
                "y_std": 1.0,
                "class_space": (bundle.class_space.tolist() if bundle.class_space is not None else []),
                "train_counts": np.bincount(train_encoded, minlength=n_classes).tolist(),
            }
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    for column in feature_columns:
        if column not in combined.columns:
            combined[column] = np.nan
    return (
        combined.loc[combined["split"] == "train"].reset_index(drop=True),
        combined.loc[combined["split"] == "valid"].reset_index(drop=True),
        combined.loc[combined["split"] == "test"].reset_index(drop=True),
        task_specs,
    )


def prediction_base_frame(frame: pd.DataFrame, bundle: SplitBundle, split: str) -> pd.DataFrame:
    columns = ["dataset_id", "subject_id", "anchor_id", "task_name", "split"]
    output = frame[columns].copy()
    output["endpoint"] = bundle.task
    output["task_type"] = bundle.label_type
    output["y_true"] = raw_y(frame, bundle)
    if bundle.label_type != "continuous":
        output["y_true_index"] = encode_y(frame, bundle)
    output["split"] = split
    return output


def save_prediction_file(
    run_id: str,
    bundle: SplitBundle,
    valid_payload: dict[str, Any],
    test_payload: dict[str, Any],
    class_space: np.ndarray | None,
) -> Path:
    rows = []
    for split_name, frame, payload in (
        ("valid", bundle.valid, valid_payload),
        ("test", bundle.test, test_payload),
    ):
        output = prediction_base_frame(frame, bundle, split_name)
        if bundle.label_type == "continuous":
            pred = np.asarray(payload["pred"], dtype=float)
            output["raw_prediction"] = pred
            output["inverse_transformed_prediction"] = pred
            output["y_pred"] = pred
        else:
            output["y_pred_uncalibrated_index"] = payload["uncal_pred"]
            output["y_pred_calibrated_index"] = payload["cal_pred"]
            output["y_pred"] = payload["cal_pred"]
            for prefix, prob in (("uncalibrated", payload["uncal_prob"]), ("calibrated", payload["cal_prob"])):
                for idx in range(prob.shape[1]):
                    label = str(class_space[idx]) if class_space is not None and idx < len(class_space) else str(idx)
                    label = label.replace(" ", "_").replace("-", "minus")
                    output[f"proba_{prefix}_{label}"] = prob[:, idx]
        rows.append(output)
    prediction = pd.concat(rows, ignore_index=True)
    path = OUT_ROOT / "predictions" / f"{run_id}.csv"
    prediction.to_csv(path, index=False)
    return path


def counts_for_bundle(bundle: SplitBundle) -> dict[str, int]:
    return {
        "num_train_participants": int(bundle.train["subject_id"].nunique()),
        "num_val_participants": int(bundle.valid["subject_id"].nunique()),
        "num_test_participants": int(bundle.test["subject_id"].nunique()),
        "num_train_task_rows": int(len(bundle.train)),
        "num_val_task_rows": int(len(bundle.valid)),
        "num_test_task_rows": int(len(bundle.test)),
    }


def metric_row_base(
    run_id: str,
    seed: int,
    bundle: SplitBundle,
    feature_set: str,
    model_family: str,
    training_scheme: str,
    config_path: Path,
    prediction_path: Path | None,
    postprocessing_type: str,
    calibration_path: Path | None,
) -> dict[str, Any]:
    row = {
        "run_id": run_id,
        "seed": seed,
        "dataset": bundle.dataset,
        "endpoint": bundle.task,
        "endpoint_display": TASK_DISPLAY.get((bundle.dataset, bundle.task), bundle.task),
        "task_type": bundle.label_type,
        "split_manifest_id": bundle.split_manifest_id,
        "feature_set": feature_set,
        "model_family": model_family,
        "training_scheme": training_scheme,
        **counts_for_bundle(bundle),
        "postprocessing_type": postprocessing_type,
        "threshold_or_bias_path": str(calibration_path.relative_to(ROOT)) if calibration_path else "",
        "config_path": str(config_path.relative_to(ROOT)),
        "prediction_path": str(prediction_path.relative_to(ROOT)) if prediction_path else "",
        "status": "ok",
        "failure_reason": "",
    }
    for name in [
        "val_primary",
        "test_primary",
        "test_r2",
        "test_rmse",
        "test_mae",
        "test_spearman",
        "test_ba",
        "test_macro_f1",
        "test_auroc",
        "test_auprc",
        "test_brier",
        "test_ece",
        "val_primary_uncalibrated",
        "test_primary_uncalibrated",
        "test_ba_uncalibrated",
        "test_macro_f1_uncalibrated",
        "test_brier_uncalibrated",
        "test_ece_uncalibrated",
        "parameter_count",
    ]:
        row[name] = math.nan
    return row


def append_seed_row(row: dict[str, Any]) -> None:
    path = OUT_ROOT / "seed_metrics.csv"
    frame = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path, keep_default_na=False)
        combined = pd.concat([old, frame], ignore_index=True)
        combined = combined.drop_duplicates(subset=["run_id"], keep="last")
    else:
        combined = frame
    combined.to_csv(path, index=False)


def normalize_seed_frame(seed: pd.DataFrame) -> pd.DataFrame:
    seed = seed.copy()
    if "model_family" in seed.columns:
        missing = seed["model_family"].astype(str).isin(["", "nan", "None"])
        null_runs = seed["run_id"].astype(str).str.contains("__null__")
        seed.loc[missing & null_runs, "model_family"] = "null"
    if "status" in seed.columns:
        seed["status"] = seed["status"].replace("", "ok")
    return seed


def record_failure(
    run_id: str,
    seed: int,
    bundle: SplitBundle,
    feature_set: str,
    model_family: str,
    training_scheme: str,
    config_path: Path,
    exc: BaseException,
) -> None:
    log_path = OUT_ROOT / "logs" / f"{run_id}.error.log"
    log_path.write_text("".join(traceback.format_exception(exc)), encoding="utf-8")
    row = metric_row_base(
        run_id,
        seed,
        bundle,
        feature_set,
        model_family,
        training_scheme,
        config_path,
        None,
        "failed",
        log_path,
    )
    row["status"] = "failed"
    row["failure_reason"] = f"{type(exc).__name__}: {exc}"
    append_seed_row(row)


def finalize_and_record_simple(
    run_id: str,
    seed: int,
    bundle: SplitBundle,
    feature_set: str,
    model_family: str,
    training_scheme: str,
    config_path: Path,
    outputs: dict[str, Any],
) -> None:
    calibration_path = OUT_ROOT / "calibration" / f"{run_id}.json"
    if bundle.label_type == "continuous":
        y_valid = encode_y(bundle.valid, bundle)
        y_test = encode_y(bundle.test, bundle)
        valid_metrics = compute_metrics("continuous", y_valid, outputs["valid_pred"])
        test_metrics = compute_metrics("continuous", y_test, outputs["test_pred"])
        prediction_path = save_prediction_file(
            run_id,
            bundle,
            {"pred": outputs["valid_pred"]},
            {"pred": outputs["test_pred"]},
            None,
        )
        write_json(calibration_path, {"postprocessing_type": "none_regression", "details": outputs.get("model_details", {})})
        row = metric_row_base(
            run_id,
            seed,
            bundle,
            feature_set,
            model_family,
            training_scheme,
            config_path,
            prediction_path,
            "none_regression",
            calibration_path,
        )
        row["val_primary"] = safe_float(valid_metrics.get("r2"))
        row["test_primary"] = safe_float(test_metrics.get("r2"))
        row["test_r2"] = safe_float(test_metrics.get("r2"))
        row["test_rmse"] = safe_float(test_metrics.get("rmse"))
        row["test_mae"] = safe_float(test_metrics.get("mae"))
        row["test_spearman"] = safe_float(test_metrics.get("spearman"))
        append_seed_row(row)
        return

    y_valid = encode_y(bundle.valid, bundle)
    y_test = encode_y(bundle.test, bundle)
    post = apply_validation_postprocessing(bundle.label_type, y_valid, outputs["valid_prob"], y_test, outputs["test_prob"])
    write_json(
        calibration_path,
        {
            "postprocessing_type": post["postprocessing_type"],
            "threshold": post["threshold"],
            "biases": post["biases"],
            "temperature": post["temperature"],
            "temperature_validation_diagnostics": post["temperature_validation_diagnostics"],
            "validation_primary_uncalibrated": post["uncalibrated_valid_metrics"]["ba"],
            "validation_primary_calibrated": post["calibrated_valid_metrics"]["ba"],
        },
    )
    prediction_path = save_prediction_file(
        run_id,
        bundle,
        {
            "uncal_pred": post["uncalibrated_valid_pred"],
            "cal_pred": post["calibrated_valid_pred"],
            "uncal_prob": post["uncalibrated_valid_prob"],
            "cal_prob": post["calibrated_valid_prob"],
        },
        {
            "uncal_pred": post["uncalibrated_test_pred"],
            "cal_pred": post["calibrated_test_pred"],
            "uncal_prob": post["uncalibrated_test_prob"],
            "cal_prob": post["calibrated_test_prob"],
        },
        bundle.class_space,
    )
    row = metric_row_base(
        run_id,
        seed,
        bundle,
        feature_set,
        model_family,
        training_scheme,
        config_path,
        prediction_path,
        post["postprocessing_type"],
        calibration_path,
    )
    row["val_primary"] = post["calibrated_valid_metrics"]["ba"]
    row["test_primary"] = post["calibrated_test_metrics"]["ba"]
    row["test_ba"] = post["calibrated_test_metrics"]["ba"]
    row["test_macro_f1"] = post["calibrated_test_metrics"]["macro_f1"]
    row["test_auroc"] = post["calibrated_test_metrics"]["auroc"]
    row["test_auprc"] = post["calibrated_test_metrics"]["auprc"]
    row["test_brier"] = post["calibrated_test_metrics"]["brier"]
    row["test_ece"] = post["calibrated_test_metrics"]["ece"]
    row["val_primary_uncalibrated"] = post["uncalibrated_valid_metrics"]["ba"]
    row["test_primary_uncalibrated"] = post["uncalibrated_test_metrics"]["ba"]
    row["test_ba_uncalibrated"] = post["uncalibrated_test_metrics"]["ba"]
    row["test_macro_f1_uncalibrated"] = post["uncalibrated_test_metrics"]["macro_f1"]
    row["test_brier_uncalibrated"] = post["uncalibrated_test_metrics"]["brier"]
    row["test_ece_uncalibrated"] = post["uncalibrated_test_metrics"]["ece"]
    append_seed_row(row)


def run_restricted_feature_experiments(args: argparse.Namespace) -> None:
    dependencies = probe_dependencies()
    write_json(OUT_ROOT / "logs" / "dependency_probe.json", dependencies)
    for dataset, tasks in PRIMARY_TASKS.items():
        for task in tasks:
            bundle = load_task_bundle(dataset, task)
            for feature_set in args.feature_sets:
                feature_cols = feature_columns_for_set(dataset, bundle.frame, feature_set)
                kept, dropped, used_constant = drop_all_nan_train_columns(bundle, feature_cols)
                for model_name in args.models:
                    if model_name == "lightgbm" and not dependencies.get("lightgbm", False):
                        missing = RuntimeError("Missing dependency: lightgbm")
                    elif model_name == "xgboost" and not dependencies.get("xgboost", False):
                        missing = RuntimeError("Missing dependency: xgboost")
                    elif model_name == "ebm" and not dependencies.get("interpret", False):
                        missing = RuntimeError("Missing dependency: interpret")
                    else:
                        missing = None
                    for seed in args.seeds:
                        run_id = f"rfv1__{dataset}__{task}__{feature_set}__{model_name}__seed{seed}"
                        config_path = OUT_ROOT / "model_configs" / f"{run_id}.json"
                        config = {
                            "run_id": run_id,
                            "seed": seed,
                            "dataset": dataset,
                            "task": task,
                            "feature_set": feature_set,
                            "model_family": model_name,
                            "training_scheme": "endpoint_wise",
                            "split_manifest": str((WINDOW_ROOT / dataset / "splits.json").relative_to(ROOT)),
                            "candidate_feature_columns": feature_cols,
                            "kept_feature_columns": kept,
                            "dropped_all_nan_train_columns": dropped,
                            "used_constant_feature_fallback": used_constant,
                            "dependency_probe": dependencies,
                            "test_used_for_selection": False,
                        }
                        write_json(config_path, config)
                        if missing is not None:
                            record_failure(run_id, seed, bundle, feature_set, model_name, "endpoint_wise", config_path, missing)
                            continue
                        if args.resume and (OUT_ROOT / "predictions" / f"{run_id}.csv").exists():
                            continue
                        try:
                            print(f"[restricted] {run_id}", flush=True)
                            outputs = run_simple_model(bundle, kept, model_name, seed)
                            finalize_and_record_simple(
                                run_id,
                                seed,
                                bundle,
                                feature_set,
                                model_name,
                                "endpoint_wise",
                                config_path,
                                outputs,
                            )
                        except BaseException as exc:  # preserve failed config/logs
                            record_failure(run_id, seed, bundle, feature_set, model_name, "endpoint_wise", config_path, exc)


def run_simple_multitask_experiments(args: argparse.Namespace) -> None:
    for dataset in PRIMARY_TASKS:
        base_bundle = load_task_bundle(dataset, PRIMARY_TASKS[dataset][0])
        feature_cols = feature_columns_for_set(dataset, base_bundle.frame, "FULL")
        kept, dropped, used_constant = drop_all_nan_train_columns(base_bundle, feature_cols)
        train_frame, valid_frame, test_frame, task_specs = prepare_multitask_frames(dataset, kept)
        for seed in args.seeds:
            run_id_base = f"rfv1__{dataset}__FULL__simple_multitask_mlp__seed{seed}"
            config_path = OUT_ROOT / "model_configs" / f"{run_id_base}.json"
            config = {
                "run_id": run_id_base,
                "seed": seed,
                "dataset": dataset,
                "tasks": PRIMARY_TASKS[dataset],
                "feature_set": "FULL",
                "model_family": "simple_multitask_mlp",
                "training_scheme": "multi_task",
                "split_manifest": str((WINDOW_ROOT / dataset / "splits.json").relative_to(ROOT)),
                "kept_feature_columns": kept,
                "dropped_all_nan_train_columns": dropped,
                "used_constant_feature_fallback": used_constant,
                "architecture": "shared MLP encoder 512-256 with task-specific linear heads; no recursive decoder; no expert multiclass head",
                "test_used_for_selection": False,
            }
            write_json(config_path, config)
            try:
                print(f"[simple-multitask] {run_id_base}", flush=True)
                model = SimpleMultiTaskMLP(input_dim=len(kept), task_specs=task_specs, seed=seed)
                outputs = model.fit_predict(train_frame, valid_frame, test_frame, kept)
                config["parameter_count"] = model.parameter_count
                config["epochs_ran"] = outputs["epochs_ran"]
                config["best_validation_macro_primary"] = outputs["best_validation_macro_primary"]
                write_json(config_path, config)
                for task in PRIMARY_TASKS[dataset]:
                    bundle = load_task_bundle(dataset, task)
                    task_key = f"{dataset}::{task}"
                    run_id = f"rfv1__{dataset}__{task}__FULL__simple_multitask_mlp__seed{seed}"
                    task_config_path = OUT_ROOT / "model_configs" / f"{run_id}.json"
                    write_json(task_config_path, {**config, "run_id": run_id, "task": task})
                    if bundle.label_type == "continuous":
                        valid_pred = outputs["valid_outputs"][task_key]["pred"]
                        test_pred = outputs["test_outputs"][task_key]["pred"]
                        payload = {"valid_pred": valid_pred, "test_pred": test_pred, "model_details": {"parameter_count": model.parameter_count}}
                    else:
                        payload = {
                            "valid_prob": outputs["valid_outputs"][task_key]["prob"],
                            "test_prob": outputs["test_outputs"][task_key]["prob"],
                            "model_details": {"parameter_count": model.parameter_count},
                        }
                    finalize_and_record_simple(
                        run_id,
                        seed,
                        bundle,
                        "FULL",
                        "simple_multitask_mlp",
                        "multi_task",
                        task_config_path,
                        payload,
                    )
                    seed_path = OUT_ROOT / "seed_metrics.csv"
                    metrics = normalize_seed_frame(pd.read_csv(seed_path, keep_default_na=False))
                    metrics.loc[metrics["run_id"] == run_id, "parameter_count"] = model.parameter_count
                    metrics.to_csv(seed_path, index=False)
            except BaseException as exc:
                for task in PRIMARY_TASKS[dataset]:
                    bundle = load_task_bundle(dataset, task)
                    record_failure(
                        f"rfv1__{dataset}__{task}__FULL__simple_multitask_mlp__seed{seed}",
                        seed,
                        bundle,
                        "FULL",
                        "simple_multitask_mlp",
                        "multi_task",
                        config_path,
                        exc,
                    )


def mctrcm_run_name(model_kind: str, dataset: str, task: str, seed: int) -> str:
    if model_kind == "locked":
        return f"final_mctrcm_{dataset}_seed{seed}"
    if model_kind == "single_task":
        return f"single_task_mctrcm_{dataset}_{task}_seed{seed}"
    if model_kind == "k1":
        return f"revised_k1_plain_{dataset}_seed{seed}"
    raise KeyError(model_kind)


def ingest_mctrcm_predictions(args: argparse.Namespace) -> None:
    kind_specs = {
        "locked": ("mctrcm_locked_k4", "multi_task_locked"),
        "single_task": ("mctrcm_locked_k4", "single_task"),
        "k1": ("mctrcm_k1_plain", "multi_task_k1_plain"),
    }
    for model_kind in args.mctrcm_kinds:
        model_family, training_scheme = kind_specs[model_kind]
        for dataset, tasks in PRIMARY_TASKS.items():
            for task in tasks:
                bundle = load_task_bundle(dataset, task)
                for seed in args.seeds:
                    run_name = mctrcm_run_name(model_kind, dataset, task, seed)
                    valid_path = MCTRCM_PRED_ROOT / f"{run_name}__valid.csv"
                    test_path = MCTRCM_PRED_ROOT / f"{run_name}__test.csv"
                    config_source = MCTRCM_LOG_ROOT / f"{run_name}__config.json"
                    summary_source = MCTRCM_LOG_ROOT / f"{run_name}__summary.json"
                    run_id = f"rfv1__{dataset}__{task}__FULL__{model_kind}_mctrcm__seed{seed}"
                    config_path = OUT_ROOT / "model_configs" / f"{run_id}.json"
                    if not valid_path.exists() or not test_path.exists():
                        write_json(
                            config_path,
                            {
                                "run_id": run_id,
                                "source_run_name": run_name,
                                "source_valid_path": str(valid_path.relative_to(ROOT)) if valid_path.exists() else str(valid_path),
                                "source_test_path": str(test_path.relative_to(ROOT)) if test_path.exists() else str(test_path),
                                "status": "missing_source_predictions",
                                "instructions": "Run scripts/run_core_mctrcm_protocol.py for single_task/k1 predictions before ingestion.",
                            },
                        )
                        record_failure(
                            run_id,
                            seed,
                            bundle,
                            "FULL",
                            model_family,
                            training_scheme,
                            config_path,
                            FileNotFoundError(f"Missing MC-TRCM source predictions for {run_name}"),
                        )
                        continue
                    source_config = read_json(config_source) if config_source.exists() else {}
                    source_summary = read_json(summary_source) if summary_source.exists() else {}
                    write_json(
                        config_path,
                        {
                            "run_id": run_id,
                            "source_run_name": run_name,
                            "model_kind": model_kind,
                            "model_family": model_family,
                            "training_scheme": training_scheme,
                            "feature_set": "FULL",
                            "source_config": source_config,
                            "source_summary": source_summary,
                            "unified_postprocessing": True,
                            "test_used_for_selection": False,
                        },
                    )
                    try:
                        valid_all = pd.read_csv(valid_path)
                        test_all = pd.read_csv(test_path)
                        valid = valid_all.loc[valid_all["task_name"].astype(str).eq(task)].reset_index(drop=True)
                        test = test_all.loc[test_all["task_name"].astype(str).eq(task)].reset_index(drop=True)
                        if valid.empty or test.empty:
                            raise ValueError(f"No rows for task {task} in source predictions {run_name}")
                        if bundle.label_type == "continuous":
                            valid_pred = pd.to_numeric(valid["y_pred"], errors="coerce").to_numpy(dtype=float)
                            test_pred = pd.to_numeric(test["y_pred"], errors="coerce").to_numpy(dtype=float)
                            finalize_and_record_simple(
                                run_id,
                                seed,
                                bundle,
                                "FULL",
                                model_family,
                                training_scheme,
                                config_path,
                                {"valid_pred": valid_pred, "test_pred": test_pred, "model_details": {"source_run_name": run_name}},
                            )
                        else:
                            proba_cols = [column for column in valid.columns if column.startswith("proba_")]
                            valid_prob = valid[proba_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
                            test_prob = test[proba_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
                            finalize_and_record_simple(
                                run_id,
                                seed,
                                bundle,
                                "FULL",
                                model_family,
                                training_scheme,
                                config_path,
                                {"valid_prob": valid_prob, "test_prob": test_prob, "model_details": {"source_run_name": run_name}},
                            )
                        if source_summary.get("parameter_count") is not None:
                            seed_path = OUT_ROOT / "seed_metrics.csv"
                            metrics = normalize_seed_frame(pd.read_csv(seed_path, keep_default_na=False))
                            metrics.loc[metrics["run_id"] == run_id, "parameter_count"] = int(source_summary["parameter_count"])
                            metrics.to_csv(seed_path, index=False)
                    except BaseException as exc:
                        record_failure(run_id, seed, bundle, "FULL", model_family, training_scheme, config_path, exc)


def build_summary() -> pd.DataFrame:
    seed_path = OUT_ROOT / "seed_metrics.csv"
    if not seed_path.exists():
        return pd.DataFrame()
    seed = normalize_seed_frame(pd.read_csv(seed_path, keep_default_na=False))
    ok = seed.loc[seed["status"].eq("ok")].copy()
    groups = ["dataset", "endpoint", "feature_set", "model_family", "training_scheme"]
    rows = []
    for key, group in ok.groupby(groups, dropna=False):
        row = dict(zip(groups, key))
        row["mean_val_primary"] = float(pd.to_numeric(group["val_primary"], errors="coerce").mean())
        row["se_val_primary"] = standard_error(group["val_primary"])
        row["mean_test_primary"] = float(pd.to_numeric(group["test_primary"], errors="coerce").mean())
        row["se_test_primary"] = standard_error(group["test_primary"])
        secondary = {}
        for metric in [
            "test_r2",
            "test_rmse",
            "test_mae",
            "test_spearman",
            "test_ba",
            "test_macro_f1",
            "test_auroc",
            "test_auprc",
            "test_brier",
            "test_ece",
            "test_ba_uncalibrated",
            "test_brier_uncalibrated",
            "test_ece_uncalibrated",
            "parameter_count",
        ]:
            value = pd.to_numeric(group.get(metric), errors="coerce")
            if value.notna().any():
                secondary[metric] = float(value.mean())
        row["mean_test_secondary_metrics"] = json.dumps(secondary, sort_keys=True)
        row["n_seeds"] = int(group["seed"].nunique())
        rows.append(row)
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary

    def lookup(dataset: str, endpoint: str, feature_set: str, model_family: str, scheme: str) -> float:
        match = summary.loc[
            summary["dataset"].eq(dataset)
            & summary["endpoint"].eq(endpoint)
            & summary["feature_set"].eq(feature_set)
            & summary["model_family"].eq(model_family)
            & summary["training_scheme"].eq(scheme),
            "mean_test_primary",
        ]
        return float(match.iloc[0]) if not match.empty else math.nan

    deltas = []
    for _, row in summary.iterrows():
        null_value = lookup(row["dataset"], row["endpoint"], row["feature_set"], "null", "endpoint_wise")
        sensor_value = lookup(row["dataset"], row["endpoint"], "SENSOR_ONLY", row["model_family"], row["training_scheme"])
        symptom_value = lookup(row["dataset"], row["endpoint"], "SYMPTOM_STATIC_CLINICAL", row["model_family"], row["training_scheme"])
        deltas.append(
            {
                "delta_vs_null": row["mean_test_primary"] - null_value if not pd.isna(null_value) else math.nan,
                "delta_vs_sensor_only": row["mean_test_primary"] - sensor_value if not pd.isna(sensor_value) else math.nan,
                "delta_vs_symptom_static_clinical": row["mean_test_primary"] - symptom_value if not pd.isna(symptom_value) else math.nan,
            }
        )
    summary = pd.concat([summary.reset_index(drop=True), pd.DataFrame(deltas)], axis=1)
    summary.to_csv(OUT_ROOT / "summary_metrics.csv", index=False)
    return summary


def best_by_validation(summary: pd.DataFrame, dataset: str, endpoint: str, feature_set: str, model_families: list[str]) -> pd.Series | None:
    subset = summary.loc[
        summary["dataset"].eq(dataset)
        & summary["endpoint"].eq(endpoint)
        & summary["feature_set"].eq(feature_set)
        & summary["model_family"].isin(model_families)
        & summary["n_seeds"].ge(1)
    ].copy()
    if subset.empty:
        return None
    subset = subset.sort_values(["mean_val_primary", "mean_test_primary"], ascending=[False, False])
    return subset.iloc[0]


def metric_string(row: pd.Series | None) -> str:
    if row is None:
        return "--"
    return fmt_metric(safe_float(row.get("mean_test_primary")), safe_float(row.get("se_test_primary")))


def build_uncalibrated_baseline_summary() -> pd.DataFrame:
    seed_path = OUT_ROOT / "seed_metrics.csv"
    if not seed_path.exists():
        return pd.DataFrame()
    seed = normalize_seed_frame(pd.read_csv(seed_path, keep_default_na=False))
    seed = seed.loc[
        seed["status"].eq("ok")
        & seed["feature_set"].eq("FULL")
        & seed["training_scheme"].eq("endpoint_wise")
        & seed["model_family"].isin(["elastic_net", "lightgbm", "xgboost", "ebm", "mlp"])
    ].copy()
    if seed.empty:
        return pd.DataFrame()
    seed["val_primary_uncalibrated_effective"] = pd.to_numeric(seed["val_primary_uncalibrated"], errors="coerce")
    missing_val = seed["val_primary_uncalibrated_effective"].isna()
    seed.loc[missing_val, "val_primary_uncalibrated_effective"] = pd.to_numeric(seed.loc[missing_val, "val_primary"], errors="coerce")
    seed["test_primary_uncalibrated_effective"] = pd.to_numeric(seed["test_primary_uncalibrated"], errors="coerce")
    missing_test = seed["test_primary_uncalibrated_effective"].isna()
    seed.loc[missing_test, "test_primary_uncalibrated_effective"] = pd.to_numeric(seed.loc[missing_test, "test_primary"], errors="coerce")
    rows = []
    for key, group in seed.groupby(["dataset", "endpoint", "model_family", "training_scheme"], dropna=False):
        rows.append(
            {
                "dataset": key[0],
                "endpoint": key[1],
                "model_family": key[2],
                "training_scheme": key[3],
                "mean_val_primary": float(group["val_primary_uncalibrated_effective"].mean()),
                "mean_test_primary": float(group["test_primary_uncalibrated_effective"].mean()),
                "se_test_primary": standard_error(group["test_primary_uncalibrated_effective"]),
                "n_seeds": int(group["seed"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def best_uncalibrated_baseline(uncalibrated: pd.DataFrame, dataset: str, endpoint: str) -> pd.Series | None:
    if uncalibrated.empty:
        return None
    subset = uncalibrated.loc[uncalibrated["dataset"].eq(dataset) & uncalibrated["endpoint"].eq(endpoint)].copy()
    if subset.empty:
        return None
    subset = subset.sort_values(["mean_val_primary", "mean_test_primary"], ascending=[False, False])
    return subset.iloc[0]


def latex_escape(value: object) -> str:
    return str(value).replace("\\", "\\textbackslash{}").replace("_", "\\_")


def generate_latex_tables(summary: pd.DataFrame) -> None:
    simple_non_null = ["elastic_net", "lightgbm", "xgboost", "ebm", "mlp"]
    rows = []
    for dataset, tasks in PRIMARY_TASKS.items():
        for endpoint in tasks:
            null_row = best_by_validation(summary, dataset, endpoint, "FULL", ["null"])
            sensor_best = best_by_validation(summary, dataset, endpoint, "SENSOR_ONLY", simple_non_null)
            symptom_best = best_by_validation(summary, dataset, endpoint, "SYMPTOM_STATIC_CLINICAL", simple_non_null)
            full_best = best_by_validation(summary, dataset, endpoint, "FULL", simple_non_null)
            locked_match = summary.loc[
                summary["dataset"].eq(dataset)
                & summary["endpoint"].eq(endpoint)
                & summary["feature_set"].eq("FULL")
                & summary["model_family"].eq("mctrcm_locked_k4")
                & summary["training_scheme"].eq("multi_task_locked")
            ]
            locked = locked_match.iloc[0] if not locked_match.empty else None
            single = summary.loc[
                summary["dataset"].eq(dataset)
                & summary["endpoint"].eq(endpoint)
                & summary["feature_set"].eq("FULL")
                & summary["model_family"].eq("mctrcm_locked_k4")
                & summary["training_scheme"].eq("single_task")
            ]
            single_row = single.iloc[0] if not single.empty else None
            multitask = best_by_validation(summary, dataset, endpoint, "FULL", ["simple_multitask_mlp"])
            rows.append(
                [
                    TASK_DISPLAY[(dataset, endpoint)],
                    metric_string(null_row),
                    f"{latex_escape(sensor_best['model_family'])} {metric_string(sensor_best)}" if sensor_best is not None else "--",
                    f"{latex_escape(symptom_best['model_family'])} {metric_string(symptom_best)}" if symptom_best is not None else "--",
                    f"{latex_escape(full_best['model_family'])} {metric_string(full_best)}" if full_best is not None else "--",
                    metric_string(locked),
                    metric_string(single_row),
                    metric_string(multitask),
                ]
            )
    write_restricted_table(rows)

    uncalibrated_baselines = build_uncalibrated_baseline_summary()
    rows = []
    for dataset, tasks in PRIMARY_TASKS.items():
        for endpoint in tasks:
            full_uncal_best = best_uncalibrated_baseline(uncalibrated_baselines, dataset, endpoint)
            full_cal_best = best_by_validation(summary, dataset, endpoint, "FULL", simple_non_null)
            single = summary.loc[
                summary["dataset"].eq(dataset)
                & summary["endpoint"].eq(endpoint)
                & summary["feature_set"].eq("FULL")
                & summary["model_family"].eq("mctrcm_locked_k4")
                & summary["training_scheme"].eq("single_task")
            ]
            locked = summary.loc[
                summary["dataset"].eq(dataset)
                & summary["endpoint"].eq(endpoint)
                & summary["feature_set"].eq("FULL")
                & summary["model_family"].eq("mctrcm_locked_k4")
                & summary["training_scheme"].eq("multi_task_locked")
            ]
            simple_mt = summary.loc[
                summary["dataset"].eq(dataset)
                & summary["endpoint"].eq(endpoint)
                & summary["feature_set"].eq("FULL")
                & summary["model_family"].eq("simple_multitask_mlp")
            ]
            k1 = summary.loc[
                summary["dataset"].eq(dataset)
                & summary["endpoint"].eq(endpoint)
                & summary["feature_set"].eq("FULL")
                & summary["model_family"].eq("mctrcm_k1_plain")
            ]
            rows.append(
                [
                    TASK_DISPLAY[(dataset, endpoint)],
                    f"{latex_escape(full_uncal_best['model_family'])} {metric_string(full_uncal_best)}" if full_uncal_best is not None else "--",
                    f"{latex_escape(full_cal_best['model_family'])} {metric_string(full_cal_best)}" if full_cal_best is not None else "--",
                    metric_string(single.iloc[0] if not single.empty else None),
                    metric_string(locked.iloc[0] if not locked.empty else None),
                    metric_string(simple_mt.iloc[0] if not simple_mt.empty else None),
                    metric_string(k1.iloc[0] if not k1.empty else None),
                ]
            )
    write_fairness_table(rows)
    write_calibration_table()


def write_restricted_table(rows: list[list[str]]) -> None:
    lines = [
        "\\begin{table}[!tbp]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2pt}",
        "\\caption{Revised restricted-feature primary endpoint results. Values are mean $\\pm$ SE over five seeds; model choices for best simple baselines use validation primary metric only.}",
        "\\label{tab:restricted_feature_primary}",
        "\\begin{tabularx}{\\linewidth}{@{}L{0.23\\linewidth}C{0.10\\linewidth}C{0.15\\linewidth}C{0.15\\linewidth}C{0.14\\linewidth}C{0.10\\linewidth}C{0.10\\linewidth}C{0.12\\linewidth}@{}}",
        "\\toprule",
        "Endpoint & Null & Sensor-only best & Symptom/static/clinical best & Full best simple & Full MC-TRCM & Single-task MC-TRCM & Simple multi-task \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(row) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabularx}", "\\end{table}", ""])
    (TABLE_ROOT / "restricted_feature_primary.tex").write_text("\n".join(lines), encoding="utf-8")


def write_fairness_table(rows: list[list[str]]) -> None:
    lines = [
        "\\begin{table}[!tbp]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2pt}",
        "\\caption{Revised fairness comparison on full features. The first baseline column uses uncalibrated validation primary metric; the second uses the unified validation-calibrated primary metric. Values are test mean $\\pm$ SE over five seeds.}",
        "\\label{tab:fairness_primary_comparison}",
        "\\begin{tabularx}{\\linewidth}{@{}L{0.24\\linewidth}C{0.15\\linewidth}C{0.15\\linewidth}C{0.12\\linewidth}C{0.12\\linewidth}C{0.12\\linewidth}C{0.12\\linewidth}@{}}",
        "\\toprule",
        "Endpoint & Endpoint-wise best baseline & Validation-calibrated best baseline & Single-task MC-TRCM & Multi-task MC-TRCM & Simple multi-task & K=1 model \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(row) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabularx}", "\\end{table}", ""])
    (TABLE_ROOT / "fairness_primary_comparison.tex").write_text("\n".join(lines), encoding="utf-8")


def write_calibration_table() -> None:
    seed_path = OUT_ROOT / "seed_metrics.csv"
    if not seed_path.exists():
        return
    seed = normalize_seed_frame(pd.read_csv(seed_path, keep_default_na=False))
    cls = seed.loc[seed["task_type"].isin(CLASSIFICATION_TYPES) & seed["status"].eq("ok")].copy()
    if cls.empty:
        return
    rows = []
    keep_models = ["elastic_net", "lightgbm", "xgboost", "ebm", "mlp", "mctrcm_locked_k4", "simple_multitask_mlp", "mctrcm_k1_plain"]
    for (dataset, endpoint, feature_set, model_family, training_scheme), group in cls.groupby(
        ["dataset", "endpoint", "feature_set", "model_family", "training_scheme"]
    ):
        if model_family not in keep_models or feature_set != "FULL":
            continue
        def numeric_metric(column: str) -> tuple[float, float]:
            values = pd.to_numeric(group[column], errors="coerce")
            return float(values.mean()), standard_error(values)

        ba_uncal_mean, ba_uncal_se = numeric_metric("test_ba_uncalibrated")
        ba_cal_mean, ba_cal_se = numeric_metric("test_ba")
        brier_uncal_mean, brier_uncal_se = numeric_metric("test_brier_uncalibrated")
        brier_cal_mean, brier_cal_se = numeric_metric("test_brier")
        ece_uncal_mean, ece_uncal_se = numeric_metric("test_ece_uncalibrated")
        ece_cal_mean, ece_cal_se = numeric_metric("test_ece")
        rows.append(
            [
                TASK_DISPLAY.get((dataset, endpoint), endpoint),
                latex_escape(f"{model_family}/{training_scheme}"),
                fmt_metric(ba_uncal_mean, ba_uncal_se),
                fmt_metric(ba_cal_mean, ba_cal_se),
                fmt_metric(brier_uncal_mean, brier_uncal_se),
                fmt_metric(brier_cal_mean, brier_cal_se),
                fmt_metric(ece_uncal_mean, ece_uncal_se),
                fmt_metric(ece_cal_mean, ece_cal_se),
            ]
        )
    lines = [
        "\\begin{table}[!tbp]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2pt}",
        "\\caption{Unified validation-only calibration diagnostics for primary classification endpoints.}",
        "\\label{tab:calibration_fair_classification}",
        "\\begin{tabularx}{\\linewidth}{@{}L{0.21\\linewidth}L{0.20\\linewidth}C{0.09\\linewidth}C{0.09\\linewidth}C{0.09\\linewidth}C{0.09\\linewidth}C{0.09\\linewidth}C{0.09\\linewidth}@{}}",
        "\\toprule",
        "Endpoint & Model & BA uncal. & BA cal. & Brier uncal. & Brier cal. & ECE uncal. & ECE cal. \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(row) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabularx}", "\\end{table}", ""])
    (TABLE_ROOT / "calibration_fair_classification.tex").write_text("\n".join(lines), encoding="utf-8")


def table_value(summary: pd.DataFrame, dataset: str, endpoint: str, feature_set: str, models: list[str]) -> tuple[str, float]:
    row = best_by_validation(summary, dataset, endpoint, feature_set, models)
    if row is None:
        return "--", math.nan
    return str(row["model_family"]), float(row["mean_test_primary"])


def generate_report(summary: pd.DataFrame) -> None:
    lines = [
        "# Revised Fairness v1 Experiment Report",
        "",
        f"Generated: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "All new artifacts are under `results/revised_fairness_v1/`; locked benchmark exports were read only and not overwritten.",
        "",
        "## Protocol Checks",
        "",
        "- Reused existing participant-level manifests from `data_interim/window_tables/*/splits.json`.",
        "- Restricted-feature simple baselines use train-only imputation/scaling and validation-only model family/calibration selection.",
        "- Classification post-processing fits temperature, binary thresholds, or class-bias parameters on validation labels only.",
        "- `SENSOR_ONLY` includes sensor missingness summaries; `SENSOR_VALUES_ONLY` removes them.",
        "- Failed or missing-source runs are retained in `seed_metrics.csv` with logs in `results/revised_fairness_v1/logs/`.",
        "",
        "## Answers",
        "",
    ]
    simple = ["elastic_net", "lightgbm", "xgboost", "ebm", "mlp"]
    answer_rows = []
    for dataset, tasks in PRIMARY_TASKS.items():
        for endpoint in tasks:
            full_name, full_value = table_value(summary, dataset, endpoint, "FULL", simple)
            sensor_name, sensor_value = table_value(summary, dataset, endpoint, "SENSOR_ONLY", simple)
            symptom_name, symptom_value = table_value(summary, dataset, endpoint, "SYMPTOM_STATIC_CLINICAL", simple)
            null_name, null_value = table_value(summary, dataset, endpoint, "FULL", ["null"])
            locked = summary.loc[
                summary["dataset"].eq(dataset)
                & summary["endpoint"].eq(endpoint)
                & summary["model_family"].eq("mctrcm_locked_k4")
                & summary["training_scheme"].eq("multi_task_locked")
            ]
            single = summary.loc[
                summary["dataset"].eq(dataset)
                & summary["endpoint"].eq(endpoint)
                & summary["model_family"].eq("mctrcm_locked_k4")
                & summary["training_scheme"].eq("single_task")
            ]
            simple_mt = summary.loc[
                summary["dataset"].eq(dataset)
                & summary["endpoint"].eq(endpoint)
                & summary["model_family"].eq("simple_multitask_mlp")
            ]
            k1 = summary.loc[
                summary["dataset"].eq(dataset)
                & summary["endpoint"].eq(endpoint)
                & summary["model_family"].eq("mctrcm_k1_plain")
            ]
            answer_rows.append(
                {
                    "endpoint": TASK_DISPLAY[(dataset, endpoint)],
                    "full_best": full_value,
                    "full_best_model": full_name,
                    "sensor_best": sensor_value,
                    "sensor_best_model": sensor_name,
                    "symptom_static_best": symptom_value,
                    "symptom_static_model": symptom_name,
                    "null": null_value,
                    "locked_mctrcm": float(locked["mean_test_primary"].iloc[0]) if not locked.empty else math.nan,
                    "single_task_mctrcm": float(single["mean_test_primary"].iloc[0]) if not single.empty else math.nan,
                    "simple_multitask": float(simple_mt["mean_test_primary"].iloc[0]) if not simple_mt.empty else math.nan,
                    "k1": float(k1["mean_test_primary"].iloc[0]) if not k1.empty else math.nan,
                }
            )
    answer = pd.DataFrame(answer_rows)
    answer.to_csv(OUT_ROOT / "report_endpoint_matrix.csv", index=False)

    def count_relation(left: str, right: str, margin: float = 0.02) -> str:
        valid = answer[[left, right]].dropna()
        if valid.empty:
            return "insufficient completed runs"
        wins = int((valid[left] > valid[right] + margin).sum())
        ties = int((abs(valid[left] - valid[right]) <= margin).sum())
        losses = int((valid[left] < valid[right] - margin).sum())
        return f"{wins} clearly higher, {ties} within +/-{margin:.2f}, {losses} lower across {len(valid)} endpoints"

    # Build the answer section with ASCII-only text for portable logs.
    answer_start = lines.index("## Answers")
    lines = lines[: answer_start + 2]
    lines.extend(
        [
            f"1. Is FULL clearly better than SENSOR_ONLY? {count_relation('full_best', 'sensor_best')}.",
            f"2. Is SYMPTOM_STATIC_CLINICAL already close to or stronger than FULL? {count_relation('symptom_static_best', 'full_best')}.",
            f"3. Is SENSOR_ONLY better than null? {count_relation('sensor_best', 'null')}.",
            f"4. Does the MC-TRCM advantage persist in the single-task setting? {count_relation('single_task_mctrcm', 'full_best')}; locked multi-task MC-TRCM vs full best simple baseline: {count_relation('locked_mctrcm', 'full_best')}.",
            f"5. Does the simple multi-task baseline approach or exceed MC-TRCM? Simple multi-task vs locked MC-TRCM: {count_relation('simple_multitask', 'locked_mctrcm')}.",
            "6. Do validation-calibrated baselines change the main conclusions? See `tables/final/calibration_fair_classification.tex`; classification rows compare uncalibrated and calibrated BA/Brier/ECE under the same validation-only post-processing.",
            f"7. Is K=1 better than or close to locked K=4 under this validation-selected protocol? K=1 vs locked K=4: {count_relation('k1', 'locked_mctrcm')}.",
            "8. Which endpoint conclusions need rewriting? Endpoints where SENSOR_ONLY is near null or SYMPTOM_STATIC_CLINICAL is near FULL should be framed as context-feature prediction rather than behavior-only sensing; inspect `report_endpoint_matrix.csv` for endpoint-level values.",
            "9. Which results are unstable or failed? Current `seed_metrics.csv` has no failed runs; use seed SE and `report_endpoint_matrix.csv` to identify unstable endpoint-level patterns.",
            "",
            "## Endpoint Matrix",
            "",
            answer.to_markdown(index=False),
            "",
        ]
    )

    failures = pd.DataFrame()
    seed_path = OUT_ROOT / "seed_metrics.csv"
    if seed_path.exists():
        seed = normalize_seed_frame(pd.read_csv(seed_path, keep_default_na=False))
        failures = seed.loc[~seed["status"].eq("ok"), ["run_id", "status", "failure_reason", "config_path", "threshold_or_bias_path"]]
    lines.extend(["## Failures", ""])
    lines.append(failures.to_markdown(index=False) if not failures.empty else "No failed runs recorded.")
    superseded_error_logs = sorted((OUT_ROOT / "logs").glob("*.error.log"))
    if failures.empty and superseded_error_logs:
        lines.append("")
        lines.append(
            f"Superseded error logs retained from corrected reruns: {len(superseded_error_logs)} files under `results/revised_fairness_v1/logs/`."
        )
    lines.append("")
    (OUT_ROOT / "EXPERIMENT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run revised fairness v1 experiments and aggregate results.")
    parser.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    parser.add_argument("--feature-sets", nargs="*", default=FEATURE_SETS, choices=FEATURE_SETS)
    parser.add_argument("--models", nargs="*", default=SIMPLE_MODELS, choices=SIMPLE_MODELS)
    parser.add_argument("--mctrcm-kinds", nargs="*", default=["locked", "single_task", "k1"], choices=["locked", "single_task", "k1"])
    parser.add_argument("--skip-restricted", action="store_true")
    parser.add_argument("--skip-simple-multitask", action="store_true")
    parser.add_argument("--skip-mctrcm-ingest", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_dirs()
    write_json(OUT_ROOT / "feature_set_definitions.json", build_feature_set_definitions())
    if not args.summary_only:
        if not args.skip_restricted:
            run_restricted_feature_experiments(args)
        if not args.skip_simple_multitask:
            run_simple_multitask_experiments(args)
        if not args.skip_mctrcm_ingest:
            ingest_mctrcm_predictions(args)
    summary = build_summary()
    if not summary.empty:
        generate_latex_tables(summary)
        generate_report(summary)
    print(f"Wrote revised fairness artifacts to {OUT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
