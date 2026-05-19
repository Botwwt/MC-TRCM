from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import traceback
from collections import Counter
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_revised_fairness_v1 import (  # noqa: E402
    apply_class_bias,
    apply_validation_postprocessing,
    balanced_accuracy,
    fit_binary_threshold,
    fit_class_bias,
    fit_temperature,
    normalize_probabilities,
    safe_float,
    standard_error,
)
from src.evaluation.metrics import compute_metrics  # noqa: E402
from src.models.baselines import build_estimator, probe_dependencies  # noqa: E402


WINDOW_ROOT = BUNDLE_ROOT / "data_interim" / "window_tables" / "psyche_d"
MCTRCM_PRED_ROOT = BUNDLE_ROOT / "outputs" / "predictions" / "mctrcm_v2"
MCTRCM_LOG_ROOT = BUNDLE_ROOT / "outputs" / "logs"
OUT_ROOT = ROOT / "results" / "psyched_sensitivity_v1"
TABLE_ROOT = ROOT / "tables" / "final"

SEEDS = [20260417, 20260418, 20260419, 20260420, 20260421]
TRAINED_MODELS = ["null", "elastic_net", "lightgbm", "xgboost", "ebm", "mlp"]
BOOTSTRAP_REPS = 1000
FEATURE_PREFIXES = ("feat_", "modality_mask_", "concept_mask_")


VARIANTS = {
    "current_binary": {
        "task_type": "binary",
        "classes": ["stable_or_improved", "worsened"],
        "primary": "ba",
        "source_mctrcm_task": "phq_change_binary",
        "description": "worsened: delta > 0; stable_or_improved: delta <= 0",
    },
    "current_multiclass": {
        "task_type": "multiclass",
        "classes": ["improved", "stable", "worsened"],
        "primary": "ba",
        "source_mctrcm_task": "phq_change_multiclass",
        "description": "improved: delta < 0; stable: delta = 0; worsened: delta > 0",
    },
    "deadzone3_binary": {
        "task_type": "binary",
        "classes": ["not_worsened", "worsened"],
        "primary": "ba",
        "source_mctrcm_task": "phq_change_binary",
        "description": "worsened: delta >= 3; not_worsened: delta < 3",
    },
    "deadzone3_multiclass": {
        "task_type": "multiclass",
        "classes": ["improved", "stable", "worsened"],
        "primary": "ba",
        "source_mctrcm_task": "phq_change_multiclass",
        "description": "improved: delta <= -3; stable: -2 <= delta <= 2; worsened: delta >= 3",
    },
    "deadzone5_binary": {
        "task_type": "binary",
        "classes": ["not_worsened", "worsened"],
        "primary": "ba",
        "source_mctrcm_task": "phq_change_binary",
        "description": "worsened: delta >= 5; not_worsened: delta < 5",
    },
    "deadzone5_multiclass": {
        "task_type": "multiclass",
        "classes": ["improved", "stable", "worsened"],
        "primary": "ba",
        "source_mctrcm_task": "phq_change_multiclass",
        "description": "improved: delta <= -5; stable: -4 <= delta <= 4; worsened: delta >= 5",
    },
    "continuous_delta_regression": {
        "task_type": "continuous",
        "classes": None,
        "primary": "r2",
        "source_mctrcm_task": None,
        "description": "target: end PHQ-9 minus start PHQ-9",
    },
}


@dataclass
class VariantBundle:
    variant: str
    task_type: str
    classes: list[str] | None
    feature_columns: list[str]
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    descriptive_only: bool
    descriptive_reason: str


def ensure_dirs() -> None:
    for path in [
        OUT_ROOT,
        OUT_ROOT / "configs",
        OUT_ROOT / "predictions",
        OUT_ROOT / "calibration",
        OUT_ROOT / "logs",
        TABLE_ROOT,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def assign_split(subject_id: object, manifest: dict[str, Any]) -> str:
    normalized = str(subject_id)
    for split, key in (("train", "train_subjects"), ("valid", "valid_subjects"), ("test", "test_subjects")):
        if normalized in {str(item) for item in manifest.get(key, [])}:
            return split
    raise KeyError(f"Subject {subject_id} is absent from PSYCHE-D split manifest.")


def parse_source_row(anchor_id: str) -> str:
    match = re.match(r"psyche_d__(.+?)__phq_change_", str(anchor_id))
    return match.group(1) if match else str(anchor_id)


def parse_anchor_order(source_row_id: str) -> int:
    parts = str(source_row_id).rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return int(parts[1])
    return 0


def build_base_frame() -> pd.DataFrame:
    windows = pd.read_parquet(WINDOW_ROOT / "windows_wide.parquet")
    labels = pd.read_csv(WINDOW_ROOT / "labels.csv")
    labels = labels.loc[(labels["label_available"] == 1) & labels["task_name"].eq("phq_change_binary")].copy()
    label_cols = [
        "anchor_id",
        "phq9_score_start",
        "phq9_score_end",
        "phq9_cat_start",
        "phq9_cat_end",
    ]
    frame = windows.loc[windows["task_name"].eq("phq_change_binary")].merge(labels[label_cols], on="anchor_id", how="inner")
    frame["source_row_id"] = frame["anchor_id"].map(parse_source_row)
    frame["anchor_order"] = frame["source_row_id"].map(parse_anchor_order)
    frame["delta_phq9"] = pd.to_numeric(frame["phq9_score_end"], errors="coerce") - pd.to_numeric(
        frame["phq9_score_start"],
        errors="coerce",
    )
    manifest = read_json(WINDOW_ROOT / "splits.json")
    frame["split"] = frame["subject_id"].map(lambda value: assign_split(value, manifest))
    frame["participant_id"] = frame["subject_id"].astype(str)
    frame = frame.sort_values(["participant_id", "anchor_order", "source_row_id"]).reset_index(drop=True)
    return frame


def label_for_variant(delta: float, variant: str) -> int | float:
    if variant == "current_binary":
        return int(delta > 0)
    if variant == "current_multiclass":
        if delta < 0:
            return 0
        if delta == 0:
            return 1
        return 2
    if variant == "deadzone3_binary":
        return int(delta >= 3)
    if variant == "deadzone3_multiclass":
        if delta <= -3:
            return 0
        if delta >= 3:
            return 2
        return 1
    if variant == "deadzone5_binary":
        return int(delta >= 5)
    if variant == "deadzone5_multiclass":
        if delta <= -5:
            return 0
        if delta >= 5:
            return 2
        return 1
    if variant == "continuous_delta_regression":
        return float(delta)
    raise KeyError(variant)


def make_variant_frame(base: pd.DataFrame, variant: str) -> pd.DataFrame:
    spec = VARIANTS[variant]
    frame = base.copy()
    frame["variant"] = variant
    frame["task_type"] = spec["task_type"]
    frame["target"] = frame["delta_phq9"].map(lambda value: label_for_variant(float(value), variant))
    if spec["classes"] is not None:
        classes = spec["classes"]
        frame["target_label"] = frame["target"].astype(int).map(lambda idx: classes[int(idx)])
    else:
        frame["target_label"] = frame["target"]
    return frame


def feature_columns(frame: pd.DataFrame) -> list[str]:
    candidates = sorted([column for column in frame.columns if column.startswith(FEATURE_PREFIXES)])
    train = frame.loc[frame["split"].eq("train")]
    kept = [column for column in candidates if train[column].notna().any()]
    if not kept:
        frame["__constant_feature__"] = 0.0
        return ["__constant_feature__"]
    return kept


def make_bundle(base: pd.DataFrame, variant: str) -> VariantBundle:
    frame = make_variant_frame(base, variant)
    features = feature_columns(frame)
    train = frame.loc[frame["split"].eq("train")].reset_index(drop=True)
    valid = frame.loc[frame["split"].eq("valid")].reset_index(drop=True)
    test = frame.loc[frame["split"].eq("test")].reset_index(drop=True)
    descriptive_only, reason = descriptive_flag(train, valid, test, variant)
    return VariantBundle(
        variant=variant,
        task_type=VARIANTS[variant]["task_type"],
        classes=VARIANTS[variant]["classes"],
        feature_columns=features,
        train=train,
        valid=valid,
        test=test,
        descriptive_only=descriptive_only,
        descriptive_reason=reason,
    )


def descriptive_flag(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame, variant: str) -> tuple[bool, str]:
    spec = VARIANTS[variant]
    if spec["task_type"] == "continuous":
        return False, ""
    n_classes = len(spec["classes"] or [])
    reasons = []
    for split_name, split in (("train", train), ("valid", valid), ("test", test)):
        counts = np.bincount(split["target"].astype(int).to_numpy(), minlength=n_classes)
        if np.any(counts == 0):
            reasons.append(f"{split_name}_class_absence={counts.tolist()}")
        min_fraction = counts.min() / max(counts.sum(), 1)
        if min_fraction < 0.05:
            reasons.append(f"{split_name}_minority_fraction={min_fraction:.3f}")
    return bool(reasons), "; ".join(reasons)


def write_label_counts(base: pd.DataFrame) -> None:
    rows = []
    for variant in VARIANTS:
        bundle = make_bundle(base, variant)
        for split_name, split in (("train", bundle.train), ("valid", bundle.valid), ("test", bundle.test)):
            row = {
                "variant": variant,
                "task_type": bundle.task_type,
                "split": split_name,
                "n_rows": int(len(split)),
                "n_participants": int(split["participant_id"].nunique()),
                "descriptive_only": bool(bundle.descriptive_only),
                "descriptive_reason": bundle.descriptive_reason,
            }
            if bundle.task_type == "continuous":
                row.update(
                    {
                        "target_mean": float(split["target"].mean()),
                        "target_sd": float(split["target"].std(ddof=1)),
                        "target_min": float(split["target"].min()),
                        "target_max": float(split["target"].max()),
                    }
                )
            else:
                counts = split["target"].astype(int).value_counts().sort_index()
                for index, label in enumerate(bundle.classes or []):
                    row[f"class_{index}_{label}"] = int(counts.get(index, 0))
            rows.append(row)
    pd.DataFrame(rows).to_csv(OUT_ROOT / "label_counts.csv", index=False)


def prepare_xy(bundle: VariantBundle, split: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    x = split[bundle.feature_columns]
    if bundle.task_type == "continuous":
        y = pd.to_numeric(split["target"], errors="coerce").to_numpy(dtype=float)
    else:
        y = split["target"].astype(int).to_numpy()
    return x, y


def build_mlp_estimator(task_type: str, seed: int) -> Any:
    from sklearn.impute import SimpleImputer
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if task_type == "continuous":
        model = MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            alpha=1e-4,
            batch_size=128,
            learning_rate_init=1e-3,
            max_iter=400,
            early_stopping=True,
            n_iter_no_change=20,
            random_state=seed,
        )
    else:
        model = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            alpha=1e-4,
            batch_size=128,
            learning_rate_init=1e-3,
            max_iter=400,
            early_stopping=True,
            n_iter_no_change=20,
            random_state=seed,
        )
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", model)])


def align_probabilities(estimator: Any, probabilities: np.ndarray, n_classes: int) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim == 1:
        probabilities = np.column_stack([1.0 - probabilities, probabilities])
    model = estimator.named_steps["model"] if hasattr(estimator, "named_steps") else estimator
    trained_classes = np.asarray(getattr(model, "classes_", np.arange(probabilities.shape[1])), dtype=int)
    aligned = np.zeros((probabilities.shape[0], n_classes), dtype=float)
    for trained_idx, trained_class in enumerate(trained_classes):
        if 0 <= int(trained_class) < n_classes and trained_idx < probabilities.shape[1]:
            aligned[:, int(trained_class)] = probabilities[:, trained_idx]
    zero = aligned.sum(axis=1) <= 0
    if np.any(zero):
        aligned[zero] = 1.0 / n_classes
    return normalize_probabilities(aligned)


def run_null(bundle: VariantBundle) -> dict[str, Any]:
    _, y_train = prepare_xy(bundle, bundle.train)
    _, y_valid = prepare_xy(bundle, bundle.valid)
    _, y_test = prepare_xy(bundle, bundle.test)
    if bundle.task_type == "continuous":
        value = float(np.mean(y_train))
        return {
            "valid_pred": np.full(len(y_valid), value),
            "test_pred": np.full(len(y_test), value),
            "details": {"train_mean": value},
        }
    n_classes = len(bundle.classes or [])
    counts = np.bincount(y_train.astype(int), minlength=n_classes).astype(float)
    priors = counts / counts.sum()
    return {
        "valid_prob": np.tile(priors.reshape(1, -1), (len(y_valid), 1)),
        "test_prob": np.tile(priors.reshape(1, -1), (len(y_test), 1)),
        "details": {"train_class_priors": priors.tolist()},
    }


def run_model(bundle: VariantBundle, model_name: str, seed: int) -> dict[str, Any]:
    if model_name == "null":
        return run_null(bundle)
    x_train, y_train = prepare_xy(bundle, bundle.train)
    x_valid, _ = prepare_xy(bundle, bundle.valid)
    x_test, _ = prepare_xy(bundle, bundle.test)
    if model_name == "mlp":
        estimator = build_mlp_estimator(bundle.task_type, seed)
    else:
        estimator = build_estimator(
            model_name=model_name,
            label_type=bundle.task_type,
            seed=seed,
            n_classes=1 if bundle.task_type == "continuous" else len(bundle.classes or []),
        )
    estimator.fit(x_train, y_train)
    if bundle.task_type == "continuous":
        return {
            "valid_pred": np.asarray(estimator.predict(x_valid), dtype=float),
            "test_pred": np.asarray(estimator.predict(x_test), dtype=float),
            "details": {"estimator": model_name},
        }
    n_classes = len(bundle.classes or [])
    return {
        "valid_prob": align_probabilities(estimator, estimator.predict_proba(x_valid), n_classes),
        "test_prob": align_probabilities(estimator, estimator.predict_proba(x_test), n_classes),
        "details": {"estimator": model_name},
    }


def classification_metrics(y_true: np.ndarray, probabilities: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    label_type = "binary" if probabilities.shape[1] == 2 else "multiclass"
    metrics = compute_metrics(label_type, y_true.astype(int), pred.astype(int), probabilities)
    return {
        "primary": safe_float(metrics.get("balanced_accuracy")),
        "ba": safe_float(metrics.get("balanced_accuracy")),
        "macro_f1": safe_float(metrics.get("macro_f1")),
        "auroc": safe_float(metrics.get("auroc")),
        "auprc": safe_float(metrics.get("auprc")),
        "brier": safe_float(metrics.get("brier_score")),
        "ece": safe_float(metrics.get("ece")),
    }


def regression_metrics(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    metrics = compute_metrics("continuous", y_true.astype(float), pred.astype(float))
    return {
        "primary": safe_float(metrics.get("r2")),
        "r2": safe_float(metrics.get("r2")),
        "rmse": safe_float(metrics.get("rmse")),
        "mae": safe_float(metrics.get("mae")),
        "spearman": safe_float(metrics.get("spearman")),
    }


def decode_from_post(probabilities: np.ndarray, post: dict[str, Any]) -> np.ndarray:
    probabilities = normalize_probabilities(probabilities)
    temp = float(post.get("temperature", 1.0))
    scaled = temperature_scale(probabilities, temp)
    if post.get("postprocessing_type") == "temperature_plus_binary_threshold":
        return (scaled[:, 1] >= float(post["threshold"])).astype(int)
    biases = np.asarray(post.get("biases", [0.0] * scaled.shape[1]), dtype=float)
    return apply_class_bias(scaled, biases).argmax(axis=1)


def temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    probabilities = normalize_probabilities(probabilities)
    logits = np.log(probabilities + 1e-12) / float(temperature)
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def participant_row_macro_classification(frame: pd.DataFrame, prob: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    values = []
    f1_values = []
    brier_values = []
    for subject, idx in frame.groupby("participant_id").indices.items():
        indices = np.asarray(idx, dtype=int)
        y = frame.iloc[indices]["target"].astype(int).to_numpy()
        p = prob[indices]
        yhat = pred[indices]
        values.append(local_balanced_accuracy(y, yhat))
        from sklearn.metrics import f1_score

        f1_values.append(safe_float(f1_score(y, yhat, average="macro")))
        target = np.zeros_like(p)
        target[np.arange(len(y)), y] = 1.0
        brier_values.append(float(np.mean(np.sum((p - target) ** 2, axis=1))) if p.shape[1] > 2 else float(np.mean((p[:, 1] - y) ** 2)))
    return {
        "primary": float(np.nanmean(values)) if values else math.nan,
        "ba": float(np.nanmean(values)) if values else math.nan,
        "macro_f1": float(np.nanmean(f1_values)) if f1_values else math.nan,
        "brier": float(np.nanmean(brier_values)) if brier_values else math.nan,
    }


def local_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    recalls = []
    for label in np.unique(y_true):
        mask = y_true == label
        recalls.append(float(np.mean(y_pred[mask] == label)))
    return float(np.mean(recalls)) if recalls else math.nan


def fast_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0:
        return math.nan
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if denom <= 0.0:
        return math.nan
    return float(1.0 - (np.sum((y_true - y_pred) ** 2) / denom))


def aggregate_label_for_participant(group: pd.DataFrame) -> int:
    labels = group["target"].astype(int).tolist()
    counts = Counter(labels)
    top_count = max(counts.values())
    tied = {label for label, count in counts.items() if count == top_count}
    if len(tied) == 1:
        return int(next(iter(tied)))
    last_label = int(group.sort_values("anchor_order").iloc[-1]["target"])
    return last_label if last_label in tied else int(sorted(tied)[0])


def participant_aggregated_classification(frame: pd.DataFrame, prob: np.ndarray, post: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for subject, idx in frame.groupby("participant_id").indices.items():
        indices = np.asarray(idx, dtype=int)
        group = frame.iloc[indices]
        rows.append(
            {
                "participant_id": subject,
                "target": aggregate_label_for_participant(group),
                "prob": prob[indices].mean(axis=0),
            }
        )
    y = np.asarray([row["target"] for row in rows], dtype=int)
    p = normalize_probabilities(np.vstack([row["prob"] for row in rows]))
    pred = decode_from_post(p, post)
    metrics = classification_metrics(y, p, pred)
    return {"metrics": metrics, "frame": pd.DataFrame({"participant_id": [row["participant_id"] for row in rows], "target": y}), "prob": p, "pred": pred}


def participant_regression(frame: pd.DataFrame, pred: np.ndarray) -> dict[str, Any]:
    temp = frame[["participant_id", "target"]].copy()
    temp["pred"] = pred
    grouped = temp.groupby("participant_id", as_index=False)[["target", "pred"]].mean()
    metrics = regression_metrics(grouped["target"].to_numpy(dtype=float), grouped["pred"].to_numpy(dtype=float))
    return {"metrics": metrics, "frame": grouped}


def save_predictions(run_id: str, bundle: VariantBundle, outputs: dict[str, Any], post: dict[str, Any] | None) -> Path:
    rows = []
    for split_name, split in (("valid", bundle.valid), ("test", bundle.test)):
        output = split[["participant_id", "source_row_id", "anchor_order", "split", "delta_phq9", "target", "target_label"]].copy()
        output["variant"] = bundle.variant
        output["task_type"] = bundle.task_type
        if bundle.task_type == "continuous":
            pred = outputs[f"{split_name}_pred"]
            output["prediction"] = pred
        else:
            prob = outputs[f"{split_name}_prob"]
            if split_name == "valid":
                cal_prob = post["calibrated_valid_prob"]
                cal_pred = post["calibrated_valid_pred"]
                uncal_pred = post["uncalibrated_valid_pred"]
            else:
                cal_prob = post["calibrated_test_prob"]
                cal_pred = post["calibrated_test_pred"]
                uncal_pred = post["uncalibrated_test_pred"]
            output["prediction_uncalibrated"] = uncal_pred
            output["prediction_calibrated"] = cal_pred
            for idx, label in enumerate(bundle.classes or []):
                safe_label = str(label).replace(" ", "_")
                output[f"proba_uncalibrated_{safe_label}"] = prob[:, idx]
                output[f"proba_calibrated_{safe_label}"] = cal_prob[:, idx]
        rows.append(output)
    path = OUT_ROOT / "predictions" / f"{run_id}.csv"
    pd.concat(rows, ignore_index=True).to_csv(path, index=False)
    return path


def append_csv(path: Path, row: dict[str, Any], subset: list[str]) -> None:
    frame = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path, keep_default_na=False)
        out = pd.concat([old, frame], ignore_index=True).drop_duplicates(subset=subset, keep="last")
    else:
        out = frame
    out.to_csv(path, index=False)


def config_common(bundle: VariantBundle, model_name: str, seed: int, run_id: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "psyche_d",
        "variant": bundle.variant,
        "label_definition": VARIANTS[bundle.variant]["description"],
        "task_type": bundle.task_type,
        "model_family": model_name,
        "seed": seed,
        "split_manifest": str((WINDOW_ROOT / "splits.json").relative_to(ROOT)),
        "feature_set": "FULL",
        "feature_columns": bundle.feature_columns,
        "selection_policy": "All thresholds/calibration use validation labels only; test is evaluated after fixed configuration.",
        "descriptive_only": bundle.descriptive_only,
        "descriptive_reason": bundle.descriptive_reason,
    }


def record_failure(bundle: VariantBundle, model_name: str, seed: int, run_id: str, config_path: Path, exc: BaseException) -> None:
    log_path = OUT_ROOT / "logs" / f"{run_id}.error.log"
    log_path.write_text("".join(traceback.format_exception(exc)), encoding="utf-8")
    row = {
        "run_id": run_id,
        "seed": seed,
        "variant": bundle.variant,
        "task_type": bundle.task_type,
        "model_family": model_name,
        "training_label_variant": bundle.variant,
        "val_primary": math.nan,
        "test_primary": math.nan,
        "status": "failed",
        "failure_reason": f"{type(exc).__name__}: {exc}",
        "config_path": str(config_path.relative_to(ROOT)),
        "prediction_path": "",
        "calibration_path": str(log_path.relative_to(ROOT)),
        "descriptive_only": bundle.descriptive_only,
        "descriptive_reason": bundle.descriptive_reason,
    }
    append_csv(OUT_ROOT / "seed_metrics_taskrow.csv", row, ["run_id"])


def evaluate_and_record(bundle: VariantBundle, model_name: str, seed: int, outputs: dict[str, Any], run_id: str, config_path: Path, training_label_variant: str) -> None:
    _, y_valid = prepare_xy(bundle, bundle.valid)
    _, y_test = prepare_xy(bundle, bundle.test)
    calibration_path = OUT_ROOT / "calibration" / f"{run_id}.json"
    if bundle.task_type == "continuous":
        valid_metrics = regression_metrics(y_valid, outputs["valid_pred"])
        test_metrics = regression_metrics(y_test, outputs["test_pred"])
        pred_path = save_predictions(run_id, bundle, outputs, None)
        write_json(calibration_path, {"postprocessing_type": "none_regression"})
        task_row = base_metric_row(run_id, seed, bundle, model_name, training_label_variant, config_path, pred_path, calibration_path)
        task_row.update(prefix_metrics("", test_metrics))
        task_row["val_primary"] = valid_metrics["primary"]
        task_row["test_primary"] = test_metrics["primary"]
        append_csv(OUT_ROOT / "seed_metrics_taskrow.csv", task_row, ["run_id"])
        part = participant_regression(bundle.test, outputs["test_pred"])
        part_row = participant_metric_row(task_row, "participant_mean_target_prediction", part["metrics"])
        append_csv(OUT_ROOT / "seed_metrics_participant.csv", part_row, ["run_id", "participant_eval_scheme"])
        return

    post = apply_validation_postprocessing(bundle.task_type, y_valid, outputs["valid_prob"], y_test, outputs["test_prob"])
    write_json(
        calibration_path,
        {
            "postprocessing_type": post["postprocessing_type"],
            "temperature": post["temperature"],
            "threshold": post["threshold"],
            "biases": post["biases"],
            "validation_primary_uncalibrated": post["uncalibrated_valid_metrics"]["ba"],
            "validation_primary_calibrated": post["calibrated_valid_metrics"]["ba"],
        },
    )
    pred_path = save_predictions(run_id, bundle, outputs, post)
    task_row = base_metric_row(run_id, seed, bundle, model_name, training_label_variant, config_path, pred_path, calibration_path)
    task_row.update(prefix_metrics("", post["calibrated_test_metrics"]))
    task_row.update(prefix_metrics("uncalibrated_", post["uncalibrated_test_metrics"]))
    task_row["val_primary"] = post["calibrated_valid_metrics"]["ba"]
    task_row["test_primary"] = post["calibrated_test_metrics"]["ba"]
    task_row["postprocessing_type"] = post["postprocessing_type"]
    append_csv(OUT_ROOT / "seed_metrics_taskrow.csv", task_row, ["run_id"])

    row_macro = participant_row_macro_classification(bundle.test, post["calibrated_test_prob"], post["calibrated_test_pred"])
    append_csv(OUT_ROOT / "seed_metrics_participant.csv", participant_metric_row(task_row, "participant_row_macro", row_macro), ["run_id", "participant_eval_scheme"])
    aggregated = participant_aggregated_classification(bundle.test, post["calibrated_test_prob"], post)
    append_csv(OUT_ROOT / "seed_metrics_participant.csv", participant_metric_row(task_row, "participant_aggregated_majority_last_tie", aggregated["metrics"]), ["run_id", "participant_eval_scheme"])


def base_metric_row(
    run_id: str,
    seed: int,
    bundle: VariantBundle,
    model_name: str,
    training_label_variant: str,
    config_path: Path,
    pred_path: Path,
    calibration_path: Path,
) -> dict[str, Any]:
    row = {
        "run_id": run_id,
        "seed": seed,
        "dataset": "psyche_d",
        "variant": bundle.variant,
        "task_type": bundle.task_type,
        "model_family": model_name,
        "training_label_variant": training_label_variant,
        "feature_set": "FULL",
        "n_train_rows": int(len(bundle.train)),
        "n_val_rows": int(len(bundle.valid)),
        "n_test_rows": int(len(bundle.test)),
        "n_train_participants": int(bundle.train["participant_id"].nunique()),
        "n_val_participants": int(bundle.valid["participant_id"].nunique()),
        "n_test_participants": int(bundle.test["participant_id"].nunique()),
        "val_primary": math.nan,
        "test_primary": math.nan,
        "status": "ok",
        "failure_reason": "",
        "postprocessing_type": "",
        "config_path": str(config_path.relative_to(ROOT)),
        "prediction_path": str(pred_path.relative_to(ROOT)),
        "calibration_path": str(calibration_path.relative_to(ROOT)),
        "descriptive_only": bundle.descriptive_only,
        "descriptive_reason": bundle.descriptive_reason,
    }
    for metric in ["r2", "rmse", "mae", "spearman", "ba", "macro_f1", "auroc", "auprc", "brier", "ece"]:
        row[f"test_{metric}"] = math.nan
        row[f"uncalibrated_test_{metric}"] = math.nan
    return row


def prefix_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    out = {}
    mapping = {
        "r2": "r2",
        "rmse": "rmse",
        "mae": "mae",
        "spearman": "spearman",
        "ba": "ba",
        "macro_f1": "macro_f1",
        "auroc": "auroc",
        "auprc": "auprc",
        "brier": "brier",
        "ece": "ece",
    }
    for source, target in mapping.items():
        if source in metrics:
            out[f"{prefix}test_{target}"] = metrics[source]
    return out


def participant_metric_row(task_row: dict[str, Any], scheme: str, metrics: dict[str, float]) -> dict[str, Any]:
    row = {key: task_row[key] for key in task_row if key not in {"prediction_path"}}
    row["participant_eval_scheme"] = scheme
    row["participant_primary"] = metrics.get("primary", math.nan)
    row["participant_r2"] = metrics.get("r2", math.nan)
    row["participant_rmse"] = metrics.get("rmse", math.nan)
    row["participant_mae"] = metrics.get("mae", math.nan)
    row["participant_spearman"] = metrics.get("spearman", math.nan)
    row["participant_ba"] = metrics.get("ba", math.nan)
    row["participant_macro_f1"] = metrics.get("macro_f1", math.nan)
    row["participant_brier"] = metrics.get("brier", math.nan)
    return row


def run_trained_models(base: pd.DataFrame, args: argparse.Namespace) -> None:
    deps = probe_dependencies()
    write_json(OUT_ROOT / "logs" / "dependency_probe.json", deps)
    for variant in args.variants:
        bundle = make_bundle(base, variant)
        for model_name in args.models:
            dep_error: BaseException | None = None
            if model_name == "lightgbm" and not deps.get("lightgbm", False):
                dep_error = RuntimeError("Missing dependency: lightgbm")
            if model_name == "xgboost" and not deps.get("xgboost", False):
                dep_error = RuntimeError("Missing dependency: xgboost")
            if model_name == "ebm" and not deps.get("interpret", False):
                dep_error = RuntimeError("Missing dependency: interpret")
            for seed in args.seeds:
                run_id = f"psychedsens__{variant}__{model_name}__seed{seed}"
                config_path = OUT_ROOT / "configs" / f"{run_id}.json"
                write_json(config_path, config_common(bundle, model_name, seed, run_id))
                if args.resume and (OUT_ROOT / "predictions" / f"{run_id}.csv").exists():
                    continue
                if dep_error is not None:
                    record_failure(bundle, model_name, seed, run_id, config_path, dep_error)
                    continue
                try:
                    print(f"[psyched sensitivity] {run_id}", flush=True)
                    outputs = run_model(bundle, model_name, seed)
                    evaluate_and_record(bundle, model_name, seed, outputs, run_id, config_path, bundle.variant)
                except BaseException as exc:
                    record_failure(bundle, model_name, seed, run_id, config_path, exc)


def load_mctrcm_probabilities(seed: int, source_task: str, split: str) -> pd.DataFrame:
    path = MCTRCM_PRED_ROOT / f"final_mctrcm_psyche_d_seed{seed}__{split}.csv"
    frame = pd.read_csv(path)
    frame = frame.loc[frame["task_name"].eq(source_task)].copy()
    frame["source_row_id"] = frame["anchor_id"].map(parse_source_row)
    if source_task == "phq_change_binary":
        ordered = ["proba_0", "proba_1"]
    else:
        # Source columns are named by raw class value. The sensitivity
        # variant class order is improved(-1), stable(0), worsened(1).
        ordered = ["proba_-1", "proba_0", "proba_1"]
    missing = [column for column in ordered if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing MC-TRCM probability columns for {source_task}: {missing}")
    output = frame[["source_row_id"] + ordered].copy()
    output = output.rename(columns={column: f"mctrcm_prob_{idx}" for idx, column in enumerate(ordered)})
    return output.reset_index(drop=True)


def ingest_locked_mctrcm(base: pd.DataFrame, args: argparse.Namespace) -> None:
    for variant in args.variants:
        spec = VARIANTS[variant]
        if spec["task_type"] == "continuous" or spec["source_mctrcm_task"] is None:
            continue
        bundle = make_bundle(base, variant)
        source_task = spec["source_mctrcm_task"]
        training_label_variant = "current_binary" if source_task == "phq_change_binary" else "current_multiclass"
        for seed in args.seeds:
            run_id = f"psychedsens__{variant}__locked_mctrcm_current_label__seed{seed}"
            config_path = OUT_ROOT / "configs" / f"{run_id}.json"
            config = config_common(bundle, "locked_mctrcm_current_label", seed, run_id)
            config.update(
                {
                    "source_mctrcm_run": f"final_mctrcm_psyche_d_seed{seed}",
                    "source_task": source_task,
                    "training_label_variant": training_label_variant,
                    "diagnostic_note": "MC-TRCM was trained on the original current label only; for deadzone variants this is a validation-only re-decoding sensitivity diagnostic, not retraining on the variant label.",
                }
            )
            write_json(config_path, config)
            try:
                valid_probs = load_mctrcm_probabilities(seed, source_task, "valid")
                test_probs = load_mctrcm_probabilities(seed, source_task, "test")
                proba_cols = [column for column in valid_probs.columns if column.startswith("mctrcm_prob_")]
                valid = bundle.valid.merge(valid_probs, on="source_row_id", how="left")
                test = bundle.test.merge(test_probs, on="source_row_id", how="left")
                if valid[proba_cols].isna().any().any() or test[proba_cols].isna().any().any():
                    raise ValueError("Missing MC-TRCM probabilities after source-row merge.")
                outputs = {
                    "valid_prob": valid[proba_cols].to_numpy(dtype=float),
                    "test_prob": test[proba_cols].to_numpy(dtype=float),
                }
                evaluate_and_record(bundle, "locked_mctrcm_current_label", seed, outputs, run_id, config_path, training_label_variant)
            except BaseException as exc:
                record_failure(bundle, "locked_mctrcm_current_label", seed, run_id, config_path, exc)


def load_predictions(run_id: str) -> pd.DataFrame:
    return pd.read_csv(OUT_ROOT / "predictions" / f"{run_id}.csv")


def select_validation_baselines() -> pd.DataFrame:
    metrics = pd.read_csv(OUT_ROOT / "seed_metrics_taskrow.csv", keep_default_na=False)
    metrics = metrics.loc[metrics["status"].eq("ok")].copy()
    metrics["val_primary"] = pd.to_numeric(metrics["val_primary"], errors="coerce")
    trained = metrics.loc[metrics["model_family"].isin([m for m in TRAINED_MODELS if m != "null"])].copy()
    rows = []
    for variant, group in trained.groupby("variant"):
        summary = group.groupby("model_family", as_index=False)["val_primary"].mean().sort_values("val_primary", ascending=False)
        if not summary.empty:
            rows.append({"variant": variant, "selected_baseline": summary.iloc[0]["model_family"], "mean_val_primary": summary.iloc[0]["val_primary"]})
    selected = pd.DataFrame(rows)
    selected.to_csv(OUT_ROOT / "validation_selected_baselines.csv", index=False)
    return selected


def build_bootstrap_cache(bundle: VariantBundle, prediction_frame: pd.DataFrame, calibration: dict[str, Any]) -> dict[str, Any]:
    frame = prediction_frame.loc[prediction_frame["split"].eq("test")].copy()
    frame["participant_id"] = frame["participant_id"].astype(str)
    participant_order = bundle.test["participant_id"].astype(str).drop_duplicates().tolist()
    groups = {str(participant): group for participant, group in frame.groupby("participant_id", sort=False)}
    cache: dict[str, Any] = {
        "participant_order": participant_order,
        "calibration": calibration,
        "task_type": bundle.task_type,
    }
    if bundle.task_type == "continuous":
        row_y = []
        row_pred = []
        part_y = []
        part_pred = []
        for participant in participant_order:
            group = groups[participant]
            y = group["target"].to_numpy(dtype=float)
            pred = group["prediction"].to_numpy(dtype=float)
            row_y.append(y)
            row_pred.append(pred)
            part_y.append(float(np.mean(y)))
            part_pred.append(float(np.mean(pred)))
        cache.update(
            {
                "row_y": row_y,
                "row_pred": row_pred,
                "part_y": np.asarray(part_y, dtype=float),
                "part_pred": np.asarray(part_pred, dtype=float),
            }
        )
        return cache

    proba_cols = [column for column in frame.columns if column.startswith("proba_calibrated_")]
    row_y = []
    row_pred = []
    row_prob = []
    row_macro = []
    agg_y = []
    agg_prob = []
    for participant in participant_order:
        group = groups[participant].copy()
        y = group["target"].to_numpy(dtype=int)
        pred = group["prediction_calibrated"].to_numpy(dtype=int)
        prob = group[proba_cols].to_numpy(dtype=float)
        row_y.append(y)
        row_pred.append(pred)
        row_prob.append(prob)
        row_macro.append(local_balanced_accuracy(y, pred))
        agg_y.append(aggregate_label_for_participant(group))
        agg_prob.append(prob.mean(axis=0))
    cache.update(
        {
            "row_y": row_y,
            "row_pred": row_pred,
            "row_prob": row_prob,
            "row_macro": np.asarray(row_macro, dtype=float),
            "agg_y": np.asarray(agg_y, dtype=int),
            "agg_prob": normalize_probabilities(np.vstack(agg_prob)),
        }
    )
    return cache


def bootstrap_metric_for_caches(caches: list[dict[str, Any]], sample_indices: np.ndarray, eval_level: str) -> float:
    seed_values = []
    for cache in caches:
        if cache["task_type"] == "continuous":
            if eval_level == "task_row":
                y = np.concatenate([cache["row_y"][idx] for idx in sample_indices])
                pred = np.concatenate([cache["row_pred"][idx] for idx in sample_indices])
                seed_values.append(fast_r2(y, pred))
            else:
                y = cache["part_y"][sample_indices]
                pred = cache["part_pred"][sample_indices]
                seed_values.append(fast_r2(y, pred))
            continue

        if eval_level == "task_row":
            y = np.concatenate([cache["row_y"][idx] for idx in sample_indices])
            pred = np.concatenate([cache["row_pred"][idx] for idx in sample_indices])
            prob = np.vstack([cache["row_prob"][idx] for idx in sample_indices])
            seed_values.append(local_balanced_accuracy(y, pred))
        elif eval_level == "participant_row_macro":
            seed_values.append(float(np.nanmean(cache["row_macro"][sample_indices])))
        else:
            y = cache["agg_y"][sample_indices]
            prob = normalize_probabilities(cache["agg_prob"][sample_indices])
            calibration = cache.get("calibration", {})
            if prob.shape[1] == 2 and calibration.get("postprocessing_type") == "temperature_plus_binary_threshold":
                pred = (prob[:, 1] >= float(calibration.get("threshold", 0.5))).astype(int)
            else:
                pred = prob.argmax(axis=1)
            seed_values.append(local_balanced_accuracy(y, pred))
    return float(np.nanmean(seed_values))


def run_bootstrap(base: pd.DataFrame, args: argparse.Namespace) -> None:
    selected = select_validation_baselines()
    metric_rows = pd.read_csv(OUT_ROOT / "seed_metrics_taskrow.csv", keep_default_na=False)
    rows = []
    for variant in args.variants:
        bundle = make_bundle(base, variant)
        selected_match = selected.loc[selected["variant"].eq(variant)]
        if selected_match.empty:
            continue
        baseline = str(selected_match.iloc[0]["selected_baseline"])
        model_specs = [("validation_selected_baseline", baseline)]
        if bundle.task_type != "continuous":
            model_specs.append(("locked_mctrcm_current_label", "locked_mctrcm_current_label"))
        eval_levels = ["task_row"]
        if bundle.task_type == "continuous":
            eval_levels.append("participant_mean_target_prediction")
        else:
            eval_levels.extend(["participant_row_macro", "participant_aggregated_majority_last_tie"])
        pred_by_model = {}
        for label, model in model_specs:
            run_ids = metric_rows.loc[
                metric_rows["variant"].eq(variant)
                & metric_rows["model_family"].eq(model)
                & metric_rows["status"].eq("ok"),
                "run_id",
            ].tolist()
            if run_ids:
                records = []
                for run_id in run_ids:
                    calibration_path = metric_rows.loc[metric_rows["run_id"].eq(run_id), "calibration_path"].iloc[0]
                    calibration = read_json(ROOT / calibration_path) if str(calibration_path).strip() else {}
                    records.append(build_bootstrap_cache(bundle, load_predictions(run_id), calibration))
                pred_by_model[label] = records
        rng = np.random.default_rng(20260429)
        participant_count = int(bundle.test["participant_id"].nunique())
        for eval_level in eval_levels:
            boot_values: dict[str, list[float]] = {label: [] for label in pred_by_model}
            delta_values = []
            for _ in range(args.bootstrap_reps):
                sample_indices = rng.integers(0, participant_count, size=participant_count)
                current_values = {}
                for label, caches in pred_by_model.items():
                    current_values[label] = bootstrap_metric_for_caches(caches, sample_indices, eval_level)
                    boot_values[label].append(current_values[label])
                if "validation_selected_baseline" in current_values and "locked_mctrcm_current_label" in current_values:
                    delta_values.append(current_values["locked_mctrcm_current_label"] - current_values["validation_selected_baseline"])
            for label, values in boot_values.items():
                arr = np.asarray(values, dtype=float)
                rows.append(
                    {
                        "variant": variant,
                        "evaluation_level": eval_level,
                        "model_role": label,
                        "model_family": baseline if label == "validation_selected_baseline" else "locked_mctrcm_current_label",
                        "n_bootstrap": int(len(arr)),
                        "mean_primary": float(np.nanmean(arr)),
                        "ci_low": float(np.nanpercentile(arr, 2.5)),
                        "ci_high": float(np.nanpercentile(arr, 97.5)),
                        "delta_role": "",
                        "descriptive_only": bundle.descriptive_only,
                        "descriptive_reason": bundle.descriptive_reason,
                    }
                )
            if delta_values:
                arr = np.asarray(delta_values, dtype=float)
                rows.append(
                    {
                        "variant": variant,
                        "evaluation_level": eval_level,
                        "model_role": "delta",
                        "model_family": "locked_mctrcm_current_label_minus_validation_selected_baseline",
                        "n_bootstrap": int(len(arr)),
                        "mean_primary": float(np.nanmean(arr)),
                        "ci_low": float(np.nanpercentile(arr, 2.5)),
                        "ci_high": float(np.nanpercentile(arr, 97.5)),
                        "delta_role": "mctrcm_minus_baseline",
                        "descriptive_only": bundle.descriptive_only,
                        "descriptive_reason": bundle.descriptive_reason,
                    }
                )
    pd.DataFrame(rows).to_csv(OUT_ROOT / "bootstrap_ci.csv", index=False)


def summarize_metrics(path: Path, value_col: str = "test_primary") -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False)
    frame = frame.loc[frame["status"].eq("ok")].copy()
    frame[value_col] = pd.to_numeric(frame[value_col], errors="coerce")
    rows = []
    for keys, group in frame.groupby(["variant", "model_family", "training_label_variant"], dropna=False):
        rows.append(
            {
                "variant": keys[0],
                "model_family": keys[1],
                "training_label_variant": keys[2],
                "mean_primary": float(group[value_col].mean()),
                "se_primary": standard_error(group[value_col]),
                "mean_val_primary": float(pd.to_numeric(group.get("val_primary"), errors="coerce").mean()) if "val_primary" in group else math.nan,
                "n_seeds": int(group["seed"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def fmt(mean: float, se: float) -> str:
    if pd.isna(mean):
        return "--"
    return f"{mean:.3f} $\\pm$ {se:.3f}"


def best_row(summary: pd.DataFrame, variant: str, models: list[str]) -> pd.Series | None:
    subset = summary.loc[summary["variant"].eq(variant) & summary["model_family"].isin(models)].copy()
    if subset.empty:
        return None
    subset = subset.sort_values(["mean_val_primary", "mean_primary"], ascending=[False, False])
    return subset.iloc[0]


def generate_tables() -> None:
    task = summarize_metrics(OUT_ROOT / "seed_metrics_taskrow.csv", "test_primary")
    part = pd.read_csv(OUT_ROOT / "seed_metrics_participant.csv", keep_default_na=False)
    part = part.loc[part["status"].eq("ok")].copy()
    part["participant_primary"] = pd.to_numeric(part["participant_primary"], errors="coerce")
    simple_models = ["elastic_net", "lightgbm", "xgboost", "ebm", "mlp"]
    lines = [
        "\\begin{table}[!tbp]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\caption{PSYCHE-D PHQ-change label-definition sensitivity under task-row evaluation. Values are test primary metric mean $\\pm$ SE over five seeds. Baseline selection uses validation primary metric only.}",
        "\\label{tab:psyched_label_sensitivity}",
        "\\begin{tabularx}{\\linewidth}{@{}L{0.23\\linewidth}C{0.13\\linewidth}C{0.16\\linewidth}C{0.16\\linewidth}C{0.16\\linewidth}C{0.12\\linewidth}@{}}",
        "\\toprule",
        "Label variant & Null & Validation-selected baseline & MLP neural & Locked MC-TRCM diagnostic & Descriptive \\\\",
        "\\midrule",
    ]
    label_counts = pd.read_csv(OUT_ROOT / "label_counts.csv", keep_default_na=False)
    for variant in VARIANTS:
        null = best_row(task, variant, ["null"])
        best = best_row(task, variant, simple_models)
        mlp = best_row(task, variant, ["mlp"])
        mct = best_row(task, variant, ["locked_mctrcm_current_label"])
        desc = label_counts.loc[label_counts["variant"].eq(variant), "descriptive_only"].astype(str).eq("True").any()
        lines.append(
            " & ".join(
                [
                    variant.replace("_", "\\_"),
                    fmt(null["mean_primary"], null["se_primary"]) if null is not None else "--",
                    f"{best['model_family']} {fmt(best['mean_primary'], best['se_primary'])}" if best is not None else "--",
                    fmt(mlp["mean_primary"], mlp["se_primary"]) if mlp is not None else "--",
                    fmt(mct["mean_primary"], mct["se_primary"]) if mct is not None else "--",
                    "yes" if desc else "no",
                ]
            )
            + " \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabularx}", "\\end{table}", ""])
    (TABLE_ROOT / "psyched_label_sensitivity.tex").write_text("\n".join(lines), encoding="utf-8")

    lines = [
        "\\begin{table}[!tbp]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\caption{PSYCHE-D task-row and participant-level primary metrics for validation-selected baselines. Participant aggregation uses majority label with last-row tie-break for classification.}",
        "\\label{tab:psyched_participant_level_metrics}",
        "\\begin{tabularx}{\\linewidth}{@{}L{0.24\\linewidth}C{0.16\\linewidth}C{0.18\\linewidth}C{0.18\\linewidth}C{0.18\\linewidth}@{}}",
        "\\toprule",
        "Label variant & Task-row & Participant row-macro & Participant aggregated & Model \\\\",
        "\\midrule",
    ]
    for variant in VARIANTS:
        best = best_row(task, variant, simple_models)
        if best is None:
            continue
        model = best["model_family"]
        task_value = fmt(best["mean_primary"], best["se_primary"])
        psub = part.loc[part["variant"].eq(variant) & part["model_family"].eq(model)]
        row_macro = psub.loc[psub["participant_eval_scheme"].eq("participant_row_macro")]
        aggregated = psub.loc[
            psub["participant_eval_scheme"].isin(["participant_aggregated_majority_last_tie", "participant_mean_target_prediction"])
        ]
        row_macro_value = (
            fmt(row_macro["participant_primary"].mean(), standard_error(row_macro["participant_primary"]))
            if not row_macro.empty
            else "--"
        )
        agg_value = (
            fmt(aggregated["participant_primary"].mean(), standard_error(aggregated["participant_primary"]))
            if not aggregated.empty
            else "--"
        )
        lines.append(" & ".join([variant.replace("_", "\\_"), task_value, row_macro_value, agg_value, str(model)]) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabularx}", "\\end{table}", ""])
    (TABLE_ROOT / "psyched_participant_level_metrics.tex").write_text("\n".join(lines), encoding="utf-8")


def generate_report() -> None:
    task = summarize_metrics(OUT_ROOT / "seed_metrics_taskrow.csv", "test_primary")
    part = pd.read_csv(OUT_ROOT / "seed_metrics_participant.csv", keep_default_na=False)
    part = part.loc[part["status"].eq("ok")].copy()
    part["participant_primary"] = pd.to_numeric(part["participant_primary"], errors="coerce")
    counts = pd.read_csv(OUT_ROOT / "label_counts.csv", keep_default_na=False)
    simple_models = ["elastic_net", "lightgbm", "xgboost", "ebm", "mlp"]
    selected_rows = []
    for variant in VARIANTS:
        best = best_row(task, variant, simple_models)
        null = best_row(task, variant, ["null"])
        mct = best_row(task, variant, ["locked_mctrcm_current_label"])
        selected_rows.append(
            {
                "variant": variant,
                "best_baseline": best["model_family"] if best is not None else "",
                "best_taskrow": best["mean_primary"] if best is not None else math.nan,
                "null_taskrow": null["mean_primary"] if null is not None else math.nan,
                "mctrcm_diagnostic": mct["mean_primary"] if mct is not None else math.nan,
            }
        )
    selected = pd.DataFrame(selected_rows)
    selected.to_csv(OUT_ROOT / "report_variant_matrix.csv", index=False)
    current_binary = selected.loc[selected["variant"].eq("current_binary"), "best_taskrow"].iloc[0]
    dz3_binary = selected.loc[selected["variant"].eq("deadzone3_binary"), "best_taskrow"].iloc[0]
    dz5_binary = selected.loc[selected["variant"].eq("deadzone5_binary"), "best_taskrow"].iloc[0]
    current_multi = selected.loc[selected["variant"].eq("current_multiclass"), "best_taskrow"].iloc[0]
    dz3_multi = selected.loc[selected["variant"].eq("deadzone3_multiclass"), "best_taskrow"].iloc[0]
    dz5_multi = selected.loc[selected["variant"].eq("deadzone5_multiclass"), "best_taskrow"].iloc[0]
    participant_drops = []
    for _, row in selected.iterrows():
        best_model = row["best_baseline"]
        if not best_model:
            continue
        task_value = row["best_taskrow"]
        psub = part.loc[
            part["variant"].eq(row["variant"])
            & part["model_family"].eq(best_model)
            & part["participant_eval_scheme"].isin(["participant_aggregated_majority_last_tie", "participant_mean_target_prediction"])
        ]
        if not psub.empty:
            participant_drops.append(task_value - float(psub["participant_primary"].mean()))
    lines = [
        "# PSYCHE-D Sensitivity v1 Experiment Report",
        "",
        f"Generated: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "This suite reuses `data_interim/window_tables/psyche_d/splits.json`; no participant was reassigned.",
        "All trained tabular and MLP models use the full prepared feature view and validation-only calibration/threshold selection.",
        "Locked MC-TRCM rows are diagnostic re-decodings of the current-label trained model under alternative labels, not retraining on deadzone labels.",
        "",
        "## Label Counts",
        "",
        counts.to_markdown(index=False),
        "",
        "## Answers",
        "",
        f"- Does the current any-change label hold under deadzone3/deadzone5? Binary best-baseline BA changes from {current_binary:.3f} to {dz3_binary:.3f} (deadzone3) and {dz5_binary:.3f} (deadzone5); multiclass BA changes from {current_multi:.3f} to {dz3_multi:.3f} and {dz5_multi:.3f}.",
        "- Are models only predicting tiny PHQ fluctuations? Deadzone3/deadzone5 task-row results remain above null for validation-selected tabular baselines, so the signal is not limited to one-point changes; however participant-aggregated rows and bootstrap intervals must govern participant-level claims.",
        f"- Are participant-level metrics lower than task-row metrics? Mean task-row minus participant-aggregated primary difference across selected baselines is {np.nanmean(participant_drops):.3f}.",
        "- Should the PSYCHE-D main conclusion be downgraded? Yes. It should be limited to processed-feature, task-row prediction; participant-level worsening/change claims are not supported uniformly.",
        "",
        "## Variant Matrix",
        "",
        selected.to_markdown(index=False),
        "",
        "## Failures",
        "",
    ]
    metrics = pd.read_csv(OUT_ROOT / "seed_metrics_taskrow.csv", keep_default_na=False)
    failures = metrics.loc[~metrics["status"].eq("ok"), ["run_id", "model_family", "variant", "failure_reason", "config_path", "calibration_path"]]
    lines.append(failures.to_markdown(index=False) if not failures.empty else "No failed seed runs recorded.")
    lines.append("")
    (OUT_ROOT / "EXPERIMENT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PSYCHE-D label sensitivity, participant metrics, and cluster bootstrap.")
    parser.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    parser.add_argument("--variants", nargs="*", default=list(VARIANTS), choices=list(VARIANTS))
    parser.add_argument("--models", nargs="*", default=TRAINED_MODELS, choices=TRAINED_MODELS)
    parser.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    parser.add_argument("--skip-trained", action="store_true")
    parser.add_argument("--skip-mctrcm-ingest", action="store_true")
    parser.add_argument("--skip-bootstrap", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_dirs()
    base = build_base_frame()
    write_label_counts(base)
    if not args.summary_only:
        if not args.skip_trained:
            run_trained_models(base, args)
        if not args.skip_mctrcm_ingest:
            ingest_locked_mctrcm(base, args)
        if not args.skip_bootstrap:
            run_bootstrap(base, args)
    if (OUT_ROOT / "seed_metrics_taskrow.csv").exists() and (OUT_ROOT / "seed_metrics_participant.csv").exists():
        generate_tables()
        generate_report()
    print(f"Wrote PSYCHE-D sensitivity artifacts to {OUT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
