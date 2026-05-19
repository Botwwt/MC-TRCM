from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BUNDLE_ROOT = ROOT
PRED_ROOT = BUNDLE_ROOT / "outputs" / "predictions" / "baselines"
OUT_PATH = ROOT / "results" / "final" / "calibrated_baseline_results_seeded.csv"


def _probability_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if str(column).startswith("proba_")]


def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return math.nan
    recalls: list[float] = []
    for label in np.unique(y_true):
        mask = y_true == label
        if np.any(mask):
            recalls.append(float(np.mean(y_pred[mask] == label)))
    return float(np.mean(recalls)) if recalls else math.nan


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return math.nan
    scores: list[float] = []
    for label in np.unique(y_true):
        tp = float(np.sum((y_true == label) & (y_pred == label)))
        fp = float(np.sum((y_true != label) & (y_pred == label)))
        fn = float(np.sum((y_true == label) & (y_pred != label)))
        denom = (2.0 * tp) + fp + fn
        scores.append((2.0 * tp / denom) if denom > 0 else 0.0)
    return float(np.mean(scores)) if scores else math.nan


def _ece(y_true: np.ndarray, probabilities: np.ndarray, n_bins: int = 10) -> float:
    if y_true.size == 0 or probabilities.size == 0:
        return math.nan
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correct = (predicted == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = float(len(y_true))
    value = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        if right == 1.0:
            mask = (confidence >= left) & (confidence <= right)
        else:
            mask = (confidence >= left) & (confidence < right)
        if not np.any(mask):
            continue
        value += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return value if total > 0 else math.nan


def _brier(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    if y_true.size == 0 or probabilities.size == 0:
        return math.nan
    target = np.zeros_like(probabilities, dtype=float)
    valid = (y_true >= 0) & (y_true < probabilities.shape[1])
    target[np.arange(len(y_true))[valid], y_true[valid]] = 1.0
    return float(np.mean(np.sum((probabilities - target) ** 2, axis=1)))


def _normalize(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities.astype(float), 1e-12, 1.0)
    return clipped / clipped.sum(axis=1, keepdims=True)


def _fit_binary_threshold(valid: pd.DataFrame, proba_cols: list[str]) -> float:
    y_true = valid["y_true_index"].to_numpy(dtype=int)
    positive = valid[proba_cols[-1]].to_numpy(dtype=float)
    candidates = np.unique(np.concatenate([np.linspace(0.05, 0.95, 37), np.quantile(positive, np.linspace(0.05, 0.95, 19))]))
    best_key = (-math.inf, -math.inf, -math.inf)
    best_threshold = 0.5
    for threshold in candidates:
        y_pred = (positive >= float(threshold)).astype(int)
        key = (
            _balanced_accuracy(y_true, y_pred),
            _macro_f1(y_true, y_pred),
            -abs(float(threshold) - 0.5),
        )
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold


def _fit_class_bias(valid: pd.DataFrame, proba_cols: list[str]) -> np.ndarray:
    y_true = valid["y_true_index"].to_numpy(dtype=int)
    probabilities = _normalize(valid[proba_cols].to_numpy(dtype=float))
    biases = np.zeros(probabilities.shape[1], dtype=float)
    grid = np.linspace(-2.0, 2.0, 9)
    for _ in range(2):
        for class_index in range(probabilities.shape[1]):
            best_key = (-math.inf, -math.inf)
            best_value = biases[class_index]
            for value in grid:
                trial = biases.copy()
                trial[class_index] = float(value)
                trial = trial - trial.mean()
                adjusted = _apply_class_bias(probabilities, trial)
                y_pred = adjusted.argmax(axis=1)
                key = (_balanced_accuracy(y_true, y_pred), _macro_f1(y_true, y_pred))
                if key > best_key:
                    best_key = key
                    best_value = float(value)
            biases[class_index] = best_value
        biases = biases - biases.mean()
    return biases


def _apply_class_bias(probabilities: np.ndarray, biases: np.ndarray) -> np.ndarray:
    logits = np.log(_normalize(probabilities)) + biases.reshape(1, -1)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def _evaluate(y_true: np.ndarray, probabilities: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "balanced_accuracy": _balanced_accuracy(y_true, y_pred),
        "macro_f1": _macro_f1(y_true, y_pred),
        "brier_score": _brier(y_true, probabilities),
        "ece": _ece(y_true, probabilities),
    }


def _process_file(path: Path) -> dict[str, object] | None:
    frame = pd.read_csv(path)
    proba_cols = _probability_columns(frame)
    required = {"split", "dataset_id", "task_name", "label_type", "y_true_index", "y_pred_index"}
    if not proba_cols or not required.issubset(frame.columns):
        return None
    valid = frame.loc[frame["split"].astype(str).eq("valid")].copy()
    test = frame.loc[frame["split"].astype(str).eq("test")].copy()
    if valid.empty or test.empty:
        return None
    y_test = test["y_true_index"].to_numpy(dtype=int)
    original_pred = test["y_pred_index"].to_numpy(dtype=int)
    original_prob = _normalize(test[proba_cols].to_numpy(dtype=float))
    original = _evaluate(y_test, original_prob, original_pred)
    label_type = str(test["label_type"].iloc[0])
    if len(proba_cols) == 2 and label_type == "binary":
        threshold = _fit_binary_threshold(valid, proba_cols)
        calibrated_prob = original_prob.copy()
        calibrated_pred = (test[proba_cols[-1]].to_numpy(dtype=float) >= threshold).astype(int)
        calibration = f"binary_threshold={threshold:.6f}"
    else:
        biases = _fit_class_bias(valid, proba_cols)
        calibrated_prob = _apply_class_bias(original_prob, biases)
        calibrated_pred = calibrated_prob.argmax(axis=1)
        calibration = "class_bias=" + ";".join(f"{value:.6f}" for value in biases)
    calibrated = _evaluate(y_test, calibrated_prob, calibrated_pred)
    stem_parts = path.stem.split("__seed")
    return {
        "dataset_id": str(test["dataset_id"].iloc[0]),
        "task_name": str(test["task_name"].iloc[0]),
        "label_type": label_type,
        "model_name": path.parent.name,
        "seed": int(stem_parts[-1]) if len(stem_parts) == 2 and stem_parts[-1].isdigit() else math.nan,
        "n_test": int(len(test)),
        "calibration": calibration,
        "original_balanced_accuracy": original["balanced_accuracy"],
        "calibrated_balanced_accuracy": calibrated["balanced_accuracy"],
        "delta_balanced_accuracy": calibrated["balanced_accuracy"] - original["balanced_accuracy"],
        "original_macro_f1": original["macro_f1"],
        "calibrated_macro_f1": calibrated["macro_f1"],
        "original_brier_score": original["brier_score"],
        "calibrated_brier_score": calibrated["brier_score"],
        "original_ece": original["ece"],
        "calibrated_ece": calibrated["ece"],
        "source_file": str(path.relative_to(ROOT)),
    }


def main() -> None:
    rows: list[dict[str, object]] = []
    for path in sorted(PRED_ROOT.glob("*/*__seed*.csv")):
        row = _process_file(path)
        if row is not None:
            rows.append(row)
    output = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(output)} calibrated baseline diagnostic rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
