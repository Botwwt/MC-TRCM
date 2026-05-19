from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_ROOT = ROOT
RESULT_ROOT = ROOT / "results" / "final"
FINAL_SEEDS = [20260417, 20260418, 20260419, 20260420, 20260421]
DATASET_LABELS = {
    "studentlife": "StudentLife",
    "deprest_cat": "DepreST-CAT",
    "psyche_d": "PSYCHE-D",
    "depresjon": "Depresjon",
    "obf": "OBF-Psychiatric",
}


PRED_ROOT = BUNDLE_ROOT / "outputs" / "predictions" / "mctrcm_v2"


def ensure_dirs() -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)


def _safe_float(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def _label_from_proba_column(column: str) -> object:
    value = column.replace("proba_", "", 1)
    try:
        return int(float(value))
    except ValueError:
        return value


def _ece(y_true: np.ndarray, probabilities: np.ndarray, labels: list[object], bins: int = 10) -> float:
    predicted_index = probabilities.argmax(axis=1)
    predicted = np.asarray([labels[index] for index in predicted_index], dtype=object)
    confidence = probabilities.max(axis=1)
    correct = predicted == y_true
    total = max(len(y_true), 1)
    value = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for left, right in zip(edges[:-1], edges[1:]):
        if right == 1.0:
            mask = (confidence >= left) & (confidence <= right)
        else:
            mask = (confidence >= left) & (confidence < right)
        if not mask.any():
            continue
        value += float(mask.sum() / total) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return value


def _classification_metrics(y_true: np.ndarray, probabilities: np.ndarray, labels: list[object]) -> dict[str, float]:
    if all(isinstance(label, (int, np.integer)) for label in labels):
        y_true = np.asarray(y_true, dtype=int)
        predicted = np.asarray([labels[index] for index in probabilities.argmax(axis=1)], dtype=int)
    else:
        y_true = np.asarray(y_true, dtype=str)
        predicted = np.asarray([labels[index] for index in probabilities.argmax(axis=1)], dtype=str)
    output = {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "macro_f1": float(f1_score(y_true, predicted, average="macro", zero_division=0)),
        "ece": _ece(y_true, probabilities, labels),
    }
    label_to_index = {label: index for index, label in enumerate(labels)}
    y_index = np.asarray([label_to_index[value] for value in y_true], dtype=int)
    one_hot = np.eye(len(labels), dtype=float)[y_index]
    output["brier_score"] = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
    try:
        if len(labels) == 2:
            positive = 1 if 1 in label_to_index else labels[-1]
            pos_index = label_to_index[positive]
            output["auroc"] = float(roc_auc_score(y_index, probabilities[:, pos_index]))
            output["auprc"] = float(average_precision_score(y_index, probabilities[:, pos_index]))
        else:
            output["auroc"] = float(roc_auc_score(y_index, probabilities, multi_class="ovr", average="macro"))
            output["auprc"] = float(average_precision_score(one_hot, probabilities, average="macro"))
    except ValueError:
        output["auroc"] = math.nan
        output["auprc"] = math.nan
    return output


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "spearman": float(pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")),
    }


def _load_split(dataset_id: str, split: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for seed in FINAL_SEEDS:
        path = PRED_ROOT / f"final_mctrcm_{dataset_id}_seed{seed}__{split}.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame["seed"] = seed
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _aggregate_regression(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    index_cols = ["dataset_id", "subject_id", "anchor_id", "task_name", "label_type", "y_true"]
    clean = frame.loc[pd.to_numeric(frame["y_true"], errors="coerce").notna()].copy()
    pivot = clean.pivot_table(index=index_cols, columns="seed", values="y_pred", aggfunc="first").reset_index()
    seed_cols = [column for column in pivot.columns if isinstance(column, (int, np.integer))]
    y_true = pd.to_numeric(pivot["y_true"], errors="coerce").to_numpy(dtype=float)
    y_pred = pivot[seed_cols].mean(axis=1).to_numpy(dtype=float)
    return y_true, y_pred


def _aggregate_probabilities(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[object]]:
    proba_cols = [column for column in frame.columns if column.startswith("proba_")]
    labels = [_label_from_proba_column(column) for column in proba_cols]
    index_cols = ["dataset_id", "subject_id", "anchor_id", "task_name", "label_type", "y_true"]
    clean = frame[index_cols + ["seed"] + proba_cols].copy()
    for column in proba_cols:
        clean[column] = pd.to_numeric(clean[column], errors="coerce").fillna(0.0)
    grouped = clean.groupby(index_cols, dropna=False)[proba_cols].mean().reset_index()
    probabilities = grouped[proba_cols].to_numpy(dtype=float)
    row_sums = probabilities.sum(axis=1, keepdims=True)
    probabilities = np.divide(probabilities, np.maximum(row_sums, 1e-12))
    if all(isinstance(label, (int, np.integer)) for label in labels):
        y_true = pd.to_numeric(grouped["y_true"], errors="coerce").astype(int).to_numpy(dtype=int)
    else:
        y_true = grouped["y_true"].astype(str).to_numpy(dtype=object)
        labels = [str(label) for label in labels]
    return y_true, probabilities, labels


def _fit_binary_threshold(y_true: np.ndarray, probabilities: np.ndarray, labels: list[object]) -> dict[str, object]:
    if all(isinstance(label, (int, np.integer)) for label in labels):
        y_true = np.asarray(y_true, dtype=int)
    else:
        y_true = np.asarray(y_true, dtype=str)
    positive = 1 if 1 in labels else labels[-1]
    pos_index = labels.index(positive)
    scores = probabilities[:, pos_index]
    candidates = np.unique(np.quantile(scores, np.linspace(0.02, 0.98, 97)))
    best_threshold = 0.5
    best_score = -np.inf
    for threshold in candidates:
        predicted = np.where(scores >= threshold, positive, labels[1 - pos_index])
        score = balanced_accuracy_score(y_true, predicted)
        if score > best_score or (math.isclose(score, best_score) and abs(float(threshold) - 0.5) < abs(best_threshold - 0.5)):
            best_score = float(score)
            best_threshold = float(threshold)
    return {"threshold": best_threshold, "positive": positive}


def _apply_binary_threshold(probabilities: np.ndarray, labels: list[object], calibration: dict[str, object]) -> np.ndarray:
    positive = calibration["positive"]
    pos_index = labels.index(positive)
    threshold = float(calibration["threshold"])
    calibrated = probabilities.copy()
    predicted_positive = calibrated[:, pos_index] >= threshold
    if threshold > 0:
        calibrated[:, pos_index] = calibrated[:, pos_index] / threshold
    neg_index = 1 - pos_index
    if threshold < 1:
        calibrated[:, neg_index] = calibrated[:, neg_index] / max(1.0 - threshold, 1e-6)
    calibrated = calibrated / np.maximum(calibrated.sum(axis=1, keepdims=True), 1e-12)
    calibrated[predicted_positive, pos_index] = np.maximum(calibrated[predicted_positive, pos_index], calibrated[predicted_positive, neg_index] + 1e-6)
    calibrated[~predicted_positive, neg_index] = np.maximum(calibrated[~predicted_positive, neg_index], calibrated[~predicted_positive, pos_index] + 1e-6)
    calibrated = calibrated / np.maximum(calibrated.sum(axis=1, keepdims=True), 1e-12)
    return calibrated


def _fit_class_bias(y_true: np.ndarray, probabilities: np.ndarray, labels: list[object]) -> dict[str, object]:
    numeric_labels = all(isinstance(label, (int, np.integer)) for label in labels)
    y_true = np.asarray(y_true, dtype=int if numeric_labels else str)
    bias = np.zeros(len(labels), dtype=float)
    grid = np.linspace(-1.5, 1.5, 25)
    pred_dtype = int if numeric_labels else str
    best_score = balanced_accuracy_score(y_true, np.asarray([labels[i] for i in probabilities.argmax(axis=1)], dtype=pred_dtype))
    for _ in range(3):
        improved = False
        for class_index in range(len(labels)):
            best_value = bias[class_index]
            for candidate in grid:
                trial = bias.copy()
                trial[class_index] = candidate
                adjusted = probabilities * np.exp(trial.reshape(1, -1))
                adjusted = adjusted / np.maximum(adjusted.sum(axis=1, keepdims=True), 1e-12)
                predicted = np.asarray([labels[i] for i in adjusted.argmax(axis=1)], dtype=pred_dtype)
                score = balanced_accuracy_score(y_true, predicted)
                if score > best_score:
                    best_score = float(score)
                    best_value = float(candidate)
                    improved = True
            bias[class_index] = best_value
        if not improved:
            break
    return {"class_bias": bias.tolist()}


def _apply_class_bias(probabilities: np.ndarray, calibration: dict[str, object]) -> np.ndarray:
    bias = np.asarray(calibration.get("class_bias", []), dtype=float)
    if bias.size != probabilities.shape[1]:
        return probabilities
    adjusted = probabilities * np.exp(bias.reshape(1, -1))
    return adjusted / np.maximum(adjusted.sum(axis=1, keepdims=True), 1e-12)


def main() -> None:
    ensure_dirs()
    rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    for dataset_id in DATASET_LABELS:
        valid = _load_split(dataset_id, "valid")
        test = _load_split(dataset_id, "test")
        if valid.empty or test.empty:
            continue
        for task_name, valid_task in valid.groupby("task_name"):
            test_task = test.loc[test["task_name"].astype(str).eq(str(task_name))]
            if test_task.empty:
                continue
            label_type = str(valid_task["label_type"].dropna().iloc[0])
            if label_type == "continuous":
                y_valid, pred_valid = _aggregate_regression(valid_task)
                y_test, pred_test = _aggregate_regression(test_task)
                valid_metrics = _regression_metrics(y_valid, pred_valid)
                test_metrics = _regression_metrics(y_test, pred_test)
                calibration = {"mode": "seed_mean_regression"}
            else:
                y_valid, proba_valid, labels = _aggregate_probabilities(valid_task)
                y_test, proba_test, test_labels = _aggregate_probabilities(test_task)
                if labels != test_labels:
                    raise ValueError(f"Class-label mismatch for {dataset_id}/{task_name}: {labels} vs {test_labels}")
                base_valid = _classification_metrics(y_valid, proba_valid, labels)
                if len(labels) == 2:
                    calibration = {"mode": "binary_threshold", **_fit_binary_threshold(y_valid, proba_valid, labels)}
                    proba_valid_cal = _apply_binary_threshold(proba_valid, labels, calibration)
                    proba_test_cal = _apply_binary_threshold(proba_test, labels, calibration)
                else:
                    calibration = {"mode": "class_bias", **_fit_class_bias(y_valid, proba_valid, labels)}
                    proba_valid_cal = _apply_class_bias(proba_valid, calibration)
                    proba_test_cal = _apply_class_bias(proba_test, calibration)
                valid_metrics = _classification_metrics(y_valid, proba_valid_cal, labels)
                test_metrics = _classification_metrics(y_test, proba_test_cal, labels)
                calibration["base_valid_balanced_accuracy"] = base_valid["balanced_accuracy"]
            row = {
                "dataset_id": dataset_id,
                "task_name": task_name,
                "label_type": label_type,
                "split": "test",
                "n_seed_members": len(FINAL_SEEDS),
            }
            row.update({name: _safe_float(value) for name, value in test_metrics.items()})
            rows.append(row)
            calibration_rows.append(
                {
                    "dataset_id": dataset_id,
                    "task_name": task_name,
                    "label_type": label_type,
                    "valid_primary": valid_metrics.get("r2", valid_metrics.get("balanced_accuracy")),
                    "test_primary": test_metrics.get("r2", test_metrics.get("balanced_accuracy")),
                    "calibration": calibration,
                }
            )
    pd.DataFrame(rows).to_csv(RESULT_ROOT / "mctrcm_seed_ensemble_metrics.csv", index=False)
    pd.DataFrame(calibration_rows).to_json(RESULT_ROOT / "mctrcm_seed_ensemble_calibration.json", orient="records", indent=2)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
