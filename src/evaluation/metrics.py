from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import pandas as pd


def _safe_float(value: Any) -> float:
    try:
        if value is None:
            return math.nan
        value = float(value)
        if math.isfinite(value):
            return value
    except (TypeError, ValueError):
        pass
    return math.nan


def _expected_calibration_error(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> float:
    if probabilities.ndim == 1:
        probabilities = np.column_stack([1.0 - probabilities, probabilities])

    predictions = probabilities.argmax(axis=1)
    confidences = probabilities.max(axis=1)
    correctness = (predictions == y_true).astype(float)

    ece = 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    for lower, upper in zip(bins[:-1], bins[1:]):
        if upper == 1.0:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)
        if not mask.any():
            continue
        accuracy = correctness[mask].mean()
        confidence = confidences[mask].mean()
        ece += mask.mean() * abs(accuracy - confidence)
    return float(ece)


def _safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return math.nan
    if np.unique(y_true).size < 2 or np.unique(y_pred).size < 2:
        return math.nan
    return _safe_float(pd.Series(y_true).corr(pd.Series(y_pred), method="spearman"))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    return {
        "mae": _safe_float(mean_absolute_error(y_true, y_pred)),
        "rmse": _safe_float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": _safe_float(r2_score(y_true, y_pred)),
        "spearman": _safe_spearman(y_true, y_pred),
    }


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        f1_score,
        roc_auc_score,
    )
    from sklearn.preprocessing import label_binarize

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)

    if probabilities.ndim == 1:
        probabilities = np.column_stack([1.0 - probabilities, probabilities])

    inferred_classes = int(max(np.max(y_true), np.max(y_pred)) + 1) if len(y_true) else probabilities.shape[1]
    n_classes = max(probabilities.shape[1], inferred_classes)
    if probabilities.shape[1] < n_classes:
        padded = np.zeros((probabilities.shape[0], n_classes), dtype=float)
        padded[:, : probabilities.shape[1]] = probabilities
        probabilities = padded

    probabilities = np.clip(probabilities, 1e-8, 1.0)
    row_sums = probabilities.sum(axis=1, keepdims=True)
    zero_rows = row_sums <= 0.0
    if zero_rows.any():
        probabilities[zero_rows] = 1.0 / n_classes
        row_sums = probabilities.sum(axis=1, keepdims=True)
    probabilities = probabilities / row_sums

    unique_classes = np.unique(y_true)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        metrics = {
            "auroc": math.nan,
            "auprc": math.nan,
            "balanced_accuracy": _safe_float(balanced_accuracy_score(y_true, y_pred)),
            "macro_f1": _safe_float(f1_score(y_true, y_pred, average="macro")),
            "brier_score": math.nan,
            "ece": _safe_float(_expected_calibration_error(y_true, probabilities)),
        }

    if n_classes == 2 and unique_classes.size <= 2:
        if unique_classes.size >= 2:
            positive_scores = probabilities[:, 1]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                metrics["auroc"] = _safe_float(roc_auc_score(y_true, positive_scores))
                metrics["auprc"] = _safe_float(average_precision_score(y_true, positive_scores))
        metrics["brier_score"] = _safe_float(brier_score_loss(y_true, probabilities[:, 1]))
        return metrics

    y_true_one_hot = label_binarize(y_true, classes=np.arange(n_classes))
    valid_columns = y_true_one_hot.sum(axis=0) > 0
    if valid_columns.sum() >= 2:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            metrics["auroc"] = _safe_float(
                roc_auc_score(
                    y_true_one_hot[:, valid_columns],
                    probabilities[:, valid_columns],
                    multi_class="ovr",
                    average="macro",
                )
            )
    if valid_columns.sum() >= 1:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            metrics["auprc"] = _safe_float(
                average_precision_score(
                    y_true_one_hot[:, valid_columns],
                    probabilities[:, valid_columns],
                    average="macro",
                )
            )

    metrics["brier_score"] = _safe_float(
        np.mean(np.sum((y_true_one_hot - probabilities) ** 2, axis=1))
    )
    return metrics


def compute_metrics(
    label_type: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray | None = None,
) -> dict[str, float]:
    if label_type == "continuous":
        return regression_metrics(
            y_true=np.asarray(y_true, dtype=float),
            y_pred=np.asarray(y_pred, dtype=float),
        )

    if probabilities is None:
        y_true_array = np.asarray(y_true, dtype=int)
        y_pred_array = np.asarray(y_pred, dtype=int)
        n_classes = int(max(np.max(y_true_array), np.max(y_pred_array)) + 1) if len(y_true_array) else 1
        probabilities = np.eye(n_classes, dtype=float)[y_pred_array]
    return classification_metrics(
        y_true=np.asarray(y_true, dtype=int),
        y_pred=np.asarray(y_pred, dtype=int),
        probabilities=np.asarray(probabilities, dtype=float),
    )
