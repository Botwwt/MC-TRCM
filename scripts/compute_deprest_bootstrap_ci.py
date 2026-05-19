from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TABLES = ROOT / "tables" / "final"
RFV1 = RESULTS / "revised_fairness_v1"
OUT = RESULTS / "deprest_bootstrap_v1"

ENDPOINTS = [
    ("phq9_reg", "DepreST-CAT PHQ-9 severity", "DepreST-CAT PHQ-9 sev.", "$R^2$"),
    ("gad7_reg", "DepreST-CAT GAD-7 severity", "DepreST-CAT GAD-7 sev.", "$R^2$"),
    ("phq9_cat", "DepreST-CAT PHQ-9 category", "DepreST-CAT PHQ-9 cat.", "BA"),
    ("gad7_cat", "DepreST-CAT GAD-7 category", "DepreST-CAT GAD-7 cat.", "BA"),
]

CONTRASTS = [
    ("full_minus_sensor", "Full simple - sensor", "full", "sensor"),
    ("context_minus_full", "Context - full simple", "context", "full"),
    ("k1_minus_k4", "$K=1-K=4$", "k1", "k4"),
    ("k4_minus_full", "$K=4$ - full simple", "k4", "full"),
    ("k1_minus_full", "$K=1$ - full simple", "k1", "full"),
]

LABEL_TO_ROW = {
    "full": ("FULL", None, "endpoint_wise"),
    "sensor": ("SENSOR_ONLY", None, "endpoint_wise"),
    "context": ("SYMPTOM_STATIC_CLINICAL", None, "endpoint_wise"),
    "k4": ("FULL", "mctrcm_locked_k4", "multi_task_locked"),
    "k1": ("FULL", "mctrcm_k1_plain", "multi_task_k1_plain"),
}


def metric_value(frame: pd.DataFrame, endpoint: str) -> float:
    if endpoint.endswith("_reg"):
        y = frame["y_true"].to_numpy(dtype=float)
        pred_col = "inverse_transformed_prediction" if "inverse_transformed_prediction" in frame else "y_pred"
        pred = frame[pred_col].to_numpy(dtype=float)
        denom = float(np.sum((y - y.mean()) ** 2))
        if denom <= 0:
            return math.nan
        return 1.0 - float(np.sum((y - pred) ** 2)) / denom

    y_col = "y_true_index" if "y_true_index" in frame else "y_true"
    p_col = "y_pred_calibrated_index" if "y_pred_calibrated_index" in frame else "y_pred"
    y = frame[y_col].to_numpy(dtype=int)
    pred = frame[p_col].to_numpy(dtype=int)
    recalls = []
    for cls in sorted(np.unique(y).tolist()):
        mask = y == cls
        if mask.any():
            recalls.append(float(np.mean(pred[mask] == cls)))
    return float(np.mean(recalls)) if recalls else math.nan


def prediction_arrays(frame: pd.DataFrame, endpoint: str) -> tuple[np.ndarray, np.ndarray]:
    if endpoint.endswith("_reg"):
        pred_col = "inverse_transformed_prediction" if "inverse_transformed_prediction" in frame else "y_pred"
        return frame["y_true"].to_numpy(dtype=float), frame[pred_col].to_numpy(dtype=float)
    y_col = "y_true_index" if "y_true_index" in frame else "y_true"
    p_col = "y_pred_calibrated_index" if "y_pred_calibrated_index" in frame else "y_pred"
    return frame[y_col].to_numpy(dtype=int), frame[p_col].to_numpy(dtype=int)


def metric_from_arrays(y: np.ndarray, pred: np.ndarray, endpoint: str, sample_idx: np.ndarray | None) -> float:
    if sample_idx is not None:
        y = y[sample_idx]
        pred = pred[sample_idx]
    if endpoint.endswith("_reg"):
        y = y.astype(float, copy=False)
        pred = pred.astype(float, copy=False)
        denom = float(np.sum((y - y.mean()) ** 2))
        if denom <= 0:
            return math.nan
        return 1.0 - float(np.sum((y - pred) ** 2)) / denom
    recalls = []
    for cls in np.unique(y):
        mask = y == cls
        if mask.any():
            recalls.append(float(np.mean(pred[mask] == cls)))
    return float(np.mean(recalls)) if recalls else math.nan


def load_test_predictions(row: pd.Series) -> pd.DataFrame:
    path = ROOT / str(row["prediction_path"])
    frame = pd.read_csv(path)
    frame = frame.loc[frame["split"].eq("test")].copy()
    frame["subject_id"] = frame["subject_id"].astype(str)
    frame = frame.sort_values("subject_id").set_index("subject_id", drop=False)
    if frame.index.has_duplicates:
        raise ValueError(f"Expected one DepreST-CAT test row per participant in {path}")
    return frame


def rows_for(seed_metrics: pd.DataFrame, endpoint: str, label: str, matrix_row: pd.Series) -> dict[int, pd.Series]:
    feature_set, model_family, training_scheme = LABEL_TO_ROW[label]
    if label == "full":
        model_family = str(matrix_row["full_best_model"])
    elif label == "sensor":
        model_family = str(matrix_row["sensor_best_model"])
    elif label == "context":
        model_family = str(matrix_row["symptom_static_model"])

    mask = (
        seed_metrics["dataset"].eq("deprest_cat")
        & seed_metrics["endpoint"].eq(endpoint)
        & seed_metrics["feature_set"].eq(feature_set)
        & seed_metrics["model_family"].eq(model_family)
        & seed_metrics["status"].eq("ok")
    )
    if training_scheme is not None:
        mask &= seed_metrics["training_scheme"].eq(training_scheme)
    rows = seed_metrics.loc[mask].copy()
    if rows.empty:
        raise ValueError(f"No seed rows for {endpoint}/{label}/{feature_set}/{model_family}")
    return {int(row["seed"]): row for _, row in rows.iterrows()}


def aligned_frames(
    seed_metrics: pd.DataFrame,
    endpoint: str,
    left: str,
    right: str,
    matrix_row: pd.Series,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    left_rows = rows_for(seed_metrics, endpoint, left, matrix_row)
    right_rows = rows_for(seed_metrics, endpoint, right, matrix_row)
    common = sorted(set(left_rows).intersection(right_rows))
    if not common:
        raise ValueError(f"No common seeds for {endpoint}/{left}/{right}")

    pairs = []
    for seed in common:
        left_frame = load_test_predictions(left_rows[seed])
        right_frame = load_test_predictions(right_rows[seed])
        subjects = left_frame.index.intersection(right_frame.index)
        if len(subjects) != len(left_frame) or len(subjects) != len(right_frame):
            raise ValueError(f"Participant mismatch for {endpoint}/{left}/{right}/seed{seed}")
        y_left = left_frame.loc[subjects, "y_true"].to_numpy(dtype=float)
        y_right = right_frame.loc[subjects, "y_true"].to_numpy(dtype=float)
        if not np.allclose(y_left, y_right):
            raise ValueError(f"Target mismatch for {endpoint}/{left}/{right}/seed{seed}")
        left_frame = left_frame.loc[subjects]
        right_frame = right_frame.loc[subjects]
        y, left_pred = prediction_arrays(left_frame, endpoint)
        y_right_arr, right_pred = prediction_arrays(right_frame, endpoint)
        if not np.allclose(y.astype(float), y_right_arr.astype(float)):
            raise ValueError(f"Target array mismatch for {endpoint}/{left}/{right}/seed{seed}")
        pairs.append((y, left_pred, right_pred))
    return pairs


def mean_diff(pairs: list[tuple[np.ndarray, np.ndarray, np.ndarray]], endpoint: str, sample_idx: np.ndarray | None) -> float:
    diffs = []
    for y, left_pred, right_pred in pairs:
        diffs.append(
            metric_from_arrays(y, left_pred, endpoint, sample_idx)
            - metric_from_arrays(y, right_pred, endpoint, sample_idx)
        )
    return float(np.nanmean(diffs))


def ci_cell(point: float, low: float, high: float) -> str:
    def fmt(value: float) -> str:
        return f"{value:+.3f}"

    return f"{fmt(point)} [{fmt(low)}, {fmt(high)}]"


def escape_tex(text: str) -> str:
    return text.replace("_", r"\_")


def write_table(summary: pd.DataFrame) -> None:
    endpoint_order = [row[2] for row in ENDPOINTS]
    summary = summary.copy()
    summary["endpoint_display"] = pd.Categorical(summary["endpoint_display"], endpoint_order, ordered=True)
    pivot = summary.pivot(index=["endpoint_display", "metric"], columns="contrast", values="cell")
    pivot = pivot.reset_index()
    pivot = pivot.sort_values("endpoint_display")

    lines = [
        r"\begin{table}[!tbp]",
        r"\centering",
        r"\begin{threeparttable}",
        r"\caption{DepreST-CAT paired participant bootstrap intervals for main fixed-split follow-up contrasts.}",
        r"\label{tab:deprest_bootstrap_ci}",
        r"\tiny",
        r"\setlength{\tabcolsep}{2pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\begin{tabularx}{\linewidth}{@{}L{0.13\linewidth}C{0.045\linewidth}C{0.155\linewidth}C{0.155\linewidth}C{0.145\linewidth}C{0.155\linewidth}C{0.155\linewidth}@{}}",
        r"\toprule",
        r"\rowcolor{tableheader}",
        r"Endpoint & Metric & \makecell{Full simple\\-- sensor} & \makecell{Context\\-- full simple} & $K=1-K=4$ & \makecell{$K=4$\\-- full simple} & \makecell{$K=1$\\-- full simple} \\",
        r"\midrule",
    ]
    for _, row in pivot.iterrows():
        lines.append(
            " & ".join(
                [
                    row["endpoint_display"],
                    row["metric"],
                    row["full_minus_sensor"],
                    row["context_minus_full"],
                    row["k1_minus_k4"],
                    row["k4_minus_full"],
                    row["k1_minus_full"],
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabularx}",
            r"\begin{tablenotes}[flushleft]",
            r"\footnotesize",
            r"\item Cells show mean paired difference [2.5\%, 97.5\% percentile interval] from 1000 participant-bootstrap resamples of the 72 DepreST-CAT test participants, averaging the paired endpoint metric difference over common seeds. Positive values favor the left-hand model or feature set. Bootstrap resamples recompute $R^2$ and BA within each resample.",
            r"\item Full simple, sensor, and context models are the validation-selected families used in Table~\ref{tab:restricted_feature_primary}; MC-TRCM contrasts use the full-feature $K=4$ locked and $K=1$ variant rows. These intervals quantify fixed-test-split resampling variability and do not make the revised follow-up diagnostics an independent confirmation.",
            r"\end{tablenotes}",
            r"\end{threeparttable}",
            r"\end{table}",
            "",
        ]
    )
    (TABLES / "deprest_bootstrap_ci.tex").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    seed_metrics = pd.read_csv(RFV1 / "seed_metrics.csv")
    matrix = pd.read_csv(RFV1 / "report_endpoint_matrix.csv")

    matrix_by_endpoint: dict[str, pd.Series] = {}
    for endpoint, matrix_display, _, _ in ENDPOINTS:
        row = matrix.loc[matrix["endpoint"].eq(matrix_display)]
        if row.empty:
            raise ValueError(f"Missing report matrix row for {matrix_display}")
        matrix_by_endpoint[endpoint] = row.iloc[0]

    rng = np.random.default_rng(20260429)
    rows: list[dict[str, object]] = []
    n_boot = 1000
    for endpoint, _, display, metric in ENDPOINTS:
        matrix_row = matrix_by_endpoint[endpoint]
        for contrast, contrast_label, left, right in CONTRASTS:
            pairs = aligned_frames(seed_metrics, endpoint, left, right, matrix_row)
            n_subjects = int(len(pairs[0][0]))
            point = mean_diff(pairs, endpoint, None)
            boot = []
            for _ in range(n_boot):
                sample_idx = rng.integers(0, n_subjects, size=n_subjects)
                boot.append(mean_diff(pairs, endpoint, sample_idx))
            low, high = np.nanpercentile(np.asarray(boot, dtype=float), [2.5, 97.5])
            rows.append(
                {
                    "dataset": "deprest_cat",
                    "endpoint": endpoint,
                    "endpoint_display": display,
                    "metric": metric,
                    "contrast": contrast,
                    "contrast_label": contrast_label,
                    "left": left,
                    "right": right,
                    "diff_point": point,
                    "ci_low": float(low),
                    "ci_high": float(high),
                    "n_boot": n_boot,
                    "n_test_participants": n_subjects,
                    "cell": ci_cell(point, float(low), float(high)),
                }
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "deprest_paired_bootstrap_ci.csv", index=False)
    write_table(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
