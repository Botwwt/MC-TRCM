from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_metrics
from src.evaluation.protocol_alignment import primary_metric_name, select_validation_best_baselines
from src.utils.constants import PROJECT_ROOT
from src.utils.io import ensure_dir, write_json

REFERENCE_METRICS_PATHS = [
    PROJECT_ROOT / "outputs" / "predictions" / "mctrcm_v2",
    PROJECT_ROOT / "outputs" / "predictions" / "mctrcm",
]
STATISTICS_SUMMARY_PATH = PROJECT_ROOT / "outputs" / "tables" / "statistics_summary.csv"
STATISTICS_AUDIT_PATH = PROJECT_ROOT / "outputs" / "logs" / "stage_09_bootstrap_audit.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 9 first-pass bootstrap statistics.")
    parser.add_argument("--reference-run", type=str, default="mctrcm_ablation_reference_k6_ssl_v1")
    parser.add_argument("--comparison-manifest", type=str, default=None)
    parser.add_argument("--n-bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260417)
    return parser.parse_args()


def _infer_label_type(frame: pd.DataFrame) -> str:
    if "label_type" in frame.columns and frame["label_type"].notna().any():
        return str(frame["label_type"].dropna().iloc[0])
    if "y_true_index" in frame.columns:
        return "categorical"
    return "continuous"


def _probability_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column.startswith("proba_") and frame[column].notna().any()]


def _task_metrics(frame: pd.DataFrame, label_type: str) -> dict[str, float]:
    if label_type == "continuous":
        y_true = pd.to_numeric(frame["y_true"], errors="coerce").to_numpy(dtype=float)
        y_pred = pd.to_numeric(frame["y_pred"], errors="coerce").to_numpy(dtype=float)
        return compute_metrics(label_type, y_true, y_pred)

    if "y_true_index" in frame.columns and "y_pred_index" in frame.columns:
        y_true = pd.to_numeric(frame["y_true_index"], errors="coerce").to_numpy(dtype=int)
        y_pred = pd.to_numeric(frame["y_pred_index"], errors="coerce").to_numpy(dtype=int)
    else:
        y_true_numeric = pd.to_numeric(frame["y_true"], errors="coerce")
        y_pred_numeric = pd.to_numeric(frame["y_pred"], errors="coerce")
        if y_true_numeric.notna().all() and y_pred_numeric.notna().all():
            y_true = y_true_numeric.to_numpy(dtype=int)
            y_pred = y_pred_numeric.to_numpy(dtype=int)
        else:
            label_space = pd.Index(sorted(set(frame["y_true"].astype(str)) | set(frame["y_pred"].astype(str))))
            mapping = {label: index for index, label in enumerate(label_space.tolist())}
            y_true = frame["y_true"].astype(str).map(mapping).to_numpy(dtype=int)
            y_pred = frame["y_pred"].astype(str).map(mapping).to_numpy(dtype=int)
    proba_columns = _probability_columns(frame)
    probabilities = frame[proba_columns].to_numpy(dtype=float) if proba_columns else None
    n_observed_classes = len(np.unique(y_true))
    metric_label_type = "binary" if label_type == "binary" and n_observed_classes <= 2 else "multiclass"
    return compute_metrics(metric_label_type, y_true, y_pred, probabilities)


def _bootstrap_metric_distribution(
    frame: pd.DataFrame,
    label_type: str,
    primary_metric: str,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[list[float], int]:
    subjects = sorted(frame["subject_id"].astype(str).unique().tolist())
    by_subject = {subject_id: frame.loc[frame["subject_id"].astype(str) == subject_id].copy() for subject_id in subjects}
    values: list[float] = []
    for _ in range(n_bootstrap):
        sampled_subjects = rng.choice(subjects, size=len(subjects), replace=True)
        sampled = pd.concat([by_subject[subject_id] for subject_id in sampled_subjects], ignore_index=True)
        metric_value = _task_metrics(sampled, label_type).get(primary_metric)
        if metric_value is not None and not np.isnan(metric_value):
            values.append(float(metric_value))
    return values, len(subjects)


def _ci(values: list[float]) -> tuple[float, float]:
    if not values:
        return np.nan, np.nan
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def _load_reference_predictions(reference_run: str) -> pd.DataFrame:
    for root in REFERENCE_METRICS_PATHS:
        path = root / f"{reference_run}__test.csv"
        if path.exists():
            frame = pd.read_csv(path)
            return frame.loc[frame["split"] == "test"].copy()
    searched = ", ".join(str(root / f"{reference_run}__test.csv") for root in REFERENCE_METRICS_PATHS)
    raise FileNotFoundError(f"Could not find reference predictions for run={reference_run}. Searched: {searched}")


def _load_baseline_predictions(model_name: str, dataset_id: str, task_name: str) -> pd.DataFrame:
    path = PROJECT_ROOT / "outputs" / "predictions" / "baselines" / model_name / f"{dataset_id}__{task_name}.csv"
    frame = pd.read_csv(path)
    return frame.loc[frame["split"] == "test"].copy()


def _load_comparison_manifest(path: str) -> pd.DataFrame:
    manifest_path = PROJECT_ROOT / path if not Path(path).is_absolute() else Path(path)
    frame = pd.read_csv(manifest_path)
    required_columns = {
        "comparison_id",
        "target_dataset",
        "task_name",
        "reference_run_name",
        "baseline_model_name",
        "baseline_experiment_id",
    }
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise ValueError(f"Comparison manifest is missing required columns: {missing}")
    frame["comparison_id"] = frame["comparison_id"].astype(str)
    frame["target_dataset"] = frame["target_dataset"].astype(str)
    frame["task_name"] = frame["task_name"].astype(str)
    return frame


def _default_comparison_manifest(reference_run: str) -> pd.DataFrame:
    reference_predictions = _load_reference_predictions(reference_run)
    baseline_lookup = select_validation_best_baselines()
    rows: list[dict[str, object]] = []
    for (dataset_id, task_name), frame in reference_predictions.groupby(["dataset_id", "task_name"], dropna=False):
        frame = frame.copy()
        label_type = _infer_label_type(frame)
        baseline_row = baseline_lookup.loc[
            (baseline_lookup["dataset_id"] == str(dataset_id)) & (baseline_lookup["task_name"] == str(task_name))
        ]
        if baseline_row.empty:
            continue
        baseline_row = baseline_row.iloc[0]
        rows.append(
            {
                "protocol_id": "LEGACY_REFERENCE_PROTOCOL",
                "table_group": "legacy_reference",
                "comparison_id": f"{dataset_id}__{task_name}",
                "target_dataset": str(dataset_id),
                "task_name": str(task_name),
                "reference_model_family": "mctrcm",
                "reference_run_name": reference_run,
                "reference_training_datasets": "",
                "allowed_train_datasets": str(dataset_id),
                "baseline_experiment_id": baseline_row["baseline_experiment_id"],
                "baseline_model_name": baseline_row["baseline_model_name"],
                "split_manifest_path": baseline_row["split_manifest_path"],
                "feature_view": baseline_row["feature_view"],
                "label_type": label_type,
                "metric": primary_metric_name(label_type),
                "seeds": "",
                "baseline_selection_metric": baseline_row["baseline_selection_metric"],
                "baseline_selection_score": baseline_row["baseline_selection_score"],
                "baseline_test_score": baseline_row["baseline_test_score"],
                "notes": "Fallback manifest generated from validation-selected comparable baselines.",
            }
        )
    return pd.DataFrame(rows).sort_values(["target_dataset", "task_name"]).reset_index(drop=True)


def _paired_bootstrap(
    reference_frame: pd.DataFrame,
    baseline_frame: pd.DataFrame,
    label_type: str,
    primary_metric: str,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, int]:
    reference_subjects = set(reference_frame["subject_id"].astype(str))
    baseline_subjects = set(baseline_frame["subject_id"].astype(str))
    common_subjects = sorted(reference_subjects & baseline_subjects)
    if not common_subjects:
        return np.nan, np.nan, np.nan, 0

    ref_by_subject = {
        subject_id: reference_frame.loc[reference_frame["subject_id"].astype(str) == subject_id].copy()
        for subject_id in common_subjects
    }
    base_by_subject = {
        subject_id: baseline_frame.loc[baseline_frame["subject_id"].astype(str) == subject_id].copy()
        for subject_id in common_subjects
    }

    deltas: list[float] = []
    for _ in range(n_bootstrap):
        sampled_subjects = rng.choice(common_subjects, size=len(common_subjects), replace=True)
        ref_sample = pd.concat([ref_by_subject[subject_id] for subject_id in sampled_subjects], ignore_index=True)
        base_sample = pd.concat([base_by_subject[subject_id] for subject_id in sampled_subjects], ignore_index=True)
        ref_metric = _task_metrics(ref_sample, label_type).get(primary_metric)
        base_metric = _task_metrics(base_sample, label_type).get(primary_metric)
        if ref_metric is None or base_metric is None or np.isnan(ref_metric) or np.isnan(base_metric):
            continue
        deltas.append(float(ref_metric - base_metric))

    point_delta = _task_metrics(reference_frame.loc[reference_frame["subject_id"].astype(str).isin(common_subjects)], label_type).get(primary_metric)
    point_baseline = _task_metrics(baseline_frame.loc[baseline_frame["subject_id"].astype(str).isin(common_subjects)], label_type).get(primary_metric)
    if point_delta is None or point_baseline is None or np.isnan(point_delta) or np.isnan(point_baseline):
        point_estimate = np.nan
    else:
        point_estimate = float(point_delta - point_baseline)
    ci_lower, ci_upper = _ci(deltas)
    return point_estimate, ci_lower, ci_upper, len(common_subjects), len(deltas)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    manifest = (
        _load_comparison_manifest(args.comparison_manifest)
        if args.comparison_manifest is not None
        else _default_comparison_manifest(args.reference_run)
    )
    reference_predictions_cache: dict[str, pd.DataFrame] = {}

    rows: list[dict[str, object]] = []
    audit_tasks: list[dict[str, object]] = []

    for manifest_row in manifest.to_dict(orient="records"):
        dataset_id = str(manifest_row["target_dataset"])
        task_name = str(manifest_row["task_name"])
        reference_run_name = str(manifest_row["reference_run_name"])
        if reference_run_name not in reference_predictions_cache:
            reference_predictions_cache[reference_run_name] = _load_reference_predictions(reference_run_name)
        reference_predictions = reference_predictions_cache[reference_run_name]
        reference_task_frame = reference_predictions.loc[
            (reference_predictions["dataset_id"] == dataset_id) & (reference_predictions["task_name"] == task_name)
        ].copy()
        if reference_task_frame.empty:
            raise ValueError(
                f"Reference predictions for run={reference_run_name} do not include dataset={dataset_id}, task={task_name}"
            )
        reference_task_frame = reference_task_frame.copy()
        label_type = str(manifest_row.get("label_type") or _infer_label_type(reference_task_frame))
        primary_metric = str(manifest_row.get("metric") or primary_metric_name(label_type))
        reference_metrics = _task_metrics(reference_task_frame, label_type)
        reference_bootstrap, n_subjects = _bootstrap_metric_distribution(
            reference_task_frame,
            label_type,
            primary_metric,
            args.n_bootstrap,
            rng,
        )
        ref_ci_lower, ref_ci_upper = _ci(reference_bootstrap)

        rows.append(
            {
                "analysis_type": "participant_bootstrap",
                "protocol_id": manifest_row.get("protocol_id"),
                "table_group": manifest_row.get("table_group"),
                "comparison_id": manifest_row.get("comparison_id"),
                "dataset_id": dataset_id,
                "task_name": task_name,
                "label_type": label_type,
                "model_name": "mctrcm",
                "reference_model_family": manifest_row.get("reference_model_family", "mctrcm"),
                "run_name": reference_run_name,
                "reference_run_name": reference_run_name,
                "allowed_train_datasets": manifest_row.get("allowed_train_datasets"),
                "split_manifest_path": manifest_row.get("split_manifest_path"),
                "feature_view": manifest_row.get("feature_view"),
                "seeds": manifest_row.get("seeds"),
                "primary_metric": primary_metric,
                "point_estimate": reference_metrics.get(primary_metric),
                "ci_lower": ref_ci_lower,
                "ci_upper": ref_ci_upper,
                "auroc": reference_metrics.get("auroc"),
                "auprc": reference_metrics.get("auprc"),
                "balanced_accuracy": reference_metrics.get("balanced_accuracy"),
                "macro_f1": reference_metrics.get("macro_f1"),
                "brier_score": reference_metrics.get("brier_score"),
                "ece": reference_metrics.get("ece"),
                "mae": reference_metrics.get("mae"),
                "rmse": reference_metrics.get("rmse"),
                "r2": reference_metrics.get("r2"),
                "spearman": reference_metrics.get("spearman"),
                "comparison_model_name": None,
                "comparison_run_name": None,
                "baseline_selection_metric": manifest_row.get("baseline_selection_metric"),
                "paired_delta_point_estimate": None,
                "paired_delta_ci_lower": None,
                "paired_delta_ci_upper": None,
                "n_subjects": n_subjects,
                "n_bootstrap_success": len(reference_bootstrap),
            }
        )

        baseline_model_name = str(manifest_row["baseline_model_name"]) if pd.notna(manifest_row["baseline_model_name"]) else ""
        if baseline_model_name:
            baseline_frame = _load_baseline_predictions(baseline_model_name, dataset_id, task_name)
            baseline_label_type = label_type
            baseline_primary_metric = primary_metric_name(baseline_label_type)
            baseline_metrics = _task_metrics(baseline_frame, baseline_label_type)
            baseline_bootstrap, baseline_subjects = _bootstrap_metric_distribution(
                baseline_frame,
                baseline_label_type,
                baseline_primary_metric,
                args.n_bootstrap,
                rng,
            )
            baseline_ci_lower, baseline_ci_upper = _ci(baseline_bootstrap)
            paired_point, paired_ci_lower, paired_ci_upper, paired_subjects, paired_success = _paired_bootstrap(
                reference_task_frame,
                baseline_frame,
                baseline_label_type,
                baseline_primary_metric,
                args.n_bootstrap,
                rng,
            )

            rows.append(
                {
                    "analysis_type": "participant_bootstrap",
                    "protocol_id": manifest_row.get("protocol_id"),
                    "table_group": manifest_row.get("table_group"),
                    "comparison_id": manifest_row.get("comparison_id"),
                    "dataset_id": dataset_id,
                    "task_name": task_name,
                    "label_type": baseline_label_type,
                    "model_name": baseline_model_name,
                    "reference_model_family": manifest_row.get("reference_model_family", "mctrcm"),
                    "run_name": manifest_row["baseline_experiment_id"],
                    "reference_run_name": reference_run_name,
                    "allowed_train_datasets": manifest_row.get("allowed_train_datasets"),
                    "split_manifest_path": manifest_row.get("split_manifest_path"),
                    "feature_view": manifest_row.get("feature_view"),
                    "seeds": manifest_row.get("seeds"),
                    "primary_metric": baseline_primary_metric,
                    "point_estimate": baseline_metrics.get(baseline_primary_metric),
                    "ci_lower": baseline_ci_lower,
                    "ci_upper": baseline_ci_upper,
                    "auroc": baseline_metrics.get("auroc"),
                    "auprc": baseline_metrics.get("auprc"),
                    "balanced_accuracy": baseline_metrics.get("balanced_accuracy"),
                    "macro_f1": baseline_metrics.get("macro_f1"),
                    "brier_score": baseline_metrics.get("brier_score"),
                    "ece": baseline_metrics.get("ece"),
                    "mae": baseline_metrics.get("mae"),
                    "rmse": baseline_metrics.get("rmse"),
                    "r2": baseline_metrics.get("r2"),
                    "spearman": baseline_metrics.get("spearman"),
                    "comparison_model_name": "mctrcm",
                    "comparison_run_name": reference_run_name,
                    "baseline_selection_metric": manifest_row.get("baseline_selection_metric"),
                    "paired_delta_point_estimate": None,
                    "paired_delta_ci_lower": None,
                    "paired_delta_ci_upper": None,
                    "n_subjects": baseline_subjects,
                    "n_bootstrap_success": len(baseline_bootstrap),
                }
            )
            rows.append(
                {
                    "analysis_type": "paired_bootstrap",
                    "protocol_id": manifest_row.get("protocol_id"),
                    "table_group": manifest_row.get("table_group"),
                    "comparison_id": manifest_row.get("comparison_id"),
                    "dataset_id": dataset_id,
                    "task_name": task_name,
                    "label_type": baseline_label_type,
                    "model_name": "mctrcm",
                    "reference_model_family": manifest_row.get("reference_model_family", "mctrcm"),
                    "run_name": reference_run_name,
                    "reference_run_name": reference_run_name,
                    "allowed_train_datasets": manifest_row.get("allowed_train_datasets"),
                    "split_manifest_path": manifest_row.get("split_manifest_path"),
                    "feature_view": manifest_row.get("feature_view"),
                    "seeds": manifest_row.get("seeds"),
                    "primary_metric": baseline_primary_metric,
                    "point_estimate": reference_metrics.get(primary_metric),
                    "ci_lower": None,
                    "ci_upper": None,
                    "auroc": None,
                    "auprc": None,
                    "balanced_accuracy": None,
                    "macro_f1": None,
                    "brier_score": None,
                    "ece": None,
                    "mae": None,
                    "rmse": None,
                    "r2": None,
                    "spearman": None,
                    "comparison_model_name": baseline_model_name,
                    "comparison_run_name": manifest_row["baseline_experiment_id"],
                    "baseline_selection_metric": manifest_row.get("baseline_selection_metric"),
                    "paired_delta_point_estimate": paired_point,
                    "paired_delta_ci_lower": paired_ci_lower,
                    "paired_delta_ci_upper": paired_ci_upper,
                    "n_subjects": paired_subjects,
                    "n_bootstrap_success": paired_success,
                }
            )

            audit_tasks.append(
                {
                    "dataset_id": dataset_id,
                    "task_name": task_name,
                    "primary_metric": baseline_primary_metric,
                    "baseline_model_name": baseline_model_name,
                    "baseline_experiment_id": manifest_row["baseline_experiment_id"],
                    "reference_run_name": reference_run_name,
                    "protocol_id": manifest_row.get("protocol_id"),
                    "comparison_id": manifest_row.get("comparison_id"),
                    "paired_subjects": paired_subjects,
                    "paired_delta_point_estimate": paired_point,
                    "paired_delta_ci_lower": paired_ci_lower,
                    "paired_delta_ci_upper": paired_ci_upper,
                }
            )
        else:
            audit_tasks.append(
                {
                    "dataset_id": dataset_id,
                    "task_name": task_name,
                    "primary_metric": primary_metric,
                    "baseline_model_name": None,
                    "baseline_experiment_id": None,
                    "reference_run_name": reference_run_name,
                    "protocol_id": manifest_row.get("protocol_id"),
                    "comparison_id": manifest_row.get("comparison_id"),
                    "paired_subjects": 0,
                    "paired_delta_point_estimate": None,
                    "paired_delta_ci_lower": None,
                    "paired_delta_ci_upper": None,
                }
            )

    output = pd.DataFrame(rows).sort_values(["dataset_id", "task_name", "analysis_type", "model_name"]).reset_index(drop=True)
    ensure_dir(STATISTICS_SUMMARY_PATH.parent)
    output.to_csv(STATISTICS_SUMMARY_PATH, index=False)
    write_json(
        STATISTICS_AUDIT_PATH,
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "reference_run": args.reference_run,
            "comparison_manifest": args.comparison_manifest,
            "n_bootstrap": args.n_bootstrap,
            "tasks": audit_tasks,
        },
    )


if __name__ == "__main__":
    main()
