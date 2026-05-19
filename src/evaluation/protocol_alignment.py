from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from src.utils.constants import PROJECT_ROOT
from src.utils.io import ensure_dir

BASELINE_RESULTS_PATH = PROJECT_ROOT / "outputs" / "tables" / "baseline_results.csv"
PROTOCOL_CONFIG_DIR = PROJECT_ROOT / "configs" / "protocols"
PROTOCOL_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "protocol_manifests"

STRICT_WITHIN_PROTOCOL_ID = "STRICT_ALIGNED_WITHIN_DATASET"

WITHIN_DATASET_FEATURE_VIEW = "canonical_window_multiscale_from_windows_wide"

EXCLUDED_COMPARISON_MODELS = {
    "gru_small",
    "lstm_small",
    "mlp_small",
    "transformer_small",
    "psyche_d_public_two_stage",
    "deprest_cat_public_time_series",
    "depresjon_public_actigraphy",
}


@dataclass
class ComparisonManifestRow:
    protocol_id: str
    table_group: str
    comparison_id: str
    target_dataset: str
    task_name: str
    reference_model_family: str
    reference_run_name: str
    reference_training_datasets: str
    allowed_train_datasets: str
    baseline_experiment_id: str
    baseline_model_name: str
    split_manifest_path: str
    feature_view: str
    label_type: str
    metric: str
    seeds: str
    baseline_selection_metric: str
    baseline_selection_score: float
    baseline_test_score: float
    notes: str


def primary_metric_name(label_type: str) -> str:
    return "r2" if str(label_type) == "continuous" else "balanced_accuracy"


def load_protocol_definition(protocol_id: str) -> dict[str, object]:
    path = PROTOCOL_CONFIG_DIR / f"{protocol_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_comparable_baseline_results() -> pd.DataFrame:
    frame = pd.read_csv(BASELINE_RESULTS_PATH)
    return frame.loc[
        (frame["status"] == "completed") & (~frame["model_name"].isin(EXCLUDED_COMPARISON_MODELS))
    ].copy()


def select_validation_best_baselines(frame: pd.DataFrame | None = None) -> pd.DataFrame:
    baseline = load_comparable_baseline_results() if frame is None else frame.copy()
    rows: list[dict[str, object]] = []
    for (dataset_id, task_name), group in baseline.groupby(["dataset_id", "task_name"], dropna=False):
        label_type = str(group["label_type"].dropna().iloc[0]) if group["label_type"].notna().any() else "categorical"
        metric = primary_metric_name(label_type)
        valid_column = f"valid_{metric}"
        test_column = f"test_{metric}"
        scores = pd.to_numeric(group[valid_column], errors="coerce")
        if not scores.notna().any():
            raise ValueError(
                f"Missing validation primary-metric scores for comparable baselines on "
                f"dataset={dataset_id}, task={task_name}, metric={metric}"
            )
        best_index = scores.idxmax()
        best_row = group.loc[best_index]
        rows.append(
            {
                "dataset_id": str(dataset_id),
                "task_name": str(task_name),
                "label_type": label_type,
                "metric": metric,
                "baseline_experiment_id": str(best_row["experiment_id"]),
                "baseline_model_name": str(best_row["model_name"]),
                "baseline_selection_metric": valid_column,
                "baseline_selection_score": float(pd.to_numeric(pd.Series([best_row[valid_column]]), errors="coerce").iloc[0]),
                "baseline_test_score": float(pd.to_numeric(pd.Series([best_row[test_column]]), errors="coerce").iloc[0]),
                "split_manifest_path": f"data_interim/window_tables/{dataset_id}/splits.json",
                "feature_view": WITHIN_DATASET_FEATURE_VIEW,
                "allowed_train_datasets": str(dataset_id),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset_id", "task_name"]).reset_index(drop=True)


def lookup_validation_best_baseline(dataset_id: str, task_name: str, frame: pd.DataFrame | None = None) -> dict[str, object]:
    selected = select_validation_best_baselines(frame)
    match = selected.loc[(selected["dataset_id"] == dataset_id) & (selected["task_name"] == task_name)]
    if match.empty:
        raise KeyError(f"No validation-selected baseline found for dataset={dataset_id}, task={task_name}")
    return match.iloc[0].to_dict()


def write_manifest_csv(path: Path, rows: list[ComparisonManifestRow]) -> pd.DataFrame:
    ensure_dir(path.parent)
    frame = pd.DataFrame([asdict(row) for row in rows])
    frame.to_csv(path, index=False)
    json_path = path.with_suffix(".json")
    json_path.write_text(frame.to_json(orient="records", indent=2), encoding="utf-8")
    return frame
