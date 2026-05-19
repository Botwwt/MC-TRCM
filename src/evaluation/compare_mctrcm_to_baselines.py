from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.protocol_alignment import select_validation_best_baselines
from src.utils.constants import PROJECT_ROOT
from src.utils.io import ensure_dir

BASELINE_RESULTS_PATH = PROJECT_ROOT / "outputs" / "tables" / "baseline_results.csv"
MCTRCM_RESULTS_PATH = PROJECT_ROOT / "outputs" / "tables" / "mctrcm_results.csv"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "tables" / "mctrcm_vs_baseline_firstpass.csv"


def _metric_name(label_type: str) -> str:
    if label_type == "continuous":
        return "r2"
    return "balanced_accuracy"


def _safe_score(frame: pd.DataFrame, metric: str, split_prefix: str = "test") -> pd.Series:
    column = f"{split_prefix}_{metric}" if f"{split_prefix}_{metric}" in frame.columns else metric
    return pd.to_numeric(frame[column], errors="coerce")


def main() -> None:
    baseline = select_validation_best_baselines()

    mctrcm = pd.read_csv(MCTRCM_RESULTS_PATH)
    valid_mctrcm = mctrcm.loc[mctrcm["split"] == "valid"].copy()
    test_mctrcm = mctrcm.loc[mctrcm["split"] == "test"].copy()

    rows: list[dict[str, object]] = []
    task_keys = sorted(
        set(zip(baseline["dataset_id"], baseline["task_name"]))
        | set(zip(valid_mctrcm["dataset_id"], valid_mctrcm["task_name"]))
        | set(zip(test_mctrcm["dataset_id"], test_mctrcm["task_name"]))
    )
    for dataset_id, task_name in task_keys:
        baseline_task = baseline.loc[(baseline["dataset_id"] == dataset_id) & (baseline["task_name"] == task_name)].copy()
        valid_task = valid_mctrcm.loc[
            (valid_mctrcm["dataset_id"] == dataset_id) & (valid_mctrcm["task_name"] == task_name)
        ].copy()
        test_task = test_mctrcm.loc[
            (test_mctrcm["dataset_id"] == dataset_id) & (test_mctrcm["task_name"] == task_name)
        ].copy()
        if baseline_task.empty or valid_task.empty or test_task.empty:
            continue

        label_type = str(
            baseline_task["label_type"].dropna().iloc[0]
            if "label_type" in baseline_task.columns and not baseline_task["label_type"].dropna().empty
            else valid_task["label_type"].dropna().iloc[0]
        )
        metric = _metric_name(label_type)

        baseline_best = baseline_task.iloc[0]
        baseline_score = float(pd.to_numeric(pd.Series([baseline_best["baseline_test_score"]]), errors="coerce").iloc[0])

        valid_scores = _safe_score(valid_task, metric)
        if valid_scores.notna().any():
            valid_best = valid_task.loc[valid_scores.idxmax()]
            mctrcm_best_candidates = test_task.loc[test_task["run_name"] == valid_best["run_name"]].copy()
            if mctrcm_best_candidates.empty:
                mctrcm_best = test_task.iloc[0]
                mctrcm_score = np.nan
            else:
                mctrcm_best = mctrcm_best_candidates.iloc[0]
                mctrcm_score = float(_safe_score(pd.DataFrame([mctrcm_best]), metric).iloc[0])
        else:
            mctrcm_best = test_task.iloc[0]
            mctrcm_score = np.nan

        delta = mctrcm_score - baseline_score if pd.notna(mctrcm_score) and pd.notna(baseline_score) else np.nan
        if pd.isna(delta):
            winner = "undetermined"
        elif delta > 0:
            winner = "mctrcm"
        elif delta < 0:
            winner = "baseline"
        else:
            winner = "tie"

        rows.append(
            {
                "dataset_id": dataset_id,
                "task_name": task_name,
                "label_type": label_type,
                "primary_metric": metric,
                "best_baseline_model": baseline_best["baseline_model_name"],
                "best_baseline_score": baseline_score,
                "baseline_selection_metric": baseline_best["baseline_selection_metric"],
                "baseline_selection_score": baseline_best["baseline_selection_score"],
                "best_mctrcm_run": mctrcm_best["run_name"],
                "best_mctrcm_score": mctrcm_score,
                "delta_mctrcm_minus_baseline": delta,
                "winner": winner,
            }
        )

    output = pd.DataFrame(rows).sort_values(["dataset_id", "task_name"]).reset_index(drop=True)
    ensure_dir(OUTPUT_PATH.parent)
    output.to_csv(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main()
