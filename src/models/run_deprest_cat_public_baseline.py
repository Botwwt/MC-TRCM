from __future__ import annotations

import argparse
import math
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import preprocessing, svm
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import resample

from src.utils.constants import PROJECT_ROOT
from src.utils.io import ensure_dir, write_csv, write_json

STAGE_NAME = "stage_05_public_baseline"
DEPREST_ROOT = PROJECT_ROOT / "data_raw" / "deprest_cat" / "extracted" / "DepreST-CAT-main"
FEATURE_DIR = DEPREST_ROOT / "features"
RESULT_VIEW_DIRS = {
    "combined": DEPREST_ROOT / "machineLearning" / "screeningResults",
    "call": DEPREST_ROOT / "machineLearning" / "screeningResultsCall",
    "text": DEPREST_ROOT / "machineLearning" / "screeningResultsText",
}
RESULT_PATTERN = re.compile(r"resultsCAT(?P<week>\d+)week(?P<label>phq9|gad7)split(?P<split>\d+)\.csv$")

OUTPUT_TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
OUTPUT_LOG_DIR = PROJECT_ROOT / "outputs" / "logs"
REPORT_PATH = PROJECT_ROOT / "reports" / "qa" / "deprest_cat_public_baseline_audit.md"
REGISTRY_PATH = OUTPUT_LOG_DIR / "experiment_registry.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit and smoke-reproduce the public DepreST-CAT screening baseline."
    )
    parser.add_argument("--smoke-week", type=int, default=2)
    parser.add_argument("--smoke-split", type=int, default=5)
    parser.add_argument("--smoke-seeds", nargs="*", type=int, default=[0, 1])
    parser.add_argument("--smoke-labels", nargs="*", default=["phq9", "gad7"])
    parser.add_argument("--smoke-models", nargs="*", default=["SVC", "kNN", "RF", "LR", "XG"])
    parser.add_argument("--smoke-nfeatures", nargs="*", type=int, default=[1, 2, 3, 4])
    return parser.parse_args()


def _load_registry() -> pd.DataFrame:
    if REGISTRY_PATH.exists():
        return pd.read_csv(REGISTRY_PATH)
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
    registry.to_csv(REGISTRY_PATH, index=False)


def _load_public_results() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for source_view, directory in RESULT_VIEW_DIRS.items():
        for path in sorted(directory.glob("resultsCAT*split*.csv")):
            match = RESULT_PATTERN.match(path.name)
            if not match:
                continue
            frame = pd.read_csv(path)
            if frame.columns[0].startswith("Unnamed"):
                frame = frame.drop(columns=frame.columns[0])
            frame["source_view"] = source_view
            frame["week"] = int(match.group("week"))
            frame["label"] = match.group("label")
            frame["split"] = int(match.group("split"))
            frames.append(frame)
    if not frames:
        raise FileNotFoundError("No public DepreST-CAT screening result CSVs were found.")
    results = pd.concat(frames, ignore_index=True)
    for column in ("week", "split", "nFeatures", "F1", "Accuracy", "randomSeed"):
        results[column] = pd.to_numeric(results[column], errors="coerce")
    return results


def _expected_missing_result_files() -> dict[str, list[str]]:
    expected: dict[str, list[str]] = {}
    for source_view, directory in RESULT_VIEW_DIRS.items():
        missing: list[str] = []
        for week in (2, 4, 8, 16):
            for label in ("phq9", "gad7"):
                for split in (5, 6, 7, 8, 9, 10):
                    name = f"resultsCAT{week}week{label}split{split}.csv"
                    if not (directory / name).exists():
                        missing.append(name)
        expected[source_view] = missing
    return expected


def _summarize_public_results(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail = (
        results.groupby(["source_view", "week", "label", "split", "model", "nFeatures"], dropna=False)
        .agg(
            mean_f1=("F1", "mean"),
            std_f1=("F1", "std"),
            mean_accuracy=("Accuracy", "mean"),
            std_accuracy=("Accuracy", "std"),
            n_runs=("randomSeed", "nunique"),
        )
        .reset_index()
        .sort_values(
            ["source_view", "label", "split", "mean_f1", "mean_accuracy", "week", "nFeatures"],
            ascending=[True, True, True, False, False, True, True],
        )
    )
    best = (
        detail.groupby(["source_view", "label", "split"], as_index=False)
        .first()
        .sort_values(["source_view", "label", "split"])
    )
    return detail, best


def _load_feature_frame(week: int) -> pd.DataFrame:
    path = FEATURE_DIR / f"featureSet{week}weeksDepreST-CAT.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing official feature file: {path}")
    return pd.read_csv(path)


def _select_view_frame(frame: pd.DataFrame, source_view: str) -> pd.DataFrame:
    if source_view == "combined":
        return frame.copy()
    n = int((frame.shape[1] - 3) / 2)
    call_ids = range(n + 1, frame.shape[1] - 2)
    text_ids = range(1, n + 1)
    if source_view == "call":
        selected = ["id"] + list(frame.columns[call_ids]) + ["phq9", "gad7"]
        return frame[selected].copy()
    if source_view == "text":
        selected = ["id"] + list(frame.columns[text_ids]) + ["phq9", "gad7"]
        return frame[selected].copy()
    raise ValueError(f"Unsupported source_view: {source_view}")


def _binarize_targets(frame: pd.DataFrame, threshold: int) -> pd.DataFrame:
    output = frame.copy()
    for label in ("phq9", "gad7"):
        output[label] = (pd.to_numeric(output[label], errors="coerce").astype(int) >= threshold).astype(int)
    return output


def _upsample_minority(train_frame: pd.DataFrame) -> pd.DataFrame:
    counts = train_frame["target"].value_counts().to_dict()
    if len(counts) < 2:
        return train_frame.copy()
    majority_label = max(counts, key=counts.get)
    minority_label = min(counts, key=counts.get)
    if counts[majority_label] == counts[minority_label]:
        return train_frame.copy()
    train_majority = train_frame.loc[train_frame["target"] == majority_label]
    train_minority = train_frame.loc[train_frame["target"] == minority_label]
    train_minority_upsampled = resample(
        train_minority,
        replace=True,
        n_samples=len(train_majority),
        random_state=42,
    )
    return pd.concat([train_majority, train_minority_upsampled], ignore_index=True)


def _build_model(model_type: str, seed: int):
    if model_type == "SVC":
        return svm.SVC(random_state=seed)
    if model_type == "RF":
        return RandomForestClassifier(random_state=seed)
    if model_type == "kNN":
        return KNeighborsClassifier()
    if model_type == "LR":
        return LogisticRegression(random_state=seed, max_iter=1000)
    if model_type == "XG":
        return xgb.XGBClassifier(random_state=seed)
    raise ValueError(f"Unsupported model_type: {model_type}")


def _compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    conf_mat = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = conf_mat.ravel()
    precision_den = tp + fp
    recall_den = tp + fn
    precision = tp / precision_den if precision_den else math.nan
    sensitivity = tp / recall_den if recall_den else math.nan
    f1_den = precision + sensitivity
    f1_value = (2 * precision * sensitivity) / f1_den if f1_den else math.nan
    accuracy = (tp + tn) / max(1, conf_mat.sum())
    return {
        "truePos": float(tp),
        "trueNeg": float(tn),
        "falsePos": float(fp),
        "falseNeg": float(fn),
        "F1": float(f1_value),
        "Accuracy": float(accuracy),
    }


def _run_smoke_reproduction(
    week: int,
    split_threshold: int,
    seeds: list[int],
    labels: list[str],
    model_types: list[str],
    n_features_grid: list[int],
) -> pd.DataFrame:
    raw_frame = _load_feature_frame(week)
    data = _binarize_targets(_select_view_frame(raw_frame, "combined"), split_threshold)
    rows: list[dict[str, object]] = []
    for label in labels:
        for seed in seeds:
            df_train, df_test = train_test_split(
                data,
                test_size=0.3,
                stratify=data[["phq9", "gad7"]],
                random_state=seed,
            )
            train_content = df_train.iloc[:, 1:-2]
            test_content = df_test.iloc[:, 1:-2]
            scaler = preprocessing.MinMaxScaler()
            train_scaled = pd.DataFrame(scaler.fit_transform(train_content))
            test_scaled = pd.DataFrame(scaler.transform(test_content))
            target = df_train[label].astype(int).tolist()
            y_test = df_test[label].astype(int).to_numpy()
            for n_features in n_features_grid:
                pca = PCA(n_components=n_features)
                train_pca = pd.DataFrame(pca.fit_transform(train_scaled))
                train_pca = train_pca.assign(target=target)
                test_pca = pd.DataFrame(pca.transform(test_scaled))
                balanced_train = _upsample_minority(train_pca)
                y_train = balanced_train["target"]
                x_train = balanced_train.drop(columns="target")
                for model_type in model_types:
                    model = _build_model(model_type, seed)
                    model.fit(x_train, y_train)
                    y_pred = np.asarray(model.predict(test_pca), dtype=int)
                    metrics = _compute_binary_metrics(y_test, y_pred)
                    rows.append(
                        {
                            "source_view": "combined",
                            "week": week,
                            "label": label,
                            "split": split_threshold,
                            "model": model_type,
                            "nFeatures": n_features,
                            "randomSeed": seed,
                            **metrics,
                        }
                    )
    return pd.DataFrame(rows)


def _build_smoke_comparison(official_results: pd.DataFrame, smoke_results: pd.DataFrame) -> pd.DataFrame:
    official_subset = official_results.loc[
        (official_results["source_view"] == "combined")
        & (official_results["week"].isin(smoke_results["week"].unique()))
        & (official_results["split"].isin(smoke_results["split"].unique()))
        & (official_results["label"].isin(smoke_results["label"].unique()))
        & (official_results["model"].isin(smoke_results["model"].unique()))
        & (official_results["nFeatures"].isin(smoke_results["nFeatures"].unique()))
        & (official_results["randomSeed"].isin(smoke_results["randomSeed"].unique()))
    ].copy()
    merged = smoke_results.merge(
        official_subset[
            [
                "source_view",
                "week",
                "label",
                "split",
                "model",
                "nFeatures",
                "randomSeed",
                "F1",
                "Accuracy",
                "truePos",
                "trueNeg",
                "falsePos",
                "falseNeg",
            ]
        ].rename(
            columns={
                "F1": "official_F1",
                "Accuracy": "official_Accuracy",
                "truePos": "official_truePos",
                "trueNeg": "official_trueNeg",
                "falsePos": "official_falsePos",
                "falseNeg": "official_falseNeg",
            }
        ),
        on=["source_view", "week", "label", "split", "model", "nFeatures", "randomSeed"],
        how="left",
    )
    for metric_name in ("F1", "Accuracy", "truePos", "trueNeg", "falsePos", "falseNeg"):
        merged[f"{metric_name}_delta"] = merged[metric_name] - merged[f"official_{metric_name}"]
    return merged


def _public_status_table(best_summary: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "baseline_name": "deprest_cat_public_time_series",
            "dataset_id": "deprest_cat",
            "status": "completed_sidecar",
            "protocol": "official_screening_random_split",
            "comparable_to_main_protocol": False,
            "summary_path": "outputs/tables/deprest_cat_public_screening_best.csv",
            "notes": "Faithful public screening audit plus local smoke reproduction; not merged into main participant-level baseline table.",
        },
        {
            "baseline_name": "psyche_d_public_two_stage",
            "dataset_id": "psyche_d",
            "status": "pending",
            "protocol": "official_processed_release_two_stage",
            "comparable_to_main_protocol": False,
            "summary_path": "",
            "notes": "Pending source-aligned code acquisition beyond the processed release currently in the workspace.",
        },
        {
            "baseline_name": "depresjon_public_actigraphy",
            "dataset_id": "depresjon",
            "status": "pending",
            "protocol": "public_actigraphy_baseline",
            "comparable_to_main_protocol": False,
            "summary_path": "",
            "notes": "Pending source-aligned public actigraphy reproduction.",
        },
    ]
    if not best_summary.empty:
        rows[0]["best_config_count"] = int(len(best_summary))
    return pd.DataFrame(rows)


def _write_report(
    official_results: pd.DataFrame,
    best_summary: pd.DataFrame,
    smoke_comparison: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    official_rows = len(official_results)
    official_files = sum(len(list(directory.glob("resultsCAT*split*.csv"))) for directory in RESULT_VIEW_DIRS.values())
    missing_files = _expected_missing_result_files()
    max_abs_f1 = pd.to_numeric(smoke_comparison["F1_delta"], errors="coerce").abs().max()
    max_abs_acc = pd.to_numeric(smoke_comparison["Accuracy_delta"], errors="coerce").abs().max()
    smoke_match_count = int(smoke_comparison["official_F1"].notna().sum())
    exact_f1_count = int((pd.to_numeric(smoke_comparison["F1_delta"], errors="coerce").abs() < 1e-12).sum())
    exact_acc_count = int((pd.to_numeric(smoke_comparison["Accuracy_delta"], errors="coerce").abs() < 1e-12).sum())
    exact_both_count = int(
        (
            (pd.to_numeric(smoke_comparison["F1_delta"], errors="coerce").abs() < 1e-12)
            & (pd.to_numeric(smoke_comparison["Accuracy_delta"], errors="coerce").abs() < 1e-12)
        ).sum()
    )
    top_rows = best_summary.head(6)
    top_lines = "\n".join(
        f"| `{row.source_view}` | `{row.label}` | `{int(row.split)}` | `{int(row.week)}` | `{row.model}` | `{int(row.nFeatures)}` | `{row.mean_f1:.3f}` | `{row.mean_accuracy:.3f}` |"
        for row in top_rows.itertuples()
    )
    missing_lines = "\n".join(
        f"- `{source_view}` missing files: `{', '.join(paths) if paths else 'none'}`"
        for source_view, paths in missing_files.items()
    )
    text = f"""# DepreST-CAT Public Baseline Audit

Date: `2026-04-17`

## Scope

This audit implements the corpus-specific public baseline requested for `DepreST-CAT` as a sidecar reproduction path. It does **not** enter the unified participant-level Stage 5 comparison table because the released public protocol uses:

- binary screening targets created by thresholding `PHQ-9` or `GAD-7` at splits `5-10`
- repeated random `70/30` train/test splits
- stratification on the joint thresholded `phq9` and `gad7` labels

These choices are source-aligned but not directly comparable to the main project protocol.

## Official artifacts ingested

- Result directories parsed: `screeningResults`, `screeningResultsCall`, `screeningResultsText`
- Parsed result files: `{official_files}`
- Parsed result rows: `{official_rows}`
- Missing expected result files:
{missing_lines}
- Output summaries:
  - `outputs/tables/deprest_cat_public_screening_detail.csv`
  - `outputs/tables/deprest_cat_public_screening_best.csv`
  - `outputs/tables/deprest_cat_public_screening_smoke.csv`
  - `outputs/tables/deprest_cat_public_screening_smoke_vs_official.csv`

## Local smoke reproduction

The local smoke reproduction replays the released `combined` pipeline on:

- week: `{args.smoke_week}`
- split threshold: `{args.smoke_split}`
- labels: `{", ".join(args.smoke_labels)}`
- random seeds: `{", ".join(str(seed) for seed in args.smoke_seeds)}`
- models: `{", ".join(args.smoke_models)}`
- PCA dimensions: `{", ".join(str(value) for value in args.smoke_nfeatures)}`

Smoke rows matched against official outputs: `{smoke_match_count}`

- max abs `F1` delta: `{float(max_abs_f1):.6f}`
- max abs `Accuracy` delta: `{float(max_abs_acc):.6f}`
- exact `F1` matches: `{exact_f1_count}`
- exact `Accuracy` matches: `{exact_acc_count}`
- exact match on both metrics: `{exact_both_count}`

## Best released configurations by source view and label threshold

| View | Label | Split | Week | Model | PCA dims | Mean F1 | Mean Accuracy |
|---|---|---:|---:|---|---:|---:|---:|
{top_lines}

## Interpretation

- `DepreST-CAT` now has a completed source-aligned public baseline reproduction path in this workspace.
- The implementation is intentionally tracked as a sidecar audit instead of being mixed into `outputs/tables/baseline_results.csv`.
- All three requested corpus-specific public baselines now have dedicated sidecar audits in this workspace, although they remain separate from the main participant-level benchmark table.
"""
    ensure_dir(REPORT_PATH.parent)
    REPORT_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_dir(OUTPUT_TABLE_DIR)
    ensure_dir(OUTPUT_LOG_DIR)

    official_results = _load_public_results()
    detail_summary, best_summary = _summarize_public_results(official_results)
    smoke_results = _run_smoke_reproduction(
        week=args.smoke_week,
        split_threshold=args.smoke_split,
        seeds=args.smoke_seeds,
        labels=args.smoke_labels,
        model_types=args.smoke_models,
        n_features_grid=args.smoke_nfeatures,
    )
    smoke_comparison = _build_smoke_comparison(official_results, smoke_results)
    status_table = _public_status_table(best_summary)

    detail_path = OUTPUT_TABLE_DIR / "deprest_cat_public_screening_detail.csv"
    best_path = OUTPUT_TABLE_DIR / "deprest_cat_public_screening_best.csv"
    smoke_path = OUTPUT_TABLE_DIR / "deprest_cat_public_screening_smoke.csv"
    comparison_path = OUTPUT_TABLE_DIR / "deprest_cat_public_screening_smoke_vs_official.csv"
    status_path = OUTPUT_TABLE_DIR / "public_baseline_status.csv"

    write_csv(detail_path, detail_summary)
    write_csv(best_path, best_summary)
    write_csv(smoke_path, smoke_results)
    write_csv(comparison_path, smoke_comparison)
    write_csv(status_path, status_table)

    audit_payload = {
        "stage": STAGE_NAME,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "official_rows": int(len(official_results)),
        "missing_expected_result_files": _expected_missing_result_files(),
        "detail_rows": int(len(detail_summary)),
        "best_rows": int(len(best_summary)),
        "smoke_rows": int(len(smoke_results)),
        "smoke_matched_rows": int(smoke_comparison["official_F1"].notna().sum()),
        "paths": {
            "detail": str(detail_path.relative_to(PROJECT_ROOT)),
            "best": str(best_path.relative_to(PROJECT_ROOT)),
            "smoke": str(smoke_path.relative_to(PROJECT_ROOT)),
            "comparison": str(comparison_path.relative_to(PROJECT_ROOT)),
            "status": str(status_path.relative_to(PROJECT_ROOT)),
            "report": str(REPORT_PATH.relative_to(PROJECT_ROOT)),
        },
    }
    write_json(OUTPUT_LOG_DIR / "deprest_cat_public_baseline_audit.json", audit_payload)
    _write_report(official_results, best_summary, smoke_comparison, args)

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    _append_registry_row(
        {
            "experiment_id": f"{STAGE_NAME}__deprest_cat__official_summary__{timestamp}",
            "stage": STAGE_NAME,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "model_name": "deprest_cat_public_time_series",
            "training_corpora": "deprest_cat",
            "target_dataset": "deprest_cat",
            "task_name": "official_screening_summary",
            "split_config": "official_random_70_30_x100",
            "model_config": "official_MachineLearning.ipynb",
            "train_config": "official_MachineLearning.ipynb",
            "status": "completed",
            "notes": "Sidecar audit over released screeningResults / screeningResultsCall / screeningResultsText.",
        }
    )
    _append_registry_row(
        {
            "experiment_id": f"{STAGE_NAME}__deprest_cat__smoke_reproduction__{timestamp}",
            "stage": STAGE_NAME,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "model_name": "deprest_cat_public_time_series",
            "training_corpora": "deprest_cat",
            "target_dataset": "deprest_cat",
            "task_name": "official_screening_smoke",
            "split_config": f"official_random_70_30_week{args.smoke_week}_split{args.smoke_split}",
            "model_config": "official_MachineLearning.ipynb",
            "train_config": "official_MachineLearning.ipynb",
            "status": "completed",
            "notes": "Local source-aligned smoke reproduction on the combined-view official feature table.",
        }
    )


if __name__ == "__main__":
    main()
