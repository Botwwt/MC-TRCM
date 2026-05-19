from __future__ import annotations

import math
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results" / "final"
TABLE_ROOT = ROOT / "tables" / "final"
DATASET_LABELS = {
    "studentlife": "StudentLife",
    "deprest_cat": "DepreST-CAT",
    "psyche_d": "PSYCHE-D",
    "depresjon": "Depresjon",
    "obf": "OBF-Psychiatric",
}


def ensure_dirs() -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)


def assert_disjoint_split_sets(dataset_id: str) -> list[str]:
    path = ROOT / "data_interim" / "window_tables" / dataset_id / "splits.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    splits = {
        "train": {str(x) for x in payload.get("train_subjects", [])},
        "valid": {str(x) for x in payload.get("valid_subjects", [])},
        "test": {str(x) for x in payload.get("test_subjects", [])},
    }
    problems: list[str] = []
    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        overlap = splits[left] & splits[right]
        if overlap:
            problems.append(f"{dataset_id}: {left}/{right} overlap: {sorted(overlap)[:5]}")
    return problems


def _ok(condition: bool, message: str, problems: list[str], warnings: list[str], *, warning: bool = False) -> None:
    if condition:
        return
    if warning:
        warnings.append(message)
    else:
        problems.append(message)


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def main() -> None:
    ensure_dirs()
    if not (RESULT_ROOT / "strict_primary_metrics.csv").exists():
        raise FileNotFoundError(
            f"Missing {RESULT_ROOT / 'strict_primary_metrics.csv'}. Generate primary metric exports before integrity checks."
        )
    frame = pd.read_csv(RESULT_ROOT / "strict_primary_metrics.csv")
    problems: list[str] = []
    warnings: list[str] = []

    for dataset_id in DATASET_LABELS:
        problems.extend(assert_disjoint_split_sets(dataset_id))

    for _, row in frame.iterrows():
        metric = str(row["metric"])
        label_type = str(row["label_type"])
        _ok(
            (label_type == "continuous" and metric == "r2")
            or (label_type != "continuous" and metric == "balanced_accuracy"),
            f"Primary metric mismatch for {row['dataset_id']}/{row['task_name']}: {label_type} uses {metric}",
            problems,
            warnings,
        )
        n = int(row["mctrcm_n_seeds"])
        se = row["mctrcm_se"]
        sd = row["mctrcm_sd"]
        if n <= 1:
            _ok(pd.isna(se), f"SE should be missing for n=1 on {row['dataset_id']}/{row['task_name']}", problems, warnings)
            warnings.append(f"{row['dataset_id']}/{row['task_name']} has n=1; final mean +/- SE table is not statistically complete.")
        else:
            expected = float(sd) / math.sqrt(n)
            _ok(
                math.isclose(float(se), expected, rel_tol=1e-9, abs_tol=1e-12),
                f"SE formula mismatch for {row['dataset_id']}/{row['task_name']}",
                problems,
                warnings,
            )
        baseline_n = int(row["best_baseline_n_seeds"])
        baseline_se = row["best_baseline_se"]
        if baseline_n <= 1:
            _ok(
                pd.isna(baseline_se),
                f"Baseline SE should be missing for n<=1 on {row['dataset_id']}/{row['task_name']}",
                problems,
                warnings,
            )
            warnings.append(f"{row['dataset_id']}/{row['task_name']} baseline has n={baseline_n}; baseline SE is incomplete.")
        delta = row["delta"]
        if pd.notna(delta):
            expected_delta = float(row["mctrcm_mean"]) - float(row["best_baseline_mean"])
            _ok(
                math.isclose(float(delta), expected_delta, rel_tol=1e-9, abs_tol=1e-12),
                f"Delta direction/value mismatch for {row['dataset_id']}/{row['task_name']}",
                problems,
                warnings,
            )

    table_sources = {
        "dataset_task_summary": TABLE_ROOT / "dataset_task_summary.tex",
        "strict_regression_primary": TABLE_ROOT / "strict_regression_primary.tex",
        "strict_classification_primary": TABLE_ROOT / "strict_classification_primary.tex",
        "full_regression_metrics": TABLE_ROOT / "full_regression_metrics.tex",
        "full_classification_metrics": TABLE_ROOT / "full_classification_metrics.tex",
        "mctrcm_prediction_ensemble": TABLE_ROOT / "mctrcm_prediction_ensemble.tex",
    }
    for name, path in table_sources.items():
        _ok(path.exists(), f"Missing generated table: {path}", problems, warnings)

    figure_sources = {
        "strict_benchmark_delta": Path("figures/final/strict_benchmark_delta.png"),
        "strict_benchmark_delta_source": Path("figures/final/strict_benchmark_delta_source.csv"),
    }
    for name, path in figure_sources.items():
        _ok(path.exists(), f"Missing generated figure/source: {path}", problems, warnings)

    report = [
        "# Metric Integrity Report",
        "",
        "## Sources",
        "",
        f"- Primary metrics: `{_rel(RESULT_ROOT / 'strict_primary_metrics.csv')}`",
        f"- Per-seed metrics: `{_rel(RESULT_ROOT / 'per_seed_primary_metrics.csv')}`",
        f"- Completed seeded baseline rows: `{_rel(RESULT_ROOT / 'baseline_results_seeded.csv')}`",
        f"- MC-TRCM prediction ensemble: `{_rel(RESULT_ROOT / 'mctrcm_seed_ensemble_metrics.csv')}`",
        f"- Baseline selection: `{_rel(RESULT_ROOT / 'validation_selected_baselines.csv')}`",
        f"- Full metric matrix: `{_rel(RESULT_ROOT / 'full_model_metrics.csv')}`",
        "",
        "## Generated Tables",
        "",
    ]
    report.extend(f"- {name}: `{_rel(path)}`" for name, path in table_sources.items())
    report.extend(["", "## Generated Figures", ""])
    report.extend(f"- {name}: `{_rel(path)}`" for name, path in figure_sources.items())
    report.extend(["", "## Problems", ""])
    report.extend([f"- {problem}" for problem in problems] or ["- None"])
    report.extend(["", "## Warnings", ""])
    report.extend([f"- {warning}" for warning in warnings] or ["- None"])
    report.extend(
        [
            "",
            "## Unresolved Issues",
            "",
            "- LightGBM, XGBoost, and EBM were not rerun in the five-seed baseline pass because the current Python environment lacks `lightgbm`, `xgboost`, and `interpret`; a package install attempt was blocked by local permissions. Older one-seed rows are retained in the raw baseline file but are excluded from final five-seed tables.",
            "- No participant-bootstrap intervals are reported in the manuscript; the main uncertainty estimate is seed SE over final version-consistent exports.",
        ]
    )
    Path("reports/metric_integrity_final.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"problems={len(problems)} warnings={len(warnings)}")
    if problems:
        for problem in problems:
            print(f"ERROR: {problem}")
        raise SystemExit(1)
    for warning in warnings:
        print(f"WARNING: {warning}")


if __name__ == "__main__":
    main()
