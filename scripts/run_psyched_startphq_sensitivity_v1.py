from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_psyched_sensitivity_v1 import (  # noqa: E402
    SEEDS,
    TRAINED_MODELS,
    VARIANTS,
    WINDOW_ROOT,
    VariantBundle,
    apply_validation_postprocessing,
    build_base_frame,
    classification_metrics,
    descriptive_flag,
    feature_columns as all_feature_columns,
    make_variant_frame,
    participant_aggregated_classification,
    participant_regression,
    participant_row_macro_classification,
    prepare_xy,
    regression_metrics,
    run_model,
)
from scripts.run_revised_fairness_v1 import standard_error  # noqa: E402
from src.models.baselines import probe_dependencies  # noqa: E402


OUT_ROOT = ROOT / "results" / "psyched_startphq_sensitivity_v1"
TABLE_ROOT = ROOT / "tables" / "final"

SENSITIVITY_VARIANTS = ["current_binary", "current_multiclass", "continuous_delta_regression"]
FEATURE_SETS = ["FULL", "NO_START_PHQ", "START_PHQ_ONLY"]
START_PHQ_VALUE_COLUMNS = {
    "feat_static_baseline_severity",
    "feat_symptom_context_mean_short",
    "feat_symptom_context_mean_medium",
    "feat_symptom_context_mean_long",
}
START_PHQ_RELATED_COLUMNS = START_PHQ_VALUE_COLUMNS | {
    "feat_symptom_context_missing_ratio_short",
    "feat_symptom_context_missing_ratio_medium",
    "feat_symptom_context_missing_ratio_long",
    "modality_mask_symptom_context",
}


def ensure_dirs() -> None:
    for path in [OUT_ROOT, OUT_ROOT / "configs", OUT_ROOT / "predictions", OUT_ROOT / "calibration", OUT_ROOT / "logs", TABLE_ROOT]:
        path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def append_csv(path: Path, row: dict[str, Any], subset: list[str]) -> None:
    frame = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path, keep_default_na=False)
        out = pd.concat([old, frame], ignore_index=True).drop_duplicates(subset=subset, keep="last")
    else:
        out = frame
    out.to_csv(path, index=False)


def feature_columns_for_set(frame: pd.DataFrame, feature_set: str) -> list[str]:
    candidates = all_feature_columns(frame)
    if feature_set == "FULL":
        kept = candidates
    elif feature_set == "NO_START_PHQ":
        kept = [column for column in candidates if column not in START_PHQ_RELATED_COLUMNS]
    elif feature_set == "START_PHQ_ONLY":
        kept = [column for column in candidates if column in START_PHQ_VALUE_COLUMNS]
    else:
        raise KeyError(feature_set)
    if not kept:
        frame["__constant_feature__"] = 0.0
        return ["__constant_feature__"]
    return kept


def make_bundle(base: pd.DataFrame, variant: str, feature_set: str) -> VariantBundle:
    frame = make_variant_frame(base, variant)
    features = feature_columns_for_set(frame, feature_set)
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


def model_applicable(model_name: str, bundle: VariantBundle) -> bool:
    if model_name == "start_end_ancova_linear":
        return bundle.task_type == "continuous"
    return model_name in TRAINED_MODELS


def run_start_end_ancova(bundle: VariantBundle) -> dict[str, Any]:
    from sklearn.linear_model import LinearRegression

    start_col = "phq9_score_start"
    end_col = "phq9_score_end"
    train = bundle.train[[start_col, end_col]].dropna().copy()
    if train.empty:
        raise ValueError("No complete train rows for start/end PHQ ANCOVA model.")
    model = LinearRegression()
    model.fit(train[[start_col]].to_numpy(dtype=float), train[end_col].to_numpy(dtype=float))

    outputs: dict[str, Any] = {
        "details": {
            "formula": "end_phq9 ~ start_phq9 trained on train rows only; delta prediction = predicted_end - observed_start",
            "intercept": float(model.intercept_),
            "coef_start_phq9": float(model.coef_[0]),
        }
    }
    for split_name, split in (("valid", bundle.valid), ("test", bundle.test)):
        start = pd.to_numeric(split[start_col], errors="coerce").to_numpy(dtype=float)
        pred_end = model.predict(start.reshape(-1, 1))
        outputs[f"{split_name}_pred"] = pred_end - start
        outputs[f"{split_name}_pred_end"] = pred_end
    return outputs


def save_predictions(run_id: str, feature_set: str, bundle: VariantBundle, outputs: dict[str, Any], post: dict[str, Any] | None) -> Path:
    rows = []
    for split_name, split in (("valid", bundle.valid), ("test", bundle.test)):
        output = split[
            [
                "participant_id",
                "source_row_id",
                "anchor_order",
                "split",
                "phq9_score_start",
                "phq9_score_end",
                "delta_phq9",
                "target",
                "target_label",
            ]
        ].copy()
        output["variant"] = bundle.variant
        output["feature_set"] = feature_set
        output["task_type"] = bundle.task_type
        if bundle.task_type == "continuous":
            output["prediction_delta"] = outputs[f"{split_name}_pred"]
            if f"{split_name}_pred_end" in outputs:
                output["prediction_end_phq9"] = outputs[f"{split_name}_pred_end"]
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


def config_common(bundle: VariantBundle, feature_set: str, model_name: str, seed: int, run_id: str) -> dict[str, Any]:
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
        "feature_set": feature_set,
        "feature_columns": bundle.feature_columns,
        "start_phq_value_columns": sorted([column for column in bundle.feature_columns if column in START_PHQ_VALUE_COLUMNS]),
        "dropped_start_phq_related_columns": sorted(START_PHQ_RELATED_COLUMNS - set(bundle.feature_columns)),
        "selection_policy": "Feature set and model family summaries select by mean validation primary metric. Thresholds/calibration use validation labels only. Test is evaluated after fixed configuration.",
        "descriptive_only": bundle.descriptive_only,
        "descriptive_reason": bundle.descriptive_reason,
    }


def base_metric_row(
    run_id: str,
    seed: int,
    bundle: VariantBundle,
    feature_set: str,
    model_name: str,
    config_path: Path,
    pred_path: Path | None,
    calibration_path: Path,
    status: str = "ok",
    failure_reason: str = "",
) -> dict[str, Any]:
    row = {
        "run_id": run_id,
        "seed": seed,
        "dataset": "psyche_d",
        "variant": bundle.variant,
        "task_type": bundle.task_type,
        "feature_set": feature_set,
        "model_family": model_name,
        "n_train_rows": int(len(bundle.train)),
        "n_val_rows": int(len(bundle.valid)),
        "n_test_rows": int(len(bundle.test)),
        "n_train_participants": int(bundle.train["participant_id"].nunique()),
        "n_val_participants": int(bundle.valid["participant_id"].nunique()),
        "n_test_participants": int(bundle.test["participant_id"].nunique()),
        "val_primary": math.nan,
        "test_primary": math.nan,
        "status": status,
        "failure_reason": failure_reason,
        "postprocessing_type": "",
        "config_path": str(config_path.relative_to(ROOT)),
        "prediction_path": str(pred_path.relative_to(ROOT)) if pred_path is not None else "",
        "calibration_path": str(calibration_path.relative_to(ROOT)),
        "descriptive_only": bundle.descriptive_only,
        "descriptive_reason": bundle.descriptive_reason,
    }
    for metric in ["r2", "rmse", "mae", "spearman", "ba", "macro_f1", "auroc", "auprc", "brier", "ece"]:
        row[f"test_{metric}"] = math.nan
        row[f"uncalibrated_test_{metric}"] = math.nan
    return row


def prefix_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}test_{key}": value for key, value in metrics.items() if key in {"r2", "rmse", "mae", "spearman", "ba", "macro_f1", "auroc", "auprc", "brier", "ece"}}


def record_failure(bundle: VariantBundle, feature_set: str, model_name: str, seed: int, run_id: str, config_path: Path, exc: BaseException) -> None:
    log_path = OUT_ROOT / "logs" / f"{run_id}.error.log"
    log_path.write_text("".join(traceback.format_exception(exc)), encoding="utf-8")
    calibration_path = log_path
    row = base_metric_row(
        run_id,
        seed,
        bundle,
        feature_set,
        model_name,
        config_path,
        None,
        calibration_path,
        status="failed",
        failure_reason=f"{type(exc).__name__}: {exc}",
    )
    append_csv(OUT_ROOT / "seed_metrics.csv", row, ["run_id"])


def participant_metric_row(task_row: dict[str, Any], scheme: str, metrics: dict[str, float]) -> dict[str, Any]:
    row = {key: task_row[key] for key in task_row if key != "prediction_path"}
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


def evaluate_and_record(bundle: VariantBundle, feature_set: str, model_name: str, seed: int, outputs: dict[str, Any], run_id: str, config_path: Path) -> None:
    _, y_valid = prepare_xy(bundle, bundle.valid)
    _, y_test = prepare_xy(bundle, bundle.test)
    calibration_path = OUT_ROOT / "calibration" / f"{run_id}.json"
    if bundle.task_type == "continuous":
        valid_metrics = regression_metrics(y_valid, outputs["valid_pred"])
        test_metrics = regression_metrics(y_test, outputs["test_pred"])
        pred_path = save_predictions(run_id, feature_set, bundle, outputs, None)
        write_json(calibration_path, {"postprocessing_type": "none_regression", **outputs.get("details", {})})
        row = base_metric_row(run_id, seed, bundle, feature_set, model_name, config_path, pred_path, calibration_path)
        row.update(prefix_metrics("", test_metrics))
        row["val_primary"] = valid_metrics["primary"]
        row["test_primary"] = test_metrics["primary"]
        append_csv(OUT_ROOT / "seed_metrics.csv", row, ["run_id"])
        part = participant_regression(bundle.test, outputs["test_pred"])
        append_csv(OUT_ROOT / "participant_metrics.csv", participant_metric_row(row, "participant_mean_target_prediction", part["metrics"]), ["run_id", "participant_eval_scheme"])
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
    pred_path = save_predictions(run_id, feature_set, bundle, outputs, post)
    row = base_metric_row(run_id, seed, bundle, feature_set, model_name, config_path, pred_path, calibration_path)
    row.update(prefix_metrics("", post["calibrated_test_metrics"]))
    row.update(prefix_metrics("uncalibrated_", post["uncalibrated_test_metrics"]))
    row["val_primary"] = post["calibrated_valid_metrics"]["ba"]
    row["test_primary"] = post["calibrated_test_metrics"]["ba"]
    row["postprocessing_type"] = post["postprocessing_type"]
    append_csv(OUT_ROOT / "seed_metrics.csv", row, ["run_id"])

    row_macro = participant_row_macro_classification(bundle.test, post["calibrated_test_prob"], post["calibrated_test_pred"])
    append_csv(OUT_ROOT / "participant_metrics.csv", participant_metric_row(row, "participant_row_macro", row_macro), ["run_id", "participant_eval_scheme"])
    aggregated = participant_aggregated_classification(bundle.test, post["calibrated_test_prob"], post)
    append_csv(OUT_ROOT / "participant_metrics.csv", participant_metric_row(row, "participant_aggregated_majority_last_tie", aggregated["metrics"]), ["run_id", "participant_eval_scheme"])


def run_all(base: pd.DataFrame, args: argparse.Namespace) -> None:
    deps = probe_dependencies()
    write_json(OUT_ROOT / "logs" / "dependency_probe.json", deps)
    for variant in args.variants:
        for feature_set in args.feature_sets:
            bundle = make_bundle(base, variant, feature_set)
            for model_name in args.models:
                if not model_applicable(model_name, bundle):
                    continue
                dep_error: BaseException | None = None
                if model_name == "lightgbm" and not deps.get("lightgbm", False):
                    dep_error = RuntimeError("Missing dependency: lightgbm")
                if model_name == "xgboost" and not deps.get("xgboost", False):
                    dep_error = RuntimeError("Missing dependency: xgboost")
                if model_name == "ebm" and not deps.get("interpret", False):
                    dep_error = RuntimeError("Missing dependency: interpret")
                for seed in args.seeds:
                    run_id = f"psychedstart__{variant}__{feature_set}__{model_name}__seed{seed}"
                    config_path = OUT_ROOT / "configs" / f"{run_id}.json"
                    write_json(config_path, config_common(bundle, feature_set, model_name, seed, run_id))
                    if args.resume and (OUT_ROOT / "predictions" / f"{run_id}.csv").exists():
                        continue
                    if dep_error is not None:
                        record_failure(bundle, feature_set, model_name, seed, run_id, config_path, dep_error)
                        continue
                    try:
                        print(f"[psyched start-PHQ] {run_id}", flush=True)
                        if model_name == "start_end_ancova_linear":
                            outputs = run_start_end_ancova(bundle)
                        else:
                            outputs = run_model(bundle, model_name, seed)
                        evaluate_and_record(bundle, feature_set, model_name, seed, outputs, run_id, config_path)
                    except BaseException as exc:
                        record_failure(bundle, feature_set, model_name, seed, run_id, config_path, exc)


def write_feature_definitions(base: pd.DataFrame) -> None:
    frame = make_variant_frame(base, "current_binary")
    candidates = all_feature_columns(frame)
    definitions = {}
    for feature_set in FEATURE_SETS:
        cols = feature_columns_for_set(frame, feature_set)
        definitions[feature_set] = {
            "description": {
                "FULL": "Prepared PSYCHE-D feature view including sensor summaries, static context, concept/missingness masks, and start/baseline PHQ context.",
                "NO_START_PHQ": "FULL after dropping start/baseline PHQ value columns and symptom-context missingness/mask columns.",
                "START_PHQ_ONLY": "Only start/baseline PHQ value columns; no sensor, static demographic, or missingness columns.",
            }[feature_set],
            "included_columns": cols,
            "excluded_columns": [column for column in candidates if column not in cols],
            "start_phq_value_columns_included": sorted([column for column in cols if column in START_PHQ_VALUE_COLUMNS]),
            "start_phq_related_columns_excluded": sorted([column for column in START_PHQ_RELATED_COLUMNS if column not in cols]),
        }
    write_json(OUT_ROOT / "feature_set_definitions.json", definitions)


def summarize() -> pd.DataFrame:
    metrics = pd.read_csv(OUT_ROOT / "seed_metrics.csv", keep_default_na=False)
    metrics = metrics.loc[metrics["status"].eq("ok")].copy()
    for column in ["val_primary", "test_primary"]:
        metrics[column] = pd.to_numeric(metrics[column], errors="coerce")
    rows = []
    for keys, group in metrics.groupby(["variant", "feature_set", "model_family"], dropna=False):
        rows.append(
            {
                "variant": keys[0],
                "feature_set": keys[1],
                "model_family": keys[2],
                "mean_val_primary": float(group["val_primary"].mean()),
                "se_val_primary": standard_error(group["val_primary"]),
                "mean_test_primary": float(group["test_primary"].mean()),
                "se_test_primary": standard_error(group["test_primary"]),
                "n_seeds": int(group["seed"].nunique()),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_ROOT / "summary_metrics.csv", index=False)
    return summary


def best_row(summary: pd.DataFrame, variant: str, feature_set: str, models: list[str]) -> pd.Series | None:
    subset = summary.loc[
        summary["variant"].eq(variant)
        & summary["feature_set"].eq(feature_set)
        & summary["model_family"].isin(models)
    ].copy()
    if subset.empty:
        return None
    subset = subset.sort_values(["mean_val_primary", "mean_test_primary"], ascending=[False, False])
    return subset.iloc[0]


def fmt(mean: float, se: float) -> str:
    if pd.isna(mean):
        return "--"
    return f"{mean:.3f} $\\pm$ {se:.3f}"


def tex_escape(value: object) -> str:
    return str(value).replace("_", "\\_")


def generate_table(summary: pd.DataFrame) -> None:
    simple = ["elastic_net", "lightgbm", "xgboost", "ebm", "mlp"]
    lines = [
        "\\begin{table}[!tbp]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\caption{PSYCHE-D start-PHQ sensitivity diagnostic. Values are task-row test primary metric mean $\\pm$ SE over five seeds. Model family selection within each feature set uses validation primary metric only. The ANCOVA row predicts end PHQ-9 from start PHQ-9 on train rows and evaluates derived delta predictions.}",
        "\\label{tab:psyched_startphq_sensitivity}",
        "\\begin{tabularx}{\\linewidth}{@{}L{0.23\\linewidth}C{0.15\\linewidth}C{0.18\\linewidth}C{0.18\\linewidth}C{0.18\\linewidth}C{0.12\\linewidth}@{}}",
        "\\toprule",
        "Endpoint variant & Null & FULL best & NO-START-PHQ best & START-PHQ-only best & ANCOVA \\\\",
        "\\midrule",
    ]
    for variant in SENSITIVITY_VARIANTS:
        null = best_row(summary, variant, "FULL", ["null"])
        full = best_row(summary, variant, "FULL", simple)
        no_start = best_row(summary, variant, "NO_START_PHQ", simple)
        start_only = best_row(summary, variant, "START_PHQ_ONLY", simple)
        ancova = best_row(summary, variant, "START_PHQ_ONLY", ["start_end_ancova_linear"])
        lines.append(
            " & ".join(
                [
                    variant.replace("_", "\\_"),
                    fmt(null["mean_test_primary"], null["se_test_primary"]) if null is not None else "--",
                    f"{tex_escape(full['model_family'])} {fmt(full['mean_test_primary'], full['se_test_primary'])}" if full is not None else "--",
                    f"{tex_escape(no_start['model_family'])} {fmt(no_start['mean_test_primary'], no_start['se_test_primary'])}" if no_start is not None else "--",
                    f"{tex_escape(start_only['model_family'])} {fmt(start_only['mean_test_primary'], start_only['se_test_primary'])}" if start_only is not None else "--",
                    fmt(ancova["mean_test_primary"], ancova["se_test_primary"]) if ancova is not None else "--",
                ]
            )
            + " \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabularx}", "\\end{table}", ""])
    (TABLE_ROOT / "psyched_startphq_sensitivity.tex").write_text("\n".join(lines), encoding="utf-8")


def generate_report(summary: pd.DataFrame) -> None:
    simple = ["elastic_net", "lightgbm", "xgboost", "ebm", "mlp"]
    matrix = []
    for variant in SENSITIVITY_VARIANTS:
        row = {"variant": variant}
        for feature_set in FEATURE_SETS:
            best = best_row(summary, variant, feature_set, simple)
            row[f"{feature_set}_best_model"] = best["model_family"] if best is not None else ""
            row[f"{feature_set}_test_primary"] = best["mean_test_primary"] if best is not None else math.nan
            row[f"{feature_set}_val_primary"] = best["mean_val_primary"] if best is not None else math.nan
        ancova = best_row(summary, variant, "START_PHQ_ONLY", ["start_end_ancova_linear"])
        row["ancova_test_primary"] = ancova["mean_test_primary"] if ancova is not None else math.nan
        matrix.append(row)
    matrix_frame = pd.DataFrame(matrix)
    matrix_frame.to_csv(OUT_ROOT / "report_startphq_matrix.csv", index=False)

    lines = [
        "# PSYCHE-D Start-PHQ Sensitivity v1 Report",
        "",
        f"Generated: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "This suite reuses `data_interim/window_tables/psyche_d/splits.json`; no participant was reassigned.",
        "It is a follow-up diagnostic for baseline-score coupling and regression-to-mean risk, not a new primary model-selection pass.",
        "All trained models use train rows for fitting and validation rows for family/threshold/calibration selection. Test rows are evaluated after those choices are fixed.",
        "",
        "## Feature Sets",
        "",
        "- FULL: complete prepared PSYCHE-D feature view.",
        "- NO_START_PHQ: FULL after dropping baseline/start PHQ value columns and symptom-context missingness/mask columns.",
        "- START_PHQ_ONLY: start/baseline PHQ value columns only.",
        "- ANCOVA: train-only linear `end PHQ-9 ~ start PHQ-9`; reported as delta prediction after subtracting observed start PHQ-9.",
        "",
        "## Matrix",
        "",
        matrix_frame.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
    ]
    for _, row in matrix_frame.iterrows():
        variant = row["variant"]
        full = row["FULL_test_primary"]
        no_start = row["NO_START_PHQ_test_primary"]
        start_only = row["START_PHQ_ONLY_test_primary"]
        lines.append(
            f"- {variant}: FULL={full:.3f}, NO_START_PHQ={no_start:.3f}, START_PHQ_ONLY={start_only:.3f}. "
            f"The no-start comparison estimates how much task-row signal remains after removing direct start-PHQ context."
        )
    lines.extend(["", "## Failures", ""])
    metrics = pd.read_csv(OUT_ROOT / "seed_metrics.csv", keep_default_na=False)
    failures = metrics.loc[~metrics["status"].eq("ok"), ["run_id", "feature_set", "model_family", "variant", "failure_reason", "config_path", "calibration_path"]]
    lines.append(failures.to_markdown(index=False) if not failures.empty else "No failed seed runs recorded.")
    lines.append("")
    (OUT_ROOT / "EXPERIMENT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PSYCHE-D start-PHQ/no-start-PHQ sensitivity diagnostics.")
    parser.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    parser.add_argument("--variants", nargs="*", default=SENSITIVITY_VARIANTS, choices=SENSITIVITY_VARIANTS)
    parser.add_argument("--feature-sets", nargs="*", default=FEATURE_SETS, choices=FEATURE_SETS)
    parser.add_argument("--models", nargs="*", default=TRAINED_MODELS + ["start_end_ancova_linear"], choices=TRAINED_MODELS + ["start_end_ancova_linear"])
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_dirs()
    base = build_base_frame()
    write_feature_definitions(base)
    if not args.summary_only:
        run_all(base, args)
    if (OUT_ROOT / "seed_metrics.csv").exists():
        summary = summarize()
        generate_table(summary)
        generate_report(summary)
    print(f"Wrote PSYCHE-D start-PHQ sensitivity artifacts to {OUT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
