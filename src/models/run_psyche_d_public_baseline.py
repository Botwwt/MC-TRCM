from __future__ import annotations

import argparse
import importlib.util
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.constants import PROJECT_ROOT
from src.utils.io import ensure_dir, write_csv, write_json

STAGE_NAME = "stage_05_public_baseline"
PSYCHE_REPO_ROOT = PROJECT_ROOT / "data_raw" / "psyche_d" / "extracted" / "PSYCHE-D-main"
DEFAULT_DATA_PATH = PROJECT_ROOT / "data_raw" / "psyche_d" / "anon_processed_df_parquet"
OUTPUT_TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
OUTPUT_LOG_DIR = PROJECT_ROOT / "outputs" / "logs"
REPORT_PATH = PROJECT_ROOT / "reports" / "qa" / "psyche_d_public_baseline_audit.md"
REGISTRY_PATH = OUTPUT_LOG_DIR / "experiment_registry.csv"
PUBLIC_STATUS_PATH = OUTPUT_TABLE_DIR / "public_baseline_status.csv"

# Copied from the official PSYCHE-D release at `combined_pipeline.py`.
PHASE_1_BASE_COLS = [
    "sleep__main_start_hour_adj__score",
    "sleep__main_start_hour_adj__intercept",
    "sleep__main_start_hour_adj__coeff",
    "sleep_main_start_hour_adj_median",
    "sleep_main_start_hour_adj_iqr",
    "sleep_main_start_hour_adj_range",
    "sleep__total_asleep_minutes__score",
    "sleep__total_asleep_minutes__intercept",
    "sleep__total_asleep_minutes__coeff",
    "sleep__awake__sum__score",
    "sleep__awake__sum__intercept",
    "sleep__awake__sum__coeff",
    "sleep__nap_count__score",
    "sleep__nap_count__intercept",
    "sleep__nap_count__coeff",
    "sleep__total_asleep_minutes__score_",
    "sleep__total_asleep_minutes__intercept_",
    "sleep__total_asleep_minutes__coeff_",
    "sleep__main_efficiency__score_",
    "sleep__main_efficiency__intercept_",
    "sleep__main_efficiency__coeff_",
    "sleep__awake__sum__score_",
    "sleep__awake__sum__intercept_",
    "sleep__awake__sum__coeff_",
    "sleep__total_in_bed_minutes__score_",
    "sleep__total_in_bed_minutes__intercept_",
    "sleep__total_in_bed_minutes__coeff_",
    "steps__awake__sum__score_",
    "steps__awake__sum__intercept_",
    "steps__awake__sum__coeff_",
    "steps__mvpa__sum__score_",
    "steps__mvpa__sum__intercept_",
    "steps__mvpa__sum__coeff_",
    "steps__light_activity__sum__score_",
    "steps__light_activity__sum__intercept_",
    "steps__light_activity__sum__coeff_",
    "steps_mvpa_sum_recent",
    "sleep_asleep_mean_recent",
    "sleep_in_bed_mean_recent",
    "sleep_ratio_asleep_in_bed_mean_recent",
    "steps_lpa_sum_recent",
]

SELECTED_COLS_1 = {
    2732: [
        "trauma",
        "comorbid_gout",
        "nonmed_stop",
        "life_meditation",
        "insurance",
        "meds_migraine",
        "sex",
        "educ",
        "pregnant",
        "med_nonmed_dnu",
        "comorbid_arthritis",
        "race_asian",
        "weight",
        "money_assistance",
        "num_migraine_days",
        "med_stop",
        "household",
        "med_dose",
        "med_start",
        "comorbid_migraines",
        "birthyear",
        "comorbid_neuropathic",
        "bmi",
        "height",
        "race_hispanic",
        "race_black",
        "money",
        "sleep_main_start_hour_adj_range",
    ],
    9845: [
        "trauma",
        "comorbid_cancer",
        "comorbid_gout",
        "nonmed_stop",
        "life_meditation",
        "insurance",
        "meds_migraine",
        "sex",
        "educ",
        "med_nonmed_dnu",
        "comorbid_arthritis",
        "weight",
        "money_assistance",
        "num_migraine_days",
        "med_stop",
        "sleep_ratio_asleep_in_bed_mean_recent",
        "med_dose",
        "med_start",
        "comorbid_migraines",
        "birthyear",
        "comorbid_neuropathic",
        "comorbid_diabetes_typ1",
        "bmi",
        "height",
        "race_hispanic",
        "race_black",
        "money",
        "sleep_main_start_hour_adj_range",
    ],
    3264: [
        "educ",
        "weight",
        "comorbid_migraines",
        "bmi",
        "comorbid_neuropathic",
        "med_dose",
        "birthyear",
        "race_black",
        "trauma",
        "comorbid_arthritis",
        "nonmed_stop",
        "comorbid_cancer",
        "race_hispanic",
        "money_assistance",
        "money",
        "height",
        "race_white",
        "med_start",
        "meds_migraine",
        "insurance",
        "num_migraine_days",
        "sex",
    ],
    4859: [
        "trauma",
        "comorbid_cancer",
        "comorbid_gout",
        "nonmed_stop",
        "life_meditation",
        "insurance",
        "meds_migraine",
        "sex",
        "educ",
        "pregnant",
        "comorbid_arthritis",
        "weight",
        "money_assistance",
        "num_migraine_days",
        "med_stop",
        "household",
        "med_dose",
        "med_start",
        "comorbid_migraines",
        "birthyear",
        "comorbid_neuropathic",
        "bmi",
        "height",
        "race_hispanic",
        "race_black",
        "money",
        "sleep__hypersomnia_count_",
        "comorbid_osteoporosis",
    ],
    9225: [
        "educ",
        "weight",
        "comorbid_migraines",
        "bmi",
        "med_nonmed_dnu",
        "comorbid_neuropathic",
        "med_dose",
        "sleep_main_start_hour_adj_range",
        "comorbid_ms",
        "birthyear",
        "race_black",
        "trauma",
        "comorbid_arthritis",
        "comorbid_cancer",
        "race_hispanic",
        "money_assistance",
        "money",
        "height",
        "race_white",
        "med_start",
        "meds_migraine",
        "insurance",
        "num_migraine_days",
        "sex",
        "comorbid_gout",
        "med_stop",
    ],
}

SELECTED_PARAMS_1 = {
    2732: {"n_estimators": 115, "max_depth": 4, "drop_rate": 0.2},
    9845: {"n_estimators": 140, "max_depth": 7, "drop_rate": 0.05},
    3264: {"n_estimators": 50, "max_depth": 4, "drop_rate": 0.0},
    4859: {"n_estimators": 85, "max_depth": 3, "drop_rate": 0.0},
    9225: {"n_estimators": 160, "max_depth": 3, "drop_rate": 0.15},
}

SELECTED_COLS_2 = {
    2732: [
        "sleep__awake__sum__coeff_",
        "sex",
        "insurance",
        "med_start",
        "med_stop",
        "med_dose",
        "nonmed_stop",
        "life_meditation",
        "life_stress",
        "med_nonmed_dnu",
        "life_red_stop_alcoh",
        "cat_m0",
        "cat_m2",
    ],
    9845: [
        "comorbid_arthritis",
        "race_black",
        "trauma",
        "med_start",
        "med_stop",
        "med_dose",
        "nonmed_stop",
        "life_meditation",
        "life_stress",
        "life_red_stop_alcoh",
        "proba_cat_0_m2",
        "cat_m0",
        "cat_m2",
    ],
    3264: [
        "comorbid_neuropathic",
        "sex",
        "race_black",
        "insurance",
        "med_start",
        "med_stop",
        "med_dose",
        "nonmed_start",
        "nonmed_stop",
        "life_meditation",
        "life_red_stop_alcoh",
        "cat_m0",
        "cat_m1",
    ],
    4859: [
        "sex",
        "insurance",
        "med_start",
        "med_stop",
        "med_dose",
        "nonmed_stop",
        "life_meditation",
        "med_nonmed_dnu",
        "life_activity_eating",
        "life_red_stop_alcoh",
        "proba_cat_1_m2",
        "cat_m0",
        "cat_m1",
    ],
    9225: [
        "money_assistance",
        "sex",
        "insurance",
        "med_start",
        "med_stop",
        "med_dose",
        "nonmed_stop",
        "life_meditation",
        "life_stress",
        "life_activity_eating",
        "life_red_stop_alcoh",
        "cat_m0",
        "cat_m1",
    ],
}

OFFICIAL_RANDOM_SEEDS = [2732, 9845, 3264, 4859, 9225]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the source-aligned fixed-configuration PSYCHE-D public baseline."
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "predictions" / "public_baselines" / "psyche_d_public_two_stage",
    )
    parser.add_argument("--seeds", nargs="*", type=int, default=OFFICIAL_RANDOM_SEEDS)
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


def _load_official_utils():
    utils_path = PSYCHE_REPO_ROOT / "utils.py"
    spec = importlib.util.spec_from_file_location("psyche_d_public_utils", utils_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import official PSYCHE-D utils from {utils_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _phase_1_input_cols(frame: pd.DataFrame, utils_module) -> list[str]:
    selected_cols_union = sorted({column for columns in SELECTED_COLS_1.values() for column in columns})
    cols = PHASE_1_BASE_COLS + selected_cols_union + [
        column
        for column in frame.columns
        if column.startswith("life") or column.startswith("med") or column.startswith("nonmed")
    ]
    return sorted(set(utils_module.SCREENER_COLS + cols))


def _prepare_data(data_path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(data_path).copy()
    frame["user_id"] = [value.rsplit("_", 1)[0] for value in frame.index]
    frame["user_mth"] = [int(value.rsplit("_", 1)[1]) for value in frame.index]
    frame["ref_label"] = ~frame["phq9_cat_end"].isna()
    frame["user_qtr"] = (frame["user_mth"] - 1) // 3
    frame["user_id_qtr"] = frame.apply(lambda row: f"{row.user_id}_{row.user_qtr}", axis=1)
    valid_quarters = frame.loc[frame["ref_label"], "user_id_qtr"].unique()
    frame = frame.loc[frame["user_id_qtr"].isin(valid_quarters)].copy()
    frame["decl"] = frame["phq9_cat_end"] > frame["phq9_cat_start"]
    return frame


def _t_interval(values: list[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    point = float(np.mean(array))
    if len(array) < 2:
        return point, float("nan"), float("nan")
    lower, upper = stats.t.interval(confidence=0.95, df=len(array) - 1, loc=point, scale=stats.sem(array))
    return point, float(lower), float(upper)


def _update_public_status() -> None:
    if PUBLIC_STATUS_PATH.exists():
        status = pd.read_csv(PUBLIC_STATUS_PATH)
    else:
        status = pd.DataFrame(
            columns=[
                "baseline_name",
                "dataset_id",
                "status",
                "protocol",
                "comparable_to_main_protocol",
                "summary_path",
                "notes",
            ]
        )
    row = {
        "baseline_name": "psyche_d_public_two_stage",
        "dataset_id": "psyche_d",
        "status": "completed_sidecar",
        "protocol": "official_two_stage_participant_split",
        "comparable_to_main_protocol": False,
        "summary_path": "outputs/tables/psyche_d_public_two_stage_summary.csv",
        "notes": "Source-aligned fixed-configuration run based on official selected seeds, feature subsets, and LightGBM heads.",
    }
    status = status.loc[status["baseline_name"] != "psyche_d_public_two_stage"].copy()
    status = pd.concat([status, pd.DataFrame([row])], ignore_index=True)
    write_csv(PUBLIC_STATUS_PATH, status)


def _write_report(seed_metrics: pd.DataFrame, summary: pd.DataFrame) -> None:
    phase_2 = summary.loc[summary["phase"] == "phase_2"].copy()
    sensitivity_row = phase_2.loc[phase_2["metric_name"] == "sensitivity"].iloc[0]
    specificity_row = phase_2.loc[phase_2["metric_name"] == "specificity"].iloc[0]
    auroc_row = phase_2.loc[phase_2["metric_name"] == "auroc"].iloc[0]
    text = f"""# PSYCHE-D Public Baseline Audit

Date: `2026-04-17`

## Scope

This audit runs the released `PSYCHE-D` two-stage benchmark against the locally downloaded official parquet release. To keep the execution tractable while staying source-aligned, the wrapper uses the official fixed seeds, selected feature subsets, and selected LightGBM parameters published in `combined_pipeline.py`.

One compatibility patch is applied explicitly and transparently: the released fixed-feature dictionary references `sleep__hypersomnia_count_` for one seed even though that column is not included in the base `PHASE_1_INPUT_COLS` list. The wrapper therefore expands the phase-1 candidate set by the union of the released `SELECTED_COLS_1` features so the fixed-feature branch is executable as written.

## Execution mode

- Data path: `data_raw/psyche_d/anon_processed_df_parquet`
- Official code source: `data_raw/psyche_d/extracted/PSYCHE-D-main/combined_pipeline.py`
- Seeds: `{", ".join(str(seed) for seed in OFFICIAL_RANDOM_SEEDS)}`
- Output tables:
  - `outputs/tables/psyche_d_public_two_stage_seed_metrics.csv`
  - `outputs/tables/psyche_d_public_two_stage_summary.csv`

## Phase-2 headline metrics

- Sensitivity: `{sensitivity_row.point_estimate:.3f}` [{sensitivity_row.ci_lower:.3f}, {sensitivity_row.ci_upper:.3f}]
- Specificity: `{specificity_row.point_estimate:.3f}` [{specificity_row.ci_lower:.3f}, {specificity_row.ci_upper:.3f}]
- AUROC: `{auroc_row.point_estimate:.3f}` [{auroc_row.ci_lower:.3f}, {auroc_row.ci_upper:.3f}]

## Interpretation

- `PSYCHE-D` is no longer a source-availability blocker in this workspace.
- This sidecar reproduction remains separate from the main participant-level benchmark table because the official task definition and metrics differ from the study-wide canonical task setup.
- The official fixed-feature route is executable on the local processed parquet release.
"""
    ensure_dir(REPORT_PATH.parent)
    REPORT_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    ensure_dir(OUTPUT_TABLE_DIR)
    ensure_dir(OUTPUT_LOG_DIR)

    utils_module = _load_official_utils()
    data_original = _prepare_data(args.data_path)
    phase_1_cols = _phase_1_input_cols(data_original, utils_module)

    feature_importances_1: list[np.ndarray] = []
    feature_importances_2: list[np.ndarray] = []
    seed_rows: list[dict[str, object]] = []

    for seed in args.seeds:
        data = data_original.copy()
        data_train, data_test = utils_module.split_participant_data_random(data_original, n_users_test=800, seed=seed)
        data_train = data_train.loc[data_train["ref_label"]].copy()
        data_test = data_test.loc[data_test["ref_label"]].copy()

        x = data[phase_1_cols]
        y = data["phq9_cat_end"]
        x_train = x.loc[data_train.index, SELECTED_COLS_1[seed]]
        x_test = x.loc[data_test.index, SELECTED_COLS_1[seed]]
        y_train = y.loc[data_train.index]
        y_test = y.loc[data_test.index]
        x_generate = x.loc[data.loc[~data["ref_label"]].index, SELECTED_COLS_1[seed]]

        model_1 = utils_module.phase_1_model(
            x_train,
            y_train,
            n_estimators=SELECTED_PARAMS_1[seed]["n_estimators"],
            max_depth=SELECTED_PARAMS_1[seed]["max_depth"],
            drop_rate=SELECTED_PARAMS_1[seed]["drop_rate"],
            seed=seed,
            importance_type="split",
        )
        y_pred_1 = model_1.predict(x_test)
        y_proba_1 = model_1.predict_proba(x_test)
        perf_1 = utils_module.get_model_performance("one", y_test, y_pred_1, y_proba_1)
        for metric_name, metric_value in perf_1.items():
            seed_rows.append(
                {
                    "phase": "phase_1",
                    "metric_name": metric_name,
                    "seed": seed,
                    "metric_value": float(metric_value),
                }
            )
        feature_importances_1.append(np.asarray(model_1.feature_importances_))

        if len(x_generate.index) > 0:
            data.loc[data.loc[~data["ref_label"]].index, utils_module.PROBA_COLS] = model_1.predict_proba(x_generate)
            data.loc[data.loc[~data["ref_label"]].index, "phq9_cat_end"] = model_1.predict(x_generate)

        x_2, y_2 = utils_module.prep_phq_decline_data(data)
        x_train_2, x_test_2, y_train_2, y_test_2 = utils_module.split_train_test_set(
            x_2, y_2, data_train.index, data_test.index
        )
        x_train_2 = x_train_2[SELECTED_COLS_2[seed]]
        x_test_2 = x_test_2[SELECTED_COLS_2[seed]]
        model_2 = utils_module.phase_2_model(x_train_2, y_train_2, seed=seed, importance_type="split")
        y_pred_2 = model_2.predict(x_test_2)
        y_proba_2 = model_2.predict_proba(x_test_2)[:, 1]
        perf_2 = utils_module.get_model_performance("two", y_test_2, y_pred_2, y_proba_2)
        for metric_name, metric_value in perf_2.items():
            seed_rows.append(
                {
                    "phase": "phase_2",
                    "metric_name": metric_name,
                    "seed": seed,
                    "metric_value": float(metric_value),
                }
            )
        feature_importances_2.append(np.asarray(model_2.feature_importances_))

    seed_metrics = pd.DataFrame(seed_rows).sort_values(["phase", "metric_name", "seed"]).reset_index(drop=True)
    summary_rows: list[dict[str, object]] = []
    for (phase, metric_name), frame in seed_metrics.groupby(["phase", "metric_name"], dropna=False):
        point, lower, upper = _t_interval(frame["metric_value"].tolist())
        summary_rows.append(
            {
                "phase": phase,
                "metric_name": metric_name,
                "point_estimate": point,
                "ci_lower": lower,
                "ci_upper": upper,
                "n_seeds": len(frame.index),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(["phase", "metric_name"]).reset_index(drop=True)

    seed_path = OUTPUT_TABLE_DIR / "psyche_d_public_two_stage_seed_metrics.csv"
    summary_path = OUTPUT_TABLE_DIR / "psyche_d_public_two_stage_summary.csv"
    fi1_path = args.output_dir / "psyche_d_public_two_stage_feature_importances_phase1.csv"
    fi2_path = args.output_dir / "psyche_d_public_two_stage_feature_importances_phase2.csv"
    write_csv(seed_path, seed_metrics)
    write_csv(summary_path, summary)
    write_csv(fi1_path, pd.DataFrame(feature_importances_1))
    write_csv(fi2_path, pd.DataFrame(feature_importances_2))
    _update_public_status()
    _write_report(seed_metrics, summary)

    audit_payload = {
        "stage": STAGE_NAME,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_path": str(args.data_path.relative_to(PROJECT_ROOT)),
        "n_rows": int(len(data_original.index)),
        "n_columns": int(len(data_original.columns)),
        "seeds": args.seeds,
        "phase_1_input_columns": len(phase_1_cols),
        "paths": {
            "seed_metrics": str(seed_path.relative_to(PROJECT_ROOT)),
            "summary": str(summary_path.relative_to(PROJECT_ROOT)),
            "feature_importances_phase1": str(fi1_path.relative_to(PROJECT_ROOT)),
            "feature_importances_phase2": str(fi2_path.relative_to(PROJECT_ROOT)),
            "report": str(REPORT_PATH.relative_to(PROJECT_ROOT)),
        },
    }
    write_json(OUTPUT_LOG_DIR / "psyche_d_public_baseline_audit.json", audit_payload)

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    _append_registry_row(
        {
            "experiment_id": f"{STAGE_NAME}__psyche_d__public_two_stage__{timestamp}",
            "stage": STAGE_NAME,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "model_name": "psyche_d_public_two_stage",
            "training_corpora": "psyche_d",
            "target_dataset": "psyche_d",
            "task_name": "official_two_stage_summary",
            "split_config": "official_participant_random_n_users_test_800",
            "model_config": "official_combined_pipeline_fixed_selected_params",
            "train_config": "official_combined_pipeline_fixed_selected_params",
            "status": "completed",
            "notes": "Source-aligned fixed-configuration wrapper over the released PSYCHE-D code and parquet artifact.",
        }
    )


if __name__ == "__main__":
    main()
