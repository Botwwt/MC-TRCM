from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocess.schema import (
    ANCHOR_COLUMNS,
    LABEL_COLUMNS,
    PARTICIPANT_COLUMNS,
    RAW_DAILY_COLUMNS,
    build_concept_columns,
    build_window_columns,
)
from src.utils.constants import CONCEPT_DEFINITIONS, PROJECT_ROOT
from src.utils.io import ensure_dir, write_csv, write_json, write_parquet_with_fallback


DATASET_ID = "depresjon"
RAW_BASE = PROJECT_ROOT / "data_raw" / DATASET_ID / "extracted" / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build canonical Depresjon tables.")
    parser.add_argument("--min-complete-days", type=int, default=5)
    parser.add_argument("--low-coverage-threshold", type=int, default=1200)
    return parser.parse_args()


def safe_float(value) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def compute_age_midpoint(age_band: str | float | None) -> float | None:
    if age_band is None or pd.isna(age_band):
        return None
    if isinstance(age_band, str) and "-" in age_band:
        left, right = age_band.split("-", 1)
        return (float(left) + float(right)) / 2.0
    try:
        return float(age_band)
    except Exception:
        return None


def winsorize_series(values: pd.Series, lower_q: float, upper_q: float) -> pd.Series:
    return values.clip(lower=lower_q, upper=upper_q)


def summarize_subject_daily(activity_frame: pd.DataFrame, low_coverage_threshold: int) -> pd.DataFrame:
    activity_frame = activity_frame.copy()
    activity_frame["hour"] = activity_frame["timestamp"].dt.hour
    activity_frame["is_night"] = activity_frame["hour"].isin([0, 1, 2, 3, 4, 5, 22, 23])

    grouped = activity_frame.groupby("date", sort=True)
    daily = grouped.agg(
        activity_sum=("activity_winsorized", "sum"),
        activity_mean=("activity_winsorized", "mean"),
        activity_sd=("activity_winsorized", "std"),
        activity_max=("activity_winsorized", "max"),
        minute_count=("activity_winsorized", "count"),
        nonzero_minutes=("activity_winsorized", lambda s: int((s > 0).sum())),
        night_sum=("activity_winsorized", lambda s: float(activity_frame.loc[s.index, "activity_winsorized"][activity_frame.loc[s.index, "is_night"]].sum())),
        day_sum=("activity_winsorized", lambda s: float(activity_frame.loc[s.index, "activity_winsorized"][~activity_frame.loc[s.index, "is_night"]].sum())),
    ).reset_index()

    daily["activity_sd"] = daily["activity_sd"].fillna(0.0)
    daily["activity_cv"] = np.where(daily["activity_mean"] > 0, daily["activity_sd"] / daily["activity_mean"], 0.0)
    daily["activity_nonzero_ratio"] = np.where(daily["minute_count"] > 0, daily["nonzero_minutes"] / daily["minute_count"], np.nan)
    daily["activity_day_night_ratio"] = np.where(daily["night_sum"] > 0, daily["day_sum"] / daily["night_sum"], np.nan)
    daily["daily_missing_rate"] = 1.0 - np.clip(daily["minute_count"] / 1440.0, 0, 1)
    daily["qa_flag"] = np.where(daily["minute_count"] < low_coverage_threshold, "low_coverage", "ok")
    daily["is_weekend"] = pd.to_datetime(daily["date"]).dt.dayofweek >= 5
    daily["study_day"] = np.arange(1, len(daily) + 1)
    return daily


def entropy_from_values(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    values = values[values >= 0]
    total = values.sum()
    if len(values) == 0 or total <= 0:
        return 0.0
    probabilities = values / total
    probabilities = probabilities[probabilities > 0]
    return float(-(probabilities * np.log(probabilities)).sum())


def slope_from_values(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    return float(np.polyfit(x, values, 1)[0])


def regularity_from_values(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) < 2:
        return 1.0
    return float(1.0 / (1.0 + np.nanstd(np.diff(values))))


def compute_window_feature_bundle(window_daily: pd.DataFrame, full_daily: pd.DataFrame) -> dict[str, float | int | None]:
    values = window_daily["activity_sum"].to_numpy(dtype=float)
    day_night_values = window_daily["activity_day_night_ratio"].dropna().to_numpy(dtype=float)
    result = {
        "mean": float(np.nanmean(values)) if len(values) else np.nan,
        "sd": float(np.nanstd(values, ddof=0)) if len(values) else np.nan,
        "cv": float(np.nanstd(values, ddof=0) / np.nanmean(values)) if len(values) and np.nanmean(values) > 0 else 0.0,
        "slope": slope_from_values(values),
        "entropy": entropy_from_values(values),
        "day_night_ratio": float(np.nanmean(day_night_values)) if len(day_night_values) else np.nan,
        "weekend_shift": np.nan,
        "regularity": regularity_from_values(values),
        "interevent_cv": np.nan,
        "ratio": np.nan,
        "missing_ratio": float(np.nanmean(window_daily["daily_missing_rate"])) if len(window_daily) else np.nan,
    }

    weekend_values = window_daily.loc[window_daily["is_weekend"], "activity_sum"]
    weekday_values = window_daily.loc[~window_daily["is_weekend"], "activity_sum"]
    if len(weekend_values) and len(weekday_values):
        result["weekend_shift"] = float(weekend_values.mean() - weekday_values.mean())

    full_mean = float(np.nanmean(full_daily["activity_sum"])) if len(full_daily) else np.nan
    if not np.isnan(full_mean) and full_mean != 0 and not np.isnan(result["mean"]):
        result["ratio"] = float(result["mean"] / full_mean)
    return result


def build_empty_window_row(subject_id: str, task_name: str, anchor_id: str) -> dict:
    row = {column: np.nan for column in build_window_columns()}
    row["dataset_id"] = DATASET_ID
    row["subject_id"] = subject_id
    row["anchor_id"] = anchor_id
    row["task_name"] = task_name
    return row


def build_window_row(subject_id: str, anchor_id: str, task_name: str, anchor_date: pd.Timestamp, daily_frame: pd.DataFrame, static_row: pd.Series) -> dict:
    row = build_empty_window_row(subject_id, task_name, anchor_id)

    row["modality_mask_activity"] = 1
    row["modality_mask_sleep"] = 0
    row["modality_mask_communication"] = 0
    row["modality_mask_phone_use"] = 0
    row["modality_mask_mobility"] = 0
    row["modality_mask_static"] = 1
    row["modality_mask_symptom_context"] = 1 if pd.notna(static_row.get("madrs1")) else 0

    concept_mask_map = {
        "c1": 1,
        "c2": 1,
        "c3": 1,
        "c4": 1,
        "c5": 0,
        "c6": 0,
        "c7": 0,
        "c8": 1 if pd.notna(static_row.get("madrs1")) else 0,
    }
    for concept_id in CONCEPT_DEFINITIONS:
        row[f"concept_mask_{concept_id}"] = concept_mask_map[concept_id]

    row["feat_static_age"] = compute_age_midpoint(static_row.get("age"))
    row["feat_static_sex"] = safe_float(static_row.get("gender"))
    row["feat_static_baseline_severity"] = safe_float(static_row.get("madrs1"))
    row["feat_static_missing_rate"] = float(static_row.isna().mean())

    eligible_daily = daily_frame.loc[daily_frame["date"] <= anchor_date].copy()
    window_targets = {"short": 3, "medium": 7, "long": 14}
    for window_name, target_days in window_targets.items():
        window_daily = eligible_daily.tail(target_days)
        bundle = compute_window_feature_bundle(window_daily, eligible_daily)
        row[f"feat_activity_mean_{window_name}"] = bundle["mean"]
        row[f"feat_activity_sd_{window_name}"] = bundle["sd"]
        row[f"feat_activity_cv_{window_name}"] = bundle["cv"]
        row[f"feat_activity_slope_{window_name}"] = bundle["slope"]
        row[f"feat_activity_entropy_{window_name}"] = bundle["entropy"]
        row[f"feat_activity_day_night_ratio_{window_name}"] = bundle["day_night_ratio"]
        row[f"feat_activity_weekend_shift_{window_name}"] = bundle["weekend_shift"]
        row[f"feat_activity_regularity_{window_name}"] = bundle["regularity"]
        row[f"feat_activity_interevent_cv_{window_name}"] = bundle["interevent_cv"]
        row[f"feat_activity_ratio_{window_name}"] = bundle["ratio"]
        row[f"feat_activity_missing_ratio_{window_name}"] = bundle["missing_ratio"]

    row["feat_symptom_context_mean_short"] = safe_float(static_row.get("madrs1"))
    row["feat_symptom_context_mean_medium"] = safe_float(static_row.get("madrs1"))
    row["feat_symptom_context_mean_long"] = safe_float(static_row.get("madrs1"))
    row["feat_symptom_context_missing_ratio_short"] = 0.0 if pd.notna(static_row.get("madrs1")) else 1.0
    row["feat_symptom_context_missing_ratio_medium"] = 0.0 if pd.notna(static_row.get("madrs1")) else 1.0
    row["feat_symptom_context_missing_ratio_long"] = 0.0 if pd.notna(static_row.get("madrs1")) else 1.0

    return row


def main() -> int:
    args = parse_args()
    ensure_dir(PROJECT_ROOT / "data_interim" / "subject_tables" / DATASET_ID)
    ensure_dir(PROJECT_ROOT / "data_interim" / "daily_tables" / DATASET_ID)
    ensure_dir(PROJECT_ROOT / "data_interim" / "window_tables" / DATASET_ID)
    ensure_dir(PROJECT_ROOT / "data_interim" / "concept_tables" / DATASET_ID)

    scores = pd.read_csv(RAW_BASE / "scores.csv")
    subject_files = sorted((RAW_BASE / "condition").glob("*.csv")) + sorted((RAW_BASE / "control").glob("*.csv"))
    if not subject_files:
        raise FileNotFoundError(f"No subject CSV files found under {RAW_BASE}")

    all_activities = []
    subject_frames: dict[str, pd.DataFrame] = {}
    for subject_path in subject_files:
        frame = pd.read_csv(subject_path)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
        frame["activity"] = pd.to_numeric(frame["activity"], errors="coerce")
        subject_id = subject_path.stem
        subject_frames[subject_id] = frame
        all_activities.append(frame["activity"].dropna().to_numpy(dtype=float))

    concatenated = np.concatenate(all_activities)
    lower_q = float(np.quantile(concatenated, 0.01))
    upper_q = float(np.quantile(concatenated, 0.99))

    participants_rows = []
    daily_rows = []
    anchor_rows = []
    label_rows = []
    window_rows = []
    included_subjects = []
    excluded_subjects = []

    for _, score_row in scores.iterrows():
        subject_id = score_row["number"]
        frame = subject_frames.get(subject_id)
        if frame is None:
            excluded_subjects.append({"subject_id": subject_id, "reason": "missing_activity_file"})
            continue

        invalid_timestamps = int(frame["timestamp"].isna().sum())
        frame = frame.dropna(subset=["timestamp", "date", "activity"]).copy()
        if frame.empty:
            excluded_subjects.append({"subject_id": subject_id, "reason": "unparseable_activity_file"})
            continue

        frame["activity_winsorized"] = winsorize_series(frame["activity"], lower_q, upper_q)
        daily = summarize_subject_daily(frame, args.low_coverage_threshold)
        if len(daily) < args.min_complete_days:
            excluded_subjects.append({"subject_id": subject_id, "reason": "insufficient_complete_days"})
            continue

        included_subjects.append(subject_id)
        baseline_group = "condition" if subject_id.startswith("condition_") else "control"
        participants_rows.append(
            {
                "dataset_id": DATASET_ID,
                "subject_id": subject_id,
                "subject_id_original": subject_id,
                "sex": score_row.get("gender"),
                "age": score_row.get("age"),
                "baseline_group": baseline_group,
                "baseline_severity": safe_float(score_row.get("madrs1")),
                "static_missing_rate": float(score_row.isna().mean()),
                "source_file": str((RAW_BASE / baseline_group / f"{subject_id}.csv").relative_to(PROJECT_ROOT)),
                "days_recorded": int(len(daily)),
                "invalid_timestamp_rows": invalid_timestamps,
                "winsorized_q01": lower_q,
                "winsorized_q99": upper_q,
            }
        )

        for _, row in daily.iterrows():
            daily_rows.append(
                {
                    "dataset_id": DATASET_ID,
                    "subject_id": subject_id,
                    "date": row["date"].date().isoformat(),
                    "study_day": int(row["study_day"]),
                    "is_weekend": bool(row["is_weekend"]),
                    "raw_activity_value": float(row["activity_sum"]),
                    "raw_sleep_value": np.nan,
                    "raw_comm_value": np.nan,
                    "raw_phone_use_value": np.nan,
                    "raw_mobility_value": np.nan,
                    "raw_symptom_value": safe_float(score_row.get("madrs1")),
                    "daily_missing_rate": float(row["daily_missing_rate"]),
                    "qa_flag": row["qa_flag"],
                    "source_file": str((RAW_BASE / baseline_group / f"{subject_id}.csv").relative_to(PROJECT_ROOT)),
                    "activity_mean": float(row["activity_mean"]),
                    "activity_sd": float(row["activity_sd"]),
                    "activity_cv": float(row["activity_cv"]),
                    "activity_nonzero_ratio": float(row["activity_nonzero_ratio"]),
                    "activity_day_night_ratio": float(row["activity_day_night_ratio"]) if pd.notna(row["activity_day_night_ratio"]) else np.nan,
                    "minute_count": int(row["minute_count"]),
                }
            )

        last_timestamp = frame["timestamp"].max()
        anchor_date = pd.Timestamp(last_timestamp).normalize()
        available_days = len(daily)

        def append_anchor(task_name: str, label_type: str, y_raw, class_label):
            anchor_id = f"{DATASET_ID}__{subject_id}__{task_name}__end"
            anchor_rows.append(
                {
                    "dataset_id": DATASET_ID,
                    "subject_id": subject_id,
                    "anchor_id": anchor_id,
                    "anchor_time": last_timestamp.isoformat(),
                    "task_name": task_name,
                    "label_type": label_type,
                    "label_time": last_timestamp.isoformat(),
                    "window_short_days": min(3, available_days),
                    "window_medium_days": min(7, available_days),
                    "window_long_days": min(14, available_days),
                    "window_mask_short": int(available_days >= 3),
                    "window_mask_medium": int(available_days >= 7),
                    "window_mask_long": int(available_days >= 14),
                }
            )
            label_rows.append(
                {
                    "dataset_id": DATASET_ID,
                    "subject_id": subject_id,
                    "anchor_id": anchor_id,
                    "task_name": task_name,
                    "label_type": label_type,
                    "y_raw": y_raw,
                    "y_std": np.nan,
                    "class_label": class_label,
                    "label_available": int(pd.notna(y_raw) if label_type != "binary" else True),
                }
            )
            window_rows.append(build_window_row(subject_id, anchor_id, task_name, anchor_date, daily, score_row))

        append_anchor("dep_binary", "binary", 1 if baseline_group == "condition" else 0, "condition" if baseline_group == "condition" else "control")
        if baseline_group == "condition" and pd.notna(score_row.get("madrs2")):
            append_anchor("madrs_reg", "continuous", float(score_row["madrs2"]), np.nan)

    participants_df = pd.DataFrame(participants_rows)
    daily_df = pd.DataFrame(daily_rows)
    anchors_df = pd.DataFrame(anchor_rows)
    labels_df = pd.DataFrame(label_rows)
    windows_df = pd.DataFrame(window_rows)
    concepts_df = pd.DataFrame(columns=build_concept_columns())

    for required, frame in [
        (PARTICIPANT_COLUMNS, participants_df),
        (RAW_DAILY_COLUMNS, daily_df),
        (ANCHOR_COLUMNS, anchors_df),
        (LABEL_COLUMNS, labels_df),
    ]:
        for column in required:
            if column not in frame.columns:
                frame[column] = np.nan

    subject_dir = PROJECT_ROOT / "data_interim" / "subject_tables" / DATASET_ID
    daily_dir = PROJECT_ROOT / "data_interim" / "daily_tables" / DATASET_ID
    window_dir = PROJECT_ROOT / "data_interim" / "window_tables" / DATASET_ID
    concept_dir = PROJECT_ROOT / "data_interim" / "concept_tables" / DATASET_ID

    write_csv(subject_dir / "participants.csv", participants_df)
    write_csv(daily_dir / "raw_daily.csv", daily_df)
    write_csv(window_dir / "anchors.csv", anchors_df)
    write_csv(window_dir / "labels.csv", labels_df)
    windows_record = write_parquet_with_fallback(window_dir / "windows_wide.parquet", windows_df)
    concepts_record = write_parquet_with_fallback(concept_dir / "concepts.parquet", concepts_df)
    write_json(
        window_dir / "splits.json",
        {
            "dataset_id": DATASET_ID,
            "split_name": "pending_generation",
            "seed": 20260416,
            "train_subjects": [],
            "valid_subjects": [],
            "test_subjects": [],
            "audit": {
                "participant_overlap_detected": False,
                "notes": "Placeholder manifest retained until Stage 4 split generation.",
            },
        },
    )

    audit_payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": DATASET_ID,
        "raw_subjects_in_scores": int(len(scores)),
        "activity_files_detected": int(len(subject_files)),
        "included_subjects": int(len(included_subjects)),
        "excluded_subjects": excluded_subjects,
        "window_export": windows_record,
        "concept_export": concepts_record,
        "winsorize_q01": lower_q,
        "winsorize_q99": upper_q,
        "assumptions": [
            "End-of-recording timestamp is used as the anchor for dep_binary and madrs_reg tasks.",
            "madrs_reg uses madrs2 as the continuous target and madrs1 as static baseline context when available.",
            "Daily activity summaries are built from winsorized minute-level actigraphy.",
        ],
    }
    write_json(PROJECT_ROOT / "outputs" / "logs" / "depresjon_preprocess_audit.json", audit_payload)

    print(
        f"Depresjon preprocessing complete: participants={len(participants_df)} "
        f"daily_rows={len(daily_df)} anchors={len(anchors_df)} labels={len(labels_df)}"
    )
    print(
        f"Window export={windows_record['format']} | Concept export={concepts_record['format']} | "
        f"winsorize=({lower_q:.3f}, {upper_q:.3f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
