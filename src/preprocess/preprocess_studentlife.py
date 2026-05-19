from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadr

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


DATASET_ID = "studentlife"
RAW_BASE = PROJECT_ROOT / "data_raw" / DATASET_ID / "extracted" / "dataset_rds"

PHQ_MAP = {
    "Not at all": 0,
    "Several days": 1,
    "More than half the days": 2,
    "Nearly every day": 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build canonical StudentLife tables from selected RDS tables.")
    parser.add_argument("--min-complete-days", type=int, default=28)
    return parser.parse_args()


def read_rds(relative_path: str) -> pd.DataFrame:
    path = RAW_BASE / relative_path
    result = pyreadr.read_r(str(path))
    return next(iter(result.values())).copy()


def epoch_seconds_to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, unit="s", errors="coerce")


def phq_total_from_row(row: pd.Series) -> float | None:
    values = []
    for question in [f"Q{i}" for i in range(1, 10)]:
        response = row.get(question)
        if pd.isna(response):
            return None
        values.append(PHQ_MAP.get(str(response), np.nan))
    if any(pd.isna(values)):
        return None
    return float(np.sum(values))


def phq_category(score: float) -> tuple[int, str]:
    if score <= 4:
        return 0, "minimal"
    if score <= 9:
        return 1, "mild"
    if score <= 14:
        return 2, "moderate"
    if score <= 19:
        return 3, "moderately_severe"
    return 4, "severe"


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


def generic_window_bundle(window_daily: pd.DataFrame, full_daily: pd.DataFrame, value_col: str, missing_col: str) -> dict[str, float | None]:
    values = window_daily[value_col].to_numpy(dtype=float)
    weekend_values = window_daily.loc[window_daily["is_weekend"], value_col]
    weekday_values = window_daily.loc[~window_daily["is_weekend"], value_col]
    result = {
        "mean": float(np.nanmean(values)) if len(values) else np.nan,
        "sd": float(np.nanstd(values, ddof=0)) if len(values) else np.nan,
        "cv": float(np.nanstd(values, ddof=0) / np.nanmean(values)) if len(values) and np.nanmean(values) > 0 else 0.0,
        "slope": slope_from_values(values),
        "entropy": entropy_from_values(values),
        "day_night_ratio": np.nan,
        "weekend_shift": float(weekend_values.mean() - weekday_values.mean()) if len(weekend_values) and len(weekday_values) else np.nan,
        "regularity": regularity_from_values(values),
        "interevent_cv": np.nan,
        "ratio": np.nan,
        "missing_ratio": float(np.nanmean(window_daily[missing_col])) if len(window_daily) else np.nan,
    }
    full_mean = float(np.nanmean(full_daily[value_col])) if len(full_daily) else np.nan
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


def build_window_row(subject_id: str, anchor_id: str, task_name: str, anchor_date: pd.Timestamp, daily_frame: pd.DataFrame, pre_phq: float | None) -> dict:
    row = build_empty_window_row(subject_id, task_name, anchor_id)
    row["modality_mask_activity"] = 0
    row["modality_mask_sleep"] = 1
    row["modality_mask_communication"] = 1
    row["modality_mask_phone_use"] = 1
    row["modality_mask_mobility"] = 1
    row["modality_mask_static"] = 1
    row["modality_mask_symptom_context"] = 1 if pre_phq is not None else 0

    concept_mask_map = {
        "c1": 0,
        "c2": 0,
        "c3": 1,
        "c4": 1,
        "c5": 1,
        "c6": 1,
        "c7": 1,
        "c8": 1 if pre_phq is not None else 0,
    }
    for concept_id in CONCEPT_DEFINITIONS:
        row[f"concept_mask_{concept_id}"] = concept_mask_map[concept_id]

    row["feat_static_age"] = np.nan
    row["feat_static_sex"] = np.nan
    row["feat_static_baseline_severity"] = pre_phq
    row["feat_static_missing_rate"] = 1.0
    row["feat_symptom_context_mean_short"] = pre_phq
    row["feat_symptom_context_mean_medium"] = pre_phq
    row["feat_symptom_context_mean_long"] = pre_phq
    row["feat_symptom_context_missing_ratio_short"] = 0.0 if pre_phq is not None else 1.0
    row["feat_symptom_context_missing_ratio_medium"] = 0.0 if pre_phq is not None else 1.0
    row["feat_symptom_context_missing_ratio_long"] = 0.0 if pre_phq is not None else 1.0

    eligible_daily = daily_frame.loc[daily_frame["date"] <= anchor_date].copy()
    for window_name, target_days in {"short": 3, "medium": 7, "long": 30}.items():
        window_daily = eligible_daily.tail(target_days)
        for family, value_col, missing_col in [
            ("sleep", "raw_sleep_value", "sleep_missing"),
            ("communication", "raw_comm_value", "comm_missing"),
            ("phone_use", "raw_phone_use_value", "phone_missing"),
            ("mobility", "raw_mobility_value", "mobility_missing"),
        ]:
            bundle = generic_window_bundle(window_daily, eligible_daily, value_col, missing_col)
            row[f"feat_{family}_mean_{window_name}"] = bundle["mean"]
            row[f"feat_{family}_sd_{window_name}"] = bundle["sd"]
            row[f"feat_{family}_cv_{window_name}"] = bundle["cv"]
            row[f"feat_{family}_slope_{window_name}"] = bundle["slope"]
            row[f"feat_{family}_entropy_{window_name}"] = bundle["entropy"]
            row[f"feat_{family}_day_night_ratio_{window_name}"] = bundle["day_night_ratio"]
            row[f"feat_{family}_weekend_shift_{window_name}"] = bundle["weekend_shift"]
            row[f"feat_{family}_regularity_{window_name}"] = bundle["regularity"]
            row[f"feat_{family}_interevent_cv_{window_name}"] = bundle["interevent_cv"]
            row[f"feat_{family}_ratio_{window_name}"] = bundle["ratio"]
            row[f"feat_{family}_missing_ratio_{window_name}"] = bundle["missing_ratio"]
    return row


def main() -> int:
    args = parse_args()
    ensure_dir(PROJECT_ROOT / "data_interim" / "subject_tables" / DATASET_ID)
    ensure_dir(PROJECT_ROOT / "data_interim" / "daily_tables" / DATASET_ID)
    ensure_dir(PROJECT_ROOT / "data_interim" / "window_tables" / DATASET_ID)
    ensure_dir(PROJECT_ROOT / "data_interim" / "concept_tables" / DATASET_ID)

    phq = read_rds("survey/PHQ-9.Rds")
    sms = read_rds("other/sms.Rds")
    call_log = read_rds("other/call_log.Rds")
    conversation = read_rds("sensing/conversation.Rds")
    dark = read_rds("sensing/dark.Rds")
    phonelock = read_rds("sensing/phonelock.Rds")
    gps = read_rds("sensing/gps.Rds")

    phq["uid"] = phq["uid"].astype(str)
    phq["phq9_total"] = phq.apply(phq_total_from_row, axis=1)
    pre_scores = phq.loc[phq["type"] == "pre", ["uid", "phq9_total"]].rename(columns={"phq9_total": "phq9_pre"})
    post_scores = phq.loc[phq["type"] == "post", ["uid", "phq9_total"]].rename(columns={"phq9_total": "phq9_post"})
    survey_scores = pre_scores.merge(post_scores, on="uid", how="outer")

    call_log = call_log.loc[call_log["CALLS_date"].notna()].copy()
    call_log["uid"] = call_log["uid"].astype(str)
    call_log["timestamp_dt"] = pd.to_datetime(call_log["CALLS_date"], unit="ms", errors="coerce")
    call_log["date"] = call_log["timestamp_dt"].dt.normalize()

    sms["uid"] = sms["uid"].astype(str)
    sms["timestamp_dt"] = epoch_seconds_to_datetime(sms["timestamp"])
    sms["date"] = sms["timestamp_dt"].dt.normalize()

    conversation["uid"] = conversation["uid"].astype(str)
    conversation["start_dt"] = epoch_seconds_to_datetime(conversation["start_timestamp"])
    conversation["end_dt"] = epoch_seconds_to_datetime(conversation["end_timestamp"])
    conversation["duration_hours"] = (conversation["end_dt"] - conversation["start_dt"]).dt.total_seconds().clip(lower=0) / 3600.0
    conversation["date"] = conversation["start_dt"].dt.normalize()

    dark["uid"] = dark["uid"].astype(str)
    dark["start_dt"] = epoch_seconds_to_datetime(dark["start_timestamp"])
    dark["end_dt"] = epoch_seconds_to_datetime(dark["end_timestamp"])
    dark["duration_hours"] = (dark["end_dt"] - dark["start_dt"]).dt.total_seconds().clip(lower=0) / 3600.0
    dark["date"] = dark["start_dt"].dt.normalize()

    phonelock["uid"] = phonelock["uid"].astype(str)
    phonelock["start_dt"] = epoch_seconds_to_datetime(phonelock["start_timestamp"])
    phonelock["end_dt"] = epoch_seconds_to_datetime(phonelock["end_timestamp"])
    phonelock["duration_hours"] = (phonelock["end_dt"] - phonelock["start_dt"]).dt.total_seconds().clip(lower=0) / 3600.0
    phonelock["date"] = phonelock["start_dt"].dt.normalize()

    gps["uid"] = gps["uid"].astype(str)
    gps["timestamp_dt"] = epoch_seconds_to_datetime(gps["timestamp"])
    gps["date"] = gps["timestamp_dt"].dt.normalize()
    gps["loc_bin"] = gps["latitude"].round(3).astype(str) + "_" + gps["longitude"].round(3).astype(str)

    call_daily = call_log.groupby(["uid", "date"], as_index=False).agg(
        call_count=("CALLS_date", "count"),
        call_duration_sum=("CALLS_duration", "sum"),
    )
    sms_daily = sms.groupby(["uid", "date"], as_index=False).agg(sms_count=("timestamp", "count"))
    convo_daily = conversation.groupby(["uid", "date"], as_index=False).agg(
        conversation_count=("start_dt", "count"),
        conversation_hours=("duration_hours", "sum"),
    )
    dark_daily = dark.groupby(["uid", "date"], as_index=False).agg(dark_hours=("duration_hours", "sum"))
    lock_daily = phonelock.groupby(["uid", "date"], as_index=False).agg(phonelock_hours=("duration_hours", "sum"))
    gps_daily = gps.groupby(["uid", "date"], as_index=False).agg(
        gps_points=("timestamp_dt", "count"),
        gps_unique_locations=("loc_bin", "nunique"),
    )

    max_sensor_time = {}
    for name, frame, time_col in [
        ("call", call_log, "timestamp_dt"),
        ("sms", sms, "timestamp_dt"),
        ("conversation", conversation, "end_dt"),
        ("dark", dark, "end_dt"),
        ("phonelock", phonelock, "end_dt"),
        ("gps", gps, "timestamp_dt"),
    ]:
        sensor_max = frame.groupby("uid")[time_col].max()
        for uid, timestamp in sensor_max.items():
            if pd.isna(timestamp):
                continue
            max_sensor_time[uid] = max(timestamp, max_sensor_time.get(uid, timestamp))

    participants_rows = []
    daily_rows = []
    anchor_rows = []
    label_rows = []
    window_rows = []
    included_subjects = []
    excluded_subjects = []

    for _, survey_row in survey_scores.iterrows():
        uid = str(survey_row["uid"])
        if pd.isna(survey_row.get("phq9_post")):
            excluded_subjects.append({"subject_id": uid, "reason": "missing_post_phq"})
            continue
        if uid not in max_sensor_time:
            excluded_subjects.append({"subject_id": uid, "reason": "missing_sensor_history"})
            continue

        frames = []
        for daily_frame in [call_daily, sms_daily, convo_daily, dark_daily, lock_daily, gps_daily]:
            frames.append(daily_frame.loc[daily_frame["uid"] == uid].copy())
        available_frames = [frame[["date"]].copy() for frame in frames if len(frame)]
        if not available_frames:
            excluded_subjects.append({"subject_id": uid, "reason": "empty_sensor_history"})
            continue
        date_min = min(frame["date"].min() for frame in frames if len(frame))
        date_max = max_sensor_time[uid].normalize()
        daily = pd.DataFrame({"date": pd.date_range(start=date_min, end=date_max, freq="D")})
        daily["uid"] = uid
        daily = daily.merge(call_daily.loc[call_daily["uid"] == uid], on=["uid", "date"], how="left")
        daily = daily.merge(sms_daily.loc[sms_daily["uid"] == uid], on=["uid", "date"], how="left")
        daily = daily.merge(convo_daily.loc[convo_daily["uid"] == uid], on=["uid", "date"], how="left")
        daily = daily.merge(dark_daily.loc[dark_daily["uid"] == uid], on=["uid", "date"], how="left")
        daily = daily.merge(lock_daily.loc[lock_daily["uid"] == uid], on=["uid", "date"], how="left")
        daily = daily.merge(gps_daily.loc[gps_daily["uid"] == uid], on=["uid", "date"], how="left")
        for col in ["call_count", "call_duration_sum", "sms_count", "conversation_count", "conversation_hours", "dark_hours", "phonelock_hours", "gps_points", "gps_unique_locations"]:
            if col not in daily.columns:
                daily[col] = np.nan
        modality_cols = {
            "sleep_missing": "dark_hours",
            "comm_missing": "call_count",
            "phone_missing": "phonelock_hours",
            "mobility_missing": "gps_unique_locations",
        }
        for missing_col, source_col in modality_cols.items():
            daily[missing_col] = daily[source_col].isna().astype(float)

        daily = daily.fillna(
            {
                "call_count": 0,
                "call_duration_sum": 0,
                "sms_count": 0,
                "conversation_count": 0,
                "conversation_hours": 0,
                "dark_hours": 0,
                "phonelock_hours": 0,
                "gps_points": 0,
                "gps_unique_locations": 0,
            }
        )
        daily["raw_comm_value"] = daily["call_count"] + daily["sms_count"] + daily["conversation_count"]
        daily["raw_sleep_value"] = daily["dark_hours"]
        daily["raw_phone_use_value"] = daily["phonelock_hours"]
        daily["raw_mobility_value"] = daily["gps_unique_locations"]
        daily["raw_activity_value"] = np.nan
        daily["raw_symptom_value"] = survey_row.get("phq9_pre")
        daily["daily_missing_rate"] = daily[["sleep_missing", "comm_missing", "phone_missing", "mobility_missing"]].mean(axis=1)
        daily["qa_flag"] = np.where(daily["daily_missing_rate"] > 0.5, "sparse_multimodal_day", "ok")
        daily["study_day"] = np.arange(1, len(daily) + 1)
        daily["is_weekend"] = daily["date"].dt.dayofweek >= 5

        if len(daily) < args.min_complete_days:
            excluded_subjects.append({"subject_id": uid, "reason": "insufficient_complete_days"})
            continue

        included_subjects.append(uid)
        participants_rows.append(
            {
                "dataset_id": DATASET_ID,
                "subject_id": uid,
                "subject_id_original": uid,
                "sex": np.nan,
                "age": np.nan,
                "baseline_group": np.nan,
                "baseline_severity": survey_row.get("phq9_pre"),
                "static_missing_rate": 1.0,
                "source_file": "survey/PHQ-9.Rds",
                "days_recorded": int(len(daily)),
                "has_pre_phq": int(pd.notna(survey_row.get("phq9_pre"))),
                "has_post_phq": int(pd.notna(survey_row.get("phq9_post"))),
            }
        )

        for _, row in daily.iterrows():
            daily_rows.append(
                {
                    "dataset_id": DATASET_ID,
                    "subject_id": uid,
                    "date": row["date"].date().isoformat(),
                    "study_day": int(row["study_day"]),
                    "is_weekend": bool(row["is_weekend"]),
                    "raw_activity_value": np.nan,
                    "raw_sleep_value": float(row["raw_sleep_value"]),
                    "raw_comm_value": float(row["raw_comm_value"]),
                    "raw_phone_use_value": float(row["raw_phone_use_value"]),
                    "raw_mobility_value": float(row["raw_mobility_value"]),
                    "raw_symptom_value": survey_row.get("phq9_pre"),
                    "daily_missing_rate": float(row["daily_missing_rate"]),
                    "qa_flag": row["qa_flag"],
                    "source_file": "studentlife_rds_first_pass",
                    "call_count": int(row["call_count"]),
                    "sms_count": int(row["sms_count"]),
                    "conversation_count": int(row["conversation_count"]),
                    "dark_hours": float(row["dark_hours"]),
                    "phonelock_hours": float(row["phonelock_hours"]),
                    "gps_unique_locations": int(row["gps_unique_locations"]),
                }
            )

        anchor_time = max_sensor_time[uid]
        anchor_date = anchor_time.normalize()
        phq_post = float(survey_row["phq9_post"])
        phq_cat_value, phq_cat_label = phq_category(phq_post)
        available_days = len(daily)
        for task_name, label_type, y_raw, class_label in [
            ("phq9_reg", "continuous", phq_post, np.nan),
            ("phq9_cat", "ordinal", phq_cat_value, phq_cat_label),
        ]:
            anchor_id = f"{DATASET_ID}__{uid}__{task_name}__post"
            anchor_rows.append(
                {
                    "dataset_id": DATASET_ID,
                    "subject_id": uid,
                    "anchor_id": anchor_id,
                    "anchor_time": anchor_time.isoformat(),
                    "task_name": task_name,
                    "label_type": label_type,
                    "label_time": anchor_time.isoformat(),
                    "window_short_days": min(3, available_days),
                    "window_medium_days": min(7, available_days),
                    "window_long_days": min(30, available_days),
                    "window_mask_short": int(available_days >= 3),
                    "window_mask_medium": int(available_days >= 7),
                    "window_mask_long": int(available_days >= 30),
                }
            )
            label_rows.append(
                {
                    "dataset_id": DATASET_ID,
                    "subject_id": uid,
                    "anchor_id": anchor_id,
                    "task_name": task_name,
                    "label_type": label_type,
                    "y_raw": y_raw,
                    "y_std": np.nan,
                    "class_label": class_label,
                    "label_available": 1,
                }
            )
            window_rows.append(build_window_row(uid, anchor_id, task_name, anchor_date, daily, survey_row.get("phq9_pre")))

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
        "survey_uids_total": int(phq["uid"].nunique()),
        "uids_with_post_phq": int(post_scores["uid"].nunique()),
        "included_subjects": int(len(included_subjects)),
        "excluded_subjects": excluded_subjects,
        "windows_export": windows_record,
        "concept_export": concepts_record,
        "assumptions": [
            "First-pass StudentLife parsing excludes the very large sensing/activity.Rds table to keep the bootstrap pipeline auditable and tractable.",
            "End-of-study anchor time is approximated by the maximum available sensor timestamp across call, sms, conversation, dark, phonelock, and gps tables.",
            "Pre PHQ-9 is used as symptom-context input and post PHQ-9 is used as the supervised endpoint.",
        ],
    }
    write_json(PROJECT_ROOT / "outputs" / "logs" / "studentlife_preprocess_audit.json", audit_payload)

    print(
        f"StudentLife preprocessing complete: participants={len(participants_df)} "
        f"daily_rows={len(daily_df)} anchors={len(anchors_df)} labels={len(labels_df)}"
    )
    print(f"Window export={windows_record['format']} | Concept export={concepts_record['format']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
