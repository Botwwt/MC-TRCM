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


DATASET_ID = "deprest_cat"
RAW_BASE = PROJECT_ROOT / "data_raw" / DATASET_ID / "extracted" / "DepreST-CAT-main"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build canonical DepreST-CAT tables.")
    parser.add_argument("--min-events", type=int, default=5)
    return parser.parse_args()


def parse_survey_timestamp(value: str) -> pd.Timestamp:
    return pd.to_datetime(str(value).strip().strip('"'), errors="coerce")


def age_midpoint(age_band: str | float | None) -> float | None:
    if age_band is None or pd.isna(age_band):
        return None
    if isinstance(age_band, str):
        age_band = age_band.strip()
        if age_band in {"", "-"}:
            return None
        if "-" in age_band:
            left, right = age_band.split("-", 1)
            try:
                return (float(left) + float(right)) / 2.0
            except ValueError:
                return None
        if age_band.endswith("+"):
            try:
                return float(age_band[:-1])
            except ValueError:
                return None
    return None


def sex_code(value: str | float | None) -> float | None:
    if value is None or pd.isna(value):
        return None
    value = str(value).strip().lower()
    if value == "woman":
        return 1.0
    if value == "man":
        return 0.0
    return None


def binary_yes(value: str | float | None) -> float | None:
    if value is None or pd.isna(value):
        return None
    value = str(value).strip().lower()
    if value.startswith("yes"):
        return 1.0
    if value.startswith("no"):
        return 0.0
    return None


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


def gad_category(score: float) -> tuple[int, str]:
    if score <= 4:
        return 0, "minimal"
    if score <= 9:
        return 1, "mild"
    if score <= 14:
        return 2, "moderate"
    return 3, "severe"


def build_empty_window_row(subject_id: str, task_name: str, anchor_id: str) -> dict:
    row = {column: np.nan for column in build_window_columns()}
    row["dataset_id"] = DATASET_ID
    row["subject_id"] = subject_id
    row["anchor_id"] = anchor_id
    row["task_name"] = task_name
    return row


def compute_daily_series_bundle(
    window_values: pd.Series,
    full_values: pd.Series,
    *,
    weekend_flags: pd.Series | None = None,
) -> dict[str, float | None]:
    window_array = pd.to_numeric(window_values, errors="coerce").to_numpy(dtype=float)
    full_array = pd.to_numeric(full_values, errors="coerce").to_numpy(dtype=float)
    valid_mask = ~np.isnan(window_array)
    valid_values = window_array[valid_mask]

    result = {
        "mean": float(np.nanmean(valid_values)) if len(valid_values) else np.nan,
        "sd": float(np.nanstd(valid_values, ddof=0)) if len(valid_values) else np.nan,
        "cv": float(np.nanstd(valid_values, ddof=0) / np.nanmean(valid_values)) if len(valid_values) and np.nanmean(valid_values) > 0 else 0.0,
        "slope": slope_from_values(valid_values),
        "entropy": entropy_from_values(valid_values),
        "regularity": regularity_from_values(valid_values),
        "ratio": np.nan,
        "missing_ratio": float(1.0 - valid_mask.mean()) if len(valid_mask) else np.nan,
        "weekend_shift": np.nan,
    }
    if weekend_flags is not None and len(window_array):
        weekend_array = weekend_flags.to_numpy(dtype=bool)
        weekend_values = window_array[weekend_array & valid_mask]
        weekday_values = window_array[(~weekend_array) & valid_mask]
        if len(weekend_values) and len(weekday_values):
            result["weekend_shift"] = float(np.nanmean(weekend_values) - np.nanmean(weekday_values))

    full_valid = full_array[~np.isnan(full_array)]
    if len(valid_values) and len(full_valid):
        full_mean = float(np.nanmean(full_valid))
        if np.isfinite(full_mean) and full_mean != 0.0:
            result["ratio"] = float(np.nanmean(valid_values) / full_mean)
    return result


def compute_comm_window_bundle(window_daily: pd.DataFrame, full_daily: pd.DataFrame, event_frame: pd.DataFrame) -> dict[str, float | None]:
    base_bundle = compute_daily_series_bundle(
        window_daily["all_logs"],
        full_daily["all_logs"],
        weekend_flags=window_daily["is_weekend"],
    )
    call_bundle = compute_daily_series_bundle(
        window_daily["all_calls"],
        full_daily["all_calls"],
        weekend_flags=window_daily["is_weekend"],
    )
    text_bundle = compute_daily_series_bundle(
        window_daily["all_text"],
        full_daily["all_text"],
        weekend_flags=window_daily["is_weekend"],
    )
    duration_bundle = compute_daily_series_bundle(
        window_daily["total_call_duration"],
        full_daily["total_call_duration"],
        weekend_flags=window_daily["is_weekend"],
    )
    contact_bundle = compute_daily_series_bundle(
        window_daily["unique_contacts"],
        full_daily["unique_contacts"],
        weekend_flags=window_daily["is_weekend"],
    )

    values = pd.to_numeric(window_daily["all_logs"], errors="coerce").to_numpy(dtype=float)
    night_ratios = window_daily["comm_day_night_ratio"].dropna().to_numpy(dtype=float)
    incoming_total = float(window_daily["incoming_logs"].sum())
    outgoing_total = float(window_daily["outgoing_logs"].sum())
    total_logs = float(window_daily["all_logs"].sum())
    total_calls = float(window_daily["all_calls"].sum())
    total_text = float(window_daily["all_text"].sum())

    event_times = event_frame["timestamp"].sort_values()
    if len(event_times) >= 2:
        gaps = event_times.diff().dropna().dt.total_seconds().to_numpy(dtype=float)
        interevent_cv = float(np.nanstd(gaps, ddof=0) / np.nanmean(gaps)) if np.nanmean(gaps) > 0 else np.nan
    else:
        interevent_cv = np.nan

    result = {
        "mean": base_bundle["mean"],
        "sd": base_bundle["sd"],
        "cv": base_bundle["cv"],
        "slope": base_bundle["slope"],
        "entropy": base_bundle["entropy"],
        "day_night_ratio": float(np.nanmean(night_ratios)) if len(night_ratios) else np.nan,
        "weekend_shift": base_bundle["weekend_shift"],
        "regularity": base_bundle["regularity"],
        "interevent_cv": interevent_cv,
        "ratio": base_bundle["ratio"],
        "missing_ratio": base_bundle["missing_ratio"],
        "call_mean": call_bundle["mean"],
        "call_sd": call_bundle["sd"],
        "call_cv": call_bundle["cv"],
        "call_slope": call_bundle["slope"],
        "call_entropy": call_bundle["entropy"],
        "call_ratio": call_bundle["ratio"],
        "text_mean": text_bundle["mean"],
        "text_sd": text_bundle["sd"],
        "text_cv": text_bundle["cv"],
        "text_slope": text_bundle["slope"],
        "text_entropy": text_bundle["entropy"],
        "text_ratio": text_bundle["ratio"],
        "duration_mean": duration_bundle["mean"],
        "duration_ratio": duration_bundle["ratio"],
        "contacts_mean": contact_bundle["mean"],
        "contacts_ratio": contact_bundle["ratio"],
        "coverage": float((window_daily["all_logs"] > 0).mean()) if len(window_daily) else np.nan,
        "call_coverage": float((window_daily["all_calls"] > 0).mean()) if len(window_daily) else np.nan,
        "text_coverage": float((window_daily["all_text"] > 0).mean()) if len(window_daily) else np.nan,
        "call_share": float(total_calls / total_logs) if total_logs > 0 else np.nan,
        "text_share": float(total_text / total_logs) if total_logs > 0 else np.nan,
        "outgoing_incoming_ratio": float(outgoing_total / incoming_total) if incoming_total > 0 else np.nan,
    }
    return result


def build_window_row(
    subject_id: str,
    anchor_id: str,
    task_name: str,
    anchor_time: pd.Timestamp,
    daily_frame: pd.DataFrame,
    event_frame: pd.DataFrame,
    survey_row: pd.Series,
) -> dict:
    row = build_empty_window_row(subject_id, task_name, anchor_id)

    row["modality_mask_activity"] = 0
    row["modality_mask_sleep"] = 0
    row["modality_mask_communication"] = 1
    row["modality_mask_phone_use"] = 0
    row["modality_mask_mobility"] = 0
    row["modality_mask_static"] = 1
    row["modality_mask_symptom_context"] = 1

    concept_mask_map = {
        "c1": 0,
        "c2": 0,
        "c3": 0,
        "c4": 0,
        "c5": 0,
        "c6": 1,
        "c7": 1,
        "c8": 1,
    }
    for concept_id in CONCEPT_DEFINITIONS:
        row[f"concept_mask_{concept_id}"] = concept_mask_map[concept_id]

    row["feat_static_age"] = age_midpoint(survey_row.get("Age"))
    row["feat_static_sex"] = sex_code(survey_row.get("Gender"))
    row["feat_static_baseline_severity"] = np.nan
    row["feat_static_missing_rate"] = float(survey_row.isna().mean())

    prior_treatment = binary_yes(survey_row.get("PriorDepressionTreatment"))
    row["feat_symptom_context_mean_short"] = prior_treatment
    row["feat_symptom_context_mean_medium"] = prior_treatment
    row["feat_symptom_context_mean_long"] = prior_treatment
    row["feat_symptom_context_missing_ratio_short"] = 0.0 if prior_treatment is not None else 1.0
    row["feat_symptom_context_missing_ratio_medium"] = 0.0 if prior_treatment is not None else 1.0
    row["feat_symptom_context_missing_ratio_long"] = 0.0 if prior_treatment is not None else 1.0

    eligible_daily = daily_frame.loc[daily_frame["date"] <= anchor_time.normalize()].copy()
    eligible_events = event_frame.loc[event_frame["timestamp"] <= anchor_time].copy()
    window_targets = {"short": 3, "medium": 7, "long": 30}
    for window_name, target_days in window_targets.items():
        window_daily = eligible_daily.tail(target_days)
        window_start = anchor_time.normalize() - pd.Timedelta(days=target_days - 1)
        window_events = eligible_events.loc[eligible_events["timestamp"] >= window_start]
        bundle = compute_comm_window_bundle(window_daily, eligible_daily, window_events)
        row[f"feat_communication_mean_{window_name}"] = bundle["mean"]
        row[f"feat_communication_sd_{window_name}"] = bundle["sd"]
        row[f"feat_communication_cv_{window_name}"] = bundle["cv"]
        row[f"feat_communication_slope_{window_name}"] = bundle["slope"]
        row[f"feat_communication_entropy_{window_name}"] = bundle["entropy"]
        row[f"feat_communication_day_night_ratio_{window_name}"] = bundle["day_night_ratio"]
        row[f"feat_communication_weekend_shift_{window_name}"] = bundle["weekend_shift"]
        row[f"feat_communication_regularity_{window_name}"] = bundle["regularity"]
        row[f"feat_communication_interevent_cv_{window_name}"] = bundle["interevent_cv"]
        row[f"feat_communication_ratio_{window_name}"] = bundle["ratio"]
        row[f"feat_communication_missing_ratio_{window_name}"] = bundle["missing_ratio"]
        row[f"feat_communication_call_mean_{window_name}"] = bundle["call_mean"]
        row[f"feat_communication_call_sd_{window_name}"] = bundle["call_sd"]
        row[f"feat_communication_call_cv_{window_name}"] = bundle["call_cv"]
        row[f"feat_communication_call_slope_{window_name}"] = bundle["call_slope"]
        row[f"feat_communication_call_entropy_{window_name}"] = bundle["call_entropy"]
        row[f"feat_communication_call_ratio_{window_name}"] = bundle["call_ratio"]
        row[f"feat_communication_text_mean_{window_name}"] = bundle["text_mean"]
        row[f"feat_communication_text_sd_{window_name}"] = bundle["text_sd"]
        row[f"feat_communication_text_cv_{window_name}"] = bundle["text_cv"]
        row[f"feat_communication_text_slope_{window_name}"] = bundle["text_slope"]
        row[f"feat_communication_text_entropy_{window_name}"] = bundle["text_entropy"]
        row[f"feat_communication_text_ratio_{window_name}"] = bundle["text_ratio"]
        row[f"feat_communication_duration_mean_{window_name}"] = bundle["duration_mean"]
        row[f"feat_communication_duration_ratio_{window_name}"] = bundle["duration_ratio"]
        row[f"feat_communication_contacts_mean_{window_name}"] = bundle["contacts_mean"]
        row[f"feat_communication_contacts_ratio_{window_name}"] = bundle["contacts_ratio"]
        row[f"feat_communication_coverage_{window_name}"] = bundle["coverage"]
        row[f"feat_communication_call_coverage_{window_name}"] = bundle["call_coverage"]
        row[f"feat_communication_text_coverage_{window_name}"] = bundle["text_coverage"]
        row[f"feat_communication_call_share_{window_name}"] = bundle["call_share"]
        row[f"feat_communication_text_share_{window_name}"] = bundle["text_share"]
        row[f"feat_communication_outgoing_incoming_ratio_{window_name}"] = bundle["outgoing_incoming_ratio"]
    return row


def main() -> int:
    args = parse_args()
    ensure_dir(PROJECT_ROOT / "data_interim" / "subject_tables" / DATASET_ID)
    ensure_dir(PROJECT_ROOT / "data_interim" / "daily_tables" / DATASET_ID)
    ensure_dir(PROJECT_ROOT / "data_interim" / "window_tables" / DATASET_ID)
    ensure_dir(PROJECT_ROOT / "data_interim" / "concept_tables" / DATASET_ID)

    surveys = pd.read_csv(RAW_BASE / "surveysDepreST-CAT.csv")
    calls = pd.read_csv(RAW_BASE / "callLogsDepreST-CAT2021.csv")
    texts = pd.read_csv(RAW_BASE / "textLogsDepreST-CAT2021.csv")

    surveys["anchor_time"] = surveys["Timestamp"].map(parse_survey_timestamp)
    calls["timestamp"] = pd.to_datetime(calls["UnixTimestamp"], unit="ms", errors="coerce", utc=True).dt.tz_localize(None)
    texts["timestamp"] = pd.to_datetime(texts["UnixTimestamp"], unit="ms", errors="coerce", utc=True).dt.tz_localize(None)

    call_events = pd.DataFrame(
        {
            "subject_id": calls["id"],
            "timestamp": calls["timestamp"],
            "contact": calls["Contact"],
            "event_type": "call",
            "is_outgoing": calls["Direction"] == 2,
            "is_incoming": calls["Direction"] != 2,
            "duration": pd.to_numeric(calls["Duration"], errors="coerce"),
        }
    )
    text_events = pd.DataFrame(
        {
            "subject_id": texts["id"],
            "timestamp": texts["timestamp"],
            "contact": texts["Contact"],
            "event_type": "text",
            "is_outgoing": texts["Direction"] == 2,
            "is_incoming": texts["Direction"] == 1,
            "duration": np.nan,
        }
    )
    events = pd.concat([call_events, text_events], ignore_index=True)
    events = events.dropna(subset=["timestamp"]).copy()
    events["date"] = events["timestamp"].dt.normalize()
    events["hour"] = events["timestamp"].dt.hour
    events["is_night"] = events["hour"].isin([0, 1, 2, 3, 4, 5, 22, 23])

    participants_rows = []
    daily_rows = []
    anchor_rows = []
    label_rows = []
    window_rows = []
    included_subjects = []
    excluded_subjects = []

    lookback_days = 180

    for _, survey_row in surveys.iterrows():
        subject_id = survey_row["id"]
        anchor_time = survey_row["anchor_time"]
        if pd.isna(anchor_time):
            excluded_subjects.append({"subject_id": subject_id, "reason": "invalid_survey_timestamp"})
            continue

        lookback_start = anchor_time - pd.Timedelta(days=lookback_days)
        subject_events = events.loc[
            (events["subject_id"] == subject_id)
            & (events["timestamp"] <= anchor_time)
            & (events["timestamp"] >= lookback_start)
        ].copy()
        if len(subject_events) < args.min_events:
            excluded_subjects.append({"subject_id": subject_id, "reason": "insufficient_event_count"})
            continue

        included_subjects.append(subject_id)
        start_date = subject_events["date"].min()
        end_date = anchor_time.normalize()
        calendar = pd.DataFrame({"date": pd.date_range(start=start_date, end=end_date, freq="D")})
        daily_agg = (
            subject_events.groupby("date", sort=True)
            .agg(
                all_logs=("event_type", "count"),
                all_calls=("event_type", lambda s: int((s == "call").sum())),
                all_text=("event_type", lambda s: int((s == "text").sum())),
                incoming_logs=("is_incoming", lambda s: int(s.sum())),
                outgoing_logs=("is_outgoing", lambda s: int(s.sum())),
                total_call_duration=("duration", "sum"),
                unique_contacts=("contact", "nunique"),
                nocturnal_logs=("is_night", lambda s: int(s.sum())),
            )
            .reset_index()
        )
        daily_agg["comm_day_night_ratio"] = np.where(
            daily_agg["nocturnal_logs"] > 0,
            (daily_agg["all_logs"] - daily_agg["nocturnal_logs"]) / daily_agg["nocturnal_logs"],
            np.nan,
        )
        daily = calendar.merge(daily_agg, on="date", how="left").fillna(
            {
                "all_logs": 0,
                "all_calls": 0,
                "all_text": 0,
                "incoming_logs": 0,
                "outgoing_logs": 0,
                "total_call_duration": 0,
                "unique_contacts": 0,
                "nocturnal_logs": 0,
            }
        )
        daily["is_weekend"] = daily["date"].dt.dayofweek >= 5
        daily["study_day"] = np.arange(1, len(daily) + 1)
        daily["daily_missing_rate"] = 0.0
        daily["qa_flag"] = "ok"

        participants_rows.append(
            {
                "dataset_id": DATASET_ID,
                "subject_id": subject_id,
                "subject_id_original": subject_id,
                "sex": survey_row.get("Gender"),
                "age": survey_row.get("Age"),
                "baseline_group": survey_row.get("appVersion"),
                "baseline_severity": np.nan,
                "static_missing_rate": float(survey_row.isna().mean()),
                "source_file": "surveysDepreST-CAT.csv",
                "event_count": int(len(subject_events)),
                "first_event_time": subject_events["timestamp"].min().isoformat(),
                "last_event_time": subject_events["timestamp"].max().isoformat(),
                "race_group": survey_row.get("Group"),
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
                    "raw_activity_value": np.nan,
                    "raw_sleep_value": np.nan,
                    "raw_comm_value": float(row["all_logs"]),
                    "raw_phone_use_value": np.nan,
                    "raw_mobility_value": np.nan,
                    "raw_symptom_value": np.nan,
                    "daily_missing_rate": float(row["daily_missing_rate"]),
                    "qa_flag": row["qa_flag"],
                    "source_file": "callLogsDepreST-CAT2021.csv;textLogsDepreST-CAT2021.csv",
                    "all_calls": int(row["all_calls"]),
                    "all_text": int(row["all_text"]),
                    "incoming_logs": int(row["incoming_logs"]),
                    "outgoing_logs": int(row["outgoing_logs"]),
                    "total_call_duration": float(row["total_call_duration"]),
                    "unique_contacts": int(row["unique_contacts"]),
                    "comm_day_night_ratio": float(row["comm_day_night_ratio"]) if pd.notna(row["comm_day_night_ratio"]) else np.nan,
                }
            )

        available_days = len(daily)
        phq_total = float(survey_row["PHQ - Total"])
        gad_total = float(survey_row["GAD - Total"])
        phq_cat_value, phq_cat_label = phq_category(phq_total)
        gad_cat_value, gad_cat_label = gad_category(gad_total)

        tasks = [
            ("phq9_reg", "continuous", phq_total, np.nan),
            ("gad7_reg", "continuous", gad_total, np.nan),
            ("phq9_cat", "ordinal", phq_cat_value, phq_cat_label),
            ("gad7_cat", "ordinal", gad_cat_value, gad_cat_label),
        ]
        for task_name, label_type, y_raw, class_label in tasks:
            anchor_id = f"{DATASET_ID}__{subject_id}__{task_name}__survey"
            anchor_rows.append(
                {
                    "dataset_id": DATASET_ID,
                    "subject_id": subject_id,
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
                    "subject_id": subject_id,
                    "anchor_id": anchor_id,
                    "task_name": task_name,
                    "label_type": label_type,
                    "y_raw": y_raw,
                    "y_std": np.nan,
                    "class_label": class_label,
                    "label_available": 1,
                }
            )
            window_rows.append(build_window_row(subject_id, anchor_id, task_name, anchor_time, daily, subject_events, survey_row))

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
        "raw_subjects_in_surveys": int(len(surveys)),
        "included_subjects": int(len(included_subjects)),
        "excluded_subjects": excluded_subjects,
        "windows_export": windows_record,
        "concept_export": concepts_record,
        "assumptions": [
            "Survey timestamp is used as the anchor time for all DepreST-CAT tasks.",
            "Communication windows use only call and text events observed at or before the survey timestamp.",
            "For the first-pass canonical parser, communication history is truncated to the 180 days before the survey anchor.",
            "Call direction code 2 is treated as outgoing; all other call direction codes are counted toward incoming/non-outgoing totals to match the released summary table.",
        ],
    }
    write_json(PROJECT_ROOT / "outputs" / "logs" / "deprest_cat_preprocess_audit.json", audit_payload)

    print(
        f"DepreST-CAT preprocessing complete: participants={len(participants_df)} "
        f"daily_rows={len(daily_df)} anchors={len(anchors_df)} labels={len(labels_df)}"
    )
    print(
        f"Window export={windows_record['format']} | Concept export={concepts_record['format']} | "
        f"excluded={len(excluded_subjects)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
