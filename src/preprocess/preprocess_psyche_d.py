from __future__ import annotations

import argparse
import re
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


DATASET_ID = "psyche_d"
RAW_FILE = PROJECT_ROOT / "data_raw" / DATASET_ID / "anon_processed_df_parquet"


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description="Build canonical PSYCHE-D tables from the official processed parquet release.").parse_args()


def sanitize_feature_name(name: str) -> str:
    name = name.lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def derive_change_labels(row: pd.Series) -> tuple[int | None, str | None, int | None]:
    start = row.get("phq9_score_start")
    end = row.get("phq9_score_end")
    if pd.isna(start) or pd.isna(end):
        return None, None, None
    delta = float(end) - float(start)
    if delta < 0:
        return -1, "improved", 0
    if delta > 0:
        return 1, "worsened", 1
    return 0, "stable", 0


def subject_id_from_index(index_value: str) -> str:
    return str(index_value).split("_", 1)[0]


def anchor_order_from_index(index_value: str) -> int | None:
    parts = str(index_value).split("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return int(parts[1])
    return None


def main() -> int:
    _ = parse_args()
    ensure_dir(PROJECT_ROOT / "data_interim" / "subject_tables" / DATASET_ID)
    ensure_dir(PROJECT_ROOT / "data_interim" / "daily_tables" / DATASET_ID)
    ensure_dir(PROJECT_ROOT / "data_interim" / "window_tables" / DATASET_ID)
    ensure_dir(PROJECT_ROOT / "data_interim" / "concept_tables" / DATASET_ID)

    df = pd.read_parquet(RAW_FILE)
    df = df.copy()
    df["source_row_id"] = df.index.astype(str)
    df["subject_id"] = df["source_row_id"].map(subject_id_from_index)
    df["anchor_order"] = df["source_row_id"].map(anchor_order_from_index)

    static_columns = [
        "sex",
        "race_white",
        "race_black",
        "race_hispanic",
        "race_asian",
        "race_other",
        "birthyear",
        "educ",
        "height",
        "weight",
        "bmi",
        "pregnant",
    ]

    participants_df = (
        df.groupby("subject_id", sort=True)[static_columns]
        .first()
        .reset_index()
    )
    participants_df["dataset_id"] = DATASET_ID
    participants_df["subject_id_original"] = participants_df["subject_id"]
    participants_df["sex"] = participants_df["sex"]
    participants_df["age"] = 2026 - participants_df["birthyear"]
    participants_df["baseline_group"] = np.nan
    participants_df["baseline_severity"] = np.nan
    participants_df["static_missing_rate"] = participants_df[static_columns].isna().mean(axis=1)
    participants_df["source_file"] = str(RAW_FILE.relative_to(PROJECT_ROOT))

    labeled_df = df.loc[df["phq9_score_start"].notna() & df["phq9_score_end"].notna()].copy()
    labeled_df[["phq_change_multiclass", "phq_change_label", "phq_change_binary"]] = labeled_df.apply(
        lambda row: pd.Series(derive_change_labels(row)), axis=1
    )

    anchor_rows = []
    label_rows = []
    window_rows = []

    label_columns = {"phq9_score_start", "phq9_score_end", "phq9_cat_start", "phq9_cat_end", "subject_id", "anchor_order", "source_row_id"}
    native_feature_columns = [column for column in df.columns if column not in label_columns]

    for _, row in labeled_df.iterrows():
        for task_name, label_type, y_raw, class_label in [
            ("phq_change_multiclass", "multiclass", int(row["phq_change_multiclass"]), row["phq_change_label"]),
            ("phq_change_binary", "binary", int(row["phq_change_binary"]), "worsened" if int(row["phq_change_binary"]) == 1 else "stable_or_improved"),
        ]:
            anchor_id = f"{DATASET_ID}__{row['source_row_id']}__{task_name}"
            anchor_rows.append(
                {
                    "dataset_id": DATASET_ID,
                    "subject_id": row["subject_id"],
                    "anchor_id": anchor_id,
                    "anchor_time": np.nan,
                    "task_name": task_name,
                    "label_type": label_type,
                    "label_time": np.nan,
                    "window_short_days": 3,
                    "window_medium_days": 7,
                    "window_long_days": 30,
                    "window_mask_short": 1,
                    "window_mask_medium": 1,
                    "window_mask_long": 1,
                    "anchor_order": row["anchor_order"],
                    "source_row_id": row["source_row_id"],
                }
            )
            label_rows.append(
                {
                    "dataset_id": DATASET_ID,
                    "subject_id": row["subject_id"],
                    "anchor_id": anchor_id,
                    "task_name": task_name,
                    "label_type": label_type,
                    "y_raw": y_raw,
                    "y_std": np.nan,
                    "class_label": class_label,
                    "label_available": 1,
                    "phq9_score_start": row["phq9_score_start"],
                    "phq9_score_end": row["phq9_score_end"],
                    "phq9_cat_start": row["phq9_cat_start"],
                    "phq9_cat_end": row["phq9_cat_end"],
                }
            )

            window_row = {column: np.nan for column in build_window_columns()}
            window_row["dataset_id"] = DATASET_ID
            window_row["subject_id"] = row["subject_id"]
            window_row["anchor_id"] = anchor_id
            window_row["task_name"] = task_name
            window_row["modality_mask_activity"] = 1
            window_row["modality_mask_sleep"] = 1
            window_row["modality_mask_communication"] = 0
            window_row["modality_mask_phone_use"] = 0
            window_row["modality_mask_mobility"] = 0
            window_row["modality_mask_static"] = 1
            window_row["modality_mask_symptom_context"] = 1
            for concept_id in CONCEPT_DEFINITIONS:
                window_row[f"concept_mask_{concept_id}"] = 1 if concept_id in {"c1", "c2", "c3", "c4", "c8"} else 0
            window_row["feat_static_age"] = 2026 - row["birthyear"] if pd.notna(row["birthyear"]) else np.nan
            window_row["feat_static_sex"] = row["sex"]
            window_row["feat_static_baseline_severity"] = row["phq9_score_start"]
            window_row["feat_static_missing_rate"] = float(pd.Series(row[static_columns]).isna().mean())
            window_row["feat_symptom_context_mean_short"] = row["phq9_score_start"]
            window_row["feat_symptom_context_mean_medium"] = row["phq9_score_start"]
            window_row["feat_symptom_context_mean_long"] = row["phq9_score_start"]
            window_row["feat_symptom_context_missing_ratio_short"] = 0.0
            window_row["feat_symptom_context_missing_ratio_medium"] = 0.0
            window_row["feat_symptom_context_missing_ratio_long"] = 0.0

            for feature_name in native_feature_columns:
                sanitized = sanitize_feature_name(feature_name)
                window_row[f"feat_native_{sanitized}"] = row[feature_name]
            window_rows.append(window_row)

    anchors_df = pd.DataFrame(anchor_rows)
    labels_df = pd.DataFrame(label_rows)
    windows_df = pd.DataFrame(window_rows)
    raw_daily_df = pd.DataFrame(columns=RAW_DAILY_COLUMNS)
    concepts_df = pd.DataFrame(columns=build_concept_columns())

    for required, frame in [
        (PARTICIPANT_COLUMNS, participants_df),
        (RAW_DAILY_COLUMNS, raw_daily_df),
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
    write_csv(daily_dir / "raw_daily.csv", raw_daily_df)
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
        "rows_total": int(len(df)),
        "rows_labeled": int(len(labeled_df)),
        "subjects_total": int(df["subject_id"].nunique()),
        "windows_export": windows_record,
        "concept_export": concepts_record,
        "assumptions": [
            "The official PSYCHE-D parquet is treated as a processed window-level release rather than reconstructed raw daily data.",
            "raw_daily.csv is intentionally left empty because day-level observations are not exposed in the downloaded release.",
            "PHQ change labels are derived from phq9_score_end - phq9_score_start with classes improved / stable / worsened.",
        ],
    }
    write_json(PROJECT_ROOT / "outputs" / "logs" / "psyche_d_preprocess_audit.json", audit_payload)

    print(
        f"PSYCHE-D preprocessing complete: participants={len(participants_df)} "
        f"labeled_rows={len(labeled_df)} anchors={len(anchors_df)} labels={len(labels_df)}"
    )
    print(f"Window export={windows_record['format']} | Concept export={concepts_record['format']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
