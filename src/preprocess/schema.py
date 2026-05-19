import pandas as pd

from src.utils.constants import (
    CANONICAL_MODALITIES,
    CONCEPT_DEFINITIONS,
    FEATURE_FAMILIES,
    FEATURE_STATISTICS,
    WINDOW_SPECS,
)

PARTICIPANT_COLUMNS = [
    "dataset_id",
    "subject_id",
    "subject_id_original",
    "sex",
    "age",
    "baseline_group",
    "baseline_severity",
    "static_missing_rate",
    "source_file",
]

RAW_DAILY_COLUMNS = [
    "dataset_id",
    "subject_id",
    "date",
    "study_day",
    "is_weekend",
    "raw_activity_value",
    "raw_sleep_value",
    "raw_comm_value",
    "raw_phone_use_value",
    "raw_mobility_value",
    "raw_symptom_value",
    "daily_missing_rate",
    "qa_flag",
    "source_file",
]

ANCHOR_COLUMNS = [
    "dataset_id",
    "subject_id",
    "anchor_id",
    "anchor_time",
    "task_name",
    "label_type",
    "label_time",
    "window_short_days",
    "window_medium_days",
    "window_long_days",
    "window_mask_short",
    "window_mask_medium",
    "window_mask_long",
]

LABEL_COLUMNS = [
    "dataset_id",
    "subject_id",
    "anchor_id",
    "task_name",
    "label_type",
    "y_raw",
    "y_std",
    "class_label",
    "label_available",
]


def build_modality_mask_columns() -> list[str]:
    return [f"modality_mask_{modality}" for modality in CANONICAL_MODALITIES]


def build_concept_mask_columns() -> list[str]:
    return [f"concept_mask_{concept_id}" for concept_id in CONCEPT_DEFINITIONS]


def build_feature_columns() -> list[str]:
    feature_columns: list[str] = []
    for family in FEATURE_FAMILIES:
        for statistic in FEATURE_STATISTICS:
            for window_name in WINDOW_SPECS:
                feature_columns.append(f"feat_{family}_{statistic}_{window_name}")
    feature_columns.extend(
        [
            "feat_static_age",
            "feat_static_sex",
            "feat_static_baseline_severity",
            "feat_static_missing_rate",
        ]
    )
    return feature_columns


def build_window_columns() -> list[str]:
    return [
        "dataset_id",
        "subject_id",
        "anchor_id",
        "task_name",
        *build_modality_mask_columns(),
        *build_concept_mask_columns(),
        *build_feature_columns(),
    ]


def build_concept_columns() -> list[str]:
    return [
        "dataset_id",
        "subject_id",
        "anchor_id",
        *[f"concept_{concept_id}" for concept_id in CONCEPT_DEFINITIONS],
        "recursion_steps",
        "concept_source_model",
    ]


def empty_frame(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series(dtype="object") for column in columns})
