from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd

from src.utils.constants import PROJECT_ROOT

FEATURE_PREFIXES = ("feat_", "modality_mask_", "concept_mask_")
WINDOW_TABLE_ROOT = PROJECT_ROOT / "data_interim" / "window_tables"


@dataclass
class TaskBundle:
    dataset_id: str
    task_name: str
    label_type: str
    feature_columns: list[str]
    dropped_feature_columns: list[str]
    train_frame: pd.DataFrame
    valid_frame: pd.DataFrame
    test_frame: pd.DataFrame


def list_dataset_tasks(dataset_id: str) -> list[dict[str, str]]:
    labels_path = WINDOW_TABLE_ROOT / dataset_id / "labels.csv"
    labels = pd.read_csv(labels_path)
    summary = (
        labels.loc[labels["label_available"] == 1, ["task_name", "label_type"]]
        .drop_duplicates()
        .sort_values(["task_name", "label_type"])
    )
    return summary.to_dict(orient="records")


def _load_split_manifest(dataset_id: str) -> dict:
    path = WINDOW_TABLE_ROOT / dataset_id / "splits.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _assign_split(subject_id: object, split_manifest: dict) -> str | None:
    normalized = str(subject_id)
    for split_name, subject_key in (
        ("train", "train_subjects"),
        ("valid", "valid_subjects"),
        ("test", "test_subjects"),
    ):
        subject_set = {str(item) for item in split_manifest.get(subject_key, [])}
        if normalized in subject_set:
            return split_name
    return None


def load_task_bundle(dataset_id: str, task_name: str) -> TaskBundle:
    dataset_dir = WINDOW_TABLE_ROOT / dataset_id
    windows_path = dataset_dir / "windows_wide.parquet"
    labels_path = dataset_dir / "labels.csv"
    split_manifest = _load_split_manifest(dataset_id)

    windows = pd.read_parquet(windows_path)
    labels = pd.read_csv(labels_path)
    labels = labels.loc[labels["label_available"] == 1].copy()
    labels = labels.loc[labels["task_name"] == task_name].copy()

    merged = windows.merge(
        labels[
            [
                "anchor_id",
                "task_name",
                "label_type",
                "y_raw",
                "class_label",
                "label_available",
            ]
        ],
        on=["anchor_id", "task_name"],
        how="inner",
    )
    merged["split"] = merged["subject_id"].map(lambda value: _assign_split(value, split_manifest))

    if merged["split"].isna().any():
        missing_subjects = sorted(merged.loc[merged["split"].isna(), "subject_id"].astype(str).unique().tolist())
        raise ValueError(
            f"Split assignment failed for dataset={dataset_id}, task={task_name}, "
            f"subjects={missing_subjects[:5]}"
        )

    label_types = sorted(merged["label_type"].dropna().astype(str).unique().tolist())
    if len(label_types) != 1:
        raise ValueError(
            f"Expected exactly one label type for dataset={dataset_id}, task={task_name}, got {label_types}"
        )

    feature_columns = sorted(
        [
            column
            for column in merged.columns
            if column.startswith(FEATURE_PREFIXES)
        ]
    )
    train_mask = merged["split"] == "train"
    dropped_feature_columns = [
        column for column in feature_columns if not merged.loc[train_mask, column].notna().any()
    ]
    kept_feature_columns = [
        column for column in feature_columns if column not in dropped_feature_columns
    ]

    return TaskBundle(
        dataset_id=dataset_id,
        task_name=task_name,
        label_type=label_types[0],
        feature_columns=kept_feature_columns,
        dropped_feature_columns=dropped_feature_columns,
        train_frame=merged.loc[merged["split"] == "train"].reset_index(drop=True),
        valid_frame=merged.loc[merged["split"] == "valid"].reset_index(drop=True),
        test_frame=merged.loc[merged["split"] == "test"].reset_index(drop=True),
    )
