import json
from pathlib import Path

from src.preprocess.schema import (
    ANCHOR_COLUMNS,
    LABEL_COLUMNS,
    PARTICIPANT_COLUMNS,
    RAW_DAILY_COLUMNS,
    build_concept_columns,
    build_window_columns,
    empty_frame,
)
from src.utils.constants import DATASET_IDS, PROJECT_ROOT
from src.utils.io import ensure_dir, write_csv, write_json, write_parquet_with_fallback


DATASET_CONFIG_DIR = PROJECT_ROOT / "configs" / "dataset_configs"


def load_dataset_config(dataset_id: str) -> dict:
    config_path = DATASET_CONFIG_DIR / f"{dataset_id}.json"
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def list_dataset_ids() -> list[str]:
    return list(DATASET_IDS)


def scan_raw_inventory(dataset_id: str) -> list[dict]:
    dataset_config = load_dataset_config(dataset_id)
    raw_dir = PROJECT_ROOT / dataset_config["raw_dir"]
    ensure_dir(raw_dir)
    inventory = []
    for path in sorted(raw_dir.rglob("*")):
        if path.is_file():
            inventory.append(
                {
                    "relative_path": str(path.relative_to(PROJECT_ROOT)),
                    "size_bytes": path.stat().st_size,
                }
            )
    return inventory


def initialize_canonical_tables(dataset_id: str) -> dict:
    subject_dir = ensure_dir(PROJECT_ROOT / "data_interim" / "subject_tables" / dataset_id)
    daily_dir = ensure_dir(PROJECT_ROOT / "data_interim" / "daily_tables" / dataset_id)
    window_dir = ensure_dir(PROJECT_ROOT / "data_interim" / "window_tables" / dataset_id)
    concept_dir = ensure_dir(PROJECT_ROOT / "data_interim" / "concept_tables" / dataset_id)

    participants_path = subject_dir / "participants.csv"
    raw_daily_path = daily_dir / "raw_daily.csv"
    anchors_path = window_dir / "anchors.csv"
    labels_path = window_dir / "labels.csv"
    splits_path = window_dir / "splits.json"
    windows_path = window_dir / "windows_wide.parquet"
    concepts_path = concept_dir / "concepts.parquet"

    write_csv(participants_path, empty_frame(PARTICIPANT_COLUMNS))
    write_csv(raw_daily_path, empty_frame(RAW_DAILY_COLUMNS))
    write_csv(anchors_path, empty_frame(ANCHOR_COLUMNS))
    write_csv(labels_path, empty_frame(LABEL_COLUMNS))
    write_json(
        splits_path,
        {
            "dataset_id": dataset_id,
            "split_name": "pending_generation",
            "seed": 20260416,
            "train_subjects": [],
            "valid_subjects": [],
            "test_subjects": [],
            "audit": {
                "participant_overlap_detected": False,
                "notes": "Placeholder split manifest created by minimal_pipeline.py",
            },
        },
    )
    windows_record = write_parquet_with_fallback(windows_path, empty_frame(build_window_columns()))
    concepts_record = write_parquet_with_fallback(concepts_path, empty_frame(build_concept_columns()))

    return {
        "dataset_id": dataset_id,
        "participants_path": str(participants_path.relative_to(PROJECT_ROOT)),
        "raw_daily_path": str(raw_daily_path.relative_to(PROJECT_ROOT)),
        "anchors_path": str(anchors_path.relative_to(PROJECT_ROOT)),
        "labels_path": str(labels_path.relative_to(PROJECT_ROOT)),
        "splits_path": str(splits_path.relative_to(PROJECT_ROOT)),
        "windows_record": windows_record,
        "concepts_record": concepts_record,
    }
