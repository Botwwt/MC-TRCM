from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.constants import DATASET_IDS, PROJECT_ROOT
from src.utils.io import write_json


SEED = 20260417


def get_strata(dataset_id: str, participants: pd.DataFrame, labels: pd.DataFrame) -> dict[str, str]:
    strata = {}
    if dataset_id == "deprest_cat":
        subset = labels.loc[labels["task_name"] == "phq9_cat", ["subject_id", "class_label"]].drop_duplicates()
        strata.update(dict(zip(subset["subject_id"], subset["class_label"])))
    elif dataset_id == "studentlife":
        subset = labels.loc[labels["task_name"] == "phq9_cat", ["subject_id", "class_label"]].drop_duplicates()
        strata.update(dict(zip(subset["subject_id"], subset["class_label"])))
    elif dataset_id == "depresjon":
        subset = labels.loc[labels["task_name"] == "dep_binary", ["subject_id", "y_raw"]].drop_duplicates()
        strata.update({row["subject_id"]: f"class_{int(row['y_raw'])}" for _, row in subset.iterrows()})
    elif dataset_id == "obf":
        subset = labels.loc[labels["task_name"] == "obf_5class", ["subject_id", "class_label"]].drop_duplicates()
        strata.update(dict(zip(subset["subject_id"], subset["class_label"])))
    else:
        for subject_id in participants["subject_id"].astype(str):
            strata[subject_id] = "all"
    for subject_id in participants["subject_id"].astype(str):
        strata.setdefault(subject_id, "all")
    return strata


def stratified_holdout(subject_ids: list[str], strata: dict[str, str], rng: np.random.Generator) -> tuple[list[str], list[str], list[str]]:
    buckets = defaultdict(list)
    for subject_id in subject_ids:
        buckets[strata[subject_id]].append(subject_id)

    train, valid, test = [], [], []
    for bucket_subjects in buckets.values():
        bucket_subjects = list(bucket_subjects)
        rng.shuffle(bucket_subjects)
        n = len(bucket_subjects)
        n_train = max(1, int(round(n * 0.6))) if n >= 3 else max(1, n - 2)
        n_valid = max(1, int(round(n * 0.2))) if n >= 5 else 1 if n >= 2 else 0
        n_test = n - n_train - n_valid
        if n_test <= 0 and n >= 3:
            n_test = 1
            n_train = max(1, n_train - 1)
        train.extend(bucket_subjects[:n_train])
        valid.extend(bucket_subjects[n_train:n_train + n_valid])
        test.extend(bucket_subjects[n_train + n_valid:])

    rng.shuffle(train)
    rng.shuffle(valid)
    rng.shuffle(test)
    return train, valid, test


def stratified_cv(subject_ids: list[str], strata: dict[str, str], folds: int, rng: np.random.Generator) -> list[list[str]]:
    buckets = defaultdict(list)
    for subject_id in subject_ids:
        buckets[strata[subject_id]].append(subject_id)

    fold_subjects = [[] for _ in range(folds)]
    for bucket_subjects in buckets.values():
        bucket_subjects = list(bucket_subjects)
        rng.shuffle(bucket_subjects)
        for index, subject_id in enumerate(bucket_subjects):
            fold_subjects[index % folds].append(subject_id)
    for fold in fold_subjects:
        rng.shuffle(fold)
    return fold_subjects


def main() -> int:
    rng = np.random.default_rng(SEED)
    audit_lines = [
        "# Split Integrity Audit",
        "",
        f"Generated: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "| Dataset | Participants | Train | Valid | Test | Overlap | Small-dataset CV |",
        "|---|---:|---:|---:|---:|---|---|",
    ]

    for dataset_id in DATASET_IDS:
        participants_path = PROJECT_ROOT / "data_interim" / "subject_tables" / dataset_id / "participants.csv"
        labels_path = PROJECT_ROOT / "data_interim" / "window_tables" / dataset_id / "labels.csv"
        participants = pd.read_csv(participants_path)
        labels = pd.read_csv(labels_path)
        subject_ids = participants["subject_id"].astype(str).tolist()
        strata = get_strata(dataset_id, participants, labels)

        train_subjects, valid_subjects, test_subjects = stratified_holdout(subject_ids, strata, rng)
        overlap = bool(set(train_subjects) & set(valid_subjects) or set(train_subjects) & set(test_subjects) or set(valid_subjects) & set(test_subjects))

        payload = {
            "dataset_id": dataset_id,
            "split_name": "participant_level_default",
            "seed": SEED,
            "train_subjects": train_subjects,
            "valid_subjects": valid_subjects,
            "test_subjects": test_subjects,
            "audit": {
                "participant_overlap_detected": overlap,
                "participant_count": len(subject_ids),
                "train_count": len(train_subjects),
                "valid_count": len(valid_subjects),
                "test_count": len(test_subjects),
            },
        }

        cv_note = "none"
        if dataset_id in {"studentlife", "depresjon"}:
            folds = stratified_cv(subject_ids, strata, folds=5, rng=rng)
            payload["cv5"] = [{"fold": index, "subjects": fold} for index, fold in enumerate(folds)]
            cv_note = "5-fold"

        write_json(PROJECT_ROOT / "data_interim" / "window_tables" / dataset_id / "splits.json", payload)
        audit_lines.append(
            f"| {dataset_id} | {len(subject_ids)} | {len(train_subjects)} | {len(valid_subjects)} | {len(test_subjects)} | {overlap} | {cv_note} |"
        )

    audit_lines.extend(
        [
            "",
            "All splits are participant-level. No anchor-level randomization was used.",
            "",
            "Small datasets `studentlife` and `depresjon` also receive a stratified 5-fold subject-level CV manifest.",
        ]
    )
    audit_path = PROJECT_ROOT / "reports" / "audits" / "split_integrity.md"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text("\n".join(audit_lines) + "\n", encoding="utf-8")
    print("Wrote participant-level split manifests and split_integrity.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
