from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.utils.constants import CANONICAL_MODALITIES, CONCEPT_DEFINITIONS, PROJECT_ROOT, WINDOW_SPECS

WINDOW_TABLE_ROOT = PROJECT_ROOT / "data_interim" / "window_tables"
WINDOW_ORDER = tuple(WINDOW_SPECS)
FEATURE_PREFIX_BY_MODALITY = {
    "activity": "feat_activity_",
    "sleep": "feat_sleep_",
    "communication": "feat_communication_",
    "phone_use": "feat_phone_use_",
    "mobility": "feat_mobility_",
    "static": "feat_static_",
    "symptom_context": "feat_symptom_context_",
}
SENSOR_MODALITIES = {"activity", "sleep", "communication", "phone_use", "mobility"}
FEATURE_SOURCE_CONDITIONS = {
    "FULL",
    "SENSOR_VALUES_ONLY",
    "SENSOR_PLUS_MISSINGNESS",
    "MISSINGNESS_ONLY",
    "SYMPTOM_STATIC_CLINICAL",
}


@dataclass
class TaskMetadata:
    task_index: int
    task_key: str
    dataset_id: str
    task_name: str
    label_type: str
    output_dim: int
    class_space: list[object] | None
    numeric_labels: bool


@dataclass
class PreparedMultiCorpusData:
    train: "MultiCorpusTorchDataset"
    valid: "MultiCorpusTorchDataset"
    test: "MultiCorpusTorchDataset"
    task_metadata: list[TaskMetadata]
    dataset_to_index: dict[str, int]
    modality_feature_columns: dict[str, list[str]]
    native_feature_columns: list[str]
    deprest_comm_feature_columns: list[str]
    feature_stats: dict[str, dict[str, list[float]]]
    temporal_slice_dims: dict[str, int]
    temporal_num_slices: int
    missing_signal_dim: int
    native_missing_signal_dim: int
    feature_source_condition: str


def _load_split_manifest(dataset_id: str) -> dict:
    path = WINDOW_TABLE_ROOT / dataset_id / "splits.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _assign_split(subject_id: object, split_manifest: dict) -> str | None:
    normalized = str(subject_id)
    for split_name, key in (
        ("train", "train_subjects"),
        ("valid", "valid_subjects"),
        ("test", "test_subjects"),
    ):
        if normalized in {str(item) for item in split_manifest.get(key, [])}:
            return split_name
    return None


def _load_dataset_frame(dataset_id: str) -> pd.DataFrame:
    dataset_dir = WINDOW_TABLE_ROOT / dataset_id
    windows = pd.read_parquet(dataset_dir / "windows_wide.parquet")
    labels = pd.read_csv(dataset_dir / "labels.csv")
    labels = labels.loc[labels["label_available"] == 1].copy()
    splits = _load_split_manifest(dataset_id)

    label_columns = ["anchor_id", "task_name", "label_type", "y_raw", "class_label", "label_available"]
    optional_label_columns = ["phq9_cat_start", "phq9_cat_end", "phq9_score_start", "phq9_score_end"]
    for column in optional_label_columns:
        if column in labels.columns:
            label_columns.append(column)

    merged = windows.merge(
        labels[label_columns],
        on=["anchor_id", "task_name"],
        how="inner",
    )
    merged["split"] = merged["subject_id"].map(lambda value: _assign_split(value, splits))
    merged["task_key"] = merged["dataset_id"].astype(str) + "::" + merged["task_name"].astype(str)
    if merged["split"].isna().any():
        raise ValueError(f"Missing split assignment in dataset={dataset_id}")
    return merged.reset_index(drop=True)


def _collect_modality_columns(
    frame: pd.DataFrame,
    *,
    include_augmented_communication_features: bool = True,
) -> dict[str, list[str]]:
    modality_feature_columns: dict[str, list[str]] = {}
    for modality in CANONICAL_MODALITIES:
        prefix = FEATURE_PREFIX_BY_MODALITY[modality]
        columns = [column for column in frame.columns if column.startswith(prefix)]
        if modality == "communication" and not include_augmented_communication_features:
            excluded_tokens = (
                "_call_",
                "_text_",
                "_duration_",
                "_contacts_",
                "_coverage_",
                "_share_",
                "_outgoing_incoming_",
            )
            columns = [column for column in columns if not any(token in column for token in excluded_tokens)]
        modality_feature_columns[modality] = sorted(columns)
    return modality_feature_columns


def _collect_native_columns(frame: pd.DataFrame, *, include_native_features: bool = True) -> list[str]:
    if not include_native_features:
        return []
    return sorted([column for column in frame.columns if column.startswith("feat_native_")])


def _collect_deprest_comm_columns(frame: pd.DataFrame) -> list[str]:
    communication_columns = sorted([column for column in frame.columns if column.startswith("feat_communication_")])
    extra_tokens = (
        "_call_",
        "_text_",
        "_duration_",
        "_contacts_",
        "_coverage_",
        "_share_",
        "_outgoing_incoming_",
    )
    return [column for column in communication_columns if any(token in column for token in extra_tokens)]


def _normalize_feature_source_condition(feature_source_condition: str | None) -> str:
    condition = str(feature_source_condition or "FULL").strip().upper()
    if condition == "SENSOR_PLUS_MISSINGNESS":
        condition = "SENSOR_PLUS_MISSINGNESS"
    if condition not in FEATURE_SOURCE_CONDITIONS:
        allowed = ", ".join(sorted(FEATURE_SOURCE_CONDITIONS))
        raise ValueError(f"Unknown feature_source_condition={feature_source_condition!r}; expected one of {allowed}")
    return condition


def _filter_columns_for_feature_source(
    modality_feature_columns: dict[str, list[str]],
    native_feature_columns: list[str],
    deprest_comm_feature_columns: list[str],
    *,
    feature_source_condition: str | None,
) -> tuple[dict[str, list[str]], list[str], list[str]]:
    condition = _normalize_feature_source_condition(feature_source_condition)
    if condition == "FULL":
        return modality_feature_columns, native_feature_columns, deprest_comm_feature_columns

    filtered: dict[str, list[str]] = {modality: [] for modality in CANONICAL_MODALITIES}
    if condition in {"SENSOR_VALUES_ONLY", "SENSOR_PLUS_MISSINGNESS"}:
        for modality in SENSOR_MODALITIES:
            filtered[modality] = list(modality_feature_columns.get(modality, []))
        return filtered, [], list(deprest_comm_feature_columns)

    if condition == "SYMPTOM_STATIC_CLINICAL":
        for modality in ("static", "symptom_context"):
            filtered[modality] = list(modality_feature_columns.get(modality, []))
        return filtered, [], []

    if condition == "MISSINGNESS_ONLY":
        return filtered, [], []

    raise AssertionError(f"Unhandled feature source condition: {condition}")


def _window_suffix(column_name: str) -> str | None:
    for window_name in WINDOW_ORDER:
        if column_name.endswith(f"_{window_name}"):
            return str(window_name)
    return None


def _infer_task_metadata(frame: pd.DataFrame) -> list[TaskMetadata]:
    task_metadata: list[TaskMetadata] = []
    for task_index, (task_key, task_frame) in enumerate(sorted(frame.groupby("task_key"), key=lambda item: item[0])):
        label_type = str(task_frame["label_type"].iloc[0])
        dataset_id = str(task_frame["dataset_id"].iloc[0])
        task_name = str(task_frame["task_name"].iloc[0])
        if label_type == "continuous":
            class_space = None
            numeric_labels = True
            output_dim = 1
        else:
            numeric_values = pd.to_numeric(task_frame["y_raw"], errors="coerce")
            if numeric_values.notna().all():
                class_space = np.sort(numeric_values.astype(int).unique()).tolist()
                numeric_labels = True
            else:
                class_space = np.sort(task_frame["y_raw"].astype(str).unique()).tolist()
                numeric_labels = False
            output_dim = len(class_space)
        task_metadata.append(
            TaskMetadata(
                task_index=task_index,
                task_key=task_key,
                dataset_id=dataset_id,
                task_name=task_name,
                label_type=label_type,
                output_dim=output_dim,
                class_space=class_space,
                numeric_labels=numeric_labels,
            )
        )
    return task_metadata


def _attach_deprest_pair_targets(
    frame: pd.DataFrame,
    task_metadata: list[TaskMetadata],
    task_to_index: dict[str, int],
) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["paired_task_mask"] = 0.0
    enriched["paired_task_index"] = -1
    enriched["paired_target_float"] = 0.0
    enriched["paired_target_index"] = -1

    task_meta_by_key = {meta.task_key: meta for meta in task_metadata}
    pair_specs = [
        ("deprest_cat::gad7_cat", "deprest_cat::gad7_reg"),
        ("deprest_cat::gad7_reg", "deprest_cat::gad7_cat"),
        ("deprest_cat::phq9_cat", "deprest_cat::phq9_reg"),
        ("deprest_cat::phq9_reg", "deprest_cat::phq9_cat"),
    ]
    for source_task_key, target_task_key in pair_specs:
        source_meta = task_meta_by_key.get(source_task_key)
        target_meta = task_meta_by_key.get(target_task_key)
        if source_meta is None or target_meta is None:
            continue
        target_rows = enriched.loc[
            enriched["task_key"] == target_task_key,
            ["subject_id", "y_raw"],
        ].copy()
        if target_rows.empty:
            continue
        target_rows = target_rows.rename(columns={"y_raw": "paired_y_raw"}).drop_duplicates(subset=["subject_id"])
        subject_to_raw = target_rows.set_index("subject_id")["paired_y_raw"]
        source_mask = enriched["task_key"] == source_task_key
        paired_raw = enriched.loc[source_mask, "subject_id"].map(subject_to_raw)
        paired_available = paired_raw.notna()
        if not paired_available.any():
            continue
        valid_index = paired_raw.index[paired_available]
        enriched.loc[valid_index, "paired_task_mask"] = 1.0
        enriched.loc[valid_index, "paired_task_index"] = int(task_to_index[target_task_key])
        if target_meta.label_type == "continuous":
            enriched.loc[valid_index, "paired_target_float"] = pd.to_numeric(
                paired_raw.loc[valid_index],
                errors="coerce",
            ).fillna(0.0)
        else:
            if target_meta.numeric_labels:
                raw_values = pd.to_numeric(paired_raw.loc[valid_index], errors="coerce").astype(int)
            else:
                raw_values = paired_raw.loc[valid_index].astype(str)
            mapping = {value: idx for idx, value in enumerate(target_meta.class_space or [])}
            enriched.loc[valid_index, "paired_target_index"] = raw_values.map(mapping).fillna(-1).astype(int)
    return enriched


def _compute_feature_stats(
    train_frame: pd.DataFrame,
    modality_feature_columns: dict[str, list[str]],
    native_feature_columns: list[str],
    deprest_comm_feature_columns: list[str],
) -> dict[str, dict[str, list[float]]]:
    feature_stats: dict[str, dict[str, list[float]]] = {}
    for modality, columns in modality_feature_columns.items():
        values = train_frame[columns].apply(pd.to_numeric, errors="coerce")
        means = values.mean(axis=0, skipna=True).fillna(0.0)
        stds = values.std(axis=0, skipna=True).replace(0.0, 1.0).fillna(1.0)
        feature_stats[modality] = {
            "mean": means.astype(float).tolist(),
            "std": stds.astype(float).tolist(),
        }
    native_values = train_frame[native_feature_columns].apply(pd.to_numeric, errors="coerce")
    native_means = native_values.mean(axis=0, skipna=True).fillna(0.0)
    native_stds = native_values.std(axis=0, skipna=True).replace(0.0, 1.0).fillna(1.0)
    feature_stats["native"] = {
        "mean": native_means.astype(float).tolist(),
        "std": native_stds.astype(float).tolist(),
    }
    deprest_comm_values = train_frame[deprest_comm_feature_columns].apply(pd.to_numeric, errors="coerce")
    deprest_comm_means = deprest_comm_values.mean(axis=0, skipna=True).fillna(0.0)
    deprest_comm_stds = deprest_comm_values.std(axis=0, skipna=True).replace(0.0, 1.0).fillna(1.0)
    feature_stats["deprest_comm"] = {
        "mean": deprest_comm_means.astype(float).tolist(),
        "std": deprest_comm_stds.astype(float).tolist(),
    }
    return feature_stats


def prepare_multicorpus_data(
    dataset_ids: list[str],
    *,
    include_native_features: bool = True,
    include_augmented_communication_features: bool = True,
    feature_source_condition: str | None = "FULL",
) -> PreparedMultiCorpusData:
    frame = pd.concat([_load_dataset_frame(dataset_id) for dataset_id in dataset_ids], ignore_index=True)
    modality_feature_columns = _collect_modality_columns(
        frame,
        include_augmented_communication_features=include_augmented_communication_features,
    )
    native_feature_columns = _collect_native_columns(frame, include_native_features=include_native_features)
    deprest_comm_feature_columns = _collect_deprest_comm_columns(frame)
    condition = _normalize_feature_source_condition(feature_source_condition)
    modality_feature_columns, native_feature_columns, deprest_comm_feature_columns = _filter_columns_for_feature_source(
        modality_feature_columns,
        native_feature_columns,
        deprest_comm_feature_columns,
        feature_source_condition=condition,
    )
    task_metadata = _infer_task_metadata(frame)
    task_to_index = {meta.task_key: meta.task_index for meta in task_metadata}
    frame = _attach_deprest_pair_targets(frame, task_metadata, task_to_index)
    dataset_to_index = {dataset_id: index for index, dataset_id in enumerate(sorted(frame["dataset_id"].unique().tolist()))}
    feature_stats = _compute_feature_stats(
        frame.loc[frame["split"] == "train"],
        modality_feature_columns,
        native_feature_columns,
        deprest_comm_feature_columns,
    )

    train_dataset = MultiCorpusTorchDataset(
        frame=frame.loc[frame["split"] == "train"].reset_index(drop=True),
        modality_feature_columns=modality_feature_columns,
        native_feature_columns=native_feature_columns,
        deprest_comm_feature_columns=deprest_comm_feature_columns,
        feature_stats=feature_stats,
        task_metadata=task_metadata,
        task_to_index=task_to_index,
        dataset_to_index=dataset_to_index,
    )
    valid_dataset = MultiCorpusTorchDataset(
        frame=frame.loc[frame["split"] == "valid"].reset_index(drop=True),
        modality_feature_columns=modality_feature_columns,
        native_feature_columns=native_feature_columns,
        deprest_comm_feature_columns=deprest_comm_feature_columns,
        feature_stats=feature_stats,
        task_metadata=task_metadata,
        task_to_index=task_to_index,
        dataset_to_index=dataset_to_index,
    )
    test_dataset = MultiCorpusTorchDataset(
        frame=frame.loc[frame["split"] == "test"].reset_index(drop=True),
        modality_feature_columns=modality_feature_columns,
        native_feature_columns=native_feature_columns,
        deprest_comm_feature_columns=deprest_comm_feature_columns,
        feature_stats=feature_stats,
        task_metadata=task_metadata,
        task_to_index=task_to_index,
        dataset_to_index=dataset_to_index,
    )

    return PreparedMultiCorpusData(
        train=train_dataset,
        valid=valid_dataset,
        test=test_dataset,
        task_metadata=task_metadata,
        dataset_to_index=dataset_to_index,
        modality_feature_columns=modality_feature_columns,
        native_feature_columns=native_feature_columns,
        deprest_comm_feature_columns=deprest_comm_feature_columns,
        feature_stats=feature_stats,
        temporal_slice_dims=dict(train_dataset.temporal_slice_dims),
        temporal_num_slices=int(train_dataset.temporal_num_slices),
        missing_signal_dim=int(train_dataset.missing_signal_dim),
        native_missing_signal_dim=int(train_dataset.native_missing_signal_dim),
        feature_source_condition=condition,
    )


class MultiCorpusTorchDataset:
    def __init__(
        self,
        frame: pd.DataFrame,
        modality_feature_columns: dict[str, list[str]],
        native_feature_columns: list[str],
        deprest_comm_feature_columns: list[str],
        feature_stats: dict[str, dict[str, list[float]]],
        task_metadata: list[TaskMetadata],
        task_to_index: dict[str, int],
        dataset_to_index: dict[str, int],
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.modality_feature_columns = modality_feature_columns
        self.native_feature_columns = native_feature_columns
        self.deprest_comm_feature_columns = deprest_comm_feature_columns
        self.feature_stats = feature_stats
        self.task_to_index = task_to_index
        self.dataset_to_index = dataset_to_index
        self.task_metadata = {meta.task_index: meta for meta in task_metadata}

        self.modality_arrays = self._prepare_modality_arrays()
        (
            self.modality_temporal_arrays,
            self.modality_temporal_masks,
            self.temporal_slice_dims,
            self.temporal_num_slices,
        ) = self._prepare_temporal_arrays()
        self.native_array = self._prepare_native_array()
        self.deprest_comm_array = self._prepare_deprest_comm_array()
        self.modality_mask = self.frame[
            [f"modality_mask_{modality}" for modality in CANONICAL_MODALITIES]
        ].to_numpy(dtype=np.float32).copy()
        if self.native_feature_columns:
            native_observed = self.frame[self.native_feature_columns].notna().any(axis=1).astype(np.float32)
            self.native_mask = native_observed.to_numpy(dtype=np.float32).copy()
        else:
            self.native_mask = np.zeros(len(self.frame), dtype=np.float32)
        if self.deprest_comm_feature_columns:
            deprest_observed = self.frame[self.deprest_comm_feature_columns].notna().any(axis=1).astype(np.float32)
            self.deprest_comm_mask = deprest_observed.to_numpy(dtype=np.float32).copy()
        else:
            self.deprest_comm_mask = np.zeros(len(self.frame), dtype=np.float32)
        self.modality_missing_signals, self.missing_signal_dim = self._prepare_modality_missing_signals()
        self.native_missing_signal, self.native_missing_signal_dim = self._prepare_native_missing_signal()
        self.concept_mask = self.frame[
            [f"concept_mask_{concept_key}" for concept_key in CONCEPT_DEFINITIONS]
        ].to_numpy(dtype=np.float32).copy()
        self.dataset_index = self.frame["dataset_id"].map(self.dataset_to_index).to_numpy(dtype=np.int64).copy()
        self.task_index = self.frame["task_key"].map(self.task_to_index).to_numpy(dtype=np.int64).copy()
        psyche_end_values = pd.to_numeric(self.frame.get("phq9_cat_end"), errors="coerce") if "phq9_cat_end" in self.frame.columns else pd.Series(np.nan, index=self.frame.index)
        self.psyche_end_mask = psyche_end_values.notna().to_numpy(dtype=np.float32).copy()
        self.psyche_end_target = psyche_end_values.fillna(-1).astype(np.int64).to_numpy(dtype=np.int64).copy()
        psyche_start_values = pd.to_numeric(self.frame.get("phq9_cat_start"), errors="coerce") if "phq9_cat_start" in self.frame.columns else pd.Series(np.nan, index=self.frame.index)
        self.psyche_start_mask = psyche_start_values.notna().to_numpy(dtype=np.float32).copy()
        self.psyche_start_target = psyche_start_values.fillna(-1).astype(np.int64).to_numpy(dtype=np.int64).copy()
        psyche_start_score_values = pd.to_numeric(self.frame.get("phq9_score_start"), errors="coerce") if "phq9_score_start" in self.frame.columns else pd.Series(np.nan, index=self.frame.index)
        self.psyche_start_score_mask = psyche_start_score_values.notna().to_numpy(dtype=np.float32).copy()
        self.psyche_start_score = psyche_start_score_values.fillna(0.0).to_numpy(dtype=np.float32).copy()
        psyche_end_score_values = pd.to_numeric(self.frame.get("phq9_score_end"), errors="coerce") if "phq9_score_end" in self.frame.columns else pd.Series(np.nan, index=self.frame.index)
        self.psyche_end_score_mask = psyche_end_score_values.notna().to_numpy(dtype=np.float32).copy()
        self.psyche_end_score = psyche_end_score_values.fillna(0.0).to_numpy(dtype=np.float32).copy()
        self.paired_task_mask = pd.to_numeric(self.frame.get("paired_task_mask", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32).copy()
        self.paired_task_index = pd.to_numeric(self.frame.get("paired_task_index", -1), errors="coerce").fillna(-1).astype(np.int64).to_numpy(dtype=np.int64).copy()
        self.paired_target_float = pd.to_numeric(self.frame.get("paired_target_float", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32).copy()
        self.paired_target_index = pd.to_numeric(self.frame.get("paired_target_index", -1), errors="coerce").fillna(-1).astype(np.int64).to_numpy(dtype=np.int64).copy()

        target_float = np.zeros(len(self.frame), dtype=np.float32)
        target_index = np.full(len(self.frame), fill_value=-1, dtype=np.int64)
        for task_index, meta in self.task_metadata.items():
            mask = self.task_index == task_index
            if meta.label_type == "continuous":
                target_float[mask] = pd.to_numeric(self.frame.loc[mask, "y_raw"], errors="coerce").astype(float)
            else:
                if meta.numeric_labels:
                    raw_values = pd.to_numeric(self.frame.loc[mask, "y_raw"], errors="coerce").astype(int)
                else:
                    raw_values = self.frame.loc[mask, "y_raw"].astype(str)
                mapping = {value: idx for idx, value in enumerate(meta.class_space or [])}
                target_index[mask] = raw_values.map(mapping).to_numpy(dtype=np.int64)
        self.target_float = target_float
        self.target_index = target_index

    def _prepare_modality_arrays(self) -> dict[str, np.ndarray]:
        arrays: dict[str, np.ndarray] = {}
        for modality, columns in self.modality_feature_columns.items():
            values = self.frame[columns].apply(pd.to_numeric, errors="coerce")
            mean = np.asarray(self.feature_stats[modality]["mean"], dtype=np.float32)
            std = np.asarray(self.feature_stats[modality]["std"], dtype=np.float32)
            normalized = (values.to_numpy(dtype=np.float32) - mean) / std
            normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
            arrays[modality] = normalized.copy()
        return arrays

    def _prepare_temporal_arrays(self) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, int], int]:
        arrays: dict[str, np.ndarray] = {}
        masks: dict[str, np.ndarray] = {}
        slice_dims: dict[str, int] = {}
        num_slices = len(WINDOW_ORDER)

        for modality, columns in self.modality_feature_columns.items():
            grouped_indices = {window_name: [] for window_name in WINDOW_ORDER}
            residual_indices: list[int] = []
            for column_index, column_name in enumerate(columns):
                window_name = _window_suffix(column_name)
                if window_name is None:
                    residual_indices.append(column_index)
                else:
                    grouped_indices[window_name].append(column_index)

            slice_dim = max(
                [len(indices) for indices in grouped_indices.values()] + ([len(residual_indices)] if residual_indices else [0])
            )
            slice_dims[modality] = int(slice_dim)
            if slice_dim <= 0:
                arrays[modality] = np.zeros((len(self.frame), num_slices, 0), dtype=np.float32)
                masks[modality] = np.zeros((len(self.frame), num_slices), dtype=np.float32)
                continue

            temporal_array = np.zeros((len(self.frame), num_slices, slice_dim), dtype=np.float32)
            temporal_mask = np.zeros((len(self.frame), num_slices), dtype=np.float32)
            modality_array = self.modality_arrays[modality]

            for slice_index, window_name in enumerate(WINDOW_ORDER):
                window_indices = grouped_indices[window_name]
                if not window_indices:
                    continue
                temporal_array[:, slice_index, : len(window_indices)] = modality_array[:, window_indices]
                window_columns = [columns[index] for index in window_indices]
                observed = self.frame[window_columns].notna().any(axis=1).to_numpy(dtype=np.float32)
                temporal_mask[:, slice_index] = observed.copy()

            if residual_indices:
                temporal_array[:, 0, : len(residual_indices)] = modality_array[:, residual_indices]
                residual_columns = [columns[index] for index in residual_indices]
                residual_observed = self.frame[residual_columns].notna().any(axis=1).to_numpy(dtype=np.float32)
                temporal_mask[:, 0] = np.maximum(temporal_mask[:, 0], residual_observed)

            arrays[modality] = temporal_array
            masks[modality] = temporal_mask
        return arrays, masks, slice_dims, num_slices

    def _prepare_native_array(self) -> np.ndarray:
        if not self.native_feature_columns:
            return np.zeros((len(self.frame), 0), dtype=np.float32)
        values = self.frame[self.native_feature_columns].apply(pd.to_numeric, errors="coerce")
        mean = np.asarray(self.feature_stats["native"]["mean"], dtype=np.float32)
        std = np.asarray(self.feature_stats["native"]["std"], dtype=np.float32)
        normalized = (values.to_numpy(dtype=np.float32) - mean) / std
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
        return normalized.copy()

    def _prepare_deprest_comm_array(self) -> np.ndarray:
        if not self.deprest_comm_feature_columns:
            return np.zeros((len(self.frame), 0), dtype=np.float32)
        values = self.frame[self.deprest_comm_feature_columns].apply(pd.to_numeric, errors="coerce")
        mean = np.asarray(self.feature_stats["deprest_comm"]["mean"], dtype=np.float32)
        std = np.asarray(self.feature_stats["deprest_comm"]["std"], dtype=np.float32)
        normalized = (values.to_numpy(dtype=np.float32) - mean) / std
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
        return normalized.copy()

    def _prepare_modality_missing_signals(self) -> tuple[dict[str, np.ndarray], int]:
        signal_dim = 6
        modality_signals: dict[str, np.ndarray] = {}
        for modality_index, modality in enumerate(CANONICAL_MODALITIES):
            signal = np.zeros((len(self.frame), signal_dim), dtype=np.float32)
            observed = self.modality_mask[:, modality_index]
            signal[:, 0] = observed
            prefix = FEATURE_PREFIX_BY_MODALITY[modality]
            window_ratios: list[np.ndarray] = []
            modality_columns = self.modality_feature_columns[modality]
            if modality_columns:
                overall_missing_rate = 1.0 - self.frame[modality_columns].notna().mean(axis=1).to_numpy(dtype=np.float32)
            else:
                overall_missing_rate = (1.0 - observed).astype(np.float32)
            overall_missing_rate_series = pd.Series(overall_missing_rate, index=self.frame.index)
            for offset, window_name in enumerate(WINDOW_ORDER, start=1):
                ratio_column = f"{prefix}missing_ratio_{window_name}"
                if ratio_column in self.frame.columns:
                    ratio = pd.to_numeric(self.frame[ratio_column], errors="coerce").fillna(overall_missing_rate_series)
                    ratio = ratio.clip(lower=0.0, upper=1.0).to_numpy(dtype=np.float32)
                elif modality == "static" and "feat_static_missing_rate" in self.frame.columns:
                    ratio = pd.to_numeric(self.frame["feat_static_missing_rate"], errors="coerce").fillna(overall_missing_rate_series)
                    ratio = ratio.clip(lower=0.0, upper=1.0).to_numpy(dtype=np.float32)
                else:
                    ratio = overall_missing_rate.astype(np.float32).copy()
                signal[:, offset] = ratio
                window_ratios.append(ratio)

            gap_days = np.maximum.reduce(
                [
                    window_ratios[window_index] * float(WINDOW_SPECS[window_name][1])
                    for window_index, window_name in enumerate(WINDOW_ORDER)
                ]
            ).astype(np.float32)
            signal[:, 4] = np.clip(gap_days / float(WINDOW_SPECS["long"][1]), 0.0, 1.0)
            signal[:, 5] = np.exp(-gap_days / 7.0).astype(np.float32)
            modality_signals[modality] = signal
        return modality_signals, signal_dim

    def _prepare_native_missing_signal(self) -> tuple[np.ndarray, int]:
        if not self.native_feature_columns:
            return np.zeros((len(self.frame), 0), dtype=np.float32), 0
        missing_rate = (
            1.0 - self.frame[self.native_feature_columns].notna().mean(axis=1).to_numpy(dtype=np.float32)
        ).astype(np.float32)
        gap_days = missing_rate * float(WINDOW_SPECS["long"][1])
        signal = np.stack(
            [
                self.native_mask,
                missing_rate,
                np.clip(gap_days / float(WINDOW_SPECS["long"][1]), 0.0, 1.0).astype(np.float32),
                np.exp(-gap_days / 7.0).astype(np.float32),
            ],
            axis=1,
        ).astype(np.float32)
        return signal, int(signal.shape[1])

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = {
            "dataset_id": str(self.frame.at[index, "dataset_id"]),
            "subject_id": str(self.frame.at[index, "subject_id"]),
            "anchor_id": str(self.frame.at[index, "anchor_id"]),
            "task_name": str(self.frame.at[index, "task_name"]),
            "task_key": str(self.frame.at[index, "task_key"]),
            "label_type": str(self.frame.at[index, "label_type"]),
            "y_raw": str(self.frame.at[index, "y_raw"]),
            "dataset_index": self.dataset_index[index],
            "task_index": self.task_index[index],
            "target_float": self.target_float[index],
            "target_index": self.target_index[index],
            "modality_mask": self.modality_mask[index],
            "native_mask": self.native_mask[index],
            "concept_mask": self.concept_mask[index],
            "native_features": self.native_array[index],
            "native_missing_signal": self.native_missing_signal[index],
            "deprest_comm_mask": self.deprest_comm_mask[index],
            "deprest_comm_features": self.deprest_comm_array[index],
            "psyche_end_mask": self.psyche_end_mask[index],
            "psyche_end_target": self.psyche_end_target[index],
            "psyche_start_mask": self.psyche_start_mask[index],
            "psyche_start_target": self.psyche_start_target[index],
            "psyche_start_score_mask": self.psyche_start_score_mask[index],
            "psyche_start_score": self.psyche_start_score[index],
            "psyche_end_score_mask": self.psyche_end_score_mask[index],
            "psyche_end_score": self.psyche_end_score[index],
            "paired_task_mask": self.paired_task_mask[index],
            "paired_task_index": self.paired_task_index[index],
            "paired_target_float": self.paired_target_float[index],
            "paired_target_index": self.paired_target_index[index],
        }
        for modality in CANONICAL_MODALITIES:
            sample[f"{modality}_features"] = self.modality_arrays[modality][index]
            sample[f"{modality}_temporal_slices"] = self.modality_temporal_arrays[modality][index]
            sample[f"{modality}_temporal_mask"] = self.modality_temporal_masks[modality][index]
            sample[f"{modality}_missing_signal"] = self.modality_missing_signals[modality][index]
        return sample

    def dataset_sample_weights(self) -> np.ndarray:
        counts = pd.Series(self.frame["dataset_id"]).value_counts()
        return self.frame["dataset_id"].map(lambda value: 1.0 / counts[value]).to_numpy(dtype=np.float32)
