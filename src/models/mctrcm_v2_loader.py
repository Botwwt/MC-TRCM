from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import torch

from src.models.mctrcm_data import PreparedMultiCorpusData, TaskMetadata, prepare_multicorpus_data
from src.models.mctrcm_v2 import MCTRCMV2
from src.models.train_mctrcm_v2 import (
    _apply_task_label_overrides,
    _build_task_head_specs,
    _derive_deprest_comm_groups,
)
from src.utils.constants import PROJECT_ROOT

CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"
LOG_DIR = PROJECT_ROOT / "outputs" / "logs"


def _load_run_config(run_name: str) -> dict[str, object]:
    path = LOG_DIR / f"{run_name}__config.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing saved run config for {run_name}: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_requested_task_index(
    requested_task_key: str,
    task_key_to_index: dict[str, int],
    task_name_to_indices: dict[str, list[int]],
) -> int:
    requested_task_key = str(requested_task_key).strip()
    if requested_task_key in task_key_to_index:
        return int(task_key_to_index[requested_task_key])
    candidate_indices = task_name_to_indices.get(requested_task_key, [])
    if len(candidate_indices) == 1:
        return int(candidate_indices[0])
    raise ValueError(f"No unique task match found for task spec {requested_task_key!r}")


def _dataset_task_indices(
    task_metadata: list[TaskMetadata],
    *,
    dataset_id: str,
    enabled: bool,
    task_names: list[str] | None,
    default_task_names: list[str] | None = None,
) -> list[int]:
    if not enabled:
        return []
    selected_names = {
        str(task_name).strip()
        for task_name in ((task_names or default_task_names) or [])
        if str(task_name).strip()
    }
    return sorted(
        int(meta.task_index)
        for meta in task_metadata
        if meta.dataset_id == dataset_id and meta.task_name in selected_names
    )


def build_mctrcm_v2_from_saved_config(
    prepared: PreparedMultiCorpusData,
    run_config: dict[str, object],
) -> tuple[MCTRCMV2, dict[str, object]]:
    model_config = dict(run_config["model_config"])
    task_metadata = prepared.task_metadata
    _apply_task_label_overrides(
        prepared,
        multiclass_task_keys=list(run_config.get("multiclass_task_keys") or []),
    )
    task_metadata = prepared.task_metadata
    task_key_to_index = {str(meta.task_key): int(meta.task_index) for meta in task_metadata}
    task_name_to_indices: dict[str, list[int]] = defaultdict(list)
    for meta in task_metadata:
        task_name_to_indices[str(meta.task_name)].append(int(meta.task_index))

    paired_regression_bridge_specs = {
        int(
            _resolve_requested_task_index(
                target_task_key,
                task_key_to_index,
                task_name_to_indices,
            )
        ): int(
            _resolve_requested_task_index(
                source_task_key,
                task_key_to_index,
                task_name_to_indices,
            )
        )
        for target_task_key, source_task_key in dict(run_config.get("paired_regression_bridge_specs") or {}).items()
    }
    deprest_comm_groups = _derive_deprest_comm_groups(prepared.deprest_comm_feature_columns)
    use_deprest_comm_features = (
        (not bool(run_config.get("disable_deprest_adapter", False)))
        or bool(run_config.get("enable_deprest_category_adapter", False))
        or bool(run_config.get("enable_deprest_coverage_routing", False))
        or bool(run_config.get("enable_deprest_concept_compatibility", False))
        or bool(run_config.get("enable_deprest_concept_contrast", False))
        or bool(run_config.get("enable_deprest_edge_specialist", False))
        or bool(run_config.get("enable_deprest_edge_ovr", False))
    )
    deprest_category_task_indices = sorted(
        int(meta.task_index)
        for meta in task_metadata
        if meta.dataset_id == "deprest_cat" and meta.task_name in {"gad7_cat", "phq9_cat"}
    )
    deprest_coverage_task_indices = _dataset_task_indices(
        task_metadata,
        dataset_id="deprest_cat",
        enabled=bool(run_config.get("enable_deprest_coverage_routing", False)),
        task_names=list(run_config.get("deprest_coverage_task_names") or []),
        default_task_names=["gad7_cat"],
    )
    deprest_concept_compat_task_indices = _dataset_task_indices(
        task_metadata,
        dataset_id="deprest_cat",
        enabled=bool(run_config.get("enable_deprest_concept_compatibility", False)),
        task_names=list(run_config.get("deprest_concept_compat_task_names") or []),
        default_task_names=["gad7_cat"],
    )
    deprest_concept_contrast_task_indices = _dataset_task_indices(
        task_metadata,
        dataset_id="deprest_cat",
        enabled=bool(run_config.get("enable_deprest_concept_contrast", False)),
        task_names=list(run_config.get("deprest_concept_contrast_task_names") or []),
        default_task_names=["gad7_cat"],
    )
    deprest_edge_task_indices = _dataset_task_indices(
        task_metadata,
        dataset_id="deprest_cat",
        enabled=bool(run_config.get("enable_deprest_edge_specialist", False)),
        task_names=list(run_config.get("deprest_edge_task_names") or []),
        default_task_names=["gad7_cat"],
    )
    deprest_edge_ovr_task_indices = _dataset_task_indices(
        task_metadata,
        dataset_id="deprest_cat",
        enabled=bool(run_config.get("enable_deprest_edge_ovr", False)),
        task_names=list(run_config.get("deprest_edge_ovr_task_names") or []),
        default_task_names=["gad7_cat"],
    )
    deprest_concept_feature_indices = [
        index
        for index, column in enumerate(prepared.deprest_comm_feature_columns)
        if any(token in column for token in ("_coverage_", "_share_", "_duration_", "_contacts_", "_outgoing_incoming_"))
    ]
    use_psyche_conditioning = (not bool(run_config.get("disable_psyche_hierarchical", False))) or bool(
        run_config.get("enable_psyche_two_stage", False)
    )
    psyche_change_task_indices = sorted(
        int(meta.task_index)
        for meta in task_metadata
        if meta.dataset_id == "psyche_d" and meta.task_name in {"phq_change_binary", "phq_change_multiclass"}
    )
    psyche_binary_correction_task_indices = sorted(
        int(meta.task_index)
        for meta in task_metadata
        if bool(run_config.get("enable_psyche_binary_correction", False))
        and meta.dataset_id == "psyche_d"
        and meta.task_name == "phq_change_binary"
    )
    psyche_native_task_indices = _dataset_task_indices(
        task_metadata,
        dataset_id="psyche_d",
        enabled=bool(run_config.get("enable_psyche_native_adapter", False)),
        task_names=list(run_config.get("psyche_native_task_names") or []),
        default_task_names=["phq_change_binary", "phq_change_multiclass"],
    )
    task_local_trm_task_indices = sorted(
        int(meta.task_index)
        for meta in task_metadata
        if meta.task_name in {
            str(task_name).strip()
            for task_name in (run_config.get("task_local_trm_task_names") or [])
            if str(task_name).strip()
        }
    )
    tree_head_task_indices = sorted(
        int(meta.task_index)
        for meta in task_metadata
        if bool(model_config.get("enable_tree_gated_head", False))
        and meta.task_name in {
            str(task_name).strip()
            for task_name in (run_config.get("tree_head_task_names") or [])
            if str(task_name).strip()
        }
    )
    auto_tree_head_label_types = {
        str(label_type).strip()
        for label_type in model_config.get("auto_tree_head_label_types", [])
        if str(label_type).strip()
    }
    if bool(model_config.get("enable_tree_gated_head", False)) and auto_tree_head_label_types:
        tree_head_task_indices = sorted(
            set(tree_head_task_indices)
            | {
                int(meta.task_index)
                for meta in task_metadata
                if meta.label_type in auto_tree_head_label_types
            }
        )
    deprest_construct_bridge_specs: dict[str, dict[str, object]] = {}
    if bool(run_config.get("enable_deprest_construct_bridge", False)):
        selected_construct_tasks = {
            str(task_name).strip()
            for task_name in (run_config.get("deprest_construct_task_names") or ["gad7_cat", "phq9_cat"])
            if str(task_name).strip()
        }
        deprest_meta_by_name = {
            str(meta.task_name): meta
            for meta in task_metadata
            if meta.dataset_id == "deprest_cat"
        }
        construct_definitions = [
            ("gad7", "gad7_cat", "gad7_reg", [5.0, 10.0, 15.0], 21.0),
            ("phq9", "phq9_cat", "phq9_reg", [5.0, 10.0, 15.0, 20.0], 27.0),
        ]
        for construct_key, categorical_task_name, regression_task_name, thresholds, score_max in construct_definitions:
            if categorical_task_name not in selected_construct_tasks:
                continue
            categorical_meta = deprest_meta_by_name.get(categorical_task_name)
            regression_meta = deprest_meta_by_name.get(regression_task_name)
            if categorical_meta is None:
                continue
            task_indices = [int(categorical_meta.task_index)]
            if regression_meta is not None:
                task_indices.append(int(regression_meta.task_index))
            deprest_construct_bridge_specs[construct_key] = {
                "task_indices": task_indices,
                "categorical_task_indices": [int(categorical_meta.task_index)],
                "thresholds": thresholds,
                "score_max": score_max,
            }
    deprest_severity_specs: dict[int, dict[str, object]] = {}
    if bool(run_config.get("enable_deprest_severity_head", False)):
        selected_severity_tasks = {
            str(task_name).strip()
            for task_name in (run_config.get("deprest_severity_task_names") or ["phq9_cat", "gad7_cat"])
            if str(task_name).strip()
        }
        for meta in task_metadata:
            if meta.dataset_id != "deprest_cat":
                continue
            if meta.task_name == "phq9_cat" and meta.task_name in selected_severity_tasks:
                deprest_severity_specs[int(meta.task_index)] = {
                    "thresholds": [5.0, 10.0, 15.0, 20.0],
                    "score_max": 27.0,
                }
            elif meta.task_name == "gad7_cat" and meta.task_name in selected_severity_tasks:
                deprest_severity_specs[int(meta.task_index)] = {
                    "thresholds": [5.0, 10.0, 15.0],
                    "score_max": 21.0,
                }
    deprest_pair_specs: dict[int, dict[str, object]] = {}
    if bool(run_config.get("enable_deprest_gad7_pair_aux", False)):
        task_meta_by_name = {
            str(meta.task_name): meta
            for meta in task_metadata
            if meta.dataset_id == "deprest_cat"
        }
        gad7_cat_meta = task_meta_by_name.get("gad7_cat")
        gad7_reg_meta = task_meta_by_name.get("gad7_reg")
        if gad7_cat_meta is not None and gad7_reg_meta is not None:
            deprest_pair_specs[int(gad7_cat_meta.task_index)] = {
                "target_task_index": int(gad7_reg_meta.task_index),
                "output_dim": 1,
            }
            deprest_pair_specs[int(gad7_reg_meta.task_index)] = {
                "target_task_index": int(gad7_cat_meta.task_index),
                "output_dim": (
                    max(len(gad7_cat_meta.class_space or []) - 1, 1)
                    if gad7_cat_meta.label_type == "ordinal"
                    else gad7_cat_meta.output_dim
                ),
            }
    deprest_task_bridge_specs: dict[int, dict[str, object]] = {}
    if bool(run_config.get("enable_deprest_gad7_reg_bridge", False)):
        task_meta_by_name = {
            str(meta.task_name): meta
            for meta in task_metadata
            if meta.dataset_id == "deprest_cat"
        }
        gad7_cat_meta = task_meta_by_name.get("gad7_cat")
        gad7_reg_meta = task_meta_by_name.get("gad7_reg")
        if gad7_cat_meta is not None and gad7_reg_meta is not None:
            deprest_task_bridge_specs[int(gad7_cat_meta.task_index)] = {
                "source_task_index": int(gad7_reg_meta.task_index),
            }

    model = MCTRCMV2(
        modality_input_dims={
            modality: len(columns)
            for modality, columns in prepared.modality_feature_columns.items()
        },
        num_datasets=len(prepared.dataset_to_index),
        task_head_specs=_build_task_head_specs(task_metadata),
        native_input_dim=0 if bool(run_config.get("disable_native_branch", False)) else len(prepared.native_feature_columns),
        temporal_slice_dims=prepared.temporal_slice_dims,
        temporal_num_slices=prepared.temporal_num_slices,
        temporal_frontend_mode=str(model_config.get("temporal_frontend_mode", "none")),
        enable_missingness_tokens=bool(model_config.get("enable_missingness_tokens", False)),
        enable_missingness_embedding=bool(model_config.get("enable_missingness_embedding", False)),
        missing_signal_dim=int(prepared.missing_signal_dim),
        native_missing_signal_dim=int(prepared.native_missing_signal_dim),
        psyche_native_input_dim=len(prepared.native_feature_columns) if bool(run_config.get("enable_psyche_native_adapter", False)) else 0,
        psyche_dataset_index=None if not use_psyche_conditioning else prepared.dataset_to_index.get("psyche_d"),
        use_psyche_two_stage=bool(run_config.get("enable_psyche_two_stage", False)),
        psyche_change_task_indices=psyche_change_task_indices if bool(run_config.get("enable_psyche_two_stage", False)) else [],
        psyche_binary_correction_task_indices=psyche_binary_correction_task_indices if bool(run_config.get("enable_psyche_two_stage", False)) else [],
        use_psyche_native_adapter=bool(run_config.get("enable_psyche_native_adapter", False)),
        use_psyche_taskwise_native_adapter=bool(run_config.get("enable_psyche_taskwise_native_adapter", False)),
        psyche_native_task_indices=psyche_native_task_indices,
        deprest_dataset_index=None if not use_deprest_comm_features else prepared.dataset_to_index.get("deprest_cat"),
        deprest_comm_input_dim=0 if not use_deprest_comm_features else len(prepared.deprest_comm_feature_columns),
        deprest_comm_group_indices={} if not use_deprest_comm_features else deprest_comm_groups,
        use_deprest_global_adapter=not bool(run_config.get("disable_deprest_adapter", False)),
        use_deprest_category_adapter=bool(run_config.get("enable_deprest_category_adapter", False)),
        deprest_category_task_indices=deprest_category_task_indices if bool(run_config.get("enable_deprest_category_adapter", False)) else [],
        use_deprest_coverage_routing=bool(run_config.get("enable_deprest_coverage_routing", False)),
        deprest_coverage_task_indices=deprest_coverage_task_indices,
        use_deprest_concept_compatibility=bool(run_config.get("enable_deprest_concept_compatibility", False)),
        deprest_concept_compat_task_indices=deprest_concept_compat_task_indices,
        deprest_concept_compat_feature_indices=deprest_concept_feature_indices,
        use_deprest_concept_contrast=bool(run_config.get("enable_deprest_concept_contrast", False)),
        deprest_concept_contrast_task_indices=deprest_concept_contrast_task_indices,
        deprest_concept_contrast_feature_indices=deprest_concept_feature_indices,
        deprest_concept_contrast_pair_indices=[5, 6],
        deprest_task_bridge_specs=deprest_task_bridge_specs,
        deprest_severity_specs=deprest_severity_specs,
        deprest_pair_specs=deprest_pair_specs,
        deprest_construct_bridge_specs=deprest_construct_bridge_specs,
        use_concept_residual=bool(run_config.get("enable_concept_residual", False)),
        shared_token_refiner_type=str(model_config.get("shared_token_refiner_type", "none")),
        shared_token_refiner_steps=int(model_config.get("shared_token_refiner_steps", 0)),
        shared_token_refiner_nograd_steps=int(model_config.get("shared_token_refiner_nograd_steps", 0)),
        task_local_trm_mode=str(model_config.get("task_local_trm_mode", "none")),
        task_local_trm_reasoning_dim=int(model_config.get("task_local_trm_reasoning_dim", model_config["token_dim"])),
        task_local_trm_h_cycles=int(model_config.get("task_local_trm_h_cycles", 0)),
        task_local_trm_l_cycles=int(model_config.get("task_local_trm_l_cycles", 0)),
        task_local_trm_task_indices=task_local_trm_task_indices,
        use_psyche_delta_bridge=bool(run_config.get("enable_psyche_delta_bridge", False)),
        psyche_delta_bridge_task_indices=psyche_change_task_indices if bool(run_config.get("enable_psyche_delta_bridge", False)) else [],
        use_deprest_edge_specialist=bool(run_config.get("enable_deprest_edge_specialist", False)),
        deprest_edge_task_indices=deprest_edge_task_indices,
        use_deprest_edge_ovr=bool(run_config.get("enable_deprest_edge_ovr", False)),
        deprest_edge_ovr_task_indices=deprest_edge_ovr_task_indices,
        enable_tree_gated_head=bool(model_config.get("enable_tree_gated_head", False)),
        tree_head_task_indices=tree_head_task_indices,
        tree_head_depth=int(model_config.get("tree_head_depth", 2)),
        use_ordered_threshold_ordinal_head=bool(model_config.get("use_ordered_threshold_ordinal_head", False)),
        token_dim=int(model_config["token_dim"]),
        encoder_hidden_dim=int(model_config["encoder_hidden_dim"]),
        transformer_layers=int(model_config["transformer_layers"]),
        transformer_heads=int(model_config["transformer_heads"]),
        transformer_ff_dim=int(model_config["transformer_ff_dim"]),
        latent_dim=int(model_config["latent_dim"]),
        dataset_embedding_dim=int(model_config["dataset_embedding_dim"]),
        task_embedding_dim=int(model_config["task_embedding_dim"]),
        disable_task_conditioning=bool(model_config.get("disable_task_conditioning", False)),
        concept_dim=int(model_config["concept_dim"]),
        concept_hidden_dim=int(model_config["concept_hidden_dim"]),
        task_feature_dim=int(model_config.get("task_feature_dim", model_config["concept_hidden_dim"])),
        output_refine_dim=int(model_config["output_refine_dim"]),
        recursion_steps=int(model_config["recursion_steps"]),
        dropout=float(model_config["dropout"]),
        activation=str(model_config["activation"]),
    )
    metadata = {
        "task_metadata": task_metadata,
        "paired_regression_bridge_task_indices": paired_regression_bridge_specs,
    }
    return model, metadata


def load_saved_mctrcm_v2_run(
    run_name: str,
    *,
    device: torch.device | str | None = None,
) -> tuple[MCTRCMV2, PreparedMultiCorpusData, dict[str, object], dict[str, object]]:
    run_config = _load_run_config(run_name)
    include_native_features = (not bool(run_config.get("disable_native_branch", False))) or bool(
        run_config.get("enable_psyche_native_adapter", False)
    )
    include_augmented_communication_features = not bool(run_config.get("base_communication_only", False))
    prepared = prepare_multicorpus_data(
        list(run_config["datasets"]),
        include_native_features=include_native_features,
        include_augmented_communication_features=include_augmented_communication_features,
    )
    model, metadata = build_mctrcm_v2_from_saved_config(prepared, run_config)
    checkpoint_path = CHECKPOINT_DIR / f"{run_name}.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device or "cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if device is not None:
        model = model.to(device)
    model.eval()
    return model, prepared, run_config, metadata
