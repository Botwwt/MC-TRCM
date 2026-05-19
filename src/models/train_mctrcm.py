from __future__ import annotations

import argparse
import json
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.evaluation.metrics import compute_metrics
from src.models.mctrcm import MCTRCM, TaskHeadSpec
from src.models.mctrcm_data import PreparedMultiCorpusData, TaskMetadata, prepare_multicorpus_data
from src.utils.constants import CANONICAL_MODALITIES, CONCEPT_DEFINITIONS, DATASET_IDS, PROJECT_ROOT
from src.utils.io import ensure_dir, write_json

CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"
PREDICTION_DIR = PROJECT_ROOT / "outputs" / "predictions" / "mctrcm"
CONCEPT_DIR = PROJECT_ROOT / "outputs" / "concept_exports"
LOG_DIR = PROJECT_ROOT / "outputs" / "logs"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
MODEL_CONFIG_PATH = PROJECT_ROOT / "configs" / "model_configs" / "mctrcm_default.json"
TRAIN_CONFIG_PATH = PROJECT_ROOT / "configs" / "train_configs" / "default_train.json"
EXPERIMENT_REGISTRY_PATH = LOG_DIR / "experiment_registry.csv"
RESULTS_TABLE_PATH = TABLE_DIR / "mctrcm_results.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the minimal MC-TRCM model.")
    parser.add_argument("--datasets", nargs="*", default=list(DATASET_IDS), choices=DATASET_IDS)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260417)
    parser.add_argument("--run-name", type=str, default="mctrcm_smoke")
    parser.add_argument("--recursion-steps", type=int, default=None)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-epochs", type=int, default=2)
    parser.add_argument("--ssl-enabled", action="store_true")
    parser.add_argument("--ssl-weight", type=float, default=0.1)
    parser.add_argument("--ssl-mask-prob", type=float, default=0.15)
    parser.add_argument("--disable-dataset-embedding", action="store_true")
    parser.add_argument("--disable-modality-mask", action="store_true")
    parser.add_argument("--disable-concept-bottleneck", action="store_true")
    parser.add_argument("--l1-lambda", type=float, default=None)
    parser.add_argument("--group-lambda", type=float, default=None)
    return parser.parse_args()


def _load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _load_registry() -> pd.DataFrame:
    if EXPERIMENT_REGISTRY_PATH.exists():
        return pd.read_csv(EXPERIMENT_REGISTRY_PATH)
    return pd.DataFrame(
        columns=[
            "experiment_id",
            "stage",
            "created_at",
            "model_name",
            "training_corpora",
            "target_dataset",
            "task_name",
            "split_config",
            "model_config",
            "train_config",
            "status",
            "notes",
        ]
    )


def _append_registry_row(row: dict[str, object]) -> None:
    registry = _load_registry()
    registry = pd.concat([registry, pd.DataFrame([row])], ignore_index=True)
    registry.to_csv(EXPERIMENT_REGISTRY_PATH, index=False)


def _load_results_table() -> pd.DataFrame:
    if RESULTS_TABLE_PATH.exists():
        return pd.read_csv(RESULTS_TABLE_PATH)
    return pd.DataFrame()


def _append_result_rows(rows: list[dict[str, object]]) -> None:
    results = _load_results_table()
    results = pd.concat([results, pd.DataFrame(rows)], ignore_index=True)
    results = results.drop_duplicates(subset=["run_name", "split", "dataset_id", "task_name"], keep="last")
    ensure_dir(RESULTS_TABLE_PATH.parent)
    results.to_csv(RESULTS_TABLE_PATH, index=False)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _move_batch(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    moved: dict[str, object] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _task_metrics_score(label_type: str, metrics: dict[str, float]) -> float:
    if label_type == "continuous":
        r2 = metrics.get("r2")
        if r2 is not None and not np.isnan(r2):
            return float(r2)
        rmse = metrics.get("rmse")
        return float(-rmse) if rmse is not None and not np.isnan(rmse) else -1e9
    balanced_accuracy = metrics.get("balanced_accuracy")
    return float(balanced_accuracy) if balanced_accuracy is not None and not np.isnan(balanced_accuracy) else -1e9


def _build_ssl_batch(
    batch: dict[str, object],
    mask_probability: float,
) -> tuple[dict[str, object], dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    ssl_batch = dict(batch)
    reconstruction_targets: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    original_mask = batch["modality_mask"]
    ssl_mask = original_mask.clone()

    for modality_index, modality in enumerate(CANONICAL_MODALITIES):
        feature_key = f"{modality}_features"
        observed = original_mask[:, modality_index] > 0.5
        random_mask = torch.rand_like(original_mask[:, modality_index]) < mask_probability
        selected = observed & random_mask
        if torch.any(selected):
            reconstruction_targets[modality] = (batch[feature_key][selected].clone(), selected)
            ssl_features = batch[feature_key].clone()
            ssl_features[selected] = 0.0
            ssl_batch[feature_key] = ssl_features
            ssl_mask[selected, modality_index] = 0.0
    ssl_batch["modality_mask"] = ssl_mask
    return ssl_batch, reconstruction_targets


def _compute_ssl_loss(
    model: MCTRCM,
    ssl_outputs: dict[str, torch.Tensor],
    reconstruction_targets: dict[str, tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    hidden = ssl_outputs["hidden"]
    losses = []
    for modality in CANONICAL_MODALITIES:
        if modality not in reconstruction_targets:
            continue
        targets, selected = reconstruction_targets[modality]
        predictions = model.reconstruct_modality(hidden[selected], modality)
        losses.append(F.mse_loss(predictions, targets))
    if not losses:
        return torch.zeros((), device=hidden.device)
    return torch.stack(losses).mean()


def _compute_batch_loss(
    model: MCTRCM,
    batch: dict[str, object],
    task_metadata: dict[int, TaskMetadata],
    l1_lambda: float,
    group_lambda: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    outputs = model(batch)
    concepts = outputs["concepts"]
    task_inputs = model.task_representation(outputs)
    total_loss = torch.zeros((), device=concepts.device)
    task_count = 0
    for task_index, meta in task_metadata.items():
        mask = batch["task_index"] == task_index
        if not torch.any(mask):
            continue
        logits = model.task_logits(task_inputs[mask], task_index)
        if meta.label_type == "continuous":
            loss = F.huber_loss(logits.squeeze(-1), batch["target_float"][mask])
        else:
            loss = F.cross_entropy(logits, batch["target_index"][mask])
        total_loss = total_loss + loss
        task_count += 1
    total_loss = total_loss / max(task_count, 1)
    total_loss = total_loss + l1_lambda * concepts.abs().mean()
    total_loss = total_loss + group_lambda * model.sparse_penalty()
    return total_loss, outputs


def _evaluate_split(
    model: MCTRCM,
    loader: DataLoader,
    task_metadata: dict[int, TaskMetadata],
    device: torch.device,
    split_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    per_task_predictions: dict[int, dict[str, list[object]]] = {}

    model.eval()
    with torch.no_grad():
        for raw_batch in loader:
            batch = _move_batch(raw_batch, device)
            outputs = model(batch)
            concepts = outputs["concepts"]
            task_inputs = model.task_representation(outputs)

            for task_index, meta in task_metadata.items():
                mask = batch["task_index"] == task_index
                if not torch.any(mask):
                    continue
                indices = mask.nonzero(as_tuple=False).squeeze(-1).detach().cpu().tolist()
                logits = model.task_logits(task_inputs[mask], task_index)

                holder = per_task_predictions.setdefault(
                    task_index,
                    {
                        "y_true": [],
                        "y_pred": [],
                        "probabilities": [],
                    },
                )

                if meta.label_type == "continuous":
                    y_true = batch["target_float"][mask].detach().cpu().numpy()
                    y_pred = logits.squeeze(-1).detach().cpu().numpy()
                    holder["y_true"].extend(y_true.tolist())
                    holder["y_pred"].extend(y_pred.tolist())
                    probabilities = None
                else:
                    probabilities = torch.softmax(logits, dim=-1).detach().cpu().numpy()
                    y_true_index = batch["target_index"][mask].detach().cpu().numpy()
                    y_pred_index = probabilities.argmax(axis=1)
                    class_space = np.asarray(meta.class_space, dtype=object)
                    y_true = class_space[y_true_index]
                    y_pred = class_space[y_pred_index]
                    holder["y_true"].extend(y_true_index.tolist())
                    holder["y_pred"].extend(y_pred_index.tolist())
                    holder["probabilities"].append(probabilities)

                concept_values = concepts[mask].detach().cpu().numpy()
                for local_index, global_index in enumerate(indices):
                    row = {
                        "split": split_name,
                        "dataset_id": raw_batch["dataset_id"][global_index],
                        "subject_id": raw_batch["subject_id"][global_index],
                        "anchor_id": raw_batch["anchor_id"][global_index],
                        "task_name": raw_batch["task_name"][global_index],
                        "task_key": raw_batch["task_key"][global_index],
                        "label_type": raw_batch["label_type"][global_index],
                        "y_true": y_true[local_index],
                        "y_pred": y_pred[local_index],
                    }
                    if meta.label_type != "continuous":
                        row["y_true_index"] = int(holder["y_true"][-len(indices) + local_index])
                        row["y_pred_index"] = int(holder["y_pred"][-len(indices) + local_index])
                        for class_position, class_value in enumerate(meta.class_space or []):
                            row[f"proba_{class_value}"] = float(probabilities[local_index, class_position])
                    for concept_position, concept_name in enumerate(CONCEPT_DEFINITIONS):
                        row[f"concept_{concept_name}"] = float(concept_values[local_index, concept_position])
                    rows.append(row)

    score_values = []
    for task_index, meta in task_metadata.items():
        holder = per_task_predictions.get(task_index)
        if holder is None or not holder["y_true"]:
            continue
        probabilities = None
        if meta.label_type != "continuous":
            probabilities = np.concatenate(holder["probabilities"], axis=0)
        metrics = compute_metrics(
            meta.label_type,
            np.asarray(holder["y_true"]),
            np.asarray(holder["y_pred"]),
            probabilities,
        )
        metrics["split"] = split_name
        metrics["dataset_id"] = meta.dataset_id
        metrics["task_name"] = meta.task_name
        metrics["task_key"] = meta.task_key
        metric_rows.append(metrics)
        score_values.append(_task_metrics_score(meta.label_type, metrics))

    metric_frame = pd.DataFrame(metric_rows)
    prediction_frame = pd.DataFrame(rows)
    score = float(np.mean(score_values)) if score_values else -1e9
    return prediction_frame, metric_frame, score


def _metric_frame_to_records(
    metric_frame: pd.DataFrame,
    run_name: str,
    recursion_steps: int,
    datasets: list[str],
    ssl_enabled: bool,
    model_config: dict[str, object],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for row in metric_frame.to_dict(orient="records"):
        record = {
            "run_name": run_name,
            "model_name": "mctrcm",
            "recursion_steps": recursion_steps,
            "datasets": "|".join(datasets),
            "ssl_enabled": ssl_enabled,
            "use_dataset_embedding": bool(model_config["use_dataset_embedding"]),
            "use_modality_mask": bool(model_config["use_modality_mask"]),
            "use_concept_bottleneck": bool(model_config["use_concept_bottleneck"]),
            "l1_lambda": float(model_config["sparse_penalty"]["l1"]),
            "group_lambda": float(model_config["sparse_penalty"]["group"]),
        }
        record.update(row)
        records.append(record)
    return records


def main() -> None:
    args = parse_args()
    _set_seed(args.seed)

    model_config = _load_json(MODEL_CONFIG_PATH)
    train_config = _load_json(TRAIN_CONFIG_PATH)
    if args.recursion_steps is not None:
        model_config["recursion_steps"] = args.recursion_steps
    if args.disable_dataset_embedding:
        model_config["use_dataset_embedding"] = False
    if args.disable_modality_mask:
        model_config["use_modality_mask"] = False
    if args.disable_concept_bottleneck:
        model_config["use_concept_bottleneck"] = False
    if args.l1_lambda is not None:
        model_config["sparse_penalty"]["l1"] = float(args.l1_lambda)
    if args.group_lambda is not None:
        model_config["sparse_penalty"]["group"] = float(args.group_lambda)
    prepared = prepare_multicorpus_data(args.datasets)
    task_metadata = {meta.task_index: meta for meta in prepared.task_metadata}
    modality_input_dims = {
        modality: len(columns)
        for modality, columns in prepared.modality_feature_columns.items()
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MCTRCM(
        modality_input_dims=modality_input_dims,
        num_datasets=len(prepared.dataset_to_index),
        task_head_specs=[
            TaskHeadSpec(task_index=meta.task_index, output_dim=meta.output_dim)
            for meta in prepared.task_metadata
        ],
        hidden_dim=model_config["hidden_dim"],
        encoder_hidden_dim=model_config["encoder_hidden_dim"],
        concept_dim=model_config["concept_dim"],
        recursion_steps=model_config["recursion_steps"],
        dropout=model_config["dropout"],
        activation=model_config["activation"],
        use_layernorm=model_config["use_layernorm"],
        use_dataset_embedding=model_config["use_dataset_embedding"],
        use_modality_mask=model_config["use_modality_mask"],
        use_concept_bottleneck=model_config["use_concept_bottleneck"],
    ).to(device)

    sample_weights = prepared.train.dataset_sample_weights()
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )
    train_loader = DataLoader(prepared.train, batch_size=args.batch_size, sampler=sampler)
    valid_loader = DataLoader(prepared.valid, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(prepared.test, batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_score = -1e9
    best_epoch = 0
    stale_epochs = 0
    best_checkpoint = ensure_dir(CHECKPOINT_DIR) / f"{args.run_name}.pt"
    history: list[dict[str, object]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        for raw_batch in train_loader:
            batch = _move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = _compute_batch_loss(
                model,
                batch,
                task_metadata,
                l1_lambda=float(model_config["sparse_penalty"]["l1"]),
                group_lambda=float(model_config["sparse_penalty"]["group"]),
            )
            if args.ssl_enabled:
                ssl_batch, reconstruction_targets = _build_ssl_batch(batch, args.ssl_mask_prob)
                if reconstruction_targets:
                    ssl_outputs = model(ssl_batch)
                    ssl_loss = _compute_ssl_loss(model, ssl_outputs, reconstruction_targets)
                    loss = loss + args.ssl_weight * ssl_loss
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))

        _, valid_metrics, valid_score = _evaluate_split(model, valid_loader, task_metadata, device, "valid")
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(epoch_losses)) if epoch_losses else None,
                "valid_score": valid_score,
            }
        )
        if valid_score > best_score:
            best_score = valid_score
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": model_config,
                    "train_config": train_config,
                    "dataset_ids": args.datasets,
                    "task_metadata": [meta.__dict__ for meta in prepared.task_metadata],
                    "feature_stats": prepared.feature_stats,
                },
                best_checkpoint,
            )
        else:
            stale_epochs += 1

        if epoch >= args.min_epochs and stale_epochs >= args.patience:
            break

    checkpoint = torch.load(best_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    valid_predictions, valid_metrics, valid_score = _evaluate_split(model, valid_loader, task_metadata, device, "valid")
    test_predictions, test_metrics, test_score = _evaluate_split(model, test_loader, task_metadata, device, "test")

    ensure_dir(PREDICTION_DIR)
    ensure_dir(CONCEPT_DIR)
    valid_predictions.to_csv(PREDICTION_DIR / f"{args.run_name}__valid.csv", index=False)
    test_predictions.to_csv(PREDICTION_DIR / f"{args.run_name}__test.csv", index=False)
    pd.concat([valid_predictions, test_predictions], ignore_index=True).to_csv(
        CONCEPT_DIR / f"{args.run_name}__concepts.csv",
        index=False,
    )

    metric_table = pd.concat([valid_metrics, test_metrics], ignore_index=True)
    metric_table.to_csv(PREDICTION_DIR / f"{args.run_name}__metrics.csv", index=False)
    _append_result_rows(
        _metric_frame_to_records(
            metric_table,
            run_name=args.run_name,
            recursion_steps=int(model_config["recursion_steps"]),
            datasets=args.datasets,
            ssl_enabled=args.ssl_enabled,
            model_config=model_config,
        )
    )

    experiment_id = f"stage_06_mctrcm__{args.run_name}__{datetime.now().strftime('%Y%m%dT%H%M%S')}"
    _append_registry_row(
        {
            "experiment_id": experiment_id,
            "stage": "stage_06_model",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "model_name": f"mctrcm_k{model_config['recursion_steps']}",
            "training_corpora": "|".join(args.datasets),
            "target_dataset": "multi_corpus_bundle",
            "task_name": "multi_task_bundle",
            "split_config": "participant_level_dataset_specific_splits",
            "model_config": str(MODEL_CONFIG_PATH.relative_to(PROJECT_ROOT)),
            "train_config": str(TRAIN_CONFIG_PATH.relative_to(PROJECT_ROOT)),
            "status": "completed",
            "notes": (
                f"run_name={args.run_name}; best_valid_score={best_score:.6f}; "
                f"final_test_score={test_score:.6f}; best_epoch={best_epoch}; "
                f"ssl_enabled={args.ssl_enabled}; "
                f"use_dataset_embedding={model_config['use_dataset_embedding']}; "
                f"use_modality_mask={model_config['use_modality_mask']}; "
                f"use_concept_bottleneck={model_config['use_concept_bottleneck']}; "
                f"l1={model_config['sparse_penalty']['l1']}; "
                f"group={model_config['sparse_penalty']['group']}"
            ),
        }
    )
    summary = {
        "run_name": args.run_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "datasets": args.datasets,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "min_epochs": args.min_epochs,
        "recursion_steps": int(model_config["recursion_steps"]),
        "ssl_enabled": args.ssl_enabled,
        "ssl_weight": args.ssl_weight,
        "ssl_mask_prob": args.ssl_mask_prob,
        "use_dataset_embedding": bool(model_config["use_dataset_embedding"]),
        "use_modality_mask": bool(model_config["use_modality_mask"]),
        "use_concept_bottleneck": bool(model_config["use_concept_bottleneck"]),
        "l1_lambda": float(model_config["sparse_penalty"]["l1"]),
        "group_lambda": float(model_config["sparse_penalty"]["group"]),
        "best_valid_score": best_score,
        "final_valid_score": valid_score,
        "final_test_score": test_score,
        "checkpoint_path": str(best_checkpoint),
        "metric_path": str(PREDICTION_DIR / f"{args.run_name}__metrics.csv"),
        "history": history,
    }
    write_json(LOG_DIR / f"{args.run_name}__summary.json", summary)


if __name__ == "__main__":
    main()
