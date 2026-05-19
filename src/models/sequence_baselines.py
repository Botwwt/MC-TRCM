from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.metrics import compute_metrics
from src.models.baseline_data import TaskBundle

WINDOW_TOKENS = ("short", "medium", "long", "static")


@dataclass
class SequenceArrays:
    train: np.ndarray
    valid: np.ndarray
    test: np.ndarray
    feature_names: list[str]
    sequence_tokens: list[str]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _suffix_token(column: str) -> tuple[str, str]:
    for token in ("short", "medium", "long"):
        suffix = f"_{token}"
        if column.endswith(suffix):
            return token, column[: -len(suffix)]
    if column.startswith("feat_"):
        return "static", column
    if column.startswith("modality_mask_") or column.startswith("concept_mask_"):
        return "shared", column
    return "static", column


def prepare_sequence_arrays(bundle: TaskBundle) -> SequenceArrays:
    feature_names: set[str] = set()
    feature_mapping: dict[str, tuple[str, str]] = {}
    for column in bundle.feature_columns:
        token, base_name = _suffix_token(column)
        feature_mapping[column] = (token, base_name)
        feature_names.add(base_name)
    ordered_features = sorted(feature_names)
    feature_index = {name: idx for idx, name in enumerate(ordered_features)}
    token_index = {token: idx for idx, token in enumerate(WINDOW_TOKENS)}

    def build_array(frame: pd.DataFrame) -> np.ndarray:
        array = np.zeros((len(frame), len(WINDOW_TOKENS), len(ordered_features)), dtype=np.float32)
        for column in bundle.feature_columns:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float32)
            token, base_name = feature_mapping[column]
            feature_position = feature_index[base_name]
            if token == "shared":
                for token_name in WINDOW_TOKENS:
                    array[:, token_index[token_name], feature_position] = values
            else:
                array[:, token_index[token], feature_position] = values
        return array

    train = build_array(bundle.train_frame)
    valid = build_array(bundle.valid_frame)
    test = build_array(bundle.test_frame)

    means = np.nanmean(train, axis=0)
    means = np.nan_to_num(means, nan=0.0)
    stds = np.nanstd(train, axis=0)
    stds = np.nan_to_num(stds, nan=1.0)
    stds[stds == 0.0] = 1.0

    def normalize(array: np.ndarray) -> np.ndarray:
        normalized = (array - means) / stds
        return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    return SequenceArrays(
        train=normalize(train),
        valid=normalize(valid),
        test=normalize(test),
        feature_names=ordered_features,
        sequence_tokens=list(WINDOW_TOKENS),
    )


class GRUSequenceModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, 16)
        self.gru = nn.GRU(input_size=16, hidden_size=16, num_layers=1, batch_first=True)
        self.head = nn.Linear(16, output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        projected = torch.relu(self.input_projection(inputs))
        _, hidden = self.gru(projected)
        return self.head(hidden[-1])


class LSTMSequenceModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, 16)
        self.lstm = nn.LSTM(input_size=16, hidden_size=16, num_layers=1, batch_first=True)
        self.head = nn.Linear(16, output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        projected = torch.relu(self.input_projection(inputs))
        _, (hidden, _) = self.lstm(projected)
        return self.head(hidden[-1])


class TransformerSequenceModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, sequence_length: int) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, 16)
        self.position_embedding = nn.Parameter(torch.zeros(1, sequence_length, 16))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=16,
            nhead=2,
            dim_feedforward=32,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.head = nn.Linear(16, output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.input_projection(inputs))
        hidden = hidden + self.position_embedding[:, : hidden.size(1), :]
        encoded = self.encoder(hidden)
        pooled = encoded.mean(dim=1)
        return self.head(pooled)


def build_sequence_model(model_name: str, input_dim: int, output_dim: int, sequence_length: int) -> nn.Module:
    if model_name == "gru_small":
        return GRUSequenceModel(input_dim=input_dim, output_dim=output_dim)
    if model_name == "lstm_small":
        return LSTMSequenceModel(input_dim=input_dim, output_dim=output_dim)
    if model_name == "transformer_small":
        return TransformerSequenceModel(input_dim=input_dim, output_dim=output_dim, sequence_length=sequence_length)
    raise KeyError(f"Unknown sequence model: {model_name}")


def _task_score(label_type: str, metrics: dict[str, float]) -> float:
    if label_type == "continuous":
        r2 = metrics.get("r2")
        if r2 is not None and not np.isnan(r2):
            return float(r2)
        rmse = metrics.get("rmse")
        return float(-rmse) if rmse is not None and not np.isnan(rmse) else -1e9
    balanced_accuracy = metrics.get("balanced_accuracy")
    return float(balanced_accuracy) if balanced_accuracy is not None and not np.isnan(balanced_accuracy) else -1e9


def _predict(model: nn.Module, features: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray | None]:
    model.eval()
    tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = model(tensor)
    logits_np = logits.detach().cpu().numpy()
    if logits_np.ndim == 1 or logits_np.shape[1] == 1:
        return logits_np.reshape(-1), None
    probabilities = torch.softmax(torch.as_tensor(logits_np), dim=-1).numpy()
    predictions = probabilities.argmax(axis=1)
    return predictions, probabilities


def fit_sequence_baseline(
    model_name: str,
    label_type: str,
    x_train: np.ndarray,
    x_valid: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_valid: np.ndarray,
    y_test: np.ndarray,
    n_classes: int,
    seed: int,
) -> dict[str, object]:
    _set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dim = 1 if label_type == "continuous" else n_classes
    model = build_sequence_model(
        model_name=model_name,
        input_dim=x_train.shape[-1],
        output_dim=output_dim,
        sequence_length=x_train.shape[1],
    ).to(device)

    x_train_tensor = torch.as_tensor(np.array(x_train, copy=True), dtype=torch.float32)
    if label_type == "continuous":
        y_train_tensor = torch.as_tensor(np.array(y_train, copy=True), dtype=torch.float32)
    else:
        y_train_tensor = torch.as_tensor(np.array(y_train, copy=True), dtype=torch.long)
    train_loader = DataLoader(TensorDataset(x_train_tensor, y_train_tensor), batch_size=64, shuffle=True)

    if label_type == "continuous":
        criterion = nn.HuberLoss()
        class_weights = None
    else:
        counts = np.bincount(y_train, minlength=n_classes).astype(np.float32)
        counts[counts == 0.0] = 1.0
        weights = len(y_train) / (n_classes * counts)
        class_weights = torch.as_tensor(weights, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_state = None
    best_score = -1e9
    stale_epochs = 0
    for epoch in range(1, 61):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            if label_type == "continuous":
                loss = criterion(logits.squeeze(-1), batch_y)
            else:
                loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

        valid_pred_index, valid_proba = _predict(model, x_valid, device)
        if label_type == "continuous":
            valid_metrics = compute_metrics(label_type, y_valid, valid_pred_index)
        else:
            valid_metrics = compute_metrics(label_type, y_valid, valid_pred_index, valid_proba)
        score = _task_score(label_type, valid_metrics)
        if score > best_score:
            best_score = score
            stale_epochs = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale_epochs += 1
        if epoch >= 10 and stale_epochs >= 8:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_pred_index, train_proba = _predict(model, x_train, device)
    valid_pred_index, valid_proba = _predict(model, x_valid, device)
    test_pred_index, test_proba = _predict(model, x_test, device)
    if label_type == "continuous":
        train_metrics = compute_metrics(label_type, y_train, train_pred_index)
        valid_metrics = compute_metrics(label_type, y_valid, valid_pred_index)
        test_metrics = compute_metrics(label_type, y_test, test_pred_index)
    else:
        train_metrics = compute_metrics(label_type, y_train, train_pred_index, train_proba)
        valid_metrics = compute_metrics(label_type, y_valid, valid_pred_index, valid_proba)
        test_metrics = compute_metrics(label_type, y_test, test_pred_index, test_proba)

    return {
        "train_pred": train_pred_index,
        "valid_pred": valid_pred_index,
        "test_pred": test_pred_index,
        "train_proba": train_proba,
        "valid_proba": valid_proba,
        "test_proba": test_proba,
        "train_metrics": train_metrics,
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,
    }
