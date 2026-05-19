from __future__ import annotations

import json
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.metrics import compute_metrics


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _task_score(label_type: str, metrics: dict[str, float]) -> float:
    if label_type == "continuous":
        r2 = metrics.get("r2")
        if r2 is not None and not np.isnan(r2):
            return float(r2)
        mae = metrics.get("mae")
        return float(-mae) if mae is not None and not np.isnan(mae) else -1e9
    balanced_accuracy = metrics.get("balanced_accuracy")
    return float(balanced_accuracy) if balanced_accuracy is not None and not np.isnan(balanced_accuracy) else -1e9


class LastStepLinearModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, token_index: int = 0) -> None:
        super().__init__()
        self.token_index = token_index
        self.head = nn.Linear(input_dim, output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(inputs[:, self.token_index, :])


class FlattenLinearModel(nn.Module):
    def __init__(self, input_dim: int, sequence_length: int, output_dim: int) -> None:
        super().__init__()
        self.head = nn.Linear(input_dim * sequence_length, output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        flattened = inputs.reshape(inputs.size(0), -1)
        return self.head(flattened)


class LinearStateSpaceReadout(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim, bias=False)
        self.transition = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        state = inputs.new_zeros(inputs.size(0), self.transition.out_features)
        for step in range(inputs.size(1)):
            state = self.transition(state) + self.input_projection(inputs[:, step, :]) + self.bias
        return self.head(state)


class KalmanReadoutModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.observation_projection = nn.Linear(input_dim, hidden_dim)
        self.transition = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.observation_model = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.correction = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        state = inputs.new_zeros(inputs.size(0), self.transition.out_features)
        for step in range(inputs.size(1)):
            predicted_state = self.transition(state)
            observation_state = self.observation_projection(inputs[:, step, :])
            innovation = observation_state - self.observation_model(predicted_state)
            state = predicted_state + self.correction(innovation)
        return self.head(state)


class PLRNNReadoutModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.linear_transition = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.piecewise_transition = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.theta = nn.Parameter(torch.zeros(hidden_dim))
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        state = inputs.new_zeros(inputs.size(0), self.linear_transition.out_features)
        for step in range(inputs.size(1)):
            piecewise = self.piecewise_transition(torch.relu(state - self.theta))
            state = self.linear_transition(state) + piecewise + self.input_projection(inputs[:, step, :])
        return self.head(state)


def _predict(model: nn.Module, features: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray | None]:
    model.eval()
    tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = model(tensor)
    logits_np = logits.detach().cpu().numpy()
    if logits_np.ndim == 1 or logits_np.shape[1] == 1:
        return logits_np.reshape(-1), None
    probabilities = torch.softmax(torch.as_tensor(logits_np), dim=-1).cpu().numpy()
    predictions = probabilities.argmax(axis=1)
    return predictions, probabilities


def _global_mean_outputs(
    *,
    label_type: str,
    y_train: np.ndarray,
    y_valid: np.ndarray,
    y_test: np.ndarray,
    n_classes: int,
) -> dict[str, object]:
    if label_type == "continuous":
        constant = float(np.mean(y_train))
        train_pred = np.full_like(y_train, fill_value=constant, dtype=float)
        valid_pred = np.full_like(y_valid, fill_value=constant, dtype=float)
        test_pred = np.full_like(y_test, fill_value=constant, dtype=float)
        return {
            "train_pred": train_pred,
            "valid_pred": valid_pred,
            "test_pred": test_pred,
            "train_proba": None,
            "valid_proba": None,
            "test_proba": None,
            "train_metrics": compute_metrics(label_type, y_train, train_pred),
            "valid_metrics": compute_metrics(label_type, y_valid, valid_pred),
            "test_metrics": compute_metrics(label_type, y_test, test_pred),
            "parameter_count": 0,
            "paper_baseline_note": json.dumps({"model": "global_mean", "mode": "constant_train_mean"}, sort_keys=True),
        }

    counts = np.bincount(y_train, minlength=n_classes).astype(np.float32)
    counts[counts == 0.0] = 1.0
    probabilities = counts / counts.sum()
    prediction = int(np.argmax(probabilities))
    train_pred = np.full(len(y_train), fill_value=prediction, dtype=int)
    valid_pred = np.full(len(y_valid), fill_value=prediction, dtype=int)
    test_pred = np.full(len(y_test), fill_value=prediction, dtype=int)
    train_proba = np.tile(probabilities, (len(y_train), 1))
    valid_proba = np.tile(probabilities, (len(y_valid), 1))
    test_proba = np.tile(probabilities, (len(y_test), 1))
    return {
        "train_pred": train_pred,
        "valid_pred": valid_pred,
        "test_pred": test_pred,
        "train_proba": train_proba,
        "valid_proba": valid_proba,
        "test_proba": test_proba,
        "train_metrics": compute_metrics(label_type, y_train, train_pred, train_proba),
        "valid_metrics": compute_metrics(label_type, y_valid, valid_pred, valid_proba),
        "test_metrics": compute_metrics(label_type, y_test, test_pred, test_proba),
        "parameter_count": 0,
        "paper_baseline_note": json.dumps({"model": "global_mean", "mode": "class_prior"}, sort_keys=True),
    }


def _build_model(
    model_name: str,
    *,
    input_dim: int,
    output_dim: int,
    sequence_length: int,
) -> tuple[nn.Module, dict[str, object]]:
    hidden_dim = max(24, min(96, input_dim // 2 if input_dim >= 8 else 24))
    if model_name == "last_step":
        model = LastStepLinearModel(input_dim=input_dim, output_dim=output_dim, token_index=0)
        return model, {"mode": "short_token_linear_readout", "hidden_dim": None}
    if model_name == "linear_regression":
        model = FlattenLinearModel(input_dim=input_dim, sequence_length=sequence_length, output_dim=output_dim)
        return model, {"mode": "flattened_linear_readout", "hidden_dim": None}
    if model_name == "var1":
        model = LinearStateSpaceReadout(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim)
        return model, {"mode": "linear_state_space_readout", "hidden_dim": hidden_dim}
    if model_name == "kalman_filter":
        model = KalmanReadoutModel(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim)
        return model, {"mode": "kalman_like_filter_readout", "hidden_dim": hidden_dim}
    if model_name == "plrnn":
        model = PLRNNReadoutModel(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim)
        return model, {"mode": "piecewise_linear_rnn_readout", "hidden_dim": hidden_dim}
    raise KeyError(f"Unknown paper-inspired baseline model: {model_name}")


def fit_paper_sequence_baseline(
    model_name: str,
    *,
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
    if model_name == "global_mean":
        return _global_mean_outputs(
            label_type=label_type,
            y_train=y_train,
            y_valid=y_valid,
            y_test=y_test,
            n_classes=n_classes,
        )

    _set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dim = 1 if label_type == "continuous" else n_classes
    model, note_payload = _build_model(
        model_name,
        input_dim=int(x_train.shape[-1]),
        output_dim=output_dim,
        sequence_length=int(x_train.shape[1]),
    )
    model = model.to(device)

    x_train_tensor = torch.as_tensor(np.array(x_train, copy=True), dtype=torch.float32)
    if label_type == "continuous":
        y_train_tensor = torch.as_tensor(np.array(y_train, copy=True), dtype=torch.float32)
    else:
        y_train_tensor = torch.as_tensor(np.array(y_train, copy=True), dtype=torch.long)

    batch_size = 128 if len(y_train) >= 2048 else 64
    train_loader = DataLoader(TensorDataset(x_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)

    if label_type == "continuous":
        criterion = nn.SmoothL1Loss()
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

    for epoch in range(1, 81):
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        valid_pred, valid_proba = _predict(model, x_valid, device)
        if label_type == "continuous":
            valid_metrics = compute_metrics(label_type, y_valid, valid_pred)
        else:
            valid_metrics = compute_metrics(label_type, y_valid, valid_pred, valid_proba)
        score = _task_score(label_type, valid_metrics)
        if score > best_score:
            best_score = score
            stale_epochs = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale_epochs += 1
        if epoch >= 12 and stale_epochs >= 10:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_pred, train_proba = _predict(model, x_train, device)
    valid_pred, valid_proba = _predict(model, x_valid, device)
    test_pred, test_proba = _predict(model, x_test, device)

    if label_type == "continuous":
        train_metrics = compute_metrics(label_type, y_train, train_pred)
        valid_metrics = compute_metrics(label_type, y_valid, valid_pred)
        test_metrics = compute_metrics(label_type, y_test, test_pred)
    else:
        train_metrics = compute_metrics(label_type, y_train, train_pred, train_proba)
        valid_metrics = compute_metrics(label_type, y_valid, valid_pred, valid_proba)
        test_metrics = compute_metrics(label_type, y_test, test_pred, test_proba)

    note_payload = {
        "model": model_name,
        "adaptation": "paper_inspired_sequence_baseline",
        **note_payload,
    }
    return {
        "train_pred": train_pred,
        "valid_pred": valid_pred,
        "test_pred": test_pred,
        "train_proba": train_proba,
        "valid_proba": valid_proba,
        "test_proba": test_proba,
        "train_metrics": train_metrics,
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,
        "parameter_count": _count_parameters(model),
        "paper_baseline_note": json.dumps(note_payload, sort_keys=True),
    }
