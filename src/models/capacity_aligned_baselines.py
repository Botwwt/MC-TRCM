from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.metrics import compute_metrics
from src.utils.constants import PROJECT_ROOT

CAPACITY_REFERENCE_SUMMARY_PATH = (
    PROJECT_ROOT / "outputs" / "logs" / "mctrcm_v2_psynativeadapter_distill_fromv181_v182__summary.json"
)
DEFAULT_TARGET_PARAMETER_COUNT = 104_503


@dataclass(frozen=True)
class CapacityBaselineConfig:
    model_name: str
    target_parameter_count: int
    parameter_count: int
    hidden_dim: int | None = None
    mlp_hidden_dims: tuple[int, int] | None = None
    transformer_dim: int | None = None
    transformer_ff_dim: int | None = None
    transformer_heads: int | None = None
    transformer_layers: int | None = None

    def to_note(self) -> str:
        payload = {key: value for key, value in asdict(self).items() if value is not None}
        return json.dumps(payload, sort_keys=True)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def resolve_target_parameter_count(default_value: int = DEFAULT_TARGET_PARAMETER_COUNT) -> int:
    if not CAPACITY_REFERENCE_SUMMARY_PATH.exists():
        return default_value
    try:
        payload = json.loads(CAPACITY_REFERENCE_SUMMARY_PATH.read_text(encoding="utf-8"))
        value = int(payload.get("parameter_count", default_value))
        return value if value > 0 else default_value
    except (ValueError, TypeError, json.JSONDecodeError):
        return default_value


class CapacityAlignedMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: tuple[int, int]) -> None:
        super().__init__()
        hidden_a, hidden_b = hidden_dims
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_a),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_a, hidden_b),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_b, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class CapacityAlignedGRU(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.input_projection(inputs))
        _, recurrent_hidden = self.gru(hidden)
        return self.head(recurrent_hidden[-1])


class CapacityAlignedLSTM(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.input_projection(inputs))
        _, (recurrent_hidden, _) = self.lstm(hidden)
        return self.head(recurrent_hidden[-1])


class CapacityAlignedTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        sequence_length: int,
        model_dim: int,
        num_heads: int,
        feedforward_dim: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, model_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, sequence_length, model_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(model_dim, output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.input_projection(inputs))
        hidden = hidden + self.position_embedding[:, : hidden.size(1), :]
        encoded = self.encoder(hidden)
        return self.head(encoded.mean(dim=1))


def _best_config(
    candidates: list[tuple[int, nn.Module, dict[str, int | tuple[int, int]]]],
    *,
    model_name: str,
    target_parameter_count: int,
) -> tuple[nn.Module, CapacityBaselineConfig]:
    best_difference = None
    best_model = None
    best_config = None
    for parameter_count, model, extra in candidates:
        difference = abs(parameter_count - target_parameter_count)
        if best_difference is None or difference < best_difference:
            best_difference = difference
            best_model = model
            best_config = CapacityBaselineConfig(
                model_name=model_name,
                target_parameter_count=target_parameter_count,
                parameter_count=parameter_count,
                hidden_dim=extra.get("hidden_dim") if "hidden_dim" in extra else None,
                mlp_hidden_dims=extra.get("mlp_hidden_dims") if "mlp_hidden_dims" in extra else None,
                transformer_dim=extra.get("transformer_dim") if "transformer_dim" in extra else None,
                transformer_ff_dim=extra.get("transformer_ff_dim") if "transformer_ff_dim" in extra else None,
                transformer_heads=extra.get("transformer_heads") if "transformer_heads" in extra else None,
                transformer_layers=extra.get("transformer_layers") if "transformer_layers" in extra else None,
            )
    if best_model is None or best_config is None:
        raise RuntimeError(f"Failed to resolve a capacity-aligned configuration for model={model_name}")
    return best_model, best_config


def build_capacity_aligned_model(
    model_name: str,
    *,
    input_dim: int,
    output_dim: int,
    sequence_length: int | None = None,
    target_parameter_count: int | None = None,
) -> tuple[nn.Module, CapacityBaselineConfig]:
    target_parameter_count = int(target_parameter_count or resolve_target_parameter_count())
    candidates: list[tuple[int, nn.Module, dict[str, int | tuple[int, int]]]] = []

    if model_name == "mlp":
        for hidden_dim in range(64, 513):
            hidden_dims = (hidden_dim, hidden_dim)
            model = CapacityAlignedMLP(input_dim=input_dim, output_dim=output_dim, hidden_dims=hidden_dims)
            candidates.append(
                (
                    _count_parameters(model),
                    model,
                    {"mlp_hidden_dims": hidden_dims},
                )
            )
        return _best_config(candidates, model_name=model_name, target_parameter_count=target_parameter_count)

    if model_name == "gru":
        for hidden_dim in range(32, 257):
            model = CapacityAlignedGRU(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim)
            candidates.append((_count_parameters(model), model, {"hidden_dim": hidden_dim}))
        return _best_config(candidates, model_name=model_name, target_parameter_count=target_parameter_count)

    if model_name == "lstm":
        for hidden_dim in range(32, 257):
            model = CapacityAlignedLSTM(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim)
            candidates.append((_count_parameters(model), model, {"hidden_dim": hidden_dim}))
        return _best_config(candidates, model_name=model_name, target_parameter_count=target_parameter_count)

    if model_name == "transformer":
        if sequence_length is None:
            raise ValueError("sequence_length is required for the capacity-aligned transformer baseline.")
        for model_dim in range(32, 257, 8):
            for num_heads in (4, 8):
                if model_dim % num_heads != 0:
                    continue
                for num_layers in (1, 2):
                    for ff_multiplier in (2, 4):
                        feedforward_dim = model_dim * ff_multiplier
                        model = CapacityAlignedTransformer(
                            input_dim=input_dim,
                            output_dim=output_dim,
                            sequence_length=sequence_length,
                            model_dim=model_dim,
                            num_heads=num_heads,
                            feedforward_dim=feedforward_dim,
                            num_layers=num_layers,
                        )
                        candidates.append(
                            (
                                _count_parameters(model),
                                model,
                                {
                                    "transformer_dim": model_dim,
                                    "transformer_ff_dim": feedforward_dim,
                                    "transformer_heads": num_heads,
                                    "transformer_layers": num_layers,
                                },
                            )
                        )
        return _best_config(candidates, model_name=model_name, target_parameter_count=target_parameter_count)

    raise KeyError(f"Unknown capacity-aligned baseline model: {model_name}")


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


def _task_score(label_type: str, metrics: dict[str, float]) -> float:
    if label_type == "continuous":
        r2 = metrics.get("r2")
        if r2 is not None and not np.isnan(r2):
            return float(r2)
        mae = metrics.get("mae")
        return float(-mae) if mae is not None and not np.isnan(mae) else -1e9
    balanced_accuracy = metrics.get("balanced_accuracy")
    return float(balanced_accuracy) if balanced_accuracy is not None and not np.isnan(balanced_accuracy) else -1e9


def fit_capacity_aligned_baseline(
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
    sequence_length: int | None = None,
    target_parameter_count: int | None = None,
) -> dict[str, object]:
    _set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dim = 1 if label_type == "continuous" else n_classes
    model, config = build_capacity_aligned_model(
        model_name=model_name,
        input_dim=int(x_train.shape[-1]),
        output_dim=output_dim,
        sequence_length=sequence_length,
        target_parameter_count=target_parameter_count,
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
        "parameter_count": config.parameter_count,
        "capacity_note": config.to_note(),
    }
