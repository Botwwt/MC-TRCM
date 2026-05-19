from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.utils.constants import CANONICAL_MODALITIES


def _activation(name: str) -> nn.Module:
    if name.lower() == "relu":
        return nn.ReLU()
    if name.lower() == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


class TinyModalityEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        activation: str,
        use_layernorm: bool,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Linear(input_dim, hidden_dim),
            _activation(activation),
        ]
        if use_layernorm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.extend(
            [
                nn.Linear(hidden_dim, output_dim),
                _activation(activation),
            ]
        )
        if use_layernorm:
            layers.append(nn.LayerNorm(output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class SharedRecursiveCore(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float, activation: str) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            _activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, hidden: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        delta = self.block(torch.cat([hidden, context], dim=-1))
        return self.norm(hidden + delta)


@dataclass
class TaskHeadSpec:
    task_index: int
    output_dim: int


class MCTRCM(nn.Module):
    def __init__(
        self,
        modality_input_dims: dict[str, int],
        num_datasets: int,
        task_head_specs: list[TaskHeadSpec],
        hidden_dim: int = 32,
        encoder_hidden_dim: int = 16,
        concept_dim: int = 8,
        recursion_steps: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_layernorm: bool = True,
        use_dataset_embedding: bool = True,
        use_modality_mask: bool = True,
        use_concept_bottleneck: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.concept_dim = concept_dim
        self.recursion_steps = recursion_steps
        self.use_dataset_embedding = use_dataset_embedding
        self.use_modality_mask = use_modality_mask
        self.use_concept_bottleneck = use_concept_bottleneck

        self.modality_encoders = nn.ModuleDict(
            {
                modality: TinyModalityEncoder(
                    input_dim=modality_input_dims[modality],
                    hidden_dim=encoder_hidden_dim,
                    output_dim=encoder_hidden_dim,
                    activation=activation,
                    use_layernorm=use_layernorm,
                )
                for modality in CANONICAL_MODALITIES
            }
        )

        dataset_embedding_dim = 8 if use_dataset_embedding else 0
        self.dataset_embedding = (
            nn.Embedding(num_datasets, dataset_embedding_dim) if use_dataset_embedding else None
        )
        mask_dim = len(CANONICAL_MODALITIES) if use_modality_mask else 0
        fusion_dim = len(CANONICAL_MODALITIES) * encoder_hidden_dim + mask_dim + dataset_embedding_dim

        self.input_heads = nn.ModuleDict(
            {
                str(dataset_index): nn.Sequential(
                    nn.Linear(fusion_dim, hidden_dim),
                    _activation(activation),
                    nn.LayerNorm(hidden_dim),
                )
                for dataset_index in range(num_datasets)
            }
        )
        self.recursive_core = SharedRecursiveCore(
            input_dim=hidden_dim + fusion_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            activation=activation,
        )
        self.concept_head = nn.Linear(hidden_dim, concept_dim)
        task_input_dim = concept_dim if use_concept_bottleneck else hidden_dim
        self.task_heads = nn.ModuleDict(
            {
                str(spec.task_index): nn.Linear(task_input_dim, spec.output_dim)
                for spec in task_head_specs
            }
        )
        self.reconstruction_heads = nn.ModuleDict(
            {
                modality: nn.Linear(hidden_dim, modality_input_dims[modality])
                for modality in CANONICAL_MODALITIES
            }
        )

    def _build_context(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        encoded_modalities = []
        modality_mask = batch["modality_mask"]
        for modality_index, modality in enumerate(CANONICAL_MODALITIES):
            encoded = self.modality_encoders[modality](batch[f"{modality}_features"])
            encoded = encoded * modality_mask[:, modality_index].unsqueeze(-1)
            encoded_modalities.append(encoded)
        pieces = [torch.cat(encoded_modalities, dim=-1)]
        if self.use_modality_mask:
            pieces.append(modality_mask)
        if self.use_dataset_embedding and self.dataset_embedding is not None:
            pieces.append(self.dataset_embedding(batch["dataset_index"]))
        return torch.cat(pieces, dim=-1)

    def _dataset_specific_input(self, context: torch.Tensor, dataset_index: torch.Tensor) -> torch.Tensor:
        hidden = torch.zeros(context.size(0), self.hidden_dim, device=context.device, dtype=context.dtype)
        for key, head in self.input_heads.items():
            mask = dataset_index == int(key)
            if mask.any():
                hidden[mask] = head(context[mask])
        return hidden

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        context = self._build_context(batch)
        hidden = self._dataset_specific_input(context, batch["dataset_index"])
        for _ in range(self.recursion_steps):
            hidden = self.recursive_core(hidden, context)
        raw_concepts = self.concept_head(hidden)
        concepts = raw_concepts * batch["concept_mask"]
        return {
            "context": context,
            "hidden": hidden,
            "raw_concepts": raw_concepts,
            "concepts": concepts,
        }

    def task_representation(self, outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.use_concept_bottleneck:
            return outputs["concepts"]
        return outputs["hidden"]

    def task_logits(self, representations: torch.Tensor, task_index: int) -> torch.Tensor:
        return self.task_heads[str(task_index)](representations)

    def reconstruct_modality(self, hidden: torch.Tensor, modality: str) -> torch.Tensor:
        return self.reconstruction_heads[modality](hidden)

    def sparse_penalty(self) -> torch.Tensor:
        penalties = []
        for head in self.task_heads.values():
            penalties.append(torch.sqrt(torch.sum(head.weight ** 2, dim=0) + 1e-8).mean())
        if not penalties:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        return torch.stack(penalties).mean()
