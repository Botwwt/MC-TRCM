from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from src.utils.constants import CANONICAL_MODALITIES, concept_keys_for_dim


def _activation(name: str) -> nn.Module:
    if name.lower() == "relu":
        return nn.ReLU()
    if name.lower() == "gelu":
        return nn.GELU()
    if name.lower() == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name}")


def _ordinal_probabilities_torch(logits: torch.Tensor) -> torch.Tensor:
    cumulative = torch.sigmoid(logits)
    cumulative = torch.cummin(cumulative, dim=1).values
    num_classes = cumulative.size(1) + 1
    probabilities = logits.new_zeros((logits.size(0), num_classes))
    probabilities[:, 0] = 1.0 - cumulative[:, 0]
    for class_index in range(1, num_classes - 1):
        probabilities[:, class_index] = torch.clamp(cumulative[:, class_index - 1] - cumulative[:, class_index], min=0.0)
    probabilities[:, -1] = torch.clamp(cumulative[:, -1], min=0.0)
    probabilities = probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return probabilities


class ModalityTokenEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        token_dim: int,
        dropout: float,
        activation: str,
    ) -> None:
        super().__init__()
        self.feature_norm = nn.LayerNorm(input_dim)
        self.token_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            _activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, token_dim),
        )
        self.token_norm = nn.LayerNorm(token_dim)
        self.gate_mlp = nn.Sequential(
            nn.Linear(input_dim, max(hidden_dim // 2, 8)),
            _activation(activation),
            nn.Linear(max(hidden_dim // 2, 8), 1),
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = self.feature_norm(features)
        token = self.token_norm(self.token_mlp(normalized))
        gate = torch.sigmoid(self.gate_mlp(normalized))
        return token * gate, gate


class TemporalFrontEnd(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        token_dim: int,
        num_slices: int,
        hidden_dim: int,
        dropout: float,
        activation: str,
        mode: str,
    ) -> None:
        super().__init__()
        self.mode = str(mode or "none").lower()
        self.num_slices = max(int(num_slices), 0)
        self.token_dim = int(token_dim)
        self.has_inputs = input_dim > 0 and self.num_slices > 0
        if not self.has_inputs or self.mode == "none":
            self.slice_norm = None
            self.slice_projection = None
            self.sequence_model = None
            self.output_layer = None
            self.recency_logits = None
            return

        self.slice_norm = nn.LayerNorm(input_dim)
        self.slice_projection = nn.Linear(input_dim, token_dim)
        self.output_layer = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, hidden_dim),
            _activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, token_dim),
        )
        if self.mode == "gru":
            self.sequence_model: nn.Module | None = nn.GRU(
                input_size=token_dim,
                hidden_size=token_dim,
                batch_first=True,
            )
            self.recency_logits = None
        elif self.mode == "recency_pool":
            self.sequence_model = None
            self.recency_logits = nn.Parameter(torch.linspace(0.75, -0.25, steps=self.num_slices))
        else:
            raise ValueError(f"Unsupported temporal_frontend_mode: {self.mode}")

    def enabled(self) -> bool:
        return self.has_inputs and self.mode != "none"

    def forward(self, slices: torch.Tensor, slice_mask: torch.Tensor) -> torch.Tensor:
        if not self.enabled() or self.slice_norm is None or self.slice_projection is None or self.output_layer is None:
            return slices.new_zeros((slices.size(0), self.token_dim))

        encoded = self.slice_projection(self.slice_norm(slices))
        valid_mask = slice_mask.unsqueeze(-1).to(encoded.dtype)
        encoded = encoded * valid_mask
        has_signal = (slice_mask.sum(dim=1, keepdim=True) > 0).to(encoded.dtype)

        if self.mode == "gru" and self.sequence_model is not None:
            sequence_outputs, _ = self.sequence_model(encoded)
            last_index = slice_mask.sum(dim=1).long().clamp_min(1) - 1
            gather_index = last_index.view(-1, 1, 1).expand(-1, 1, sequence_outputs.size(-1))
            pooled = sequence_outputs.gather(1, gather_index).squeeze(1)
        elif self.mode == "recency_pool" and self.recency_logits is not None:
            recency_logits = self.recency_logits.view(1, -1).expand(slice_mask.size(0), -1)
            recency_logits = recency_logits.masked_fill(slice_mask < 0.5, -1e4)
            pooled_weights = torch.softmax(recency_logits, dim=-1) * slice_mask
            pooled_weights = pooled_weights / pooled_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
            pooled = torch.sum(encoded * pooled_weights.unsqueeze(-1), dim=1)
        else:
            pooled = torch.sum(encoded, dim=1) / valid_mask.sum(dim=1).clamp_min(1.0)

        return self.output_layer(pooled) * has_signal


class SoftTreeHead(nn.Module):
    def __init__(
        self,
        *,
        feature_dim: int,
        output_dim: int,
        depth: int,
        hidden_dim: int,
        dropout: float,
        activation: str,
    ) -> None:
        super().__init__()
        self.depth = max(int(depth), 1)
        self.num_leaves = 2 ** self.depth
        self.split_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(feature_dim),
                    nn.Linear(feature_dim, hidden_dim),
                    _activation(activation),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 1),
                )
                for _ in range(self.depth)
            ]
        )
        self.leaf_experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(feature_dim),
                    nn.Linear(feature_dim, hidden_dim),
                    _activation(activation),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, output_dim),
                )
                for _ in range(self.num_leaves)
            ]
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        path_probabilities = features.new_ones((features.size(0), 1))
        for split_layer in self.split_layers:
            split_probability = torch.sigmoid(split_layer(features))
            path_probabilities = torch.cat(
                [
                    path_probabilities * (1.0 - split_probability),
                    path_probabilities * split_probability,
                ],
                dim=1,
            )
        leaf_outputs = torch.stack([expert(features) for expert in self.leaf_experts], dim=1)
        return torch.sum(leaf_outputs * path_probabilities.unsqueeze(-1), dim=1)

    def sparse_penalty(self) -> torch.Tensor:
        penalties = []
        for expert in self.leaf_experts:
            final_linear = expert[-1]
            penalties.append(torch.sqrt(torch.sum(final_linear.weight ** 2, dim=0) + 1e-8).mean())
        if not penalties:
            return torch.tensor(0.0, device=self.leaf_experts[0][-1].weight.device)
        return torch.stack(penalties).mean()


class OrderedThresholdOrdinalHead(nn.Module):
    def __init__(
        self,
        *,
        feature_dim: int,
        num_classes: int,
        hidden_dim: int,
        dropout: float,
        activation: str,
    ) -> None:
        super().__init__()
        if int(num_classes) < 2:
            raise ValueError("OrderedThresholdOrdinalHead requires at least two classes.")
        self.num_classes = int(num_classes)
        self.score_net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            _activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.first_threshold = nn.Parameter(torch.tensor(0.0))
        self.raw_threshold_increments = nn.Parameter(torch.zeros(max(self.num_classes - 2, 0)))

    def thresholds(self) -> torch.Tensor:
        if self.num_classes == 2:
            return self.first_threshold.view(1)
        increments = F.softplus(self.raw_threshold_increments) + 1e-4
        return torch.cat(
            [
                self.first_threshold.view(1),
                self.first_threshold + torch.cumsum(increments, dim=0),
            ],
            dim=0,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        score = self.score_net(features)
        thresholds = self.thresholds().to(device=features.device, dtype=features.dtype).view(1, -1)
        return score - thresholds


class FiLMAdapter(nn.Module):
    def __init__(self, condition_dim: int, target_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, target_dim * 2),
        )
        self.target_dim = target_dim

    def forward(self, tensor: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        scale_shift = self.network(condition)
        scale, shift = scale_shift.chunk(2, dim=-1)
        while scale.dim() < tensor.dim():
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)
        return tensor * (1.0 + 0.1 * torch.tanh(scale)) + 0.1 * shift


class DatasetConditionedTransformerBlock(nn.Module):
    def __init__(
        self,
        token_dim: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float,
        condition_dim: int,
        use_film: bool = True,
    ) -> None:
        super().__init__()
        self.use_film = bool(use_film)
        self.attn_norm = nn.LayerNorm(token_dim)
        self.ffn_norm = nn.LayerNorm(token_dim)
        self.attn_film = (
            FiLMAdapter(condition_dim=condition_dim, target_dim=token_dim, hidden_dim=token_dim)
            if self.use_film
            else None
        )
        self.ffn_film = (
            FiLMAdapter(condition_dim=condition_dim, target_dim=token_dim, hidden_dim=token_dim)
            if self.use_film
            else None
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(token_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, token_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tokens: torch.Tensor,
        condition: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        attn_inputs = self.attn_norm(tokens)
        if self.attn_film is not None:
            attn_inputs = self.attn_film(attn_inputs, condition)
        attn_outputs, _ = self.attention(
            attn_inputs,
            attn_inputs,
            attn_inputs,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        tokens = tokens + self.dropout(attn_outputs)
        ffn_inputs = self.ffn_norm(tokens)
        if self.ffn_film is not None:
            ffn_inputs = self.ffn_film(ffn_inputs, condition)
        tokens = tokens + self.dropout(self.ffn(ffn_inputs))
        return tokens


class DatasetConditionedMLPTokenMixerBlock(nn.Module):
    def __init__(
        self,
        token_dim: int,
        token_count: int,
        feedforward_dim: int,
        dropout: float,
        condition_dim: int,
        activation: str,
        use_film: bool = True,
    ) -> None:
        super().__init__()
        self.use_film = bool(use_film)
        mixer_hidden_dim = max(token_count * 2, 8)
        self.token_norm = nn.LayerNorm(token_dim)
        self.channel_norm = nn.LayerNorm(token_dim)
        self.token_film = (
            FiLMAdapter(condition_dim=condition_dim, target_dim=token_dim, hidden_dim=token_dim)
            if self.use_film
            else None
        )
        self.channel_film = (
            FiLMAdapter(condition_dim=condition_dim, target_dim=token_dim, hidden_dim=token_dim)
            if self.use_film
            else None
        )
        self.token_mixer = nn.Sequential(
            nn.Linear(token_count, mixer_hidden_dim),
            _activation(activation),
            nn.Dropout(dropout),
            nn.Linear(mixer_hidden_dim, token_count),
        )
        self.channel_mlp = nn.Sequential(
            nn.Linear(token_dim, feedforward_dim),
            _activation(activation),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, token_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tokens: torch.Tensor,
        condition: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        valid_mask = (~padding_mask).unsqueeze(-1).to(tokens.dtype)
        token_inputs = self.token_norm(tokens)
        if self.token_film is not None:
            token_inputs = self.token_film(token_inputs, condition)
        token_inputs = token_inputs * valid_mask
        mixed_tokens = self.token_mixer(token_inputs.transpose(1, 2)).transpose(1, 2) * valid_mask
        tokens = tokens + self.dropout(mixed_tokens)
        channel_inputs = self.channel_norm(tokens)
        if self.channel_film is not None:
            channel_inputs = self.channel_film(channel_inputs, condition)
        channel_inputs = channel_inputs * valid_mask
        tokens = tokens + self.dropout(self.channel_mlp(channel_inputs)) * valid_mask
        return tokens


class TaskConditionedRoutingBlock(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        context_dim: int,
        hidden_dim: int,
        num_experts: int,
        dropout: float,
        activation: str,
    ) -> None:
        super().__init__()
        routing_input_dim = feature_dim + context_dim
        self.route_net = nn.Sequential(
            nn.Linear(routing_input_dim, hidden_dim),
            _activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_experts),
        )
        self.gate_net = nn.Sequential(
            nn.Linear(routing_input_dim, hidden_dim),
            _activation(activation),
            nn.Linear(hidden_dim, feature_dim),
        )
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(feature_dim, hidden_dim),
                    _activation(activation),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, feature_dim),
                )
                for _ in range(num_experts)
            ]
        )
        self.output_scale = nn.Parameter(torch.tensor(-1.5))
        nn.init.zeros_(self.route_net[-1].weight)
        nn.init.zeros_(self.route_net[-1].bias)
        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.constant_(self.gate_net[-1].bias, -1.5)
        for expert in self.experts:
            nn.init.normal_(expert[-1].weight, mean=0.0, std=0.01)
            nn.init.zeros_(expert[-1].bias)

    def forward(self, features: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        routing_input = torch.cat([features, context], dim=-1)
        route_weights = torch.softmax(self.route_net(routing_input), dim=-1)
        expert_outputs = torch.stack([expert(features) for expert in self.experts], dim=1)
        mixed_output = torch.sum(expert_outputs * route_weights.unsqueeze(-1), dim=1)
        gate = torch.sigmoid(self.gate_net(routing_input))
        scale = torch.sigmoid(self.output_scale)
        routed_features = features + scale * gate * mixed_output
        return routed_features, route_weights, scale.view(1, 1).expand(features.size(0), 1)


class TaskConditionedConceptCompatibility(nn.Module):
    def __init__(
        self,
        concept_dim: int,
        context_dim: int,
        hidden_dim: int,
        dropout: float,
        activation: str,
    ) -> None:
        super().__init__()
        input_dim = concept_dim + context_dim
        self.delta_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            _activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, concept_dim),
        )
        self.gate_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            _activation(activation),
            nn.Linear(hidden_dim, concept_dim),
        )
        self.output_scale = nn.Parameter(torch.tensor(-1.0))
        nn.init.normal_(self.delta_net[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.delta_net[-1].bias)
        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.constant_(self.gate_net[-1].bias, -1.0)

    def forward(self, concepts: torch.Tensor, context: torch.Tensor, concept_mask: torch.Tensor) -> torch.Tensor:
        compatibility_input = torch.cat([concepts, context], dim=-1)
        delta = torch.tanh(self.delta_net(compatibility_input))
        gate = torch.sigmoid(self.gate_net(compatibility_input))
        scale = torch.sigmoid(self.output_scale)
        corrected = concepts + scale * gate * delta * concept_mask
        return corrected * concept_mask


class TaskConditionedConceptContrastCompatibility(nn.Module):
    def __init__(
        self,
        context_dim: int,
        pair_indices: list[int],
        hidden_dim: int,
        dropout: float,
        activation: str,
    ) -> None:
        super().__init__()
        if len(pair_indices) != 2:
            raise ValueError(f"Expected exactly 2 pair indices, got {pair_indices}")
        self.left_index = int(pair_indices[0])
        self.right_index = int(pair_indices[1])
        input_dim = 2 + context_dim
        self.delta_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            _activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.gate_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            _activation(activation),
            nn.Linear(hidden_dim, 1),
        )
        self.output_scale = nn.Parameter(torch.tensor(-1.25))
        nn.init.normal_(self.delta_net[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.delta_net[-1].bias)
        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.constant_(self.gate_net[-1].bias, -1.25)

    def forward(self, concepts: torch.Tensor, context: torch.Tensor, concept_mask: torch.Tensor) -> torch.Tensor:
        pair = torch.stack([concepts[:, self.left_index], concepts[:, self.right_index]], dim=-1)
        contrast_input = torch.cat([pair, context], dim=-1)
        delta = torch.tanh(self.delta_net(contrast_input))
        gate = torch.sigmoid(self.gate_net(contrast_input))
        scale = torch.sigmoid(self.output_scale)
        signed_delta = scale * gate * delta
        corrected = concepts.clone()
        corrected[:, self.left_index] = corrected[:, self.left_index] + signed_delta[:, 0] * concept_mask[:, self.left_index]
        corrected[:, self.right_index] = corrected[:, self.right_index] - signed_delta[:, 0] * concept_mask[:, self.right_index]
        return corrected * concept_mask


class TaskLocalTRMRefiner(nn.Module):
    def __init__(
        self,
        *,
        mode: str,
        latent_dim: int,
        concept_dim: int,
        concept_hidden_dim: int,
        output_refine_dim: int,
        conditioning_dim: int,
        modality_gate_dim: int,
        deprest_context_dim: int,
        psyche_context_dim: int,
        reasoning_dim: int,
        feedforward_dim: int,
        num_heads: int,
        h_cycles: int,
        l_cycles: int,
        dropout: float,
        activation: str,
    ) -> None:
        super().__init__()
        self.mode = str(mode).lower()
        self.h_cycles = max(int(h_cycles), 0)
        self.l_cycles = max(int(l_cycles), 0)
        self.token_count = 5

        if self.mode == "attention":
            self.reasoning_block: nn.Module | None = DatasetConditionedTransformerBlock(
                token_dim=reasoning_dim,
                num_heads=num_heads,
                feedforward_dim=feedforward_dim,
                dropout=dropout,
                condition_dim=conditioning_dim,
            )
        elif self.mode == "mlp":
            self.reasoning_block = DatasetConditionedMLPTokenMixerBlock(
                token_dim=reasoning_dim,
                token_count=self.token_count,
                feedforward_dim=feedforward_dim,
                dropout=dropout,
                condition_dim=conditioning_dim,
                activation=activation,
            )
        elif self.mode == "none":
            self.reasoning_block = None
        else:
            raise ValueError(f"Unsupported task-local TRM mode: {self.mode}")

        self.latent_token = nn.Linear(latent_dim, reasoning_dim)
        self.concept_token = nn.Linear(concept_dim, reasoning_dim)
        self.feature_token = nn.Linear(concept_hidden_dim, reasoning_dim)
        self.output_token = nn.Linear(output_refine_dim, reasoning_dim)
        self.condition_token = nn.Linear(conditioning_dim, reasoning_dim)
        self.generic_context_encoder = nn.Sequential(
            nn.Linear(conditioning_dim + modality_gate_dim, reasoning_dim),
            _activation(activation),
            nn.Dropout(dropout),
            nn.Linear(reasoning_dim, reasoning_dim),
        )
        self.deprest_context_encoder = (
            nn.Sequential(
                nn.Linear(deprest_context_dim, reasoning_dim),
                _activation(activation),
                nn.Dropout(dropout),
                nn.Linear(reasoning_dim, reasoning_dim),
            )
            if deprest_context_dim > 0
            else None
        )
        self.psyche_context_encoder = (
            nn.Sequential(
                nn.Linear(psyche_context_dim, reasoning_dim),
                _activation(activation),
                nn.Dropout(dropout),
                nn.Linear(reasoning_dim, reasoning_dim),
            )
            if psyche_context_dim > 0
            else None
        )
        self.high_init = nn.Parameter(torch.randn(1, self.token_count, reasoning_dim) * 0.02)
        self.low_init = nn.Parameter(torch.randn(1, self.token_count, reasoning_dim) * 0.02)
        self.token_norm = nn.LayerNorm(reasoning_dim)
        self.state_norm = nn.LayerNorm(reasoning_dim)
        self.readout_norm = nn.LayerNorm(reasoning_dim)
        self.latent_delta = nn.Linear(reasoning_dim, latent_dim)
        self.latent_gate = nn.Linear(reasoning_dim, latent_dim)
        self.concept_delta = nn.Linear(reasoning_dim, concept_dim)
        self.concept_gate = nn.Linear(reasoning_dim, concept_dim)
        self.feature_delta = nn.Linear(reasoning_dim, concept_hidden_dim)
        self.feature_gate = nn.Linear(reasoning_dim, concept_hidden_dim)
        self.output_delta = nn.Linear(reasoning_dim, output_refine_dim)
        self.output_gate = nn.Linear(reasoning_dim, output_refine_dim)
        self.latent_scale = nn.Parameter(torch.tensor(-2.5))
        self.concept_scale = nn.Parameter(torch.tensor(-2.5))
        self.feature_scale = nn.Parameter(torch.tensor(-2.25))
        self.output_scale = nn.Parameter(torch.tensor(-2.5))

        for layer in (
            self.latent_delta,
            self.concept_delta,
            self.feature_delta,
            self.output_delta,
        ):
            nn.init.normal_(layer.weight, mean=0.0, std=0.02)
            nn.init.zeros_(layer.bias)
        for layer in (
            self.latent_gate,
            self.concept_gate,
            self.feature_gate,
            self.output_gate,
        ):
            nn.init.zeros_(layer.weight)
            nn.init.constant_(layer.bias, -2.0)

    def enabled(self) -> bool:
        return self.reasoning_block is not None and self.h_cycles > 0 and self.l_cycles > 0

    def _run_reasoning_block(
        self,
        state_tokens: torch.Tensor,
        input_tokens: torch.Tensor,
        conditioning: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.reasoning_block is None:
            return state_tokens
        hidden = self.token_norm(state_tokens + input_tokens)
        hidden = self.reasoning_block(hidden, conditioning, padding_mask)
        return self.state_norm(hidden)

    def _apply_residual_update(
        self,
        base: torch.Tensor,
        hidden: torch.Tensor,
        delta_layer: nn.Linear,
        gate_layer: nn.Linear,
        scale: nn.Parameter,
    ) -> torch.Tensor:
        normalized = self.readout_norm(hidden)
        delta = torch.tanh(delta_layer(normalized))
        gate = torch.sigmoid(gate_layer(normalized))
        return base + torch.sigmoid(scale) * gate * delta

    def forward(
        self,
        *,
        latent: torch.Tensor,
        concepts: torch.Tensor,
        task_features: torch.Tensor,
        output_state: torch.Tensor,
        conditioning: torch.Tensor,
        modality_gates: torch.Tensor,
        concept_mask: torch.Tensor,
        deprest_context: torch.Tensor | None = None,
        deprest_context_mask: torch.Tensor | None = None,
        psyche_context: torch.Tensor | None = None,
        psyche_context_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.enabled():
            return latent, concepts, task_features.new_zeros(task_features.shape), output_state

        context_token = self.condition_token(conditioning)
        generic_context = torch.cat([conditioning, modality_gates], dim=-1)
        context_token = context_token + self.generic_context_encoder(generic_context)
        if self.deprest_context_encoder is not None and deprest_context is not None:
            deprest_token = self.deprest_context_encoder(deprest_context)
            if deprest_context_mask is not None:
                deprest_token = deprest_token * deprest_context_mask.unsqueeze(-1).to(deprest_token.dtype)
            context_token = context_token + deprest_token
        if self.psyche_context_encoder is not None and psyche_context is not None:
            psyche_token = self.psyche_context_encoder(psyche_context)
            if psyche_context_mask is not None:
                psyche_token = psyche_token * psyche_context_mask.unsqueeze(-1).to(psyche_token.dtype)
            context_token = context_token + psyche_token

        input_tokens = torch.stack(
            [
                self.latent_token(latent),
                self.concept_token(concepts),
                self.feature_token(task_features),
                self.output_token(output_state),
                context_token,
            ],
            dim=1,
        )
        z_h = self.state_norm(input_tokens + self.high_init.expand(input_tokens.size(0), -1, -1))
        z_l = self.state_norm(input_tokens + self.low_init.expand(input_tokens.size(0), -1, -1))
        padding_mask = torch.zeros(
            (input_tokens.size(0), self.token_count),
            dtype=torch.bool,
            device=input_tokens.device,
        )
        if self.h_cycles > 1:
            with torch.no_grad():
                for _ in range(self.h_cycles - 1):
                    for _ in range(self.l_cycles):
                        z_l = self._run_reasoning_block(z_l, z_h + input_tokens, conditioning, padding_mask)
                    z_h = self._run_reasoning_block(z_h, z_l, conditioning, padding_mask)
        for _ in range(self.l_cycles):
            z_l = self._run_reasoning_block(z_l, z_h + input_tokens, conditioning, padding_mask)
        z_h = self._run_reasoning_block(z_h, z_l, conditioning, padding_mask)

        mixed = 0.5 * (z_h + z_l)
        refined_latent = self._apply_residual_update(latent, mixed[:, 0], self.latent_delta, self.latent_gate, self.latent_scale)
        refined_concepts = self._apply_residual_update(
            concepts,
            mixed[:, 1],
            self.concept_delta,
            self.concept_gate,
            self.concept_scale,
        ) * concept_mask
        feature_delta = self._apply_residual_update(
            task_features.new_zeros(task_features.shape),
            mixed[:, 2],
            self.feature_delta,
            self.feature_gate,
            self.feature_scale,
        )
        refined_output_state = self._apply_residual_update(
            output_state,
            mixed[:, 3],
            self.output_delta,
            self.output_gate,
            self.output_scale,
        )
        return refined_latent, refined_concepts, feature_delta, refined_output_state


class ConstructSeverityBridge(nn.Module):
    def __init__(
        self,
        *,
        feature_dim: int,
        score_max: float,
        thresholds: list[float],
        hidden_dim: int,
        dropout: float,
        activation: str,
    ) -> None:
        super().__init__()
        if not thresholds:
            raise ValueError("ConstructSeverityBridge requires at least one threshold.")
        self.score_max = float(score_max)
        self.score_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            _activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.residual_head = nn.Sequential(
            nn.Linear(feature_dim + 1, hidden_dim),
            _activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(thresholds)),
        )
        self.log_scale = nn.Parameter(torch.tensor(0.0))
        self.mix_logit = nn.Parameter(torch.tensor(-1.75))
        self.register_buffer("thresholds", torch.tensor(thresholds, dtype=torch.float32))
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_score = self.score_head(features)
        predicted_score = torch.sigmoid(raw_score) * self.score_max
        scale = F.softplus(self.log_scale).view(1, 1) + 0.5
        bridge_logits = (predicted_score - self.thresholds.view(1, -1)) * scale
        residual_input = torch.cat([features, predicted_score / max(self.score_max, 1.0)], dim=-1)
        bridge_logits = bridge_logits + 0.15 * self.residual_head(residual_input)
        mix = torch.sigmoid(self.mix_logit).view(1, 1)
        return predicted_score, bridge_logits, mix


class PsycheDeltaBridge(nn.Module):
    def __init__(
        self,
        *,
        feature_dim: int,
        num_start_categories: int,
        score_max: float,
        hidden_dim: int,
        dropout: float,
        activation: str,
    ) -> None:
        super().__init__()
        self.score_max = float(score_max)
        self.start_category_embedding = nn.Embedding(num_start_categories, hidden_dim)
        self.start_score_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            _activation(activation),
            nn.Linear(hidden_dim, hidden_dim),
        )
        bridge_input_dim = feature_dim + hidden_dim + hidden_dim
        self.delta_head = nn.Sequential(
            nn.Linear(bridge_input_dim, hidden_dim),
            _activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        residual_input_dim = bridge_input_dim + 1
        self.binary_residual = nn.Sequential(
            nn.Linear(residual_input_dim, hidden_dim),
            _activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.multiclass_residual = nn.Sequential(
            nn.Linear(residual_input_dim, hidden_dim),
            _activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )
        self.delta_scale = nn.Parameter(torch.tensor(0.9))
        self.stable_margin = nn.Parameter(torch.tensor(0.0))
        self.binary_bias = nn.Parameter(torch.tensor(0.0))
        self.binary_mix_logit = nn.Parameter(torch.tensor(-1.5))
        self.multiclass_mix_logit = nn.Parameter(torch.tensor(-1.5))
        nn.init.zeros_(self.binary_residual[-1].weight)
        nn.init.zeros_(self.binary_residual[-1].bias)
        nn.init.zeros_(self.multiclass_residual[-1].weight)
        nn.init.zeros_(self.multiclass_residual[-1].bias)

    def forward(
        self,
        *,
        features: torch.Tensor,
        start_score: torch.Tensor,
        start_category: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        start_categories = start_category.clamp_min(0)
        start_repr = self.start_category_embedding(start_categories)
        start_score_repr = self.start_score_encoder(start_score.unsqueeze(-1))
        bridge_input = torch.cat([features, start_repr, start_score_repr], dim=-1)
        predicted_delta = torch.tanh(self.delta_head(bridge_input)) * self.score_max
        predicted_end_score = (start_score.unsqueeze(-1) + predicted_delta).clamp_min(0.0).clamp_max(self.score_max)
        normalized_delta = predicted_delta / max(self.score_max, 1.0)
        residual_input = torch.cat([bridge_input, normalized_delta], dim=-1)
        margin = F.softplus(self.stable_margin) + 0.25
        core_binary = self.delta_scale * predicted_delta + self.binary_bias
        binary_logits = core_binary + 0.15 * self.binary_residual(residual_input)
        core_multiclass = torch.cat(
            [
                -(predicted_delta + margin),
                margin - predicted_delta.abs(),
                predicted_delta - margin,
            ],
            dim=-1,
        )
        multiclass_logits = core_multiclass + 0.15 * self.multiclass_residual(residual_input)
        return {
            "predicted_delta": predicted_delta,
            "predicted_end_score": predicted_end_score,
            "binary_logits": binary_logits,
            "multiclass_logits": multiclass_logits,
            "binary_mix": torch.sigmoid(self.binary_mix_logit).view(1, 1),
            "multiclass_mix": torch.sigmoid(self.multiclass_mix_logit).view(1, 1),
        }


class PsycheNativeAdapter(nn.Module):
    def __init__(
        self,
        *,
        feature_dim: int,
        native_dim: int,
        start_context_dim: int,
        hidden_dim: int,
        dropout: float,
        activation: str,
        task_specific: bool = False,
        task_indices: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.native_encoder = nn.Sequential(
            nn.LayerNorm(native_dim),
            nn.Linear(native_dim, hidden_dim),
            _activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        fusion_dim = feature_dim + hidden_dim + start_context_dim
        self.task_specific = bool(task_specific) and bool(task_indices)
        self.delta_net = None
        self.gate_net = None
        self.task_delta_nets = nn.ModuleDict()
        self.task_gate_nets = nn.ModuleDict()
        self.task_output_scales = nn.ParameterDict()
        if self.task_specific:
            for task_index in sorted({int(task_id) for task_id in (task_indices or [])}):
                task_id_str = str(int(task_index))
                self.task_delta_nets[task_id_str] = nn.Sequential(
                    nn.Linear(fusion_dim, hidden_dim),
                    _activation(activation),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, feature_dim),
                )
                nn.init.zeros_(self.task_delta_nets[task_id_str][-1].weight)
                nn.init.zeros_(self.task_delta_nets[task_id_str][-1].bias)
                self.task_gate_nets[task_id_str] = nn.Sequential(
                    nn.Linear(fusion_dim, hidden_dim),
                    _activation(activation),
                    nn.Linear(hidden_dim, feature_dim),
                )
                self.task_output_scales[task_id_str] = nn.Parameter(torch.tensor(-1.25))
        else:
            self.delta_net = nn.Sequential(
                nn.Linear(fusion_dim, hidden_dim),
                _activation(activation),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, feature_dim),
            )
            nn.init.zeros_(self.delta_net[-1].weight)
            nn.init.zeros_(self.delta_net[-1].bias)
            self.gate_net = nn.Sequential(
                nn.Linear(fusion_dim, hidden_dim),
                _activation(activation),
                nn.Linear(hidden_dim, feature_dim),
            )
        self.output_scale = nn.Parameter(torch.tensor(-1.25))

    def forward(
        self,
        *,
        features: torch.Tensor,
        native_features: torch.Tensor,
        start_context: torch.Tensor | None = None,
        task_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        native_context = self.native_encoder(native_features)
        if start_context is None:
            fusion_input = torch.cat([features, native_context], dim=-1)
        else:
            fusion_input = torch.cat([features, native_context, start_context], dim=-1)
        if self.task_specific:
            if task_index is None:
                raise ValueError("task_index is required when task_specific=True for PsycheNativeAdapter")
            refined = features.clone()
            for task_id_str, delta_net in self.task_delta_nets.items():
                task_mask = task_index == int(task_id_str)
                if not torch.any(task_mask):
                    continue
                delta = torch.tanh(delta_net(fusion_input[task_mask]))
                gate = torch.sigmoid(self.task_gate_nets[task_id_str](fusion_input[task_mask]))
                scale = torch.sigmoid(self.task_output_scales[task_id_str])
                refined[task_mask] = features[task_mask] + scale * gate * delta
            return refined
        delta = torch.tanh(self.delta_net(fusion_input))
        gate = torch.sigmoid(self.gate_net(fusion_input))
        return features + torch.sigmoid(self.output_scale) * gate * delta


@dataclass
class TaskHeadSpecV2:
    task_index: int
    label_type: str
    num_classes: int | None
    output_dim: int


class MCTRCMV2(nn.Module):
    def __init__(
        self,
        modality_input_dims: dict[str, int],
        num_datasets: int,
        task_head_specs: list[TaskHeadSpecV2],
        *,
        native_input_dim: int = 0,
        temporal_slice_dims: dict[str, int] | None = None,
        temporal_num_slices: int = 0,
        temporal_frontend_mode: str = "none",
        enable_missingness_tokens: bool = False,
        enable_missingness_embedding: bool = False,
        missing_signal_dim: int = 0,
        native_missing_signal_dim: int = 0,
        psyche_native_input_dim: int = 0,
        psyche_dataset_index: int | None = None,
        psyche_aux_num_classes: int = 5,
        use_psyche_two_stage: bool = False,
        psyche_change_task_indices: list[int] | None = None,
        psyche_binary_correction_task_indices: list[int] | None = None,
        use_psyche_native_adapter: bool = False,
        use_psyche_taskwise_native_adapter: bool = False,
        psyche_native_task_indices: list[int] | None = None,
        deprest_dataset_index: int | None = None,
        deprest_comm_input_dim: int = 0,
        deprest_comm_group_indices: dict[str, list[int]] | None = None,
        use_deprest_global_adapter: bool = False,
        use_deprest_category_adapter: bool = False,
        deprest_category_task_indices: list[int] | None = None,
        use_deprest_coverage_routing: bool = False,
        deprest_coverage_task_indices: list[int] | None = None,
        use_deprest_concept_compatibility: bool = False,
        deprest_concept_compat_task_indices: list[int] | None = None,
        deprest_concept_compat_feature_indices: list[int] | None = None,
        use_deprest_concept_contrast: bool = False,
        deprest_concept_contrast_task_indices: list[int] | None = None,
        deprest_concept_contrast_feature_indices: list[int] | None = None,
        deprest_concept_contrast_pair_indices: list[int] | None = None,
        deprest_task_bridge_specs: dict[int, dict[str, object]] | None = None,
        deprest_severity_specs: dict[int, dict[str, object]] | None = None,
        deprest_pair_specs: dict[int, dict[str, object]] | None = None,
        deprest_construct_bridge_specs: dict[str, dict[str, object]] | None = None,
        use_concept_residual: bool = False,
        shared_token_refiner_type: str = "none",
        shared_token_refiner_steps: int = 0,
        shared_token_refiner_nograd_steps: int = 0,
        task_local_trm_mode: str = "none",
        task_local_trm_reasoning_dim: int | None = None,
        task_local_trm_h_cycles: int = 0,
        task_local_trm_l_cycles: int = 0,
        task_local_trm_task_indices: list[int] | None = None,
        use_psyche_delta_bridge: bool = False,
        psyche_delta_bridge_task_indices: list[int] | None = None,
        use_deprest_edge_specialist: bool = False,
        deprest_edge_task_indices: list[int] | None = None,
        use_deprest_edge_ovr: bool = False,
        deprest_edge_ovr_task_indices: list[int] | None = None,
        enable_tree_gated_head: bool = False,
        tree_head_task_indices: list[int] | None = None,
        tree_head_depth: int = 2,
        use_ordered_threshold_ordinal_head: bool = False,
        ordinal_aux_class_task_indices: list[int] | None = None,
        token_dim: int = 64,
        encoder_hidden_dim: int = 96,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        transformer_ff_dim: int = 128,
        latent_dim: int = 96,
        dataset_embedding_dim: int = 16,
        task_embedding_dim: int = 16,
        disable_task_conditioning: bool = False,
        conditioning_mode: str = "dataset_task",
        film_conditioning_mode: str = "both",
        concept_dim: int = 8,
        concept_hidden_dim: int = 32,
        task_feature_dim: int | None = None,
        output_refine_dim: int = 16,
        recursion_steps: int = 4,
        modality_dropout: float = 0.0,
        missingness_embedding_norm: bool = False,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.latent_dim = latent_dim
        self.concept_dim = concept_dim
        self.use_concept_bottleneck = bool(concept_dim > 0)
        self.concept_keys = concept_keys_for_dim(concept_dim)
        self.task_feature_dim = int(task_feature_dim or concept_hidden_dim)
        concept_hidden_dim = self.task_feature_dim
        self.recursion_steps = recursion_steps
        self.output_refine_dim = output_refine_dim
        self.native_input_dim = native_input_dim
        self.temporal_slice_dims = {key: int(value) for key, value in (temporal_slice_dims or {}).items()}
        self.temporal_num_slices = max(int(temporal_num_slices), 0)
        self.temporal_frontend_mode = str(temporal_frontend_mode or "none").lower()
        self.enable_missingness_tokens = bool(enable_missingness_tokens)
        self.enable_missingness_embedding = bool(enable_missingness_embedding)
        self.modality_dropout = float(max(0.0, min(1.0, modality_dropout)))
        self.missingness_embedding_norm = bool(missingness_embedding_norm)
        self.missing_signal_dim = max(int(missing_signal_dim), 0)
        self.native_missing_signal_dim = max(int(native_missing_signal_dim), 0)
        self.psyche_native_input_dim = psyche_native_input_dim
        self.psyche_dataset_index = psyche_dataset_index
        self.use_psyche_two_stage = use_psyche_two_stage
        self.psyche_change_task_indices = {int(task_index) for task_index in (psyche_change_task_indices or [])}
        self.psyche_binary_correction_task_indices = {
            int(task_index) for task_index in (psyche_binary_correction_task_indices or [])
        }
        self.use_psyche_native_adapter = use_psyche_native_adapter
        self.use_psyche_taskwise_native_adapter = use_psyche_taskwise_native_adapter
        self.psyche_native_task_indices = {int(task_index) for task_index in (psyche_native_task_indices or [])}
        self.deprest_dataset_index = deprest_dataset_index
        self.deprest_comm_group_indices = deprest_comm_group_indices or {}
        self.use_deprest_global_adapter = use_deprest_global_adapter
        self.use_deprest_category_adapter = use_deprest_category_adapter
        self.deprest_category_task_indices = {int(task_index) for task_index in (deprest_category_task_indices or [])}
        self.use_deprest_coverage_routing = use_deprest_coverage_routing
        self.deprest_coverage_task_indices = {int(task_index) for task_index in (deprest_coverage_task_indices or [])}
        self.use_deprest_concept_compatibility = use_deprest_concept_compatibility
        self.deprest_concept_compat_task_indices = {
            int(task_index) for task_index in (deprest_concept_compat_task_indices or [])
        }
        self.deprest_concept_compat_feature_indices = [
            int(index) for index in (deprest_concept_compat_feature_indices or [])
        ]
        self.use_deprest_concept_contrast = use_deprest_concept_contrast
        self.deprest_concept_contrast_task_indices = {
            int(task_index) for task_index in (deprest_concept_contrast_task_indices or [])
        }
        self.deprest_concept_contrast_feature_indices = [
            int(index) for index in (deprest_concept_contrast_feature_indices or [])
        ]
        self.deprest_concept_contrast_pair_indices = [
            int(index) for index in (deprest_concept_contrast_pair_indices or [])
        ]
        self.deprest_task_bridge_specs = deprest_task_bridge_specs or {}
        self.deprest_severity_specs = deprest_severity_specs or {}
        self.deprest_pair_specs = deprest_pair_specs or {}
        self.deprest_construct_bridge_specs = deprest_construct_bridge_specs or {}
        self.use_concept_residual = use_concept_residual
        self.shared_token_refiner_type = str(shared_token_refiner_type or "none").lower()
        self.shared_token_refiner_steps = max(int(shared_token_refiner_steps), 0)
        self.shared_token_refiner_nograd_steps = max(int(shared_token_refiner_nograd_steps), 0)
        self.task_local_trm_mode = str(task_local_trm_mode or "none").lower()
        self.task_local_trm_reasoning_dim = int(task_local_trm_reasoning_dim or token_dim)
        self.task_local_trm_h_cycles = max(int(task_local_trm_h_cycles), 0)
        self.task_local_trm_l_cycles = max(int(task_local_trm_l_cycles), 0)
        self.task_local_trm_task_indices = {int(task_index) for task_index in (task_local_trm_task_indices or [])}
        self.use_psyche_delta_bridge = use_psyche_delta_bridge
        self.psyche_delta_bridge_task_indices = {
            int(task_index) for task_index in (psyche_delta_bridge_task_indices or [])
        }
        self.use_deprest_edge_specialist = use_deprest_edge_specialist
        self.deprest_edge_task_indices = {int(task_index) for task_index in (deprest_edge_task_indices or [])}
        self.use_deprest_edge_ovr = use_deprest_edge_ovr
        self.deprest_edge_ovr_task_indices = {int(task_index) for task_index in (deprest_edge_ovr_task_indices or [])}
        self.enable_tree_gated_head = bool(enable_tree_gated_head)
        self.tree_head_task_indices = {int(task_index) for task_index in (tree_head_task_indices or [])}
        self.tree_head_depth = max(int(tree_head_depth), 1)
        self.use_ordered_threshold_ordinal_head = bool(use_ordered_threshold_ordinal_head)
        self.ordinal_aux_class_task_indices = {int(task_index) for task_index in (ordinal_aux_class_task_indices or [])}
        self.task_specs = {spec.task_index: spec for spec in task_head_specs}
        self.communication_token_index = CANONICAL_MODALITIES.index("communication")
        self.total_token_count = len(CANONICAL_MODALITIES) + (1 if native_input_dim > 0 else 0)

        self.modality_encoders = nn.ModuleDict(
            {
                modality: ModalityTokenEncoder(
                    input_dim=modality_input_dims[modality],
                    hidden_dim=encoder_hidden_dim,
                    token_dim=token_dim,
                    dropout=dropout,
                    activation=activation,
                )
                for modality in CANONICAL_MODALITIES
            }
        )
        self.modality_embeddings = nn.Parameter(torch.randn(len(CANONICAL_MODALITIES), token_dim) * 0.02)
        self.temporal_frontends = nn.ModuleDict(
            {
                modality: TemporalFrontEnd(
                    input_dim=int(self.temporal_slice_dims.get(modality, 0)),
                    token_dim=token_dim,
                    num_slices=self.temporal_num_slices,
                    hidden_dim=max(token_dim, encoder_hidden_dim),
                    dropout=dropout,
                    activation=activation,
                    mode=self.temporal_frontend_mode,
                )
                for modality in CANONICAL_MODALITIES
            }
        )
        self.temporal_scales = nn.ParameterDict(
            {
                modality: nn.Parameter(torch.tensor(-1.25))
                for modality in CANONICAL_MODALITIES
                if self.temporal_frontends[modality].enabled()
            }
        )
        self.modality_missing_tokens = nn.ParameterDict(
            {
                modality: nn.Parameter(torch.randn(token_dim) * 0.02)
                for modality in CANONICAL_MODALITIES
            }
            if self.enable_missingness_tokens
            else {}
        )
        self.modality_missing_projections = nn.ModuleDict(
            {
                modality: nn.Sequential(
                    nn.Linear(self.missing_signal_dim, max(token_dim // 2, 8)),
                    _activation(activation),
                    nn.Linear(max(token_dim // 2, 8), token_dim),
                )
                for modality in CANONICAL_MODALITIES
                if self.enable_missingness_embedding and self.missing_signal_dim > 0
            }
        )
        self.modality_missing_norms = nn.ModuleDict(
            {
                modality: nn.LayerNorm(self.missing_signal_dim)
                for modality in CANONICAL_MODALITIES
                if self.enable_missingness_embedding
                and self.missingness_embedding_norm
                and self.missing_signal_dim > 0
            }
        )
        self.modality_missing_gate_logits = nn.ParameterDict(
            {
                modality: nn.Parameter(torch.tensor(-0.25))
                for modality in CANONICAL_MODALITIES
                if self.enable_missingness_tokens
            }
        )
        self.native_encoder = (
            ModalityTokenEncoder(
                input_dim=native_input_dim,
                hidden_dim=encoder_hidden_dim,
                token_dim=token_dim,
                dropout=dropout,
                activation=activation,
            )
            if native_input_dim > 0
            else None
        )
        self.native_embedding = nn.Parameter(torch.randn(token_dim) * 0.02) if native_input_dim > 0 else None
        self.native_temporal_scale = None
        self.native_missing_token = (
            nn.Parameter(torch.randn(token_dim) * 0.02)
            if native_input_dim > 0 and self.enable_missingness_tokens
            else None
        )
        self.native_missing_projection = (
            nn.Sequential(
                nn.Linear(self.native_missing_signal_dim, max(token_dim // 2, 8)),
                _activation(activation),
                nn.Linear(max(token_dim // 2, 8), token_dim),
            )
            if native_input_dim > 0 and self.enable_missingness_embedding and self.native_missing_signal_dim > 0
            else None
        )
        self.native_missing_norm = (
            nn.LayerNorm(self.native_missing_signal_dim)
            if (
                native_input_dim > 0
                and self.enable_missingness_embedding
                and self.missingness_embedding_norm
                and self.native_missing_signal_dim > 0
            )
            else None
        )
        self.native_missing_gate_logit = (
            nn.Parameter(torch.tensor(-0.25))
            if native_input_dim > 0 and self.enable_missingness_tokens
            else None
        )
        self.dataset_embedding = nn.Embedding(num_datasets, dataset_embedding_dim)
        self.task_embedding = nn.Embedding(len(task_head_specs), task_embedding_dim)
        self.disable_task_conditioning = bool(disable_task_conditioning)
        normalized_conditioning_mode = str(conditioning_mode or "dataset_task").lower()
        if self.disable_task_conditioning:
            normalized_conditioning_mode = "none"
        if normalized_conditioning_mode not in {"dataset_task", "dataset_only", "task_only", "none"}:
            raise ValueError(f"Unsupported conditioning_mode: {conditioning_mode}")
        self.conditioning_mode = normalized_conditioning_mode
        normalized_film_mode = str(film_conditioning_mode or "both").lower()
        if normalized_film_mode not in {"both", "pre", "post", "none"}:
            raise ValueError(f"Unsupported film_conditioning_mode: {film_conditioning_mode}")
        self.film_conditioning_mode = normalized_film_mode
        self.use_token_film = normalized_film_mode in {"both", "pre"}
        self.use_block_film = normalized_film_mode in {"both", "post"}
        self.use_latent_film = normalized_film_mode in {"both", "post"}

        conditioning_dim = dataset_embedding_dim + task_embedding_dim
        self.conditioning_dim = conditioning_dim
        self.token_adapter = FiLMAdapter(condition_dim=conditioning_dim, target_dim=token_dim, hidden_dim=token_dim)
        self.transformer_blocks = nn.ModuleList(
            [
                DatasetConditionedTransformerBlock(
                    token_dim=token_dim,
                    num_heads=transformer_heads,
                    feedforward_dim=transformer_ff_dim,
                    dropout=dropout,
                    condition_dim=conditioning_dim,
                    use_film=self.use_block_film,
                )
                for _ in range(transformer_layers)
            ]
        )
        if self.shared_token_refiner_type == "attention":
            self.shared_token_refiner: nn.Module | None = DatasetConditionedTransformerBlock(
                token_dim=token_dim,
                num_heads=transformer_heads,
                feedforward_dim=transformer_ff_dim,
                dropout=dropout,
                condition_dim=conditioning_dim,
                use_film=self.use_block_film,
            )
        elif self.shared_token_refiner_type == "mlp":
            self.shared_token_refiner = DatasetConditionedMLPTokenMixerBlock(
                token_dim=token_dim,
                token_count=self.total_token_count,
                feedforward_dim=transformer_ff_dim,
                dropout=dropout,
                condition_dim=conditioning_dim,
                activation=activation,
                use_film=self.use_block_film,
            )
        elif self.shared_token_refiner_type == "none":
            self.shared_token_refiner = None
        else:
            raise ValueError(f"Unsupported shared_token_refiner_type: {self.shared_token_refiner_type}")
        self.shared_token_refiner_residual_scale = (
            nn.Parameter(torch.tensor(0.05))
            if self.shared_token_refiner is not None and self.shared_token_refiner_steps > 0
            else None
        )
        self.pool_projection = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, latent_dim),
            _activation(activation),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        self.latent_adapter = FiLMAdapter(condition_dim=conditioning_dim, target_dim=latent_dim, hidden_dim=latent_dim)
        self.recursive_cell = nn.GRUCell(
            input_size=latent_dim + concept_dim + output_refine_dim + conditioning_dim,
            hidden_size=latent_dim,
        )
        self.latent_norm = nn.LayerNorm(latent_dim)
        self.concept_head = nn.Linear(latent_dim, concept_dim) if self.use_concept_bottleneck else None
        self.concept_post = (
            nn.Sequential(
                nn.LayerNorm(concept_dim),
                nn.Linear(concept_dim, concept_hidden_dim),
                _activation(activation),
                nn.Dropout(dropout),
                nn.LayerNorm(concept_hidden_dim),
            )
            if self.use_concept_bottleneck
            else None
        )
        self.latent_task_adapter = (
            nn.Sequential(
                nn.LayerNorm(latent_dim + conditioning_dim),
                nn.Linear(latent_dim + conditioning_dim, concept_hidden_dim),
                _activation(activation),
                nn.Dropout(dropout),
                nn.LayerNorm(concept_hidden_dim),
            )
            if not self.use_concept_bottleneck
            else None
        )
        self.concept_residual = (
            nn.Sequential(
                nn.Linear(latent_dim + conditioning_dim, concept_hidden_dim),
                _activation(activation),
                nn.Dropout(dropout),
                nn.Linear(concept_hidden_dim, concept_hidden_dim),
            )
            if use_concept_residual
            else None
        )
        self.concept_residual_gate = (
            nn.Sequential(
                nn.Linear(latent_dim + conditioning_dim, concept_hidden_dim),
                nn.Sigmoid(),
            )
            if use_concept_residual
            else None
        )
        self.task_heads = nn.ModuleDict(
            {
                str(spec.task_index): (
                    SoftTreeHead(
                        feature_dim=concept_hidden_dim,
                        output_dim=spec.output_dim,
                        depth=self.tree_head_depth,
                        hidden_dim=max(16, concept_hidden_dim),
                        dropout=dropout,
                        activation=activation,
                    )
                    if self.enable_tree_gated_head and spec.task_index in self.tree_head_task_indices
                    else OrderedThresholdOrdinalHead(
                        feature_dim=concept_hidden_dim,
                        num_classes=int(spec.num_classes or 0),
                        hidden_dim=max(16, concept_hidden_dim),
                        dropout=dropout,
                        activation=activation,
                    )
                    if self.use_ordered_threshold_ordinal_head and spec.label_type == "ordinal"
                    else nn.Sequential(
                        nn.Linear(concept_hidden_dim, concept_hidden_dim),
                        _activation(activation),
                        nn.Dropout(dropout),
                        nn.Linear(concept_hidden_dim, spec.output_dim),
                    )
                )
                for spec in task_head_specs
            }
        )
        self.ordinal_aux_class_heads = nn.ModuleDict(
            {
                str(spec.task_index): nn.Sequential(
                    nn.Linear(concept_hidden_dim, concept_hidden_dim),
                    _activation(activation),
                    nn.Dropout(dropout),
                    nn.Linear(concept_hidden_dim, int(spec.num_classes or spec.output_dim)),
                )
                for spec in task_head_specs
                if spec.label_type == "ordinal"
                and int(spec.task_index) in self.ordinal_aux_class_task_indices
                and int(spec.num_classes or 0) > 1
            }
        )
        self.deprest_severity_score_heads = nn.ModuleDict()
        self.deprest_severity_residual_heads = nn.ModuleDict()
        self.deprest_severity_log_scales = nn.ParameterDict()
        self.deprest_severity_buffer_names: dict[str, str] = {}
        self.deprest_severity_score_max: dict[str, float] = {}
        for task_index, severity_spec in self.deprest_severity_specs.items():
            task_id_str = str(int(task_index))
            thresholds = [float(value) for value in severity_spec.get("thresholds", [])]
            if not thresholds:
                continue
            score_max = float(severity_spec.get("score_max", 27.0))
            self.deprest_severity_score_heads[task_id_str] = nn.Sequential(
                nn.Linear(concept_hidden_dim, concept_hidden_dim),
                _activation(activation),
                nn.Dropout(dropout),
                nn.Linear(concept_hidden_dim, 1),
            )
            residual_head = nn.Sequential(
                nn.Linear(concept_hidden_dim, max(8, concept_hidden_dim // 2)),
                _activation(activation),
                nn.Dropout(dropout),
                nn.Linear(max(8, concept_hidden_dim // 2), len(thresholds)),
            )
            nn.init.zeros_(residual_head[-1].weight)
            nn.init.zeros_(residual_head[-1].bias)
            self.deprest_severity_residual_heads[task_id_str] = residual_head
            self.deprest_severity_log_scales[task_id_str] = nn.Parameter(torch.tensor(0.0))
            buffer_name = f"_deprest_severity_thresholds_{task_id_str}"
            self.register_buffer(buffer_name, torch.tensor(thresholds, dtype=torch.float32))
            self.deprest_severity_buffer_names[task_id_str] = buffer_name
            self.deprest_severity_score_max[task_id_str] = score_max
        self.deprest_pair_heads = nn.ModuleDict()
        for task_index, pair_spec in self.deprest_pair_specs.items():
            task_id_str = str(int(task_index))
            output_dim = int(pair_spec.get("output_dim", 1))
            self.deprest_pair_heads[task_id_str] = nn.Sequential(
                nn.Linear(concept_hidden_dim, concept_hidden_dim),
                _activation(activation),
                nn.Dropout(dropout),
                nn.Linear(concept_hidden_dim, output_dim),
            )
        self.deprest_construct_bridges = nn.ModuleDict()
        self.deprest_construct_task_to_key: dict[int, str] = {}
        self.deprest_construct_categorical_task_indices: set[int] = set()
        for construct_key, bridge_spec in sorted(self.deprest_construct_bridge_specs.items()):
            thresholds = [float(value) for value in bridge_spec.get("thresholds", [])]
            if not thresholds:
                continue
            self.deprest_construct_bridges[str(construct_key)] = ConstructSeverityBridge(
                feature_dim=concept_hidden_dim,
                score_max=float(bridge_spec.get("score_max", 27.0)),
                thresholds=thresholds,
                hidden_dim=max(16, concept_hidden_dim),
                dropout=dropout,
                activation=activation,
            )
            for task_index in bridge_spec.get("task_indices", []):
                self.deprest_construct_task_to_key[int(task_index)] = str(construct_key)
            self.deprest_construct_categorical_task_indices.update(
                int(task_index) for task_index in bridge_spec.get("categorical_task_indices", [])
            )
        call_dim = len(self.deprest_comm_group_indices.get("call", []))
        text_dim = len(self.deprest_comm_group_indices.get("text", []))
        meta_dim = len(self.deprest_comm_group_indices.get("meta", []))
        self.deprest_call_encoder = (
            nn.Sequential(
                nn.LayerNorm(call_dim),
                nn.Linear(call_dim, max(32, token_dim // 2)),
                _activation(activation),
                nn.Dropout(dropout),
                nn.Linear(max(32, token_dim // 2), token_dim),
            )
            if deprest_dataset_index is not None and deprest_comm_input_dim > 0 and call_dim > 0
            else None
        )
        self.deprest_text_encoder = (
            nn.Sequential(
                nn.LayerNorm(text_dim),
                nn.Linear(text_dim, max(32, token_dim // 2)),
                _activation(activation),
                nn.Dropout(dropout),
                nn.Linear(max(32, token_dim // 2), token_dim),
            )
            if deprest_dataset_index is not None and deprest_comm_input_dim > 0 and text_dim > 0
            else None
        )
        self.deprest_meta_encoder = (
            nn.Sequential(
                nn.LayerNorm(meta_dim),
                nn.Linear(meta_dim, max(24, token_dim // 2)),
                _activation(activation),
                nn.Dropout(dropout),
                nn.Linear(max(24, token_dim // 2), token_dim),
            )
            if deprest_dataset_index is not None and deprest_comm_input_dim > 0 and meta_dim > 0
            else None
        )
        self.deprest_branch_gate = (
            nn.Sequential(
                nn.LayerNorm(meta_dim),
                nn.Linear(meta_dim, max(24, token_dim // 2)),
                _activation(activation),
                nn.Linear(max(24, token_dim // 2), 4),
            )
            if deprest_dataset_index is not None and deprest_comm_input_dim > 0 and meta_dim > 0
            else None
        )
        self.deprest_adapter = (
            nn.Sequential(
                nn.Linear(token_dim * 2 + conditioning_dim, latent_dim),
                _activation(activation),
                nn.Dropout(dropout),
                nn.Linear(latent_dim, latent_dim),
            )
            if (
                use_deprest_global_adapter
                and deprest_dataset_index is not None
                and deprest_comm_input_dim > 0
            )
            else None
        )
        self.deprest_category_adapter = (
            nn.Sequential(
                nn.Linear(token_dim * 2 + conditioning_dim, concept_hidden_dim),
                _activation(activation),
                nn.Dropout(dropout),
                nn.Linear(concept_hidden_dim, concept_hidden_dim),
            )
            if (
                use_deprest_category_adapter
                and deprest_dataset_index is not None
                and deprest_comm_input_dim > 0
                and deprest_category_task_indices
            )
            else None
        )
        self.deprest_edge_context_dim = token_dim * 2 + conditioning_dim + 1
        self.deprest_routing_context_dim = self.deprest_edge_context_dim + self.total_token_count
        self.deprest_coverage_routing_modules = nn.ModuleDict()
        if (
            use_deprest_coverage_routing
            and deprest_dataset_index is not None
            and deprest_comm_input_dim > 0
            and self.deprest_coverage_task_indices
        ):
            routing_hidden_dim = max(16, concept_hidden_dim)
            for task_index in sorted(self.deprest_coverage_task_indices):
                task_id_str = str(int(task_index))
                self.deprest_coverage_routing_modules[task_id_str] = TaskConditionedRoutingBlock(
                    feature_dim=concept_hidden_dim,
                    context_dim=self.deprest_routing_context_dim,
                    hidden_dim=routing_hidden_dim,
                    num_experts=3,
                    dropout=dropout,
                    activation=activation,
                )
        self.deprest_concept_compat_modules = nn.ModuleDict()
        self.deprest_concept_compat_context_dim = len(self.deprest_concept_compat_feature_indices) + self.total_token_count
        if (
            use_deprest_concept_compatibility
            and self.use_concept_bottleneck
            and deprest_dataset_index is not None
            and deprest_comm_input_dim > 0
            and self.deprest_concept_compat_task_indices
            and self.deprest_concept_compat_feature_indices
        ):
            compat_hidden_dim = max(12, concept_hidden_dim // 2)
            for task_index in sorted(self.deprest_concept_compat_task_indices):
                task_id_str = str(int(task_index))
                self.deprest_concept_compat_modules[task_id_str] = TaskConditionedConceptCompatibility(
                    concept_dim=concept_dim,
                    context_dim=self.deprest_concept_compat_context_dim,
                    hidden_dim=compat_hidden_dim,
                    dropout=dropout,
                    activation=activation,
                )
        self.deprest_concept_contrast_modules = nn.ModuleDict()
        self.deprest_concept_contrast_context_dim = len(self.deprest_concept_contrast_feature_indices) + self.total_token_count
        if (
            use_deprest_concept_contrast
            and self.use_concept_bottleneck
            and deprest_dataset_index is not None
            and deprest_comm_input_dim > 0
            and self.deprest_concept_contrast_task_indices
            and self.deprest_concept_contrast_feature_indices
            and len(self.deprest_concept_contrast_pair_indices) == 2
            and max(self.deprest_concept_contrast_pair_indices) < self.concept_dim
        ):
            contrast_hidden_dim = max(8, concept_hidden_dim // 3)
            for task_index in sorted(self.deprest_concept_contrast_task_indices):
                task_id_str = str(int(task_index))
                self.deprest_concept_contrast_modules[task_id_str] = TaskConditionedConceptContrastCompatibility(
                    context_dim=self.deprest_concept_contrast_context_dim,
                    pair_indices=self.deprest_concept_contrast_pair_indices,
                    hidden_dim=contrast_hidden_dim,
                    dropout=dropout,
                    activation=activation,
                )
        self.deprest_task_bridge_modules = nn.ModuleDict()
        for task_index, bridge_spec in self.deprest_task_bridge_specs.items():
            task_id_str = str(int(task_index))
            source_task_index = int(bridge_spec.get("source_task_index", -1))
            source_spec = self.task_specs.get(source_task_index)
            if source_spec is None:
                continue
            self.deprest_task_bridge_modules[task_id_str] = nn.Sequential(
                nn.Linear(source_spec.output_dim, max(8, concept_hidden_dim // 2)),
                _activation(activation),
                nn.Dropout(dropout),
                nn.Linear(max(8, concept_hidden_dim // 2), concept_hidden_dim),
            )
        self.deprest_edge_specialist_heads = nn.ModuleDict()
        if (
            use_deprest_edge_specialist
            and deprest_dataset_index is not None
            and self.deprest_edge_task_indices
        ):
            edge_hidden_dim = max(16, concept_hidden_dim)
            for task_index in sorted(self.deprest_edge_task_indices):
                task_id_str = str(int(task_index))
                task_spec = self.task_specs.get(int(task_index))
                if task_spec is None or task_spec.label_type != "ordinal" or int(task_spec.output_dim) < 2:
                    continue
                edge_head = nn.Sequential(
                    nn.Linear(concept_hidden_dim + self.deprest_edge_context_dim, edge_hidden_dim),
                    _activation(activation),
                    nn.Dropout(dropout),
                    nn.Linear(edge_hidden_dim, 4),
                )
                nn.init.zeros_(edge_head[-1].weight)
                nn.init.zeros_(edge_head[-1].bias)
                self.deprest_edge_specialist_heads[task_id_str] = edge_head
        self.deprest_edge_ovr_heads = nn.ModuleDict()
        self.deprest_edge_ovr_mix_logits = nn.ParameterDict()
        if (
            use_deprest_edge_ovr
            and deprest_dataset_index is not None
            and self.deprest_edge_ovr_task_indices
        ):
            edge_hidden_dim = max(16, concept_hidden_dim)
            for task_index in sorted(self.deprest_edge_ovr_task_indices):
                task_id_str = str(int(task_index))
                task_spec = self.task_specs.get(int(task_index))
                if task_spec is None or task_spec.label_type != "ordinal" or int(task_spec.output_dim) < 2:
                    continue
                edge_head = nn.Sequential(
                    nn.Linear(concept_hidden_dim + self.deprest_edge_context_dim + task_spec.output_dim, edge_hidden_dim),
                    _activation(activation),
                    nn.Dropout(dropout),
                    nn.Linear(edge_hidden_dim, 2),
                )
                nn.init.zeros_(edge_head[-1].weight)
                nn.init.constant_(edge_head[-1].bias, -1.5)
                self.deprest_edge_ovr_heads[task_id_str] = edge_head
                self.deprest_edge_ovr_mix_logits[task_id_str] = nn.Parameter(torch.tensor(-0.75))
        self.psyche_end_head = (
            nn.Sequential(
                nn.Linear(concept_hidden_dim, concept_hidden_dim),
                _activation(activation),
                nn.Dropout(dropout),
                nn.Linear(concept_hidden_dim, psyche_aux_num_classes),
            )
            if psyche_dataset_index is not None
            else None
        )
        self.psyche_end_adapter = (
            nn.Sequential(
                nn.Linear(psyche_aux_num_classes, concept_hidden_dim),
                _activation(activation),
                nn.Linear(concept_hidden_dim, concept_hidden_dim),
            )
            if psyche_dataset_index is not None
            else None
        )
        self.psyche_start_embedding = (
            nn.Embedding(psyche_aux_num_classes, concept_hidden_dim)
            if psyche_dataset_index is not None and use_psyche_two_stage
            else None
        )
        self.psyche_start_score_encoder = (
            nn.Sequential(
                nn.Linear(1, max(8, concept_hidden_dim // 2)),
                _activation(activation),
                nn.Linear(max(8, concept_hidden_dim // 2), concept_hidden_dim),
            )
            if psyche_dataset_index is not None and use_psyche_two_stage
            else None
        )
        self.psyche_two_stage_adapter = (
            nn.Sequential(
                nn.Linear(concept_hidden_dim * 3 + psyche_aux_num_classes, concept_hidden_dim),
                _activation(activation),
                nn.Dropout(dropout),
                nn.Linear(concept_hidden_dim, concept_hidden_dim),
            )
            if psyche_dataset_index is not None and use_psyche_two_stage
            else None
        )
        self.psyche_binary_correction_input_dim = 1 + psyche_aux_num_classes + concept_hidden_dim + concept_hidden_dim
        self.psyche_binary_correction_heads = nn.ModuleDict()
        if (
            psyche_dataset_index is not None
            and use_psyche_two_stage
            and self.psyche_binary_correction_task_indices
        ):
            correction_hidden_dim = max(16, concept_hidden_dim // 2)
            for task_index in sorted(self.psyche_binary_correction_task_indices):
                task_id_str = str(int(task_index))
                correction_head = nn.Sequential(
                    nn.Linear(self.psyche_binary_correction_input_dim, correction_hidden_dim),
                    _activation(activation),
                    nn.Dropout(dropout),
                    nn.Linear(correction_hidden_dim, 2),
                )
                nn.init.zeros_(correction_head[-1].weight)
                nn.init.zeros_(correction_head[-1].bias)
                self.psyche_binary_correction_heads[task_id_str] = correction_head
        self.output_refiners = nn.ModuleDict(
            {
                str(spec.task_index): nn.Sequential(
                    nn.Linear(spec.output_dim, output_refine_dim),
                    _activation(activation),
                    nn.Linear(output_refine_dim, output_refine_dim),
                )
                for spec in task_head_specs
            }
        )
        self.task_local_trm_refiner = TaskLocalTRMRefiner(
            mode=self.task_local_trm_mode,
            latent_dim=latent_dim,
            concept_dim=concept_dim,
            concept_hidden_dim=concept_hidden_dim,
            output_refine_dim=output_refine_dim,
            conditioning_dim=conditioning_dim,
            modality_gate_dim=self.total_token_count,
            deprest_context_dim=self.deprest_edge_context_dim,
            psyche_context_dim=(
                (psyche_aux_num_classes + concept_hidden_dim + concept_hidden_dim)
                if psyche_dataset_index is not None and use_psyche_two_stage
                else 0
            ),
            reasoning_dim=self.task_local_trm_reasoning_dim,
            feedforward_dim=transformer_ff_dim,
            num_heads=transformer_heads,
            h_cycles=self.task_local_trm_h_cycles,
            l_cycles=self.task_local_trm_l_cycles,
            dropout=dropout,
            activation=activation,
        )
        self.psyche_delta_bridge = (
            PsycheDeltaBridge(
                feature_dim=concept_hidden_dim,
                num_start_categories=psyche_aux_num_classes,
                score_max=27.0,
                hidden_dim=max(16, concept_hidden_dim),
                dropout=dropout,
                activation=activation,
            )
            if self.use_psyche_delta_bridge and psyche_dataset_index is not None and self.psyche_delta_bridge_task_indices
            else None
        )
        self.psyche_native_adapter = (
            PsycheNativeAdapter(
                feature_dim=concept_hidden_dim,
                native_dim=psyche_native_input_dim,
                start_context_dim=(concept_hidden_dim * 2) if use_psyche_two_stage else 0,
                hidden_dim=max(32, concept_hidden_dim),
                dropout=dropout,
                activation=activation,
                task_specific=self.use_psyche_taskwise_native_adapter,
                task_indices=sorted(self.psyche_native_task_indices),
            )
            if (
                self.use_psyche_native_adapter
                and psyche_dataset_index is not None
                and psyche_native_input_dim > 0
                and self.psyche_native_task_indices
            )
            else None
        )

    def _select_group(self, features: torch.Tensor, indices: list[int]) -> torch.Tensor | None:
        if not indices:
            return None
        index_tensor = torch.as_tensor(indices, dtype=torch.long, device=features.device)
        return torch.index_select(features, dim=1, index=index_tensor)

    def _resolve_concept_mask(self, raw_mask: torch.Tensor) -> torch.Tensor:
        if self.concept_dim <= 0:
            return raw_mask.new_zeros((raw_mask.size(0), 0))
        if raw_mask.size(1) == self.concept_dim:
            return raw_mask
        if raw_mask.size(1) > self.concept_dim:
            return raw_mask[:, : self.concept_dim]
        padding = raw_mask.new_ones((raw_mask.size(0), self.concept_dim - raw_mask.size(1)))
        return torch.cat([raw_mask, padding], dim=1)

    def _training_modality_mask(self, modality_mask: torch.Tensor) -> torch.Tensor:
        if not self.training or self.modality_dropout <= 0.0:
            return modality_mask
        observed = modality_mask > 0.5
        if not torch.any(observed):
            return modality_mask
        keep_random = torch.rand_like(modality_mask) >= self.modality_dropout
        dropped = modality_mask * keep_random.to(modality_mask.dtype)
        originally_nonempty = observed.any(dim=1)
        now_empty = (dropped > 0.5).sum(dim=1) <= 0
        repair_rows = originally_nonempty & now_empty
        if torch.any(repair_rows):
            dropped = dropped.clone()
            observed_float = observed.to(modality_mask.dtype)
            random_scores = torch.rand_like(modality_mask).masked_fill(~observed, -1.0)
            repair_indices = random_scores.argmax(dim=1)
            dropped[repair_rows] = 0.0
            dropped[repair_rows, repair_indices[repair_rows]] = observed_float[repair_rows, repair_indices[repair_rows]]
        return dropped

    def _project_task_features(
        self,
        latent: torch.Tensor,
        conditioning: torch.Tensor,
        concepts: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_concept_bottleneck and self.concept_post is not None:
            return self.concept_post(concepts)
        if self.latent_task_adapter is None:
            raise RuntimeError("Latent task adapter is unavailable when concept bottleneck is disabled.")
        return self.latent_task_adapter(torch.cat([latent, conditioning], dim=-1))

    def _head_sparsity_penalty(self, head: nn.Module) -> torch.Tensor:
        if hasattr(head, "sparse_penalty"):
            return getattr(head, "sparse_penalty")()
        if isinstance(head, nn.Sequential):
            final_linear = head[-1]
            return torch.sqrt(torch.sum(final_linear.weight ** 2, dim=0) + 1e-8).mean()
        return torch.tensor(0.0, device=self.modality_embeddings.device)

    def _encode_modalities(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = []
        gates = []
        modality_mask = self._training_modality_mask(batch["modality_mask"])
        for modality_index, modality in enumerate(CANONICAL_MODALITIES):
            token, gate = self.modality_encoders[modality](batch[f"{modality}_features"])
            observed = modality_mask[:, modality_index].unsqueeze(-1)
            if modality in self.temporal_scales:
                temporal_token = self.temporal_frontends[modality](
                    batch[f"{modality}_temporal_slices"],
                    batch[f"{modality}_temporal_mask"],
                )
                token = token + torch.sigmoid(self.temporal_scales[modality]) * temporal_token
            if modality in self.modality_missing_projections:
                missing_signal = batch[f"{modality}_missing_signal"]
                if modality in self.modality_missing_norms:
                    missing_signal = self.modality_missing_norms[modality](missing_signal)
                token = token + 0.2 * self.modality_missing_projections[modality](missing_signal)
            if self.enable_missingness_tokens:
                missing_token = self.modality_missing_tokens[modality].unsqueeze(0)
                token = token * observed + missing_token * (1.0 - observed)
                missing_gate = torch.sigmoid(self.modality_missing_gate_logits[modality]).view(1, 1)
                gate = gate * observed + missing_gate * (1.0 - observed)
                token_presence = torch.ones_like(observed)
            else:
                token = token * observed
                gate = gate * observed
                token_presence = observed
            tokens.append(token)
            gates.append(gate)
        token_tensor = torch.stack(tokens, dim=1)
        gate_tensor = torch.cat(gates, dim=-1)
        token_tensor = token_tensor + self.modality_embeddings.unsqueeze(0)
        token_mask = torch.ones_like(modality_mask) if self.enable_missingness_tokens else modality_mask
        if self.native_encoder is not None:
            native_token, native_gate = self.native_encoder(batch["native_features"])
            native_observed = batch["native_mask"].unsqueeze(-1)
            if self.native_missing_projection is not None:
                native_missing_signal = batch["native_missing_signal"]
                if self.native_missing_norm is not None:
                    native_missing_signal = self.native_missing_norm(native_missing_signal)
                native_token = native_token + 0.2 * self.native_missing_projection(native_missing_signal)
            if self.enable_missingness_tokens and self.native_missing_token is not None:
                native_token = native_token * native_observed + self.native_missing_token.unsqueeze(0) * (1.0 - native_observed)
                if self.native_missing_gate_logit is not None:
                    native_gate = native_gate * native_observed + torch.sigmoid(self.native_missing_gate_logit).view(1, 1) * (1.0 - native_observed)
                native_token_mask = torch.ones_like(native_observed)
            else:
                native_token = native_token * native_observed
                native_gate = native_gate * native_observed
                native_token_mask = native_observed
            native_token = native_token + self.native_embedding.unsqueeze(0)
            token_tensor = torch.cat([token_tensor, native_token.unsqueeze(1)], dim=1)
            gate_tensor = torch.cat([gate_tensor, native_gate], dim=-1)
            token_mask = torch.cat([token_mask, native_token_mask], dim=1)
        return token_tensor, gate_tensor, token_mask

    def _pool_tokens(self, tokens: torch.Tensor, modality_mask: torch.Tensor) -> torch.Tensor:
        weights = modality_mask.unsqueeze(-1)
        summed = (tokens * weights).sum(dim=1)
        normalizer = weights.sum(dim=1).clamp_min(1.0)
        return summed / normalizer

    def _apply_task_heads(
        self,
        concept_features: torch.Tensor,
        task_index: torch.Tensor,
        psyche_binary_correction_context: torch.Tensor | None = None,
        psyche_delta_bridge_state: dict[str, torch.Tensor] | None = None,
        deprest_edge_context: torch.Tensor | None = None,
        deprest_routing_context: torch.Tensor | None = None,
    ) -> tuple[
        dict[int, torch.Tensor],
        torch.Tensor,
        dict[int, torch.Tensor],
        dict[int, torch.Tensor],
        dict[int, torch.Tensor],
        dict[int, torch.Tensor],
        dict[int, torch.Tensor],
        dict[int, torch.Tensor],
    ]:
        logits_by_task: dict[int, torch.Tensor] = {}
        paired_logits_by_task: dict[int, torch.Tensor] = {}
        edge_ovr_logits_by_task: dict[int, torch.Tensor] = {}
        edge_ovr_probs_by_task: dict[int, torch.Tensor] = {}
        edge_ovr_mix_scale_by_task: dict[int, torch.Tensor] = {}
        deprest_construct_scores_by_task: dict[int, torch.Tensor] = {}
        ordinal_aux_logits_by_task: dict[int, torch.Tensor] = {}
        refine_summary = concept_features.new_zeros((concept_features.size(0), self.output_refine_dim))
        for task_id_str, head in self.task_heads.items():
            task_id = int(task_id_str)
            mask = task_index == task_id
            if not torch.any(mask):
                continue
            head_features = concept_features[mask]
            if (
                deprest_routing_context is not None
                and task_id_str in self.deprest_coverage_routing_modules
            ):
                routing_context = deprest_routing_context[mask]
                head_features, _, _ = self.deprest_coverage_routing_modules[task_id_str](head_features, routing_context)
            if task_id in self.deprest_task_bridge_specs and task_id_str in self.deprest_task_bridge_modules:
                source_task_index = int(self.deprest_task_bridge_specs[task_id].get("source_task_index", -1))
                source_task_id_str = str(source_task_index)
                if source_task_id_str in self.task_heads:
                    source_logits = self._compute_task_logits(source_task_id_str, head_features).detach()
                    bridge_context = self.deprest_task_bridge_modules[task_id_str](source_logits)
                    head_features = head_features + 0.2 * bridge_context
            logits = self._compute_task_logits(task_id_str, head_features)
            construct_key = self.deprest_construct_task_to_key.get(task_id)
            if construct_key is not None and construct_key in self.deprest_construct_bridges:
                bridge_score, bridge_logits, bridge_mix = self.deprest_construct_bridges[construct_key](head_features)
                deprest_construct_scores_by_task[task_id] = bridge_score
                if (
                    task_id in self.deprest_construct_categorical_task_indices
                    and logits.dim() == 2
                    and bridge_logits.shape == logits.shape
                ):
                    logits = logits + bridge_mix * (bridge_logits - logits)
            if (
                deprest_edge_context is not None
                and task_id_str in self.deprest_edge_specialist_heads
                and logits.dim() == 2
                and logits.size(1) >= 2
            ):
                edge_context = deprest_edge_context[mask]
                edge_input = torch.cat([head_features, edge_context], dim=-1)
                edge_delta, edge_gate = self.deprest_edge_specialist_heads[task_id_str](edge_input).chunk(2, dim=-1)
                edge_adjustment = 0.75 * torch.tanh(edge_delta) * torch.sigmoid(edge_gate)
                logits = logits.clone()
                logits[:, 0] = logits[:, 0] - edge_adjustment[:, 0]
                logits[:, -1] = logits[:, -1] + edge_adjustment[:, 1]
            if (
                psyche_binary_correction_context is not None
                and task_id_str in self.psyche_binary_correction_heads
            ):
                correction_context = psyche_binary_correction_context[mask]
                correction_input = torch.cat([logits, correction_context], dim=-1)
                correction_delta, correction_gate = self.psyche_binary_correction_heads[task_id_str](correction_input).chunk(2, dim=-1)
                logits = logits + 0.75 * torch.tanh(correction_delta) * torch.sigmoid(correction_gate)
            if psyche_delta_bridge_state is not None and task_id in self.psyche_delta_bridge_task_indices:
                task_spec = self.task_specs.get(task_id)
                if task_spec is not None:
                    if task_spec.label_type == "binary":
                        bridge_logits = psyche_delta_bridge_state["binary_logits"][mask]
                        bridge_mix = psyche_delta_bridge_state["binary_mix"]
                    elif task_spec.label_type == "multiclass":
                        bridge_logits = psyche_delta_bridge_state["multiclass_logits"][mask]
                        bridge_mix = psyche_delta_bridge_state["multiclass_mix"]
                    else:
                        bridge_logits = None
                        bridge_mix = None
                    if bridge_logits is not None and bridge_mix is not None and bridge_logits.shape == logits.shape:
                        local_bridge_mask = psyche_delta_bridge_state["mask"][mask]
                        if torch.any(local_bridge_mask):
                            logits = logits.clone()
                            logits[local_bridge_mask] = logits[local_bridge_mask] + bridge_mix * (
                                bridge_logits[local_bridge_mask] - logits[local_bridge_mask]
                            )
            if (
                deprest_edge_context is not None
                and task_id_str in self.deprest_edge_ovr_heads
                and logits.dim() == 2
            ):
                edge_context = deprest_edge_context[mask]
                edge_input = torch.cat([head_features, edge_context, logits], dim=-1)
                edge_logits = self.deprest_edge_ovr_heads[task_id_str](edge_input)
                edge_ovr_logits_by_task[task_id] = edge_logits
                base_probabilities = _ordinal_probabilities_torch(logits)
                mix_scale = torch.sigmoid(self.deprest_edge_ovr_mix_logits[task_id_str]).view(1, 1)
                edge_probabilities = self._edge_ovr_probabilities_torch(
                    base_probabilities,
                    edge_logits,
                    mix_scale,
                )
                edge_ovr_probs_by_task[task_id] = edge_probabilities
                edge_ovr_mix_scale_by_task[task_id] = mix_scale.expand(edge_probabilities.size(0), 1)
            logits_by_task[task_id] = logits
            if task_id_str in self.deprest_pair_heads:
                paired_logits_by_task[task_id] = self.deprest_pair_heads[task_id_str](head_features)
            if task_id_str in self.ordinal_aux_class_heads:
                ordinal_aux_logits_by_task[task_id] = self.ordinal_aux_class_heads[task_id_str](head_features)
            refine_summary[mask] = self.output_refiners[task_id_str](logits)
        return (
            logits_by_task,
            refine_summary,
            paired_logits_by_task,
            edge_ovr_logits_by_task,
            edge_ovr_probs_by_task,
            edge_ovr_mix_scale_by_task,
            deprest_construct_scores_by_task,
            ordinal_aux_logits_by_task,
        )

    def _edge_ovr_probabilities_torch(
        self,
        base_probabilities: torch.Tensor,
        edge_logits: torch.Tensor,
        mix_scale: torch.Tensor,
    ) -> torch.Tensor:
        edge_mass = torch.sigmoid(edge_logits) * mix_scale
        edge_total = edge_mass.sum(dim=1, keepdim=True)
        max_total = torch.full_like(edge_total, 0.95)
        downscale = torch.minimum(torch.ones_like(edge_total), max_total / edge_total.clamp_min(1e-6))
        edge_mass = edge_mass * downscale
        remaining = (1.0 - edge_mass.sum(dim=1, keepdim=True)).clamp_min(1e-6)
        probabilities = base_probabilities * remaining
        probabilities[:, 0] = probabilities[:, 0] + edge_mass[:, 0]
        probabilities[:, -1] = probabilities[:, -1] + edge_mass[:, 1]
        probabilities = probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return probabilities

    def _compute_task_logits(self, task_id_str: str, features: torch.Tensor) -> torch.Tensor:
        if task_id_str in self.deprest_severity_score_heads:
            raw_score = self.deprest_severity_score_heads[task_id_str](features)
            score_max = float(self.deprest_severity_score_max[task_id_str])
            predicted_score = torch.sigmoid(raw_score) * score_max
            thresholds = getattr(self, self.deprest_severity_buffer_names[task_id_str]).view(1, -1)
            scale = F.softplus(self.deprest_severity_log_scales[task_id_str]).view(1, 1) + 0.5
            residual = 0.1 * self.deprest_severity_residual_heads[task_id_str](features)
            return (predicted_score - thresholds) * scale + residual
        return self.task_heads[task_id_str](features)

    def _build_deprest_adapter_inputs(
        self,
        tokens: torch.Tensor,
        batch: dict[str, torch.Tensor],
        conditioning: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[None, None, None]:
        if self.deprest_dataset_index is None or batch["deprest_comm_features"].size(1) == 0:
            return None, None, None
        deprest_mask = (batch["dataset_index"] == int(self.deprest_dataset_index)) & (batch["deprest_comm_mask"] > 0.5)
        if not torch.any(deprest_mask):
            return None, None, None

        extra_features = batch["deprest_comm_features"]
        call_features = self._select_group(extra_features, self.deprest_comm_group_indices.get("call", []))
        text_features = self._select_group(extra_features, self.deprest_comm_group_indices.get("text", []))
        meta_features = self._select_group(extra_features, self.deprest_comm_group_indices.get("meta", []))

        combined_repr = tokens[:, self.communication_token_index, :]
        call_repr = self.deprest_call_encoder(call_features) if self.deprest_call_encoder is not None and call_features is not None else torch.zeros_like(combined_repr)
        text_repr = self.deprest_text_encoder(text_features) if self.deprest_text_encoder is not None and text_features is not None else torch.zeros_like(combined_repr)
        meta_repr = self.deprest_meta_encoder(meta_features) if self.deprest_meta_encoder is not None and meta_features is not None else torch.zeros_like(combined_repr)

        if self.deprest_branch_gate is not None and meta_features is not None:
            gate_output = self.deprest_branch_gate(meta_features)
            branch_weights = torch.softmax(gate_output[:, :3], dim=-1)
            coverage_scale = torch.sigmoid(gate_output[:, 3:4])
        else:
            branch_weights = torch.full((combined_repr.size(0), 3), fill_value=1.0 / 3.0, dtype=combined_repr.dtype, device=combined_repr.device)
            coverage_scale = torch.ones((combined_repr.size(0), 1), dtype=combined_repr.dtype, device=combined_repr.device)

        stacked_repr = torch.stack([combined_repr, call_repr, text_repr], dim=1)
        gated_repr = (stacked_repr * branch_weights.unsqueeze(-1)).sum(dim=1)
        adapter_input = torch.cat([gated_repr, meta_repr, conditioning], dim=-1)
        return deprest_mask, adapter_input, coverage_scale

    def _shared_token_refiner_step(
        self,
        tokens: torch.Tensor,
        input_tokens: torch.Tensor,
        conditioning: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.shared_token_refiner is None or self.shared_token_refiner_residual_scale is None:
            return tokens
        valid_mask = (~padding_mask).unsqueeze(-1).to(tokens.dtype)
        refinement_input = tokens + input_tokens
        updated_tokens = self.shared_token_refiner(refinement_input, conditioning, padding_mask)
        delta = (updated_tokens - refinement_input) * valid_mask
        residual_scale = torch.tanh(self.shared_token_refiner_residual_scale)
        return tokens + residual_scale * delta

    def _run_shared_token_refiner(
        self,
        tokens: torch.Tensor,
        conditioning: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.shared_token_refiner is None or self.shared_token_refiner_steps <= 0:
            return tokens
        input_tokens = tokens
        refined_tokens = tokens
        no_grad_steps = min(self.shared_token_refiner_nograd_steps, max(self.shared_token_refiner_steps - 1, 0))
        if no_grad_steps > 0:
            with torch.no_grad():
                for _ in range(no_grad_steps):
                    refined_tokens = self._shared_token_refiner_step(
                        refined_tokens,
                        input_tokens,
                        conditioning,
                        padding_mask,
                    )
        for _ in range(self.shared_token_refiner_steps - no_grad_steps):
            refined_tokens = self._shared_token_refiner_step(
                refined_tokens,
                input_tokens,
                conditioning,
                padding_mask,
            )
        return refined_tokens

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor | list[dict[str, torch.Tensor]]]:
        dataset_embedding = self.dataset_embedding(batch["dataset_index"])
        task_embedding = self.task_embedding(batch["task_index"])
        if self.conditioning_mode == "dataset_only":
            task_embedding = torch.zeros_like(task_embedding)
        elif self.conditioning_mode == "task_only":
            dataset_embedding = torch.zeros_like(dataset_embedding)
        conditioning = torch.cat([dataset_embedding, task_embedding], dim=-1)
        if self.conditioning_mode == "none":
            conditioning = torch.zeros(
                (conditioning.size(0), self.conditioning_dim),
                device=conditioning.device,
                dtype=conditioning.dtype,
            )
        concept_mask = self._resolve_concept_mask(batch["concept_mask"])

        tokens, modality_gates, token_mask = self._encode_modalities(batch)
        empty_token_rows = token_mask.sum(dim=1) <= 0.0
        if torch.any(empty_token_rows):
            tokens = tokens.clone()
            token_mask = token_mask.clone()
            tokens[empty_token_rows, 0, :] = self.modality_embeddings[0].view(1, -1)
            token_mask[empty_token_rows, 0] = 1.0
        if self.use_token_film:
            tokens = self.token_adapter(tokens, conditioning)
        padding_mask = token_mask < 0.5
        for block in self.transformer_blocks:
            tokens = block(tokens, conditioning, padding_mask)
        tokens = self._run_shared_token_refiner(tokens, conditioning, padding_mask)

        pooled = self._pool_tokens(tokens, token_mask)
        base_latent = self.pool_projection(pooled)
        deprest_mask, deprest_adapter_input, deprest_coverage_scale = self._build_deprest_adapter_inputs(tokens, batch, conditioning)
        deprest_edge_context = None
        deprest_routing_context = None
        if (
            (self.deprest_edge_specialist_heads or self.deprest_edge_ovr_heads)
            and deprest_adapter_input is not None
            and deprest_coverage_scale is not None
        ):
            deprest_edge_context = torch.cat([deprest_adapter_input, deprest_coverage_scale], dim=-1)
        if deprest_adapter_input is not None and deprest_coverage_scale is not None:
            deprest_routing_context = torch.cat([deprest_adapter_input, deprest_coverage_scale, modality_gates], dim=-1)
        if self.deprest_adapter is not None and deprest_adapter_input is not None and deprest_mask is not None and deprest_coverage_scale is not None:
            adapter_residual = self.deprest_adapter(deprest_adapter_input) * deprest_coverage_scale
            base_latent = base_latent.clone()
            base_latent[deprest_mask] = base_latent[deprest_mask] + adapter_residual[deprest_mask]
        latent = self.latent_adapter(base_latent, conditioning) if self.use_latent_film else base_latent

        concept_state = latent.new_zeros((latent.size(0), self.concept_dim))
        output_state = latent.new_zeros((latent.size(0), self.output_refine_dim))
        steps: list[dict[str, torch.Tensor]] = []
        for _ in range(self.recursion_steps):
            cell_input = torch.cat([base_latent, concept_state, output_state, conditioning], dim=-1)
            latent = self.latent_norm(self.recursive_cell(cell_input, latent))
            if self.concept_head is not None:
                raw_concepts = self.concept_head(latent)
                concepts = raw_concepts * concept_mask
            else:
                raw_concepts = latent.new_zeros((latent.size(0), 0))
                concepts = raw_concepts
            concept_features = self._project_task_features(latent, conditioning, concepts)
            psyche_end_logits = None
            task_features = concept_features
            concept_compat_context = None
            if self.deprest_concept_compat_modules and self.deprest_concept_compat_feature_indices:
                concept_compat_context = batch["deprest_comm_features"].new_zeros(
                    (batch["deprest_comm_features"].size(0), self.deprest_concept_compat_context_dim)
                )
                if batch["deprest_comm_features"].size(1) > 0:
                    feature_index_tensor = torch.as_tensor(
                        self.deprest_concept_compat_feature_indices,
                        dtype=torch.long,
                        device=batch["deprest_comm_features"].device,
                    )
                    selected_comm = torch.index_select(batch["deprest_comm_features"], dim=1, index=feature_index_tensor)
                    concept_compat_context[:, : selected_comm.size(1)] = selected_comm
                concept_compat_context[:, -self.total_token_count :] = modality_gates
            concept_contrast_context = None
            if self.deprest_concept_contrast_modules and self.deprest_concept_contrast_feature_indices:
                concept_contrast_context = batch["deprest_comm_features"].new_zeros(
                    (batch["deprest_comm_features"].size(0), self.deprest_concept_contrast_context_dim)
                )
                if batch["deprest_comm_features"].size(1) > 0:
                    feature_index_tensor = torch.as_tensor(
                        self.deprest_concept_contrast_feature_indices,
                        dtype=torch.long,
                        device=batch["deprest_comm_features"].device,
                    )
                    selected_comm = torch.index_select(batch["deprest_comm_features"], dim=1, index=feature_index_tensor)
                    concept_contrast_context[:, : selected_comm.size(1)] = selected_comm
                concept_contrast_context[:, -self.total_token_count :] = modality_gates
            if concept_compat_context is not None or concept_contrast_context is not None:
                selected_task_ids = {
                    int(task_id_str) for task_id_str in self.deprest_concept_compat_modules.keys()
                } | {
                    int(task_id_str) for task_id_str in self.deprest_concept_contrast_modules.keys()
                }
                for task_id in sorted(selected_task_ids):
                    compat_mask = (
                        (batch["task_index"] == int(task_id))
                        & (batch["dataset_index"] == int(self.deprest_dataset_index))
                        & (batch["deprest_comm_mask"] > 0.5)
                    )
                    if not torch.any(compat_mask):
                        continue
                    task_features = task_features.clone()
                    corrected_concepts = concepts[compat_mask]
                    task_id_str = str(int(task_id))
                    compat_module = (
                        self.deprest_concept_compat_modules[task_id_str]
                        if task_id_str in self.deprest_concept_compat_modules
                        else None
                    )
                    if compat_module is not None and concept_compat_context is not None:
                        corrected_concepts = compat_module(
                            corrected_concepts,
                            concept_compat_context[compat_mask],
                            concept_mask[compat_mask],
                        )
                    contrast_module = (
                        self.deprest_concept_contrast_modules[task_id_str]
                        if task_id_str in self.deprest_concept_contrast_modules
                        else None
                    )
                    if contrast_module is not None and concept_contrast_context is not None:
                        corrected_concepts = contrast_module(
                            corrected_concepts,
                            concept_contrast_context[compat_mask],
                            concept_mask[compat_mask],
                        )
                    task_features[compat_mask] = self._project_task_features(
                        latent[compat_mask],
                        conditioning[compat_mask],
                        corrected_concepts,
                    )
            if self.concept_residual is not None and self.concept_residual_gate is not None:
                residual_input = torch.cat([latent, conditioning], dim=-1)
                residual_features = self.concept_residual(residual_input)
                residual_gate = self.concept_residual_gate(residual_input)
                task_features = task_features + 0.2 * residual_gate * residual_features
            if (
                self.deprest_category_adapter is not None
                and deprest_adapter_input is not None
                and deprest_mask is not None
                and deprest_coverage_scale is not None
                and self.deprest_category_task_indices
            ):
                category_mask = deprest_mask & torch.zeros_like(deprest_mask, dtype=torch.bool)
                for task_id in self.deprest_category_task_indices:
                    category_mask = category_mask | (deprest_mask & (batch["task_index"] == int(task_id)))
                if torch.any(category_mask):
                    task_features = task_features.clone()
                    category_context = self.deprest_category_adapter(deprest_adapter_input) * deprest_coverage_scale
                    task_features[category_mask] = task_features[category_mask] + category_context[category_mask]
            if self.psyche_end_head is not None and self.psyche_end_adapter is not None:
                psyche_end_logits = self.psyche_end_head(concept_features)
                psyche_end_probs = torch.softmax(psyche_end_logits, dim=-1)
                psyche_mask = batch["dataset_index"] == int(self.psyche_dataset_index)
                psyche_context_mask = psyche_mask
                start_embed = None
                start_score_repr = None
                if torch.any(psyche_mask):
                    task_features = task_features.clone()
                    psyche_context = self.psyche_end_adapter(psyche_end_probs)
                    task_features[psyche_mask] = task_features[psyche_mask] + psyche_context[psyche_mask]
                psyche_binary_correction_context = None
                if (
                    self.psyche_two_stage_adapter is not None
                    and self.psyche_start_embedding is not None
                    and self.psyche_start_score_encoder is not None
                    and (self.psyche_change_task_indices or self.psyche_binary_correction_heads)
                ):
                    psyche_change_mask = psyche_mask & (batch["psyche_start_mask"] > 0.5) & (batch["psyche_start_score_mask"] > 0.5)
                    psyche_context_mask = psyche_change_mask
                    start_indices = batch["psyche_start_target"].clamp_min(0)
                    start_embed = self.psyche_start_embedding(start_indices)
                    start_score_repr = self.psyche_start_score_encoder(batch["psyche_start_score"].unsqueeze(-1))
                    if self.psyche_binary_correction_heads:
                        psyche_binary_correction_context = torch.cat(
                            [psyche_end_probs, start_embed, start_score_repr],
                            dim=-1,
                        )
                    if torch.any(psyche_change_mask) and self.psyche_change_task_indices:
                        conditioned_mask = psyche_change_mask & torch.zeros_like(psyche_change_mask, dtype=torch.bool)
                        for task_id in self.psyche_change_task_indices:
                            conditioned_mask = conditioned_mask | (psyche_change_mask & (batch["task_index"] == int(task_id)))
                        if torch.any(conditioned_mask):
                            task_features = task_features.clone()
                            two_stage_input = torch.cat([task_features, psyche_end_probs, start_embed, start_score_repr], dim=-1)
                            two_stage_context = self.psyche_two_stage_adapter(two_stage_input)
                            task_features[conditioned_mask] = task_features[conditioned_mask] + 0.2 * two_stage_context[conditioned_mask]
                if self.psyche_native_adapter is not None and self.psyche_native_task_indices:
                    psyche_native_mask = psyche_mask & (batch["native_mask"] > 0.5)
                    if torch.any(psyche_native_mask):
                        native_task_mask = psyche_native_mask & torch.zeros_like(psyche_native_mask, dtype=torch.bool)
                        for task_id in self.psyche_native_task_indices:
                            native_task_mask = native_task_mask | (psyche_native_mask & (batch["task_index"] == int(task_id)))
                        if torch.any(native_task_mask):
                            task_features = task_features.clone()
                            start_context = None
                            if start_embed is not None and start_score_repr is not None:
                                start_context = torch.cat([start_embed, start_score_repr], dim=-1)
                            task_features[native_task_mask] = self.psyche_native_adapter(
                                features=task_features[native_task_mask],
                                native_features=batch["native_features"][native_task_mask],
                                start_context=start_context[native_task_mask] if start_context is not None else None,
                                task_index=batch["task_index"][native_task_mask],
                            )
            else:
                psyche_binary_correction_context = None
                psyche_context_mask = None
            psyche_delta_bridge_state = None
            if self.psyche_delta_bridge is not None and self.psyche_dataset_index is not None:
                psyche_delta_mask = (
                    (batch["dataset_index"] == int(self.psyche_dataset_index))
                    & (batch["psyche_start_mask"] > 0.5)
                    & (batch["psyche_start_score_mask"] > 0.5)
                )
                if torch.any(psyche_delta_mask):
                    psyche_delta_bridge_state = self.psyche_delta_bridge(
                        features=task_features,
                        start_score=batch["psyche_start_score"],
                        start_category=batch["psyche_start_target"],
                    )
                    psyche_delta_bridge_state["mask"] = psyche_delta_mask
            if self.task_local_trm_refiner.enabled():
                trm_task_mask = torch.ones_like(batch["task_index"], dtype=torch.bool)
                if self.task_local_trm_task_indices:
                    trm_task_mask = torch.zeros_like(batch["task_index"], dtype=torch.bool)
                    for task_id in self.task_local_trm_task_indices:
                        trm_task_mask = trm_task_mask | (batch["task_index"] == int(task_id))
                if torch.any(trm_task_mask):
                    prior_task_features = task_features
                    refined_latent, refined_concepts, feature_delta, refined_output_state = self.task_local_trm_refiner(
                        latent=latent,
                        concepts=concepts,
                        task_features=task_features,
                        output_state=output_state,
                        conditioning=conditioning,
                        modality_gates=modality_gates,
                        concept_mask=concept_mask,
                        deprest_context=deprest_edge_context,
                        deprest_context_mask=deprest_mask,
                        psyche_context=psyche_binary_correction_context,
                        psyche_context_mask=psyche_context_mask,
                    )
                    latent = latent.clone()
                    concepts = concepts.clone()
                    task_features = task_features.clone()
                    output_state = output_state.clone()
                    latent[trm_task_mask] = refined_latent[trm_task_mask]
                    concepts[trm_task_mask] = refined_concepts[trm_task_mask]
                    output_state[trm_task_mask] = refined_output_state[trm_task_mask]
                    concept_features = self._project_task_features(latent, conditioning, concepts)
                    task_features = prior_task_features.clone()
                    task_features[trm_task_mask] = concept_features[trm_task_mask] + feature_delta[trm_task_mask]
            (
                logits_by_task,
                output_state,
                paired_logits_by_task,
                edge_ovr_logits_by_task,
                edge_ovr_probs_by_task,
                edge_ovr_mix_scale_by_task,
                deprest_construct_scores_by_task,
                ordinal_aux_logits_by_task,
            ) = self._apply_task_heads(
                task_features,
                batch["task_index"],
                psyche_binary_correction_context=psyche_binary_correction_context,
                psyche_delta_bridge_state=psyche_delta_bridge_state,
                deprest_edge_context=deprest_edge_context,
                deprest_routing_context=deprest_routing_context,
            )
            steps.append(
                {
                    "predictive_latent": latent,
                    "latent": latent,
                    "raw_concepts": raw_concepts,
                    "concepts": concepts,
                    "concept_features": concept_features,
                    "task_features": task_features,
                    "modality_gates": modality_gates,
                    "psyche_end_logits": psyche_end_logits,
                    "logits_by_task": logits_by_task,
                    "paired_logits_by_task": paired_logits_by_task,
                    "edge_ovr_logits_by_task": edge_ovr_logits_by_task,
                    "edge_ovr_probs_by_task": edge_ovr_probs_by_task,
                    "edge_ovr_mix_scale_by_task": edge_ovr_mix_scale_by_task,
                    "deprest_construct_scores_by_task": deprest_construct_scores_by_task,
                    "ordinal_aux_logits_by_task": ordinal_aux_logits_by_task,
                    "psyche_delta_bridge_state": psyche_delta_bridge_state,
                }
            )
            concept_state = concepts
        return {
            "tokens": tokens,
            "base_latent": base_latent,
            "modality_gates": modality_gates,
            "steps": steps,
        }

    def shared_parameters(self) -> list[nn.Parameter]:
        modules = [
            self.modality_encoders,
            self.temporal_frontends,
            self.dataset_embedding,
            self.task_embedding,
            self.token_adapter,
            self.transformer_blocks,
            self.pool_projection,
            self.latent_adapter,
            self.recursive_cell,
            self.latent_norm,
        ]
        if self.enable_missingness_embedding and self.modality_missing_projections:
            modules.append(self.modality_missing_projections)
        if self.native_missing_projection is not None:
            modules.append(self.native_missing_projection)
        if self.concept_head is not None:
            modules.append(self.concept_head)
        if self.concept_post is not None:
            modules.append(self.concept_post)
        if self.latent_task_adapter is not None:
            modules.append(self.latent_task_adapter)
        if self.shared_token_refiner is not None:
            modules.append(self.shared_token_refiner)
        if self.native_encoder is not None:
            modules.append(self.native_encoder)
        if self.deprest_call_encoder is not None:
            modules.append(self.deprest_call_encoder)
        if self.deprest_text_encoder is not None:
            modules.append(self.deprest_text_encoder)
        if self.deprest_meta_encoder is not None:
            modules.append(self.deprest_meta_encoder)
        if self.deprest_branch_gate is not None:
            modules.append(self.deprest_branch_gate)
        if self.deprest_adapter is not None:
            modules.append(self.deprest_adapter)
        if self.deprest_category_adapter is not None:
            modules.append(self.deprest_category_adapter)
        if self.deprest_edge_specialist_heads:
            modules.append(self.deprest_edge_specialist_heads)
        if self.deprest_edge_ovr_heads:
            modules.append(self.deprest_edge_ovr_heads)
        if self.deprest_pair_heads:
            modules.append(self.deprest_pair_heads)
        if self.concept_residual is not None:
            modules.append(self.concept_residual)
        if self.concept_residual_gate is not None:
            modules.append(self.concept_residual_gate)
        if self.psyche_end_head is not None:
            modules.append(self.psyche_end_head)
        if self.psyche_end_adapter is not None:
            modules.append(self.psyche_end_adapter)
        if self.psyche_start_embedding is not None:
            modules.append(self.psyche_start_embedding)
        if self.psyche_start_score_encoder is not None:
            modules.append(self.psyche_start_score_encoder)
        if self.psyche_two_stage_adapter is not None:
            modules.append(self.psyche_two_stage_adapter)
        parameters: list[nn.Parameter] = []
        for module in modules:
            parameters.extend(list(module.parameters()))
        if self.shared_token_refiner_residual_scale is not None:
            parameters.append(self.shared_token_refiner_residual_scale)
        for parameter in self.temporal_scales.values():
            parameters.append(parameter)
        for parameter in self.deprest_edge_ovr_mix_logits.values():
            parameters.append(parameter)
        parameters.append(self.modality_embeddings)
        if self.enable_missingness_tokens:
            for parameter in self.modality_missing_tokens.values():
                parameters.append(parameter)
            for parameter in self.modality_missing_gate_logits.values():
                parameters.append(parameter)
        if self.native_embedding is not None:
            parameters.append(self.native_embedding)
        if self.enable_missingness_tokens and self.native_missing_token is not None:
            parameters.append(self.native_missing_token)
        if self.enable_missingness_tokens and self.native_missing_gate_logit is not None:
            parameters.append(self.native_missing_gate_logit)
        return parameters

    def sparse_penalty(self) -> torch.Tensor:
        penalties = []
        for head in self.task_heads.values():
            penalties.append(self._head_sparsity_penalty(head))
        if not penalties:
            return torch.tensor(0.0, device=self.modality_embeddings.device)
        return torch.stack(penalties).mean()
