"""Multitask graph and graph-free forecasting models."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn

from .features import MODEL_FEATURES
from .graph import ResidualAdjacency, residual_edge_mask, validate_adjacency
from .models_v2 import (
    HEADLIGHT_V2_MODEL_TYPES,
    HeadlightV2Model,
    build_headlight_v2_model,
    v2_learned_edge_artifact,
)


FORECAST_OUTPUTS = (
    "request_q50",
    "request_q90",
    "cpu_q90",
    "p95_latency_q90",
    "sla_violation_logit",
)
MODEL_VARIANTS = ("full", "static_adjacency", "no_graph", "no_attention")


class GraphBlock(nn.Module):
    def __init__(self, hidden_width: int) -> None:
        super().__init__()
        self.self_projection = nn.Linear(hidden_width, hidden_width)
        self.neighbor_projection = nn.Linear(hidden_width, hidden_width, bias=False)
        self.normalization = nn.LayerNorm(hidden_width)

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        neighbors = torch.einsum("ij,bjh->bih", adjacency, features)
        update = functional.gelu(
            self.self_projection(features) + self.neighbor_projection(neighbors)
        )
        return self.normalization(features + update)


class HeadlightForecastModel(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        input_size: int,
        hidden_width: int,
        graph_blocks: int,
        static_adjacency: torch.Tensor,
        observed_mask: torch.Tensor,
        variant: str = "full",
        residual_bound: float = 0.3,
    ) -> None:
        super().__init__()
        if variant not in MODEL_VARIANTS:
            raise ValueError(f"unsupported model variant {variant!r}")
        self.num_nodes = num_nodes
        self.input_size = input_size
        self.hidden_width = hidden_width
        self.graph_blocks_count = graph_blocks
        self.variant = variant
        self.graph_enabled = variant != "no_graph"
        self.attention_enabled = variant != "no_attention"
        residual_enabled = variant in {"full", "no_attention"}
        mask = residual_edge_mask(
            static_adjacency.detach().cpu().numpy(),
            observed_mask.detach().cpu().numpy().astype(bool),
        )
        self.adjacency = ResidualAdjacency(
            static_adjacency,
            torch.tensor(mask, dtype=torch.float32, device=static_adjacency.device),
            residual_bound=residual_bound if residual_enabled else 0.0,
            learnable=residual_enabled,
        )
        self.input_projection = nn.Linear(input_size, hidden_width)
        self.temporal_score = nn.Linear(hidden_width, 1, bias=False)
        self.graph_blocks = nn.ModuleList(
            GraphBlock(hidden_width) for _ in range(graph_blocks if self.graph_enabled else 0)
        )
        self.output_head = nn.Sequential(
            nn.Linear(hidden_width, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, len(FORECAST_OUTPUTS)),
        )

    def effective_adjacency(self) -> torch.Tensor:
        if not self.graph_enabled:
            return torch.eye(
                self.num_nodes,
                dtype=self.adjacency.static_adjacency.dtype,
                device=self.adjacency.static_adjacency.device,
            )
        return self.adjacency()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4 or features.shape[-2:] != (self.num_nodes, self.input_size):
            raise ValueError(
                f"expected [batch,time,{self.num_nodes},{self.input_size}], got {tuple(features.shape)}"
            )
        encoded = functional.gelu(self.input_projection(features))
        if self.attention_enabled:
            weights = torch.softmax(self.temporal_score(encoded), dim=1)
            hidden = torch.sum(encoded * weights, dim=1)
        else:
            hidden = encoded[:, -1]
        adjacency = self.effective_adjacency()
        for block in self.graph_blocks:
            hidden = block(hidden, adjacency)
        return self.output_head(hidden)


class PerServiceLSTM(nn.Module):
    """Shared per-service LSTM with no graph or neighbor inputs."""

    def __init__(self, num_nodes: int, input_size: int, hidden_width: int) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.input_size = input_size
        self.hidden_width = hidden_width
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_width, batch_first=True)
        self.output_head = nn.Linear(hidden_width, len(FORECAST_OUTPUTS))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4 or features.shape[-2:] != (self.num_nodes, self.input_size):
            raise ValueError(
                f"expected [batch,time,{self.num_nodes},{self.input_size}], got {tuple(features.shape)}"
            )
        batch, steps, nodes, feature_count = features.shape
        per_service = features.permute(0, 2, 1, 3).reshape(
            batch * nodes, steps, feature_count
        )
        output, _ = self.lstm(per_service)
        return self.output_head(output[:, -1]).reshape(
            batch, nodes, len(FORECAST_OUTPUTS)
        )


def decode_forecasts(raw: torch.Tensor) -> torch.Tensor:
    positive = functional.softplus(raw[..., :4])
    request_q90 = torch.maximum(positive[..., 1:2], positive[..., 0:1])
    probability = torch.sigmoid(raw[..., 4:5])
    return torch.cat(
        (
            positive[..., 0:1],
            request_q90,
            positive[..., 2:4],
            probability,
        ),
        dim=-1,
    )


def pinball_loss(prediction: torch.Tensor, target: torch.Tensor, quantile: float) -> torch.Tensor:
    valid = torch.isfinite(target)
    if not torch.any(valid):
        return prediction.sum() * 0.0
    error = target[valid] - prediction[valid]
    return torch.mean(torch.maximum(quantile * error, (quantile - 1.0) * error))


def multitask_loss(raw: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """35% request, 25% CPU, 30% p95, and 10% SLA classification."""
    if target.shape[-1] != 4:
        raise ValueError("multitask targets must be request, CPU, p95, and SLA")
    positive = functional.softplus(raw[..., :4])
    request_loss = 0.5 * (
        pinball_loss(positive[..., 0], target[..., 0], 0.50)
        + pinball_loss(positive[..., 1], target[..., 0], 0.90)
    )
    cpu_loss = pinball_loss(positive[..., 2], target[..., 1], 0.90)
    latency_loss = pinball_loss(positive[..., 3], target[..., 2], 0.90)
    valid_sla = torch.isfinite(target[..., 3])
    sla_loss = (
        functional.binary_cross_entropy_with_logits(
            raw[..., 4][valid_sla],
            target[..., 3][valid_sla],
        )
        if torch.any(valid_sla)
        else raw[..., 4].sum() * 0.0
    )
    total = (
        0.35 * request_loss
        + 0.25 * cpu_loss
        + 0.30 * latency_loss
        + 0.10 * sla_loss
    )
    return total, {
        "request": request_loss,
        "cpu": cpu_loss,
        "p95_latency": latency_loss,
        "sla": sla_loss,
    }


def build_model(config: dict[str, Any], adjacency_payload: dict[str, Any], device: torch.device) -> nn.Module:
    num_nodes = int(config["num_nodes"])
    input_size = int(config.get("input_size", len(MODEL_FEATURES)))
    model_type = str(config["model_type"])
    if model_type in HEADLIGHT_V2_MODEL_TYPES:
        matrix = validate_adjacency(adjacency_payload, tuple(config["node_order"]))
    else:
        matrix = np.asarray(adjacency_payload["matrix"], dtype=np.float32)
    observed = np.asarray(
        adjacency_payload.get("observed_mask", matrix > 0),
        dtype=np.float32,
    )
    if matrix.shape != (num_nodes, num_nodes) or observed.shape != matrix.shape:
        raise ValueError("model adjacency dimensions do not match num_nodes")
    if model_type == "headlight":
        return HeadlightForecastModel(
            num_nodes=num_nodes,
            input_size=input_size,
            hidden_width=int(config["hidden_width"]),
            graph_blocks=int(config["graph_blocks"]),
            static_adjacency=torch.tensor(matrix, dtype=torch.float32, device=device),
            observed_mask=torch.tensor(observed, dtype=torch.float32, device=device),
            variant=str(config["variant"]),
            residual_bound=float(config["residual_bound"]),
        ).to(device)
    if model_type == "lstm":
        return PerServiceLSTM(
            num_nodes=num_nodes,
            input_size=input_size,
            hidden_width=int(config["hidden_width"]),
        ).to(device)
    if model_type in HEADLIGHT_V2_MODEL_TYPES:
        return build_headlight_v2_model(
            config,
            static_adjacency=torch.tensor(matrix, dtype=torch.float32, device=device),
            observed_mask=torch.tensor(observed, dtype=torch.float32, device=device),
        ).to(device)
    raise ValueError(f"unsupported model_type {model_type!r}")


def learned_edge_artifact(model: nn.Module, node_order: tuple[str, ...]) -> dict[str, Any]:
    if isinstance(model, HeadlightV2Model):
        return v2_learned_edge_artifact(model, node_order)
    if not isinstance(model, HeadlightForecastModel):
        return {"schema_version": "headlight.learned-edges/v1", "edges": []}
    residual = model.adjacency.residual().detach().cpu().numpy()
    effective = model.effective_adjacency().detach().cpu().numpy()
    edges = []
    for source_index, source in enumerate(node_order):
        for destination_index, destination in enumerate(node_order):
            contribution = float(residual[source_index, destination_index])
            if abs(contribution) <= 1e-12:
                continue
            edges.append(
                {
                    "source": source,
                    "destination": destination,
                    "residual": contribution,
                    "effective_weight": float(effective[source_index, destination_index]),
                }
            )
    return {
        "schema_version": "headlight.learned-edges/v1",
        "variant": model.variant,
        "residual_bound": model.adjacency.residual_bound,
        "edges": edges,
    }
