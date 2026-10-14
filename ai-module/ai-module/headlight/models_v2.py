"""Headlight causal temporal, directional graph, and candidate models."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as functional
from torch import nn

from .contracts_v2 import (
    DERIVED_REQUEST_FEATURES,
    REQUIRED_EDGE_FEATURES,
    REQUIRED_NODE_FEATURES,
)
from .graph import ResidualAdjacency, residual_edge_mask


HEADLIGHT_V2_MODEL_SCHEMA_VERSION = "headlight.model-config/v2"
HEADLIGHT_V2_MODEL_TYPES = ("headlight-v2", "headlight-v2-no-action")
HEADLIGHT_V2_VARIANTS = {
    "headlight-v2": "full-action-conditioned",
    "headlight-v2-no-action": "no-action-conditioning",
}

def validate_v2_model_config(config: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "model_type",
        "num_nodes",
        "node_order",
        "node_feature_names",
        "edge_feature_names",
        "service_type_ids",
        "num_service_types",
        "horizons_seconds",
        "endpoint_names",
        "sampling_interval_seconds",
        "hidden_width",
        "tcn_dilations",
        "tcn_kernel_size",
        "graph_blocks",
        "service_embedding_dim",
        "service_type_embedding_dim",
        "position_encoding_dim",
        "short_slope_window",
        "long_slope_window",
        "num_clusters",
        "readiness_feature_size",
        "readiness_feature_names",
        "residual_bound",
        "dropout",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Headlight model config is missing fields: {', '.join(missing)}")
    if config["schema_version"] != HEADLIGHT_V2_MODEL_SCHEMA_VERSION:
        raise ValueError("unsupported Headlight model config schema")
    model_type = str(config["model_type"])
    if model_type not in HEADLIGHT_V2_MODEL_TYPES:
        raise ValueError(f"unsupported Headlight model type {model_type!r}")
    num_nodes = int(config["num_nodes"])
    if num_nodes <= 0:
        raise ValueError("Headlight num_nodes must be positive")
    node_order = tuple(str(name) for name in config["node_order"])
    if len(node_order) != num_nodes or len(node_order) != len(set(node_order)):
        raise ValueError("Headlight node_order must contain one unique name per node")
    node_features = tuple(str(name) for name in config["node_feature_names"])
    missing_node = sorted(set(REQUIRED_NODE_FEATURES) - set(node_features))
    if missing_node:
        raise ValueError(f"Headlight node features are missing: {', '.join(missing_node)}")
    if len(node_features) != len(set(node_features)):
        raise ValueError("Headlight node feature names must be unique")
    edge_features = tuple(str(name) for name in config["edge_feature_names"])
    missing_edge = sorted(set(REQUIRED_EDGE_FEATURES) - set(edge_features))
    if missing_edge:
        raise ValueError(f"Headlight edge features are missing: {', '.join(missing_edge)}")
    if len(edge_features) != len(set(edge_features)):
        raise ValueError("Headlight edge feature names must be unique")
    service_type_ids = tuple(int(value) for value in config["service_type_ids"])
    num_service_types = int(config["num_service_types"])
    if len(service_type_ids) != num_nodes:
        raise ValueError("Headlight service_type_ids must contain one value per node")
    if num_service_types <= 0 or any(value < 0 or value >= num_service_types for value in service_type_ids):
        raise ValueError("Headlight service type IDs are outside the configured vocabulary")
    horizons = tuple(int(value) for value in config["horizons_seconds"])
    if not horizons or tuple(sorted(set(horizons))) != horizons or any(value <= 0 for value in horizons):
        raise ValueError("Headlight horizons_seconds must be unique, positive, and sorted")
    endpoint_names = tuple(str(value) for value in config["endpoint_names"])
    if not endpoint_names or len(endpoint_names) != len(set(endpoint_names)) or any(
        not value.strip() for value in endpoint_names
    ):
        raise ValueError("Headlight endpoint_names must be non-empty and unique")
    if int(config["sampling_interval_seconds"]) <= 0:
        raise ValueError("Headlight sampling interval must be positive")
    if int(config["hidden_width"]) <= 0:
        raise ValueError("Headlight hidden_width must be positive")
    dilations = tuple(int(value) for value in config["tcn_dilations"])
    if not dilations or any(value <= 0 for value in dilations):
        raise ValueError("Headlight TCN dilations must be positive")
    if int(config["tcn_kernel_size"]) < 2:
        raise ValueError("Headlight TCN kernel size must be at least two")
    if int(config["graph_blocks"]) <= 0:
        raise ValueError("Headlight must contain at least one directional graph block")
    for name in ("service_embedding_dim", "service_type_embedding_dim", "position_encoding_dim"):
        if int(config[name]) <= 0:
            raise ValueError(f"Headlight {name} must be positive")
    short_window = int(config["short_slope_window"])
    long_window = int(config["long_slope_window"])
    if short_window < 2 or long_window <= short_window:
        raise ValueError("Headlight slope windows must satisfy 2 <= short < long")
    if int(config["num_clusters"]) <= 0 or int(config["readiness_feature_size"]) <= 0:
        raise ValueError("Headlight cluster and readiness dimensions must be positive")
    readiness_names = tuple(str(value) for value in config["readiness_feature_names"])
    if (
        len(readiness_names) != int(config["readiness_feature_size"])
        or len(readiness_names) != len(set(readiness_names))
        or not readiness_names
        or readiness_names[0] != "ready_fraction"
    ):
        raise ValueError(
            "Headlight readiness features must be unique, match readiness_feature_size, "
            "and begin with ready_fraction"
        )
    if not 0.0 <= float(config["residual_bound"]) <= 0.3:
        raise ValueError("Headlight residual_bound must be between zero and 0.3")
    if not 0.0 <= float(config["dropout"]) < 1.0:
        raise ValueError("Headlight dropout must be in [0, 1)")
    if model_type == "headlight-v2-no-action":
        calibration = config.get("capacity_risk_calibration")
        if not isinstance(calibration, dict):
            raise ValueError("no-action-conditioning requires capacity_risk_calibration")
        calibration_fields = {
            "schema_version",
            "service_capacity_rps",
            "cluster_capacity_weights",
            "latency_floor_ms",
            "latency_scale_ms",
            "risk_temperature",
            "risk_bias",
            "readiness_penalty",
            "endpoint_risk_bias",
        }
        missing_calibration = sorted(calibration_fields - set(calibration))
        if missing_calibration:
            raise ValueError(
                "capacity-risk calibration is missing fields: "
                + ", ".join(missing_calibration)
            )
        if calibration["schema_version"] != "headlight.capacity-risk-calibration/v2":
            raise ValueError("unsupported capacity-risk calibration schema")
        service_capacity = tuple(float(value) for value in calibration["service_capacity_rps"])
        cluster_weights = tuple(float(value) for value in calibration["cluster_capacity_weights"])
        endpoint_bias = tuple(float(value) for value in calibration["endpoint_risk_bias"])
        if len(service_capacity) != num_nodes or any(
            not math.isfinite(value) or value <= 0 for value in service_capacity
        ):
            raise ValueError("capacity-risk service capacities must be positive and match num_nodes")
        if len(cluster_weights) != int(config["num_clusters"]) or any(
            not math.isfinite(value) or value <= 0 for value in cluster_weights
        ):
            raise ValueError("capacity-risk cluster weights must be positive and match num_clusters")
        if len(endpoint_bias) != len(endpoint_names) or any(
            not math.isfinite(value) for value in endpoint_bias
        ):
            raise ValueError("capacity-risk endpoint bias must match endpoint_names and be finite")
        latency_floor = float(calibration["latency_floor_ms"])
        latency_scale = float(calibration["latency_scale_ms"])
        risk_temperature = float(calibration["risk_temperature"])
        risk_bias = float(calibration["risk_bias"])
        readiness_penalty = float(calibration["readiness_penalty"])
        if not math.isfinite(latency_floor) or latency_floor < 0:
            raise ValueError("capacity-risk latency floor must be non-negative")
        if (
            not math.isfinite(latency_scale)
            or latency_scale <= 0
            or not math.isfinite(risk_temperature)
            or risk_temperature <= 0
            or not math.isfinite(risk_bias)
            or not math.isfinite(readiness_penalty)
            or readiness_penalty < 0
        ):
            raise ValueError("capacity-risk latency scale and temperature must be positive")


class CausalDilatedTCNBlock(nn.Module):
    """Two causal dilated convolutions with residual and channel normalization."""

    def __init__(self, hidden_width: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.conv1 = nn.Conv1d(hidden_width, hidden_width, kernel_size, dilation=dilation)
        self.conv2 = nn.Conv1d(hidden_width, hidden_width, kernel_size, dilation=dilation)
        self.norm1 = nn.LayerNorm(hidden_width)
        self.norm2 = nn.LayerNorm(hidden_width)
        self.dropout = nn.Dropout(dropout)

    def _convolve(self, values: torch.Tensor, convolution: nn.Conv1d) -> torch.Tensor:
        left_padding = (self.kernel_size - 1) * self.dilation
        return convolution(functional.pad(values, (left_padding, 0)))

    @staticmethod
    def _normalize(values: torch.Tensor, normalization: nn.LayerNorm) -> torch.Tensor:
        return normalization(values.transpose(1, 2)).transpose(1, 2)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        update = self._convolve(values, self.conv1)
        update = functional.gelu(self._normalize(update, self.norm1))
        update = self.dropout(update)
        update = self._convolve(update, self.conv2)
        update = self.dropout(functional.gelu(update))
        return self._normalize(values + update, self.norm2)


class LagPositionEncoding(nn.Module):
    """Deterministic sinusoidal encoding of causal lag from the decision point."""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.dimension = int(dimension)

    def forward(self, steps: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        lag = torch.arange(steps - 1, -1, -1, device=device, dtype=dtype)
        half = max(1, math.ceil(self.dimension / 2))
        frequencies = torch.exp(
            torch.arange(half, device=device, dtype=dtype)
            * (-math.log(10_000.0) / max(1, half - 1))
        )
        angles = lag[:, None] * frequencies[None, :]
        encoded = torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)
        return encoded[:, : self.dimension]


def aggregate_directional_messages(
    caller_messages: torch.Tensor,
    backpressure_messages: torch.Tensor,
    adjacency: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate `[source,destination]` messages in opposite causal directions."""

    caller_to_callee = torch.einsum("sd,bsdh->bdh", adjacency, caller_messages)
    callee_to_caller = torch.einsum("sd,bsdh->bsh", adjacency, backpressure_messages)
    return caller_to_callee, callee_to_caller


class DirectionalGraphBlockV2(nn.Module):
    """Distinct workload, backpressure, and self-state transformations."""

    def __init__(self, hidden_width: int, edge_feature_names: tuple[str, ...], dropout: float) -> None:
        super().__init__()
        self.edge_feature_names = edge_feature_names
        caller_names = ("call_rate", "call_ratio", "cross_cluster", "missing", "stale")
        backpressure_names = ("latency_ms", "error_rate", "cross_cluster", "missing", "stale")
        self.caller_edge_indices = tuple(edge_feature_names.index(name) for name in caller_names)
        self.backpressure_edge_indices = tuple(
            edge_feature_names.index(name) for name in backpressure_names
        )
        self.caller_edge_encoder = nn.Sequential(
            nn.Linear(len(self.caller_edge_indices), hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, hidden_width),
        )
        self.backpressure_edge_encoder = nn.Sequential(
            nn.Linear(len(self.backpressure_edge_indices), hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, hidden_width),
        )
        self.caller_message = nn.Linear(hidden_width, hidden_width, bias=False)
        self.backpressure_message = nn.Linear(hidden_width, hidden_width, bias=False)
        self.self_projection = nn.Linear(hidden_width, hidden_width)
        self.merge = nn.Linear(hidden_width * 3, hidden_width)
        self.normalization = nn.LayerNorm(hidden_width)
        self.dropout = nn.Dropout(dropout)

    def split_edge_features(self, edge_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            edge_features[..., list(self.caller_edge_indices)],
            edge_features[..., list(self.backpressure_edge_indices)],
        )

    def forward(
        self,
        hidden: torch.Tensor,
        edge_features: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, nodes, width = hidden.shape
        source = hidden[:, :, None, :].expand(batch, nodes, nodes, width)
        destination = hidden[:, None, :, :].expand(batch, nodes, nodes, width)
        caller_edges, backpressure_edges = self.split_edge_features(edge_features)
        caller_messages = self.caller_message(source) + self.caller_edge_encoder(caller_edges)
        backpressure_messages = (
            self.backpressure_message(destination)
            + self.backpressure_edge_encoder(backpressure_edges)
        )
        caller_to_callee, callee_to_caller = aggregate_directional_messages(
            caller_messages,
            backpressure_messages,
            adjacency,
        )
        self_state = self.self_projection(hidden)
        update = functional.gelu(
            self.merge(torch.cat((self_state, caller_to_callee, callee_to_caller), dim=-1))
        )
        output = self.normalization(hidden + self.dropout(update))
        return output, {
            "caller_to_callee": caller_to_callee,
            "callee_to_caller": callee_to_caller,
            "self_state": self_state,
        }


class CalibratedCapacityRiskSurrogate(nn.Module):
    """Fixed capacity calibration used by the no-action-conditioning ablation."""

    def __init__(
        self,
        service_capacity_rps: torch.Tensor,
        cluster_capacity_weights: torch.Tensor,
        latency_floor_ms: float,
        latency_scale_ms: float,
        risk_temperature: float,
        risk_bias: float,
        readiness_penalty: float,
        endpoint_risk_bias: torch.Tensor,
    ) -> None:
        super().__init__()
        if torch.any(service_capacity_rps <= 0) or torch.any(cluster_capacity_weights <= 0):
            raise ValueError("capacity-risk calibration values must be positive")
        if latency_floor_ms < 0 or latency_scale_ms <= 0 or risk_temperature <= 0:
            raise ValueError("capacity-risk latency and temperature values are invalid")
        self.register_buffer("service_capacity_rps", service_capacity_rps.to(dtype=torch.float32))
        self.register_buffer("cluster_capacity_weights", cluster_capacity_weights.to(dtype=torch.float32))
        self.register_buffer("endpoint_risk_bias", endpoint_risk_bias.to(dtype=torch.float32))
        self.latency_floor_ms = float(latency_floor_ms)
        self.latency_scale_ms = float(latency_scale_ms)
        self.risk_temperature = float(risk_temperature)
        self.risk_bias = float(risk_bias)
        self.readiness_penalty = float(readiness_penalty)

    def forward(
        self,
        demand_q90: torch.Tensor,
        candidate_replicas: torch.Tensor,
        candidate_placement: torch.Tensor,
        readiness: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if candidate_replicas.ndim == 4 and candidate_replicas.shape[-1] == 1:
            candidate_replicas = candidate_replicas[..., 0]
        if candidate_replicas.ndim != 3:
            raise ValueError("candidate_replicas must be [batch,candidate,node]")
        if candidate_placement.ndim != 4:
            raise ValueError("candidate_placement must be [batch,candidate,node,cluster]")
        if readiness.ndim != 4 or readiness.shape[-1] < 1:
            raise ValueError("readiness must be [batch,candidate,node,feature]")
        if candidate_placement.shape[:-1] != candidate_replicas.shape:
            raise ValueError("candidate replica and placement dimensions differ")
        if demand_q90.ndim != 3:
            raise ValueError("demand_q90 must be [batch,node,horizon]")
        if demand_q90.shape[:2] != (
            candidate_replicas.shape[0],
            candidate_replicas.shape[2],
        ):
            raise ValueError("candidate batch and node dimensions must match demand_q90")
        if readiness.shape[:3] != candidate_replicas.shape:
            raise ValueError("candidate replica and readiness dimensions differ")
        if not torch.allclose(candidate_placement.sum(dim=-1), candidate_replicas, atol=1e-5):
            raise ValueError("candidate replicas must equal the placement sum")
        if candidate_placement.shape[-1] != len(self.cluster_capacity_weights):
            raise ValueError("candidate placement cluster dimension is incompatible")
        effective_replicas = torch.sum(
            candidate_placement * self.cluster_capacity_weights,
            dim=-1,
        ).clamp_min(1e-6)
        capacity = effective_replicas * self.service_capacity_rps[None, None, :]
        load_ratio = demand_q90[:, None, :, :] / capacity[..., None]
        ready_fraction = readiness[..., 0].clamp(0.0, 1.0)
        service_risk_logit = (
            (load_ratio - 1.0) * self.risk_temperature
            + self.risk_bias
            + (1.0 - ready_fraction[..., None]) * self.readiness_penalty
        )
        risk_logit = torch.amax(service_risk_logit, dim=2)[:, :, None, :]
        risk_logit = risk_logit + self.endpoint_risk_bias[None, None, :, None]
        return {
            "predicted_p95_latency_ms": self.latency_floor_ms
            + self.latency_scale_ms * functional.softplus(risk_logit),
            "sla_violation_probability": torch.sigmoid(risk_logit),
            "source": "calibrated-capacity-surrogate",
        }


class HeadlightV2Model(nn.Module):
    """Shared v2 demand backbone with an optional learned action-conditioned head."""

    def __init__(
        self,
        config: dict[str, Any],
        static_adjacency: torch.Tensor,
        observed_mask: torch.Tensor,
    ) -> None:
        super().__init__()
        validate_v2_model_config(config)
        self.config = dict(config)
        self.model_type = str(config["model_type"])
        self.variant = HEADLIGHT_V2_VARIANTS[self.model_type]
        self.action_conditioning = self.model_type == "headlight-v2"
        self.num_nodes = int(config["num_nodes"])
        self.node_feature_names = tuple(str(name) for name in config["node_feature_names"])
        self.edge_feature_names = tuple(str(name) for name in config["edge_feature_names"])
        self.horizons_seconds = tuple(int(value) for value in config["horizons_seconds"])
        self.endpoint_names = tuple(str(value) for value in config["endpoint_names"])
        self.hidden_width = int(config["hidden_width"])
        self.num_clusters = int(config["num_clusters"])
        self.readiness_feature_size = int(config["readiness_feature_size"])
        self.sampling_interval_seconds = int(config["sampling_interval_seconds"])
        self.short_slope_window = int(config["short_slope_window"])
        self.long_slope_window = int(config["long_slope_window"])
        self.demand_output_scale_rps = float(config.get("demand_output_scale_rps", 1.0))
        self.latency_output_scale_ms = float(config.get("latency_output_scale_ms", 1.0))
        if self.demand_output_scale_rps <= 0 or self.latency_output_scale_ms <= 0:
            raise ValueError("Headlight output scales must be positive")
        self.request_index = self.node_feature_names.index("request_rate")

        residual_mask = residual_edge_mask(
            static_adjacency.detach().cpu().numpy(),
            observed_mask.detach().cpu().numpy().astype(bool),
        )
        self.adjacency = ResidualAdjacency(
            static_adjacency,
            torch.tensor(residual_mask, dtype=torch.float32, device=static_adjacency.device),
            residual_bound=float(config["residual_bound"]),
            learnable=float(config["residual_bound"]) > 0,
        )
        service_embedding_dim = int(config["service_embedding_dim"])
        service_type_embedding_dim = int(config["service_type_embedding_dim"])
        position_dimension = int(config["position_encoding_dim"])
        self.service_embedding = nn.Embedding(self.num_nodes, service_embedding_dim)
        self.service_type_embedding = nn.Embedding(
            int(config["num_service_types"]), service_type_embedding_dim
        )
        self.register_buffer(
            "service_type_ids",
            torch.tensor(config["service_type_ids"], dtype=torch.long),
        )
        self.position_encoding = LagPositionEncoding(position_dimension)
        input_width = (
            len(self.node_feature_names)
            + len(DERIVED_REQUEST_FEATURES)
            + service_embedding_dim
            + service_type_embedding_dim
            + position_dimension
        )
        self.input_projection = nn.Linear(input_width, self.hidden_width)
        self.temporal_blocks = nn.ModuleList(
            CausalDilatedTCNBlock(
                self.hidden_width,
                int(config["tcn_kernel_size"]),
                int(dilation),
                float(config["dropout"]),
            )
            for dilation in config["tcn_dilations"]
        )
        self.graph_blocks = nn.ModuleList(
            DirectionalGraphBlockV2(
                self.hidden_width,
                self.edge_feature_names,
                float(config["dropout"]),
            )
            for _ in range(int(config["graph_blocks"]))
        )
        horizon_count = len(self.horizons_seconds)
        self.demand_head = nn.Sequential(
            nn.Linear(self.hidden_width, self.hidden_width),
            nn.GELU(),
            nn.Linear(self.hidden_width, horizon_count * 2),
        )
        self.candidate_node_encoder: nn.Module | None = None
        self.endpoint_embedding: nn.Embedding | None = None
        self.performance_output_head: nn.Module | None = None
        self.capacity_risk_surrogate: CalibratedCapacityRiskSurrogate | None = None
        if self.action_conditioning:
            performance_input = (
                self.hidden_width
                + horizon_count * 2
                + 1
                + self.num_clusters
                + self.readiness_feature_size
            )
            self.candidate_node_encoder = nn.Sequential(
                nn.Linear(performance_input, self.hidden_width),
                nn.GELU(),
                nn.Linear(self.hidden_width, self.hidden_width),
                nn.GELU(),
            )
            self.endpoint_embedding = nn.Embedding(len(self.endpoint_names), self.hidden_width)
            self.performance_output_head = nn.Linear(self.hidden_width, horizon_count * 3)
        else:
            calibration = config["capacity_risk_calibration"]
            service_capacity = calibration.get("service_capacity_rps")
            cluster_weights = calibration.get("cluster_capacity_weights")
            if len(service_capacity) != self.num_nodes or len(cluster_weights) != self.num_clusters:
                raise ValueError("capacity-risk calibration dimensions do not match the model")
            self.capacity_risk_surrogate = CalibratedCapacityRiskSurrogate(
                torch.tensor(service_capacity, dtype=torch.float32, device=static_adjacency.device),
                torch.tensor(cluster_weights, dtype=torch.float32, device=static_adjacency.device),
                float(calibration["latency_floor_ms"]),
                float(calibration["latency_scale_ms"]),
                float(calibration["risk_temperature"]),
                float(calibration["risk_bias"]),
                float(calibration["readiness_penalty"]),
                torch.tensor(
                    calibration["endpoint_risk_bias"],
                    dtype=torch.float32,
                    device=static_adjacency.device,
                ),
            )

    def effective_adjacency(self) -> torch.Tensor:
        return self.adjacency()

    def residual_sparsity_loss(self) -> torch.Tensor:
        return torch.mean(torch.abs(self.adjacency.residual()))

    def _request_derivatives(self, node_features: torch.Tensor) -> torch.Tensor:
        request_rate = node_features[..., self.request_index : self.request_index + 1]
        difference = torch.cat(
            (
                torch.zeros_like(request_rate[:, :1]),
                request_rate[:, 1:] - request_rate[:, :-1],
            ),
            dim=1,
        ) / float(self.sampling_interval_seconds)

        def slope(window: int) -> torch.Tensor:
            steps = request_rate.shape[1]
            position = torch.arange(steps, device=request_rate.device)
            start = torch.clamp(position - (window - 1), min=0)
            earlier = request_rate.index_select(1, start)
            elapsed = (position - start).clamp_min(1).to(request_rate.dtype)
            elapsed = elapsed[None, :, None, None] * float(self.sampling_interval_seconds)
            values = (request_rate - earlier) / elapsed
            values[:, 0] = 0.0
            return values

        return torch.cat(
            (difference, slope(self.short_slope_window), slope(self.long_slope_window)),
            dim=-1,
        )

    def _encode_temporal(self, node_features: torch.Tensor) -> torch.Tensor:
        batch, steps, nodes, _ = node_features.shape
        derived = self._request_derivatives(node_features)
        service_ids = torch.arange(nodes, device=node_features.device)
        service = self.service_embedding(service_ids)[None, None, :, :].expand(batch, steps, -1, -1)
        service_type = self.service_type_embedding(self.service_type_ids)
        service_type = service_type[None, None, :, :].expand(batch, steps, -1, -1)
        position = self.position_encoding(
            steps,
            device=node_features.device,
            dtype=node_features.dtype,
        )[None, :, None, :].expand(batch, -1, nodes, -1)
        encoded = functional.gelu(
            self.input_projection(
                torch.cat((node_features, derived, service, service_type, position), dim=-1)
            )
        )
        encoded = encoded.permute(0, 2, 3, 1).reshape(
            batch * nodes,
            self.hidden_width,
            steps,
        )
        for block in self.temporal_blocks:
            encoded = block(encoded)
        return encoded[:, :, -1].reshape(batch, nodes, self.hidden_width)

    def _decode_demand(self, raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (*raw.shape[:-1], len(self.horizons_seconds), 2)
        per_horizon = raw.reshape(shape)
        q50 = functional.softplus(per_horizon[..., 0]) * self.demand_output_scale_rps
        q90 = q50 + functional.softplus(per_horizon[..., 1]) * self.demand_output_scale_rps
        return q50, q90

    def _learned_candidate_performance(
        self,
        hidden: torch.Tensor,
        demand_q50: torch.Tensor,
        demand_q90: torch.Tensor,
        candidate_replicas: torch.Tensor,
        candidate_placement: torch.Tensor,
        readiness: torch.Tensor,
    ) -> dict[str, torch.Tensor | str]:
        if (
            self.candidate_node_encoder is None
            or self.endpoint_embedding is None
            or self.performance_output_head is None
        ):
            raise RuntimeError("the no-action-conditioning model has no learned performance head")
        if candidate_replicas.ndim == 4 and candidate_replicas.shape[-1] == 1:
            candidate_replicas = candidate_replicas[..., 0]
        if candidate_replicas.ndim != 3:
            raise ValueError("candidate_replicas must be [batch,candidate,node]")
        if candidate_placement.shape != (*candidate_replicas.shape, self.num_clusters):
            raise ValueError("candidate_placement has incompatible dimensions")
        if readiness.shape != (*candidate_replicas.shape, self.readiness_feature_size):
            raise ValueError("candidate readiness has incompatible dimensions")
        if not torch.allclose(candidate_placement.sum(dim=-1), candidate_replicas, atol=1e-5):
            raise ValueError("candidate replicas must equal the placement sum")
        batch, candidates, nodes = candidate_replicas.shape
        if batch != hidden.shape[0]:
            raise ValueError("candidate batch size must match observation batch size")
        if nodes != self.num_nodes:
            raise ValueError("candidate node dimension must match the model node count")
        state = hidden[:, None, :, :].expand(batch, candidates, nodes, -1)
        demand = torch.cat((demand_q50, demand_q90), dim=-1)
        demand = demand[:, None, :, :].expand(batch, candidates, nodes, -1)
        action = torch.cat(
            (candidate_replicas[..., None], candidate_placement, readiness),
            dim=-1,
        )
        candidate_nodes = self.candidate_node_encoder(
            torch.cat((state, demand, action), dim=-1)
        )
        endpoint_ids = torch.arange(len(self.endpoint_names), device=hidden.device)
        endpoint = self.endpoint_embedding(endpoint_ids)
        endpoint_attention = torch.softmax(
            torch.einsum("bcnh,eh->bcen", candidate_nodes, endpoint)
            / math.sqrt(float(self.hidden_width)),
            dim=-1,
        )
        pooled = torch.einsum("bcen,bcnh->bceh", endpoint_attention, candidate_nodes)
        endpoint_state = (
            pooled
            + endpoint[None, None, :, :]
        )
        raw = self.performance_output_head(endpoint_state)
        raw = raw.reshape(
            batch,
            candidates,
            len(self.endpoint_names),
            len(self.horizons_seconds),
            3,
        )
        return {
            "predicted_p95_latency_ms": (
                functional.softplus(raw[..., 0]) * self.latency_output_scale_ms
            ),
            "sla_violation_probability": torch.sigmoid(raw[..., 1]),
            "success_probability": torch.sigmoid(raw[..., 2]),
            "source": "learned-action-conditioned",
        }

    def score_candidates_with_capacity_surrogate(
        self,
        demand_q90: torch.Tensor,
        candidate_replicas: torch.Tensor,
        candidate_placement: torch.Tensor,
        readiness: torch.Tensor,
    ) -> dict[str, torch.Tensor | str]:
        if self.capacity_risk_surrogate is None:
            raise RuntimeError("capacity surrogate is available only for no-action-conditioning")
        normalized_replicas = (
            candidate_replicas[..., 0]
            if candidate_replicas.ndim == 4 and candidate_replicas.shape[-1] == 1
            else candidate_replicas
        )
        if normalized_replicas.ndim != 3:
            raise ValueError("candidate_replicas must be [batch,candidate,node]")
        if candidate_placement.shape != (*normalized_replicas.shape, self.num_clusters):
            raise ValueError("candidate_placement has incompatible dimensions")
        if readiness.shape != (*normalized_replicas.shape, self.readiness_feature_size):
            raise ValueError("candidate readiness has incompatible dimensions")
        expected_demand_shape = (
            normalized_replicas.shape[0],
            self.num_nodes,
            len(self.horizons_seconds),
        )
        if tuple(demand_q90.shape) != expected_demand_shape:
            raise ValueError(
                "candidate batch size and model dimensions must match demand_q90; "
                f"expected {expected_demand_shape}, got {tuple(demand_q90.shape)}"
            )
        return self.capacity_risk_surrogate(
            demand_q90,
            normalized_replicas,
            candidate_placement,
            readiness,
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        candidate_replicas: torch.Tensor | None = None,
        candidate_placement: torch.Tensor | None = None,
        readiness: torch.Tensor | None = None,
        *,
        return_diagnostics: bool = False,
    ) -> dict[str, Any]:
        if node_features.ndim != 4 or node_features.shape[2:] != (
            self.num_nodes,
            len(self.node_feature_names),
        ):
            raise ValueError("node_features must be [batch,time,node,node_feature]")
        expected_edge = (
            node_features.shape[0],
            node_features.shape[1],
            self.num_nodes,
            self.num_nodes,
            len(self.edge_feature_names),
        )
        if tuple(edge_features.shape) != expected_edge:
            raise ValueError(f"edge_features must have shape {expected_edge}")
        if not torch.all(torch.isfinite(node_features)) or not torch.all(torch.isfinite(edge_features)):
            raise ValueError("Headlight inputs must be finite with explicit missing/stale masks")
        candidate_values = (candidate_replicas, candidate_placement, readiness)
        if any(value is not None for value in candidate_values) and not all(
            value is not None for value in candidate_values
        ):
            raise ValueError("candidate replicas, placement, and readiness must be provided together")
        if not self.action_conditioning and any(value is not None for value in candidate_values):
            raise ValueError(
                "no-action-conditioning excludes candidates from the learned forward path; "
                "use score_candidates_with_capacity_surrogate"
            )
        hidden = self._encode_temporal(node_features)
        graph_diagnostics: list[dict[str, torch.Tensor]] = []
        adjacency = self.effective_adjacency()
        latest_edges = edge_features[:, -1]
        for block in self.graph_blocks:
            hidden, diagnostics = block(hidden, latest_edges, adjacency)
            graph_diagnostics.append(diagnostics)
        demand_q50, demand_q90 = self._decode_demand(self.demand_head(hidden))
        candidate_performance = None
        if self.action_conditioning and candidate_replicas is not None:
            candidate_performance = self._learned_candidate_performance(
                hidden,
                demand_q50,
                demand_q90,
                candidate_replicas,
                candidate_placement,
                readiness,
            )
        result: dict[str, Any] = {
            "demand_q50": demand_q50,
            "demand_q90": demand_q90,
            "candidate_performance": candidate_performance,
            "variant": self.variant,
        }
        if return_diagnostics:
            result["diagnostics"] = {
                "hidden_state": hidden,
                "graph_paths": graph_diagnostics,
                "effective_adjacency": adjacency,
                "residual_sparsity": self.residual_sparsity_loss(),
            }
        return result


def build_headlight_v2_model(
    config: dict[str, Any],
    static_adjacency: torch.Tensor,
    observed_mask: torch.Tensor,
) -> HeadlightV2Model:
    return HeadlightV2Model(config, static_adjacency, observed_mask)


def v2_learned_edge_artifact(
    model: HeadlightV2Model,
    node_order: tuple[str, ...],
) -> dict[str, Any]:
    residual = model.adjacency.residual().detach().cpu()
    effective = model.effective_adjacency().detach().cpu()
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
        "schema_version": "headlight.learned-edges/v2",
        "orientation": "source-to-destination",
        "paths": ["caller-to-callee", "callee-to-caller", "self-state"],
        "residual_bound": model.adjacency.residual_bound,
        "sparsity_penalty": "mean-absolute-residual",
        "edges": edges,
    }


def model_parameter_count(model: nn.Module, *, trainable_only: bool = False) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if not trainable_only or parameter.requires_grad
    )


def estimate_peak_training_memory_bytes(
    model: HeadlightV2Model,
    *,
    batch_size: int,
    lookback: int,
    candidate_count: int,
) -> int:
    """Conservative FP32 Adam estimate for model state plus retained activations."""

    parameters = model_parameter_count(model)
    parameter_optimizer_bytes = parameters * 16  # weights, gradients, and two Adam moments
    node_activations = (
        batch_size
        * lookback
        * model.num_nodes
        * model.hidden_width
        * (6 + 4 * len(model.temporal_blocks) + 6 * len(model.graph_blocks))
        * 4
    )
    edge_activations = (
        batch_size
        * lookback
        * model.num_nodes
        * model.num_nodes
        * len(model.edge_feature_names)
        * 4
    )
    graph_pair_activations = (
        batch_size
        * model.num_nodes
        * model.num_nodes
        * model.hidden_width
        * len(model.graph_blocks)
        * 8  # two paths and a conservative retained-intermediate multiplier
        * 4
    )
    candidate_activations = (
        batch_size
        * candidate_count
        * model.num_nodes
        * (model.hidden_width + len(model.horizons_seconds) * 4)
        * 4
    )
    endpoint_activations = (
        batch_size
        * candidate_count
        * len(model.endpoint_names)
        * (model.hidden_width + len(model.horizons_seconds) * 2)
        * 4
    )
    framework_margin = 256 * 1024 * 1024
    return int(
        1.5
        * (
            parameter_optimizer_bytes
            + node_activations
            + edge_activations
            + graph_pair_activations
            + candidate_activations
            + endpoint_activations
        )
        + framework_margin
    )
