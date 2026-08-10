"""Static traffic adjacency and bounded learned residual adjacency."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import torch
from torch import nn


ADJACENCY_SCHEMA_VERSION = "headlight.adjacency/v1"


def normalize_adjacency(matrix: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Row-normalize a non-negative directed adjacency matrix."""
    if isinstance(matrix, torch.Tensor):
        nonnegative = torch.clamp(matrix, min=0.0)
        denominator = torch.clamp(nonnegative.sum(dim=-1, keepdim=True), min=1e-12)
        return nonnegative / denominator
    nonnegative_np = np.clip(np.asarray(matrix, dtype=np.float32), 0.0, None)
    denominator_np = np.clip(nonnegative_np.sum(axis=-1, keepdims=True), 1e-12, None)
    return (nonnegative_np / denominator_np).astype(np.float32)


def build_static_adjacency(
    node_order: Iterable[str],
    edges: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    nodes = tuple(node_order)
    index = {name: position for position, name in enumerate(nodes)}
    traffic = np.zeros((len(nodes), len(nodes)), dtype=np.float64)
    edge_totals: dict[tuple[str, str], float] = {}
    for edge in edges:
        source = str(edge.get("source") or edge.get("source_workload") or "")
        destination = str(edge.get("destination") or edge.get("destination_workload") or "")
        if source not in index or destination not in index:
            continue
        try:
            weight = float(edge.get("weight", edge.get("request_rate", edge.get("requests", 1.0))))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(weight) or weight <= 0:
            continue
        traffic[index[source], index[destination]] += weight
        edge_totals[(source, destination)] = edge_totals.get((source, destination), 0.0) + weight
    observed = traffic > 0
    np.fill_diagonal(traffic, traffic.diagonal() + 1.0)
    normalized = normalize_adjacency(traffic)
    observed_edges = []
    for (source, destination), weight in sorted(edge_totals.items()):
        source_total = sum(
            value for (candidate_source, _), value in edge_totals.items() if candidate_source == source
        )
        observed_edges.append(
            {
                "source": source,
                "destination": destination,
                "weight": weight,
                "traffic_share": weight / source_total if source_total else 0.0,
            }
        )
    return {
        "schema_version": ADJACENCY_SCHEMA_VERSION,
        "node_order": list(nodes),
        "matrix": normalized.tolist(),
        "observed_mask": observed.astype(int).tolist(),
        "observed_edges": observed_edges,
        "self_loops": True,
        "normalization": "row",
    }


def validate_adjacency(payload: dict[str, Any], node_order: Iterable[str]) -> np.ndarray:
    nodes = tuple(node_order)
    if payload.get("schema_version") != ADJACENCY_SCHEMA_VERSION:
        raise ValueError("unsupported adjacency schema")
    if tuple(payload.get("node_order", [])) != nodes:
        raise ValueError("adjacency node order differs from the service profile")
    matrix = np.asarray(payload.get("matrix"), dtype=np.float32)
    if matrix.shape != (len(nodes), len(nodes)):
        raise ValueError("adjacency matrix has an incompatible shape")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0):
        raise ValueError("adjacency matrix must be finite and non-negative")
    if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("adjacency matrix must be row-normalized")
    return matrix


def residual_edge_mask(static_adjacency: np.ndarray, observed_mask: np.ndarray | None = None) -> np.ndarray:
    adjacency = np.asarray(static_adjacency, dtype=np.float32)
    observed = (
        np.asarray(observed_mask, dtype=bool)
        if observed_mask is not None
        else adjacency > 0
    )
    if observed.shape != adjacency.shape or adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("static adjacency and observed mask must be matching square matrices")
    two_hop = (observed.astype(np.int8) @ observed.astype(np.int8)) > 0
    allowed = observed | two_hop
    np.fill_diagonal(allowed, True)
    return allowed


class ResidualAdjacency(nn.Module):
    """A = normalize(A_static + bounded residual), masked to local topology."""

    def __init__(
        self,
        static_adjacency: torch.Tensor,
        allowed_mask: torch.Tensor,
        residual_bound: float = 0.3,
        learnable: bool = True,
    ) -> None:
        super().__init__()
        if not 0 <= residual_bound <= 0.3:
            raise ValueError("residual_bound must be between 0 and 0.3")
        if static_adjacency.shape != allowed_mask.shape:
            raise ValueError("static adjacency and residual mask shapes differ")
        self.register_buffer("static_adjacency", static_adjacency.to(dtype=torch.float32))
        self.register_buffer("allowed_mask", allowed_mask.to(dtype=torch.float32))
        self.residual_bound = float(residual_bound)
        self.residual_logits = nn.Parameter(
            torch.zeros_like(self.static_adjacency),
            requires_grad=learnable and residual_bound > 0,
        )

    def residual(self) -> torch.Tensor:
        return (
            torch.tanh(self.residual_logits)
            * self.allowed_mask
            * self.residual_bound
        )

    def forward(self) -> torch.Tensor:
        return normalize_adjacency(self.static_adjacency + self.residual())
