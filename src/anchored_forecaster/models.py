"""Pure-PatchTST anchored dynamic-graph forecaster.

The local anchor receives only target WATER histories.  The graph branch can
use all observed heterogeneous nodes and contributes a state- and lead-
dependent bounded residual correction.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class PatchTSTAnchor(nn.Module):
    """Channel-independent PatchTST-style forecast over WATER targets only."""

    def __init__(
        self,
        num_water_nodes: int,
        lookback: int,
        horizon: int,
        hidden_dim: int = 64,
        patch_len: int = 8,
        stride: int = 4,
        depth: int = 2,
        nhead: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim % nhead != 0:
            raise ValueError("hidden_dim must be divisible by nhead")
        self.num_water_nodes = num_water_nodes
        self.patch_len = min(patch_len, lookback)
        self.stride = max(1, stride)
        self.num_patches = 1 + max(0, lookback - self.patch_len) // self.stride
        self.patch_encoder = nn.Linear(self.patch_len, hidden_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, self.num_patches, hidden_dim))
        nn.init.trunc_normal_(self.positional_encoding, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.decoder = nn.Sequential(
            nn.LayerNorm(hidden_dim * self.num_patches),
            nn.Linear(hidden_dim * self.num_patches, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon),
        )

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor | None = None) -> torch.Tensor:
        water = x[:, : self.num_water_nodes, :]
        water_mask = torch.ones_like(water) if x_mask is None else x_mask[:, : self.num_water_nodes, :]
        positions = torch.arange(water.shape[-1], device=water.device).view(1, 1, -1)
        last_positions = positions.masked_fill(~water_mask.bool(), -1).amax(dim=-1, keepdim=True)
        gather_positions = last_positions.clamp_min(0)
        last = water.gather(-1, gather_positions)
        last = torch.where(last_positions >= 0, last, torch.zeros_like(last))
        centered = (water - last) * water_mask
        patches = centered.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        patches = patches[:, :, : self.num_patches, :]
        batch_size, nodes, patches_n, patch_len = patches.shape
        encoded = self.patch_encoder(patches.reshape(batch_size * nodes, patches_n, patch_len))
        encoded = encoded + self.positional_encoding[:, :patches_n]
        encoded = self.encoder(encoded)
        flat = encoded.reshape(batch_size, nodes, patches_n * encoded.shape[-1])
        return self.decoder(flat) + last


class AnchoredDynamicGraphForecaster(nn.Module):
    """PatchTST anchor plus an event-conditioned bounded graph residual.

    The first ``num_water_nodes`` input channels are WATER targets.  The
    dynamic graph encodes WATER, RAIN, WELL, PUMP, and GATE nodes from the
    observation window, their masks, types, and coordinates.  It never uses
    observations after the forecast issue time.
    """

    def __init__(
        self,
        num_nodes: int,
        num_water_nodes: int,
        lookback: int,
        horizon: int,
        node_type: torch.Tensor,
        coords: torch.Tensor,
        hidden_dim: int = 64,
        type_dim: int = 8,
        top_k: int = 20,
        dropout: float = 0.1,
        graph_gate_bias_init: float = -6.0,
        max_graph_delta_short: float = 0.005,
        max_graph_delta_long: float = 0.035,
        variant: str = "ours",
    ) -> None:
        super().__init__()
        if num_nodes < 2:
            raise ValueError("The dynamic graph requires at least two observed nodes")
        if node_type.numel() != num_nodes or coords.shape != (num_nodes, 2):
            raise ValueError("node_type and coords must describe every observed node")
        self.num_nodes = num_nodes
        self.num_water_nodes = num_water_nodes
        self.lookback = lookback
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        self.top_k = min(max(1, top_k), num_nodes)
        valid_variants = {
            "ours", "no_graph_correction",
            "no_event_context", "no_correction_bound", "no_event_context_bound",
            "fixed_graph", "without_neighbor_water", "without_rain",
            "without_well", "without_pump_gate", "without_nonwater",
        }
        if variant not in valid_variants:
            raise ValueError(f"Unknown anchored forecaster variant: {variant}")
        self.variant = variant

        # Target WATER histories provide the temporal anchor.
        self.anchor = PatchTSTAnchor(
            num_water_nodes=num_water_nodes,
            lookback=lookback,
            horizon=horizon,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.register_buffer("node_type", node_type.clone().long())
        coords = coords.clone().float()
        coord_mean = coords.mean(dim=0, keepdim=True)
        coord_std = coords.std(dim=0, keepdim=True).clamp_min(1e-6)
        self.register_buffer("coords", (coords - coord_mean) / coord_std)
        budget = torch.linspace(max_graph_delta_short, max_graph_delta_long, steps=horizon)
        self.register_buffer("correction_budget", budget.view(1, 1, horizon).float())

        self.type_embedding = nn.Embedding(5, type_dim)
        self.coord_encoder = nn.Sequential(nn.Linear(2, type_dim), nn.GELU(), nn.Linear(type_dim, type_dim))
        self.input_encoder = nn.Linear(2 + 2 * type_dim, hidden_dim)
        self.state_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.type_pair_bias = nn.Parameter(torch.zeros(5, 5))
        self.distance_scale = nn.Parameter(torch.tensor(0.1))
        self.event_encoder = nn.Sequential(
            nn.LayerNorm(10),
            nn.Linear(10, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.graph_update = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.graph_decoder = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon),
        )
        self.graph_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, horizon),
        )
        nn.init.zeros_(self.graph_decoder[-1].weight)
        nn.init.zeros_(self.graph_decoder[-1].bias)
        nn.init.zeros_(self.graph_gate[-1].weight)
        nn.init.constant_(self.graph_gate[-1].bias, graph_gate_bias_init)

    def _node_context(self, batch_size: int) -> torch.Tensor:
        type_embedding = self.type_embedding(self.node_type)
        coordinate_embedding = self.coord_encoder(self.coords)
        context = torch.cat([type_embedding, coordinate_embedding], dim=-1)
        return context.unsqueeze(0).expand(batch_size, -1, -1)

    def _distance_bias(self) -> torch.Tensor:
        distance = torch.cdist(self.coords, self.coords, p=2)
        return -F.softplus(self.distance_scale) * distance

    def _excluded_source_types(self) -> set[int]:
        if self.variant == "without_neighbor_water":
            return {0}
        if self.variant == "without_rain":
            return {1}
        if self.variant == "without_well":
            return {2}
        if self.variant == "without_pump_gate":
            return {3, 4}
        if self.variant == "without_nonwater":
            return {1, 2, 3, 4}
        return set()

    def _dynamic_graph(self, state: torch.Tensor) -> torch.Tensor:
        type_bias = self.type_pair_bias[self.node_type][:, self.node_type]
        if self.variant == "fixed_graph":
            scores = type_bias.unsqueeze(0) + self._distance_bias().unsqueeze(0)
            scores = scores.expand(state.shape[0], -1, -1)
        else:
            query = self.query(state)
            key = self.key(state)
            scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.hidden_dim)
            scores = scores + type_bias.unsqueeze(0) + self._distance_bias().unsqueeze(0)
        self_edges = torch.eye(self.num_nodes, device=scores.device, dtype=torch.bool).unsqueeze(0)
        scores = scores.masked_fill(self_edges, -1e9)
        excluded_types = self._excluded_source_types()
        source_mask = torch.zeros(self.num_nodes, device=scores.device, dtype=torch.bool)
        if excluded_types:
            for type_index in excluded_types:
                source_mask |= self.node_type == type_index
            scores = scores.masked_fill(source_mask.view(1, 1, -1), -1e9)
        if self.top_k < self.num_nodes:
            source_indices = torch.topk(scores, k=self.top_k, dim=-1).indices
            retained = torch.zeros_like(scores, dtype=torch.bool)
            retained.scatter_(-1, source_indices, True)
            scores = scores.masked_fill(~retained, -1e9)
        graph = torch.softmax(scores, dim=-1)
        valid_edges = ~self_edges
        if excluded_types:
            valid_edges = valid_edges & ~source_mask.view(1, 1, -1)
        graph = graph * valid_edges.to(graph.dtype)
        return graph / graph.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    @staticmethod
    def _masked_summary(values: torch.Tensor, masks: torch.Tensor, recent_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
        recent_values = values[:, :, -recent_steps:]
        recent_masks = masks[:, :, -recent_steps:]
        magnitude = (recent_values.abs() * recent_masks).sum(dim=(1, 2))
        magnitude = magnitude / recent_masks.sum(dim=(1, 2)).clamp_min(1.0)

        positions = torch.arange(values.shape[-1], device=values.device).view(1, 1, -1)
        first_positions = positions.masked_fill(~masks.bool(), values.shape[-1]).amin(dim=-1)
        last_positions = positions.masked_fill(~masks.bool(), -1).amax(dim=-1)
        valid_nodes = last_positions >= 0
        first = values.gather(-1, first_positions.clamp_max(values.shape[-1] - 1).unsqueeze(-1)).squeeze(-1)
        last = values.gather(-1, last_positions.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        trend = ((last - first) * valid_nodes).sum(dim=1) / valid_nodes.sum(dim=1).clamp_min(1)
        return magnitude, trend

    def _regime_summary(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        features: list[torch.Tensor] = []
        excluded_types = self._excluded_source_types()
        for type_index in range(5):
            indices = torch.nonzero(self.node_type == type_index, as_tuple=False).flatten()
            if indices.numel() == 0 or type_index in excluded_types:
                zeros = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
                features.extend([zeros, zeros])
                continue
            values = x.index_select(1, indices)
            masks = x_mask.index_select(1, indices)
            features.extend(self._masked_summary(values, masks, min(self.lookback, 24)))
        return torch.stack(features, dim=-1)

    def _components(self, x: torch.Tensor, x_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size, num_nodes, lookback = x.shape
        if num_nodes != self.num_nodes or lookback != self.lookback:
            raise ValueError(f"Expected x shape [B,{self.num_nodes},{self.lookback}], got {tuple(x.shape)}")
        x = x * x_mask
        anchor_prediction = self.anchor(x, x_mask)
        context = self._node_context(batch_size).unsqueeze(2).expand(-1, -1, lookback, -1)
        sequence = torch.stack([x, x_mask], dim=-1)
        sequence = torch.cat([sequence, context], dim=-1)
        sequence = self.input_encoder(sequence.reshape(batch_size * num_nodes, lookback, -1))
        state, _ = self.state_gru(sequence)
        state = state[:, -1, :].reshape(batch_size, num_nodes, self.hidden_dim)
        graph = self._dynamic_graph(state)
        message = torch.matmul(graph, self.value(state))
        graph_state = state + self.graph_update(torch.cat([state, message], dim=-1))
        water_state = graph_state[:, : self.num_water_nodes, :]
        regime = self.event_encoder(self._regime_summary(x, x_mask))
        if self.variant in {"no_event_context", "no_event_context_bound"}:
            regime = torch.zeros_like(regime)
        regime = regime.unsqueeze(1).expand(-1, self.num_water_nodes, -1)
        raw_delta = self.graph_decoder(water_state)
        if self.variant in {"no_correction_bound", "no_event_context_bound"}:
            graph_delta = raw_delta
        else:
            graph_delta = self.correction_budget.to(x.device) * torch.tanh(raw_delta)
        gate_input = torch.cat([state[:, : self.num_water_nodes, :], regime], dim=-1)
        gate = torch.sigmoid(self.graph_gate(gate_input))
        if self.variant == "no_graph_correction":
            correction = torch.zeros_like(graph_delta)
        else:
            correction = gate * graph_delta
        return {
            "prediction": anchor_prediction + correction,
            "anchor_prediction": anchor_prediction,
            "graph": graph,
            "regime": regime,
            "gate": gate,
            "graph_delta": graph_delta,
            "correction": correction,
            "correction_budget": self.correction_budget.to(x.device).expand_as(graph_delta),
        }

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor | None = None,
        return_graph: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if x_mask is None:
            x_mask = torch.ones_like(x)
        outputs = self._components(x, x_mask)
        if return_graph:
            return outputs["prediction"], outputs["graph"]
        return outputs["prediction"]

    def forward_with_diagnostics(self, x: torch.Tensor, x_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Return the prediction and interpretable graph-correction tensors."""

        if x_mask is None:
            x_mask = torch.ones_like(x)
        return self._components(x, x_mask)
