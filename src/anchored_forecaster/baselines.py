"""Baseline models for WATER-stage forecasting.

Each model accepts a ``[batch, nodes, lookback]`` tensor and an observation
mask, and returns forecasts for WATER nodes. Architecture settings and input
adapter descriptions are defined in ``BASELINE_CONFIGS``.
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn

from .models import PatchTSTAnchor


BASELINE_CONFIGS: Dict[str, Dict[str, object]] = {
    "nlinear": {"description": "last-value-centered linear map", "hidden_dim": 0, "input_adapter": "native WATER mask"},
    "patchtst": {"layers": 3, "hidden_dim": 128, "heads": 16, "patch_len": 16, "stride": 8, "revin": True, "dropout": 0.2, "input_adapter": "native WATER mask"},
    "itransformer": {"hidden_dim": 512, "heads": 8, "layers": 2, "ffn_dim": 2048, "dropout": 0.1, "input_adapter": "mask token + type/coordinate token embeddings"},
    "timesnet": {"blocks": 2, "top_periods": 5, "hidden_dim": 32, "inception_kernels": 6, "dropout": 0.1, "input_adapter": "mask projection + static type/coordinate bias"},
    "fouriergnn": {"spectral_embedding": 256, "hidden_dim": 512, "stages": 3, "sparsity": 0.01, "input_adapter": "per-node value/mask/type/coordinate projection"},
    "mtgnn": {"layers": 3, "node_embedding": 40, "residual_channels": 32, "skip_channels": 64, "end_channels": 128, "dropout": 0.3, "input_adapter": "mask/type/coordinate channels before native learned graph"},
    "autotimes": {"backbone": "frozen-gpt2-base", "token_length": 24, "width": 256, "layers": 2, "dropout": 0.1, "input_adapter": "value + observation-mask token channels"},
    "graphwavenet": {"blocks": 4, "dilation_layers": 4, "residual_channels": 32, "skip_channels": 256, "end_channels": 512, "dropout": 0.3, "input_adapter": "value/mask/type/coordinate node features"},
}


class _NodeMetadata:
    """Static per-node metadata for multivariate baselines."""

    def __init__(self, node_type: torch.Tensor, coords: torch.Tensor, width: int = 4) -> None:
        self.node_type = node_type.long().clone()
        coords = coords.float()
        self.coords = (coords - coords.mean(0, keepdim=True)) / coords.std(0, keepdim=True).clamp_min(1e-6)
        self.width = width


class _NodeInputAdapter(nn.Module):
    """Project per-node values, observation masks, and static metadata."""

    def __init__(self, node_type: torch.Tensor, coords: torch.Tensor, width: int = 4) -> None:
        super().__init__()
        metadata = _NodeMetadata(node_type, coords, width)
        self.register_buffer("node_type", metadata.node_type)
        self.register_buffer("coords", metadata.coords)
        self.type_embedding = nn.Embedding(5, width)
        self.coord_projection = nn.Linear(2, width)
        self.value_mask_projection = nn.Linear(2 + 2 * width, 1)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return ("type_embedding", "coord_projection", "value_mask_projection")

    def static_features(self, batch_size: int, length: int) -> torch.Tensor:
        static = torch.cat(
            [self.type_embedding(self.node_type), self.coord_projection(self.coords)], dim=-1
        )
        return static.unsqueeze(0).unsqueeze(2).expand(batch_size, -1, length, -1)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        values = values * mask
        features = torch.cat([values.unsqueeze(-1), mask.unsqueeze(-1), self.static_features(
            values.shape[0], values.shape[-1]
        )], dim=-1)
        return self.value_mask_projection(features).squeeze(-1)


class NLinear(nn.Module):
    """NLinear-style normalized last-value-centered direct forecaster."""

    def __init__(self, num_water_nodes: int, lookback: int, horizon: int) -> None:
        super().__init__()
        self.num_water_nodes = num_water_nodes
        self.linear = nn.Linear(lookback, horizon)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor | None = None) -> torch.Tensor:
        water = x[:, : self.num_water_nodes, :]
        water_mask = torch.ones_like(water) if x_mask is None else x_mask[:, : self.num_water_nodes, :]
        positions = torch.arange(water.shape[-1], device=water.device).view(1, 1, -1)
        last_positions = positions.masked_fill(~water_mask.bool(), -1).amax(dim=-1, keepdim=True)
        last = water.gather(-1, last_positions.clamp_min(0))
        last = torch.where(last_positions >= 0, last, torch.zeros_like(last))
        return self.linear((water - last) * water_mask) + last


class PatchTSTBaseline(nn.Module):
    """Channel-independent PatchTST with masked normalization."""

    def __init__(self, num_water_nodes: int, lookback: int, horizon: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.num_water_nodes = num_water_nodes
        self.anchor = PatchTSTAnchor(
            num_water_nodes, lookback, horizon, hidden_dim=128, patch_len=16,
            stride=8, depth=3, nhead=16, dropout=dropout,
        )

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor | None = None) -> torch.Tensor:
        water = x[:, : self.num_water_nodes, :]
        water_mask = torch.ones_like(water) if x_mask is None else x_mask[:, : self.num_water_nodes, :]
        count = water_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        mean = ((water * water_mask).sum(dim=-1, keepdim=True) / count).detach()
        variance = (((water - mean).square() * water_mask).sum(dim=-1, keepdim=True) / count)
        std = variance.add(1e-5).sqrt().detach()
        normalized = x.clone()
        normalized[:, : self.num_water_nodes, :] = ((water - mean) / std) * water_mask
        return self.anchor(normalized, x_mask) * std + mean


class ITransformer(nn.Module):
    """iTransformer with value, observation-mask, and node-metadata embeddings."""

    def __init__(self, num_nodes: int, num_water_nodes: int, lookback: int, horizon: int,
                 d_model: int = 512, nhead: int = 8, layers: int = 2,
                 ff_dim: int = 2048, dropout: float = 0.1,
                 node_type: torch.Tensor | None = None,
                 coords: torch.Tensor | None = None) -> None:
        super().__init__()
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")
        self.num_nodes = num_nodes
        self.num_water_nodes = num_water_nodes
        self.token = nn.Linear(lookback, d_model)
        self.mask_token = nn.Linear(lookback, d_model, bias=False)
        self.type_embedding = nn.Embedding(5, d_model)
        self.coord_projection = nn.Linear(2, d_model, bias=False)
        if node_type is None or coords is None:
            raise ValueError("iTransformer requires node_type and coords for the input adapter")
        self.register_buffer("node_type", node_type.long().clone())
        coords = coords.float()
        self.register_buffer(
            "coords",
            (coords - coords.mean(0, keepdim=True)) / coords.std(0, keepdim=True).clamp_min(1e-6),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, horizon))

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor | None = None) -> torch.Tensor:
        if x_mask is None:
            x_mask = torch.ones_like(x)
        x = x * x_mask
        metadata = self.type_embedding(self.node_type) + self.coord_projection(self.coords)
        tokens = self.token(x) + self.mask_token(x_mask)
        tokens = tokens + metadata.unsqueeze(0)
        encoded = self.encoder(tokens)
        return self.head(encoded[:, : self.num_water_nodes, :])


class GraphWaveNet(nn.Module):
    """Graph WaveNet with static node-feature input adaptation."""

    def __init__(self, num_nodes: int, num_water_nodes: int, lookback: int, horizon: int,
                 channels: int = 32, dropout: float = 0.3,
                 node_type: torch.Tensor | None = None,
                 coords: torch.Tensor | None = None) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.num_water_nodes = num_water_nodes
        if node_type is None or coords is None:
            raise ValueError("GraphWaveNet requires node_type and coords for the input adapter")
        self.register_buffer("node_type", node_type.long().clone())
        coords = coords.float()
        self.register_buffer(
            "coords",
            (coords - coords.mean(0, keepdim=True)) / coords.std(0, keepdim=True).clamp_min(1e-6),
        )
        self.type_embedding = nn.Embedding(5, 4)
        self.coord_projection = nn.Linear(2, 4)
        self.node_a = nn.Parameter(torch.randn(num_nodes, 10) * 0.02)
        self.node_b = nn.Parameter(torch.randn(10, num_nodes) * 0.02)
        self.input_proj = nn.Conv1d(10, channels, kernel_size=1)
        self.temporal = nn.ModuleList()
        self.skip = nn.ModuleList()
        for _block in range(4):
            for layer in range(4):
                dilation = 2 ** layer
                self.temporal.append(nn.Conv1d(channels, 2 * channels, kernel_size=2, dilation=dilation))
                self.skip.append(nn.Conv1d(channels, 256, kernel_size=1))
        self.end = nn.Sequential(nn.ReLU(), nn.Conv1d(256, 512, 1), nn.ReLU(), nn.Conv1d(512, horizon, 1))
        self.dropout = dropout
        self.lookback = lookback
        self.horizon = horizon

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor | None = None) -> torch.Tensor:
        if x_mask is None:
            x_mask = torch.ones_like(x)
        x = x * x_mask
        adjacency = torch.softmax(F.relu(self.node_a @ self.node_b), dim=-1)
        static = torch.cat([self.type_embedding(self.node_type), self.coord_projection(self.coords)], dim=-1)
        static = static.unsqueeze(0).unsqueeze(-1).expand(x.shape[0], -1, -1, x.shape[-1])
        features = torch.cat([x.unsqueeze(2), x_mask.unsqueeze(2), static], dim=2)
        b, n, c, t = features.shape
        state = self.input_proj(features.reshape(b * n, c, t)).reshape(b, n, -1, t)
        skip_total = None
        for convolution, skip_layer in zip(self.temporal, self.skip):
            dilation = convolution.dilation[0]
            temporal = convolution(F.pad(state.reshape(b * n, state.shape[2], state.shape[3]), (dilation, 0)))
            left, right = temporal.chunk(2, dim=1)
            temporal = (torch.tanh(left) * torch.sigmoid(right)).reshape(b, n, -1, t)
            mixed = torch.einsum("ij,bjct->bict", adjacency, temporal)
            state = state + F.dropout(mixed, p=self.dropout, training=self.training)
            current_skip = skip_layer(state.reshape(b * n, state.shape[2], t))[..., -1:]
            skip_total = current_skip if skip_total is None else skip_total + current_skip
        forecast = self.end(skip_total).squeeze(-1).reshape(b, n, self.horizon)
        return forecast[:, : self.num_water_nodes]


class _TimesBlock(nn.Module):
    def __init__(self, hidden_dim: int, top_periods: int = 5, kernels: int = 6, dropout: float = 0.1) -> None:
        super().__init__()
        kernel_sizes = [2 * index + 1 for index in range(kernels)]
        self.convolutions = nn.ModuleList([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, size), padding=(0, size // 2))
            for size in kernel_sizes
        ])
        self.top_periods = top_periods
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,D]. Dominant periods are selected from the current batch,
        # following the defining TimesNet temporal-2D representation.
        spectrum = torch.fft.rfft(x, dim=1)
        amplitude = spectrum.abs().mean(dim=(0, 2))
        amplitude[0] = 0
        k = min(self.top_periods, max(1, amplitude.numel() - 1))
        frequencies = torch.topk(amplitude, k=k).indices.clamp_min(1)
        outputs = []
        weights = []
        length = x.shape[1]
        for frequency in frequencies.tolist():
            period = max(1, length // frequency)
            padded_length = math.ceil(length / period) * period
            padded = F.pad(x, (0, 0, 0, padded_length - length))
            image = padded.reshape(x.shape[0], padded_length // period, period, x.shape[2]).permute(0, 3, 1, 2)
            transformed = torch.stack([convolution(image) for convolution in self.convolutions]).mean(dim=0)
            outputs.append(transformed.permute(0, 2, 3, 1).reshape(x.shape[0], padded_length, x.shape[2])[:, :length])
            weights.append(spectrum[:, frequency].abs().mean(dim=-1))
        mixture = torch.stack(outputs, dim=-1)
        period_weights = torch.softmax(torch.stack(weights, dim=-1), dim=-1).view(x.shape[0], 1, 1, -1)
        return x + self.dropout((mixture * period_weights).sum(dim=-1))


class TimesNet(nn.Module):
    """Two-block TimesNet with a per-input mask/metadata adapter."""

    def __init__(self, num_nodes: int, num_water_nodes: int, lookback: int, horizon: int,
                 hidden_dim: int = 32, blocks: int = 2, dropout: float = 0.1,
                 node_type: torch.Tensor | None = None,
                 coords: torch.Tensor | None = None) -> None:
        super().__init__()
        self.num_water_nodes = num_water_nodes
        self.horizon = horizon
        if node_type is None or coords is None:
            raise ValueError("TimesNet requires node_type and coords for the input adapter")
        self.register_buffer("node_type", node_type.long().clone())
        coords = coords.float()
        self.register_buffer(
            "coords",
            (coords - coords.mean(0, keepdim=True)) / coords.std(0, keepdim=True).clamp_min(1e-6),
        )
        self.type_embedding = nn.Embedding(5, 1)
        self.coord_projection = nn.Linear(2, 1)
        self.input_projection = nn.Linear(num_nodes, hidden_dim)
        self.mask_projection = nn.Linear(num_nodes, hidden_dim, bias=False)
        self.blocks = nn.ModuleList([_TimesBlock(hidden_dim, 5, 6, dropout) for _ in range(blocks)])
        self.normalization = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim * lookback, num_water_nodes * horizon)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor | None = None) -> torch.Tensor:
        if x_mask is None:
            x_mask = torch.ones_like(x)
        x = x * x_mask
        static = self.type_embedding(self.node_type).squeeze(-1) + self.coord_projection(self.coords).squeeze(-1)
        static = static.unsqueeze(0).unsqueeze(1).expand(x.shape[0], x.shape[-1], -1)
        encoded = self.input_projection(x.transpose(1, 2))
        encoded = encoded + self.mask_projection(x_mask.transpose(1, 2))
        encoded = encoded + self.input_projection(static)
        for block in self.blocks:
            encoded = self.normalization(block(encoded))
        return self.head(encoded.flatten(1)).reshape(x.shape[0], self.num_water_nodes, self.horizon)


class _ComplexLinear(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.real = nn.Linear(input_dim, output_dim)
        self.imag = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.complex(self.real(x.real) - self.imag(x.imag), self.real(x.imag) + self.imag(x.real))


class FourierGNN(nn.Module):
    """Spectral graph baseline with a per-node input adapter."""

    def __init__(self, num_nodes: int, num_water_nodes: int, lookback: int, horizon: int,
                 spectral_embedding: int = 256, hidden_dim: int = 512,
                 stages: int = 3, sparsity: float = 0.01,
                 node_type: torch.Tensor | None = None,
                 coords: torch.Tensor | None = None) -> None:
        super().__init__()
        self.num_water_nodes = num_water_nodes
        self.sparsity = sparsity
        if node_type is None or coords is None:
            raise ValueError("FourierGNN requires node_type and coords for the input adapter")
        self.node_adapter = _NodeInputAdapter(node_type, coords)
        dimensions = [num_nodes, spectral_embedding] + [hidden_dim] * max(0, stages - 2) + [spectral_embedding]
        self.stages = nn.ModuleList([_ComplexLinear(dimensions[index], dimensions[index + 1]) for index in range(len(dimensions) - 1)])
        self.output_projection = _ComplexLinear(spectral_embedding, num_water_nodes)
        self.head = nn.Linear(lookback, horizon)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor | None = None) -> torch.Tensor:
        if x_mask is None:
            x_mask = torch.ones_like(x)
        x = self.node_adapter(x, x_mask)
        spectrum = torch.fft.rfft(x, dim=-1).transpose(1, 2)
        for stage in self.stages:
            spectrum = stage(spectrum)
            spectrum = torch.complex(F.gelu(spectrum.real), F.gelu(spectrum.imag))
            spectrum = torch.where(spectrum.abs() >= self.sparsity, spectrum, torch.zeros_like(spectrum))
        spectrum = self.output_projection(spectrum).transpose(1, 2)
        temporal = torch.fft.irfft(spectrum, n=x.shape[-1], dim=-1)
        return self.head(temporal)


class MTGNN(nn.Module):
    """Learned-graph temporal convolution with static node features."""

    def __init__(self, num_nodes: int, num_water_nodes: int, lookback: int, horizon: int,
                 layers: int = 3, residual_channels: int = 32, dropout: float = 0.3,
                 node_type: torch.Tensor | None = None,
                 coords: torch.Tensor | None = None) -> None:
        super().__init__()
        self.num_water_nodes = num_water_nodes
        self.horizon = horizon
        if node_type is None or coords is None:
            raise ValueError("MTGNN requires node_type and coords for the input adapter")
        self.register_buffer("node_type", node_type.long().clone())
        coords = coords.float()
        self.register_buffer(
            "coords",
            (coords - coords.mean(0, keepdim=True)) / coords.std(0, keepdim=True).clamp_min(1e-6),
        )
        self.type_embedding = nn.Embedding(5, 4)
        self.coord_projection = nn.Linear(2, 4)
        self.node_a = nn.Parameter(torch.randn(num_nodes, 40) * 0.02)
        self.node_b = nn.Parameter(torch.randn(40, num_nodes) * 0.02)
        self.temporal = nn.ModuleList()
        self.skip = nn.ModuleList()
        for layer in range(layers):
            dilation = 2 ** layer
            self.temporal.append(nn.Conv1d(residual_channels, residual_channels * 2, 3, dilation=dilation))
            self.skip.append(nn.Conv1d(residual_channels, 64, 1))
        self.input_projection = nn.Conv1d(10, residual_channels, 1)
        self.end = nn.Sequential(nn.ReLU(), nn.Conv1d(64, 128, 1), nn.ReLU(), nn.Conv1d(128, horizon, 1))
        self.dropout = dropout

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor | None = None) -> torch.Tensor:
        if x_mask is None:
            x_mask = torch.ones_like(x)
        x = x * x_mask
        adjacency = torch.softmax(F.relu(self.node_a @ self.node_b), dim=-1)
        x = torch.einsum("ij,bjt->bit", adjacency, x)
        x_mask = torch.einsum("ij,bjt->bit", adjacency, x_mask)
        b, n, l = x.shape
        static = torch.cat([self.type_embedding(self.node_type), self.coord_projection(self.coords)], dim=-1)
        static = static.unsqueeze(0).unsqueeze(-1).expand(b, -1, -1, l)
        static = torch.einsum("ij,bjct->bict", adjacency, static)
        features = torch.cat([x.unsqueeze(2), x_mask.unsqueeze(2), static], dim=2)
        state = self.input_projection(features.reshape(b * n, 10, l)).reshape(b, n, -1, l)
        skip_total = None
        for convolution, skip in zip(self.temporal, self.skip):
            dilation = convolution.dilation[0]
            convolved = convolution(F.pad(state.reshape(b * n, state.shape[2], l), (2 * dilation, 0)))
            left, right = convolved.chunk(2, dim=1)
            convolved = (torch.tanh(left) * torch.sigmoid(right)).reshape(b, n, -1, l)
            state = state + F.dropout(torch.einsum("ij,bjct->bict", adjacency, convolved), self.dropout, self.training)
            current_skip = skip(state.reshape(b * n, state.shape[2], l))[..., -1:]
            skip_total = current_skip if skip_total is None else skip_total + current_skip
        forecast = self.end(skip_total).squeeze(-1).reshape(b, n, self.horizon)
        return forecast[:, : self.num_water_nodes]


class AutoTimes(nn.Module):
    """AutoTimes adapter around a frozen pretrained GPT-2 backbone.

    ``backbone`` may be a Hugging Face model name or local directory.  Weight
    download happens only when this baseline is instantiated.
    """

    def __init__(self, num_water_nodes: int, lookback: int, horizon: int,
                 backbone: str = "gpt2", token_length: int = 24,
                 width: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        try:
            from transformers import GPT2Model
        except ImportError as exc:
            raise ImportError("AutoTimes requires `pip install transformers`") from exc
        self.num_water_nodes = num_water_nodes
        self.token_length = token_length
        self.backbone = GPT2Model.from_pretrained(backbone)
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        embed_dim = self.backbone.config.n_embd
        # Encode values and observation masks into GPT-2 input embeddings.
        self.encoder = nn.Sequential(nn.Linear(2 * token_length, width), nn.GELU(), nn.Dropout(dropout), nn.Linear(width, embed_dim))
        tokens = math.ceil(lookback / token_length)
        self.decoder = nn.Sequential(nn.Linear(tokens * embed_dim, width), nn.GELU(), nn.Dropout(dropout), nn.Linear(width, horizon))

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor | None = None) -> torch.Tensor:
        if x_mask is not None:
            x = x * x_mask
        else:
            x_mask = torch.ones_like(x)
        water = x[:, : self.num_water_nodes]
        water_mask = x_mask[:, : self.num_water_nodes]
        b, n, l = water.shape
        pad = (-l) % self.token_length
        if pad:
            water = F.pad(water, (pad, 0))
            water_mask = F.pad(water_mask, (pad, 0))
        tokens = water.unfold(-1, self.token_length, self.token_length)
        mask_tokens = water_mask.unfold(-1, self.token_length, self.token_length)
        tokens = torch.cat([tokens, mask_tokens], dim=-1)
        embeddings = self.encoder(tokens).reshape(b * n, tokens.shape[2], -1)
        encoded = self.backbone(inputs_embeds=embeddings).last_hidden_state
        return self.decoder(encoded.reshape(b, n, -1))
