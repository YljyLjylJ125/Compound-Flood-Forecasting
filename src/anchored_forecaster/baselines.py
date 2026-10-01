"""Dependency-light implementations of the paper's comparison models.

The release intentionally keeps the baselines in PyTorch instead of pulling in
large third-party forecasting frameworks.  Each class accepts the same
``[batch, nodes, lookback]`` tensor as the proposed model and returns forecasts
for WATER nodes only.  The architecture settings in ``BASELINE_CONFIGS`` are
recorded separately from the compact implementations so experiments remain
auditable and easy to replace with the original upstream implementations.
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn

from .models import PatchTSTAnchor


BASELINE_CONFIGS: Dict[str, Dict[str, object]] = {
    "nlinear": {"description": "last-value-centered linear map", "hidden_dim": 0},
    "patchtst": {"layers": 3, "hidden_dim": 128, "heads": 16, "patch_len": 16, "stride": 8, "revin": True, "dropout": 0.2},
    "itransformer": {"hidden_dim": 512, "heads": 8, "layers": 2, "ffn_dim": 2048, "dropout": 0.1},
    "timesnet": {"blocks": 2, "top_periods": 5, "hidden_dim": 32, "inception_kernels": 6, "dropout": 0.1},
    "fouriergnn": {"spectral_embedding": 256, "hidden_dim": 512, "stages": 3, "sparsity": 0.01},
    "mtgnn": {"layers": 3, "node_embedding": 40, "residual_channels": 32, "skip_channels": 64, "end_channels": 128, "dropout": 0.3},
    "autotimes": {"backbone": "frozen-gpt2-base", "token_length": 24, "width": 256, "layers": 2, "dropout": 0.1},
    "graphwavenet": {"blocks": 4, "dilation_layers": 4, "residual_channels": 32, "skip_channels": 256, "end_channels": 512, "dropout": 0.3},
}


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
    """Paper-sized PatchTST adapter using the shared channel-independent core."""

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


class _TokenForecaster(nn.Module):
    """Token encoder used by iTransformer and the compact multivariate controls."""

    def __init__(self, num_nodes: int, num_water_nodes: int, lookback: int, horizon: int,
                 node_type: torch.Tensor, coords: torch.Tensor, d_model: int = 128,
                 nhead: int = 8, layers: int = 2, ff_dim: int | None = None,
                 dropout: float = 0.1, full_info: bool = True) -> None:
        super().__init__()
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")
        self.num_nodes = num_nodes
        self.num_water_nodes = num_water_nodes
        self.full_info = full_info
        self.register_buffer("node_type", node_type.clone().long())
        coords = coords.float()
        coords = (coords - coords.mean(0, keepdim=True)) / coords.std(0, keepdim=True).clamp_min(1e-6)
        self.register_buffer("coords", coords)
        self.type_embedding = nn.Embedding(5, min(16, d_model // 4))
        self.coord_embedding = nn.Sequential(nn.Linear(2, min(16, d_model // 4)), nn.GELU())
        metadata_dim = self.type_embedding.embedding_dim + min(16, d_model // 4)
        self.token = nn.Linear((2 * lookback + metadata_dim) if full_info else lookback, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ff_dim or d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, horizon))

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor | None = None) -> torch.Tensor:
        if x_mask is None:
            x_mask = torch.ones_like(x)
        x = x * x_mask
        batch = x.shape[0]
        if self.full_info:
            meta = torch.cat([self.type_embedding(self.node_type), self.coord_embedding(self.coords)], dim=-1)
            meta = meta.unsqueeze(0).expand(batch, -1, -1)
            token_input = torch.cat([x, x_mask, meta], dim=-1)
        else:
            token_input = x
        tokens = self.token(token_input)
        encoded = self.encoder(tokens)
        return self.head(encoded[:, : self.num_water_nodes, :])


class ITransformer(_TokenForecaster):
    """Original value-only iTransformer baseline over the observed network."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("d_model", 512)
        kwargs.setdefault("nhead", 8)
        kwargs.setdefault("layers", 2)
        kwargs.setdefault("ff_dim", 2048)
        kwargs.setdefault("full_info", False)
        super().__init__(*args, **kwargs)


class GraphWaveNet(nn.Module):
    """Graph WaveNet-style adaptive graph and dilated temporal convolution."""

    def __init__(self, num_nodes: int, num_water_nodes: int, lookback: int, horizon: int,
                 channels: int = 32, dropout: float = 0.3) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.num_water_nodes = num_water_nodes
        self.node_a = nn.Parameter(torch.randn(num_nodes, 10) * 0.02)
        self.node_b = nn.Parameter(torch.randn(10, num_nodes) * 0.02)
        self.input_proj = nn.Conv1d(1, channels, kernel_size=1)
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
        features = x.unsqueeze(2)
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
    """Two-block, top-five-period TimesNet baseline."""

    def __init__(self, num_nodes: int, num_water_nodes: int, lookback: int, horizon: int,
                 hidden_dim: int = 32, blocks: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.num_water_nodes = num_water_nodes
        self.horizon = horizon
        self.input_projection = nn.Linear(num_nodes, hidden_dim)
        self.blocks = nn.ModuleList([_TimesBlock(hidden_dim, 5, 6, dropout) for _ in range(blocks)])
        self.normalization = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim * lookback, num_water_nodes * horizon)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor | None = None) -> torch.Tensor:
        if x_mask is not None:
            x = x * x_mask
        encoded = self.input_projection(x.transpose(1, 2))
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
    """Spectral graph baseline with learned complex frequency mixing."""

    def __init__(self, num_nodes: int, num_water_nodes: int, lookback: int, horizon: int,
                 spectral_embedding: int = 256, hidden_dim: int = 512,
                 stages: int = 3, sparsity: float = 0.01) -> None:
        super().__init__()
        self.num_water_nodes = num_water_nodes
        self.sparsity = sparsity
        dimensions = [num_nodes, spectral_embedding] + [hidden_dim] * max(0, stages - 2) + [spectral_embedding]
        self.stages = nn.ModuleList([_ComplexLinear(dimensions[index], dimensions[index + 1]) for index in range(len(dimensions) - 1)])
        self.output_projection = _ComplexLinear(spectral_embedding, num_water_nodes)
        self.head = nn.Linear(lookback, horizon)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor | None = None) -> torch.Tensor:
        if x_mask is not None:
            x = x * x_mask
        spectrum = torch.fft.rfft(x, dim=-1).transpose(1, 2)
        for stage in self.stages:
            spectrum = stage(spectrum)
            spectrum = torch.complex(F.gelu(spectrum.real), F.gelu(spectrum.imag))
            spectrum = torch.where(spectrum.abs() >= self.sparsity, spectrum, torch.zeros_like(spectrum))
        spectrum = self.output_projection(spectrum).transpose(1, 2)
        temporal = torch.fft.irfft(spectrum, n=x.shape[-1], dim=-1)
        return self.head(temporal)


class MTGNN(nn.Module):
    """Learned-graph temporal convolution baseline."""

    def __init__(self, num_nodes: int, num_water_nodes: int, lookback: int, horizon: int,
                 layers: int = 3, residual_channels: int = 32, dropout: float = 0.3) -> None:
        super().__init__()
        self.num_water_nodes = num_water_nodes
        self.horizon = horizon
        self.node_a = nn.Parameter(torch.randn(num_nodes, 40) * 0.02)
        self.node_b = nn.Parameter(torch.randn(40, num_nodes) * 0.02)
        self.temporal = nn.ModuleList()
        self.skip = nn.ModuleList()
        for layer in range(layers):
            dilation = 2 ** layer
            self.temporal.append(nn.Conv1d(residual_channels, residual_channels * 2, 3, dilation=dilation))
            self.skip.append(nn.Conv1d(residual_channels, 64, 1))
        self.input_projection = nn.Conv1d(1, residual_channels, 1)
        self.end = nn.Sequential(nn.ReLU(), nn.Conv1d(64, 128, 1), nn.ReLU(), nn.Conv1d(128, horizon, 1))
        self.dropout = dropout

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor | None = None) -> torch.Tensor:
        if x_mask is not None:
            x = x * x_mask
        adjacency = torch.softmax(F.relu(self.node_a @ self.node_b), dim=-1)
        x = torch.einsum("ij,bjt->bit", adjacency, x)
        b, n, l = x.shape
        state = self.input_projection(x.reshape(b * n, 1, l)).reshape(b, n, -1, l)
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
        self.encoder = nn.Sequential(nn.Linear(token_length, width), nn.GELU(), nn.Dropout(dropout), nn.Linear(width, embed_dim))
        tokens = math.ceil(lookback / token_length)
        self.decoder = nn.Sequential(nn.Linear(tokens * embed_dim, width), nn.GELU(), nn.Dropout(dropout), nn.Linear(width, horizon))

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor | None = None) -> torch.Tensor:
        if x_mask is not None:
            x = x * x_mask
        water = x[:, : self.num_water_nodes]
        b, n, l = water.shape
        pad = (-l) % self.token_length
        if pad:
            water = F.pad(water, (pad, 0))
        tokens = water.unfold(-1, self.token_length, self.token_length)
        embeddings = self.encoder(tokens).reshape(b * n, tokens.shape[2], -1)
        encoded = self.backbone(inputs_embeds=embeddings).last_hidden_state
        return self.decoder(encoded.reshape(b, n, -1))
