"""Model registry for the manuscript baselines and ablations."""

from __future__ import annotations

from torch import nn

from .baselines import (
    AutoTimes,
    FourierGNN,
    GraphWaveNet,
    ITransformer,
    MTGNN,
    NLinear,
    PatchTSTBaseline,
    TimesNet,
)
from .models import AnchoredDynamicGraphForecaster


MODEL_NAMES = (
    "ours", "nlinear", "patchtst", "itransformer", "timesnet",
    "fouriergnn", "mtgnn", "autotimes", "graphwavenet",
    "no_graph_correction", "no_event_context", "no_correction_bound",
    "no_event_context_bound",
    "fixed_graph", "without_neighbor_water", "without_rain", "without_well",
    "without_pump_gate", "without_nonwater",
)


def build_model(name: str, data, hidden_dim: int = 64, top_k: int = 20,
                dropout: float = 0.1, autotimes_backbone: str = "gpt2") -> nn.Module:
    """Build a model against one initialized ``SFBenchDataModule``."""

    common = {
        "num_nodes": data.metadata.num_nodes,
        "num_water_nodes": data.metadata.num_water_nodes,
        "lookback": data.train.lookback,
        "horizon": data.train.horizon,
    }
    metadata = {"node_type": data.metadata.node_type, "coords": data.metadata.coords}
    if name == "nlinear":
        return NLinear(common["num_water_nodes"], common["lookback"], common["horizon"])
    if name == "patchtst":
        return PatchTSTBaseline(common["num_water_nodes"], common["lookback"], common["horizon"])
    if name == "itransformer":
        return ITransformer(**common, **metadata, dropout=dropout)
    if name == "timesnet":
        return TimesNet(**common, dropout=dropout)
    if name == "fouriergnn":
        return FourierGNN(**common)
    if name == "mtgnn":
        return MTGNN(**common, dropout=0.3)
    if name == "autotimes":
        return AutoTimes(common["num_water_nodes"], common["lookback"], common["horizon"], backbone=autotimes_backbone)
    if name == "graphwavenet":
        return GraphWaveNet(**common, dropout=0.3)
    variant = name if name != "ours" else "ours"
    if variant not in MODEL_NAMES:
        raise ValueError(f"Unknown model {name!r}; choose from {MODEL_NAMES}")
    return AnchoredDynamicGraphForecaster(
        **common, **metadata, hidden_dim=hidden_dim, top_k=top_k,
        dropout=dropout, variant=variant,
    )


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def active_parameter_count(model: nn.Module, x, x_mask) -> int:
    """Count trainable parameters connected to the selected forward path."""

    was_training = model.training
    # cuDNN RNN backward is available only in training mode. Dropout changes
    # values, not graph connectivity, so it does not affect this count.
    model.train()
    model.zero_grad(set_to_none=True)
    model(x[:1], x_mask[:1]).sum().backward()
    count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    )
    model.zero_grad(set_to_none=True)
    model.train(was_training)
    return count
