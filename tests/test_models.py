from types import ModuleType, SimpleNamespace
import sys

import torch

from anchored_forecaster import build_model


def fake_data(nodes=12, water=4, lookback=48, horizon=72):
    metadata = SimpleNamespace(
        num_nodes=nodes,
        num_water_nodes=water,
        node_type=torch.tensor([0] * water + [1, 1, 2, 2, 3, 4, 4, 4][: nodes - water]),
        coords=torch.randn(nodes, 2),
    )
    train = SimpleNamespace(lookback=lookback, horizon=horizon)
    return SimpleNamespace(metadata=metadata, train=train)


def test_paper_models_and_ablations_share_output_contract():
    data = fake_data()
    x = torch.randn(2, 12, 48)
    mask = torch.ones_like(x)
    names = [
        "ours", "nlinear", "patchtst", "itransformer", "timesnet",
        "fouriergnn", "mtgnn", "graphwavenet", "fixed_graph",
        "no_event_context", "no_correction_bound", "no_event_context_bound",
        "no_graph_correction", "without_neighbor_water", "without_rain",
        "without_well", "without_pump_gate", "without_nonwater",
    ]
    for name in names:
        prediction = build_model(name, data)(x, mask)
        assert prediction.shape == (2, 4, 72), name
        assert torch.isfinite(prediction).all(), name


def test_bounded_correction_and_anchor_only_ablation():
    data = fake_data()
    x = torch.randn(2, 12, 48)
    mask = torch.ones_like(x)
    ours = build_model("ours", data)
    outputs = ours.forward_with_diagnostics(x, mask)
    assert torch.all(outputs["correction"].abs() <= outputs["correction_budget"] + 1e-7)
    assert torch.allclose(outputs["prediction"], outputs["anchor_prediction"] + outputs["correction"])

    anchor_only = build_model("no_graph_correction", data)
    outputs = anchor_only.forward_with_diagnostics(x, mask)
    assert torch.equal(outputs["correction"], torch.zeros_like(outputs["correction"]))
    assert torch.allclose(outputs["prediction"], outputs["anchor_prediction"])


def test_source_ablation_does_not_use_removed_source_values():
    data = fake_data()
    x = torch.randn(2, 12, 48)
    mask = torch.ones_like(x)
    cases = {
        "without_neighbor_water": ([0], slice(1, None)),
        "without_rain": ([4, 5], slice(None)),
        "without_well": ([6, 7], slice(None)),
        "without_pump_gate": ([8, 9, 10, 11], slice(None)),
        "without_nonwater": (list(range(4, 12)), slice(None)),
    }
    for name, (removed_nodes, checked_targets) in cases.items():
        model = build_model(name, data).eval()
        torch.nn.init.normal_(model.graph_decoder[-1].weight)
        torch.nn.init.normal_(model.graph_decoder[-1].bias)
        perturbed = x.clone()
        perturbed[:, removed_nodes] += 100.0 * torch.randn_like(perturbed[:, removed_nodes])
        with torch.no_grad():
            baseline = model(x, mask)
            changed = model(perturbed, mask)
        assert torch.allclose(baseline[:, checked_targets], changed[:, checked_targets], atol=1e-6), name


def test_masked_values_never_change_predictions():
    data = fake_data()
    x = torch.randn(2, 12, 48)
    mask = torch.ones_like(x)
    mask[:, :, -3:] = 0
    mask[:, 5:8, 10:20] = 0
    perturbed = x.clone()
    perturbed[~mask.bool()] = 1e4 * torch.randn_like(perturbed[~mask.bool()])
    for name in ("ours", "nlinear", "patchtst", "itransformer", "graphwavenet"):
        model = build_model(name, data).eval()
        with torch.no_grad():
            baseline = model(x, mask)
            changed = model(perturbed, mask)
        assert torch.allclose(baseline, changed, atol=1e-6), name


def test_fair_input_adapters_forward_backward_and_mask_sensitivity():
    data = fake_data()
    x = torch.randn(2, 12, 48)
    mask = torch.ones_like(x)
    mask[:, 2, -3:] = 0
    adapted = ["itransformer", "timesnet", "fouriergnn", "mtgnn", "graphwavenet"]
    for name in adapted:
        model = build_model(name, data).train()
        prediction = model(x, mask)
        assert prediction.shape == (2, 4, 72), name
        assert torch.isfinite(prediction).all(), name
        prediction.sum().backward()
        assert any(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad), name
        model.eval()
        with torch.no_grad():
            masked_prediction = model(x, mask)
            unmasked_prediction = model(x, torch.ones_like(mask))
        assert not torch.allclose(masked_prediction, unmasked_prediction), name


def test_fair_input_adapters_use_context_and_metadata():
    data = fake_data()
    x = torch.randn(2, 12, 48)
    mask = torch.ones_like(x)
    for name in ("itransformer", "timesnet", "fouriergnn", "mtgnn", "graphwavenet"):
        model = build_model(name, data).eval()
        with torch.no_grad():
            baseline = model(x, mask)
            context_changed = x.clone()
            context_changed[:, 7, -3:] += 5.0
            context_prediction = model(context_changed, mask)
            assert not torch.allclose(baseline, context_prediction), name
            metadata = model.node_adapter if hasattr(model, "node_adapter") else model
            metadata.coords[7, 0] += 3.0
            metadata.type_embedding.weight[1, 0] += 3.0
            metadata_prediction = model(x, mask)
        assert not torch.allclose(baseline, metadata_prediction), name


def test_autotimes_mask_adapter_without_downloading_backbone(monkeypatch):
    class TinyBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(n_embd=16)

        def forward(self, inputs_embeds):
            return SimpleNamespace(last_hidden_state=inputs_embeds + 0.1)

    transformers = ModuleType("transformers")
    transformers.GPT2Model = SimpleNamespace(from_pretrained=lambda *args, **kwargs: TinyBackbone())
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    data = fake_data()
    x = torch.randn(2, 12, 48)
    mask = torch.ones_like(x)
    mask[:, 2, -3:] = 0
    model = build_model("autotimes", data).eval()
    with torch.no_grad():
        masked_prediction = model(x, mask)
        unmasked_prediction = model(x, torch.ones_like(mask))
    assert masked_prediction.shape == (2, 4, 72)
    assert torch.isfinite(masked_prediction).all()
    assert not torch.allclose(masked_prediction, unmasked_prediction)
