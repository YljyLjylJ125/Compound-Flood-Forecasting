"""Forecasting metrics for water-level prediction."""

from __future__ import annotations

from typing import Dict

import torch


def _safe_mask(mask: torch.Tensor | None, target: torch.Tensor) -> torch.Tensor:
    if mask is None:
        return torch.ones_like(target)
    return mask.to(device=target.device, dtype=target.dtype)


def masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    mask_t = _safe_mask(mask, target)
    return (torch.abs(pred - target) * mask_t).sum() / mask_t.sum().clamp_min(1.0)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    mask_t = _safe_mask(mask, target)
    return (((pred - target) ** 2) * mask_t).sum() / mask_t.sum().clamp_min(1.0)


def masked_rmse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    return torch.sqrt(masked_mse(pred, target, mask).clamp_min(0.0))


def nse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    mask_t = _safe_mask(mask, target)
    numerator = (((pred - target) ** 2) * mask_t).sum()
    target_mean = (target * mask_t).sum() / mask_t.sum().clamp_min(1.0)
    denominator = (((target - target_mean) ** 2) * mask_t).sum().clamp_min(1e-8)
    return 1.0 - numerator / denominator


def kge(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    mask_t = _safe_mask(mask, target).bool()
    pred_v = pred[mask_t]
    target_v = target[mask_t]
    if pred_v.numel() < 2:
        return torch.tensor(float("nan"), device=pred.device)
    pred_mean = pred_v.mean()
    target_mean = target_v.mean()
    pred_std = pred_v.std().clamp_min(1e-8)
    target_std = target_v.std().clamp_min(1e-8)
    corr = torch.corrcoef(torch.stack([pred_v, target_v]))[0, 1].nan_to_num(0.0)
    alpha = pred_std / target_std
    beta = pred_mean / target_mean.clamp_min(1e-8)
    return 1.0 - torch.sqrt((corr - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)


def peak_magnitude_error(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    mask_t = _safe_mask(mask, target).bool()
    valid_series = mask_t.any(dim=-1)
    if not bool(valid_series.any()):
        return torch.tensor(float("nan"), device=target.device)
    pred_peak = pred.masked_fill(~mask_t, float("-inf")).max(dim=-1).values
    target_peak = target.masked_fill(~mask_t, float("-inf")).max(dim=-1).values
    return torch.abs(pred_peak[valid_series] - target_peak[valid_series]).mean()


def event_detection_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    threshold: float | torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    """Threshold-event detection metrics.

    The threshold should be derived from the train period or from an external
    hydrological standard. Passing a test-derived threshold would be leakage.
    """

    if threshold is None:
        return {}
    mask_t = _safe_mask(mask, target).bool()
    threshold_t = torch.as_tensor(threshold, device=target.device, dtype=target.dtype)
    target_event = (target >= threshold_t) & mask_t
    pred_event = (pred >= threshold_t) & mask_t
    tp = (pred_event & target_event).sum().float()
    fp = (pred_event & ~target_event & mask_t).sum().float()
    fn = (~pred_event & target_event & mask_t).sum().float()
    precision = tp / (tp + fp).clamp_min(1.0)
    recall = tp / (tp + fn).clamp_min(1.0)
    target_event_node = target_event.any(dim=-1)
    pred_event_node = pred_event.any(dim=-1)
    both_event_node = target_event_node & pred_event_node
    target_cross = target_event.float().argmax(dim=-1).float()
    pred_cross = pred_event.float().argmax(dim=-1).float()
    first_crossing_error = (
        torch.abs(pred_cross[both_event_node] - target_cross[both_event_node]).mean()
        if bool(both_event_node.any())
        else torch.tensor(float("nan"), device=target.device)
    )
    return {
        "pod": recall,
        "far": fp / (tp + fp).clamp_min(1.0),
        "csi": tp / (tp + fp + fn).clamp_min(1.0),
        "f1_event": 2 * precision * recall / (precision + recall).clamp_min(1e-8),
        "event_support": target_event.sum().float(),
        "event_node_support": target_event_node.sum().float(),
        "first_crossing_timing_error": first_crossing_error,
        "first_crossing_hit_rate": both_event_node.sum().float() / target_event_node.sum().float().clamp_min(1.0),
    }


@torch.no_grad()
def forecast_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    extreme_threshold: float | torch.Tensor | None = None,
) -> Dict[str, float]:
    metrics = {
        "mae": masked_mae(pred, target, mask),
        "mse": masked_mse(pred, target, mask),
        "rmse": masked_rmse(pred, target, mask),
        "nse": nse(pred, target, mask),
        "kge": kge(pred, target, mask),
        "peak_magnitude_error": peak_magnitude_error(pred, target, mask),
    }
    metrics.update(event_detection_metrics(pred, target, mask, extreme_threshold))
    return {key: float(value.detach().cpu()) for key, value in metrics.items()}
