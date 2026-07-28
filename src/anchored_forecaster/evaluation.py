"""Shared evaluation helpers for raw-stage and episode-level reporting."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import torch
from torch import nn

from .data import SFBenchDataModule
from .metrics import evaluate_episode_windows, forecast_metrics, physical_error_metrics, station_thresholds


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device, non_blocking=True) for name, value in batch.items()}


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: Iterable[dict[str, torch.Tensor]],
    dataset,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return all predictions, targets, and masks in original WATER-stage units."""

    model.eval()
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        device_batch = move_batch(batch, device)
        prediction_normalized = model(device_batch["x"], device_batch["x_mask"])
        predictions.append(dataset.denormalize_water(prediction_normalized.detach().cpu()))
        targets.append(batch["y_raw"].detach().cpu())
        masks.append(batch["y_mask"].detach().cpu())
    if not predictions:
        raise RuntimeError("Evaluation loader produced no batches")
    return torch.cat(predictions), torch.cat(targets), torch.cat(masks)


def water_station_names(data: SFBenchDataModule) -> list[str]:
    return [data.metadata.station_names[int(index)] for index in data.metadata.water_indices.tolist()]


def evaluate_model(
    model: nn.Module,
    data: SFBenchDataModule,
    phase: str,
    device: torch.device,
    quantile: float = 0.95,
    issue_stride: int = 24,
    min_duration: int = 3,
    merge_gap: int = 6,
    max_batches: int | None = None,
) -> tuple[dict[str, float], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """Evaluate forecast and episode metrics for one split phase.

    Per-station high-water thresholds are estimated from valid training values
    only.  Episode labels are created after forecasting and are not used by the
    training objective.
    """

    dataset = {"train": data.train, "val": data.val, "test": data.test}[phase]
    predictions, targets, masks = collect_predictions(
        model,
        data.loader(phase, shuffle=False),
        dataset,
        device,
        max_batches=max_batches,
    )
    water_indices = data.metadata.water_indices
    thresholds = station_thresholds(
        data.train.values[water_indices].float(),
        data.train.masks[water_indices].bool(),
        quantile=quantile,
    )
    names = water_station_names(data)
    record_metrics = forecast_metrics(predictions, targets, masks)
    episode_metrics, episode_rows, station_episode_rows = evaluate_episode_windows(
        predictions,
        targets,
        masks,
        thresholds,
        names,
        issue_stride=issue_stride,
        min_duration=min_duration,
        merge_gap=merge_gap,
    )
    physical_metrics, physical_rows = physical_error_metrics(predictions, targets, masks, names)
    metrics = {
        **record_metrics,
        **episode_metrics,
        **physical_metrics,
        "episode_quantile": float(quantile),
        "episode_issue_stride_h": float(issue_stride),
        "episode_min_duration_h": float(min_duration),
        "episode_merge_gap_h": float(merge_gap),
    }
    return metrics, episode_rows, station_episode_rows, physical_rows


def write_rows(path: str | Path, rows: list[dict[str, object]]) -> None:
    """Write a UTF-8 CSV with the union of fields in order of first appearance."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
