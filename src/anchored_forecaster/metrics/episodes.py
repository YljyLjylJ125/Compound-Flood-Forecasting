"""Episode-level high-water and physical-unit forecast metrics."""

from __future__ import annotations

from collections.abc import Iterable
import math

import numpy as np
import torch


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else float("nan")


def _safe_mean(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else float("nan")


def station_thresholds(
    train_values: torch.Tensor,
    train_masks: torch.Tensor,
    quantile: float = 0.95,
) -> torch.Tensor:
    """Estimate one raw-unit threshold per station from valid training values."""

    thresholds = []
    for node in range(train_values.shape[0]):
        valid = train_values[node][train_masks[node].bool()]
        if valid.numel() == 0:
            thresholds.append(torch.tensor(float("nan"), dtype=train_values.dtype))
        else:
            thresholds.append(torch.quantile(valid.float(), quantile).to(train_values.dtype))
    return torch.stack(thresholds)


def _raw_runs(active: torch.Tensor) -> list[tuple[int, int]]:
    padded = torch.cat(
        [
            torch.zeros(1, dtype=torch.bool),
            active.bool().cpu(),
            torch.zeros(1, dtype=torch.bool),
        ]
    )
    changes = padded[1:].to(torch.int8) - padded[:-1].to(torch.int8)
    starts = torch.nonzero(changes == 1, as_tuple=False).flatten().tolist()
    ends = torch.nonzero(changes == -1, as_tuple=False).flatten().tolist()
    return list(zip(starts, ends))


def extract_episodes(
    values: torch.Tensor,
    valid_mask: torch.Tensor,
    threshold: float,
    min_duration: int = 3,
    merge_gap: int = 6,
) -> list[dict[str, float | int]]:
    """Extract and merge contiguous threshold exceedance episodes."""

    values = values.detach().float().cpu()
    valid_mask = valid_mask.detach().bool().cpu()
    active = (values >= float(threshold)) & valid_mask
    runs = _raw_runs(active)
    merged: list[list[int]] = []
    for start, end in runs:
        if merged and start - merged[-1][1] <= merge_gap:
            merged[-1][1] = end
        else:
            merged.append([start, end])

    episodes: list[dict[str, float | int]] = []
    for start, end in merged:
        segment_mask = valid_mask[start:end]
        segment_values = values[start:end]
        exceedance_count = int(((segment_values >= threshold) & segment_mask).sum())
        if exceedance_count < min_duration:
            continue
        valid_values = segment_values.masked_fill(~segment_mask, float("-inf"))
        peak_offset = int(valid_values.argmax())
        excess = torch.clamp(segment_values - float(threshold), min=0.0) * segment_mask.float()
        episodes.append(
            {
                "start": int(start),
                "end": int(end),
                "duration": int(end - start),
                "exceedance_duration": exceedance_count,
                "peak_index": int(start + peak_offset),
                "peak_value": float(segment_values[peak_offset]),
                "volume": float(excess.sum()),
            }
        )
    return episodes


def match_episodes(
    true_episodes: list[dict[str, float | int]],
    pred_episodes: list[dict[str, float | int]],
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Greedily maximize temporal overlap with one-to-one assignments."""

    candidates: list[tuple[int, int, int, int]] = []
    for true_idx, true_event in enumerate(true_episodes):
        for pred_idx, pred_event in enumerate(pred_episodes):
            overlap = min(int(true_event["end"]), int(pred_event["end"])) - max(
                int(true_event["start"]), int(pred_event["start"])
            )
            if overlap <= 0:
                continue
            onset_gap = abs(int(pred_event["start"]) - int(true_event["start"]))
            candidates.append((-overlap, onset_gap, true_idx, pred_idx))
    candidates.sort()

    used_true: set[int] = set()
    used_pred: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _neg_overlap, _onset_gap, true_idx, pred_idx in candidates:
        if true_idx in used_true or pred_idx in used_pred:
            continue
        used_true.add(true_idx)
        used_pred.add(pred_idx)
        matches.append((true_idx, pred_idx))
    misses = [idx for idx in range(len(true_episodes)) if idx not in used_true]
    false_alarms = [idx for idx in range(len(pred_episodes)) if idx not in used_pred]
    return matches, misses, false_alarms


def _episode_summary(rows: list[dict[str, object]]) -> dict[str, float]:
    true_count = sum(1 for row in rows if row["match_status"] in {"matched", "miss"})
    pred_count = sum(1 for row in rows if row["match_status"] in {"matched", "false_alarm"})
    matched = [row for row in rows if row["match_status"] == "matched"]
    tp = len(matched)
    fn = sum(1 for row in rows if row["match_status"] == "miss")
    fp = sum(1 for row in rows if row["match_status"] == "false_alarm")
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall) if math.isfinite(precision + recall) else float("nan")
    return {
        "episode_true_support": float(true_count),
        "episode_pred_support": float(pred_count),
        "episode_matched": float(tp),
        "episode_missed": float(fn),
        "episode_false_alarms": float(fp),
        "episode_precision": precision,
        "episode_pod": recall,
        "episode_far": _safe_div(fp, tp + fp),
        "episode_f1": f1,
        "episode_csi": _safe_div(tp, tp + fn + fp),
        "episode_miss_rate": _safe_div(fn, tp + fn),
        "episode_signed_onset_lag_h": _safe_mean(row["signed_onset_lag_h"] for row in matched),
        "episode_onset_mae_h": _safe_mean(abs(float(row["signed_onset_lag_h"])) for row in matched),
        "episode_peak_magnitude_error": _safe_mean(row["peak_magnitude_error"] for row in matched),
        "episode_peak_magnitude_mae": _safe_mean(abs(float(row["peak_magnitude_error"])) for row in matched),
        "episode_duration_bias_h": _safe_mean(row["duration_bias_h"] for row in matched),
        "episode_duration_mae_h": _safe_mean(abs(float(row["duration_bias_h"])) for row in matched),
        "episode_volume_error": _safe_mean(row["volume_error"] for row in matched),
        "episode_volume_mae": _safe_mean(abs(float(row["volume_error"])) for row in matched),
    }


def evaluate_episode_windows(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    thresholds: torch.Tensor,
    station_names: list[str] | tuple[str, ...],
    issue_stride: int = 24,
    min_duration: int = 3,
    merge_gap: int = 6,
) -> tuple[dict[str, float], list[dict[str, object]], list[dict[str, object]]]:
    """Evaluate one-to-one episodes on regularly sampled forecast issues."""

    pred = pred.detach().float().cpu()
    target = target.detach().float().cpu()
    mask = mask.detach().bool().cpu()
    thresholds = thresholds.detach().float().cpu()
    if pred.shape != target.shape or pred.shape != mask.shape:
        raise ValueError("pred, target and mask must have identical [samples,nodes,horizon] shapes")
    if thresholds.numel() != pred.shape[1]:
        raise ValueError("threshold count must equal the number of WATER nodes")

    details: list[dict[str, object]] = []
    for issue_idx in range(0, pred.shape[0], max(1, issue_stride)):
        for node in range(pred.shape[1]):
            threshold = float(thresholds[node])
            if not math.isfinite(threshold):
                continue
            true_events = extract_episodes(
                target[issue_idx, node], mask[issue_idx, node], threshold, min_duration, merge_gap
            )
            pred_events = extract_episodes(
                pred[issue_idx, node], mask[issue_idx, node], threshold, min_duration, merge_gap
            )
            matches, misses, false_alarms = match_episodes(true_events, pred_events)
            common = {
                "issue_index": issue_idx,
                "node": node,
                "station": station_names[node],
                "threshold": threshold,
            }
            for true_idx, pred_idx in matches:
                true_event = true_events[true_idx]
                pred_event = pred_events[pred_idx]
                details.append(
                    {
                        **common,
                        "match_status": "matched",
                        "true_start": true_event["start"],
                        "true_end": true_event["end"],
                        "pred_start": pred_event["start"],
                        "pred_end": pred_event["end"],
                        "signed_onset_lag_h": int(pred_event["start"]) - int(true_event["start"]),
                        "peak_magnitude_error": float(pred_event["peak_value"]) - float(true_event["peak_value"]),
                        "duration_bias_h": int(pred_event["duration"]) - int(true_event["duration"]),
                        "volume_error": float(pred_event["volume"]) - float(true_event["volume"]),
                    }
                )
            for true_idx in misses:
                true_event = true_events[true_idx]
                details.append(
                    {
                        **common,
                        "match_status": "miss",
                        "true_start": true_event["start"],
                        "true_end": true_event["end"],
                        "pred_start": "",
                        "pred_end": "",
                        "signed_onset_lag_h": float("nan"),
                        "peak_magnitude_error": float("nan"),
                        "duration_bias_h": float("nan"),
                        "volume_error": float("nan"),
                    }
                )
            for pred_idx in false_alarms:
                pred_event = pred_events[pred_idx]
                details.append(
                    {
                        **common,
                        "match_status": "false_alarm",
                        "true_start": "",
                        "true_end": "",
                        "pred_start": pred_event["start"],
                        "pred_end": pred_event["end"],
                        "signed_onset_lag_h": float("nan"),
                        "peak_magnitude_error": float("nan"),
                        "duration_bias_h": float("nan"),
                        "volume_error": float("nan"),
                    }
                )

    station_rows: list[dict[str, object]] = []
    for node, station in enumerate(station_names):
        node_rows = [row for row in details if int(row["node"]) == node]
        station_rows.append({"node": node, "station": station, **_episode_summary(node_rows)})
    summary = _episode_summary(details)
    for metric in (
        "episode_f1",
        "episode_csi",
        "episode_pod",
        "episode_far",
        "episode_miss_rate",
        "episode_onset_mae_h",
        "episode_peak_magnitude_mae",
        "episode_duration_mae_h",
        "episode_volume_mae",
    ):
        summary[f"{metric}_station_macro"] = _safe_mean(row[metric] for row in station_rows)
    summary["episode_issue_stride_h"] = float(issue_stride)
    summary["episode_min_duration_h"] = float(min_duration)
    summary["episode_merge_gap_h"] = float(merge_gap)
    return summary, details, station_rows


def physical_error_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    station_names: list[str] | tuple[str, ...],
) -> tuple[dict[str, float], list[dict[str, object]]]:
    """Compute micro and station-macro errors in the dataset's raw units."""

    pred = pred.detach().float().cpu()
    target = target.detach().float().cpu()
    mask = mask.detach().bool().cpu()
    station_rows: list[dict[str, object]] = []
    for node, station in enumerate(station_names):
        valid = mask[:, node, :]
        error = pred[:, node, :][valid] - target[:, node, :][valid]
        station_rows.append(
            {
                "node": node,
                "station": station,
                "support": int(error.numel()),
                "physical_mse": float(error.square().mean()) if error.numel() else float("nan"),
                "physical_mae": float(error.abs().mean()) if error.numel() else float("nan"),
                "physical_bias": float(error.mean()) if error.numel() else float("nan"),
            }
        )
    valid_all = mask
    error_all = pred[valid_all] - target[valid_all]
    station_mae = [float(row["physical_mae"]) for row in station_rows if math.isfinite(float(row["physical_mae"]))]
    summary = {
        "physical_mse_micro": float(error_all.square().mean()) if error_all.numel() else float("nan"),
        "physical_mae_micro": float(error_all.abs().mean()) if error_all.numel() else float("nan"),
        "physical_bias_micro": float(error_all.mean()) if error_all.numel() else float("nan"),
        "physical_mae_station_macro": _safe_mean(station_mae),
        "physical_mae_station_median": float(np.median(station_mae)) if station_mae else float("nan"),
        "physical_mae_station_p90": float(np.percentile(station_mae, 90)) if station_mae else float("nan"),
        "physical_mae_station_worst": max(station_mae) if station_mae else float("nan"),
    }
    return summary, station_rows
