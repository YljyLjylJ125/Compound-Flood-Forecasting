#!/usr/bin/env python3
"""Evaluate a released-model checkpoint without retraining."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from anchored_forecaster import AnchoredDynamicGraphForecaster, SFBenchDataModule
from anchored_forecaster.evaluation import evaluate_model, write_rows
from anchored_forecaster.reproducibility import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="S_7", choices=[f"S_{index}" for index in range(8)])
    parser.add_argument("--part", default=0, type=int, choices=[0, 1, 2])
    parser.add_argument("--lookback", default="2D")
    parser.add_argument("--horizon", default="3D")
    parser.add_argument("--span", default="0H")
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--norm-clip", default=20.0, type=float)
    parser.add_argument("--episode-quantile", default=0.95, type=float)
    parser.add_argument("--episode-issue-stride", default=24, type=int)
    parser.add_argument("--episode-min-duration", default=3, type=int)
    parser.add_argument("--episode-merge-gap", default=6, type=int)
    parser.add_argument("--seed", default=2025, type=int)
    parser.add_argument("--device", default="")
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--max-eval-batches", default=None, type=int)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    data = SFBenchDataModule(
        args.dataset_root,
        split=args.split,
        part=args.part,
        lookback=args.lookback,
        horizon=args.horizon,
        span=args.span,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        norm_clip=args.norm_clip,
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint["model_config"]
    if config["num_nodes"] != data.metadata.num_nodes or config["num_water_nodes"] != data.metadata.num_water_nodes:
        raise ValueError("Checkpoint node layout does not match the requested split and station partition")
    if config["lookback"] != data.train.lookback or config["horizon"] != data.train.horizon:
        raise ValueError("Checkpoint window configuration does not match the requested evaluation")
    model = AnchoredDynamicGraphForecaster(
        num_nodes=config["num_nodes"],
        num_water_nodes=config["num_water_nodes"],
        lookback=config["lookback"],
        horizon=config["horizon"],
        node_type=data.metadata.node_type,
        coords=data.metadata.coords,
        hidden_dim=config["hidden_dim"],
        top_k=config["top_k"],
        dropout=config["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    metrics, episode_rows, station_episode_rows, physical_rows = evaluate_model(
        model,
        data,
        "test",
        device,
        quantile=args.episode_quantile,
        issue_stride=args.episode_issue_stride,
        min_duration=args.episode_min_duration,
        merge_gap=args.episode_merge_gap,
        max_batches=args.max_eval_batches,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(output_dir / "episode_matches.csv", episode_rows)
    write_rows(output_dir / "episode_by_station.csv", station_episode_rows)
    write_rows(output_dir / "physical_metrics_by_station.csv", physical_rows)
    payload = {"checkpoint": str(args.checkpoint), "config": vars(args), "test_metrics": metrics}
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main(parse_args())
