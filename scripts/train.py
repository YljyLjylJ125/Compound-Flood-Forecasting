#!/usr/bin/env python3
"""Train the released pure-PatchTST anchored dynamic-graph forecaster."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from anchored_forecaster import AnchoredDynamicGraphForecaster, SFBenchDataModule
from anchored_forecaster.evaluation import evaluate_model, move_batch, write_rows
from anchored_forecaster.metrics import masked_mse
from anchored_forecaster.reproducibility import seed_everything


def build_model(args: argparse.Namespace, data: SFBenchDataModule) -> AnchoredDynamicGraphForecaster:
    return AnchoredDynamicGraphForecaster(
        num_nodes=data.metadata.num_nodes,
        num_water_nodes=data.metadata.num_water_nodes,
        lookback=data.train.lookback,
        horizon=data.train.horizon,
        node_type=data.metadata.node_type,
        coords=data.metadata.coords,
        hidden_dim=args.hidden_dim,
        top_k=args.top_k,
        dropout=args.dropout,
    )


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def train(args: argparse.Namespace) -> dict[str, object]:
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
    model = build_model(args, data).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = data.loader("train", shuffle=True)
    best_validation_mse = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        for batch_index, batch in enumerate(train_loader):
            if args.max_train_batches is not None and batch_index >= args.max_train_batches:
                break
            batch = move_batch(batch, device)
            prediction = model(batch["x"], batch["x_mask"])
            # Event labels and episode metrics are intentionally excluded here.
            loss = masked_mse(prediction, batch["y"], batch["y_mask"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            batches += 1

        validation_metrics, _, _, _ = evaluate_model(
            model,
            data,
            "val",
            device,
            quantile=args.episode_quantile,
            issue_stride=args.episode_issue_stride,
            min_duration=args.episode_min_duration,
            merge_gap=args.episode_merge_gap,
            max_batches=args.max_eval_batches,
        )
        validation_mse = float(validation_metrics["mse"])
        train_loss = total_loss / max(batches, 1)
        print(
            f"epoch={epoch:03d} train_masked_mse={train_loss:.6f} "
            f"validation_mse_raw={validation_mse:.6f}",
            flush=True,
        )
        if validation_mse < best_validation_mse:
            best_validation_mse = validation_mse
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if args.early_stopping_patience and epochs_without_improvement >= args.early_stopping_patience:
            print(f"early_stop_epoch={epoch} best_validation_mse={best_validation_mse:.6f}", flush=True)
            break

    if best_state is None:
        raise RuntimeError("No checkpoint was selected; verify the validation data and metric")
    model.load_state_dict(best_state)
    model_config = {
        "num_nodes": data.metadata.num_nodes,
        "num_water_nodes": data.metadata.num_water_nodes,
        "lookback": data.train.lookback,
        "horizon": data.train.horizon,
        "hidden_dim": args.hidden_dim,
        "top_k": args.top_k,
        "dropout": args.dropout,
    }
    checkpoint_path = output_dir / "best_model.pt"
    torch.save({"state_dict": model.state_dict(), "model_config": model_config, "args": vars(args)}, checkpoint_path)

    validation_metrics, _, _, _ = evaluate_model(
        model,
        data,
        "val",
        device,
        quantile=args.episode_quantile,
        issue_stride=args.episode_issue_stride,
        min_duration=args.episode_min_duration,
        merge_gap=args.episode_merge_gap,
        max_batches=args.max_eval_batches,
    )
    test_metrics, episode_rows, station_episode_rows, physical_rows = evaluate_model(
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
    write_rows(output_dir / "episode_matches.csv", episode_rows)
    write_rows(output_dir / "episode_by_station.csv", station_episode_rows)
    write_rows(output_dir / "physical_metrics_by_station.csv", physical_rows)
    result = {
        "checkpoint": str(checkpoint_path),
        "config": vars(args),
        "model": model_config,
        "data": {
            "processed_root": str(data.processed_root),
            "train_windows": len(data.train),
            "validation_windows": len(data.val),
            "test_windows": len(data.test),
        },
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
    }
    write_json(output_dir / "metrics.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, help="SFBench directory containing Processed_hour/")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="S_7", choices=[f"S_{index}" for index in range(8)])
    parser.add_argument("--part", default=0, type=int, choices=[0, 1, 2])
    parser.add_argument("--lookback", default="2D")
    parser.add_argument("--horizon", default="3D")
    parser.add_argument("--span", default="0H")
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--early-stopping-patience", default=0, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--weight-decay", default=1e-5, type=float)
    parser.add_argument("--hidden-dim", default=64, type=int)
    parser.add_argument("--top-k", default=20, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--grad-clip", default=1.0, type=float)
    parser.add_argument("--norm-clip", default=20.0, type=float)
    parser.add_argument("--episode-quantile", default=0.95, type=float)
    parser.add_argument("--episode-issue-stride", default=24, type=int)
    parser.add_argument("--episode-min-duration", default=3, type=int)
    parser.add_argument("--episode-merge-gap", default=6, type=int)
    parser.add_argument("--seed", default=2025, type=int)
    parser.add_argument("--device", default="")
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--max-train-batches", default=None, type=int)
    parser.add_argument("--max-eval-batches", default=None, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
