#!/usr/bin/env python3
"""Train a manuscript model or ablation under the shared paper protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from anchored_forecaster import MODEL_NAMES, SFBenchDataModule, active_parameter_count, build_model, trainable_parameter_count
from anchored_forecaster.baselines import BASELINE_CONFIGS
from anchored_forecaster.evaluation import (
    collect_predictions, evaluate_predictions, move_batch,
    save_prediction_archive, write_rows,
)
from anchored_forecaster.metrics import masked_mse


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


@torch.no_grad()
def normalized_validation_mse(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    max_batches: int | None = None,
) -> float:
    """Select checkpoints in the standardized target space used for training."""

    model.eval()
    squared_error = 0.0
    valid_count = 0.0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        device_batch = move_batch(batch, device)
        prediction = model(device_batch["x"], device_batch["x_mask"])
        mask = device_batch["y_mask"]
        squared_error += float((((prediction - device_batch["y"]) ** 2) * mask).sum().cpu())
        valid_count += float(mask.sum().cpu())
    if valid_count <= 0:
        raise RuntimeError("Validation contains no observed WATER targets")
    return squared_error / valid_count


def train(args: argparse.Namespace) -> dict[str, object]:
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
    model = build_model(
        args.model, data, hidden_dim=args.hidden_dim, top_k=args.top_k,
        dropout=args.dropout, autotimes_backbone=args.autotimes_backbone,
    ).to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    train_loader = data.loader("train", shuffle=True)
    best_validation_mse = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_rows: list[dict[str, object]] = []

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

        validation_mse = normalized_validation_mse(
            model, data.loader("val", shuffle=False), device,
            max_batches=args.max_eval_batches,
        )
        train_loss = total_loss / max(batches, 1)
        history_rows.append({"epoch": epoch, "train_masked_mse_normalized": train_loss, "validation_mse_normalized": validation_mse})
        print(
            f"epoch={epoch:03d} train_masked_mse={train_loss:.6f} "
            f"validation_mse_normalized={validation_mse:.6f}",
            flush=True,
        )
        if validation_mse < best_validation_mse:
            best_validation_mse = validation_mse
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if args.early_stopping_patience and epochs_without_improvement >= args.early_stopping_patience:
            print(f"early_stop_epoch={epoch} best_validation_mse_normalized={best_validation_mse:.6f}", flush=True)
            break

    if best_state is None:
        raise RuntimeError("No checkpoint was selected; verify the validation data and metric")
    model.load_state_dict(best_state)
    parameter_batch = move_batch(next(iter(data.loader("train", shuffle=False))), device)
    active_parameters = active_parameter_count(model, parameter_batch["x"], parameter_batch["x_mask"])
    model_config = {
        "model_name": args.model,
        "num_nodes": data.metadata.num_nodes,
        "num_water_nodes": data.metadata.num_water_nodes,
        "lookback": data.train.lookback,
        "horizon": data.train.horizon,
        "hidden_dim": args.hidden_dim,
        "top_k": args.top_k,
        "dropout": args.dropout,
        "trainable_parameters": trainable_parameter_count(model),
        "active_trainable_parameters": active_parameters,
        "architecture_config": BASELINE_CONFIGS.get(
            args.model,
            {
                "hidden_dim": args.hidden_dim,
                "dynamic_top_k": args.top_k,
                "dropout": args.dropout,
                "anchor": "channel-independent latest-valid-centered PatchTST",
            },
        ),
    }
    checkpoint_path = output_dir / "best_model.pt"
    torch.save({"state_dict": model.state_dict(), "model_config": model_config, "args": vars(args)}, checkpoint_path)

    write_rows(output_dir / "training_history.csv", history_rows)
    val_prediction, val_target, val_mask = collect_predictions(
        model, data.loader("val", shuffle=False), data.val, device,
        max_batches=args.max_eval_batches,
    )
    validation_metrics, _, _, _ = evaluate_predictions(
        val_prediction, val_target, val_mask, data,
        quantile=args.episode_quantile,
        issue_stride=args.episode_issue_stride,
        min_duration=args.episode_min_duration,
        merge_gap=args.episode_merge_gap,
        dataset=data.val,
    )
    test_prediction, test_target, test_mask = collect_predictions(
        model, data.loader("test", shuffle=False), data.test, device,
        max_batches=args.max_eval_batches,
    )
    test_metrics, episode_rows, station_episode_rows, physical_rows = evaluate_predictions(
        test_prediction, test_target, test_mask, data,
        quantile=args.episode_quantile,
        issue_stride=args.episode_issue_stride,
        min_duration=args.episode_min_duration,
        merge_gap=args.episode_merge_gap,
        dataset=data.test,
    )
    if not args.no_save_predictions:
        save_prediction_archive(
            output_dir / "test_predictions.npz", test_prediction, test_target,
            test_mask, data.test, issue_stride=args.episode_issue_stride,
        )
    write_rows(output_dir / "episode_matches.csv", episode_rows)
    write_rows(output_dir / "episode_by_station.csv", station_episode_rows)
    write_rows(output_dir / "physical_metrics_by_station.csv", physical_rows)
    run_row = {
        "model": args.model,
        "variant": args.model,
        "split": args.split.replace("_", ""),
        "part": args.part,
        "repeat_id": args.repeat_id,
        "horizon": data.train.horizon,
        "q": args.episode_quantile,
        "mae": test_metrics["mae"],
        "mse": test_metrics["mse"],
        "episode_f1": test_metrics["episode_f1"],
        "onset_mae": test_metrics["episode_onset_mae_h"],
        "peak_mae": test_metrics["episode_peak_magnitude_mae"],
        "duration_mae": test_metrics["episode_duration_mae_h"],
        "checkpoint_id": str(checkpoint_path),
        "prediction_version": "test_predictions.npz" if not args.no_save_predictions else "not_saved",
        "evaluator_version": "release-v3-daily-issues",
        "model_data_contract_version": "paper-strict-mask-aware-v5",
        "trainable_parameters": trainable_parameter_count(model),
        "active_trainable_parameters": active_parameters,
    }
    write_rows(output_dir / "run_metrics.csv", [run_row])
    result = {
        "checkpoint": str(checkpoint_path),
        "config": vars(args),
        "checkpoint_selection": {
            "metric": "masked_mse",
            "space": "normalized_training_target",
            "best_validation_value": best_validation_mse,
        },
        "model": model_config,
        "data": {
            "processed_root": str(data.processed_root),
            "train_windows": len(data.train),
            "validation_windows": len(data.val),
            "test_windows": len(data.test),
        },
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "run_row": run_row,
    }
    write_json(output_dir / "metrics.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, help="SFBench directory containing Processed_hour/")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="ours", choices=MODEL_NAMES)
    parser.add_argument("--split", default="S_7", choices=["S_5", "S_6", "S_7"])
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
    parser.add_argument("--repeat-id", default=1, type=int, help="Identifier for this repeated experiment")
    parser.add_argument("--device", default="")
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--max-train-batches", default=None, type=int)
    parser.add_argument("--max-eval-batches", default=None, type=int)
    parser.add_argument("--autotimes-backbone", default="gpt2")
    parser.add_argument("--no-save-predictions", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
