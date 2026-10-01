#!/usr/bin/env python3
"""Recompute the manuscript S7/3D threshold-sensitivity experiment."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anchored_forecaster.data import SFBenchDataModule  # noqa: E402
from anchored_forecaster.evaluation import evaluate_predictions  # noqa: E402


PAPER_MODELS = {
    "ours", "nlinear", "patchtst", "itransformer", "timesnet",
    "fouriergnn", "mtgnn", "autotimes", "graphwavenet",
}


def parse_run(path: Path) -> dict[str, object]:
    pieces = path.parts
    h_index = len(pieces) - 2
    return {
        "model": pieces[h_index - 4],
        "split": pieces[h_index - 3],
        "part": int(pieces[h_index - 2].split("_")[1]),
        "seed": int(pieces[h_index - 1].split("_")[1]),
        "horizon": int(pieces[h_index][1:]),
    }


def aggregate_thresholds(raw: pd.DataFrame) -> pd.DataFrame:
    metrics = ("episode_f1", "onset_mae", "peak_mae", "duration_mae")
    long = raw.melt(
        id_vars=["model", "split", "part", "seed", "horizon", "q"],
        value_vars=list(metrics),
        var_name="metric",
        value_name="value",
    )
    per_part = (
        long.groupby(["model", "split", "part", "horizon", "q", "metric"], dropna=False)["value"]
        .agg(seed_mean="mean", n_seeds="count")
        .reset_index()
    )
    return (
        per_part.groupby(["model", "split", "horizon", "q", "metric"], dropna=False)["seed_mean"]
        .agg(mean_across_parts="mean", std_across_parts=lambda values: values.std(ddof=0), n_parts="count")
        .reset_index()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=ROOT / "artifacts" / "paper_reproduction")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data" / "sf2bench")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "paper_reproduction")
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    data_cache: dict[int, SFBenchDataModule] = {}
    for path in sorted(args.artifact_root.glob("*/S_7/part_*/seed_*/h72/test_predictions.npz")):
        metadata = parse_run(path)
        if metadata["model"] not in PAPER_MODELS:
            continue
        part = int(metadata["part"])
        if part not in data_cache:
            data_cache[part] = SFBenchDataModule(
                args.dataset_root,
                split="S_7",
                part=part,
                lookback="2D",
                horizon="3D",
                batch_size=64,
                norm_clip=20.0,
            )
        archive = np.load(path, allow_pickle=False)
        prediction = torch.from_numpy(archive["prediction"])
        target = torch.from_numpy(archive["target"])
        mask = torch.from_numpy(archive["mask"]).bool()
        for quantile in (0.70, 0.80, 0.90, 0.95):
            metrics, _details, _stations, _physical = evaluate_predictions(
                prediction,
                target,
                mask,
                data_cache[part],
                quantile=quantile,
                issue_stride=1,
                min_duration=3,
                merge_gap=6,
            )
            rows.append({
                **metadata,
                "q": quantile,
                "episode_f1": metrics["episode_f1"],
                "onset_mae": metrics["episode_onset_mae_h"],
                "peak_mae": metrics["episode_peak_magnitude_mae"],
                "duration_mae": metrics["episode_duration_mae_h"],
            })
    if not rows:
        raise FileNotFoundError(
            "No S7/3D main-model prediction archives were found under "
            f"{args.artifact_root}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(rows).sort_values(["model", "part", "seed", "q"])
    raw.to_csv(args.output_dir / "threshold_sensitivity_raw.csv", index=False)
    aggregate_thresholds(raw).to_csv(
        args.output_dir / "table_11_threshold_sensitivity.csv",
        index=False,
    )
    print(f"Wrote threshold-sensitivity tables for {len(rows)} evaluations")


if __name__ == "__main__":
    main()
