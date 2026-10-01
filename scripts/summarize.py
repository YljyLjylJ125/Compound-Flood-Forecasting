#!/usr/bin/env python3
"""Generate manuscript main, split-specific, and ablation result tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
METRICS = ("mae", "mse", "episode_f1", "onset_mae", "peak_mae", "duration_mae")
PAPER_MODELS = {
    "ours", "nlinear", "patchtst", "itransformer", "timesnet",
    "fouriergnn", "mtgnn", "autotimes", "graphwavenet",
}
ARCHITECTURE_ABLATIONS = {
    "fixed_graph", "no_event_context", "no_correction_bound",
    "no_event_context_bound", "no_graph_correction",
}
SOURCE_ABLATIONS = {
    "without_neighbor_water", "without_rain", "without_well",
    "without_pump_gate", "without_nonwater",
}


def load_runs(artifact_root: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(artifact_root.glob("**/run_metrics.csv")):
        frame = pd.read_csv(path)
        frame["artifact_dir"] = str(path.parent)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No run_metrics.csv files found under {artifact_root}")
    runs = pd.concat(frames, ignore_index=True)
    key = ["variant", "split", "part", "repeat_id", "horizon", "q"]
    duplicate = runs.duplicated(key, keep=False)
    if duplicate.any():
        raise ValueError(f"Duplicate run keys:\n{runs.loc[duplicate, key].to_string(index=False)}")
    return runs.sort_values(key).reset_index(drop=True)


def long_metrics(runs: pd.DataFrame) -> pd.DataFrame:
    identifiers = ["model", "variant", "split", "part", "repeat_id", "horizon", "q", "artifact_dir"]
    return runs.melt(
        id_vars=identifiers,
        value_vars=list(METRICS),
        var_name="metric",
        value_name="value",
    )


def split_results(runs: pd.DataFrame) -> pd.DataFrame:
    """Average repeated experiments per part, then summarize the official parts."""

    per_part = (
        long_metrics(runs)
        .groupby(["variant", "split", "part", "horizon", "q", "metric"], dropna=False)["value"]
        .agg(repeat_mean="mean", n_repeats="count")
        .reset_index()
    )
    result = (
        per_part.groupby(["variant", "split", "horizon", "q", "metric"], dropna=False)["repeat_mean"]
        .agg(mean_raw="mean", std_across_parts_raw=lambda values: values.std(ddof=0), n_parts="count")
        .reset_index()
    )
    result["reported_scale"] = np.where(result["metric"].isin(["mae", "mse"]), 100.0, 1.0)
    result["mean_reported_scale"] = result["mean_raw"] * result["reported_scale"]
    result["std_across_parts_reported_scale"] = result["std_across_parts_raw"] * result["reported_scale"]
    return result


def main_results(split_table: pd.DataFrame) -> pd.DataFrame:
    """Reproduce Tables 1--3 by averaging the S5/S6/S7 table statistics."""

    main = split_table[split_table["variant"].isin(PAPER_MODELS)]
    result = (
        main.groupby(["variant", "horizon", "q", "metric", "reported_scale"], dropna=False)
        .agg(
            mean_raw=("mean_raw", "mean"),
            std_across_parts_raw=("std_across_parts_raw", "mean"),
            n_splits=("split", "nunique"),
        )
        .reset_index()
    )
    result["mean_reported_scale"] = result["mean_raw"] * result["reported_scale"]
    result["std_across_parts_reported_scale"] = result["std_across_parts_raw"] * result["reported_scale"]
    return result


def ablation_results(split_table: pd.DataFrame) -> pd.DataFrame:
    selected = split_table[
        (split_table["split"] == "S7")
        & (split_table["horizon"] == 72)
        & (split_table["q"] == 0.95)
    ].copy()
    full = selected[selected["variant"] == "ours"][["metric", "mean_raw"]].rename(
        columns={"mean_raw": "full_model_mean_raw"}
    )
    groups = []
    for group_name, variants in (
        ("architecture", ARCHITECTURE_ABLATIONS),
        ("source", SOURCE_ABLATIONS),
    ):
        current = selected[selected["variant"].isin(variants)].merge(full, on="metric", how="left")
        current["ablation_group"] = group_name
        lower_is_better = current["metric"] != "episode_f1"
        current["relative_degradation_pct"] = np.where(
            lower_is_better,
            100.0 * (current["mean_raw"] - current["full_model_mean_raw"]) / current["full_model_mean_raw"].abs(),
            100.0 * (current["full_model_mean_raw"] - current["mean_raw"]) / current["full_model_mean_raw"].abs(),
        )
        groups.append(current)
    return pd.concat(groups, ignore_index=True) if groups else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=ROOT / "artifacts" / "paper_reproduction")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "paper_reproduction")
    args = parser.parse_args()

    runs = load_runs(args.artifact_root)
    allowed = PAPER_MODELS | ARCHITECTURE_ABLATIONS | SOURCE_ABLATIONS
    unexpected = sorted(set(runs["variant"]) - allowed)
    if unexpected:
        raise ValueError(f"Non-manuscript variants found in paper artifact root: {unexpected}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(args.output_dir / "paper_run_level_metrics.csv", index=False)
    split_table = split_results(runs)
    split_table.to_csv(args.output_dir / "tables_8_10_split_results.csv", index=False)
    main_results(split_table).to_csv(args.output_dir / "tables_1_3_main_results.csv", index=False)
    ablation_results(split_table).to_csv(args.output_dir / "figure_4_ablation.csv", index=False)
    print(f"Wrote manuscript tables from {len(runs)} completed runs")


if __name__ == "__main__":
    main()
