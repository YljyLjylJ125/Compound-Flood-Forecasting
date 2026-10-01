#!/usr/bin/env python3
"""Run the manuscript main comparison and S7/3D ablation matrix."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PAPER_MODELS = (
    "ours", "nlinear", "patchtst", "itransformer", "timesnet",
    "fouriergnn", "mtgnn", "autotimes", "graphwavenet",
)
ARCHITECTURE_ABLATIONS = (
    "fixed_graph", "no_event_context", "no_correction_bound",
    "no_event_context_bound", "no_graph_correction",
)
SOURCE_ABLATIONS = (
    "without_neighbor_water", "without_rain", "without_well",
    "without_pump_gate", "without_nonwater",
)
HORIZONS = ("1D", "3D", "5D", "7D")


@dataclass(frozen=True)
class Run:
    suite: str
    model: str
    split: str
    part: int
    repeat_id: int
    horizon: str

    @property
    def relative_dir(self) -> Path:
        hours = {"1D": 24, "3D": 72, "5D": 120, "7D": 168}[self.horizon]
        return Path(self.model) / self.split / f"part_{self.part}" / f"repeat_{self.repeat_id}" / f"h{hours}"


def paper_matrix(
    splits: tuple[str, ...] = ("S_5", "S_6", "S_7"),
    parts: tuple[int, ...] = (0, 1, 2),
    repeat_ids: tuple[int, ...] = (1, 2, 3),
) -> list[Run]:
    runs = [
        Run("main", model, split, part, repeat_id, horizon)
        for model in PAPER_MODELS
        for split in splits
        for part in parts
        for repeat_id in repeat_ids
        for horizon in HORIZONS
    ]
    if "S_7" in splits:
        for suite, variants in (
            ("architecture_ablation", ARCHITECTURE_ABLATIONS),
            ("source_ablation", SOURCE_ABLATIONS),
        ):
            runs.extend(
                Run(suite, variant, "S_7", part, repeat_id, "3D")
                for variant in variants
                for part in parts
                for repeat_id in repeat_ids
            )
    return runs


def complete(path: Path) -> bool:
    required = ("metrics.json", "run_metrics.csv", "best_model.pt", "test_predictions.npz")
    if not all((path / filename).exists() for filename in required):
        return False
    try:
        row = json.loads((path / "metrics.json").read_text(encoding="utf-8"))["run_row"]
    except (OSError, KeyError, json.JSONDecodeError):
        return False
    return (
        row.get("evaluator_version") == "release-v3-daily-issues"
        and row.get("model_data_contract_version") == "paper-strict-mask-aware-v5"
    )


def execute(run: Run, args: argparse.Namespace) -> None:
    output = args.artifact_root / run.relative_dir
    if complete(output) and not args.rerun:
        print(
            f"SKIP {run.model} {run.split} part={run.part} "
            f"repeat={run.repeat_id} horizon={run.horizon}",
            flush=True,
        )
        return
    output.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, str(ROOT / "scripts" / "train.py"),
        "--dataset-root", str(args.dataset_root),
        "--output-dir", str(output),
        "--model", run.model,
        "--split", run.split,
        "--part", str(run.part),
        "--repeat-id", str(run.repeat_id),
        "--horizon", run.horizon,
        "--epochs", "10",
        "--batch-size", "64",
        "--num-workers", str(args.num_workers),
    ]
    if args.fast_dev_run:
        command += ["--max-train-batches", "1", "--max-eval-batches", "1", "--epochs", "1"]
    print(
        f"RUN  {run.model} {run.split} part={run.part} "
        f"repeat={run.repeat_id} horizon={run.horizon}",
        flush=True,
    )
    with (output / "run.log").open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            if line.startswith("epoch=") or line.startswith("early_stop"):
                print(f"  {line.strip()}", flush=True)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data" / "sf2bench")
    parser.add_argument("--artifact-root", type=Path, default=ROOT / "artifacts" / "paper_reproduction")
    parser.add_argument(
        "--suite",
        action="append",
        choices=["main", "architecture_ablation", "source_ablation"],
    )
    parser.add_argument("--model", action="append", choices=PAPER_MODELS + ARCHITECTURE_ABLATIONS + SOURCE_ABLATIONS)
    parser.add_argument("--splits", nargs="+", choices=["S_5", "S_6", "S_7"], default=["S_5", "S_6", "S_7"])
    parser.add_argument("--parts", nargs="+", type=int, choices=[0, 1, 2], default=[0, 1, 2])
    parser.add_argument("--repeat-ids", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.artifact_root = args.artifact_root.resolve()
    runs = paper_matrix(tuple(args.splits), tuple(args.parts), tuple(args.repeat_ids))
    selected = [
        run for run in runs
        if (not args.suite or run.suite in args.suite)
        and (not args.model or run.model in args.model)
    ]
    if args.suite and "main" not in args.suite and any(
        suite in args.suite for suite in ("architecture_ablation", "source_ablation")
    ):
        selected.extend(
            run for run in runs
            if run.suite == "main" and run.model == "ours"
            and run.split == "S_7" and run.horizon == "3D"
        )
    selected = list({run.relative_dir: run for run in selected}.values())
    if args.max_runs:
        selected = [
            run for run in selected
            if args.rerun or not complete(args.artifact_root / run.relative_dir)
        ][:args.max_runs]
    print(f"Selected {len(selected)} paper-reproduction runs", flush=True)
    if args.list:
        for run in selected:
            print(run)
        return
    for index, run in enumerate(selected, 1):
        print(f"[{index}/{len(selected)}]", flush=True)
        execute(run, args)


if __name__ == "__main__":
    main()
