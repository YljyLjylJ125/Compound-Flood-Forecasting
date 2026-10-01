from pathlib import Path
import sys

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from posthoc import aggregate_thresholds  # noqa: E402
from summarize import METRICS, ablation_results, main_results, split_results  # noqa: E402


def sample_runs() -> pd.DataFrame:
    rows = []
    variants = ("ours", "patchtst", "fixed_graph", "without_rain")
    for variant_index, variant in enumerate(variants):
        splits = ("S7",) if variant not in {"ours", "patchtst"} else ("S5", "S6", "S7")
        for split_index, split in enumerate(splits):
            for part in (0, 1, 2):
                for seed in (42, 43, 44):
                    base = 1.0 + variant_index + split_index / 10 + part / 100 + seed / 10000
                    row = {
                        "model": variant,
                        "variant": variant,
                        "split": split,
                        "part": part,
                        "seed": seed,
                        "horizon": 72,
                        "q": 0.95,
                        "artifact_dir": "unused",
                    }
                    row.update({metric: base for metric in METRICS})
                    if variant != "ours":
                        row["episode_f1"] = 1.0 / base
                    rows.append(row)
    return pd.DataFrame(rows)


def test_paper_aggregation_averages_seeds_then_spatial_parts():
    split = split_results(sample_runs())
    assert set(split["n_parts"]) == {3}
    ours_s7_mae = split[
        (split["variant"] == "ours")
        & (split["split"] == "S7")
        & (split["metric"] == "mae")
    ].iloc[0]
    assert ours_s7_mae["reported_scale"] == 100.0
    assert ours_s7_mae["mean_reported_scale"] == ours_s7_mae["mean_raw"] * 100.0

    main = main_results(split)
    ours_main = main[(main["variant"] == "ours") & (main["metric"] == "mae")].iloc[0]
    assert ours_main["n_splits"] == 3


def test_ablation_degradation_uses_metric_direction():
    table = ablation_results(split_results(sample_runs()))
    fixed_mae = table[(table["variant"] == "fixed_graph") & (table["metric"] == "mae")].iloc[0]
    fixed_f1 = table[(table["variant"] == "fixed_graph") & (table["metric"] == "episode_f1")].iloc[0]
    assert fixed_mae["relative_degradation_pct"] > 0
    assert fixed_f1["relative_degradation_pct"] > 0


def test_threshold_table_averages_seeds_within_each_part():
    rows = []
    for part in (0, 1, 2):
        for seed in (42, 43, 44):
            rows.append({
                "model": "ours",
                "split": "S_7",
                "part": part,
                "seed": seed,
                "horizon": 72,
                "q": 0.95,
                "episode_f1": 0.8 + part / 100,
                "onset_mae": 1.0,
                "peak_mae": 0.1,
                "duration_mae": 10.0,
            })
    table = aggregate_thresholds(pd.DataFrame(rows))
    assert set(table["n_parts"]) == {3}
    assert len(table) == 4
