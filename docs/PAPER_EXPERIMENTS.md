# Manuscript experiment map

This page maps every experiment reported in the manuscript to the released
code path. Experiments introduced after the submitted paper are intentionally
not included in this repository.

## Experiment inventory

| Manuscript item | Scope | Models or variants | Code path |
|---|---|---|---|
| Tables 1--3 | S5/S6/S7, Parts 0--2, 1D/3D/5D/7D | Ours and eight baselines | `scripts/run_paper.py --suite main` |
| Tables 8--10 | The same runs, grouped by S5, S6, and S7 | Ours and eight baselines | `scripts/run_paper.py --suite main` |
| Figure 3 / Table 11 | S7, 3D, q in 0.70/0.80/0.90/0.95 | Ours and eight baselines | `scripts/posthoc.py` on the saved main-run predictions |
| Figure 4, architecture | S7, 3D, q=0.95 | Fixed graph, no event context, no bound, neither control, no graph correction | `scripts/run_paper.py --suite architecture_ablation` |
| Figure 4, sources | S7, 3D, q=0.95 | Remove neighbor WATER, RAIN, WELL, PUMP/GATE, or all non-WATER | `scripts/run_paper.py --suite source_ablation` |

The main registry contains `NLinear`, `PatchTST`, `iTransformer`, `TimesNet`,
`FourierGNN`, `MTGNN`, `AutoTimes`, and `Graph WaveNet`. Their architecture
settings are in `configs/models.yaml`; the shared optimization and episode
settings are in `configs/paper.yaml`.

## Enumerate before running

The complete original-paper reconstruction has 1,062 training runs: 972 main
comparison runs and 90 S7/3D ablation runs. This command prints the exact
matrix without starting a job:

```bash
python scripts/run_paper.py --list
```

Each experiment is run three times, with repeat identifiers 1, 2, and 3 used
to distinguish output directories and run-level records. These identifiers
only label runs; they do not control model initialization or data shuffling.
Every model, including Graph WaveNet, uses all three repetitions.

## Run the original-paper experiments

Run the main tables and the two ablation groups separately so each stage is
resumable:

```bash
python scripts/run_paper.py --suite main
python scripts/run_paper.py --suite architecture_ablation
python scripts/run_paper.py --suite source_ablation
```

Each run writes its own checkpoint, training history, `metrics.json`,
`run_metrics.csv`, and prediction archive below `artifacts/paper_reproduction/`.
Existing complete runs are skipped unless `--rerun` is supplied.

For two GPUs, partition the official spatial parts without changing any model
or optimization setting:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_paper.py --parts 0 2
CUDA_VISIBLE_DEVICES=1 python scripts/run_paper.py --parts 1
```

## Generate paper-facing tables

After training, consolidate the run-level records and recompute the four
threshold settings from saved predictions:

```bash
python scripts/summarize.py \
  --artifact-root artifacts/paper_reproduction \
  --output-dir results/paper_reproduction

python scripts/posthoc.py \
  --artifact-root artifacts/paper_reproduction \
  --dataset-root data/sf2bench \
  --output-dir results/paper_reproduction
```

`paper_run_level_metrics.csv` is the provenance-preserving source table.
`tables_1_3_main_results.csv` and `tables_8_10_split_results.csv` reproduce the
main and split-specific comparisons; `table_11_threshold_sensitivity.csv`
supplies Figure 3/Table 11; and `figure_4_ablation.csv` supplies both Figure 4
ablation groups. Generated tables are ignored by Git and are not bundled as
historical manuscript artifacts.

## Fixed protocol

- Inputs are the preceding 48 hourly observations; labels are the immediately
  following 24, 72, 120, or 168 hours.
- Only S5, S6, and S7 and official Parts 0, 1, and 2 are used.
- Normalization and station thresholds use valid training observations only.
- Optimization is AdamW for 10 epochs, learning rate `1e-3`, weight decay
  `1e-5`, batch size 64, and gradient clipping at 1.0.
- The selected checkpoint has the lowest validation masked MSE on the
  normalized target scale.
- Main episode evaluation uses q=0.95, a 24-hour issue stride, at least three
  exceedance hours, a merge gap of at most six hours, and one-to-one matching.

See [DATA.md](DATA.md) for the input contract and
[REPRODUCIBILITY.md](REPRODUCIBILITY.md) for limitations of reconstructing
experiments whose original checkpoints and logs were not available.
