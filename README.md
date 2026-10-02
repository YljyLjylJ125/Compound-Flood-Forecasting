# Multi-Source Dynamic Graph Forecasting

Official code release for **Multi-Source Dynamic Graph Learning for
Compound-Flood Forecasting in Managed Coastal Systems**. The task is
multi-step WATER-stage forecasting: every issue uses the preceding 48 hourly
observations and predicts the next 24, 72, 120, or 168 hours.

The released protocol uses SF2Bench chronological splits `S_5`, `S_6`, and
`S_7`, with three official spatial partitions per split. The model combines a
target-only PatchTST anchor with a heterogeneous dynamic-graph residual whose
magnitude is state- and lead-dependent and explicitly bounded.

## Repository layout

```text
configs/                    Paper protocol and model configurations
data/partitions/            Official three-part station maps
docs/                       Data, model, and reproduction details
scripts/download_data.py    Verified SF2Bench downloader
scripts/train.py            Shared trainer for all models and variants
scripts/evaluate.py         Checkpoint and prediction evaluation
scripts/run_paper.py        Manuscript main-table and ablation matrix
scripts/summarize.py        Tables 1--3, 8--10, and Figure 4 aggregation
scripts/posthoc.py          Figure 3 / Table 11 threshold sensitivity
src/anchored_forecaster/    Data, models, metrics, and evaluation code
tests/                      Fast unit and contract tests
```

Large datasets, checkpoints, predictions, and generated result CSV files are
deliberately excluded from the Git repository. This directory contains only
the original manuscript's model, baseline, ablation, evaluation, and table
generation code.

## Installation

Python 3.8+ and PyTorch 2.0+ are supported.

Package dependencies and optional test dependencies are defined in
`pyproject.toml`. The SF2Bench download script is provided in
`scripts/download_data.py`.

The downloader retrieves Harvard Dataverse DOI
`10.7910/DVN/TU5UXE`, verifies the published archive size and MD5 checksum,
and extracts the expected layout:

```text
data/sf2bench/Processed_hour/{WATER,RAIN,WELL,PUMP,GATE}/S_*/<station>/
```

See [docs/DATA.md](docs/DATA.md) for column semantics and split dates.

## Train and evaluate

The shared training script supports the proposed model, baselines, and
ablation variants under the paper protocol. Each run specifies a dataset
root, output directory, model, chronological split, spatial partition,
repeat identifier, and forecast horizon.

The output contains the lowest-validation-MSE checkpoint, training history,
original-unit predictions, run-level metrics, and episode match records.
The resumable manuscript matrix writes to `artifacts/paper_reproduction/` by
default.

The evaluation script assesses saved checkpoints using the same data
boundaries, spatial partition, and forecast horizon as the corresponding
training run.

## Models and ablations

The model registry includes all eight paper baselines (`NLinear`, `PatchTST`,
`iTransformer`, `TimesNet`, `FourierGNN`, `MTGNN`, `AutoTimes`, and Graph
WaveNet), the proposed model, and the source/architecture ablations reported
in the paper. Architecture hyperparameters are recorded in `configs/models.yaml`.

The complete table-by-table experiment matrix and aggregation rules are in
[docs/PAPER_EXPERIMENTS.md](docs/PAPER_EXPERIMENTS.md). 

The baseline input audit, final input matrix, and minimal fair-input adapters
are documented in
[baseline_fair_adaptation_report.md](baseline_fair_adaptation_report.md).

## Evaluation contract

- Normalization statistics and high-water thresholds use valid training data only.
- Missing inputs are zeroed after normalization and accompanied by explicit masks.
- WATER predictions are inverse-transformed before all reported metrics.
- Main high-water thresholds are station-specific training `q=0.95` quantiles.
- Episodes require three exceedance hours and merge gaps of at most six hours.
- Forecast issues are sampled every 24 hours for episode evaluation.
- One run-level row is one model/variant, split, part, repeat, horizon, and quantile.

## Tests

Unit and contract tests are provided in `tests/`. GitHub Actions runs the test
suite on Python 3.9 and 3.11, including the AutoTimes adapter test with a mock
backbone.

## Citation

Citation metadata is provided in [CITATION.cff](CITATION.cff). The SF2Bench
dataset must also be cited according to its Dataverse record.
