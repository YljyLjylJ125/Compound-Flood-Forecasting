# Anchored Dynamic-Graph Forecasting for High-Water Processes



The model uses a pure PatchTST anchor based only on a target WATER station's history. An input-dependent graph then uses WATER, RAIN, WELL, PUMP, and GATE observations to construct an event-conditioned, lead-dependent bounded residual correction. Training uses masked MSE; high-water episode labels are used only for evaluation.

```text
src/anchored_forecaster/
  data.py                 SFBench loader and training-only normalization
  models.py               Proposed pure-PatchTST anchored dynamic-graph model
  evaluation.py           Evaluation utilities
  metrics/                Full-record and high-water episode metrics
  reproducibility.py      Random-seed utility
scripts/
  train.py                Training and test evaluation
  evaluate.py             Checkpoint evaluation
data/partitions/          Official station-partition JSON files
```

Install Python 3.9+ with `torch`, `numpy`, and `pandas`. Download the processed SFBench/SF2Bench data from [DOI:10.7910/DVN/TU5UXE](https://doi.org/10.7910/DVN/TU5UXE). 

Expected data layout:

```text
<dataset-root>/Processed_hour/{WATER,RAIN,WELL,PUMP,GATE}/S_*/<station>/
```

Each station directory must contain a CSV with `TIMESTAMP`, `CONFIDENCE`, and `INTERPOLATED_VALUE`, plus its location JSON file. The official station-partition files needed for the `part=0/1/2` protocol are included under `data/partitions/`.



The script writes the selected checkpoint, full-record metrics, and episode-level results. The default episode threshold is the station-specific training-period $q=0.95$ quantile; it can be changed with `--episode-quantile` without changing training.
