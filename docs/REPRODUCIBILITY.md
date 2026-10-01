# Reproducibility notes

## What was present upstream

The initial repository contained the proposed model, a data loader, episode
metrics, and single-model train/evaluate scripts. It did not contain baseline
implementations, checkpoints, prediction archives, original experiment logs,
environment metadata, or table-generation code. Therefore,
the release distinguishes manuscript values from newly rerun values and never
claims that a new run recovered an unavailable historical artifact.

## Runtime environment

Results can differ across PyTorch, CUDA, and hardware versions; record those
versions beside result artifacts.

## Aggregation

Metrics are first calculated independently for every split/part/repeat run. The
paper table generator first averages the three repeated experiments within each
spatial part, then reports the mean and population standard deviation across
Parts 0, 1, and 2. Main Tables 1--3 average the corresponding S5, S6, and S7
split statistics, matching the hierarchy printed in the manuscript. Episode
records are pooled within a run before TP/FP/FN and matched-event errors are
calculated.

## Paper table scope audit

The PDF's Table 10 is the detailed S7 table across four horizons. Figure 3's
caption and the Table 11 explanatory paragraph explicitly say S7/3D; however,
Table 11's printed q=.95 Ours values (Episode F1 0.652 and Duration MAE 12.330)
match the main S5/S6/S7 aggregate, not Table 10's S7 values (0.587 and 16.189).
The PDF therefore contains an internal scope/value inconsistency. The release
does not silently relabel one as the other: `scripts/posthoc.py` follows the
Figure 3 caption and Table 11 explanatory paragraph and labels newly computed
threshold sensitivity as S7/3D.

The upstream interpolation provenance is unavailable with the processed
dataset; the release reads `INTERPOLATED_VALUE` with the supplied confidence
mask and does not claim a causal-interpolation audit pass.
