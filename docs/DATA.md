# SF2Bench data contract

## Scope

The paper uses only chronological blocks `S_5`, `S_6`, and `S_7`:

| Split | Train | Validation | Test |
|---|---|---|---|
| S5 | 2010-2012 | 2013 | 2014 |
| S6 | 2015-2017 | 2018 | 2019 |
| S7 | 2020-2021 | 2022 | 2023 |

Each block is evaluated with official spatial parts 0, 1, and 2. Partition JSON
files are versioned under `data/partitions/`; observations remain in the
separately downloaded SF2Bench archive.

## Station files

Each category directory contains station folders with a CSV and location JSON.
The loader reads:

- `TIMESTAMP`: hourly time index;
- `INTERPOLATED_VALUE`: SF2Bench's released, processed numeric series;
- `CONFIDENCE`: used to derive the explicit validity mask;
- `X COORD` and `Y COORD`: static spatial metadata.

`INTERPOLATED_VALUE` is a dataset storage column, not a prediction target with
a different scientific meaning. WATER rows provide both the historical target
series and future labels; RAIN, WELL, PUMP, and GATE rows are historical
context only. A value with a non-positive confidence is masked and set to zero
after train-only normalization, so it cannot affect the model input or loss.

## Window boundary

For a forecast issue at hour `t`, the input contains exactly
`[t-47, ..., t]`. The label begins at `t+1` and covers the selected horizon.
No window crosses a train/validation/test phase boundary. Means, standard
deviations, and station-specific episode thresholds are fitted independently
for every split/part using valid training observations only.

## Download verification

- DOI: `10.7910/DVN/TU5UXE`
- Dataverse file ID: `11275874`
- Filename: `dataset.tar-1.gz`
- Size: `2,136,442,311` bytes
- MD5: `0e82b123b27d2aa64d941812ed2cf709`

