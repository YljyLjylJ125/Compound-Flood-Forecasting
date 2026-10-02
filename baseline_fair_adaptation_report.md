# Baseline Fair-Input Adaptation Report

The report is based on the released code path: dataset window -> shared
`x`/`x_mask` batch -> model forward -> original backbone -> WATER head. The
adapter status is `NATIVE` when the baseline already consumed the field,
`ADAPTED` when only a small input-side projection/embedding was added, and
`NOT_ADAPTED` when using the field would require changing the baseline's
central interaction mechanism.

## Table 1. Adaptation status

| Model | Other WATER | RAIN/WELL/PUMP/GATE | Mask | Type | Coord | Reason |
|---|---|---|---|---|---|---|
| NLinear | NOT_ADAPTED | NOT_ADAPTED | NATIVE | NOT_ADAPTED | NOT_ADAPTED | Channel-independent local linear baseline; native valid-value mask handling |
| PatchTST | NOT_ADAPTED | NOT_ADAPTED | NATIVE | NOT_ADAPTED | NOT_ADAPTED | Channel-independent PatchTST; native masked anchor |
| iTransformer | NATIVE | NATIVE | ADAPTED | ADAPTED | ADAPTED | Variable-token attention already mixes all nodes; additive token-side adapters |
| TimesNet | NATIVE | NATIVE | ADAPTED | ADAPTED | ADAPTED | Multivariate input projection already mixes nodes; adapters are before TimesBlocks |
| FourierGNN | NATIVE | NATIVE | ADAPTED | ADAPTED | ADAPTED | All series are spectral nodes; per-node adapter precedes unchanged spectral core |
| MTGNN | NATIVE | NATIVE | ADAPTED | ADAPTED | ADAPTED | Native learned graph already mixes all nodes; input channels add mask and metadata |
| AutoTimes | NOT_ADAPTED | NOT_ADAPTED | ADAPTED | NOT_ADAPTED | NOT_ADAPTED | Per-series frozen-backbone design retained; only token mask channel added |
| Graph WaveNet | NATIVE | NATIVE | ADAPTED | ADAPTED | ADAPTED | Native adaptive graph and temporal convolutions retained; node features widened |

## Table 2. Final actual inputs

| Model | Target WATER | Other WATER | RAIN/WELL/PUMP/GATE | Mask | Type | Coord |
|---|---:|---:|---:|---:|---:|---:|
| NLinear | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ |
| PatchTST | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ |
| iTransformer | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| TimesNet | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| FourierGNN | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| MTGNN | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| AutoTimes | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ |
| Graph WaveNet | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Ours | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

## Table 3. Architecture preservation

Adapter counts are measured at the largest official S5/S6/S7 partition for
each model; the ratio is relative to the original local-architecture parameter
count. AutoTimes' external frozen GPT-2 weights are excluded from both sides,
so the reported ratio is conservative for the trainable local adapter.

| Model | Core architecture changed? | Adapter params | Adapter ratio |
|---|---:|---:|---:|
| NLinear | No | 0 | 0.000% |
| PatchTST | No | 0 | 0.000% |
| iTransformer | No | 28,160 | 0.442% |
| TimesNet | No | 3,528 | 0.154% |
| FourierGNN | No | 43 | 0.008% |
| MTGNN | No | 320 | 0.701% |
| AutoTimes | No | 6,144 | 0.998% |
| Graph WaveNet | No | 320 | 0.086% |

## Table 4. Fairness checks

| Check | Result |
|---|---|
| Same chronological split | PASS |
| Same spatial partition | PASS |
| Same WATER target IDs | PASS |
| Same issue times | PASS |
| Same lookback | PASS |
| Same horizon | PASS |
| Train-only normalization | PASS |
| Same loss mask | PASS |
| Same evaluation samples | PASS |
| No future observations | PASS |

The adapted models were checked with one-batch forward/backward tests,
finite-output checks, masked-value invariance checks, mask sensitivity checks,
and metadata/context perturbation checks. The release test suite contains
these checks in
`tests/test_models.py`.
