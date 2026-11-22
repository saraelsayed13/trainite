# Review guide: four length-generalization experiments

This branch collects the four experiments on one tree so they can be reviewed
side by side. Each experiment is one config; the dataset carries the format
switches, so the diff between experiments is small and explicit.

Task and metric: character-level multi-digit addition. Train on short operands,
test on longer ones, score **exact match** (the whole generated answer must be
correct) and **teacher-forced token accuracy** (diagnostic).

## The four experiments

| # | Experiment | Config | Format | Model | Depth x width | Train -> test | OOD exact | OOD token |
|---|---|---|---|---|---|---|---|---|
| 1 | Baseline | `config_baseline.yaml` | MSB-first | RoPE | 3L x 128 | 2-4 -> 5 digits | **0.00%** | **23.37%** |
| 2 | Zero-pad | `config_zero_pad.yaml` | fixed width | RoPE | 3L x 128 | width 4 -> 5 | **0 / 1000** | - |
| 3 | Reversed | `config_reversed.yaml` | LSB-first | RoPE | 3L x 128 | 2-4 -> 5 digits | **0.00%** | **36.34%** |
| 4 | Abacus | `config_abacus_local.yaml` | LSB-first | Abacus (NoPE) | 8L x 256 | 2-6 -> 7-12 digits | **9.88%** | **62.46%** |

Two problems are being attacked separately:

- **Carry direction** (row 3): a carry travels right to left. Generating the
  most significant digit first requires the whole carry chain up front.
- **Column identity** (row 4): input digit `k` must map to output digit `k`.
  RoPE only exposes relative distance, which changes with number length.

## The matched-depth comparison (the clean claim)

Large scale, same data, optimizer, and LSB-first format:

| Run | Depth x width | Column embedding | OOD exact | OOD token |
|---|---|---|---|---|
| RoPE control (`config_large_local.yaml`) | 4L x 256 | RoPE | 0.00% | 14.18% |
| Abacus (`config_abacus_local.yaml` at commit `0c7ae5a`) | **4L** x 256 | learned table | **3.08%** | 59.82% |
| Abacus (`config_abacus_local.yaml` at this tip) | **8L** x 256 | sinusoidal | **9.88%** | 62.46% |

The 4L comparison changes one variable (the positional scheme). The 9.88% run
changes **two** (embedding and depth), so it is not a clean ablation.

Per-length cliff for the 8L sinusoidal model: 7 digits **100%**, 8 digits
**~77-86%**, 9 digits **2-6%**, 10-12 digits **0%**. Seven digits is the
longest answer seen in training.

## What each experiment changes in code

All four share the same dataset file, `dataset_impl/arithmetic_addition.py`.
The format switches are three flags:

```python
reverse_digits: bool = False   # LSB-first operands and answer
pad_to_width: bool = False     # zfill operands to max_digits, answer to max_digits + 1
```

```python
if self.pad_to_width:
    width = self.max_digits
    source = f"{str(a).zfill(width)}+{str(b).zfill(width)}"
    target = original_target.zfill(width + 1)
elif self.reverse_digits:
    source = f"{str(a)[::-1]}+{str(b)[::-1]}"
    target = original_target[::-1]
else:
    source, target = original_source, original_target
```

The model change is confined to `models/abacus_transformer.py`:

1. `compute_column_ids()` scans each sequence and numbers digits per number
   (units = 1, tens = 2, ...), with 0 for non-digits.
2. `AbacusEmbedding` maps those ids through a fixed sinusoidal basis, a learned
   linear projection, and one shared learned vector for non-digits.
3. `Attention` is plain causal attention: RoPE is removed entirely.
4. `forward()` adds the column embedding to the token embedding once, before
   the block stack.

The RoPE control is unchanged in `models/rope_transformer.py`. Batching and the
loss (`CausalLMCollateFn`, `CausalLMCrossEntropyLoss`) live in one shared module,
`models/causal_lm.py`, used by both models, so the two model files differ only in
the positional scheme.

## How to run

```bash
cd examples/arithmetic_addition

# experiment 1-3 (small, ~minutes per epoch on a laptop GPU)
uv run python main.py config_baseline.yaml
uv run python main.py config_zero_pad.yaml
uv run python main.py config_reversed.yaml

# experiment 4 and its RoPE control (large, RTX 2050: ~4h each)
uv run python main.py config_large_local.yaml
uv run python main.py config_abacus_local.yaml

# evaluate a checkpoint, per-length exact match
uv run python evaluate.py config_abacus_local.yaml --checkpoint outputs/<run>/<timestamp>/best.pt
```

## Where the numbers come from

The recorded runs were produced by the per-experiment branches before this
review branch existed; the three small configs here were added to collect all
four setups in one tree. Each recorded run folder contains a snapshot of the
config that actually ran.

Re-running a config here will not reproduce those numbers bit-for-bit: train and
validation are now drawn from one pool with `random_split`
(`utils.build_split_datasets`), where the recorded runs used explicit
`train_size` / `val_size` ranges. Expect the same ordering of effects, not the
same last digit.

| Result | Run folder | Note |
|---|---|---|
| Baseline, 5 digits | `outputs/adamw_baseline_20ep/20261025_161348` | 3L, 50k, 20 epochs, MSB-first |
| Zero-pad, 0/1000 at 5 | `outputs/arithmetic_addition_zero_pad/20261025_153329` | 26/300 epochs, evaluated after |
| Reversed, 36.34% token | `outputs/baseline-adamw/20261106_135401` | 3L, 50k |
| Abacus small, 1.70% | `outputs/abacus-adamw/20261106_142853` | 3L, learned table |
| RoPE large, 0.00% | `outputs/baseline-adamw-large-local/20261107_101401` | 4L, 150k |
| Abacus learned 4L, 3.08% | `outputs/abacus-adamw-large-local/20261107_122627` | commit `0c7ae5a` |
| Abacus sinusoidal 8L, 9.88% | `outputs/sinabacus-8l-local/20261107_223116` | current code |

## Limitations

- No learned-vs-sinusoidal ablation at matched depth.
- Zero-padding is a diagnostic, not a fair head-to-head (26 of 300 epochs).
- This is not a reproduction of McLeish et al. (2024): 8 layers, 150k examples,
  6 training digits, versus 16 layers, ~20M examples, up to 20 digits.
