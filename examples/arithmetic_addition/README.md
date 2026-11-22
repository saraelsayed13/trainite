# Arithmetic Addition: RoPE baseline vs. Abacus

> **Reviewing this branch?** Start with [`REVIEW.md`](REVIEW.md). It maps the
> four experiments (baseline, zero-pad, reversed, Abacus) to configs, code
> changes, and recorded results.

This example contains a controlled comparison between two decoder-only
Transformers on synthetic multi-digit addition:

| Experiment | Positional signal | Optimizer | Config |
| --- | --- | --- | --- |
| Baseline (natural order) | RoPE (rotary relative positions) | AdamW | `config_baseline.yaml` |
| Zero-pad | RoPE, fixed-width operands | AdamW | `config_zero_pad.yaml` |
| Reversed | RoPE, LSB-first digits | AdamW | `config_reversed.yaml` (3L ablation), `config.yaml` (4L small) |
| SinAbacus | Sinusoidal Abacus digit-column embeddings | AdamW | `config_abacus.yaml` |

There are three size/budget variants, plus a smaller set of format ablations:

| Variant | Configs | Train | OOD test | Epochs | Depth (RoPE / SinAbacus) | Use |
| --- | --- | --- | --- | --- | --- | --- |
| Format ablations | `config_baseline.yaml`, `config_zero_pad.yaml`, `config_reversed.yaml` | 2-4 / 50k | 5 / 1k | 300 | 3 / - | format checks |
| Small | `config.yaml`, `config_abacus.yaml` | 2-6 / 100k | 7-10 / 2k | 300 | 4 / 8 | quick local runs |
| Large | `config_large.yaml`, `config_abacus_large.yaml` | 2-6 / 200k | 7-12 / 5k | 150 | 4 / 8 | Modal / fast GPU |
| Local | `config_large_local.yaml`, `config_abacus_local.yaml` | 2-6 / 150k | 7-12 / 5k | 150 | 4 / 8 | RTX 2050 |

The variants use the same data format, loss, and optimizer; they differ in
dataset size, OOD range, and model width (128 for small, 256 for large/local).
The local variant keeps the large width and OOD test but uses a 150k training
pool and a 150-epoch budget.

The format ablations are a separate, smaller budget (128 wide, 3 layers, 50k
training problems) used to isolate the digit-format change from model size.
`Epochs` is the configured budget, not the number of epochs actually recorded:
the baseline run used 20 and the zero-pad run 26 before being stopped.

**Depth caveat.** The large RoPE control (`config_large_local.yaml`) stays at 4
layers (its original configuration, already trained), while the SinAbacus runs
use 8 layers as requested. Depth is therefore a confound when comparing the two:
any OOD gain cannot be attributed to the positional encoding alone. Width, heads,
FFN, dropout, data, tokenizer, and optimizer schedule are otherwise matched. To
make the comparison clean, retrain RoPE at 8 layers with
`config_large_local.yaml` after setting `model.num_layers: 8` (about 4h20m on
the RTX 2050).

## Data and fairness

`dataset_impl/arithmetic_addition.py` has two independent format switches, both
off by default:

- **Natural order** (MSB-first, no switch): `config_baseline.yaml`.
- **Fixed width**: `config_zero_pad.yaml` zero-pads each operand to
  `max_digits` and the answer to `max_digits + 1`, so a carry always has a slot.
- **Reversed** (LSB-first): `config_reversed.yaml` and the 4L small
  `config.yaml` used as the RoPE control for the Abacus runs.

Reversal is what the Abacus runs use: it makes the column id of an
already-generated token independent of how many digits are still to come. Every
SinAbacus run and its RoPE control use LSB-first, so the comparison is not
confounded by digit order. The three format ablations deliberately vary the
format instead, which is why they are a separate, smaller budget.

Splits: train and validation always come from **one** pool of unique problems,
split with `random_split`, so the two sets cannot overlap.

| Variant | Train / val | OOD test |
| --- | --- | --- |
| Format ablations | 50,000 / 1,000 from a 51k pool, 2-4 digits, `seed=42` | 5 digits, 1,000, `seed=200` |
| Small | 100,000 / 1,000 from a 101k pool, 2-6 digits, `seed=42` | 7-10 digits, 2,000, `seed=200` |
| Large | 200,000 / 5,000 from a 205k pool, 2-6 digits, `seed=42` | 7-12 digits, 5,000, `seed=200` |
| Local | 150,000 / 5,000 from a 155k pool, 2-6 digits, `seed=42` | 7-12 digits, 5,000, `seed=200` |

The test set is never carved out of that pool: it is built separately with a
different digit range and seed, because it is meant to be out of distribution.

The unmodified (MSB-first) strings are always kept as `original_source` /
`original_target` metadata and used only for readable logs; the format switches
change what the model sees, not what the logs show.

## Abacus implementation

`models/abacus_transformer.py` keeps the column-id scheme from the official
implementation of “Transformers Can Do Arithmetic with the Right Embeddings”
(McLeish et al., 2024, https://arxiv.org/abs/2405.17399), but replaces the
paper's learned lookup table with a functional sinusoidal encoding:

- column ids are consecutive and one-based from the start of each number
  (the units digit, because data is LSB-first);
- every column id `c` maps to `[sin(c * f), cos(c * f)]` over a fixed geometric
  frequency bank, then through a learned linear projection;
- all non-digit tokens share one learned vector (the old slot `0`);
- a single offset `k ~ U[0, max_offset]` is sampled per training batch and added
  to digit positions only; inference uses offset `0`;
- the Abacus embedding is **added** to the token embedding, not concatenated.

Why sinusoidal: the paper's learned table is an absolute positional embedding
and cannot represent column indices never seen in training. With 2-6 digit
operands the longest answer is 7 digits, so the row for column 8 is never
trained and accuracy falls off a cliff at 8-digit operands. A sinusoidal
function gives every column, seen or unseen, a unique and smoothly related
vector, which is what allows the model to align digits beyond the training
length. The fixed basis is mixed by a learned `hidden_size x hidden_size`
projection (about 65k parameters at width 256); the only other positional
parameter is the shared non-digit vector.

Paper add-ons outside the positional-encoding change — input injection,
progressive loss, and the depth-recurrent/looped transformer — are intentionally
excluded, so the shared trunk and training compute match the baseline.

The critical consistency property is: appending generated digits never changes
an earlier token's column id, so teacher-forced training and autoregressive
generation assign identical ids.

## Setup

The example is self-contained: it does **not** import `trainite.*` at runtime.
PyTorch, PyTorch-Ignite, TensorBoard, and W&B are used.

Using the existing environment:

```bash
cd examples/arithmetic_addition
source .venv/bin/activate
```

Using `uv` from scratch:

```bash
cd examples/arithmetic_addition
uv sync
```

## Training

Run from `examples/arithmetic_addition`:

```bash
./.venv/bin/python main.py config.yaml          # baseline-adamw
./.venv/bin/python main.py config_abacus.yaml   # sinabacus-8l-small
```

Large run on Modal (L4 by default, ~4-5h for both; L40S ~2.5-3h):

```bash
modal run modal_train.py                    # config_large.yaml then config_abacus_large.yaml
modal run modal_train.py --gpu L40S
```

Modal requires a payment method on file even though the $30/month credit covers
this run. Outputs are written to the `arithmetic-addition-outputs` Volume:

```bash
modal volume get arithmetic-addition-outputs / ./modal_outputs
```

Local run (RTX 2050, both models sequentially, 150k samples, 150 epochs each):

```bash
cd examples/arithmetic_addition
./.venv/bin/python main.py config_large_local.yaml
./.venv/bin/python main.py config_abacus_local.yaml
```

Or both in one shell:

```bash
./.venv/bin/python main.py config_large_local.yaml && \
  ./.venv/bin/python main.py config_abacus_local.yaml
```

Each run creates a timestamped directory:

```text
outputs/<run_name>/<timestamp>/
├── best.pt                 # lowest validation loss
├── last.pt                 # most recent epoch
├── config.yaml             # resolved config for this run
├── output.log              # console output, appended by evaluate.py too
├── eval_context.json       # best epoch/global step and final eval metrics
├── wandb_run_id.txt        # W&B run id used by evaluate.py
├── tensorboard/            # TensorBoard event files
└── wandb/                  # W&B run files (offline runs live here)
```

Checkpoint selection is identical for both experiments: the checkpoint with the
lowest ID validation loss. Final ID and OOD metrics are logged at that best
checkpoint's epoch/global step, and `evaluate.py` reuses the same association
from `eval_context.json`.

### W&B behavior

`wandb.mode: auto` means:

- `online` when `WANDB_API_KEY` is set;
- otherwise `offline` (no network required).

Run names are `baseline-adamw` / `sinabacus-8l-small` for the small configs,
`baseline-adamw-large` / `sinabacus-8l` for the large configs, and
`baseline-adamw-large-local` / `sinabacus-8l-local` for the local
configs, all in project `arithmetic-addition`. Logging failures are caught and
downgraded to warnings so training never depends on W&B. To force offline mode:

```bash
WANDB_MODE=offline ./.venv/bin/python main.py config.yaml
```

To upload a completed offline run:

```bash
./.venv/bin/wandb sync outputs/baseline-adamw/<timestamp>/wandb/offline-run-*
```

## Evaluation

`evaluate.py` reloads a run's best checkpoint, evaluates ID (validation) and OOD
(test) splits with the same `evaluation.evaluate_split` helper used by training,
appends to that run's `output.log`, and writes to the **same** TensorBoard
directory and W&B run (via `wandb_run_id.txt`). It never creates a new run
directory or W&B run.

```bash
./.venv/bin/python evaluate.py config.yaml
./.venv/bin/python evaluate.py config_abacus.yaml
```

Useful options:

```bash
--checkpoint PATH        # explicit best.pt (default: newest run for the config)
--run-dir DIRECTORY      # run directory containing best.pt
--max-examples N         # limit samples per split (smoke testing)
--skip-wandb
--skip-tensorboard
```

Exact match is the primary metric; teacher-forced token accuracy is secondary.

## Dashboards

TensorBoard:

```bash
./.venv/bin/tensorboard --logdir outputs
```

W&B (when online): project `arithmetic-addition`, with run names matching the
config variant (`*-large`, `*-large-local`). Offline runs can be viewed after
`wandb sync`.

Logged metrics:

| Tag | Meaning |
| --- | --- |
| `training/batch_loss` | smoothed per-iteration training loss |
| `training/loss`, `validation/loss` | epoch-level train/validation loss |
| `training/token_accuracy`, `validation/token_accuracy` | epoch-level token accuracy |
| `training/learning_rate` | learning rate at epoch end |
| `progress/epoch`, `progress/global_step` | epoch and optimizer step |
| `eval/id/exact_match`, `eval/ood/exact_match` | primary final metrics |
| `eval/id/token_accuracy`, `eval/ood/token_accuracy` | secondary final metrics |
| `eval/{id,ood}/exact_match/len_AxB` | exact match per operand-length bucket |
| `eval/best_val_loss` | validation loss of the selected checkpoint |
| `inference/training`, `inference/validation` | qualitative sample text summaries |

## Verification status

Verified locally:

- both experiments complete short smoke runs with TensorBoard and offline W&B
  logging;
- two identical seeded short runs produced identical losses for both models;
- `evaluate.py` loads the best checkpoint, appends to `output.log`, and writes
  to the same TensorBoard directory/W&B run id;
- the 4L small baseline completed a full 300-epoch run on the RTX 2050 with ID
  exact match 1.0000; OOD accuracy then falls sharply with operand length (the
  per-length breakdown is recorded in `REVIEW.md`);
- measured RTX 2050 epoch times: 4x128 / 100k is ~19-21 s; 4x256 / 200k is
  ~75 s for the RoPE baseline and ~85 s for SinAbacus 8L. At 150k that is
  ~53 s and ~89 s respectively, so the local baseline takes ~2h12m and the
  local SinAbacus 8L run takes ~3h50m (150 epochs each).

**Not verified here:** convergence and final OOD quality for the large and
local variants. Run both configs in a variant on the same hardware and
compare their best-validation-loss checkpoints before drawing conclusions.
Because the SinAbacus configs use 8 layers and the RoPE configs use 4, a clean
attribution of any OOD gain to the positional encoding requires retraining RoPE
at 8 layers as described above.

The zero-pad and reversed configs are format diagnostics, not part of the
positional-encoding comparison; `REVIEW.md` lists all four experiments with their
recorded numbers and the caveats around them.
