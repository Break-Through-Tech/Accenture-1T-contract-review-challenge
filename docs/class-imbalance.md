# Task #12: Class imbalance experiment

**Status: experiment ready, results pending.** No arm has been trained on real folds yet,
so no strategy has been picked. Task #13 needs the pick. The result goes at the top of this
page when the runs finish.

## The question

About 81% of training chunks have no priority clause, and any one category is positive in
only 1% to 4% of chunks. Which of `pos_weight`, focal loss, or negative downsampling should
final training (task #13) use?

## What is compared

Every arm trains legal-bert (task #11, revision `15b570cb`) on the same five contract folds
(`docs/tuned-baseline-results/folds.csv`) with the same settings: 2 epochs, batch size 8,
learning rate 2e-5, 10% warmup, seed 42. Only the imbalance strategy changes.

| Arm | Change | Setting | Why this setting |
|---|---|---|---|
| `bce` | None. This is the control | | Shows whether any fix helps at all |
| `pos_weight` | Positive entries weigh more in BCE | Negatives/positives per category, from that fold's training rows only (about 25 to 86) | Standard definition, nothing to tune |
| `focal` | Easy entries count for less | gamma 2, alpha 0.25 | Published defaults (Lin et al. 2017) |
| `downsample` | Drop all-negative chunks from training | 2 negatives per positive chunk, redrawn each epoch | Keeps all positives and sees more negatives over a run. Validation is never downsampled |

Each strategy gets one setting, chosen in advance. There is no budget to tune each one, so
a loss here means "this setting lost", not "this method can never work".

## Decisions made before seeing results

These are fixed in [src/imbalance.py](../src/imbalance.py), so the pick cannot be fitted to
the numbers.

**Primary metric: macro F1 on held-out folds, at thresholds tuned on the other folds.**
These strategies shift predicted probabilities by very different amounts. At a shared 0.5
threshold we would mostly measure that shift. Tuning thresholds on the rows being scored
would flatter every arm, so each fold's thresholds come from the other folds only.

**Weak categories:** Change Of Control, Termination For Convenience, Exclusivity,
Non-Compete. These are the four lowest in the tuned TF-IDF cross-validation.

**Decision rule:**

1. Keep the fix with the best macro F1, plus any fix not clearly worse than it.
2. Of those, keep the best on weak-category F1, plus any not clearly worse than it.
3. If several remain, the data cannot separate them. Take the simplest, in this order:
   `pos_weight` (no settings, no data dropped), `focal` (two settings), `downsample` (drops data).

"Clearly" means the 95% interval on the paired difference excludes zero. The interval comes
from resampling contracts, because chunks from one contract are not independent.

The control cannot be picked, because the task is to choose a fix. The output still says how
the pick compares with it, so "no measurable gain over plain BCE" gets reported.

## Data: re-chunked for legal-bert

Task #11 chose legal-bert on condition that the data is re-chunked with its tokenizer. That
had not been done, and 801 of the BERT-sized chunks overflow 512 tokens under legal-bert. The
`prepare` step does it in `data/interim/imbalance/`. Default chunks, shared code defaults, and
the baseline results are unchanged.

| Training split | BERT chunks | legal-bert chunks |
|---|---:|---:|
| Chunks | 14,266 | 14,023 |
| Longest chunk, with special tokens | 512 | 512 |
| Fully contained annotations (of 3,241) | 3,229 | 3,228 |
| Chunks with a positive label | 2,659 | 2,644 |

Saved folds assign whole contracts, so they still apply. Fold sizes are 2,786 to 2,820 chunks.

**Task #13:** train on these chunks, and make the test split with the same tokenizer. The
`prepare` step only builds the training split.

## Run it

Open [notebooks/imbalance_colab.ipynb](../notebooks/imbalance_colab.ipynb) in Colab with a GPU
runtime. It saves each finished run to Google Drive and skips finished runs when restarted, so
a disconnect costs at most the run in progress.

**Time.** 20 runs. Task #11 measured about 9 chunks per second on a T4 without mixed precision,
which gives about 44 minutes per full arm, 25 for `downsample`, and **about 13 hours in total**.
This pipeline adds mixed precision, which should be faster, but that has not been measured. The
notebook's first runs show the real speed.

A free Colab session will not last that long. Teammates can share the work: each person sets
`FOLDS` to different folds, and `summarize` accepts several run directories. The notebook runs
every arm on one fold before starting the next, and only folds finished by all four arms are
compared, so stopping early still gives a fair comparison on fewer folds. At least two folds are
required, and the result records how many were used.

The same steps from a shell:

```bash
uv run --locked python -B -m scripts.imbalance_experiment prepare
uv run --locked python -B -m scripts.imbalance_experiment train --arm focal --validation-fold 0 --output-dir RUNS
uv run --locked python -B -m scripts.imbalance_experiment summarize --runs-dir RUNS --output-dir docs/class-imbalance-results
```

`summarize` writes these files. The decision is in `metadata.json`.

- `arm_summary.csv`: one row per arm
- `per_category.csv`: precision, recall, F1, and average precision per arm and category
- `per_fold.csv`: scores per fold
- `pairwise_bootstrap.csv`: differences between arms with 95% intervals
- `thresholds.csv`: thresholds used for each held-out fold
- `metadata.json`: the decision, folds used, and input hashes

## Limitations

- One seed. Folds and contract resampling measure data variation, not training randomness.
- Short training. Two epochs with a fixed learning rate may favor a strategy that learns
  quickly. Task #13's sweep could change the absolute scores.
- Each fold's thresholds are tuned on predictions from models that trained on that fold. This
  is much less biased than tuning on the scored rows, but not fully independent.
- These scores use different chunks from task #9's baseline, so a comparison is only rough. For
  that comparison, `macro_f1_thresholds_tuned_in_sample` uses the same tuning method as the
  baseline's 0.6729. Task #14 owns the formal comparison.
- Downsampling and `pos_weight` inflate probabilities. Thresholds correct the decisions but not
  the probabilities, so `validation_bce_mean` is reported in case November's risk scoring uses
  probabilities directly.

## Verification

```bash
uv run --locked python -B -m pytest -q -p no:cacheprovider
uv run --locked ruff check --no-cache src/imbalance.py src/loss.py scripts/imbalance_experiment.py tests/test_imbalance.py
```

Done on CPU: all four arms ran 4 optimizer steps with the real legal-bert weights on a few
contracts, and `summarize` refused those runs because they are smoke tests. The mixed-precision
path needs a GPU, so it has not been run yet.
