# Task #12: Class imbalance

## Result: use pos_weight

We tested three ways of handling the class imbalance (pos_weight, focal loss and downsampling
negatives) plus plain BCE as a baseline. pos_weight won on macro F1 and on the weak categories,
so that's what task #13 should use.

All 4 options were trained on all 5 folds (20 runs, 1 epoch each). Full results are in
[class-imbalance-results/](class-imbalance-results/).

| Option | Macro F1 | Weak-category F1 | Macro avg precision |
|---|---:|---:|---:|
| bce (no fix) | 0.568 | 0.278 | 0.583 |
| **pos_weight** | **0.697** | **0.550** | **0.716** |
| focal | 0.665 | 0.477 | 0.693 |
| downsample | 0.603 | 0.350 | 0.595 |

How much pos_weight beat each one by (95% interval from resampling contracts in brackets):

| vs | Macro F1 | Weak-category F1 |
|---|---|---|
| bce | +0.129 [0.113, 0.144] | +0.272 [0.240, 0.303] |
| focal | +0.032 [0.019, 0.045] | +0.073 [0.047, 0.100] |
| downsample | +0.094 [0.080, 0.108] | +0.200 [0.169, 0.231] |

None of the intervals include 0, and pos_weight had the best macro F1 on every fold, so it
wasn't just one good split. Since it was clearly best on macro F1, the decision rule (below)
picked it at step 1.

Weak categories (F1):

| Category | bce | pos_weight | focal | downsample | TF-IDF baseline (task #9) |
|---|---:|---:|---:|---:|---:|
| Change Of Control | 0.19 | **0.42** | 0.36 | 0.26 | 0.52 |
| Termination For Convenience | 0.48 | **0.64** | 0.64 | 0.51 | 0.57 |
| Exclusivity | 0.33 | **0.61** | 0.52 | 0.44 | 0.48 |
| Non-Compete | 0.11 | **0.52** | 0.38 | 0.18 | 0.41 |

With no fix, the model basically never finds Non-Compete or Change Of Control. pos_weight
helped those the most. The easy categories (Governing Law, Anti-Assignment, Audit Rights)
were about the same for every option.

### Notes for task #13

- Use `make_loss(ARMS["pos_weight"], ...)` from [src/imbalance.py](../src/imbalance.py), with
  the weights calculated from the training data only.
- Tune a threshold per category. pos_weight makes the predicted probabilities higher, so the
  best thresholds end up around 0.65 to 0.97 (see `thresholds.csv`). Just using 0.5 drops macro
  F1 by about 0.07.
- The probabilities themselves are off (validation BCE 0.090 vs 0.050 for plain BCE). If the
  risk scoring later uses the probabilities directly, they'll need to be calibrated.
- Change Of Control is still worse than the TF-IDF baseline (0.42 vs 0.52).
- For a rough comparison with the baseline: with thresholds tuned the same way the baseline's
  0.673 was, pos_weight gets 0.709. The chunks aren't the same though, so the real comparison
  is task #14's job.

## Setup

About 81% of training chunks don't have any priority clause, and each category shows up in
only 1 to 4% of chunks. That's the imbalance we're trying to fix.

Every option trains legal-bert (from task #11, revision `15b570cb`) on the same 5 folds from
`docs/tuned-baseline-results/folds.csv`, with batch size 8, learning rate 2e-5, 10% warmup and
seed 42. We planned on 2 epochs but ran 1 to save time. The only thing that changes between
runs is how the imbalance is handled.

| Option | What it does | Setting |
|---|---|---|
| bce | Nothing, it's the baseline | |
| pos_weight | Positives count more in the loss | negatives/positives for each category, from the training fold (about 25 to 86) |
| focal | Easy examples count less | gamma 2, alpha 0.25 (defaults from Lin et al. 2017) |
| downsample | Drops some chunks with no labels from training | 2 negative chunks per positive chunk, re-sampled every epoch. Validation isn't downsampled |

We only tried one setting for each option, so if one loses it just means that setting lost.

## How the pick was made

We decided all of this before running anything (it's in
[src/imbalance.py](../src/imbalance.py)) so we couldn't pick based on what looked good.

**Main metric:** macro F1 on the held-out fold. Each fold's thresholds are tuned on the other
folds. We didn't use a flat 0.5 threshold because each option shifts the probabilities by a
different amount, and we didn't tune on the same rows we score because that inflates the numbers.

**Weak categories:** Change Of Control, Termination For Convenience, Exclusivity and
Non-Compete, the 4 lowest scores in the TF-IDF baseline.

**Rule:**

1. Keep the option with the best macro F1, and any that aren't clearly worse.
2. Out of those, keep the best on the weak categories, and any that aren't clearly worse.
3. If there's still a tie, go with the simplest: pos_weight, then focal, then downsample.

"Clearly worse" means the 95% interval for the difference doesn't include 0. The intervals
come from resampling whole contracts, since chunks from the same contract aren't independent.

Plain BCE can't be picked since the task is to choose a fix, but we still compare against it.

## Data

Task #11 picked legal-bert as long as we re-chunk with its tokenizer, which hadn't been done
yet (801 of the old chunks are longer than 512 tokens with legal-bert). The `prepare` step does
that and saves to `data/interim/imbalance/`. The default chunks and the baseline results
aren't touched.

| Training split | Old BERT chunks | legal-bert chunks |
|---|---:|---:|
| Chunks | 14,266 | 14,023 |
| Longest chunk (with special tokens) | 512 | 512 |
| Annotations fully inside a chunk (of 3,241) | 3,229 | 3,228 |
| Chunks with at least one label | 2,659 | 2,644 |

The folds are split by contract so they still work. Each fold has 2,786 to 2,820 chunks.

Task #13 should train on these chunks and re-chunk the test split with the same tokenizer
(`prepare` only does the training split).

## How to run it

Open [notebooks/imbalance_colab.ipynb](../notebooks/imbalance_colab.ipynb) in Colab with a T4
GPU and run all. Finished runs get saved to Google Drive and skipped next time, so if Colab
disconnects you just run all again.

Each run took about 6 minutes (about 3.5 for downsample), so all 20 took under 2 hours plus
around 15 minutes of setup. You can also split folds between people by giving each person
different `FOLDS`, and `summarize` can take more than one runs folder. You need at least 2
folds finished by every option to get a result.

From a terminal:

```bash
uv run --locked python -B -m scripts.imbalance_experiment prepare
uv run --locked python -B -m scripts.imbalance_experiment train --arm focal --validation-fold 0 --output-dir RUNS
uv run --locked python -B -m scripts.imbalance_experiment summarize --runs-dir RUNS --output-dir docs/class-imbalance-results
```

`summarize` saves:

- `arm_summary.csv`: one row per option
- `per_category.csv`: precision, recall, F1 and average precision for each category
- `per_fold.csv`: scores for each fold
- `pairwise_bootstrap.csv`: differences between options with 95% intervals
- `thresholds.csv`: thresholds used on each fold
- `metadata.json`: the pick, which folds were used, and data hashes

## Limitations

- Only one seed, so we don't know how much random training noise matters.
- Only 1 epoch. That could favor whatever learns fastest, but pos_weight won every fold by a
  decent margin so we don't think 2 epochs would change the pick. The scores themselves will
  change in task #13.
- Thresholds for a fold are tuned on models that saw that fold in training, so they're not
  totally independent.
- The chunks are different from task #9's, so comparing with the baseline is only rough.
- pos_weight and downsampling make the probabilities too high. Tuned thresholds fix the
  predictions but not the probabilities.

## Checks

```bash
uv run --locked python -B -m pytest -q -p no:cacheprovider
uv run --locked ruff check --no-cache src/imbalance.py src/loss.py scripts/imbalance_experiment.py tests/test_imbalance.py
```

Tests and lint pass. A quick test run of all 4 options worked on CPU, and the full runs were
done on a Colab T4.
