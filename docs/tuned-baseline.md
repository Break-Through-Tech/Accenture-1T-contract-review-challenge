# Task #9: Shared eval module + tuned baseline

Two things, both aimed at October. First, one scoring module that every model in
this project uses, so the transformer and the two baselines are compared on
identical code. Second, a decision threshold per category instead of one shared
0.5, which raises the baseline the transformer has to beat.

## Why

Scoring started inside the TF-IDF baseline. Task #8 then imported `score` from
task #7's module, and October's transformer would have been the third caller of
a function that lives in a model. The metric definitions, the thresholds and the
cross-validation folds are not properties of a model; they belong to the
comparison. They now live in [src/evaluation.py](../src/evaluation.py).

The thresholds matter for a separate reason. The task #7 numbers used 0.5 for
all ten categories, which [docs/tfidf-baseline.md](tfidf-baseline.md) lists as a
limitation. A single threshold is too low for a category whose vocabulary is
shared with another and too high for one with few positive chunks. Beating a
baseline that is handicapped by an untuned knob proves nothing, so the knob is
tuned before the transformer is measured against it.

## Run it

Needs the task #6 outputs for both splits. Outputs are ignored by Git except the
tables under `docs/`:

```bash
uv sync --locked
uv run --locked python -B -m scripts.export_chunks --split train
uv run --locked python -B -m scripts.label_chunks  --split train
uv run --locked python -B -m scripts.export_chunks --split test
uv run --locked python -B -m scripts.label_chunks  --split test
uv run --locked python -B -m scripts.tuned_baseline
```

The last command takes about three minutes and writes to
`docs/tuned-baseline-results/`:

- `folds.csv`: the contract-to-fold assignment, committed on purpose (see below)
- `thresholds.csv`: the chosen threshold per category, with the F1 it gained
- `train_cv_default.csv`, `train_cv_tuned.csv`: out-of-fold metrics, both ways
- `test_default.csv`, `test_tuned.csv`: test metrics, both ways
- `metadata.json`: config, input hashes, counts, grid, chosen thresholds

It also writes `data/interim/models/tfidf_oof_scores.parquet` (ignored by Git):
the out-of-fold score and fold number for every training chunk, which is what a
later model needs to tune its own thresholds on the same folds.

## The shared module

| Function | Purpose |
|---|---|
| `score` | Per-category precision/recall/F1/AP plus macro and micro rows |
| `resolve_thresholds` | Expands a float, a mapping or a sequence to one value per category |
| `apply_thresholds` | Scores to 0/1 predictions, per-category cutoffs |
| `tune_thresholds` | Picks the F1-maximising threshold per category |
| `threshold_map` | Tuning table to a `{category: threshold}` mapping |
| `contract_folds` | Contract-to-fold assignment, grouped so chunks cannot leak |
| `fold_of_rows` | Fold number per chunk row, from a saved `folds.csv` |

`score` now reports the threshold each row used, so a saved result always states
how its decisions were made. The macro and micro rows leave the threshold empty
when the categories do not share one, rather than implying a single value.

Masking is unchanged from task #6: a (chunk, category) pair whose clause is only
partly inside the chunk is excluded for that category, and the chunk still counts
for the other nine.

`src.tfidf_baseline` re-exports `score` and `METRIC_COLUMNS` so its existing
callers keep working. `src.keyword_baseline` now imports from
`src.evaluation` directly.

## Method

| Step | Choice |
|---|---|
| Scores tuned on | Out-of-fold predictions on the 408 training contracts |
| Folds | 5, grouped by contract, `GroupKFold`, no shuffle |
| Grid | 0.05 to 0.95 in steps of 0.01 (91 candidates) |
| Objective | F1 per category, each category independently |
| Ties | Broken towards the lower threshold |
| Test split | Scored once, at thresholds already frozen |

Each training chunk is scored by a model fitted without its contract, so the
128-token overlap between a contract's chunks cannot leak into the scores the
thresholds are chosen on.

Ties go to the lower threshold because equal F1 at two cutoffs means the lower
one trades precision for recall, and this project treats a missed clause as
worse than one a reviewer dismisses — the overview names High-Risk Recall as the
metric that matters most for triage.

### Why folds.csv is committed

`GroupKFold` does not shuffle, so the assignment depends only on which contracts
are present and how many chunks each has — the same table every run, no seed to
record. It is committed anyway, for two reasons. It makes October's comparison
exact: the transformer can be cross-validated on these folds with
`fold_of_rows`, rather than on a fresh partition that happens to use the same
algorithm. And it is a readable record of the split, so a fold's contracts can
be inspected when a category behaves oddly in one fold.

408 rows, one per training contract, with its fold and chunk count.

## Chosen thresholds

From out-of-fold predictions on the 408 training contracts. `f1_gain` is the
change against 0.5 on those same predictions.

| Category | Threshold | CV F1 at 0.5 | CV F1 tuned | Gain |
|---|---:|---:|---:|---:|
| Cap On Liability | 0.50 | 0.778 | 0.778 | 0.000 |
| Uncapped Liability | 0.58 | 0.641 | 0.659 | +0.017 |
| Governing Law | 0.50 | 0.941 | 0.941 | 0.000 |
| Anti-Assignment | 0.57 | 0.817 | 0.828 | +0.011 |
| Change Of Control | 0.45 | 0.499 | 0.520 | +0.022 |
| Termination For Convenience | 0.49 | 0.567 | 0.567 | +0.000 |
| Exclusivity | 0.28 | 0.430 | 0.478 | +0.048 |
| Non-Compete | 0.17 | 0.295 | 0.411 | **+0.116** |
| Renewal Term | 0.54 | 0.763 | 0.763 | +0.001 |
| Audit Rights | 0.33 | 0.784 | 0.784 | +0.016 |
| **Macro** | | **0.650** | **0.673** | **+0.023** |

Three categories kept 0.5, and the two largest moves are both downward, towards
recall, on the categories the baseline was worst at.

## What happened on test

Tuning worked on the folds it was tuned on and did not transfer.

| Split | Macro F1 at 0.5 | Macro F1 tuned | Micro F1 at 0.5 | Micro F1 tuned |
|---|---:|---:|---:|---:|
| Training, out-of-fold | 0.650 | **0.673** | 0.718 | 0.714 |
| Official test | **0.683** | 0.666 | **0.716** | 0.692 |

Per category on test, tuned minus 0.5:

| Category | F1 at 0.5 | F1 tuned | Change |
|---|---:|---:|---:|
| Non-Compete | 0.725 | 0.599 | **−0.127** |
| Exclusivity | 0.511 | 0.486 | −0.026 |
| Audit Rights | 0.720 | 0.696 | −0.024 |
| Uncapped Liability | 0.667 | 0.652 | −0.014 |
| Termination For Convenience | 0.554 | 0.544 | −0.011 |
| Renewal Term | 0.824 | 0.818 | −0.005 |
| Cap On Liability | 0.736 | 0.736 | 0.000 |
| Governing Law | 0.868 | 0.868 | 0.000 |
| Anti-Assignment | 0.788 | 0.789 | +0.001 |
| Change Of Control | 0.442 | 0.477 | **+0.035** |

**Non-Compete is three quarters of the loss.** Task #7 had already flagged it as
the unstable category: F1 0.295 in cross-validation against 0.725 on test. The
tuner saw only the 0.295, dropped its threshold from 0.50 to 0.17 for the
largest CV gain of any category, and on test that turned a category which was
already the baseline's third best into its worst trade — predicted positives
went from 38 to 94 and precision from 0.868 to 0.468, buying recall that was not
needed.

Change Of Control is the one clear win, +0.035, and it is the category whose CV
and test scores agreed beforehand (0.499 and 0.442).

So the lesson is not that threshold tuning is wrong. It is that a threshold
tuned on a category whose cross-validation and test behaviour already disagree
inherits that disagreement, and F1 on 408 contracts is not a stable enough
target to move a cutoff 0.33 on 198 positive chunks.

## The bar for October

**Macro F1 0.683, micro F1 0.716, at a shared 0.5 threshold** — the task #7
numbers, unchanged. Tuning was tried honestly and lost, so the untuned
configuration stands, and it now stands on evidence rather than on nobody having
checked.

Per category, the bar is the `f1` column of
[test_default.csv](tuned-baseline-results/test_default.csv). The transformer
should beat Change Of Control (0.442), Exclusivity (0.511) and Termination For
Convenience (0.554) comfortably; Governing Law (0.868) and Renewal Term (0.824)
are close to what word counts can do and will be harder to improve on.

For the transformer's own thresholds, keep the machinery and add a guard:

- Tune on these folds with `fold_of_rows`, so the comparison is exact.
- Before accepting a threshold, check the gain holds on a fold the threshold was
  not chosen on. A gain that appears in aggregate but not per fold is noise.
- Treat a large move on a low-support category as a red flag, not a win.
  Non-Compete has 198 positive training chunks and 53 on test.

## Limitations

- CV F1 at tuned thresholds (0.673) is optimistic: the thresholds were chosen on
  the same out-of-fold predictions they are scored on. Nested cross-validation
  would remove that bias at five times the compute. The test column is the
  unbiased one, which is why the conclusion rests on it.
- F1 weights precision and recall equally. The project's own evaluation section
  says High-Risk Recall matters more for triage, so a recall-weighted objective
  (Fβ with β > 1) is defensible and untried here.
- Thresholds are tuned per category independently. Categories that share
  vocabulary — Cap On Liability with Uncapped Liability, Exclusivity with
  Non-Compete — are not prevented from both firing on the same chunk.
- Everything is still scored per chunk, not per contract or per annotated
  clause, and five of the 737 test annotations are not fully inside any chunk.
- Folds are the same every run only while the set of training contracts is the
  same. Re-exporting task #6 with a different contract set changes them.

## Verification

```bash
uv run --locked python -B -m pytest -q -p no:cacheprovider
uv run --locked ruff check --no-cache src/evaluation.py scripts/tuned_baseline.py tests/test_evaluation.py
```

28 tests cover threshold resolution from all three input forms, per-category
cutoffs, the reported threshold column, masked-pair exclusion, the tuning
objective and its tie-break, a category with no positives, fold determinism
under row reordering, and the round trip through a saved `folds.csv`.
