# Task #13: Final training + single test run

**Status:** pipeline built and tested offline. GPU runs pending. Fill in the Results section
from `docs/final-model-results/` after the Colab run.

## What it does

Trains the final contract-clause classifier with everything earlier tasks decided, then scores
the official test split exactly once.

| Setting | Value | Decided in |
|---|---|---|
| Encoder | `nlpaueb/legal-bert-base-uncased`, revision `15b570cb` | Task #11 |
| Chunks | Re-chunked with legal-bert's tokenizer, train **and** test | Task #11/#12 |
| Imbalance fix | `pos_weight` (negatives/positives per category, from training rows only) | Task #12 |
| Folds | `docs/tuned-baseline-results/folds.csv`, 5 folds grouped by contract | Task #9 |
| Thresholds | One per category, tuned on out-of-fold predictions | Task #9 |
| Batch size, warmup, weight decay | 8, 10% linear warmup then linear decay, 0.01 | Task #12 |

## Steps

| Step | Command | Uses test data? |
|---|---|---|
| prepare | `python -m scripts.final_training prepare` | Chunks it, does not score it |
| sweep | `... sweep --learning-rate LR --validation-fold F --output-dir RUNS` | No |
| select | `... select --runs-dir RUNS` | No |
| final | `... final --output-dir FINAL` | No |
| test | `... test --checkpoint FINAL/checkpoint --output-dir FINAL` | **Yes, once** |

Run it all from [notebooks/final_training_colab.ipynb](../notebooks/final_training_colab.ipynb)
on a T4. Everything is saved to Google Drive and finished runs are skipped, so after a
disconnect you just run all cells again.

## Sweep

- **Grid** (fixed before running): learning rate 2e-5 and 3e-5, epochs 1 to 3.
- Each (learning rate, fold) run trains for 3 epochs and saves held-out predictions after every
  epoch, so one run covers 3 epoch settings. That makes 10 runs for 6 configs.
- The learning-rate schedule always spans 3 epochs. The final model uses that same schedule and
  stops at the chosen epoch, so it is trained the same way as the config that was scored. A test
  checks this exactly.

**Scoring a config:** cross-fitted macro F1, where each fold's thresholds are tuned on the other
4 folds (same method as Task #12), so the number isn't flattered by tuning on the rows being
scored.

**Picking a config** (rule fixed before running, in `src/final_training.select_config`):

1. The leader is the config with the best cross-fitted macro F1.
2. Keep every config that is not clearly worse than the leader. "Clearly worse" means the 95%
   interval from resampling contracts excludes 0.
3. Of those, take the fewest epochs, then the lowest learning rate.

Step 3 prefers the cheaper, less-overfit option when the data can't tell configs apart.

## Thresholds

Tuned per category on the chosen config's out-of-fold predictions (grid 0.01 to 0.99).
pos_weight pushes probabilities up, so expect thresholds well above 0.5 (Task #12 saw 0.65 to
0.97).

Following Task #9's Non-Compete lesson, `thresholds.csv` also shows how much each category's
threshold moves when one fold is left out, and flags:

- **unstable threshold**: fold-to-fold standard deviation above 0.10
- **low support**: fewer than 200 positive training chunks

Flagged thresholds are still used, because changing them after seeing test results would be
tuning on test. They are reported as a caveat.

## Final model and the single test run

`final` trains on all 408 training contracts and saves to Drive (not Git):

- `checkpoint/encoder/`: legal-bert weights
- `checkpoint/head.pt`: classification head
- `checkpoint/tokenizer.json`
- `checkpoint/classifier.json`: category order, thresholds, settings

It reloads the saved checkpoint and checks that it gives the same predictions before marking
the run finished.

`test` scores the 102 test contracts at the frozen thresholds, and at 0.5 for reference. It
writes `test_run.json` and **refuses to run again** after that, so there is exactly one reported
test result. Smoke runs can't write to `docs/final-model-results/`.

## Outputs

To commit (`docs/final-model-results/`):

| File | Contents |
|---|---|
| `sweep_summary.csv` | Cross-fitted macro F1, weak-category F1, AP for all 6 configs |
| `sweep_pairwise_bootstrap.csv` | Config-vs-config differences with 95% intervals |
| `selection.json` | Picked config, why, thresholds, flags, data hashes |
| `cv_per_category.csv` | Per-category CV metrics of the picked config |
| `thresholds.csv` | Final thresholds with fold stability and flags |
| `test_per_category.csv` | **Test result** at frozen thresholds |
| `test_at_0.5.csv` | Test result at 0.5, for reference only |
| `test_run.json` | When the test was scored, headline numbers, lock |

Stays in Drive (share with Tasks #14 and #15):

- `final_model/checkpoint/`
- `final_model/test_predictions.parquet`: chunk ID plus 10 probabilities
- `sweep/oof_predictions_selected.parquet`: out-of-fold training predictions

## Results

_To fill in after the run._

| | Macro F1 | Micro F1 | Macro AP |
|---|---:|---:|---:|
| TF-IDF baseline, test (Task #7/#9 bar) | 0.683 | 0.716 | 0.742 |
| legal-bert, CV (chosen config) | | | |
| legal-bert, **test** | | | |

The test chunks differ from the baseline's (legal-bert vs BERT tokenizer), so this table is a
rough comparison. The exact side-by-side, with confidence intervals and contract-level metrics,
is Task #14.

## Compute

On a Colab T4, Task #12 took about 6 minutes per training epoch on one fold.

| Step | Estimate |
|---|---|
| Setup + prepare | ~15 min |
| Sweep: 10 runs × 3 epochs | ~3–3.5 h |
| Final model | ~25 min |
| Test | ~3 min |

To share the sweep, each person sets `FOLDS` in the notebook to different folds, then one person
copies the others' `sweep` folders into their Drive and passes all of them to `--runs-dir`.

## Limitations

- One seed. Seed noise is not measured.
- Only 2 learning rates × 3 epochs. Other settings (batch size, warmup, weight decay) are carried
  over from Task #12, not tuned.
- Thresholds are tuned on predictions from fold models, but applied to a model trained on all
  408 contracts. That model saw 25% more data, so its probabilities can shift slightly.
- pos_weight makes probabilities too high. Thresholds fix the decisions, not the probabilities;
  calibrate before using them as risk scores.
- Everything is scored per chunk.

## Checks

```bash
uv run --locked python -B -m pytest -q -p no:cacheprovider
uv run --locked ruff check --no-cache src/final_training.py scripts/final_training.py tests/test_final_training.py
```

These cover:

- per-epoch saving and resume
- the epoch-snapshot/final-run equivalence
- pos_weight from training rows only
- the selection rule (clear winner, ties)
- threshold flags
- the end-to-end sweep, select, final and test path
- the checkpoint round trip
- the one-test-run lock and smoke-run guards

`prepare` was also run on the real CUAD files (408 train / 102 test contracts) with a stand-in
BERT tokenizer and reproduced Task #7's label files exactly. Colab uses legal-bert's tokenizer,
and the step prints `train_matches_task12` to confirm the chunks match Task #12's.
