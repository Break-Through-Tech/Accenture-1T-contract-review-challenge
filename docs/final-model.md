# Task #13: Final training + single test run

**Status:** done. Final model trained and scored once on the official test split
(12 Nov 2026).

**Headline:** on the test split (102 contracts, 3,029 chunks), legal-bert with pos_weight and
per-category thresholds reaches **macro F1 0.803**, micro F1 0.829 and macro average precision
0.831. The TF-IDF baseline bar is macro F1 0.683 (see Results for why that comparison is rough).

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

**How the reported run was done:** a first Colab attempt was lost when the laptop slept
(Google Drive could not be mounted, so nothing was saved). The reported run used a Kaggle
notebook (Tesla T4, "Save & Run All") that executes the same commands in one cell:
`prepare`, `sweep` for folds 0–4, `select`, `final`, `test`. Same code (branch
`task-13-final-training`, commit `e889adb`), same data hashes (`train_matches_task12: true`).

## Sweep

- **Grid** (fixed before running): learning rate 2e-5 and 3e-5, epochs 1 to 3.
- **As run:** learning rate **2e-5 only**, epochs 1 to 3 (5 runs, 3 configs). 3e-5 was dropped
  to fit the deadline after the first attempt was lost, before any results were seen. 2e-5 is
  the rate Task #12 used.
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

Also committed, for Tasks #14 and #15:

- `test_predictions.parquet`: contract, chunk ID and the 10 category probabilities for every
  test chunk (0.2 MB). Apply the thresholds in `selection.json` to get decisions.

Not in Git:

- **Model weights** (`final_model/checkpoint/`, about 440 MB): kept in the Kaggle notebook's
  Output.
- **Out-of-fold training predictions:** were not exported from the Kaggle run.

## Results

### Sweep (5-fold CV on training contracts, cross-fitted thresholds)

| Config | Macro F1 | Weak-category F1 | Macro AP | Micro F1 |
|---|---:|---:|---:|---:|
| 2e-5, 1 epoch | 0.684 | 0.524 | 0.695 | 0.742 |
| 2e-5, 2 epochs | 0.755 | 0.656 | 0.777 | 0.793 |
| **2e-5, 3 epochs (picked)** | **0.765** | **0.669** | **0.786** | **0.806** |

3 epochs beat 2 epochs on macro F1 by 0.010 (95% interval 0.001 to 0.019, resampling
contracts), so step 1 of the rule decided it: no config was tied with the leader. Weak
categories are Change Of Control, Exclusivity and Termination For Convenience.

### Official test split (scored once)

102 contracts, 3,029 chunks, 721 positive (chunk, category) pairs. Thresholds frozen from CV.

| Category | Threshold | Support | Precision | Recall | F1 | AP |
|---|---:|---:|---:|---:|---:|---:|
| Cap On Liability | 0.93 | 105 | 0.863 | 0.838 | 0.850 | 0.866 |
| Uncapped Liability | 0.94 | 16 | 0.846 | 0.688 | 0.759 | 0.836 |
| Governing Law | 0.71 | 126 | 0.898 | 0.976 | 0.935 | 0.932 |
| Anti-Assignment | 0.88 | 129 | 0.878 | 0.837 | 0.857 | 0.912 |
| Change Of Control | 0.68 | 56 | 0.818 | 0.643 | 0.720 | 0.721 |
| Termination For Convenience | 0.90 | 43 | 0.762 | 0.744 | 0.753 | 0.792 |
| Exclusivity | 0.88 | 71 | 0.671 | 0.718 | 0.694 | 0.687 |
| Non-Compete | 0.86 | 53 | 0.737 | 0.792 | 0.764 | 0.813 |
| Renewal Term | 0.95 | 32 | 0.763 | 0.906 | 0.829 | 0.883 |
| Audit Rights | 0.95 | 90 | 0.914 | 0.822 | 0.866 | 0.868 |
| **Macro average** | | 721 | **0.815** | **0.797** | **0.803** | **0.831** |
| **Micro average** | | 721 | 0.833 | 0.824 | **0.829** | |

At a plain 0.5 threshold the same predictions give macro F1 0.768 (precision 0.706, recall
0.857), so the CV-tuned thresholds add about 0.035 macro F1, mostly by cutting false positives.

### Against the baseline

| Model | Test macro F1 | Test micro F1 | Test macro AP |
|---|---:|---:|---:|
| TF-IDF + logistic regression, 0.5 threshold (Task #7/#9 bar) | 0.683 | 0.716 | 0.742 |
| TF-IDF, tuned thresholds (Task #9) | 0.666 | 0.692 | 0.742 |
| **legal-bert + pos_weight, tuned thresholds (this task)** | **0.803** | **0.829** | **0.831** |

The baseline was scored on BERT-tokenizer chunks (3,096 test chunks) and this model on
legal-bert chunks (3,029), so this is a rough comparison. The exact side-by-side on shared
contracts, with confidence intervals and contract-level metrics, is Task #14.

### What the numbers say

- **The weakest categories improved the most.**
  - **Change Of Control:** 0.442 → 0.720 F1 versus TF-IDF.
  - **Termination For Convenience:** 0.554 → 0.753.
  - **Exclusivity:** 0.511 → 0.694, but it is still the weakest category, with AP 0.687.
- **Formulaic clauses stay strong.** Governing Law 0.935, Audit Rights 0.866, Anti-Assignment
  0.857.
- **Uncapped Liability** has only 16 positive test chunks. Its 0.759 F1 rests on 13 predictions,
  so treat it as noisy.
- **Non-Compete** again scores higher on test (0.764) than in CV (0.617), the same gap the
  baseline showed (Task #9). It looks like a property of the split, not of the model, and is
  left to Task #15.
- **Test vs CV.** Test macro F1 (0.803) is above CV (0.765). Two reasons:
  - The final model saw all 408 contracts, while each fold model saw about 326.
  - The test split has fewer positives in the rare categories, so their scores move more.

### Threshold flags

From `thresholds.csv`. All flagged thresholds were kept as tuned; none was changed after seeing
test results.

| Category | Flag | Detail |
|---|---|---|
| Anti-Assignment | unstable threshold | Leave-one-fold-out thresholds range 0.55–0.88 (std 0.18). F1 is flat across that range (0.844 at 0.5 vs 0.850 tuned), so the choice matters little |
| Uncapped Liability, Change Of Control, Non-Compete | low support | Fewer than 200 positive training chunks |

## Compute

Measured on the Kaggle T4 run:

| Step | Time |
|---|---|
| Setup + prepare | ~3 min |
| Sweep: 5 runs × 3 epochs | ~23 min per run, ~1 h 57 min total |
| Final model (3 epochs, all 408 contracts) | ~25 min |
| Test | ~1 min |
| **Total** | **~2 h 30 min** |

To share the sweep, each person sets `FOLDS` in the notebook to different folds, then one person
copies the others' `sweep` folders into their Drive and passes all of them to `--runs-dir`.

## Limitations

- One seed. Seed noise is not measured.
- Only one learning rate was run (see Sweep).
- Only 3 epoch settings. Other settings (batch size, warmup, weight decay) are carried
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
