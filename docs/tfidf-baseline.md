# Task #7: TF-IDF baseline

A reproducible multi-label classifier over the task #6 chunks. It is the floor that the
transformer models have to beat, and a check that the labels are learnable at all.

## Run it

The classifier reads the labeled chunk tables from task #6. Build them for both splits
first (outputs are ignored by Git):

```bash
uv sync --locked
uv run --locked python -B -m scripts.export_chunks --split train
uv run --locked python -B -m scripts.label_chunks  --split train
uv run --locked python -B -m scripts.export_chunks --split test
uv run --locked python -B -m scripts.label_chunks  --split test
uv run --locked python -B -m scripts.tfidf_baseline
```

The last command takes about two and a half minutes and writes:

- `docs/tfidf-baseline-results/test_per_category.csv`
- `docs/tfidf-baseline-results/train_cv_per_category.csv`
- `docs/tfidf-baseline-results/top_terms.csv`
- `docs/tfidf-baseline-results/metadata.json`
- `data/interim/models/tfidf_baseline.joblib` and `test_scores.parquet` (ignored by Git)

## Method

| Step | Choice |
|---|---|
| Input | Chunk `text`, exactly as exported in task #6 |
| Features | TF-IDF, word unigrams and bigrams, lowercased, `min_df=2`, `max_df=0.95`, sublinear TF |
| Stop words | Kept. Words such as "not", "shall" and "without" carry meaning in clauses |
| Model | One logistic regression per category (one-vs-rest), `C=10`, liblinear, seed 42 |
| Imbalance | `class_weight="balanced"` |
| Decision threshold | 0.5 for every category |
| Categories | The ten in `PRIORITY_CATEGORIES`, in that order |

The vocabulary is fitted on training chunks only (324,756 terms). Settings are fixed in
`BaselineConfig`; none of them was tuned, on either split.

Task #6 masks (chunk, category) pairs where a clause is only partly inside the chunk.
Those pairs are left out of fitting and scoring for that category. The chunk still
counts for the other nine.

## Results on the official test split

102 contracts, 3,096 chunks. Support is the number of positive chunks.

| Category | Support | Precision | Recall | F1 | Avg. precision |
|---|---:|---:|---:|---:|---:|
| Cap On Liability | 107 | 0.826 | 0.664 | 0.736 | 0.818 |
| Uncapped Liability | 17 | 0.516 | 0.941 | 0.667 | 0.659 |
| Governing Law | 116 | 0.812 | 0.931 | 0.868 | 0.931 |
| Anti-Assignment | 134 | 0.816 | 0.761 | 0.788 | 0.846 |
| Change Of Control | 58 | 0.679 | 0.328 | 0.442 | 0.573 |
| Termination For Convenience | 44 | 0.491 | 0.636 | 0.554 | 0.586 |
| Exclusivity | 68 | 0.523 | 0.500 | 0.511 | 0.567 |
| Non-Compete | 53 | 0.868 | 0.623 | 0.725 | 0.781 |
| Renewal Term | 31 | 0.757 | 0.903 | 0.824 | 0.864 |
| Audit Rights | 90 | 0.797 | 0.656 | 0.720 | 0.795 |
| **Macro average** | 718 | 0.709 | 0.694 | 0.683 | 0.742 |
| **Micro average** | 718 | 0.739 | 0.694 | 0.716 | |

## Results on the training split, 5-fold by contract

408 contracts, 14,266 chunks. Folds are grouped by contract because chunks from one
contract overlap by 128 tokens. Use these numbers, not the test ones, for any tuning.

| Category | Support | Precision | Recall | F1 | Avg. precision |
|---|---:|---:|---:|---:|---:|
| Cap On Liability | 532 | 0.844 | 0.722 | 0.778 | 0.836 |
| Uncapped Liability | 150 | 0.570 | 0.733 | 0.641 | 0.649 |
| Governing Law | 495 | 0.921 | 0.962 | 0.941 | 0.978 |
| Anti-Assignment | 512 | 0.836 | 0.799 | 0.817 | 0.861 |
| Change Of Control | 181 | 0.546 | 0.459 | 0.498 | 0.497 |
| Termination For Convenience | 236 | 0.512 | 0.636 | 0.567 | 0.595 |
| Exclusivity | 315 | 0.518 | 0.368 | 0.430 | 0.460 |
| Non-Compete | 198 | 0.633 | 0.192 | 0.295 | 0.383 |
| Renewal Term | 210 | 0.777 | 0.748 | 0.762 | 0.822 |
| Audit Rights | 425 | 0.817 | 0.725 | 0.768 | 0.843 |
| **Macro average** | 3,254 | 0.697 | 0.634 | 0.650 | 0.692 |
| **Micro average** | 3,254 | 0.753 | 0.686 | 0.718 | |

## What the numbers say

- Categories with fixed wording score highest: Governing Law, Renewal Term,
  Anti-Assignment, Cap On Liability. Their top terms are the phrases a reviewer would
  search for ("governed by", "laws of", "assign", "consequential").
- Change Of Control, Exclusivity and Termination For Convenience are the weakest on
  both splits. These depend on who may do what, which word counts do not capture.
- Uncapped Liability shares its vocabulary with Cap On Liability ("damages",
  "consequential"), so recall is high and precision is about one in two.
- Non-Compete is unstable: F1 is 0.295 in cross-validation and 0.725 on test. Treat its
  test score as optimistic.
- Uncapped Liability has only 17 positive test chunks and Renewal Term 31, so their test
  scores move a lot with a few chunks.

## Limitations

- Scores are per chunk, not per contract or per annotated clause. One clause can be
  positive in two overlapping chunks.
- Five of the 737 test annotations are not fully inside any chunk, so no chunk-level
  model can be credited for them.
- A single 0.5 threshold is used for all categories. Task #9 tuned one threshold per
  category on the cross-validation folds: it gained 0.023 macro F1 there and lost
  0.017 on test, so these numbers stand as the baseline. See
  [the tuned baseline](tuned-baseline.md).
- Bag-of-words ignores word order beyond two-word phrases, negation scope and defined
  terms.

## Verification

```bash
uv run --locked python -B -m pytest -q -p no:cacheprovider
uv run --locked ruff check --no-cache src/tfidf_baseline.py scripts/tfidf_baseline.py tests/test_tfidf_baseline.py
```
