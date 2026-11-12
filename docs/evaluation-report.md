# Task 14: Evaluation report

The baseline run compares one TF-IDF model at the default 0.5 threshold and
at the category thresholds saved by Task 9. The script fits on the official
training split and evaluates on the official test split. Thresholds stay frozen.
Pass the Task 13 files to add Legal-BERT to the contract-level comparison.

Run from the repository root with the project's dependencies installed:

```sh
uv run --locked python -B -m scripts.export_chunks --split train
uv run --locked python -B -m scripts.label_chunks --split train
uv run --locked python -B -m scripts.export_chunks --split test
uv run --locked python -B -m scripts.label_chunks --split test
uv run --locked python -B -m scripts.evaluation_report
```

Outputs in `data/interim/evaluation/`:

- `chunk_metrics.csv` uses Task 6's chunk labels and partial-clause masks.
- `contract_metrics.csv` takes the maximum score per category over all chunks
  in each contract. A category is positive in the ground truth if CUAD
  annotates it anywhere in the contract. This keeps the ground truth fixed
  when chunk boundaries change.
- `contract_f1_intervals.csv` gives percentile 95% intervals from 2,000
  contract bootstrap draws. All configurations use the same draws for each
  paired difference. With Task 13 inputs, this also includes Legal-BERT
  versus each TF-IDF configuration. These intervals reflect variation across
  test contracts with the fitted model and thresholds held fixed.
- `contract_pr_curves.svg` and `contract_pr_curves.csv` show the precision
  and recall tradeoff by category. Both TF-IDF configurations share each
  curve because they use the same scores. The dashed line shows prevalence.

The contract metrics evaluate whether a clause type is present somewhere in
a contract. Finding the exact passage is a separate evaluation question.

## Results on the 102 held-out contracts

| Configuration | Chunk macro F1 | Contract macro precision | Contract macro recall | Contract macro F1 | Contract micro F1 | Contract macro AP |
|---|---:|---:|---:|---:|---:|---:|
| TF-IDF, 0.5 thresholds | 0.6833 | 0.8317 | 0.8471 | 0.8290 | 0.8686 | 0.8928 |
| TF-IDF, tuned thresholds | 0.6663 | 0.7956 | 0.8778 | 0.8273 | 0.8647 | 0.8928 |
| Legal-BERT, tuned thresholds | 0.8026 | 0.9120 | 0.8848 | 0.8956 | 0.9231 | 0.9518 |

Legal-BERT improved contract macro F1 by **6.66 percentage points** over
default TF-IDF (paired 95% interval **+2.50 to +11.07 points**) and **6.83
points** over tuned TF-IDF (**+2.81 to +10.99 points**). It had higher
contract F1 in all 10 categories than either TF-IDF configuration. The
largest gains over default TF-IDF were in Change Of Control (+18.53 points),
Uncapped Liability (+11.11), and Termination For Convenience (+10.92).
Recall fell for Uncapped Liability (1.000 to 0.769, 13 positive contracts) and
Termination For Convenience (0.793 to 0.759), even as precision improved.

The TF-IDF chunk results reproduce Task 9. The Legal-BERT chunk result comes
from Task 13 and uses 3,029 chunks, compared with 3,096 for TF-IDF. The
contract comparison uses the same 102 contracts and CUAD labels for all
three configurations. It compares the full pipelines, including their
different tokenizers and chunk boundaries.

See the [per-category metrics](evaluation-results/contract_metrics.csv),
[paired confidence intervals](evaluation-results/contract_f1_intervals.csv),
and [precision–recall curves](evaluation-results/contract_pr_curves.svg).

## Transformer handoff

Use the original Task 13 outputs from the [results commit](https://github.com/Break-Through-Tech/Accenture-1T-contract-review-challenge/tree/dc2a934b9002662fe70dc5e5194fe455e0825948/docs/final-model-results). The predictions
contain `contract`, `chunk_id`, and a score column named for each priority
category. The run record contains the frozen per-category thresholds and
test-set counts. Use both files from the same run.

```sh
uv run --locked python -B -m scripts.evaluation_report \
  --transformer-predictions /path/to/final_model/test_predictions.parquet \
  --transformer-run /path/to/results/test_run.json
```

The script checks the official test contract set, exported row count, unique
chunk IDs, named score columns, and frozen thresholds. All three configurations
use the same CUAD contract labels and maximum-over-chunks aggregation. The
comparison measures each pipeline with its own tokenizer. For Legal-BERT
chunk-level metrics, use Task 13's `test_per_category.csv` and report its
chunking setup separately from the TF-IDF `chunk_metrics.csv`.
