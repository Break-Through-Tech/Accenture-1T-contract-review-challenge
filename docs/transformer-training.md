# Transformer training and smoke tests

Run from the project root with task #6 training labels and the committed
`docs/tuned-baseline-results/folds.csv`. The entry point is `src/train.py`.
The official 102-contract test split is not used by this command.

## CPU pipeline smoke test

```powershell
uv run --locked python -B -m src.train --tiny --max-contracts 20 --batch-size 8 --learning-rate 0.001 --device cpu --output-dir data/interim/models/transformer-smoke-tiny-20
```

This runs one full epoch with a small randomly initialized BERT, not pretrained
BERT. It checks training mechanics and cannot establish useful model accuracy.
With seed 42 and saved validation fold 0, selection gives 16 training contracts
(619 chunks) and four validation contracts (38 chunks), without changing fold
assignments. Selection depends on identifiers, not annotation content. An actual
run completed all 78 optimizer steps, processed all 619 training chunks, evaluated
all 38 validation chunks, and verified checkpoint reloading.

## Pretrained CPU check

```powershell
uv run --locked python -B -m src.train --max-contracts 20 --max-steps 2 --batch-size 2 --device cpu --output-dir data/interim/models/transformer-smoke-pretrained-20
```

This loads the pinned BERT base weights, initializes the classification head,
and caps training at two optimizer updates. It selects the same 20 contracts but
processes only four training chunks, then evaluates all 38 validation chunks.
It checks the pretrained path; it is not a full training epoch or an accuracy
benchmark. The first run downloads pretrained weights. `--local-files-only`
requires tokenizer and model artifacts already present in the Hugging Face cache.

Both smoke runs completed successfully on CPU and verified checkpoint reloads:

| Run | Optimizer steps | Processed training chunks | Validation chunks | Validation BCE |
|---|---:|---:|---:|---:|
| Tiny random BERT, full epoch | 78 | 619 | 38 | 0.181413 |
| Pretrained BERT, capped check | 2 | 4 | 38 | 0.642935 |

These losses come from different training budgets and architectures. A low loss
on this mostly negative dataset does not establish clause-detection quality.
Inspect per-category recall/F1 and use full-fold training for model comparisons.

## GPU training

```powershell
uv run --locked python -B -m src.train --device cuda --epochs 3 --batch-size 8 --output-dir data/interim/models/transformer-fold0
```

Without `--max-contracts`, all training contracts are partitioned using the saved
folds (326 training / 82 validation for fold 0). Without `--max-steps`, every
training chunk is processed each epoch. The default learning rate is 2e-5 with
AdamW, weight decay 0.01, and gradient norm clipping at 1. This initial trainer
uses full precision. It has no scheduler, class weights, early stopping, threshold
tuning, or checkpoint resume. The final epoch is saved; validation is reported
after every epoch. This is one held-out fold, not five-fold cross-validation.

Train and validation loss are weighted by valid label entries, not batch size.
Entirely masked training batches skip the optimizer, including weight decay.
Nonfinite loss or gradients fail the run. Validation metrics reuse
`src.evaluation.score` at a fixed 0.5 threshold, excluding masked entries.
Categories with no valid validation entries report undefined metrics, excluded
from macro averages; categories with valid entries but no positives still report
zero precision/recall/F1 under the existing scorer convention.

## Artifacts

Each run requires a new or empty output directory and writes:

- `checkpoint/`: model weights, configuration with category order, tokenizer JSON.
- `validation_metrics.csv`: per-category, macro, and micro validation metrics.
- `validation_predictions.parquet`: contract, chunk ID, and ten probabilities,
  in validation row order.
- `selected_contracts.csv`: saved fold membership and train/validation role.
- `metadata.json`: arguments, seed, model revision, package versions, input
  hashes, partition counts, processed chunk count, steps, epoch losses, and
  confirmation that checkpoint reloading preserved logits on a validation batch.

The checkpoint supports inference/further loading through
`ContractClassifier.from_pretrained(path)`. It does not store optimizer state.
Do not compare these small smoke-run metrics with the full-data baselines.

## Colab launcher

Open `notebooks/train_colab.ipynb` in Colab, select a GPU runtime, and run its
cells in order. Configure the repository branch containing these changes, or
upload a complete checkout to its `REPO_DIR`. Cloning does not include local,
uncommitted changes. Existing checkouts are used without pulling or overwriting.

The notebook creates an isolated Python 3.12 environment from `uv.lock`, verifies
GPU execution in that environment, runs tests, regenerates and validates training
chunks/labels, and delegates training to `python -m src.train`. Default settings
run one full pretrained epoch on 20 contracts, with no optimizer-step cap.
Set `MAX_CONTRACTS = None` for full-fold training. Each run uses a timestamped
output directory, and the last cell downloads an archive including the checkpoint.

The notebook schema and Python cells were validated locally. A dry run executes
every cell with external actions mocked and checks its generated arguments against
the real training CLI. Hosted Colab GPU execution has not been performed; the
notebook's GPU gate must pass before training. The locked CUDA runtime requires a
compatible GPU driver; GPU allocation alone does not establish compatibility.

## Verification commands

```powershell
uv run --locked python -B -m pytest -q -p no:cacheprovider
uv run --locked ruff check --no-cache src/train.py src/evaluation.py tests/test_train.py tests/test_evaluation.py
```
