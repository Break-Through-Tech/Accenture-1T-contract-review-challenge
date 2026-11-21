# Task #18: Inference + clause register

## What this is

The clause register lists, for each of the 102 test contracts, every priority clause the final
model found: the contract, the category, where the evidence is (character offsets), the evidence
text, and how confident the model was. It's the input for the risk scorer (#17) and the triage
pipeline (#19).

- Full register (497 entries): Google Drive, `contract-review/inference/clause_register.csv`
  (and `.parquet`)
- Sample (3 contracts): [inference-results/clause_register_sample.csv](inference-results/clause_register_sample.csv)
- Check against the gold clauses: [inference-results/register_check.csv](inference-results/register_check.csv)
- Run summary: [inference-results/summary.json](inference-results/summary.json)

## How it works

1. **Chunk** the test contracts with legal-bert's tokenizer (510 tokens, 128 overlap), the same
   way task #13 did.
2. **Score** every chunk with the final model from task #13 (legal-bert + pos_weight, 3 epochs).
3. **Threshold** each category with the per-category thresholds tuned in CV (`selection.json`
   from task #13, 0.68 to 0.95).
4. **Merge** windows. Chunks overlap, so a clause near the edge of one chunk also shows up in the
   next one. For each contract and category, any run of positive chunks that overlap gets merged
   into one entry, so the same clause isn't counted twice. The entry's confidence is the highest
   chunk probability in it.
5. **Attach evidence**: the contract text for the merged span, with leading and trailing
   whitespace trimmed (offsets are adjusted to match, so `text[start:end] == evidence_text`).

Merging turned 747 positive chunks into 497 entries, so 250 would have been double counts.
Most entries are 1 chunk (290) or 2 chunks (173).

### Register columns

| Column | Meaning |
|---|---|
| `contract` | CUAD contract title |
| `category` | One of the 10 priority categories |
| `start`, `end` | Character offsets in the contract text (`end` excluded) |
| `confidence` | Highest model probability among the merged chunks |
| `threshold` | The CV threshold for that category, so #17/#19 can compute the review boundary from the rubric (`t + 0.10 × (1 - t)`) |
| `n_chunks` | How many chunks were merged into this entry |
| `chunk_ids` | The merged chunk IDs, separated by `;` |
| `evidence_text` | The contract text from `start` to `end` |

## Where the scores came from

The task #13 weights are in the Kaggle notebook's Output, not in Drive, so I didn't rerun the
model for this. Task #13 already saved the final model's probabilities for every test chunk
(`docs/final-model-results/test_predictions.parquet`), and that's what the register is built from.
It's the same model on the same chunks: re-chunking the test split here gives exactly the same
3,029 chunk IDs in the same order.

To rerun the model itself, upload the task #13 `checkpoint/` folder to Drive and run
[notebooks/inference_colab.ipynb](../notebooks/inference_colab.ipynb) on a T4. Its `predict` step
also compares the new probabilities to task #13's and prints the biggest difference.

## How good is it

I checked the register against the gold CUAD clauses in the test split. An entry counts as right
if it overlaps a gold clause of the same category.

| Category | Entries | Gold clauses | Entry precision | Clause recall | Contract-level F1 |
|---|---:|---:|---:|---:|---:|
| Cap On Liability | 69 | 118 | 0.83 | 0.90 | 0.93 |
| Uncapped Liability | 12 | 16 | 0.83 | 0.69 | 0.83 |
| Governing Law | 94 | 90 | 0.91 | 0.98 | 1.00 |
| Anti-Assignment | 87 | 137 | 0.90 | 0.88 | 0.96 |
| Change Of Control | 31 | 62 | 0.90 | 0.76 | 0.89 |
| Termination For Convenience | 32 | 41 | 0.78 | 0.71 | 0.79 |
| Exclusivity | 49 | 78 | 0.71 | 0.82 | 0.86 |
| Non-Compete | 42 | 59 | 0.71 | 0.90 | 0.84 |
| Renewal Term | 30 | 31 | 0.83 | 0.97 | 0.91 |
| Audit Rights | 51 | 105 | 0.92 | 0.92 | 0.95 |
| **Average** | **497** | **737** | **0.83** | **0.85** | **0.90** |

- **Entry precision**: how many entries overlap a real clause of that category.
- **Clause recall**: how many gold clauses are covered by an entry.
- **Contract-level F1**: just "does this contract have this clause type or not". This is what
  the missing-clause check in the rubric depends on (for example, no Cap On Liability).

Gold clauses outnumber entries because one merged entry often covers several gold clauses that
sit close together. 14 contracts have no entries at all. 12 of them really have no priority
clauses (mostly short joint filing agreements). The other 2 were missed: INGEVITYCORP (4 gold
clauses) and ThriventVariableInsuranceAccountB (1).

## Notes for #17 and #19

- **Evidence is long.** The model classifies whole chunks, so it knows which chunk a clause is in
  but not where inside it. Evidence is the whole merged span, a median of about 2,700
  characters. The red-flag patterns in `src/risk.py` could match text from a neighboring clause
  in the same span. Using gold spans first (like #19 plans) avoids this; for model output,
  narrowing the evidence would be a good next step.
- **Confidence is high almost everywhere.** pos_weight pushes probabilities up, so the median
  confidence is 0.997 and only 11 entries are below 0.9. The numbers aren't calibrated, so the
  confidence check will send fewer clauses to review than it would with calibrated scores.
- **Nearby clauses can merge.** If two different clauses of the same category are in back-to-back
  chunks, they become one entry.
- The thresholds came from CV and weren't changed after looking at test results.

## How to run

```bash
# build the register from task #13's saved probabilities (runs on a laptop, about a minute)
uv run --locked python -B -m scripts.inference register --output-dir OUT

# rerun the model first (needs the task #13 checkpoint and a GPU; see the Colab notebook)
uv run --locked python -B -m scripts.inference predict --checkpoint CHECKPOINT --output OUT/test_chunk_predictions.parquet
uv run --locked python -B -m scripts.inference register --predictions OUT/test_chunk_predictions.parquet --output-dir OUT
```

`register` writes the full register to `OUT` and the sample, check and summary to
`docs/inference-results/`.

## Checks

```bash
uv run --locked python -B -m pytest -q -p no:cacheprovider tests/test_inference.py
uv run --locked ruff check --no-cache src/inference.py scripts/inference.py tests/test_inference.py
```

The tests cover merging overlapping windows, keeping categories and contracts separate,
whitespace trimming with correct offsets, the gold check, the unlabeled prediction path
matching the labeled one, and `predict` → `register` end to end with a tiny model.
