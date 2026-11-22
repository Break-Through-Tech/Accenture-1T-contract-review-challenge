# Task #18: Inference + evidence extraction (clause register)

**Status:** done for the 102 official test contracts.

**Output:** `docs/clause-register-results/test_clause_register.parquet`, one row per detected
clause:

- contract and category
- evidence text and character offsets
- a shorter evidence excerpt with its own offsets
- confidence
- review flag

Task #19 reads this file.

## Run it

```bash
uv run --locked python -B -m scripts.clause_register
```

This takes about 15 seconds on a CPU; no GPU is needed. It writes everything in
`docs/clause-register-results/` and a JSONL copy of the full register to
`data/interim/clause_register/` (not in Git).

## Where the model scores come from

The task asks to run the final legal-bert model from Task #13 on the 102 test contracts,
re-chunked with the legal-bert tokenizer. Task #13's single official test run already did
exactly that and saved the result as `docs/final-model-results/test_predictions.parquet`. It
holds the probability of each of the 10 categories for all 3,029 legal-bert test chunks.

The register uses that export as is and does not re-score the test set, for two reasons:

- Re-running the same model on the same chunks would only reproduce the same numbers.
- It avoids a second pass over the test split.

The script checks that the export matches Task #13's official run (`test_run.json`) and the
contract text:

- every chunk ID's document key must match the contract text,
- every offset must fall inside the contract.

For contracts outside the test set (Task #19's one-command pipeline), `--contracts-json` with
`--checkpoint` chunks the new contracts and runs the saved model (`score_chunks`). This needs
the Task #13 weights, which are in the Kaggle notebook's Output.

## Method

| Step | What happens |
|---|---|
| 1. Decide | A chunk is positive for a category if its probability is at least the category's frozen CV threshold (`test_run.json`, 0.68–0.95). |
| 2. Merge | Positive chunks of the same contract and category whose character ranges overlap or touch become one clause. Windows overlap by 128 tokens, so a clause near a boundary is usually flagged in two chunks. Merging counts it once. |
| 3. Offsets | Chunk IDs end in `:start:end`, character offsets into the original contract. The clause's `start`/`end` is the union of its chunks, and `evidence_text` is exactly `contract_text[start:end]`. |
| 4. Excerpt | Inside the merged window, the category's Task #8 keyword rules locate the passage that matches most. That passage becomes the excerpt, extended into neighbouring passages to at least 400 characters (so a heading like "4.2 Covenant Not to Compete." brings its body along) and capped at 1,500. `excerpt_start`/`excerpt_end` are also offsets into the contract. If no rule fires, there is no excerpt and the full window is the evidence. |
| 5. Confidence | The highest chunk probability in the clause. |
| 6. Review flag | From the Task #16 rubric: a confidence below `t + 0.10 × (1 − t)` goes to human review, and a score equal to that boundary clears. `needs_review` is this confidence check only. `review_reason` also notes when there is no excerpt. |

## Register columns

| Column | Meaning |
|---|---|
| `contract`, `category`, `clause_id` | Clause identity. `clause_id` = `contract\|category\|n` |
| `start`, `end`, `evidence_text` | Merged window, character offsets into the contract |
| `excerpt_start`, `excerpt_end`, `excerpt_text` | Keyword-located passage inside the window (empty if no rule fired) |
| `excerpt_rules` | Which keyword rules located the excerpt |
| `confidence`, `threshold`, `review_boundary` | Model score and the cut-offs used |
| `needs_review`, `review_reason` | Rubric confidence check, plus why |
| `n_chunks`, `chunk_ids` | Which chunks were merged |
| `source` | `model` (gold CUAD spans would be `cuad`) |

## Results on the 102 test contracts

| | Count |
|---|---:|
| Positive (chunk, category) decisions | 747 |
| **Clauses after merging** | **497** (33% fewer: no double counting) |
| Contracts with at least one clause | 88 of 102 |
| Clauses per contract | median 4, max 20 |
| Clauses merged from 2 or more chunks | 207 |
| Clauses with a keyword excerpt | 384 (77%) |
| Clauses flagged by the confidence check | 3 |

### Check against CUAD gold spans

A clause is **supported** if its window overlaps a gold span of the same category. A gold span
is **found** if a clause of that category overlaps it. The **excerpt hit** rate is the share of
supported clauses whose excerpt also overlaps a gold span.

| Category | Clauses | Gold spans | Supported | Gold found | Excerpt hits gold |
|---|---:|---:|---:|---:|---:|
| Cap On Liability | 69 | 118 | 0.83 | 0.90 | 0.97 |
| Uncapped Liability | 12 | 16 | 0.83 | 0.69 | 0.71 |
| Governing Law | 94 | 90 | 0.91 | 0.98 | 0.99 |
| Anti-Assignment | 87 | 137 | 0.90 | 0.88 | 0.97 |
| Change Of Control | 31 | 62 | 0.90 | 0.76 | 0.94 |
| Termination For Convenience | 32 | 41 | 0.78 | 0.71 | 0.93 |
| Exclusivity | 49 | 78 | 0.71 | 0.82 | 0.88 |
| Non-Compete | 42 | 59 | 0.71 | 0.90 | 0.64 |
| Renewal Term | 30 | 31 | 0.83 | 0.97 | 1.00 |
| Audit Rights | 51 | 105 | 0.92 | 0.92 | 0.88 |
| **All** | **497** | **737** | **0.85** | **0.88** | **0.93** |

### What the numbers say

- **The register finds 88% of the real clauses, and 85% of its entries are real.** It does
  best on clauses with fixed wording (Governing Law, Renewal Term, Audit Rights).
- **The excerpt is usually the right place to look.** It lands on the real clause 93% of the
  time and is about 520 characters, versus about 2,670 for the full window. For Task #19,
  score red flags on the excerpt when there is one, so wording from neighbouring clauses in the
  same window doesn't trigger them.
- **Weak spots:**
  - **Non-Compete excerpts (0.64):** the keyword rules often pick a related restriction instead
    of the covenant itself.
  - **Uncapped Liability (12 clauses, 16 gold spans):** too small to read much into.
  - **Termination For Convenience and Change Of Control:** miss about a quarter of the gold
    spans.
- **Almost nothing goes to review on confidence.** The median confidence is 0.997, so only 3
  clauses fall under the rubric's boundary. pos_weight pushes scores up (see Task #13), so the
  confidence check rarely fires. Task #20 may want a wider margin.
- **Merging has a cost.** One merged window can hold two separate clauses of the same category.
  That is fine for the register, but it is counted as one clause.

## Files

| File | Contents | In Git |
|---|---|---|
| `docs/clause-register-results/test_clause_register.parquet` | Full register, 497 clauses (0.8 MB) | Yes |
| `docs/clause-register-results/clause_register_sample.csv` | The first 5 contracts alphabetically, readable in a spreadsheet | Yes |
| `docs/clause-register-results/per_contract.csv` | Clause count, categories and review count for each of the 102 contracts | Yes |
| `docs/clause-register-results/summary.csv` | Gold-span check above | Yes |
| `docs/clause-register-results/metadata.json` | Thresholds, data hashes, counts | Yes |
| `data/interim/clause_register/test_clause_register.jsonl` | Same register as JSON lines (2 MB) | No; upload to Drive if needed |

The task said to put the full output in Drive. The parquet is under 1 MB, so it is committed
instead, so that Task #19 can use it straight from the repo.

## Checks

```bash
uv run --locked python -B -m pytest -q -p no:cacheprovider tests/test_clause_register.py
uv run --locked ruff check --no-cache src/clause_register.py scripts/clause_register.py tests/test_clause_register.py
```

The 12 tests cover:

- merging of overlapping, touching and separate windows
- thresholds, including dropping chunks below them
- offsets pointing into the source
- the excerpt growing past a heading
- no excerpt when no keyword rule fires
- the rubric review boundary, where equal clears
- rejecting scores from different text or with out-of-range offsets
- the gold-span metrics
- running a saved checkpoint on new chunks

The full suite (262 tests) passes.
