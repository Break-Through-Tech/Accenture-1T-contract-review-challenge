# Task #6: Chunking implementation

**Owners:** Thierno and Nia Lam  
**Issue:** [Task #6: Chunking implementation](https://github.com/Break-Through-Tech/Accenture-1T-contract-review-challenge/issues/6)

## Goal

Implement the strategy from task #5, validate the output, and produce labeled chunks for the TF-IDF and keyword baselines.

Start from updated `main`, which includes the merged [chunking strategy](docs/chunking-strategy.md).

## Workload split

| Responsibility | Thierno | Nia |
|---|---|---|
| Main task | Create chunks from full contracts | Match chunks with annotated evidence and assign labels |
| Implementation | Paragraph-aware chunking, 510-token text limit, 128-token overlap, original character offsets | Category labels, partial-evidence handling, label validity masks, evidence records |
| Validation | Token limits, exact text slices, text coverage, final short chunks, repeatable output | Multiple categories, repeated spans, partial evidence, and chunks with no priority annotations |
| Files | `src/chunking.py`, chunking tests, pipeline/export script | `src/chunk_labels.py`, labeling tests, label-validation report |
| Deliverable | Unlabeled chunk table | Labeled chunk table and evidence table |

File names above are proposed so we can avoid editing the same files.

## Shared input format

One row represents one chunk.

| Column | Meaning |
|---|---|
| `contract` | Contract identifier from the existing loader |
| `split` | Official `train` or `test` split |
| `chunk_id` | Stable, unique chunk identifier |
| `start` | Starting character position in the original contract |
| `end` | Ending character position, excluded |
| `text` | Exactly `contract_text[start:end]` |
| `n_tokens` | Token count using the reference tokenizer from task #5 |

Thierno produces this table. Nia's labeling code accepts it together with the existing `spans` table.

## Labeling agreement

Use the category order already defined in `src.cuad_data.PRIORITY_CATEGORIES`.

- A fully contained annotation makes its category positive.
- No overlapping annotation means zero for that category relative to CUAD's labels.
- Partial evidence without a complete annotation is ambiguous. Flag it rather than treating it as negative.
- A complete annotation plus another partial annotation of the same category remains positive.
- A chunk can have multiple positive categories.
- Keep chunks with no priority annotations.

Nia adds:

- `labels`: ten binary category values.
- `label_mask`: ten flags indicating which labels are valid for training.
- A separate evidence table containing `chunk_id`, category, original annotation offsets, overlap offsets, and whether the annotation is fully contained.

## Async workflow

1. Agree on this format and create a small shared example containing complete evidence, partial evidence, and unrelated text.
2. Create separate branches from updated `main`:
   - Thierno: `chunking-implementation`
   - Nia: `chunk-labeling`
3. Work independently. Nia can use the shared example before the chunker is ready.
4. Open separate PRs with implementation and tests.
5. Thierno connects both components and runs the training export. Nia reviews the resulting labels and evidence report.

## Completion checks

- Original contract text and annotation offsets remain unchanged.
- The chunker works without receiving annotations.
- Every chunk meets the agreed token limit.
- All non-whitespace contract text is covered.
- Overlapping chunks from a contract stay in the same split.
- Partial evidence remains distinguishable from negative labels.
- Exported tables reload successfully and preserve labels and evidence.
- Training results reproduce task #5's reference figures, or any differences are explained:
  - 14,266 chunks.
  - 3,229 of 3,241 priority annotations fully contained.
- Process the official test set only after the rules are fixed. Do not tune the strategy using test results.
