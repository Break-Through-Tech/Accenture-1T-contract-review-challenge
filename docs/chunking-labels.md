# Task #6 chunk labeling handoff

Nia's implementation turns the unlabeled chunk table from [the chunking handoff](chunking-implementation.md) into labeled chunks and a separate evidence table, following the labeling agreement in the [shared plan](https://github.com/Break-Through-Tech/Accenture-1T-contract-review-challenge/blob/task-6-plan/plan.md). Thierno owns chunk creation and the training export. Task #6 remains open until both components are integrated and validated.

## Run the labeling step

Run the chunking export first (see [chunking-implementation.md](chunking-implementation.md)), then from the repository root:

```bash
uv run --locked python -B -m scripts.label_chunks --split train
```

It reads `data/interim/chunks/train/chunks.parquet` and rebuilds the annotation spans from the source JSON with the existing loader, so no stale cached tables are used. Outputs, which Git ignores like the chunk files, are written to `data/interim/labels/train/`:

- `labeled_chunks.parquet`
- `evidence.parquet`
- `label_report.json`
- `label_report.md`

Each run replaces these files. The script validates the Parquet files it saved, not the in-memory tables. For the training split it also lists any difference from the task #5 reference figures. Keep the frozen chunking defaults and do not run `--split test` until the rules are final.

## What the labeling does

Chunks and annotations are both ranges of original character positions in a contract (`end` excluded). For each chunk, the code looks at the priority-category annotations from the same contract and checks whether the ranges overlap. An annotation is **fully contained** when `chunk.start <= span.start` and `span.end <= chunk.end`. An annotation that only touches a chunk boundary does not overlap it.

Only the ten categories in `PRIORITY_CATEGORIES` (`src.cuad_data`) are used, in that order. Annotations in other categories are ignored. Labels are decided from character positions only, and the contract text and offsets are never modified.

| Evidence in the chunk, per category | `labels` | `label_mask` |
|---|---:|---:|
| At least one fully contained annotation | 1 | True |
| No overlapping annotation | 0 | True |
| Only partially overlapping annotations (ambiguous) | 0 | False |

A full annotation plus a partial one of the same category stays positive, and both are kept in the evidence table. A chunk can have several positive categories, and a partial category does not mask any other category. Chunks with no priority evidence are kept as all-zero rows with every flag True.

## Output tables

`labeled_chunks.parquet` has the seven chunk columns from the chunking handoff, unchanged and in the same order, plus two list columns:

| Column | Type | Meaning |
|---|---|---|
| `labels` | list of 10 ints | 0 or 1 per category, in `PRIORITY_CATEGORIES` order |
| `label_mask` | list of 10 bools | True when the label is valid for training, False when ambiguous |

`evidence.parquet` has one row per overlapping chunk and annotation pair and refers back to the chunk through the same `chunk_id`:

| Column | Type | Meaning |
|---|---|---|
| `chunk_id` | string | Chunk identifier from the chunk table |
| `contract` | string | Contract identifier |
| `category` | string | One of the ten priority categories |
| `span_start` | int64 | Original annotation start, inclusive |
| `span_end` | int64 | Original annotation end, exclusive |
| `overlap_start` | int64 | Start of the overlap between chunk and annotation |
| `overlap_end` | int64 | End of the overlap, exclusive |
| `fully_contained` | bool | Whether the whole annotation lies inside the chunk |

Exact duplicate annotations (same contract, category, start and end) become one evidence row. The current loader already removes them. Evidence rows are sorted by contract, chunk position, category order and span position.

## Use the labels directly

```python
import pandas as pd
from src.labeling import label_arrays, label_chunk_table
from src.cuad_data import build_split

chunks = pd.read_parquet("data/interim/chunks/train/chunks.parquet")
spans = build_split("train").priority()
labeled, evidence = label_chunk_table(chunks, spans)
labels, mask = label_arrays(labeled)  # (n_chunks, 10) int8 and bool arrays
```

`label_arrays` also works on a table reloaded from Parquet, where list columns come back as arrays. For a TF-IDF baseline with one binary classifier per category, fit category `i` only on the rows where `mask[:, i]` is True and report how many rows were excluded. At evaluation, keep all chunks, and report boundary coverage next to classifier metrics so excluded fragments do not hide misses. Treat `label_mask = False` as "unknown", never as "negative". Any validation split must be made by contract, never by chunk.

## Verification

```bash
uv run --locked python -B -m pytest -q -p no:cacheprovider tests/test_labeling.py
uv run --locked ruff check --no-cache src/labeling.py scripts/label_chunks.py tests/test_labeling.py
```

`validate_labels` recomputes every label, mask and evidence row with plain loops, independently of the pandas code, and fails on any mismatch. It also checks that the original chunk columns are unchanged, that labels are binary, that a positive label is never masked, and that overlap offsets lie inside their annotation. It returns the counts used in the report.

The 19 tests cover the shared example, boundary-touching spans, multiple categories in one chunk, full plus partial evidence for one category, repeated spans, an annotation split across two chunks, chunks without annotations, non-priority categories, a Parquet round trip, empty tables, input checks, and corrupted labels, masks, evidence and chunk columns being rejected. They use synthetic data and the shared fixture, so they need no network access or CUAD files.

## Training results

Fill this table from `data/interim/labels/train/label_report.md` after the real run.

| Check | Task #5 reference | This run |
|---|---:|---:|
| Chunks | 14,266 | TBD |
| Priority annotations | 3,241 | TBD |
| Fully contained annotations | 3,229 | TBD |
| Annotations not fully contained | 12 | TBD |
| Chunks with no priority overlap | 11,305 (79.24%) | TBD |
| Chunks with a partial-only category | 410 | TBD |

Explain any difference from the reference figures here.

## Limitations

- The 12 training annotations that are not fully contained in any chunk never receive a positive label. They are counted and listed in the report, not forced into a chunk.
- Labels depend on the chunk boundaries. If the chunking configuration changes, relabel everything.
- A chunk label means "a complete annotation of this category lies inside this chunk". It is not a contract-level answer, and an unlabeled chunk does not prove the clause is absent from the contract.
- Overlapping chunks repeat evidence. Count each original annotation once when reporting results, and deduplicate evidence when combining chunks into contract-level predictions.