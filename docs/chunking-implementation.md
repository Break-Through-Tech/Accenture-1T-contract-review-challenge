# Task #6 chunking handoff

Thierno's implementation produces the unlabeled table agreed in the [shared plan](https://github.com/Break-Through-Tech/Accenture-1T-contract-review-challenge/blob/task-6-plan/plan.md). Nia owns annotation matching, labels, validity masks, and the evidence table. Task #6 remains open until those components are integrated and validated.

## Run the training export

From the repository root, with CUAD's `train_separate_questions.json` in `data/`:

```bash
uv sync --locked
uv run --locked python -B -m scripts.export_chunks --split train
```

The first run downloads the pinned tokenizer file used in task #5. It does not download model weights or train a model. Add `--local-files-only` to use an existing tokenizer cache without network access.

Outputs are ignored by Git:

- `data/interim/chunks/train/chunks.parquet`
- `data/interim/chunks/train/metadata.json`

Each run replaces these files. Metadata records the configuration, tokenizer revision and package version, input and output hashes, validation counts, and annotation containment. The exporter rebuilds the existing loader's tables from the source JSON, avoiding stale cached tables. Chunk creation reads only contract identifiers and original text. Annotation containment is measured afterward.

The same command supports `--split test` with `data/test.json`. Keep the frozen defaults and do not tune on test results. The implementation checks below use training data only.

## Nia's input

```python
import pandas as pd
from src.cuad_data import build_split

chunks = pd.read_parquet("data/interim/chunks/train/chunks.parquet")
spans = build_split("train").spans
```

Join on `contract`, then compare original character offsets. These are Python string positions, not byte positions. `end` is excluded.

| Column | Type | Meaning |
|---|---|---|
| `contract` | string | Existing CUAD contract identifier |
| `split` | string | Official `train` or `test` partition |
| `chunk_id` | string | SHA-256 of split, contract identifier and original text, followed by start and end |
| `start` | int64 | Inclusive source character position |
| `end` | int64 | Exclusive source character position |
| `text` | string | Exact original `text[start:end]` |
| `n_tokens` | int64 | Reference token count without special tokens |

Rows are sorted by contract and start. IDs remain stable when input rows are reordered. Changes to source text, split, contract identifier, or boundaries change the ID. Keep the ID when adding labels so the evidence table can refer back to it.

The small [shared example](../tests/fixtures/chunking_example.json) contains `contracts`, `chunks`, and `spans` arrays. It includes complete evidence, partial evidence, and chunks with no evidence. Its smaller 12-token budget and 3-token overlap make it readable; production defaults remain 510 and 128. Load the arrays directly into DataFrames to develop labeling independently of the dataset or tokenizer download.

Use `PRIORITY_CATEGORIES` from `src.cuad_data` for the ten-element label and mask order. A fully contained annotation is positive. Partial-only evidence is masked as ambiguous. Keep chunks with no priority evidence. Nia's module will add labels and masks and produce a separate evidence table; this export does not yet do that.

## Use the chunker directly

```python
from src.chunking import chunk_contracts, load_tokenizer, validate_chunks
from src.cuad_data import build_split

train = build_split("train")
tokenizer = load_tokenizer()
chunks = chunk_contracts(train.contracts, "train", tokenizer=tokenizer)
checks = validate_chunks(train.contracts, chunks, "train", tokenizer=tokenizer)
```

The default strategy prefers paragraph boundaries, then sentence boundaries, then a hard token cutoff. Overlap uses full-document token positions. Slicing inside a word can change tokenization, so each resulting slice is counted again and shortened if necessary. See [the strategy](chunking-strategy.md) for the selection criteria.

Short final chunks and chunks with no annotations are retained. Empty or whitespace-only contracts produce no chunks. Invalid input or a configuration that cannot advance within its token budget raises an error instead of silently dropping text. Custom tokenizers must have padding and truncation disabled.

## Verification

```bash
uv run --locked python -B -m pytest -q -p no:cacheprovider
uv run --locked ruff check --no-cache src/chunking.py scripts/export_chunks.py tests/test_chunking.py
```

Tests cover natural boundary preference, hard cutoffs, overlap, short tails, Unicode, WordPiece slice recounting, deterministic IDs, source preservation, malformed output, and a Parquet round trip. Unit tests use an in-memory tokenizer and synthetic data, so they do not need network access or CUAD files.

The real training export reproduced task #5:

| Check | Result |
|---|---:|
| Contracts | 408 |
| Chunks | 14,266 |
| Maximum text tokens per chunk | 510 |
| Total processed tokens | 6,321,734 |
| Fully contained priority annotations | 3,229 / 3,241 |
| Priority annotations without full containment | 12 |

All source slices, non-whitespace coverage, advancing boundaries, unique IDs, split membership, and token counts passed validation. The saved Parquet table reloaded without changes. The remaining 12 annotations are the known boundary and length limitations from task #5, not missing source text. Label validation remains Nia's part of the handoff.
