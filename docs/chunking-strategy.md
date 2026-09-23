# Chunking strategy — September, task #5

[Issue #5](https://github.com/Break-Through-Tech/Accenture-1T-contract-review-challenge/issues/5)
asks for a comparison of paragraph-based and token-limited chunking, optional overlap,
and a documented rationale. This document recommends the starting configuration for
Task #6. The accompanying script is a boundary experiment, not the production chunker
or a trained classifier.

## Recommendation

Use **paragraph-aware windows with a maximum of 510 reference tokens of contract text
and 128 tokens of overlap**. Prefer ending near a paragraph break, then a sentence
break, and fall back to the token limit when neither is available.

On all 408 official training contracts, this configuration put **3,229 of 3,241 priority
annotations (99.630%) completely inside at least one chunk within the size limit**.
It generated 14,266 chunks and processed 1.390 times the original token count.

This is evidence-containment coverage, **not classification accuracy or model recall**.
Full containment does not guarantee that every cross-reference or exception needed to
interpret a clause is also in the chunk. No test contracts were used in this experiment.

## What we start from

Use the shared `src.cuad_data` loader:

- `contracts`: original text, contract identifier, and character length.
- `spans`: annotated category, original evidence text, and `[start, end)` character offsets.
- `PRIORITY_CATEGORIES`: the existing ordered list of ten categories.

Do not change `contracts.text`, strip headers, collapse whitespace, or reconstruct
chunks by decoding token IDs. Every emitted chunk must be a slice of the original
string. The experiment builds training tables from the raw file rather than trusting
a previously generated cache, and the loader verifies every original annotation offset.

The ten targets are Cap On Liability, Uncapped Liability, Governing Law,
Anti-Assignment, Change Of Control, Termination For Convenience, Exclusivity,
Non-Compete, Renewal Term, and Audit Rights.

## Why paragraph boundaries alone are insufficient

Blank-line formatting varies across CUAD. A block may be a heading, a lettered list
item, a whole clause, or a large portion of a page. The raw JSON's `paragraphs` field
is not a usable paragraph segmentation: it holds the entire contract.

In the training contracts, 120 priority annotations cross blank-line boundaries.
For example, the WhiteSmoke promotion/distribution agreement has an annotated
`Uncapped Liability` passage at characters 33,707–34,283 that includes introductory
language and five lettered items separated by blank lines. Treating each block as
an independent chunk separates the items from their context.

With the reference tokenizer, 1,628 of 62,233 nonempty blank-line blocks exceed the
510-token budget; the largest contains 5,054 tokens. Paragraph-only segmentation
contains 3,121 priority spans if unlimited-size blocks count, but only 2,261 are
contained in blocks that actually fit the budget.

## Token counting

For this experiment only, use the tokenizer from `google-bert/bert-base-uncased`,
revision `86b5e0934494bd15c9632b12f734a8a67f723594`, with `tokenizers==0.22.1`.
Only `tokenizer.json` is downloaded; there are no model weights or training.

A token is a tokenizer-defined piece of text, not necessarily a word. The 510-token
text budget leaves room for the two special tokens used by a single-sequence BERT
input within its 512-position limit. See the [reference model configuration](https://huggingface.co/google-bert/bert-base-uncased/blob/86b5e0934494bd15c9632b12f734a8a67f723594/config.json).

This does not select October's model. If a different tokenizer or input structure
is adopted, recalculate the budget and rerun the comparison. TF-IDF can use the raw
chunk text with its own feature extraction; it need not use BERT tokens as features.

## Compared approaches and results

All methods use the same original text and 3,241 priority annotations. Boundaries are
chosen from text only, without access to annotations. Annotations are used afterward
to measure containment.

| Method | Overlap | Chunks | Intact priority spans within budget | Coverage | Token work |
|---|---:|---:|---:|---:|---:|
| Paragraph-only, unbounded | None | 62,233 | 2,261 | 69.762% | 1.000× |
| Fixed window | 0 | 9,124 | 2,839 | 87.596% | 1.000× |
| Fixed window | 64 | 10,340 | 3,100 | 95.649% | 1.140× |
| Fixed window | 128 | 11,965 | 3,198 | 98.673% | 1.325× |
| Fixed window | 256 | 17,698 | 3,226 | 99.537% | 1.973× |
| Paragraph-aware | 0 | 10,387 | 3,204 | 98.858% | 1.000× |
| Paragraph-aware | 64 | 12,041 | 3,215 | 99.198% | 1.164× |
| **Paragraph-aware** | **128** | **14,266** | **3,229** | **99.630%** | **1.390×** |
| Paragraph-aware | 256 | 22,652 | 3,236 | 99.846% | 2.252× |

`Token work` is the sum of freshly tokenized chunk lengths divided by the original
full-contract token count (4,547,807). It is a text-volume proxy, not a runtime or
memory benchmark; it excludes padding and special tokens. All fixed and
paragraph-aware candidates passed the size limit. Paragraph-only is a diagnostic
baseline; its oversized blocks would need splitting before model use.

The 128-overlap recommendation favors preserving evidence at moderate duplication.
Compared with the fixed-window 128 baseline, it contains 31 additional annotations.
Increasing paragraph-aware overlap from 128 to 256 recovers seven additional
annotations but adds 8,386 chunks and about 62% more processed tokens. The
paragraph-aware zero-overlap alternative is useful if compute cost becomes the
primary constraint: it contains 25 fewer spans than the recommended configuration
while processing approximately the original token volume.

The complete measured tables and provenance are in [chunking-results](chunking-results/).

## Exact proposed boundary rules

1. Process each contract independently. Carry its identifier and official split
   through every output. Never combine text from different contracts.
2. Encode the unmodified contract with truncation and padding disabled and request
   original character offsets. Do not count special tokens in the 510-text-token budget.
3. Starting at token index `i`, the hard candidate end is `min(i + 510, token_count)`.
   If the remaining document fits, take it and finish.
4. Otherwise choose the furthest blank-line boundary in the latter half of that
   window. The experiment recognizes blank lines with `\n[ \t]*\n+`. Pack adjacent
   blocks together; do not emit every heading or list item as a separate chunk.
5. If there is no eligible paragraph boundary, choose the furthest heuristic
   sentence boundary in the same range: `.`, `!`, or `?`, optional closing quotes
   or brackets, followed by whitespace. This is a heuristic, not a legal sentence
   parser; abbreviations and numbered provisions remain a limitation.
6. If neither boundary exists, end at the token limit. In the experiment's token
   coordinates, an eligible natural boundary must be at least
   `max(i + 255, i + overlap + 1, previous_end + 1)`. This avoids tiny nonfinal
   chunks and repeated windows that add no new text.
7. Map the boundary to the start character of the next token. Use character 0 for
   the first chunk and the full string length for the last. Slice the original text.
   This retains inter-token whitespace as part of the slices.
8. Retokenize that exact slice to verify its actual size. Slicing inside a word can
   change WordPiece tokenization; shrink the end if it exceeds 510. Do not silently
   truncate during later model ingestion.
9. Start the next window 128 full-document token positions before this window's end.
   **Overlap means repeated tokens, not the distance moved forward.** With a full
   510-token window, the normal advance is 382 tokens; natural boundaries can shorten
   it. Avoid splitting duplicate character offsets and require forward progress.
10. Keep the final short chunk. Preserve all non-whitespace source text, even when
    it has no priority annotations. No padding is stored in the text dataset.

Natural-boundary preference applies to chunk ends. Overlap can make the next chunk
start inside a sentence or paragraph; that is deliberate repeated context. Hard
fallbacks can also split words. These limitations are why the actual substring is
retokenized and containment is measured rather than assumed.

## Three concrete examples

These contracts were selected at approximately the 10th, 50th, and 90th percentiles
of training character length, sorted by length and identifier. They illustrate
formatting variation; all 408 contracts were used for the comparison above.

| Contract | Words | Reference tokens | Recommended chunks | Intact priority spans |
|---|---:|---:|---:|---:|
| Solutions Vending services agreement | 1,122 | 1,510 | 5 | 0 of 0 |
| AUL American Unit Trust servicing agreement | 5,607 | 7,018 | 21 | 6 of 6 |
| Zebra Technologies IP agreement | 19,581 | 26,839 | 84 | 10 of 10 |

The Solutions Vending example remains useful: it demonstrates text with none of the
ten target annotations, rather than a contract to discard. Full identifiers and
original first-chunk text are saved in the generated `samples.json` for inspection.

## Coverage by category for the recommendation

| Category | Fully contained / annotated spans | Coverage |
|---|---:|---:|
| Cap On Liability | 553 / 554 | 99.819% |
| Uncapped Liability | 151 / 151 | 100.000% |
| Governing Law | 374 / 374 | 100.000% |
| Anti-Assignment | 515 / 517 | 99.613% |
| Change Of Control | 191 / 191 | 100.000% |
| Termination For Convenience | 204 / 205 | 99.512% |
| Exclusivity | 331 / 332 | 99.699% |
| Non-Compete | 198 / 200 | 99.000% |
| Renewal Term | 177 / 179 | 98.883% |
| Audit Rights | 535 / 538 | 99.442% |

## Evidence and labeling handoff to Task #6

The implementation should expose one record per chunk with `contract`, `split`,
`chunk_id`, `start`, `end`, `text`, and `n_tokens`, plus annotation evidence metadata.
For every overlapping priority span, retain its category and original span offsets,
its overlap with the chunk, and whether it is fully contained. Keep category order
in `src.cuad_data.PRIORITY_CATEGORIES`; do not redefine another list.

Recommended label semantics:

- Fully contained evidence makes that category positive for the chunk.
- A category with no overlapping annotation is zero relative to the CUAD labels.
- A category that has partial evidence but no fully contained evidence is ambiguous:
  flag it separately, rather than silently making it negative or claiming complete evidence.
- If a category has one fully contained span plus another partial span, it remains
  positive, with both evidence records retained.

Task #6 should implement an explicit per-category validity mask alongside the labels.
For the TF-IDF baseline, a separate binary classifier per category can omit ambiguous
rows for that category during fitting. Report the exclusions. At evaluation, keep
all chunks for inference, retain contract-level evidence, and report boundary coverage
alongside classifier metrics so excluded fragments do not hide end-to-end misses.

The recommended configuration produces 410 chunks with at least one partial-only
category. That is different from the 12 globally uncovered spans: a span may be
partial in one chunk and complete in an overlapping neighbor. Separately, 11,305 of
14,266 chunks (79.24%) have no overlap with any priority annotation. Retain these
negative examples; any later downsampling must be training-only and documented.

## Limitations and validation requirements

- Twelve priority annotations are not fully contained under the recommended settings.
  One annotation contains 558 reference tokens, exceeding the budget by itself; overlap
  alone cannot put that whole passage into a 510-token chunk. The other eleven are
  boundary misses. Keep all fragments and log these cases; do not truncate or drop
  the source evidence to manufacture 100% coverage.
- A larger overlap, context retrieval, or a longer-context model can be investigated
  later. No annotation-driven special windows should be inserted into this experiment:
  they would not be available for unseen contracts.
- An annotation can cover a list or reference another section. Complete containment
  is only a proxy for adequate legal context, not a judgment about risk.
- Text with no annotation is not proof that a clause is absent or that the contract
  is low risk. Redactions remain in the original text.
- Chunk boundaries and overlap duplicate evidence. Evaluate each original span once
  when reporting containment and deduplicate evidence during contract aggregation.
- The original train/test split stays unchanged. Any validation partition must be
  chosen by contract inside the 408 training contracts before fitting or tuning models.
  Keep all overlapping chunks from one contract in the same partition.
- The experiment checks nonempty chunks, valid and advancing character boundaries,
  complete coverage of non-whitespace source text, and actual token budgets across all
  nine configurations. Task #6 must additionally verify its emitted text equals
  `contract_text[start:end]`, validate label/evidence joins, and test inference without
  supplying annotations.

## Reproduce

From the repository root:

```bash
uv sync --locked
uv run --locked --with tokenizers==0.22.1 python -B scripts/compare_chunking.py
```

The extra tokenizer package is scoped to the experiment command; project dependency
files are unchanged. The first run needs network access to obtain the pinned package
and tokenizer. The tokenizer's normal cache can satisfy subsequent downloads.

The script rebuilds training tables in memory, compares nine configurations, and
writes `summary.csv`, `per_category.csv`, `per_contract.csv`, `metadata.json`,
`uncovered_spans.json`, and `samples.json` to `data/interim/chunking_strategy/`.
The summary, category table, and metadata in `docs/chunking-results/` are a committed
snapshot of this experiment; rerunning does not overwrite that review snapshot.
No raw data or pre-existing implementation is changed, and no model is trained.
