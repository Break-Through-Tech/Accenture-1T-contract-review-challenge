# Model selection: encoder and tokenizer decision

**Decision:** `nlpaueb/legal-bert-base-uncased`, conditional on re-chunking with its tokenizer (see Consequences).
**Status:** based on one seed (42). Two extra seeds for bert and legal-bert are optional and would firm this up.
**Run:** Colab T4 GPU, run `selection-20261028-214409`, 2026-10-28. Official test split not used.
**Revisions:** bert `86b5e093`, legal-bert `15b570cb`, deberta-v3 `8ccc9b6f` (full SHAs in `tokenizer_meta.json`).

## Why this was needed
Chunks were built to fit 512 tokens under the `bert-base-uncased` tokenizer. Another encoder splits the
same text differently, so chunks can overflow 512 tokens. `ChunkDataset` raises on overflow instead of
truncating annotated evidence.

## Tokenizer recount (14,266 training chunks, 3,254 positive labels)
| Tokenizer | Mean | p99 | Max | Over 512 | Positives in over-512 chunks | Total tokens vs BERT |
|---|---:|---:|---:|---:|---:|---:|
| bert-base-uncased | 445.1 | 512 | 512 | 0 | 0 | 1.000 |
| legal-bert-base-uncased | 437.4 | 530 | 635 | 801 (5.6%) | 107 (3.3%) | 0.983 |
| deberta-v3-base | 429.7 | 516 | 656 | 214 (1.5%) | 26 (0.8%) | 0.965 |

legal-bert's legal vocabulary does not shorten text on average, but its tail is longer: 5.6% of chunks
overflow and carry 107 labeled positives that would be lost if those chunks were dropped.

## Short fine-tuning test
60 fold-0 training contracts (1,904 chunks), 2 epochs, batch 4, lr 2e-5, seed 42, scored on the full
fold-0 validation split (82 contracts, 2,657 chunks). Chunks over 512 under any tokenizer were excluded
for every candidate, so all models saw identical data. Same CLS + linear head for all.

| Candidate | Macro AP | Macro F1 @0.5 | Macro F1 best thr.* | Val BCE | s/step | Peak GPU GiB |
|---|---:|---:|---:|---:|---:|---:|
| bert-base-uncased | 0.266 | 0.127 | 0.261 | 0.0884 | 0.447 | 2.5 |
| legal-bert-base-uncased | **0.448** | **0.343** | **0.495** | **0.0695** | 0.456 | 2.5 |
| deberta-v3-base | 0.263 | 0.078 | 0.239 | 0.0880 | 0.649 | 6.0 |

*Tuned on the validation fold itself, so optimistic; use only to compare candidates.

## Reasoning
legal-bert's macro average precision is about 68% higher than BERT's and it beats BERT in 8 of 10
categories (largest gains: Governing Law 0.97 vs 0.67, Anti-Assignment 0.91 vs 0.42, Termination For
Convenience 0.31 vs 0.05). It costs the same time and memory as BERT. DeBERTa-v3 matched BERT on
quality while being about 45% slower and using 2.4x the GPU memory, so it has no case. The gap between
legal-bert and the others is far larger than we expect from seed noise, but only one seed was run.

## Consequences of the decision
- **Re-chunk with the legal-bert tokenizer** so no chunk exceeds 512 tokens and the 107 affected positives are kept.
  This changes chunks and labels, so re-run `export_chunks` and `label_chunks`, and expect fold assignments to need checking.
- legal-bert is a BERT architecture, so the existing `ContractClassifier` and `load_pretrained_model` work
  after changing `TOKENIZER_MODEL` / `TOKENIZER_REVISION` to legal-bert and its pinned SHA above.
- Scores here are not comparable to earlier full-data baselines: small training budget, and about 7% of
  validation chunks were excluded for length.

## Limits
Small training budget, one validation fold, one seed. Change Of Control, Exclusivity, Non-Compete and Renewal Term
are weak for every model (AP 0.02 to 0.17), likely undertrained rather than a model difference. Treat gaps
there as ties.
