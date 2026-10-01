"""Attach CUAD priority labels, validity masks and evidence to unlabeled chunks.

Input is the seven-column chunk table from src.chunking plus the CUAD spans table.
Offsets are original Python string positions; ``end`` is excluded.

Label semantics (per chunk, per category in PRIORITY_CATEGORIES order):

=================================  ======  ==========
evidence in the chunk              label   label_mask
=================================  ======  ==========
>= 1 fully contained annotation    1       True
no overlapping annotation          0       True
only partially overlapping         0       False  (ambiguous; excluded from fitting)
=================================  ======  ==========

A fully contained annotation plus a partial one of the same category stays positive.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from src.cuad_data import PRIORITY_CATEGORIES

CHUNK_COLUMNS = ["contract", "split", "chunk_id", "start", "end", "text", "n_tokens"]
SPAN_COLUMNS = ["contract", "category", "start", "end"]
EVIDENCE_COLUMNS = [
    "chunk_id",
    "contract",
    "category",
    "span_start",
    "span_end",
    "overlap_start",
    "overlap_end",
    "fully_contained",
]


def _check_chunks(chunks: pd.DataFrame) -> None:
    missing = set(CHUNK_COLUMNS) - set(chunks.columns)
    if missing:
        raise ValueError(f"chunks is missing columns: {sorted(missing)}")
    if chunks.chunk_id.duplicated().any():
        raise ValueError("chunk_id values must be unique")
    if (chunks.start >= chunks.end).any():
        raise ValueError("chunks must satisfy start < end")


def _priority_spans(spans: pd.DataFrame, categories) -> pd.DataFrame:
    """Keep priority-category rows; spans of other categories are ignored."""
    missing = set(SPAN_COLUMNS) - set(spans.columns)
    if missing:
        raise ValueError(f"spans is missing columns: {sorted(missing)}")
    priority = spans.loc[spans.category.isin(categories), SPAN_COLUMNS]
    if (priority.start >= priority.end).any():
        raise ValueError("spans must satisfy start < end")
    return priority


def build_evidence(
    chunks: pd.DataFrame, spans: pd.DataFrame, categories=PRIORITY_CATEGORIES
) -> pd.DataFrame:
    """One row per (chunk, priority annotation) pair that overlaps.

    Exact duplicate annotations (same contract, category, start, end) are collapsed
    so repeated spans cannot be double counted. Annotations are matched only within
    the same contract.
    """
    _check_chunks(chunks)
    categories = list(categories)
    spans = _priority_spans(spans, categories).drop_duplicates()
    left = chunks[["chunk_id", "contract", "start", "end"]].rename(
        columns={"start": "chunk_start", "end": "chunk_end"}
    )
    right = spans.rename(columns={"start": "span_start", "end": "span_end"})
    merged = left.merge(right, on="contract", how="inner")
    hit = merged[
        (merged.chunk_start < merged.span_end) & (merged.span_start < merged.chunk_end)
    ].copy()
    hit["overlap_start"] = np.maximum(hit.chunk_start, hit.span_start)
    hit["overlap_end"] = np.minimum(hit.chunk_end, hit.span_end)
    hit["fully_contained"] = (hit.chunk_start <= hit.span_start) & (
        hit.span_end <= hit.chunk_end
    )
    hit["category"] = pd.Categorical(hit.category, categories=categories, ordered=True)
    hit = hit.sort_values(
        ["contract", "chunk_start", "chunk_id", "category", "span_start", "span_end"]
    )
    hit["category"] = hit.category.astype(str)
    out = hit[EVIDENCE_COLUMNS].reset_index(drop=True)
    return out.astype(
        {
            "span_start": "int64",
            "span_end": "int64",
            "overlap_start": "int64",
            "overlap_end": "int64",
            "fully_contained": "bool",
        }
    )


def label_chunks(
    chunks: pd.DataFrame, evidence: pd.DataFrame, categories=PRIORITY_CATEGORIES
) -> pd.DataFrame:
    """Return a copy of chunks with ``labels`` (0/1 ints) and ``label_mask`` (bool).

    Both columns hold one length-len(categories) list per chunk, in category order.
    """
    _check_chunks(chunks)
    categories = list(categories)
    index = {category: i for i, category in enumerate(categories)}
    unknown = set(evidence.category) - set(index)
    if unknown:
        raise ValueError(f"evidence has unknown categories: {sorted(unknown)}")
    row_of = pd.Series(np.arange(len(chunks)), index=chunks.chunk_id)
    if not evidence.chunk_id.isin(row_of.index).all():
        raise ValueError("evidence refers to unknown chunk_id values")
    positive = np.zeros((len(chunks), len(categories)), dtype=bool)
    touched = np.zeros_like(positive)
    if len(evidence):
        rows = row_of.loc[evidence.chunk_id].to_numpy()
        cols = evidence.category.map(index).to_numpy()
        full = evidence.fully_contained.to_numpy(dtype=bool)
        positive[rows[full], cols[full]] = True
        touched[rows, cols] = True
    ambiguous = touched & ~positive
    result = chunks.copy()
    result["labels"] = positive.astype(int).tolist()
    result["label_mask"] = (~ambiguous).tolist()
    return result


def label_arrays(
    labeled: pd.DataFrame, n_categories: int = len(PRIORITY_CATEGORIES)
) -> tuple[np.ndarray, np.ndarray]:
    """Return (labels int8, label_mask bool) arrays; works after a Parquet reload.

    Shape is always (n_chunks, n_categories), including for an empty table.
    """
    labels = np.array(labeled["labels"].tolist(), dtype=np.int8)
    mask = np.array(labeled["label_mask"].tolist(), dtype=bool)
    return labels.reshape(-1, n_categories), mask.reshape(-1, n_categories)


def label_chunk_table(
    chunks: pd.DataFrame, spans: pd.DataFrame, categories=PRIORITY_CATEGORIES
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience wrapper: returns (labeled_chunks, evidence)."""
    evidence = build_evidence(chunks, spans, categories)
    return label_chunks(chunks, evidence, categories), evidence


def validate_labels(
    chunks: pd.DataFrame,
    spans: pd.DataFrame,
    labeled: pd.DataFrame,
    evidence: pd.DataFrame,
    categories=PRIORITY_CATEGORIES,
) -> dict:
    """Recompute everything with plain loops and compare; raise ValueError on mismatch.

    ``spans`` should be the raw priority rows (no de-duplication) so that containment
    counts are comparable with task #5 (3,229 of 3,241 on the training split).
    Returns a dictionary of summary counts for the validation report.
    """
    categories = list(categories)
    k = len(categories)
    _check_chunks(chunks)
    if list(labeled.columns) != list(chunks.columns) + ["labels", "label_mask"]:
        raise ValueError(
            "Labeled table must be the chunk table plus labels and label_mask"
        )
    if not labeled[list(chunks.columns)].equals(chunks):
        raise ValueError("Chunk columns were modified by labeling")
    if list(evidence.columns) != EVIDENCE_COLUMNS:
        raise ValueError("Unexpected evidence columns or column order")
    labels, mask = label_arrays(labeled, k)
    if labels.shape != (len(chunks), k) or mask.shape != labels.shape:
        raise ValueError(f"labels and label_mask must have {k} values per chunk")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("labels must be binary")
    if (~mask & (labels == 1)).any():
        raise ValueError("A positive label cannot be masked")

    priority = _priority_spans(spans, categories)
    by_contract = defaultdict(list)
    for s in priority.itertuples(index=False):
        by_contract[s.contract].append((s.category, int(s.start), int(s.end)))

    expected_rows, expected_labels, expected_mask = set(), [], []
    chunk_bounds = defaultdict(list)
    no_overlap = partial_only_chunks = multi_label = 0
    for c in chunks.itertuples(index=False):
        chunk_bounds[c.contract].append((int(c.start), int(c.end)))
        full, part = set(), set()
        seen = set()
        for cat, s, e in by_contract.get(c.contract, []):
            if not (c.start < e and s < c.end):
                continue
            is_full = c.start <= s and e <= c.end
            (full if is_full else part).add(cat)
            if (cat, s, e) not in seen:  # exact duplicates collapse to one record
                seen.add((cat, s, e))
                expected_rows.add(
                    (c.chunk_id, cat, s, e, max(c.start, s), min(c.end, e), is_full)
                )
        expected_labels.append([int(cat in full) for cat in categories])
        expected_mask.append(
            [not (cat in part and cat not in full) for cat in categories]
        )
        no_overlap += not (full or part)
        partial_only_chunks += bool(part - full)
        multi_label += len(full) >= 2
    if not np.array_equal(
        labels, np.array(expected_labels, dtype=np.int8).reshape(-1, k)
    ):
        raise ValueError("labels differ from the independent recomputation")
    if not np.array_equal(mask, np.array(expected_mask, dtype=bool).reshape(-1, k)):
        raise ValueError("label_mask differs from the independent recomputation")

    actual_rows = {
        (a, b, int(c), int(d), int(e), int(f), bool(g))
        for a, b, c, d, e, f, g in evidence[
            [
                "chunk_id",
                "category",
                "span_start",
                "span_end",
                "overlap_start",
                "overlap_end",
                "fully_contained",
            ]
        ].itertuples(index=False, name=None)
    }
    if len(actual_rows) != len(evidence):
        raise ValueError("Evidence contains duplicate rows")
    if actual_rows != expected_rows:
        raise ValueError("Evidence table differs from the independent recomputation")
    if not evidence.chunk_id.isin(chunks.chunk_id).all():
        raise ValueError("Evidence refers to unknown chunks")
    if not (
        (evidence.overlap_start >= evidence.span_start)
        & (evidence.overlap_end <= evidence.span_end)
        & (evidence.overlap_start < evidence.overlap_end)
    ).all():
        raise ValueError("Evidence overlap offsets are inconsistent")

    contained = 0
    uncovered = []
    for contract, items in by_contract.items():
        bounds = chunk_bounds.get(contract, [])
        for cat, s, e in items:
            if any(a <= s and e <= b for a, b in bounds):
                contained += 1
            else:
                uncovered.append(
                    {"contract": contract, "category": cat, "start": s, "end": e}
                )
    total = len(priority)
    per_category = {
        cat: {
            "positive_chunks": int(labels[:, i].sum()),
            "masked_chunks": int((~mask[:, i]).sum()),
        }
        for i, cat in enumerate(categories)
    }
    return {
        "chunks": len(chunks),
        "evidence_rows": len(evidence),
        "priority_spans": total,
        "duplicate_span_rows": int(priority.duplicated().sum()),
        "fully_contained_spans": contained,
        "not_fully_contained_spans": total - contained,
        "uncovered_spans": uncovered,
        "chunks_without_priority_overlap": no_overlap,
        "chunks_without_priority_overlap_pct": round(
            100 * no_overlap / max(len(chunks), 1), 2
        ),
        "chunks_with_partial_only_category": partial_only_chunks,
        "chunks_with_multiple_positive_categories": multi_label,
        "chunks_with_any_positive": int((labels.sum(axis=1) > 0).sum()),
        "per_category": per_category,
    }