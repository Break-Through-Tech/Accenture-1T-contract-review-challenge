import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.cuad_data import PRIORITY_CATEGORIES
from src.labeling import (
    EVIDENCE_COLUMNS,
    build_evidence,
    label_arrays,
    label_chunk_table,
    label_chunks,
    validate_labels,
)

CAT = {c: i for i, c in enumerate(PRIORITY_CATEGORIES)}


def chunk_table(bounds, contract="c1"):
    return pd.DataFrame(
        [
            {
                "contract": contract,
                "split": "train",
                "chunk_id": f"{contract}:{a}:{b}",
                "start": a,
                "end": b,
                "text": "x" * (b - a),
                "n_tokens": 1,
            }
            for a, b in bounds
        ],
        columns=["contract", "split", "chunk_id", "start", "end", "text", "n_tokens"],
    )


def span_table(rows, contract="c1"):
    return pd.DataFrame(
        [
            {"contract": contract, "category": c, "start": s, "end": e, "text": "t"}
            for c, s, e in rows
        ],
        columns=["contract", "category", "start", "end", "text"],
    )


def test_shared_example():
    sample = json.loads(
        (Path(__file__).parent / "fixtures/chunking_example.json").read_text()
    )
    chunks = pd.DataFrame(sample["chunks"])
    spans = pd.DataFrame(sample["spans"])
    labeled, evidence = label_chunk_table(chunks, spans)
    labels, mask = label_arrays(labeled)

    gov, audit = CAT["Governing Law"], CAT["Audit Rights"]
    assert labels[0, gov] == 1 and mask[0, gov]  # complete evidence
    assert labels[1, audit] == 1 and mask[1, audit]  # complete in neighbour
    assert labels[2, audit] == 0 and not mask[2, audit]  # partial only: ambiguous
    assert labels[3:].sum() == 0 and mask[3:].all()  # no evidence
    assert labels.sum() == 2 and (~mask).sum() == 1
    assert len(labeled) == len(chunks) == 5
    assert evidence.fully_contained.tolist() == [True, True, False]
    validate_labels(chunks, spans, labeled, evidence)


def test_boundary_touching_is_not_overlap():
    chunks = chunk_table([(0, 10), (10, 20)])
    spans = span_table([("Governing Law", 10, 15)])
    evidence = build_evidence(chunks, spans)
    assert evidence.chunk_id.tolist() == ["c1:10:20"]


def test_multiple_categories_in_one_chunk():
    chunks = chunk_table([(0, 100)])
    spans = span_table([("Governing Law", 5, 20), ("Audit Rights", 30, 50)])
    labeled, evidence = label_chunk_table(chunks, spans)
    labels, mask = label_arrays(labeled)
    assert labels[0].sum() == 2 and mask.all()
    assert labels[0, CAT["Governing Law"]] == labels[0, CAT["Audit Rights"]] == 1
    assert len(evidence) == 2


def test_full_plus_partial_same_category_stays_positive():
    chunks = chunk_table([(0, 50)])
    spans = span_table([("Exclusivity", 10, 20), ("Exclusivity", 40, 80)])
    labeled, evidence = label_chunk_table(chunks, spans)
    labels, mask = label_arrays(labeled)
    i = CAT["Exclusivity"]
    assert labels[0, i] == 1 and mask[0, i]
    assert evidence.fully_contained.tolist() == [True, False]  # both retained


def test_partial_in_one_category_does_not_mask_others():
    chunks = chunk_table([(0, 50)])
    spans = span_table([("Exclusivity", 40, 80), ("Audit Rights", 1, 5)])
    labels, mask = label_arrays(label_chunk_table(chunks, spans)[0])
    assert not mask[0, CAT["Exclusivity"]] and labels[0, CAT["Exclusivity"]] == 0
    assert mask[0, CAT["Audit Rights"]] and labels[0, CAT["Audit Rights"]] == 1
    assert mask[0].sum() == len(PRIORITY_CATEGORIES) - 1


def test_repeated_spans_are_collapsed_in_evidence_but_not_in_counts():
    chunks = chunk_table([(0, 50), (30, 90)])
    spans = span_table([("Governing Law", 35, 45)] * 2)
    labeled, evidence = label_chunk_table(chunks, spans)
    assert len(evidence) == 2  # one record per chunk, not per duplicate
    report = validate_labels(chunks, spans, labeled, evidence)
    assert report["priority_spans"] == 2
    assert report["duplicate_span_rows"] == 1
    assert report["fully_contained_spans"] == 2


def test_span_spanning_two_chunks_is_partial_in_both_unless_overlap_covers_it():
    chunks = chunk_table([(0, 50), (40, 90)])
    spans = span_table([("Non-Compete", 30, 60)])
    labeled, evidence = label_chunk_table(chunks, spans)
    labels, mask = label_arrays(labeled)
    i = CAT["Non-Compete"]
    assert labels[:, i].tolist() == [0, 0] and mask[:, i].tolist() == [False, False]
    assert evidence.overlap_start.tolist() == [30, 40]
    assert evidence.overlap_end.tolist() == [50, 60]
    report = validate_labels(chunks, spans, labeled, evidence)
    assert report["not_fully_contained_spans"] == 1
    assert report["uncovered_spans"][0]["category"] == "Non-Compete"


def test_non_priority_categories_and_other_contracts_are_ignored():
    chunks = chunk_table([(0, 50)])
    spans = pd.concat(
        [
            span_table([("Parties", 0, 10)]),
            span_table([("Governing Law", 0, 10)], contract="other"),
        ]
    )
    labeled, evidence = label_chunk_table(chunks, spans)
    assert evidence.empty and list(evidence.columns) == EVIDENCE_COLUMNS
    labels, mask = label_arrays(labeled)
    assert labels.sum() == 0 and mask.all()


def test_chunks_without_annotations_are_kept():
    chunks = chunk_table([(0, 10), (10, 20)])
    labeled, _ = label_chunk_table(chunks, span_table([]))
    assert len(labeled) == 2
    assert label_arrays(labeled)[0].sum() == 0


def test_does_not_modify_inputs_and_keeps_order():
    chunks = chunk_table([(0, 50), (40, 90)])
    before = chunks.copy()
    spans = span_table([("Governing Law", 5, 10)])
    labeled, _ = label_chunk_table(chunks, spans)
    pd.testing.assert_frame_equal(chunks, before)
    assert labeled.chunk_id.tolist() == chunks.chunk_id.tolist()


def test_parquet_round_trip(tmp_path):
    chunks = chunk_table([(0, 50), (40, 90)])
    labeled, evidence = label_chunk_table(
        chunks, span_table([("Audit Rights", 45, 48), ("Governing Law", 10, 60)])
    )
    labeled.to_parquet(tmp_path / "l.parquet", index=False)
    evidence.to_parquet(tmp_path / "e.parquet", index=False)
    back, back_evidence = (
        pd.read_parquet(tmp_path / "l.parquet"),
        pd.read_parquet(tmp_path / "e.parquet"),
    )
    for a, b in zip(label_arrays(labeled), label_arrays(back)):
        assert np.array_equal(a, b)
    pd.testing.assert_frame_equal(evidence, back_evidence)
    validate_labels(
        chunks,
        span_table([("Audit Rights", 45, 48), ("Governing Law", 10, 60)]),
        back,
        back_evidence,
    )


def test_inputs_are_checked():
    chunks = chunk_table([(0, 10)])
    with pytest.raises(ValueError, match="missing columns"):
        build_evidence(chunks, pd.DataFrame({"contract": ["c1"]}))
    with pytest.raises(ValueError, match="start < end"):
        build_evidence(chunks, span_table([("Governing Law", 5, 5)]))
    with pytest.raises(ValueError, match="unique"):
        build_evidence(pd.concat([chunks, chunks]), span_table([]))
    bad = pd.DataFrame(
        {"chunk_id": ["nope"], "category": ["Audit Rights"], "fully_contained": [True]}
    )
    with pytest.raises(ValueError, match="unknown chunk_id"):
        label_chunks(chunks, bad)


@pytest.fixture
def good():
    chunks = chunk_table([(0, 50), (40, 90), (90, 120)])
    spans = span_table([("Governing Law", 5, 10), ("Audit Rights", 45, 60)])
    labeled, evidence = label_chunk_table(chunks, spans)
    return chunks, spans, labeled, evidence


def test_validation_passes_and_counts(good):
    report = validate_labels(*good)
    assert report["chunks"] == 3
    assert report["chunks_without_priority_overlap"] == 1
    assert report["chunks_with_partial_only_category"] == 1
    assert report["fully_contained_spans"] == 2


def test_validation_detects_tampered_labels(good):
    chunks, spans, labeled, evidence = good
    labeled = labeled.copy()
    labeled.at[2, "labels"] = [1] + [0] * 9
    with pytest.raises(ValueError, match="labels differ"):
        validate_labels(chunks, spans, labeled, evidence)


def test_validation_detects_tampered_mask(good):
    chunks, spans, labeled, evidence = good
    labeled = labeled.copy()
    labeled.at[0, "label_mask"] = [True] * 10
    labeled.at[1, "label_mask"] = [True] * 10
    with pytest.raises(ValueError, match="label_mask differs"):
        validate_labels(chunks, spans, labeled, evidence)


def test_validation_detects_positive_but_masked(good):
    chunks, spans, labeled, evidence = good
    labeled = labeled.copy()
    labeled.at[0, "label_mask"] = [False] * 10
    with pytest.raises(ValueError, match="cannot be masked"):
        validate_labels(chunks, spans, labeled, evidence)


def test_validation_detects_bad_evidence(good):
    chunks, spans, labeled, evidence = good
    changed = evidence.copy()
    changed.loc[0, "fully_contained"] = False
    with pytest.raises(ValueError, match="Evidence table differs"):
        validate_labels(chunks, spans, labeled, changed)
    with pytest.raises(ValueError, match="Evidence table differs"):
        validate_labels(chunks, spans, labeled, evidence.iloc[1:])


def test_validation_detects_modified_chunk_columns(good):
    chunks, spans, labeled, evidence = good
    labeled = labeled.copy()
    labeled.loc[0, "text"] = "changed"
    with pytest.raises(ValueError, match="modified"):
        validate_labels(chunks, spans, labeled, evidence)


def test_empty_chunk_table():
    chunks = chunk_table([])
    labeled, evidence = label_chunk_table(chunks, span_table([("Audit Rights", 0, 5)]))
    labels, mask = label_arrays(labeled)
    assert labels.shape == mask.shape == (0, len(PRIORITY_CATEGORIES))
    assert evidence.empty
    report = validate_labels(chunks, span_table([]), labeled, evidence)
    assert report["chunks"] == 0 and report["chunks_without_priority_overlap_pct"] == 0