"""Offline tests for Task #18's clause register."""

import numpy as np
import pandas as pd
import pytest

from src.clause_register import (
    CATEGORIES,
    REGISTER_COLUMNS,
    build_register,
    document_key,
    locate_excerpt,
    match_gold,
    merge_windows,
    parse_chunk_ids,
    review_boundary,
    score_chunks,
)

LAW = "This Agreement shall be governed by the laws of the State of New York. "
FILLER = "The parties will cooperate on marketing activities in good faith. "
TEXT = FILLER * 10 + LAW + FILLER * 10 + LAW + FILLER * 10
CONTRACTS = pd.DataFrame({"contract": ["c1"], "text": [TEXT]})
T = {c: 0.5 for c in CATEGORIES}


def chunk(start, end, **probs):
    row = {"contract": "c1", "chunk_id": f"{document_key('c1', TEXT, 'test')}:{start}:{end}"}
    row.update({c: 0.01 for c in CATEGORIES})
    row.update({c.replace("_", " "): p for c, p in probs.items()})
    return row


def preds(*rows):
    return pd.DataFrame(list(rows))


GL = "Governing_Law"


def test_parse_chunk_ids_reads_offsets():
    out = parse_chunk_ids(preds(chunk(5, 40)))
    assert (out.start.iloc[0], out.end.iloc[0]) == (5, 40)


def test_merge_windows_overlap_touch_and_gap():
    merged = merge_windows([0, 50, 100, 300], [60, 100, 150, 400])
    assert [(s, e) for s, e, _ in merged] == [(0, 150), (300, 400)]
    assert merged[0][2] == [0, 1, 2]


def test_overlapping_positive_chunks_become_one_clause():
    reg = build_register(preds(
        chunk(0, 800, **{GL: 0.9}), chunk(600, 1400, **{GL: 0.97}), chunk(1400, 2000),
    ), CONTRACTS, T)
    assert list(reg.columns) == REGISTER_COLUMNS
    assert len(reg) == 1
    r = reg.iloc[0]
    assert (r.start, r.end, r.n_chunks) == (0, 1400, 2)
    assert r.confidence == 0.97 and r.category == "Governing Law"
    assert r.evidence_text == TEXT[0:1400]


def test_separate_positive_chunks_stay_separate_and_below_threshold_is_dropped():
    reg = build_register(preds(
        chunk(0, 500, **{GL: 0.9}), chunk(900, 1300, **{GL: 0.4}), chunk(1500, 2000, **{GL: 0.8}),
    ), CONTRACTS, T)
    assert list(zip(reg.start, reg.end)) == [(0, 500), (1500, 2000)]


def test_excerpt_offsets_point_into_the_source_and_cover_the_clause():
    i = TEXT.index("This Agreement shall be governed")
    reg = build_register(preds(chunk(i - 300, i + 400, **{GL: 0.9})), CONTRACTS, T)
    r = reg.iloc[0]
    assert TEXT[r.excerpt_start:r.excerpt_end] == r.excerpt_text
    assert r.excerpt_start <= i < r.excerpt_end
    assert "governing" in r.excerpt_rules or r.excerpt_rules


def test_review_boundary_follows_the_rubric():
    assert review_boundary(0.8) == 0.82
    low = {**T, "Governing Law": 0.8}
    reg = build_register(preds(chunk(0, 500, **{GL: 0.82}), chunk(900, 1300, **{GL: 0.81})),
                         CONTRACTS, low)
    assert reg.needs_review.tolist() == [False, True]  # equal to the boundary clears it
    assert "low confidence" in reg.review_reason.iloc[1]


def test_heading_alone_is_extended_into_the_body():
    text = "4.2 Covenant Not to Compete. " + "The Consultant shall not engage in any business. " * 15
    found = locate_excerpt(text, "Non-Compete")
    assert found is not None
    a, b, _ = found
    assert a == 0 and b - a >= 400


def test_no_keyword_hit_gives_no_excerpt():
    assert locate_excerpt(FILLER * 5, "Governing Law") is None
    reg = build_register(preds(chunk(0, 300, **{GL: 0.9})), CONTRACTS, T)
    assert pd.isna(reg.excerpt_start.iloc[0]) and reg.excerpt_text.iloc[0] == ""


def test_rejects_scores_from_different_text():
    bad = preds(chunk(0, 300, **{GL: 0.9}))
    bad.loc[0, "chunk_id"] = "deadbeef:0:300"
    with pytest.raises(ValueError, match="do not match"):
        build_register(bad, CONTRACTS, T)


def test_rejects_offsets_outside_text_and_bad_thresholds():
    with pytest.raises(ValueError, match="outside"):
        build_register(preds(chunk(0, len(TEXT) + 5)), CONTRACTS, T)
    with pytest.raises(ValueError, match="one threshold"):
        build_register(preds(chunk(0, 300)), CONTRACTS, {"Governing Law": 0.5})


def test_match_gold_precision_recall_and_dedup():
    i = TEXT.index("This Agreement")
    j = TEXT.index("This Agreement", i + 1)
    reg = build_register(preds(
        chunk(i - 50, i + 300, **{GL: 0.9}), chunk(i + 200, i + 500, **{GL: 0.9}),
        chunk(i + 540, j - 40, **{GL: 0.9}),  # separate window, no gold inside
    ), CONTRACTS, T)
    gold = pd.DataFrame({"contract": ["c1", "c1"], "category": ["Governing Law"] * 2,
                         "start": [i, j], "end": [i + len(LAW), j + len(LAW)]})
    table = match_gold(reg, gold, pd.Series({"Governing Law": 3})).set_index("category")
    row = table.loc["Governing Law"]
    assert (row.positive_chunks, row.clauses) == (3, 2)
    assert row.clause_precision == 0.5 and row.span_recall == 0.5


def test_score_chunks_runs_a_saved_checkpoint(tmp_path):
    from src.final_training import save_checkpoint
    from tests.test_final_training import tiny_model, tiny_tokenizer

    save_checkpoint(tiny_model(), tiny_tokenizer(), tmp_path / "ckpt", {"thresholds": T})
    chunks = pd.DataFrame({"contract": ["a", "a"], "chunk_id": ["k:0:3", "k:3:9"],
                           "text": ["law law", "other"]})
    out = score_chunks(chunks, tmp_path / "ckpt", device="cpu")
    assert list(out.columns) == ["contract", "chunk_id", *CATEGORIES]
    probs = out[CATEGORIES].to_numpy()
    assert probs.shape == (2, 10) and np.all((probs >= 0) & (probs <= 1))
