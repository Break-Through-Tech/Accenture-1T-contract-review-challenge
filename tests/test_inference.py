"""Offline tests for task #18: window merging, evidence offsets, register check."""

import argparse
import json

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.inference import run_predict, run_register
from src.cuad_data import PRIORITY_CATEGORIES
from src.dataset import ChunkDataset
from src.inference import (
    REGISTER_COLUMNS,
    add_evidence,
    chunk_offsets,
    merge_windows,
    predict_chunks,
    score_register,
)
from tests.test_final_training import tiny_model, tiny_tokenizer

LAW, CAP = "Governing Law", "Cap On Liability"
THRESHOLDS = dict.fromkeys(PRIORITY_CATEGORIES, 0.5)


def windows(contract, spans_and_probs):
    """Prediction rows; each item is (start, end, {category: probability})."""
    rows = []
    for start, end, probs in spans_and_probs:
        row = {"contract": contract, "chunk_id": f"key-{contract}:{start}:{end}"}
        row.update(dict.fromkeys(PRIORITY_CATEGORIES, 0.0))
        row.update(probs)
        rows.append(row)
    return pd.DataFrame(rows)


def test_chunk_offsets_come_from_the_id():
    offsets = chunk_offsets(pd.Series(["abc:0:10", "a:b:5:9"]))
    assert offsets.start.tolist() == [0, 5] and offsets.end.tolist() == [10, 9]
    with pytest.raises(ValueError):
        chunk_offsets(pd.Series(["abc:9:3"]))


def test_overlapping_positive_windows_become_one_clause():
    predictions = windows("c", [
        (0, 100, {LAW: 0.6}), (80, 180, {LAW: 0.9}), (160, 260, {LAW: 0.7}),
        (240, 340, {LAW: 0.1}), (320, 420, {LAW: 0.8}),
    ])
    register = merge_windows(predictions, THRESHOLDS)
    assert register[["start", "end"]].values.tolist() == [[0, 260], [320, 420]]
    assert register.confidence.tolist() == [0.9, 0.8]
    assert register.n_chunks.tolist() == [3, 1]
    assert register.chunk_ids[0].count(";") == 2


def test_each_category_and_contract_is_merged_separately():
    predictions = pd.concat([
        windows("a", [(0, 100, {LAW: 0.9, CAP: 0.2}), (80, 180, {CAP: 0.95})]),
        windows("b", [(0, 100, {LAW: 0.9})]),
    ])
    register = merge_windows(predictions, {**THRESHOLDS, CAP: 0.9})
    got = register[["contract", "category", "start", "end"]].values.tolist()
    assert got == [["a", LAW, 0, 100], ["a", CAP, 80, 180], ["b", LAW, 0, 100]]


def test_threshold_is_inclusive_and_every_category_needs_one():
    predictions = windows("c", [(0, 10, {LAW: 0.5})])
    assert len(merge_windows(predictions, THRESHOLDS)) == 1
    with pytest.raises(ValueError):
        merge_windows(predictions, {LAW: 0.5})


def test_no_positive_windows_gives_an_empty_register():
    register = merge_windows(windows("c", [(0, 10, {})]), THRESHOLDS)
    assert register.empty and list(register.columns) == REGISTER_COLUMNS[:-1]


def test_evidence_matches_the_source_and_drops_edge_whitespace():
    contracts = pd.DataFrame({"contract": ["c"], "text": ["  This Agreement is governed by law.  \n"]})
    register = add_evidence(merge_windows(windows("c", [(0, 39, {LAW: 0.9})]), THRESHOLDS),
                            contracts)
    row = register.iloc[0]
    assert row.evidence_text == "This Agreement is governed by law."
    assert contracts.text[0][row.start:row.end] == row.evidence_text
    assert list(register.columns) == REGISTER_COLUMNS


def test_evidence_rejects_spans_past_the_text():
    contracts = pd.DataFrame({"contract": ["c"], "text": ["short"]})
    with pytest.raises(ValueError):
        add_evidence(merge_windows(windows("c", [(0, 50, {LAW: 0.9})]), THRESHOLDS), contracts)


def test_register_check_counts_overlaps_and_contract_presence():
    register = pd.DataFrame({
        "contract": ["a", "a", "b"], "category": [LAW, LAW, LAW],
        "start": [0, 500, 0], "end": [100, 600, 100],
    })
    spans = pd.DataFrame({
        "contract": ["a", "a", "c"], "category": [LAW, LAW, LAW],
        "start": [50, 300, 0], "end": [60, 310, 10],
    })
    table = score_register(register, spans, ["a", "b", "c"]).set_index("category")
    law = table.loc[LAW]
    assert law.entries == 3 and law.gold_clauses == 3
    assert law.entry_precision == pytest.approx(1 / 3, abs=1e-4)  # only a's first entry hits gold
    assert law.clause_recall == pytest.approx(1 / 3, abs=1e-4)  # only a's first clause is found
    assert law.contract_precision == 0.5 and law.contract_recall == 0.5  # a right, b and c wrong
    assert table.loc["macro avg", "entries"] == 3


def test_predict_chunks_matches_the_labeled_path():
    rows = pd.DataFrame({
        "contract": ["c", "c", "d"], "chunk_id": ["c:0:3", "c:2:5", "d:0:3"],
        "text": ["law law", "other", "law other"],
    })
    model, tok = tiny_model(), tiny_tokenizer()
    proba = predict_chunks(model, rows, tok, batch_size=2, device="cpu")
    labeled = rows.assign(labels=[[0] * 10] * 3, label_mask=[[True] * 10] * 3)
    batch = next(iter(torch.utils.data.DataLoader(ChunkDataset(labeled, tok), batch_size=3)))
    with torch.no_grad():
        expected = model(input_ids=batch["input_ids"],
                         attention_mask=batch["attention_mask"]).logits.sigmoid().numpy()
    assert proba.shape == (3, 10)
    np.testing.assert_allclose(proba, expected, atol=1e-6)


@pytest.fixture
def raw(tmp_path):
    """A two-contract test.json in CUAD's format."""
    texts = {
        "Alpha": "This Agreement is governed by the laws of Ohio. other other other",
        "Beta": "other other. Liability is capped at fees paid. other",
    }
    gold = {"Alpha": ("Governing Law", "governed by the laws of Ohio"),
            "Beta": ("Cap On Liability", "Liability is capped at fees paid")}
    data = []
    for title, text in texts.items():
        category, answer = gold[title]
        data.append({"title": title, "paragraphs": [{"context": text, "qas": [{
            "id": f"{title}__{category}",
            "answers": [{"text": answer, "answer_start": text.index(answer)}],
        }]}]})
    (tmp_path / "test.json").write_text(json.dumps({"data": data}))
    return tmp_path


def test_predict_then_register_end_to_end(raw):
    tok = tiny_tokenizer()
    predicted = raw / "drive" / "predictions.parquet"
    report = run_predict(argparse.Namespace(
        checkpoint=None, output=predicted, compare=None, raw_dir=raw, batch_size=4,
        local_files_only=True,
    ), tokenizer=tok, model=tiny_model())
    assert report["contracts"] == 2

    selection = raw / "selection.json"
    selection.write_text(json.dumps({"thresholds": dict.fromkeys(PRIORITY_CATEGORIES, 0.0)}))
    summary = run_register(argparse.Namespace(
        predictions=predicted, selection=selection, output_dir=raw / "drive",
        results_dir=raw / "docs", raw_dir=raw,
    ))
    register = pd.read_parquet(raw / "drive" / "clause_register.parquet")
    # threshold 0 makes every window positive: each contract fits in one window
    assert summary["register_entries"] == len(register) == 2 * len(PRIORITY_CATEGORIES)
    assert summary["windows_merged_away"] == 0
    assert (raw / "docs" / "clause_register_sample.csv").exists()
    check = pd.read_csv(raw / "docs" / "register_check.csv").set_index("category")
    assert check.loc["Governing Law", "clause_recall"] == 1.0
