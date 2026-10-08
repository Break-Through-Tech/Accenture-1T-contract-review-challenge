import numpy as np
import pandas as pd
import pytest

from src.tfidf_baseline import (
    METRIC_COLUMNS,
    BaselineConfig,
    cross_validate,
    evaluate,
    fit,
    score,
)

CATS = ["Governing Law", "Audit Rights"]
CONFIG = BaselineConfig(min_df=1, max_df=1.0)
LAW = "this agreement shall be governed by the laws of the state of new york"
AUDIT = "licensor may audit and inspect the books and records of licensee"
OTHER = "the parties shall cooperate in good faith on marketing activities"


def table(rows):
    """rows: (contract, text, labels, mask)."""
    return pd.DataFrame(
        [
            {
                "contract": c,
                "chunk_id": f"{c}:{i}",
                "text": t,
                "labels": list(y),
                "label_mask": list(m),
            }
            for i, (c, t, y, m) in enumerate(rows)
        ]
    )


def toy(n_contracts=6):
    rows = []
    for k in range(n_contracts):
        c = f"c{k}"
        rows += [
            (c, f"{LAW} section {k}", (1, 0), (True, True)),
            (c, f"{AUDIT} section {k}", (0, 1), (True, True)),
            (c, f"{LAW} {AUDIT}", (1, 1), (True, True)),
            (c, f"{OTHER} section {k}", (0, 0), (True, True)),
        ]
    return table(rows)


def test_fit_predict_is_multilabel_and_deterministic():
    data = toy()
    model = fit(data, CATS, CONFIG)
    pred = model.predict(data["text"])
    assert pred.shape == (len(data), 2)
    assert pred.tolist() == data["labels"].tolist()
    again = fit(data, CATS, CONFIG).predict_proba(data["text"])
    assert np.array_equal(model.predict_proba(data["text"]), again)


def test_top_terms_are_category_specific():
    terms = fit(toy(), CATS, CONFIG).top_terms(n=20)
    assert list(terms.columns) == ["category", "rank", "term", "weight"]
    law = set(terms[terms.category == "Governing Law"].term)
    audit = set(terms[terms.category == "Audit Rights"].term)
    assert "governed" in law and "audit" not in law
    assert "audit" in audit and "governed" not in audit


def test_masked_pairs_are_excluded_from_scoring():
    labels = np.array([[1, 0], [0, 1], [0, 0]], dtype=np.int8)
    mask = np.array([[True, True], [True, True], [False, True]])
    proba = np.array([[0.9, 0.1], [0.1, 0.9], [0.9, 0.1]])  # row 3 would be a FP
    out = score(labels, mask, proba, CATS)
    assert list(out.columns) == METRIC_COLUMNS
    law = out[out.category == "Governing Law"].iloc[0]
    assert (law.precision, law.recall, law.f1) == (1.0, 1.0, 1.0)
    assert (law.scored_chunks, law.masked_chunks) == (2, 1)
    assert out.category.tolist()[-2:] == ["macro avg", "micro avg"]


def test_masked_pairs_are_excluded_from_fitting():
    base = toy()
    # A masked row whose label would otherwise teach "audit" -> Governing Law.
    noisy = pd.concat(
        [base, table([("c9", AUDIT, (1, 0), (False, True))] * 20)], ignore_index=True
    )
    proba = fit(noisy, CATS, CONFIG).predict_proba([AUDIT])
    assert proba[0, 0] < 0.5


def test_score_counts_false_positives_and_negatives():
    labels = np.array([[1], [1], [0], [0]], dtype=np.int8)
    mask = np.ones((4, 1), dtype=bool)
    proba = np.array([[0.9], [0.2], [0.8], [0.1]])
    row = score(labels, mask, proba, ["x"]).iloc[0]
    assert (row.precision, row.recall, row.f1) == (0.5, 0.5, 0.5)


def test_cross_validation_keeps_contracts_together():
    data = toy(n_contracts=6)
    out = cross_validate(data, CATS, CONFIG, n_splits=3)
    assert out[out.category == "macro avg"].f1.iloc[0] == 1.0


def test_evaluate_matches_score():
    data = toy()
    model = fit(data, CATS, CONFIG)
    metrics, proba = evaluate(model, data)
    assert proba.shape == (len(data), 2)
    assert metrics[metrics.category == "micro avg"].f1.iloc[0] == 1.0


def test_rejects_category_without_positives():
    data = table(
        [("c1", LAW, (1, 0), (True, True)), ("c2", OTHER, (0, 0), (True, True))]
    )
    with pytest.raises(ValueError, match="Audit Rights"):
        fit(data, CATS, CONFIG)


def test_rejects_missing_columns():
    with pytest.raises(ValueError, match="missing columns"):
        fit(pd.DataFrame({"text": ["a"]}), CATS, CONFIG)
