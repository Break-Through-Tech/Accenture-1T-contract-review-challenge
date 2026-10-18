import numpy as np
import pandas as pd
import pytest

from src.evaluation import (
    DEFAULT_THRESHOLD,
    FOLD_COLUMNS,
    METRIC_COLUMNS,
    TUNING_COLUMNS,
    apply_thresholds,
    contract_folds,
    fold_of_rows,
    resolve_thresholds,
    score,
    threshold_map,
    tune_thresholds,
)

CATS = ["Governing Law", "Audit Rights"]


def chunks(contracts) -> pd.DataFrame:
    """Minimal chunk table: one row per entry, labels/masks unused by folds."""
    return pd.DataFrame({"contract": list(contracts)})


# --- thresholds -------------------------------------------------------------


def test_scalar_threshold_applies_to_every_category():
    assert resolve_thresholds(0.3, CATS).tolist() == [0.3, 0.3]


def test_mapping_threshold_follows_category_order():
    out = resolve_thresholds({"Audit Rights": 0.8, "Governing Law": 0.2}, CATS)
    assert out.tolist() == [0.2, 0.8]


def test_sequence_threshold_is_taken_in_order():
    assert resolve_thresholds([0.2, 0.8], CATS).tolist() == [0.2, 0.8]


def test_mapping_must_cover_every_category():
    with pytest.raises(ValueError, match="no threshold given"):
        resolve_thresholds({"Governing Law": 0.2}, CATS)


def test_mapping_rejects_unknown_category():
    with pytest.raises(ValueError, match="unknown categories"):
        resolve_thresholds({"Governing Law": 0.2, "Nope": 0.5}, CATS)


def test_sequence_length_must_match():
    with pytest.raises(ValueError, match="expected 2 thresholds"):
        resolve_thresholds([0.2, 0.4, 0.6], CATS)


def test_threshold_rejects_nan_and_negative():
    with pytest.raises(ValueError, match="finite"):
        resolve_thresholds([0.2, float("nan")], CATS)
    with pytest.raises(ValueError, match="negative"):
        resolve_thresholds([-0.1, 0.5], CATS)


def test_each_category_uses_its_own_threshold():
    proba = np.array([[0.4, 0.4]])
    out = apply_thresholds(proba, {"Governing Law": 0.3, "Audit Rights": 0.9}, CATS)
    assert out.tolist() == [[1, 0]]


def test_score_equal_to_threshold_counts_as_positive():
    assert apply_thresholds(np.array([[0.5, 0.5]]), 0.5, CATS).tolist() == [[1, 1]]


def test_apply_thresholds_rejects_wrong_width():
    with pytest.raises(ValueError, match="2 columns"):
        apply_thresholds(np.zeros((3, 5)), 0.5, CATS)


# --- scoring ----------------------------------------------------------------


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


def test_score_reports_the_threshold_it_used():
    labels = np.array([[1, 1]], dtype=np.int8)
    mask = np.ones((1, 2), dtype=bool)
    proba = np.array([[0.6, 0.6]])
    out = score(labels, mask, proba, CATS, {"Governing Law": 0.3, "Audit Rights": 0.9})
    assert out.set_index("category").loc["Governing Law", "threshold"] == 0.3
    assert out.set_index("category").loc["Audit Rights", "threshold"] == 0.9
    # Ten categories cannot share one number, so the average rows stay empty.
    assert pd.isna(out.set_index("category").loc["macro avg", "threshold"])


def test_shared_threshold_is_named_on_the_average_rows():
    labels = np.array([[1, 1]], dtype=np.int8)
    mask = np.ones((1, 2), dtype=bool)
    out = score(labels, mask, np.array([[0.9, 0.9]]), CATS, 0.4)
    assert out.set_index("category").loc["macro avg", "threshold"] == 0.4


def test_a_higher_threshold_trades_recall_for_precision():
    labels = np.array([[1], [1], [0]], dtype=np.int8)
    mask = np.ones((3, 1), dtype=bool)
    proba = np.array([[0.9], [0.4], [0.6]])
    low = score(labels, mask, proba, ["x"], 0.3).iloc[0]
    high = score(labels, mask, proba, ["x"], 0.8).iloc[0]
    assert (low.recall, low.precision) == (1.0, 0.6667)  # score() rounds to 4dp
    assert (high.recall, high.precision) == (0.5, 1.0)


def test_score_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="disagree in shape"):
        score(
            np.zeros((3, 2), dtype=np.int8),
            np.ones((3, 2), dtype=bool),
            np.zeros((4, 2)),
            CATS,
        )


# --- tuning -----------------------------------------------------------------


def test_tuning_finds_the_threshold_that_separates_the_classes():
    # Positives score 0.6, negatives 0.4: 0.5 is perfect, 0.3 is not.
    labels = np.array([[1], [1], [0], [0]], dtype=np.int8)
    mask = np.ones((4, 1), dtype=bool)
    proba = np.array([[0.6], [0.6], [0.4], [0.4]])
    out = tune_thresholds(labels, mask, proba, ["x"], grid=[0.3, 0.5, 0.7])
    assert list(out.columns) == TUNING_COLUMNS
    row = out.iloc[0]
    assert row.threshold == 0.5
    assert row.f1 == 1.0


def test_tuning_beats_the_default_on_a_skewed_category():
    # Every positive scores below 0.5, so the shared threshold finds nothing.
    labels = np.array([[1], [1], [0], [0]], dtype=np.int8)
    mask = np.ones((4, 1), dtype=bool)
    proba = np.array([[0.3], [0.25], [0.05], [0.05]])
    row = tune_thresholds(labels, mask, proba, ["x"], grid=[0.1, 0.5, 0.9]).iloc[0]
    assert row.default_f1 == 0.0
    assert row.threshold == 0.1
    assert row.f1 == 1.0
    assert row.f1_gain == 1.0


def test_ties_are_broken_towards_recall():
    # 0.2 and 0.3 both give the same F1; the lower one is kept.
    labels = np.array([[1], [0]], dtype=np.int8)
    mask = np.ones((2, 1), dtype=bool)
    proba = np.array([[0.5], [0.1]])
    row = tune_thresholds(labels, mask, proba, ["x"], grid=[0.2, 0.3]).iloc[0]
    assert row.threshold == 0.2


def test_tuning_keeps_the_default_for_a_category_with_no_positives():
    labels = np.zeros((3, 1), dtype=np.int8)
    mask = np.ones((3, 1), dtype=bool)
    row = tune_thresholds(labels, mask, np.full((3, 1), 0.2), ["x"]).iloc[0]
    assert row.threshold == DEFAULT_THRESHOLD
    assert row.support == 0
    assert np.isnan(row.f1)


def test_tuning_ignores_masked_pairs():
    labels = np.array([[1], [1], [0]], dtype=np.int8)
    mask = np.array([[True], [True], [False]])  # the masked row would force 0.9
    proba = np.array([[0.6], [0.6], [0.8]])
    row = tune_thresholds(labels, mask, proba, ["x"], grid=[0.5, 0.7, 0.9]).iloc[0]
    assert row.threshold == 0.5
    assert row.f1 == 1.0


def test_tuning_rejects_an_unsorted_grid():
    labels = np.array([[1], [0]], dtype=np.int8)
    mask = np.ones((2, 1), dtype=bool)
    with pytest.raises(ValueError, match="strictly increasing"):
        tune_thresholds(labels, mask, np.array([[0.6], [0.2]]), ["x"], grid=[0.5, 0.3])


def test_threshold_map_round_trips_into_score():
    labels = np.array([[1, 1]], dtype=np.int8)
    mask = np.ones((1, 2), dtype=bool)
    proba = np.array([[0.6, 0.6]])
    tuned = tune_thresholds(labels, mask, proba, CATS, grid=[0.4, 0.8])
    chosen = threshold_map(tuned)
    assert set(chosen) == set(CATS)
    out = score(labels, mask, proba, CATS, chosen)
    assert out.set_index("category").loc[CATS, "threshold"].tolist() == [0.4, 0.4]


# --- folds ------------------------------------------------------------------


def test_every_contract_lands_in_exactly_one_fold():
    data = chunks(["a", "a", "b", "c", "c", "c", "d", "e", "f"])
    folds = contract_folds(data, n_splits=3)
    assert list(folds.columns) == FOLD_COLUMNS
    assert len(folds) == 6
    assert folds.fold.nunique() == 3
    assert folds.chunks.sum() == len(data)
    assert folds.contract.is_monotonic_increasing


def test_folds_are_the_same_every_run():
    data = chunks(["a", "a", "b", "c", "d", "e"])
    first = contract_folds(data, n_splits=3)
    second = contract_folds(data.sample(frac=1, random_state=1), n_splits=3)
    pd.testing.assert_frame_equal(first, second)


def test_fold_of_rows_maps_chunks_back_to_their_contract_fold():
    data = chunks(["a", "b", "a", "c", "d", "e"])
    folds = contract_folds(data, n_splits=3)
    rows = fold_of_rows(data, folds)
    lookup = dict(zip(folds.contract, folds.fold))
    assert rows.tolist() == [lookup[c] for c in data.contract]


def test_fold_of_rows_rejects_an_unknown_contract():
    folds = contract_folds(chunks(["a", "b", "c", "d"]), n_splits=2)
    with pytest.raises(ValueError, match="have no fold"):
        fold_of_rows(chunks(["a", "zzz"]), folds)


def test_folds_reject_too_many_splits():
    with pytest.raises(ValueError, match="n_splits must be between"):
        contract_folds(chunks(["a", "b"]), n_splits=5)


def test_folds_reject_an_empty_table():
    with pytest.raises(ValueError, match="empty"):
        contract_folds(chunks([]), n_splits=2)
