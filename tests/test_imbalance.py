"""Offline tests: a tiny random encoder, no model or tokenizer downloads."""

import argparse
import json
import math

import numpy as np
import pandas as pd
import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from torch.nn import functional as F
from transformers import BertConfig, BertModel

from scripts.imbalance_experiment import run_summarize, run_train
from src.candidates import EncoderClassifier
from src.cuad_data import PRIORITY_CATEGORIES
from src.imbalance import (
    ARMS,
    WEAK_CATEGORIES,
    choose_arm,
    cross_fit_decisions,
    epoch_rows,
    make_loss,
    paired_bootstrap,
    positive_weights,
)
from src.loss import masked_bce_with_logits, masked_focal_with_logits

# ------------------------------------------------------------------ losses


def test_pos_weight_multiplies_positive_entries_only():
    logits = torch.zeros(1, 2)
    labels = torch.tensor([[1.0, 0.0]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    loss = masked_bce_with_logits(logits, labels, mask, torch.tensor([3.0, 100.0]))
    assert loss.item() == pytest.approx((3 * math.log(2) + math.log(2)) / 2)


def test_pos_weight_follows_category_when_entries_are_masked():
    logits = torch.zeros(1, 3)
    labels = torch.tensor([[1.0, 1.0, 1.0]])
    mask = torch.tensor([[False, False, True]])
    weights = torch.tensor([100.0, 100.0, 2.0])
    assert masked_bce_with_logits(logits, labels, mask, weights).item() == pytest.approx(
        2 * math.log(2)
    )


def test_pos_weight_of_one_is_plain_bce_and_wrong_shape_raises():
    logits = torch.tensor([[2.0, -1.0], [0.5, 3.0]])
    labels = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    mask = torch.ones(2, 2, dtype=torch.bool)
    torch.testing.assert_close(
        masked_bce_with_logits(logits, labels, mask, torch.ones(2)),
        masked_bce_with_logits(logits, labels, mask),
    )
    with pytest.raises(ValueError):
        masked_bce_with_logits(logits, labels, mask, torch.ones(3))


def test_focal_known_value():
    # p=0.5 for both entries: bce=log 2, modulating factor 0.5**2.
    logits, labels = torch.zeros(1, 2), torch.tensor([[1.0, 0.0]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    expected = (0.25 + 0.75) * 0.25 * math.log(2) / 2
    assert masked_focal_with_logits(logits, labels, mask).item() == pytest.approx(expected)


def test_focal_reduces_to_half_bce_without_focusing():
    logits = torch.tensor([[2.0, -1.0, 0.3]])
    labels = torch.tensor([[1.0, 0.0, 0.0]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    torch.testing.assert_close(
        masked_focal_with_logits(logits, labels, mask, gamma=0.0, alpha=0.5),
        F.binary_cross_entropy_with_logits(logits, labels) / 2,
    )


def test_focal_discounts_easy_entries_more_than_hard_ones():
    labels, mask = torch.tensor([[0.0]]), torch.ones(1, 1, dtype=torch.bool)
    easy, hard = torch.tensor([[-4.0]]), torch.tensor([[2.0]])

    def ratio(logits):
        return (
            masked_focal_with_logits(logits, labels, mask)
            / masked_bce_with_logits(logits, labels, mask)
        )

    assert ratio(easy) < ratio(hard)


def test_focal_ignores_masked_values_and_handles_empty_batches():
    labels = torch.tensor([[1.0, 0.0]])
    mask = torch.tensor([[True, False]])
    logits = torch.tensor([[0.0, float("nan")]], requires_grad=True)
    loss = masked_focal_with_logits(logits, labels, mask)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad[0, 1] == 0
    empty = torch.full((1, 2), float("nan"), requires_grad=True)
    assert masked_focal_with_logits(empty, labels, torch.zeros(1, 2, dtype=torch.bool)).item() == 0


def test_focal_extreme_logits_have_finite_gradients():
    logits = torch.tensor([[1000.0, -1000.0]], requires_grad=True)
    loss = masked_focal_with_logits(
        logits, torch.tensor([[1.0, 0.0]]), torch.ones(1, 2, dtype=torch.bool)
    )
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()


@pytest.mark.parametrize("kwargs", [{"gamma": -1.0}, {"alpha": 1.5}])
def test_focal_invalid_settings_raise(kwargs):
    with pytest.raises(ValueError):
        masked_focal_with_logits(
            torch.zeros(1, 1), torch.zeros(1, 1), torch.ones(1, 1, dtype=torch.bool), **kwargs
        )


# --------------------------------------------------------------- strategies


def test_positive_weights_ignore_masked_entries():
    labels = np.array([[1, 0], [0, 1], [0, 0], [0, 0]])
    mask = np.array([[True, True], [True, True], [False, True], [True, True]])
    np.testing.assert_allclose(positive_weights(labels, mask), [2.0, 3.0])
    with pytest.raises(ValueError, match="positive"):
        positive_weights(np.zeros((2, 2)), np.ones((2, 2), dtype=bool))


def test_every_arm_has_a_working_loss():
    labels = np.array([[1, 0], [0, 1], [0, 0]])
    mask = np.ones_like(labels, dtype=bool)
    logits = torch.zeros(3, 2)
    y, valid = torch.tensor(labels, dtype=torch.float32), torch.tensor(mask)
    losses = {
        name: make_loss(arm, labels, mask, torch.device("cpu"))(logits, y, valid).item()
        for name, arm in ARMS.items()
    }
    assert losses["bce"] == losses["downsample"] == pytest.approx(math.log(2))
    assert losses["pos_weight"] > losses["bce"] > losses["focal"] > 0


def test_downsampling_keeps_every_positive_and_samples_negatives_by_epoch():
    labels = np.zeros((100, 2), dtype=int)
    labels[:10, 0] = 1
    labels[5:15, 1] = 1  # 15 chunks have a positive; a chunk may have two
    arm = ARMS["downsample"]
    first, second = epoch_rows(labels, arm, 42, 0), epoch_rows(labels, arm, 42, 1)
    assert len(first) == len(second) == 15 + 30
    assert set(range(15)) <= set(first) and set(range(15)) <= set(second)
    assert len(set(first)) == len(first)
    assert not np.array_equal(first, second)
    np.testing.assert_array_equal(first, epoch_rows(labels, arm, 42, 0))
    assert (labels[np.setdiff1d(np.arange(100), first)].sum(axis=1) == 0).all()


def test_other_arms_keep_all_rows_and_downsampling_is_capped_by_available_negatives():
    labels = np.array([[1], [1], [0]])
    for name in ("bce", "pos_weight", "focal"):
        np.testing.assert_array_equal(epoch_rows(labels, ARMS[name], 42, 0), [0, 1, 2])
    np.testing.assert_array_equal(epoch_rows(labels, ARMS["downsample"], 42, 0), [0, 1, 2])


# ---------------------------------------------------------------- comparison


def test_cross_fit_thresholds_come_only_from_other_folds():
    # Fold 0 separates at 0.8, fold 1 at 0.2. Each fold must receive the other's cut,
    # which misclassifies its own rows; tuning on itself would be perfect.
    proba = np.array([[0.9], [0.7], [0.3], [0.1]])
    labels = np.array([[1], [0], [1], [0]])
    mask = np.ones_like(labels, dtype=bool)
    fold_of = np.array([0, 0, 1, 1])
    decisions, table = cross_fit_decisions(labels, mask, proba, fold_of, ["only"])
    assert table.loc[table.fold == 0, "only"].item() == pytest.approx(0.11)
    assert table.loc[table.fold == 1, "only"].item() == pytest.approx(0.71)
    np.testing.assert_array_equal(decisions.ravel(), [1, 1, 0, 0])
    with pytest.raises(ValueError, match="two folds"):
        cross_fit_decisions(labels, mask, proba, np.zeros(4, dtype=int), ["only"])


def test_bootstrap_is_paired_and_respects_masks():
    categories = ["a", "b"]
    labels = np.array([[1, 0]] * 10 + [[0, 1]] * 10)
    mask = np.ones_like(labels, dtype=bool)
    contracts = np.array([f"c{i // 2}" for i in range(20)])
    perfect = labels.astype(np.int8)
    worse = perfect.copy()
    worse[[0, 2, 4, 6, 8], 0] = 0  # one miss in each of the five contracts with a positive
    pairs = paired_bootstrap(
        labels, mask, {"good": perfect, "bad": worse, "same": perfect.copy()},
        contracts, categories, weak=("a",), samples=300,
    )
    macro = pairs[(pairs.arm_a == "good") & (pairs.arm_b == "bad") & (pairs.metric == "macro_f1")].iloc[0]
    expected_bad = (2 * 5 / (2 * 5 + 5) + 1.0) / 2
    assert macro.difference == pytest.approx(1 - expected_bad)
    assert macro.ci_low > 0
    same = pairs[(pairs.arm_a == "good") & (pairs.arm_b == "same")]
    assert (same.difference == 0).all() and (same.ci_low == 0).all() and (same.ci_high == 0).all()
    # A false positive on a masked entry must not count.
    extra = perfect.copy()
    extra[0, 1] = 1
    mask[0, 1] = False
    masked = paired_bootstrap(labels, mask, {"x": perfect, "y": extra}, contracts, categories,
                              weak=("a",), samples=10)
    assert (masked.difference == 0).all()


def _pairs(entries):
    rows = []
    for metric, a, b, difference, low, high in entries:
        rows.append({"arm_a": a, "arm_b": b, "metric": metric, "difference": difference,
                     "ci_low": low, "ci_high": high})
    return pd.DataFrame(rows)


def _comparisons(macro, weak):
    """Every pair needed by choose_arm; ``macro``/``weak`` map (a, b) to (low, high)."""
    arms = ["bce", "pos_weight", "focal", "downsample"]
    entries = []
    for i, a in enumerate(arms):
        for b in arms[i + 1:]:
            for metric, table in (("macro_f1", macro), ("weak_f1", weak)):
                low, high = table.get((a, b), (-0.01, 0.01))
                entries.append((metric, a, b, (low + high) / 2, low, high))
    return _pairs(entries)


def _summary(macro, weak):
    return pd.DataFrame({"arm": list(macro), "macro_f1": list(macro.values()),
                         "weak_f1": [weak[a] for a in macro]})


def test_clear_macro_winner_is_picked_regardless_of_simplicity():
    summary = _summary({"bce": .6, "pos_weight": .6, "focal": .7, "downsample": .6},
                       {"bce": .5, "pos_weight": .5, "focal": .5, "downsample": .9})
    wins = {("pos_weight", "focal"): (-.2, -.05), ("focal", "downsample"): (.05, .2),
            ("bce", "focal"): (-.2, -.05)}
    result = choose_arm(summary, _comparisons(wins, {}))
    assert result["pick"] == "focal"
    assert result["decided_by"] == "macro_f1"
    assert result["clearly_beats_control_macro_f1"]


def test_macro_tie_is_settled_by_weak_categories():
    summary = _summary({"bce": .6, "pos_weight": .7, "focal": .69, "downsample": .5},
                       {"bce": .4, "pos_weight": .4, "focal": .6, "downsample": .9})
    macro = {("pos_weight", "downsample"): (.1, .3)}
    weak = {("pos_weight", "focal"): (-.3, -.1)}
    result = choose_arm(summary, _comparisons(macro, weak))
    # downsample has the best weak F1 but was already out on macro F1.
    assert result["tied_on_macro_f1"] == ["pos_weight", "focal"]
    assert result["pick"] == "focal"
    assert result["decided_by"] == "weak_f1"


def test_full_tie_falls_to_the_simplest_and_flags_no_gain_over_control():
    summary = _summary({"bce": .7, "pos_weight": .70, "focal": .71, "downsample": .69},
                       {"bce": .5, "pos_weight": .5, "focal": .51, "downsample": .5})
    result = choose_arm(summary, _comparisons({}, {}))
    assert result["pick"] == "pos_weight"
    assert result["decided_by"] == "simplicity"
    assert not result["clearly_beats_control_macro_f1"]
    assert not result["clearly_worse_than_control_macro_f1"]


def test_weak_categories_are_real_categories():
    assert set(WEAK_CATEGORIES) <= set(PRIORITY_CATEGORIES)


# ---------------------------------------------------------------- end to end


@pytest.fixture
def workspace(tmp_path):
    tokenizer = Tokenizer(models.WordPiece({
        "[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "law": 4, "other": 5,
    }, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.BertProcessing(("[SEP]", 3), ("[CLS]", 2))
    rows, generator = [], np.random.default_rng(0)
    for contract in range(9):
        for chunk in range(4):
            positive = chunk == 0
            labels = [int(positive)] * 10
            rows.append({
                "contract": f"c{contract}", "chunk_id": f"{contract}:{chunk}", "split": "train",
                "text": "law law" if positive else "other", "labels": labels,
                "label_mask": [True] * 10,
            })
    generator.shuffle(rows)
    rows = pd.DataFrame(rows)
    folds = pd.DataFrame({"contract": [f"c{i}" for i in range(9)], "fold": [i % 3 for i in range(9)]})
    rows.to_parquet(tmp_path / "labels.parquet", index=False)
    folds.to_csv(tmp_path / "folds.csv", index=False)
    return tmp_path, tokenizer


def _train_args(root, arm, fold, **overrides):
    values = {
        "arm": arm, "validation_fold": fold, "output_dir": root / "runs",
        "labels": root / "labels.parquet", "folds": root / "folds.csv",
        "seed": 42, "subset_seed": 42, "epochs": 1, "batch_size": 4,
        "learning_rate": 1e-3, "weight_decay": 0.01, "warmup_ratio": 0.1,
        "local_files_only": True, "train_contracts": None,
        "validation_contracts": None, "max_steps": None,
    }
    return argparse.Namespace(**{**values, **overrides})


def _tiny_model():
    torch.manual_seed(0)
    return EncoderClassifier(BertModel(BertConfig(
        vocab_size=6, hidden_size=8, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=8, max_position_embeddings=512,
    )))


def test_runs_resume_and_summarize_on_complete_folds_only(workspace):
    root, tokenizer = workspace
    for arm in ARMS:
        for fold in (0, 1):
            result = run_train(_train_args(root, arm, fold), tokenizer=tokenizer, model=_tiny_model())
            assert not result["smoke"]
    # fold 2 is finished by only one arm, and must be left out rather than compared.
    run_train(_train_args(root, "focal", 2), tokenizer=tokenizer, model=_tiny_model())

    downsampled = json.loads((root / "runs/downsample-fold0-seed42/result.json").read_text())
    plain = json.loads((root / "runs/bce-fold0-seed42/result.json").read_text())
    assert downsampled["chunks_per_epoch"] < plain["chunks_per_epoch"] == plain["train_chunks"]
    # Validation is never downsampled.
    assert downsampled["validation_chunks"] == plain["validation_chunks"] == 12

    finished = root / "runs/bce-fold0-seed42"
    marker = (finished / "result.json").stat().st_mtime_ns
    run_train(_train_args(root, "bce", 0), tokenizer=tokenizer, model=None)  # would fail if rerun
    assert (finished / "result.json").stat().st_mtime_ns == marker

    # An interrupted run has no result.json and is started over.
    interrupted = root / "runs/bce-fold2-seed42"
    interrupted.mkdir()
    (interrupted / "stale.txt").write_text("partial")
    run_train(_train_args(root, "bce", 2), tokenizer=tokenizer, model=_tiny_model())
    assert not (interrupted / "stale.txt").exists()
    assert (interrupted / "result.json").exists()
    (interrupted / "result.json").unlink()

    out = root / "summary"
    metadata = run_summarize(argparse.Namespace(
        runs_dir=[root / "runs"], output_dir=out, labels=root / "labels.parquet",
        folds=root / "folds.csv", seed=42, bootstrap=50,
    ))
    assert metadata["folds_compared"] == [0, 1]
    assert ["focal", 2] in metadata["runs_ignored_incomplete_fold"]
    assert metadata["decision"]["pick"] in ("pos_weight", "focal", "downsample")
    summary = pd.read_csv(out / "arm_summary.csv")
    assert set(summary.arm) == set(ARMS)
    per_category = pd.read_csv(out / "per_category.csv")
    assert set(PRIORITY_CATEGORIES) <= set(per_category.category)
    assert len(pd.read_csv(out / "per_fold.csv")) == 8
    assert len(pd.read_csv(out / "thresholds.csv")) == 8


def test_smoke_runs_cannot_be_summarized(workspace):
    root, tokenizer = workspace
    run_train(_train_args(root, "bce", 0, max_steps=1), tokenizer=tokenizer, model=_tiny_model())
    with pytest.raises(ValueError, match="smoke"):
        run_summarize(argparse.Namespace(
            runs_dir=[root / "runs"], output_dir=root / "summary",
            labels=root / "labels.parquet", folds=root / "folds.csv", seed=42, bootstrap=10,
        ))


def test_arms_start_from_the_same_weights_and_see_the_same_validation_rows(workspace):
    root, tokenizer = workspace
    predictions = {}
    for arm in ("bce", "focal"):
        run_train(_train_args(root, arm, 0, max_steps=None), tokenizer=tokenizer, model=_tiny_model())
        predictions[arm] = pd.read_parquet(root / f"runs/{arm}-fold0-seed42/validation_predictions.parquet")
    assert list(predictions["bce"].chunk_id) == list(predictions["focal"].chunk_id)
    assert not np.allclose(
        np.array(predictions["bce"].probabilities.tolist()),
        np.array(predictions["focal"].probabilities.tolist()),
    )
