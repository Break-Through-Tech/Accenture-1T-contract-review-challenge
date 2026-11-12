"""Offline tests for task #13: tiny random encoder, no downloads."""

import argparse
import json

import numpy as np
import pandas as pd
import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import BertConfig, BertModel

from scripts.final_training import run_final, run_select, run_sweep, run_test
from src.candidates import EncoderClassifier
from src.cuad_data import PRIORITY_CATEGORIES
from src.final_training import (
    final_thresholds,
    load_checkpoint,
    predict,
    save_checkpoint,
    select_config,
    train_encoder,
)

K = len(PRIORITY_CATEGORIES)


def tiny_tokenizer():
    tok = Tokenizer(models.WordPiece({
        "[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "law": 4, "other": 5,
    }, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.post_processor = processors.BertProcessing(("[SEP]", 3), ("[CLS]", 2))
    return tok


def tiny_model(seed=0):
    torch.manual_seed(seed)
    return EncoderClassifier(BertModel(BertConfig(
        vocab_size=6, hidden_size=8, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=8, max_position_embeddings=512,
    )))


def chunk_rows(split, contracts, prefix="c"):
    rows = []
    for c in range(contracts):
        for k in range(4):
            positive = k == 0
            rows.append({
                "contract": f"{prefix}{c}", "chunk_id": f"{prefix}{c}:{k}", "split": split,
                "text": "law law" if positive else "other",
                "labels": [int(positive)] * K, "label_mask": [True] * K,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def ws(tmp_path):
    train = chunk_rows("train", 9)
    test = chunk_rows("test", 3, prefix="t")
    train.sample(frac=1, random_state=0).to_parquet(tmp_path / "train.parquet", index=False)
    test.to_parquet(tmp_path / "test.parquet", index=False)
    pd.DataFrame({"contract": [f"c{i}" for i in range(9)], "fold": [i % 3 for i in range(9)]}
                 ).to_csv(tmp_path / "folds.csv", index=False)
    return tmp_path


def sweep_args(root, lr, fold, **extra):
    values = {
        "learning_rate": lr, "validation_fold": fold, "output_dir": root / "runs",
        "labels": root / "train.parquet", "folds": root / "folds.csv", "epochs": 2, "batch_size": 4,
        "seed": 42, "train_contracts": None, "validation_contracts": None, "max_steps": None,
        "local_files_only": True,
    }
    return argparse.Namespace(**{**values, **extra})


def select_args(root):
    return argparse.Namespace(
        labels=root / "train.parquet", runs_dir=[root / "runs"], output_dir=root / "results",
        folds=root / "folds.csv", seed=42, bootstrap=50, local_files_only=True,
    )


def run_full_sweep(root, lrs=(1e-3, 3e-3), folds=(0, 1, 2)):
    tok = tiny_tokenizer()
    for lr in lrs:
        for fold in folds:
            run_sweep(sweep_args(root, lr, fold), tokenizer=tok, model=tiny_model())


# ------------------------------------------------------------------ sweep


def test_sweep_saves_every_epoch_and_resumes(ws):
    tok = tiny_tokenizer()
    result = run_sweep(sweep_args(ws, 1e-3, 0), tokenizer=tok, model=tiny_model())
    run_dir = ws / "runs" / "lr0.001-fold0-seed42"
    assert [e["epoch"] for e in result["epoch_scores"]] == [1, 2]
    for epoch in (1, 2):
        preds = pd.read_parquet(run_dir / f"epoch{epoch}_predictions.parquet")
        assert len(preds) == 12  # 3 validation contracts x 4 chunks
        assert len(preds.probabilities.iloc[0]) == K
    assert not result["smoke"]
    stamp = (run_dir / "result.json").stat().st_mtime_ns
    run_sweep(sweep_args(ws, 1e-3, 0), tokenizer=tok, model=None)  # model=None would fail if rerun
    assert (run_dir / "result.json").stat().st_mtime_ns == stamp

    broken = ws / "runs" / "lr0.001-fold1-seed42"
    broken.mkdir()
    (broken / "partial.txt").write_text("x")
    run_sweep(sweep_args(ws, 1e-3, 1), tokenizer=tok, model=tiny_model())
    assert not (broken / "partial.txt").exists()


def test_epoch_snapshot_matches_a_run_that_stops_there():
    # sweep scores epoch 1 of a 2-epoch schedule; final with epochs=1 on that schedule
    # must reproduce it exactly, or the selected config is not what gets trained
    rows = chunk_rows("train", 4)
    tok = tiny_tokenizer()
    snapshot = {}

    def keep(epoch, model):
        snapshot[epoch] = predict(model, rows, tok, device="cpu")[1]

    train_encoder(tiny_model(), rows, tok, epochs=2, learning_rate=1e-2, device="cpu",
                  on_epoch_end=keep)
    stopped = tiny_model()
    train_encoder(stopped, rows, tok, epochs=1, schedule_epochs=2, learning_rate=1e-2,
                  device="cpu")
    np.testing.assert_allclose(predict(stopped, rows, tok, device="cpu")[1], snapshot[1],
                               atol=1e-6)
    assert not np.allclose(snapshot[1], snapshot[2])


def test_pos_weight_comes_from_training_rows():
    rows = chunk_rows("train", 4)  # 1 positive : 3 negatives per category
    out = train_encoder(tiny_model(), rows, tiny_tokenizer(), epochs=1, learning_rate=1e-3,
                        device="cpu", max_steps=1)
    assert set(out["positive_weights"].values()) == {3.0}
    with pytest.raises(ValueError):
        train_encoder(tiny_model(), rows, tiny_tokenizer(), epochs=3, schedule_epochs=2,
                      learning_rate=1e-3, device="cpu")


# ----------------------------------------------------------------- select


def _summary_and_pairs(macro, ci):
    summary = pd.DataFrame([
        {"config": n, "learning_rate": lr, "epochs": ep, "macro_f1": m}
        for (n, lr, ep), m in zip(
            [("lr2e-05-ep3", 2e-5, 3), ("lr3e-05-ep2", 3e-5, 2), ("lr2e-05-ep2", 2e-5, 2)], macro)
    ])
    pairs = []
    names = list(summary.config)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            low, high = ci[(a, b)]
            pairs.append({"arm_a": a, "arm_b": b, "metric": "macro_f1",
                          "difference": (low + high) / 2, "ci_low": low, "ci_high": high})
    return summary, pd.DataFrame(pairs)


def test_clear_winner_is_picked_even_with_more_epochs():
    summary, pairs = _summary_and_pairs([0.75, 0.70, 0.69], {
        ("lr2e-05-ep3", "lr3e-05-ep2"): (0.02, 0.08),
        ("lr2e-05-ep3", "lr2e-05-ep2"): (0.03, 0.09),
        ("lr3e-05-ep2", "lr2e-05-ep2"): (-0.01, 0.03),
    })
    decision = select_config(summary, pairs)
    assert decision["pick"] == "lr2e-05-ep3" and decision["decided_by"] == "macro_f1"


def test_ties_go_to_fewest_epochs_then_lowest_lr():
    summary, pairs = _summary_and_pairs([0.75, 0.74, 0.74], {
        ("lr2e-05-ep3", "lr3e-05-ep2"): (-0.01, 0.03),
        ("lr2e-05-ep3", "lr2e-05-ep2"): (-0.01, 0.03),
        ("lr3e-05-ep2", "lr2e-05-ep2"): (-0.02, 0.02),
    })
    decision = select_config(summary, pairs)
    assert decision["pick"] == "lr2e-05-ep2"
    assert decision["leader"] == "lr2e-05-ep3"
    assert decision["decided_by"] == "fewest_epochs_then_lowest_lr"


def test_thresholds_flag_low_support_and_cover_every_category():
    rng = np.random.default_rng(0)
    n = 300
    labels = (rng.random((n, K)) < 0.1).astype(int)
    proba = np.clip(labels * 0.6 + rng.random((n, K)) * 0.5, 0, 1)
    rows = pd.DataFrame({
        "contract": [f"c{i // 10}" for i in range(n)], "chunk_id": [str(i) for i in range(n)],
        "labels": labels.tolist(), "label_mask": [[True] * K] * n, "text": "x", "split": "train",
    })
    fold_of = np.array([(i // 10) % 3 for i in range(n)])
    table = final_thresholds(rows, fold_of, proba)
    assert list(table.category) == list(PRIORITY_CATEGORIES)
    assert table.flag.str.contains("low support").all()  # ~30 positives each
    assert (table.fold_threshold_min <= table.fold_threshold_max).all()


def test_select_end_to_end_and_incomplete_learning_rates_are_left_out(ws):
    run_full_sweep(ws, lrs=(1e-3, 3e-3))
    run_sweep(sweep_args(ws, 5e-3, 0), tokenizer=tiny_tokenizer(), model=tiny_model())
    selection = run_select(select_args(ws))
    assert selection["learning_rates_incomplete"] == [5e-3]
    assert selection["learning_rates_compared"] == [1e-3, 3e-3]
    assert selection["decision"]["pick"] in {
        f"lr{lr:g}-ep{ep}" for lr in (1e-3, 3e-3) for ep in (1, 2)}
    assert set(selection["thresholds"]) == set(PRIORITY_CATEGORIES)
    assert selection["schedule_epochs"] == 2 and not selection["official_test_split_used"]
    out = ws / "results"
    assert len(pd.read_csv(out / "sweep_summary.csv")) == 4  # 2 lrs x 2 epochs
    assert len(pd.read_csv(out / "thresholds.csv")) == K
    assert (ws / "runs" / "oof_predictions_selected.parquet").exists()


def test_smoke_sweep_runs_cannot_be_selected(ws):
    run_sweep(sweep_args(ws, 1e-3, 0, max_steps=1), tokenizer=tiny_tokenizer(),
              model=tiny_model())
    with pytest.raises(ValueError, match="smoke"):
        run_select(select_args(ws))


# ------------------------------------------------------------ final + test


def test_checkpoint_round_trip(tmp_path):
    rows = chunk_rows("train", 2)
    model, tok = tiny_model(), tiny_tokenizer()
    save_checkpoint(model, tok, tmp_path / "ckpt", {"thresholds": {}})
    restored = load_checkpoint(tmp_path / "ckpt")
    np.testing.assert_allclose(predict(model.eval(), rows, tok, device="cpu")[1],
                               predict(restored, rows, tok, device="cpu")[1], atol=1e-6)


def test_final_then_one_locked_test_run(ws):
    run_full_sweep(ws, lrs=(1e-3,))
    run_select(select_args(ws))
    final_dir = ws / "final"
    result = run_final(argparse.Namespace(
        labels=ws / "train.parquet", selection=ws / "results" / "selection.json",
        output_dir=final_dir, train_contracts=None, max_steps=None, local_files_only=True,
    ), tokenizer=tiny_tokenizer(), model=tiny_model())
    assert result["train_contracts"] == 9 and result["checkpoint_reload_verified"]
    assert result["schedule_epochs"] == 2

    test_args = argparse.Namespace(
        labels=ws / "test.parquet", checkpoint=final_dir / "checkpoint",
        selection=ws / "results" / "selection.json", results_dir=ws / "results",
        output_dir=ws / "drive", batch_size=8, test_contracts=None, smoke_ok=False,
        allow_second_test_run=False, local_files_only=True,
    )
    record = run_test(test_args, tokenizer=tiny_tokenizer())
    assert record["test_contracts"] == 3 and not record["smoke"]
    assert (ws / "results" / "test_per_category.csv").exists()
    preds = pd.read_parquet(ws / "drive" / "test_predictions.parquet")
    assert list(preds.columns) == ["contract", "chunk_id", *PRIORITY_CATEGORIES]
    with pytest.raises(SystemExit, match="already scored"):
        run_test(test_args, tokenizer=tiny_tokenizer())


def test_test_run_refuses_training_rows_and_smoke_checkpoints(ws):
    run_full_sweep(ws, lrs=(1e-3,))
    run_select(select_args(ws))
    run_final(argparse.Namespace(
        labels=ws / "train.parquet", selection=ws / "results" / "selection.json",
        output_dir=ws / "final", train_contracts=None, max_steps=1, local_files_only=True,
    ), tokenizer=tiny_tokenizer(), model=tiny_model())
    args = argparse.Namespace(
        labels=ws / "test.parquet", checkpoint=ws / "final" / "checkpoint",
        selection=ws / "results" / "selection.json", results_dir=ws / "smoke-results",
        output_dir=ws / "drive", batch_size=8, test_contracts=None, smoke_ok=False,
        allow_second_test_run=False, local_files_only=True,
    )
    with pytest.raises(SystemExit, match="not reportable"):
        run_test(args, tokenizer=tiny_tokenizer())
    args.smoke_ok, args.labels = True, ws / "train.parquet"
    with pytest.raises(ValueError, match="test split"):
        run_test(args, tokenizer=tiny_tokenizer())
    assert not (ws / "smoke-results" / "test_run.json").exists()


def test_selection_json_is_valid_json(ws):
    run_full_sweep(ws, lrs=(1e-3,))
    run_select(select_args(ws))
    data = json.loads((ws / "results" / "selection.json").read_text())
    assert data["arm"] == "pos_weight"
    assert data["encoder"] == "nlpaueb/legal-bert-base-uncased"
