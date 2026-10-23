import json

import numpy as np
import pandas as pd
import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, processors

from src.model import ContractClassifier
from src.train import TrainingConfig, evaluate, run_training, select_contracts


@pytest.fixture
def inputs(tmp_path):
    tokenizer = Tokenizer(models.WordPiece({
        "[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "law": 4,
    }, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.BertProcessing(("[SEP]", 3), ("[CLS]", 2))
    rows = pd.DataFrame([
        {"contract": f"c{i}", "chunk_id": f"{i}:{j}", "split": "train",
         "text": "law", "labels": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
         "label_mask": [True] * 9 + [False]}
        for i in range(6) for j in range(2)
    ])
    folds = pd.DataFrame({"contract": [f"c{i}" for i in range(6)], "fold": [0, 0, 1, 1, 2, 2]})
    rows.to_parquet(tmp_path / "labels.parquet", index=False)
    folds.to_csv(tmp_path / "folds.csv", index=False)
    config = TrainingConfig(
        labels=tmp_path / "labels.parquet", folds=tmp_path / "folds.csv",
        output_dir=tmp_path / "run", tiny=True, cpu_threads=1, device="cpu",
        max_contracts=4, max_steps=2,
    )
    return config, tokenizer, rows, folds


def test_smoke_training_outputs_and_repeatability(inputs, tmp_path):
    config, tokenizer, _rows, _folds = inputs
    previous_threads = torch.get_num_threads()
    try:
        result = run_training(config, tokenizer=tokenizer)
        assert result["optimizer_steps"] == 2
        assert result["processed_train_chunks"] == 4
        assert result["checkpoint_reload_verified"]
        assert result["train_contracts"] + result["validation_contracts"] == 4
        assert np.isfinite(result["history"][0]["validation_loss"])
        saved = json.loads((config.output_dir / "metadata.json").read_text())
        assert saved["mode"] == "tiny-random"
        predictions = pd.read_parquet(config.output_dir / "validation_predictions.parquet")
        assert len(predictions) == result["validation_chunks"]
        assert np.asarray(predictions.probabilities.tolist()).shape[1] == 10
        selected = pd.read_csv(config.output_dir / "selected_contracts.csv")
        assert set(selected.loc[selected.role == "train", "contract"]).isdisjoint(
            selected.loc[selected.role == "validation", "contract"]
        )
        assert set(predictions.contract) == set(selected.loc[selected.role == "validation", "contract"])
        metrics = pd.read_csv(config.output_dir / "validation_metrics.csv")
        assert set(metrics.category).issuperset({"macro avg", "micro avg"})
        assert (config.output_dir / "checkpoint/tokenizer.json").exists()
        model = ContractClassifier.from_pretrained(config.output_dir / "checkpoint", local_files_only=True)
        assert model.classifier.out_features == 10
        with pytest.raises(ValueError, match="empty"):
            run_training(config, tokenizer=tokenizer)
        config.output_dir = tmp_path / "repeat"
        repeated = run_training(config, tokenizer=tokenizer)
        assert repeated["history"] == result["history"]
    finally:
        torch.set_num_threads(previous_threads)


def test_subset_deterministic_whole_contracts(inputs):
    _, _, rows, folds = inputs
    train, validation = select_contracts(rows, folds, validation_fold=0, count=4)
    train2, val2 = select_contracts(rows.iloc[::-1], folds.iloc[::-1], validation_fold=0, count=4)
    assert set(train.contract) == set(train2.contract)
    assert set(validation.contract) == set(val2.contract)
    assert set(train.contract).isdisjoint(validation.contract)
    assert len(train) + len(validation) == 8
    with pytest.raises(ValueError, match="count"):
        select_contracts(rows, folds, validation_fold=0, count=7)


def test_all_masked_batches_skip_optimizer_updates(inputs):
    config, tokenizer, rows, _folds = inputs
    for index in rows.index[rows.contract == "c2"]:
        rows.at[index, "labels"] = [0] * 10
        rows.at[index, "label_mask"] = [False] * 10
    rows.to_parquet(config.labels, index=False)
    config.max_contracts = None
    config.max_steps = None
    config.batch_size = 1
    previous_threads = torch.get_num_threads()
    try:
        result = run_training(config, tokenizer=tokenizer)
        assert result["history"][0]["skipped_batches"] == 2
        assert result["optimizer_steps"] == 6
        assert result["processed_train_chunks"] == 6
    finally:
        torch.set_num_threads(previous_threads)


@pytest.mark.parametrize("name,value", [
    ("epochs", 0), ("batch_size", 0), ("learning_rate", float("nan")),
    ("weight_decay", -1), ("max_contracts", 1), ("max_steps", 0),
])
def test_invalid_configuration(name, value):
    config = TrainingConfig()
    setattr(config, name, value)
    with pytest.raises(ValueError):
        config.validate()


def test_evaluation_weights_by_valid_entries():
    class FakeModel:
        def eval(self):
            pass

        def __call__(self, **batch):
            from types import SimpleNamespace
            return SimpleNamespace(loss=batch["loss"], logits=torch.zeros(1, 10))

    loader = [
        {"label_mask": torch.tensor([[True] * 10]), "loss": torch.tensor(2.0)},
        {"label_mask": torch.tensor([[True] + [False] * 9]), "loss": torch.tensor(4.0)},
    ]
    loss, probabilities = evaluate(FakeModel(), loader, torch.device("cpu"))
    assert loss == pytest.approx(24 / 11)
    assert probabilities.shape == (2, 10)
