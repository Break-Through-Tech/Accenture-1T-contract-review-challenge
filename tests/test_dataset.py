"""Offline dataset tests with a small, real BERT-style tokenizer."""

import pandas as pd
import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from torch.utils.data import DataLoader

from src.cuad_data import PRIORITY_CATEGORIES
from src.dataset import ChunkDataset, split_by_fold


@pytest.fixture
def tokenizer():
    tok = Tokenizer(models.WordPiece({
        "[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3,
        "law": 4, "audit": 5,
    }, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.post_processor = processors.BertProcessing(("[SEP]", 3), ("[CLS]", 2))
    return tok


@pytest.fixture
def rows():
    return pd.DataFrame([
        {"contract": contract, "split": "train", "chunk_id": str(i),
         "text": text, "labels": labels, "label_mask": mask}
        for i, (contract, text, labels, mask) in enumerate([
            ("a", "law audit", [0, 0, 1, 0, 0, 0, 0, 0, 0, 1], [True] * 10),
            ("a", "law", [0] * 10, [False] + [True] * 9),
            ("b", "audit", [0] * 10, [True] * 10),
            ("c", "law", [0] * 10, [True] * 10),
        ])
    ], index=[10, 20, 30, 40])


def test_exact_tokens_padding_labels_and_masks(rows, tokenizer):
    dataset = ChunkDataset(rows, tokenizer, max_length=6)
    item = dataset[0]
    assert dataset.categories == tuple(PRIORITY_CATEGORIES)
    assert item["input_ids"].tolist() == [2, 4, 5, 3, 0, 0]
    assert item["attention_mask"].tolist() == [1, 1, 1, 1, 0, 0]
    assert item["input_ids"].dtype == item["attention_mask"].dtype == torch.long
    assert item["labels"].dtype == torch.float32
    assert item["label_mask"].dtype == torch.bool
    assert item["labels"].tolist() == rows.iloc[0].labels
    assert dataset[1]["label_mask"].tolist() == rows.iloc[1].label_mask
    assert dataset.rows.chunk_id.tolist() == rows.chunk_id.tolist()
    item["labels"][2] = 0
    assert dataset[0]["labels"][2] == 1


def test_dataloader_batches_and_empty_dataset(rows, tokenizer):
    dataset = ChunkDataset(rows, tokenizer)
    batches = list(DataLoader(dataset, batch_size=3))
    assert batches[0]["input_ids"].shape == (3, 512)
    assert batches[0]["labels"].shape == (3, 10)
    assert batches[1]["label_mask"].shape == (1, 10)
    assert len(ChunkDataset(rows.iloc[:0], tokenizer)) == 0


def test_token_budget_never_truncates(rows, tokenizer):
    rows.loc[10, "text"] = " ".join(["law"] * 510)
    assert ChunkDataset(rows, tokenizer)[0]["attention_mask"].sum() == 512
    rows.loc[10, "text"] += " law"
    with pytest.raises(ValueError, match="Re-chunk"):
        ChunkDataset(rows, tokenizer)[0]


def test_tokenizer_settings_are_isolated(rows, tokenizer):
    tokenizer.enable_truncation(max_length=3)
    tokenizer.enable_padding(length=10)
    item = ChunkDataset(rows, tokenizer, max_length=6)[0]
    assert item["input_ids"].tolist() == [2, 4, 5, 3, 0, 0]
    assert tokenizer.truncation["max_length"] == 3
    assert tokenizer.padding["length"] == 10


@pytest.mark.parametrize("column,value", [
    ("labels", [0] * 9), ("labels", [0.5] * 10),
    ("labels", [float("nan")] * 10), ("label_mask", [2] * 10),
    ("text", "  "), ("contract", None),
])
def test_reject_bad_data(rows, tokenizer, column, value):
    rows.at[10, column] = value
    with pytest.raises(ValueError):
        ChunkDataset(rows, tokenizer)


def test_reject_positive_mask_duplicates_and_missing_columns(rows, tokenizer):
    bad = rows.copy()
    bad.at[10, "label_mask"] = [False] * 10
    with pytest.raises(ValueError, match="Positive"):
        ChunkDataset(bad, tokenizer)
    bad = rows.copy()
    bad.loc[20, "chunk_id"] = "0"
    with pytest.raises(ValueError, match="unique"):
        ChunkDataset(bad, tokenizer)
    with pytest.raises(ValueError, match="Missing"):
        ChunkDataset(rows.drop(columns="labels"), tokenizer)


@pytest.fixture
def folds():
    return pd.DataFrame({"contract": ["a", "b", "c"], "fold": [0, 1, 1]})


def test_saved_folds_keep_contracts_together(rows, folds):
    train, val = split_by_fold(rows, folds, validation_fold=0)
    assert train.chunk_id.tolist() == ["2", "3"]
    assert val.chunk_id.tolist() == ["0", "1"]
    assert set(train.contract).isdisjoint(val.contract)
    train2, val2 = split_by_fold(rows, folds.iloc[::-1], validation_fold=0)
    pd.testing.assert_frame_equal(train, train2)
    pd.testing.assert_frame_equal(val, val2)


@pytest.mark.parametrize("case", ["duplicate", "missing", "fraction", "test", "empty", "unknown"])
def test_reject_invalid_fold_splits(rows, folds, case):
    validation_fold = 0
    if case == "duplicate":
        folds = pd.concat([folds, folds.iloc[:1]])
    elif case == "missing":
        folds = folds.iloc[1:]
    elif case == "fraction":
        folds["fold"] = [0.5, 1, 1]
    elif case == "test":
        rows.loc[10, "split"] = "test"
    elif case == "empty":
        rows = rows.iloc[:2]
    else:
        validation_fold = 9
    with pytest.raises(ValueError):
        split_by_fold(rows, folds, validation_fold=validation_fold)
