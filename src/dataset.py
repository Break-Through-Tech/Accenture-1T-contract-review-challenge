"""PyTorch inputs for labeled CUAD chunks, in PRIORITY_CATEGORIES order.

Reuse src.chunking.load_tokenizer() so training uses the chunk export's pinned
BERT vocabulary. Inputs include special tokens and are padded to max_length.
Oversized inputs raise rather than silently dropping annotated evidence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from tokenizers import Tokenizer
from torch.utils.data import Dataset

from src.cuad_data import PRIORITY_CATEGORIES
from src.evaluation import fold_of_rows


class ChunkDataset(Dataset):
    """One chunk per item; float32 labels and bool masks have shape (10,).

    ``label_mask=True`` means that category is included in the loss. Row metadata
    stays in ``rows`` in prediction order, separate from model tensor inputs.
    """

    categories = tuple(PRIORITY_CATEGORIES)

    def __init__(
        self, rows: pd.DataFrame, tokenizer: Tokenizer, *, max_length: int = 512
    ):
        required = {"contract", "chunk_id", "text", "labels", "label_mask"}
        missing = required - set(rows.columns)
        if missing:
            raise ValueError(f"Missing dataset columns: {sorted(missing)}")
        if type(max_length) is not int or not 3 <= max_length <= 512:
            raise ValueError("max_length must be an integer between 3 and 512")
        if rows.chunk_id.isna().any() or rows.chunk_id.duplicated().any():
            raise ValueError("chunk_id must be non-null and unique")
        if not rows.contract.map(lambda x: isinstance(x, str) and bool(x)).all():
            raise ValueError("contract must contain nonempty strings")
        if not rows.text.map(lambda x: isinstance(x, str) and bool(x.strip())).all():
            raise ValueError("text must contain nonempty strings")

        arrays = []
        for column in ("labels", "label_mask"):
            try:
                values = np.asarray(rows[column].tolist())
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{column} must contain length-10 binary vectors") from exc
            if len(rows) == 0:
                values = values.reshape(0, len(self.categories))
            if values.shape != (len(rows), len(self.categories)):
                raise ValueError(f"{column} must contain length-10 binary vectors")
            if not np.isin(values, [0, 1]).all():
                raise ValueError(f"{column} must contain binary values")
            arrays.append(values)
        labels, masks = arrays
        if ((labels == 1) & (masks == 0)).any():
            raise ValueError("Positive labels cannot be masked")

        # Own the tokenizer settings; never truncate or inherit caller padding.
        self.tokenizer = Tokenizer.from_str(tokenizer.to_str())
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        self.pad_id = self.tokenizer.token_to_id("[PAD]")
        if self.pad_id is None:
            raise ValueError("Tokenizer must define [PAD]")
        if self.tokenizer.num_special_tokens_to_add(False) != 2:
            raise ValueError("Expected BERT tokenizer with two special tokens")
        self.rows = rows.copy(deep=True).reset_index(drop=True)
        self.labels = torch.tensor(labels.astype(np.float32))
        self.label_mask = torch.tensor(masks.astype(bool))
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        ids = self.tokenizer.encode(row.text, add_special_tokens=True).ids
        if len(ids) > self.max_length:
            raise ValueError(
                f"Chunk {row.chunk_id} has {len(ids)} tokens including special "
                f"tokens; max_length={self.max_length}. Re-chunk instead of truncating."
            )
        padding = self.max_length - len(ids)
        return {
            "input_ids": torch.tensor(ids + [self.pad_id] * padding, dtype=torch.long),
            "attention_mask": torch.tensor(
                [1] * len(ids) + [0] * padding, dtype=torch.long
            ),
            "labels": self.labels[index].clone(),
            "label_mask": self.label_mask[index].clone(),
        }


def split_by_fold(
    rows: pd.DataFrame, folds: pd.DataFrame, *, validation_fold: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return training and validation rows using saved contract assignments.

    Accepts a subset of training contracts (e.g. a smoke run), but both resulting
    partitions must contain rows. Official test rows must never enter this split.
    """
    if not {"contract", "fold"}.issubset(folds.columns):
        raise ValueError("folds must contain contract and fold")
    if folds.contract.isna().any() or folds.contract.duplicated().any():
        raise ValueError("folds must have one non-null row per contract")
    numbers = pd.to_numeric(folds.fold, errors="coerce")
    if numbers.isna().any() or not np.isfinite(numbers).all():
        raise ValueError("Fold numbers must be nonnegative integers")
    if ((numbers < 0) | (numbers != np.floor(numbers))).any():
        raise ValueError("Fold numbers must be nonnegative integers")
    if type(validation_fold) is not int or validation_fold not in set(numbers):
        raise ValueError("validation_fold must be a saved fold number")
    if "split" not in rows or not rows.split.eq("train").all():
        raise ValueError("Only rows marked split='train' can use training folds")
    assigned = fold_of_rows(rows, folds)
    train = rows.loc[assigned != validation_fold].copy().reset_index(drop=True)
    validation = rows.loc[assigned == validation_fold].copy().reset_index(drop=True)
    if train.empty or validation.empty:
        raise ValueError("Both training and validation partitions must be nonempty")
    return train, validation
