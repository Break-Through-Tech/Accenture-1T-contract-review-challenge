"""Offline tests: no model or tokenizer downloads."""

import pandas as pd
import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import BertConfig, BertModel

from scripts.model_selection import select_subset
from src.candidates import (
    CANDIDATES,
    EncoderClassifier,
    fresh_tokenizer,
    length_summary,
    token_lengths,
)
from src.cuad_data import PRIORITY_CATEGORIES
from src.loss import masked_bce_with_logits


@pytest.fixture
def tokenizer():
    tok = Tokenizer(models.WordPiece(
        {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "law": 4}, unk_token="[UNK]"
    ))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.post_processor = processors.BertProcessing(("[SEP]", 3), ("[CLS]", 2))
    return tok


def test_registry_has_the_three_candidates_and_pins_bert():
    assert set(CANDIDATES) == {"bert", "legal-bert", "deberta-v3"}
    assert CANDIDATES["bert"].revision  # chunks were built with this exact revision
    assert CANDIDATES["legal-bert"].model == "nlpaueb/legal-bert-base-uncased"
    assert CANDIDATES["deberta-v3"].model == "microsoft/deberta-v3-base"


def test_token_lengths_include_special_tokens_and_ignore_caller_settings(tokenizer):
    tokenizer.enable_truncation(max_length=3)
    tokenizer.enable_padding(length=10)
    lengths = token_lengths(tokenizer, ["law", "law law law law"])
    assert lengths.tolist() == [3, 6]
    assert tokenizer.truncation["max_length"] == 3  # caller's tokenizer untouched
    assert fresh_tokenizer(tokenizer).truncation is None


def test_length_summary_counts_oversized_chunks_and_their_positives():
    lengths = [100, 512, 513, 700]
    positives = [0, 1, 2, 3]
    stats = length_summary(lengths, positives)
    assert stats["over_limit"] == 2
    assert stats["over_limit_pct"] == 50.0
    assert stats["positives_in_over_limit"] == 5
    assert stats["positives_total"] == 6
    assert stats["max"] == 700
    with pytest.raises(ValueError):
        length_summary([], [])
    with pytest.raises(ValueError):
        length_summary([1, 2], [1])


@pytest.fixture
def model():
    config = BertConfig(
        vocab_size=32, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=32, hidden_dropout_prob=0, attention_probs_dropout_prob=0,
    )
    return EncoderClassifier(BertModel(config), dropout=0)


def test_classifier_loss_matches_masked_bce_and_inference_has_no_loss(model):
    ids = torch.tensor([[2, 5, 3, 0], [2, 6, 7, 3]])
    attention = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
    labels = torch.zeros(2, 10)
    mask = torch.tensor([[True] * 9 + [False]] * 2)
    output = model(ids, attention, labels, mask)
    assert output.logits.shape == (2, len(PRIORITY_CATEGORIES))
    torch.testing.assert_close(output.loss, masked_bce_with_logits(output.logits, labels, mask))
    assert model(ids, attention).loss is None
    with pytest.raises(ValueError, match="together"):
        model(ids, attention, labels)
    output.loss.backward()
    assert model.classifier.weight.grad[9].abs().sum() == 0  # masked category


def test_padding_does_not_change_predictions(model):
    model.eval()
    ids = torch.tensor([[2, 5, 3, 0, 0]])
    attention = torch.tensor([[1, 1, 1, 0, 0]])
    other = ids.clone()
    other[0, 3:] = 20
    with torch.no_grad():
        torch.testing.assert_close(model(ids, attention).logits, model(other, attention).logits)


def test_subset_is_whole_contracts_and_independent_of_row_order():
    rows = pd.DataFrame({"contract": list("aabbccdd"), "chunk_id": list("12345678")})
    first = select_subset(rows, 2, seed=1)
    second = select_subset(rows.iloc[::-1], 2, seed=1)
    assert first.contract.nunique() == 2
    assert set(first.contract) == set(second.contract)
    assert select_subset(rows, None, seed=1).equals(rows)
    assert len(select_subset(rows, 10, seed=1)) == len(rows)
