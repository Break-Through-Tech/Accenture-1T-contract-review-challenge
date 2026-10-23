import pytest
import torch
from transformers import BertConfig, BertModel

from src.cuad_data import PRIORITY_CATEGORIES
from src.loss import masked_bce_with_logits
from src.model import ContractClassifier, configure_labels


@pytest.fixture
def config():
    return configure_labels(BertConfig(
        vocab_size=32, hidden_size=16, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=32,
        max_position_embeddings=512, hidden_dropout_prob=0,
        attention_probs_dropout_prob=0,
    ))


@pytest.fixture
def batch():
    return {
        "input_ids": torch.tensor([[2, 5, 6, 3, 0], [2, 7, 3, 0, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]]),
        "labels": torch.zeros(2, 10),
        "label_mask": torch.tensor([[True] * 9 + [False]] * 2),
    }


def test_forward_and_masked_loss(config, batch):
    model = ContractClassifier(config)
    output = model(**batch)
    assert output.logits.shape == (2, 10)
    assert torch.isfinite(output.logits).all()
    torch.testing.assert_close(output.loss, masked_bce_with_logits(
        output.logits, batch["labels"], batch["label_mask"]
    ))
    assert model.config.id2label == dict(enumerate(PRIORITY_CATEGORIES))
    inference = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    assert inference.loss is None
    assert inference.logits.sigmoid().shape == (2, 10)


def test_backward_reaches_encoder_and_head_and_masks_category(config, batch):
    model = ContractClassifier(config)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = model(**batch).loss
    before.backward()
    assert model.bert.embeddings.word_embeddings.weight.grad.abs().sum() > 0
    assert model.classifier.weight.grad[:9].abs().sum() > 0
    assert torch.all(model.classifier.weight.grad[9] == 0)
    assert model.classifier.bias.grad[9] == 0
    optimizer.step()
    assert model(**batch).loss < before


def test_padding_does_not_change_predictions(config, batch):
    model = ContractClassifier(config).eval()
    ids = batch["input_ids"].clone()
    ids[batch["attention_mask"] == 0] = 20
    with torch.no_grad():
        original = model(batch["input_ids"], batch["attention_mask"]).logits
        changed = model(ids, batch["attention_mask"]).logits
    torch.testing.assert_close(original, changed)


def test_save_reload_preserves_logits_and_label_order(config, batch, tmp_path):
    model = ContractClassifier(config).eval()
    model.save_pretrained(tmp_path)
    restored = ContractClassifier.from_pretrained(tmp_path, local_files_only=True).eval()
    with torch.no_grad():
        torch.testing.assert_close(model(**batch).logits, restored(**batch).logits)
    assert restored.config.id2label == dict(enumerate(PRIORITY_CATEGORIES))


def test_base_encoder_checkpoint_loads_without_changing_encoder(config, tmp_path):
    base = BertModel(config)
    base.save_pretrained(tmp_path)
    model = ContractClassifier.from_pretrained(tmp_path, config=config, local_files_only=True)
    for name, value in base.state_dict().items():
        torch.testing.assert_close(value, model.bert.state_dict()[name])


@pytest.mark.parametrize("missing", ["labels", "label_mask"])
def test_reject_labels_without_masks(config, batch, missing):
    del batch[missing]
    with pytest.raises(ValueError, match="together"):
        ContractClassifier(config)(**batch)


def test_reject_wrong_category_mapping(config):
    config.id2label[0] = "wrong category"
    with pytest.raises(ValueError, match="order"):
        ContractClassifier(config)


def test_full_sequence_length(config):
    model = ContractClassifier(config).eval()
    with torch.no_grad():
        output = model(torch.ones(1, 512, dtype=torch.long), torch.ones(1, 512))
    assert output.logits.shape == (1, 10)
