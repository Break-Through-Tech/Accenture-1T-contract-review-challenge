"""BERT encoder and independent classification logits for CUAD categories."""

from __future__ import annotations

from torch import Tensor, nn
from transformers import BertConfig, BertModel, BertPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

from src.chunking import TOKENIZER_MODEL, TOKENIZER_REVISION
from src.cuad_data import PRIORITY_CATEGORIES
from src.loss import masked_bce_with_logits


def configure_labels(config: BertConfig) -> BertConfig:
    """Set the project's fixed output order on a new training configuration."""
    config.id2label = dict(enumerate(PRIORITY_CATEGORIES))
    config.label2id = {label: i for i, label in enumerate(PRIORITY_CATEGORIES)}
    config.problem_type = "multi_label_classification"
    return config


class ContractClassifier(BertPreTrainedModel):
    """Fine-tune BERT with ten independent outputs and optional masked BCE.

    Use ``load_pretrained_model`` to start from the pinned base BERT weights.
    Standard ``save_pretrained`` / ``from_pretrained`` persist trained models.
    Labels and masks must be supplied together; otherwise only logits return.
    """

    def __init__(self, config: BertConfig):
        super().__init__(config)
        expected = dict(enumerate(PRIORITY_CATEGORIES))
        if config.id2label != expected or config.label2id != {
            label: i for i, label in expected.items()
        }:
            raise ValueError("Model configuration must use PRIORITY_CATEGORIES order")
        self.bert = BertModel(config)
        dropout = config.classifier_dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob if dropout is None else dropout)
        self.classifier = nn.Linear(config.hidden_size, len(PRIORITY_CATEGORIES))
        self.post_init()

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        token_type_ids: Tensor | None = None,
        labels: Tensor | None = None,
        label_mask: Tensor | None = None,
    ) -> SequenceClassifierOutput:
        if (labels is None) != (label_mask is None):
            raise ValueError("Supply labels and label_mask together")
        encoded = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        logits = self.classifier(self.dropout(encoded.pooler_output))
        loss = None
        if labels is not None:
            loss = masked_bce_with_logits(logits, labels, label_mask)
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoded.hidden_states,
            attentions=encoded.attentions,
        )


def load_pretrained_model(*, local_files_only: bool = False) -> ContractClassifier:
    """Load the base encoder matching chunk tokenization; initialize a new head.

    Downloads full BERT weights if absent from cache. For an already trained
    checkpoint use ContractClassifier.from_pretrained(checkpoint_path) instead.
    """
    config = BertConfig.from_pretrained(
        TOKENIZER_MODEL, revision=TOKENIZER_REVISION, local_files_only=local_files_only
    )
    return ContractClassifier.from_pretrained(
        TOKENIZER_MODEL,
        revision=TOKENIZER_REVISION,
        config=configure_labels(config),
        local_files_only=local_files_only,
    )
