"""Candidate encoders and tokenizer-length helpers for model selection.

The shipped pipeline is BERT-specific (src.model.ContractClassifier). This module
adds what the model-selection experiment needs without changing that code:
a registry of the three candidates, a way to get each one's tokenizer as a
``tokenizers.Tokenizer`` (what ChunkDataset expects), token-length statistics,
and one classifier head that works for any Hugging Face encoder so the
comparison differs only in the encoder.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from tokenizers import Tokenizer
from torch import Tensor, nn
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput

from src.chunking import TOKENIZER_MODEL, TOKENIZER_REVISION
from src.cuad_data import PRIORITY_CATEGORIES
from src.loss import masked_bce_with_logits

MAX_LENGTH = 512


@dataclass(frozen=True)
class Candidate:
    key: str
    model: str
    revision: str | None  # None: resolved to the current commit SHA at run time


# bert is pinned to the exact revision the chunks were built with. The other two
# resolve to a commit SHA at run time, which is recorded in every result file;
# paste those SHAs here to pin them once a winner is chosen.
CANDIDATES = {
    "bert": Candidate("bert", TOKENIZER_MODEL, TOKENIZER_REVISION),
    "legal-bert": Candidate("legal-bert", "nlpaueb/legal-bert-base-uncased", None),
    "deberta-v3": Candidate("deberta-v3", "microsoft/deberta-v3-base", None),
}


def resolve_revision(candidate: Candidate, *, local_files_only: bool = False) -> str | None:
    """Return the pinned revision, or the repository's current commit SHA."""
    if candidate.revision or local_files_only:
        return candidate.revision
    from huggingface_hub import HfApi

    return HfApi().model_info(candidate.model).sha


def fresh_tokenizer(tokenizer: Tokenizer) -> Tokenizer:
    """Copy a tokenizer with truncation and padding off, so lengths are true."""
    copy = Tokenizer.from_str(tokenizer.to_str())
    copy.no_truncation()
    copy.no_padding()
    return copy


def load_backend_tokenizer(
    candidate: Candidate, revision: str | None, *, local_files_only: bool = False
) -> Tokenizer:
    """Load a candidate's fast tokenizer as a raw ``tokenizers.Tokenizer``.

    deberta-v3 ships only a SentencePiece model, so the fast tokenizer is
    converted on load; that needs the ``sentencepiece`` and ``protobuf`` packages.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        candidate.model, revision=revision, use_fast=True, local_files_only=local_files_only
    )
    if not tokenizer.is_fast:
        raise RuntimeError(f"{candidate.model}: no fast tokenizer available")
    return fresh_tokenizer(tokenizer.backend_tokenizer)


def token_lengths(tokenizer: Tokenizer, texts) -> np.ndarray:
    """Token count per text including special tokens, with no truncation."""
    encodings = fresh_tokenizer(tokenizer).encode_batch(list(texts), add_special_tokens=True)
    return np.fromiter((len(e.ids) for e in encodings), dtype=np.int64, count=len(encodings))


def length_summary(lengths, positives, limit: int = MAX_LENGTH) -> dict:
    """Length statistics plus how much labeled evidence sits in oversized chunks.

    ``positives`` is the number of positive labels in each chunk.
    """
    lengths, positives = np.asarray(lengths), np.asarray(positives)
    if lengths.size == 0 or lengths.shape != positives.shape:
        raise ValueError("lengths and positives must be matching, nonempty 1D arrays")
    over = lengths > limit
    return {
        "chunks": int(lengths.size),
        "total_tokens": int(lengths.sum()),
        "mean": round(float(lengths.mean()), 1),
        "median": float(np.median(lengths)),
        "p95": float(np.percentile(lengths, 95)),
        "p99": float(np.percentile(lengths, 99)),
        "max": int(lengths.max()),
        "over_limit": int(over.sum()),
        "over_limit_pct": round(100 * float(over.mean()), 3),
        "positives_total": int(positives.sum()),
        "positives_in_over_limit": int(positives[over].sum()),
    }


class EncoderClassifier(nn.Module):
    """Any HF encoder + CLS vector + dropout + linear head with ten logits.

    The same head is used for every candidate (final-layer CLS state, no BERT
    pooler) so differences in results come from the encoder, not the head. Same
    call signature and masked loss as ContractClassifier, so src.train.evaluate
    works on it unchanged.
    """

    def __init__(self, encoder: nn.Module, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(encoder.config.hidden_size, len(PRIORITY_CATEGORIES))
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    @classmethod
    def from_pretrained(cls, candidate: Candidate, revision, *, local_files_only=False):
        encoder = AutoModel.from_pretrained(
            candidate.model, revision=revision, local_files_only=local_files_only
        )
        return cls(encoder)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor | None = None,
        label_mask: Tensor | None = None,
    ) -> SequenceClassifierOutput:
        if (labels is None) != (label_mask is None):
            raise ValueError("Supply labels and label_mask together")
        hidden = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        ).last_hidden_state
        logits = self.classifier(self.dropout(hidden[:, 0]))
        loss = None
        if labels is not None:
            loss = masked_bce_with_logits(logits, labels, label_mask)
        return SequenceClassifierOutput(loss=loss, logits=logits)
