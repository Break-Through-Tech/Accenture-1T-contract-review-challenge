"""Annotation-independent CUAD chunking with original character offsets."""

from __future__ import annotations

import bisect
import hashlib
import json
import re
from dataclasses import dataclass

import pandas as pd
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

TOKENIZER_MODEL = "google-bert/bert-base-uncased"
TOKENIZER_REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"
CHUNK_DTYPES = {
    "contract": "str",
    "split": "str",
    "chunk_id": "str",
    "start": "int64",
    "end": "int64",
    "text": "str",
    "n_tokens": "int64",
}


@dataclass(frozen=True)
class ChunkingConfig:
    max_tokens: int = 510
    overlap: int = 128

    def __post_init__(self):
        if type(self.max_tokens) is not int or not 1 <= self.max_tokens <= 510:
            raise ValueError("max_tokens must be an integer between 1 and 510")
        if type(self.overlap) is not int or not 0 <= self.overlap < self.max_tokens:
            raise ValueError("overlap must be an integer from 0 to max_tokens - 1")


DEFAULT_CONFIG = ChunkingConfig()


def load_tokenizer(*, local_files_only: bool = False) -> Tokenizer:
    """Load the task #5 reference counter; no model weights are downloaded."""
    path = hf_hub_download(
        TOKENIZER_MODEL,
        "tokenizer.json",
        revision=TOKENIZER_REVISION,
        local_files_only=local_files_only,
    )
    tokenizer = Tokenizer.from_file(path)
    tokenizer.no_truncation()
    tokenizer.no_padding()
    return tokenizer


def _check_inputs(contracts, split, tokenizer):
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    if not {"contract", "text"}.issubset(contracts.columns):
        raise ValueError("contracts must contain contract and text columns")
    for column in ("contract", "text"):
        if not contracts[column].map(lambda value: isinstance(value, str)).all():
            raise ValueError(f"{column} values must be strings")
    if contracts.contract.duplicated().any():
        raise ValueError("contract identifiers must be unique within a split")
    if tokenizer.truncation is not None or tokenizer.padding is not None:
        raise ValueError("Disable tokenizer truncation and padding before chunking")


def _document_key(contract, text, split):
    payload = json.dumps([split, contract, text], ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _windows(text, tokenizer, config):
    if not text.strip():
        return
    offsets = tokenizer.encode(text, add_special_tokens=False).offsets
    starts = [start for start, _ in offsets]
    if not starts:
        raise ValueError("Non-whitespace document produced no tokens")
    boundaries = [
        sorted(
            {bisect.bisect_left(starts, m.end()) for m in re.finditer(pattern, text)}
        )
        for pattern in (r"\n[ \t]*\n+", r"[.!?][\"')\]]*\s+")
    ]
    i, previous_end = 0, 0
    while i < len(starts):
        j = min(i + config.max_tokens, len(starts))
        if j < len(starts):
            minimum = max(
                i + (config.max_tokens + 1) // 2,
                i + config.overlap + 1,
                previous_end + 1,
            )
            for candidates in boundaries:
                k = bisect.bisect_right(candidates, j) - 1
                if k >= 0 and candidates[k] >= minimum:
                    j = candidates[k]
                    break
        start = 0 if i == 0 else starts[i]
        end = len(text) if j == len(starts) else starts[j]
        size = len(tokenizer.encode(text[start:end], add_special_tokens=False).ids)
        # Slicing inside a word can change its WordPiece count.
        while size > config.max_tokens and j > i + 1:
            j -= 1
            while j > i + 1 and starts[j] == starts[j - 1]:
                j -= 1
            end = starts[j]
            size = len(tokenizer.encode(text[start:end], add_special_tokens=False).ids)
        if size > config.max_tokens or size == 0 or j <= previous_end:
            raise ValueError("Cannot advance chunk boundaries within the token budget")
        yield start, end, size
        if j == len(starts):
            break
        previous_end = j
        next_i = max(i + 1, j - config.overlap)
        while next_i > 0 and starts[next_i] == starts[next_i - 1]:
            next_i -= 1
        if next_i <= i:
            raise ValueError(
                "Overlap cannot advance past tokens sharing a character offset"
            )
        i = next_i


def chunk_contracts(
    contracts: pd.DataFrame,
    split: str,
    *,
    tokenizer: Tokenizer,
    config: ChunkingConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Return seven-column chunks, sorted by contract and source position.

    Only contract identifiers and original text are read. IDs depend on split,
    contract, original text and boundaries, not input ordering or annotations.
    Whitespace-only documents produce no rows; short final chunks are retained.
    """
    _check_inputs(contracts, split, tokenizer)
    rows = []
    for contract in contracts.sort_values("contract").itertuples(index=False):
        key = _document_key(contract.contract, contract.text, split)
        for start, end, size in _windows(contract.text, tokenizer, config):
            rows.append(
                {
                    "contract": contract.contract,
                    "split": split,
                    "chunk_id": f"{key}:{start}:{end}",
                    "start": start,
                    "end": end,
                    "text": contract.text[start:end],
                    "n_tokens": size,
                }
            )
    return pd.DataFrame(rows, columns=list(CHUNK_DTYPES)).astype(CHUNK_DTYPES)


def validate_chunks(
    contracts: pd.DataFrame,
    chunks: pd.DataFrame,
    split: str,
    *,
    tokenizer: Tokenizer,
    config: ChunkingConfig = DEFAULT_CONFIG,
) -> dict:
    """Check the exported interface against source text and actual token counts."""
    _check_inputs(contracts, split, tokenizer)
    if list(chunks.columns) != list(CHUNK_DTYPES):
        raise ValueError("Unexpected chunk columns or column order")
    if chunks.isna().any().any() or chunks.chunk_id.duplicated().any():
        raise ValueError("Chunks contain missing values or duplicate IDs")
    if not chunks.split.eq(split).all():
        raise ValueError("Chunks contain a different split")
    if not chunks.contract.isin(contracts.contract).all():
        raise ValueError("Chunks contain unknown contracts")
    for column in ("start", "end", "n_tokens"):
        if not pd.api.types.is_integer_dtype(chunks[column]):
            raise ValueError(f"{column} must have an integer dtype")
    groups = {name: group for name, group in chunks.groupby("contract", sort=False)}
    for contract in contracts.itertuples(index=False):
        source = contract.text
        key = _document_key(contract.contract, source, split)
        group = groups.get(contract.contract, chunks.iloc[:0])
        covered_end, previous_start = 0, -1
        for chunk in group.itertuples(index=False):
            if not 0 <= chunk.start < chunk.end <= len(source):
                raise ValueError("Invalid character offsets")
            if chunk.text != source[chunk.start : chunk.end]:
                raise ValueError("Chunk text differs from the original source slice")
            if chunk.chunk_id != f"{key}:{chunk.start}:{chunk.end}":
                raise ValueError("Chunk ID does not match its source and boundaries")
            size = len(tokenizer.encode(chunk.text, add_special_tokens=False).ids)
            if size != chunk.n_tokens or not 0 < size <= config.max_tokens:
                raise ValueError("Incorrect token count or token budget exceeded")
            if chunk.start <= previous_start or chunk.end <= covered_end:
                raise ValueError("Chunks must advance through the document")
            if source[covered_end : chunk.start].strip():
                raise ValueError("Uncovered non-whitespace text between chunks")
            covered_end, previous_start = chunk.end, chunk.start
        if source[covered_end:].strip():
            raise ValueError("Uncovered non-whitespace document ending")
    return {
        "contracts": len(contracts),
        "chunks": len(chunks),
        "processed_tokens": int(chunks.n_tokens.sum()),
        "max_chunk_tokens": int(chunks.n_tokens.max()) if len(chunks) else 0,
    }
