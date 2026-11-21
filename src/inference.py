"""Task #18: turn chunk probabilities into a clause register.

The final model (task #13) scores fixed windows of about 510 tokens that overlap by
128 tokens. A clause near a window edge lands in two windows, so counting positive
windows would count it twice. Here each run of overlapping positive windows for one
category becomes a single register entry: the merged character span, its text, and
the highest window probability as its confidence.

The model only says which window a clause is in, not where inside it, so evidence
is the whole merged window (usually 1,500 to 4,000 characters).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.cuad_data import PRIORITY_CATEGORIES
from src.dataset import ChunkDataset

REGISTER_COLUMNS = [
    "contract", "category", "start", "end", "confidence", "threshold", "n_chunks",
    "chunk_ids", "evidence_text",
]


def chunk_offsets(chunk_ids: pd.Series) -> pd.DataFrame:
    """Start and end character offsets from IDs shaped ``key:start:end`` (src.chunking)."""
    parts = chunk_ids.str.rsplit(":", n=2, expand=True)
    if parts.shape[1] != 3 or parts.isna().any().any():
        raise ValueError("chunk_id must look like key:start:end")
    offsets = pd.DataFrame({"start": parts[1].astype("int64"), "end": parts[2].astype("int64")})
    if (offsets.start >= offsets.end).any():
        raise ValueError("chunk_id offsets must satisfy start < end")
    return offsets


def predict_chunks(model, chunks: pd.DataFrame, tokenizer, *, batch_size=16, device=None):
    """Probabilities for unlabeled chunks, shape (len(chunks), 10), in row order."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    rows = chunks[["contract", "chunk_id", "text"]].copy()
    # ChunkDataset expects labels; inference has none, and they are never read here
    rows["labels"] = [[0] * len(PRIORITY_CATEGORIES)] * len(rows)
    rows["label_mask"] = [[True] * len(PRIORITY_CATEGORIES)] * len(rows)
    loader = DataLoader(ChunkDataset(rows, tokenizer), batch_size=batch_size, shuffle=False)
    model.to(device).eval()
    out = []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            ).logits
            out.append(logits.sigmoid().float().cpu().numpy())
    if not out:
        return np.zeros((0, len(PRIORITY_CATEGORIES)), dtype=np.float32)
    return np.concatenate(out)


def merge_windows(
    predictions: pd.DataFrame, thresholds: dict, categories=PRIORITY_CATEGORIES
) -> pd.DataFrame:
    """One row per clause: positive windows that overlap are merged into one span.

    ``predictions`` has contract, chunk_id and one probability column per category.
    A window is positive when its probability is at or above that category's threshold.
    """
    missing = [c for c in categories if c not in thresholds]
    if missing:
        raise ValueError(f"No threshold for {missing}")
    table = predictions.reset_index(drop=True)
    table = pd.concat([table, chunk_offsets(table.chunk_id)], axis=1)
    table = table.sort_values(["contract", "start"], kind="stable")
    rows = []
    for category in categories:
        threshold = thresholds[category]
        positive = table.loc[table[category] >= threshold]
        for contract, windows in positive.groupby("contract", sort=True):
            current = None
            for chunk_id, start, end, probability in zip(
                windows.chunk_id, windows.start, windows.end, windows[category]
            ):
                if current is not None and start < current["end"]:
                    current["end"] = max(current["end"], int(end))
                    current["confidence"] = max(current["confidence"], float(probability))
                    current["chunk_ids"].append(chunk_id)
                    continue
                if current is not None:
                    rows.append(current)
                current = {
                    "contract": contract, "category": category,
                    "start": int(start), "end": int(end),
                    "confidence": float(probability), "threshold": threshold,
                    "chunk_ids": [chunk_id],
                }
            rows.append(current)
    register = pd.DataFrame(rows, columns=REGISTER_COLUMNS[:-1])
    register["confidence"] = register.confidence.round(6)
    register["n_chunks"] = register.chunk_ids.map(len)
    register["chunk_ids"] = register.chunk_ids.map(";".join)
    return register.sort_values(["contract", "start", "category"], ignore_index=True)


def add_evidence(register: pd.DataFrame, contracts: pd.DataFrame) -> pd.DataFrame:
    """Attach the source text of each span, trimming whitespace and moving offsets to match."""
    text_of = dict(zip(contracts.contract, contracts.text))
    unknown = set(register.contract) - set(text_of)
    if unknown:
        raise ValueError(f"Register has contracts with no source text: {sorted(unknown)[:3]}")
    register = register.copy()
    starts, ends, evidence = [], [], []
    for row in register.itertuples(index=False):
        source = text_of[row.contract]
        if row.end > len(source):
            raise ValueError(f"{row.contract}: span ends past the contract text")
        piece = source[row.start:row.end]
        lead = len(piece) - len(piece.lstrip())
        trail = len(piece) - len(piece.rstrip())
        starts.append(row.start + lead)
        ends.append(row.end - trail)
        evidence.append(source[row.start + lead:row.end - trail])
    register["start"], register["end"], register["evidence_text"] = starts, ends, evidence
    return register[REGISTER_COLUMNS]


def _overlaps(a_start, a_end, b_start, b_end) -> np.ndarray:
    """(len(a), len(b)) boolean matrix of half-open interval overlap."""
    return (np.asarray(a_start)[:, None] < np.asarray(b_end)[None, :]) & (
        np.asarray(b_start)[None, :] < np.asarray(a_end)[:, None]
    )


def _prf(tp, fp, fn) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def score_register(
    register: pd.DataFrame, spans: pd.DataFrame, contracts, categories=PRIORITY_CATEGORIES
) -> pd.DataFrame:
    """Check the register against CUAD's gold clauses, per category.

    - entry precision: share of register entries that overlap a gold clause of their category
    - clause recall: share of gold clauses that overlap a register entry of their category
    - contract-level precision/recall/F1: does the contract have that clause type at all
      (what the missing-clause risk signal depends on)
    """
    contracts = sorted(set(contracts))
    gold = spans.loc[spans.category.isin(categories) & spans.contract.isin(contracts)]
    rows = []
    for category in categories:
        found = register.loc[register.category == category]
        truth = gold.loc[gold.category == category]
        hit_entries = hit_clauses = 0
        for contract in contracts:
            f = found.loc[found.contract == contract]
            t = truth.loc[truth.contract == contract]
            if len(f) and len(t):
                both = _overlaps(f.start, f.end, t.start, t.end)
                hit_entries += int(both.any(axis=1).sum())
                hit_clauses += int(both.any(axis=0).sum())
        has_found, has_truth = set(found.contract), set(truth.contract)
        contract_level = _prf(
            len(has_found & has_truth), len(has_found - has_truth), len(has_truth - has_found)
        )
        rows.append({
            "category": category,
            "entries": len(found), "gold_clauses": len(truth),
            "entry_precision": round(hit_entries / len(found), 4) if len(found) else 0.0,
            "clause_recall": round(hit_clauses / len(truth), 4) if len(truth) else 0.0,
            "contracts_predicted": len(has_found), "contracts_with_clause": len(has_truth),
            **{f"contract_{k}": v for k, v in contract_level.items()},
            "median_evidence_chars": int(found.end.sub(found.start).median()) if len(found) else 0,
        })
    table = pd.DataFrame(rows)
    macro = {"category": "macro avg"}
    for column in ("entry_precision", "clause_recall", "contract_precision",
                   "contract_recall", "contract_f1"):
        macro[column] = round(float(table[column].mean()), 4)
    for column in ("entries", "gold_clauses", "contracts_predicted", "contracts_with_clause"):
        macro[column] = int(table[column].sum())
    macro["median_evidence_chars"] = (
        int(register.end.sub(register.start).median()) if len(register) else 0
    )
    return pd.concat([table, pd.DataFrame([macro])], ignore_index=True)
