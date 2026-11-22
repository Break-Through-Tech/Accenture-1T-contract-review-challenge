"""Task #18: turn chunk-level model scores into a per-contract clause register.

Input: one row per chunk with its probabilities for the 10 priority categories
(Task #13's official test export, or `score_chunks` on new contracts).

Steps:
1. Decide: chunk is positive for a category if probability >= the frozen CV threshold.
2. Merge: positive chunks of the same contract + category whose character ranges
   overlap or touch become ONE clause, so a clause sitting in the 128-token overlap
   between two windows is not counted twice.
3. Evidence: keep the merged window (text + offsets) and, where the category's
   keyword rules (Task #8) fire inside it, a shorter excerpt around the matches.
4. Confidence: highest chunk probability in the clause; review flag from the Task #16
   rubric (score below t + 0.10 * (1 - t) goes to human review).

Offsets are character offsets into the original contract text, so evidence can be
checked against the source and against CUAD's gold spans.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import re

import numpy as np
import pandas as pd

from src.cuad_data import PRIORITY_CATEGORIES
from src.keyword_baseline import RULES

CATEGORIES = list(PRIORITY_CATEGORIES)
REVIEW_MARGIN = 0.10  # rubric section 4
EXCERPT_MAX_CHARS = 1500  # cap on the keyword-located excerpt
EXCERPT_MIN_CHARS = 400  # headings alone are not evidence; grow into the clause body

REGISTER_COLUMNS = [
    "contract", "category", "clause_id", "start", "end", "evidence_text",
    "excerpt_start", "excerpt_end", "excerpt_text", "excerpt_rules",
    "confidence", "threshold", "review_boundary", "needs_review", "review_reason",
    "n_chunks", "chunk_ids", "source",
]

# sentence-ish breaks: blank line, or end punctuation followed by a capital/number/bracket
_BREAK = re.compile(r"\n\s*\n|(?<=[.;:])\s+(?=[A-Z0-9(\"“])")


# ------------------------------------------------------------------ inputs


def document_key(contract: str, text: str, split: str) -> str:
    """Same key src.chunking puts at the front of every chunk_id."""
    payload = json.dumps([split, contract, text], ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parse_chunk_ids(predictions: pd.DataFrame) -> pd.DataFrame:
    """Add start/end parsed from chunk_id ('<key>:<start>:<end>')."""
    parts = predictions["chunk_id"].str.rsplit(":", n=2, expand=True)
    if parts.shape[1] != 3 or parts.isna().any().any():
        raise ValueError("chunk_id must look like '<key>:<start>:<end>'")
    out = predictions.copy()
    out["key"] = parts[0]
    out["start"] = parts[1].astype(int)
    out["end"] = parts[2].astype(int)
    return out


def check_against_source(chunks: pd.DataFrame, contracts: pd.DataFrame, split: str) -> None:
    """Fail loudly if a chunk does not belong to the contract text we have."""
    text_of = dict(zip(contracts.contract, contracts.text))
    if missing := set(chunks.contract) - set(text_of):
        raise ValueError(f"{len(missing)} contracts in the scores are not in the source")
    keys = {c: document_key(c, text_of[c], split) for c in chunks.contract.unique()}
    if (chunks.key != chunks.contract.map(keys)).any():
        raise ValueError("chunk_id keys do not match the contract text (different data?)")
    lengths = chunks.contract.map({c: len(t) for c, t in text_of.items()})
    if ((chunks.start < 0) | (chunks.end > lengths) | (chunks.start >= chunks.end)).any():
        raise ValueError("Chunk offsets fall outside the contract text")


def review_boundary(threshold: float) -> float:
    return round(threshold + REVIEW_MARGIN * (1 - threshold), 6)


# ------------------------------------------------------------------ merging


def merge_windows(starts, ends) -> list[tuple[int, int, list[int]]]:
    """Merge overlapping/touching [start, end) ranges; returns (start, end, member idx)."""
    order = np.argsort(np.asarray(starts), kind="stable")
    merged: list[list] = []
    for i in order:
        s, e = int(starts[i]), int(ends[i])
        if merged and s <= merged[-1][1]:  # overlaps or touches previous
            merged[-1][1] = max(merged[-1][1], e)
            merged[-1][2].append(int(i))
        else:
            merged.append([s, e, [int(i)]])
    return [(s, e, idx) for s, e, idx in merged]


def _passages(text: str) -> list[tuple[int, int]]:
    """Split text into sentence-ish passages; offsets relative to text."""
    cuts = [0] + [m.end() for m in _BREAK.finditer(text)] + [len(text)]
    out = []
    for a, b in itertools.pairwise(cuts):
        seg = text[a:b]
        lead = len(seg) - len(seg.lstrip())
        trail = len(seg.rstrip())
        if trail > lead:
            out.append((a + lead, a + trail))
    return out


def locate_excerpt(text: str, category: str, rules=RULES) -> tuple[int, int, list[str]] | None:
    """Passage(s) inside ``text`` where the category's keyword rules fire.

    Takes the passage with the most rule matches (ties: earliest) and grows it with
    neighbouring passages that also match, up to EXCERPT_MAX_CHARS. None if no rule fires.
    """
    cat_rules = [r for r in rules if r.category == category]
    hits = []  # (start, rule name)
    for rule in cat_rules:
        hits += [(m.start(), rule.name) for m in rule.regex.finditer(text)]
    if not hits:
        return None
    passages = _passages(text) or [(0, len(text))]

    def matches(p):
        a, b = p
        return [name for pos, name in hits if a <= pos < b]

    counts = [len(matches(p)) for p in passages]
    best = int(np.argmax(counts))
    lo = hi = best
    # grow while neighbours also contain matches and we stay under the cap
    while True:
        grew = False
        for j in (lo - 1, hi + 1):
            if 0 <= j < len(passages) and counts[j] > 0:
                a = passages[min(lo, j)][0]
                b = passages[max(hi, j)][1]
                if b - a <= EXCERPT_MAX_CHARS:
                    lo, hi = min(lo, j), max(hi, j)
                    grew = True
        if not grew:
            break
    # too short (often just a heading like "4.2 Non-Compete."): extend forward into the
    # clause body first, then backward, until EXCERPT_MIN_CHARS
    while passages[hi][1] - passages[lo][0] < EXCERPT_MIN_CHARS:
        if hi + 1 < len(passages) and passages[hi + 1][1] - passages[lo][0] <= EXCERPT_MAX_CHARS:
            hi += 1
        elif lo > 0 and passages[hi][1] - passages[lo - 1][0] <= EXCERPT_MAX_CHARS:
            lo -= 1
        else:
            break
    a, b = passages[lo][0], passages[hi][1]
    if b - a > EXCERPT_MAX_CHARS:  # one huge passage: centre on the first hit inside it
        first = min(pos for pos, _ in hits if passages[best][0] <= pos < passages[best][1])
        a = max(passages[best][0], first - EXCERPT_MAX_CHARS // 3)
        b = min(passages[best][1], a + EXCERPT_MAX_CHARS)
    names = sorted({name for pos, name in hits if a <= pos < b})
    return a, b, names


# ------------------------------------------------------------------ register


def build_register(
    predictions: pd.DataFrame,
    contracts: pd.DataFrame,
    thresholds: dict[str, float],
    *,
    split: str = "test",
    source: str = "model",
) -> pd.DataFrame:
    """One row per detected clause (contract x category x merged window)."""
    if set(thresholds) != set(CATEGORIES):
        raise ValueError("Need exactly one threshold per priority category")
    for c, t in thresholds.items():
        if not (np.isfinite(t) and 0 <= t <= 1):
            raise ValueError(f"Threshold for {c!r} must be in [0, 1]")
    if missing := {"contract", "chunk_id", *CATEGORIES} - set(predictions.columns):
        raise ValueError(f"Predictions missing columns: {sorted(missing)}")
    if predictions.chunk_id.duplicated().any():
        raise ValueError("Duplicate chunk_id in predictions")
    chunks = parse_chunk_ids(predictions)
    check_against_source(chunks, contracts, split)
    text_of = dict(zip(contracts.contract, contracts.text))

    rows = []
    for category in CATEGORIES:
        t = float(thresholds[category])
        boundary = review_boundary(t)
        prob = chunks[category].to_numpy(dtype=float)
        if not np.isfinite(prob).all():
            raise ValueError(f"Non-finite probabilities for {category!r}")
        positive = chunks[prob >= t]
        for contract, group in positive.groupby("contract", sort=True):
            text = text_of[contract]
            for k, (s, e, idx) in enumerate(merge_windows(group.start.to_numpy(),
                                                          group.end.to_numpy())):
                members = group.iloc[idx]
                confidence = round(float(members[category].max()), 6)
                window = text[s:e]
                found = locate_excerpt(window, category)
                if found:
                    a, b, names = found
                    ex_start, ex_end, ex_text = s + a, s + b, window[a:b]
                else:
                    ex_start = ex_end = None
                    ex_text, names = "", []
                reasons = []
                if confidence < boundary:
                    reasons.append("low confidence")
                if not found:
                    reasons.append("no keyword excerpt; read full window")
                rows.append({
                    "contract": contract, "category": category,
                    "clause_id": f"{contract}|{category}|{k}",
                    "start": int(s), "end": int(e), "evidence_text": window,
                    "excerpt_start": ex_start, "excerpt_end": ex_end,
                    "excerpt_text": ex_text, "excerpt_rules": ";".join(names),
                    "confidence": confidence, "threshold": t,
                    "review_boundary": boundary,
                    "needs_review": confidence < boundary,  # rubric: confidence check only
                    "review_reason": "; ".join(reasons),
                    "n_chunks": len(members),
                    "chunk_ids": ";".join(members.sort_values("start").chunk_id),
                    "source": source,
                })
    register = pd.DataFrame(rows, columns=REGISTER_COLUMNS)
    for col in ("excerpt_start", "excerpt_end"):
        register[col] = register[col].astype("Int64")
    return register.sort_values(["contract", "start", "category"], ignore_index=True)


# ------------------------------------------------------------------ checks vs gold


def _overlaps(a_start, a_end, spans: pd.DataFrame) -> pd.Series:
    return (spans.start < a_end) & (spans.end > a_start)


def match_gold(register: pd.DataFrame, gold: pd.DataFrame, chunk_positives: pd.Series):
    """Per-category comparison against CUAD gold spans (overlap on characters).

    - clause precision: share of register clauses overlapping a gold span of that category
    - span recall: share of gold spans overlapped by a register clause of that category
    - excerpt hit rate: share of supported clauses whose excerpt also overlaps a gold span
    - dedup: positive chunks vs merged clauses
    """
    gold = gold[gold.category.isin(CATEGORIES)]
    g_by = {k: v for k, v in gold.groupby(["contract", "category"])}
    r_by = {k: v for k, v in register.groupby(["contract", "category"])}
    rows = []
    for category in CATEGORIES:
        reg = register[register.category == category]
        sup = exc_hit = exc_n = 0
        for r in reg.itertuples(index=False):
            g = g_by.get((r.contract, category))
            hit = g is not None and _overlaps(r.start, r.end, g).any()
            sup += hit
            if hit and pd.notna(r.excerpt_start):
                exc_n += 1
                exc_hit += _overlaps(r.excerpt_start, r.excerpt_end, g).any()
        spans = gold[gold.category == category]
        found = 0
        for s in spans.itertuples(index=False):
            rr = r_by.get((s.contract, category))
            found += rr is not None and bool(((rr.start < s.end) & (rr.end > s.start)).any())
        with_ex = reg.excerpt_start.notna()
        rows.append({
            "category": category,
            "positive_chunks": int(chunk_positives.get(category, 0)),
            "clauses": len(reg),
            "contracts_with_clause": reg.contract.nunique(),
            "gold_spans": len(spans),
            "clause_precision": sup / len(reg) if len(reg) else np.nan,
            "span_recall": found / len(spans) if len(spans) else np.nan,
            "with_excerpt": float(with_ex.mean()) if len(reg) else np.nan,
            "excerpts_checked": exc_n,
            "excerpt_hits_gold": exc_hit / exc_n if exc_n else np.nan,
            "median_window_chars": float(reg.evidence_text.str.len().median()) if len(reg) else np.nan,
            "median_excerpt_chars": float(reg.excerpt_text[with_ex].str.len().median()) if with_ex.any() else np.nan,
            "needs_review": int(reg.needs_review.sum()),
        })
    table = pd.DataFrame(rows)
    tot = {
        "category": "all",
        "positive_chunks": int(table.positive_chunks.sum()),
        "clauses": int(table.clauses.sum()),
        "contracts_with_clause": register.contract.nunique(),
        "gold_spans": int(table.gold_spans.sum()),
        "clause_precision": float((table.clause_precision * table.clauses).sum() / table.clauses.sum()),
        "span_recall": float((table.span_recall * table.gold_spans).sum() / table.gold_spans.sum()),
        "with_excerpt": float(register.excerpt_start.notna().mean()),
        "excerpts_checked": int(table.excerpts_checked.sum()),
        "excerpt_hits_gold": float((table.excerpt_hits_gold.fillna(0) * table.excerpts_checked).sum()
                                   / max(table.excerpts_checked.sum(), 1)),
        "median_window_chars": float(register.evidence_text.str.len().median()),
        "median_excerpt_chars": float(register.excerpt_text[register.excerpt_start.notna()].str.len().median()),
        "needs_review": int(register.needs_review.sum()),
    }
    return pd.concat([table, pd.DataFrame([tot])], ignore_index=True)


# ------------------------------------------------------------------ new contracts


def score_chunks(chunks: pd.DataFrame, checkpoint_dir, *, batch_size: int = 16, device=None):
    """Run the saved Task #13 model on unlabeled chunks; returns the predictions table.

    For contracts outside the test split (Task #19's one-command pipeline). The
    test split itself uses Task #13's official export instead of re-scoring.
    """
    from pathlib import Path

    from tokenizers import Tokenizer

    from src.final_training import load_checkpoint, predict  # torch only when needed

    checkpoint_dir = Path(checkpoint_dir)
    model = load_checkpoint(checkpoint_dir)
    tokenizer = Tokenizer.from_file(str(checkpoint_dir / "tokenizer.json"))
    rows = chunks[["contract", "chunk_id", "text"]].copy()
    k = len(CATEGORIES)
    rows["labels"] = [[0] * k] * len(rows)  # unlabeled: placeholders, ignored here
    rows["label_mask"] = [[True] * k] * len(rows)
    _, proba = predict(model, rows, tokenizer, batch_size=batch_size, device=device)
    out = rows[["contract", "chunk_id"]].copy()
    for i, c in enumerate(CATEGORIES):
        out[c] = proba[:, i].round(6)
    return out
