"""Training-only boundary experiment for issue #5, not the production chunker.

Run from the repository root:
    uv run --locked --with tokenizers==0.22.1 python -B scripts/compare_chunking.py

Writes diagnostic tables to data/interim/chunking_strategy/. No model is trained.
"""

from __future__ import annotations

import bisect
import hashlib
import itertools
import json
import math
from collections import Counter, defaultdict
from importlib.metadata import version
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from src.cuad_data import DATA_DIR, PRIORITY_CATEGORIES, build_split

MODEL = "google-bert/bert-base-uncased"
REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"
LIMIT = 510
OUT = DATA_DIR / "interim" / "chunking_strategy"


def natural_breaks(text, starts):
    """Map formatting/sentence breaks to full-document token indices."""
    import re

    paragraphs = sorted(
        {bisect.bisect_left(starts, m.end()) for m in re.finditer(r"\n[ \t]*\n+", text)}
    )
    sentences = sorted(
        {
            bisect.bisect_left(starts, m.end())
            for m in re.finditer(r"[.!?][\"')\]]*\s+", text)
        }
    )
    return paragraphs, sentences


def windows(text, offsets, tokenizer, overlap, hybrid):
    """Choose boundaries from text only; annotations are unavailable here.

    Prefer a paragraph boundary in the latter half of the available window;
    otherwise a heuristic sentence boundary, then the hard token limit.
    Every nonfinal chunk must extend beyond the previous chunk's end.
    """
    starts = [a for a, _ in offsets]
    n = len(starts)
    paragraphs, sentences = natural_breaks(text, starts)
    result = []
    i, previous_end = 0, 0
    while i < n:
        hard_end = min(i + LIMIT, n)
        j, reason = hard_end, "token_limit"
        if hard_end == n:
            reason = "document_end"
        elif hybrid:
            minimum = max(i + math.ceil(LIMIT / 2), i + overlap + 1, previous_end + 1)
            for candidates, label in [
                (paragraphs, "paragraph"),
                (sentences, "sentence"),
            ]:
                k = bisect.bisect_right(candidates, hard_end) - 1
                if k >= 0 and candidates[k] >= minimum:
                    j, reason = candidates[k], label
                    break
        a = 0 if i == 0 else starts[i]
        b = len(text) if j == n else starts[j]
        size = len(tokenizer.encode(text[a:b], add_special_tokens=False).ids)
        # A WordPiece at a sliced word boundary can tokenize differently.
        # Measure the actual substring; never trust only j-i.
        while size > LIMIT and j > i + 1:
            j -= 1
            while j > i + 1 and starts[j] == starts[j - 1]:
                j -= 1
            b = starts[j]
            size = len(tokenizer.encode(text[a:b], add_special_tokens=False).ids)
            reason = "retokenization_limit"
        if size > LIMIT or j <= previous_end:
            raise ValueError("Invalid token budget or non-advancing window")
        result.append((a, b, size, reason))
        if j == n:
            break
        previous_end = j
        i = max(i + 1, j - overlap)
        while i > 0 and starts[i] == starts[i - 1]:
            i -= 1
    return result


def paragraph_chunks(text, tokenizer):
    import re

    bounds = [0] + [m.end() for m in re.finditer(r"\n[ \t]*\n+", text)] + [len(text)]
    return [
        (
            a,
            b,
            len(tokenizer.encode(text[a:b], add_special_tokens=False).ids),
            "paragraph",
        )
        for a, b in itertools.pairwise(bounds)
        if text[a:b].strip()
    ]


def validate_chunks(text, chunks):
    """Check original offsets, complete non-whitespace coverage, and ordering."""
    end = 0
    for a, b, size, _ in chunks:
        assert 0 <= a < b <= len(text) and size > 0
        assert not text[end:a].strip(), "Uncovered non-whitespace text"
        assert b > end, "Chunk does not extend coverage"
        end = b
    assert not text[end:].strip(), "Uncovered document ending"


def main():
    token_path = hf_hub_download(MODEL, "tokenizer.json", revision=REVISION)
    tokenizer = Tokenizer.from_file(token_path)
    tokenizer.no_truncation()
    tokenizer.no_padding()
    # Build afresh rather than relying on possibly stale cached tables.
    train = build_split("train")
    assert len(train.contracts) == 408
    priority = train.priority()
    assert len(priority) == 3241
    by_contract = {
        name: list(group.itertuples()) for name, group in priority.groupby("contract")
    }
    configs = [("paragraph_only", None, False)]
    configs += [
        (f"{mode}_510_{overlap}", overlap, mode == "hybrid")
        for mode in ("fixed", "hybrid")
        for overlap in (0, 64, 128, 256)
    ]
    totals = {name: Counter() for name, _, _ in configs}
    categories = defaultdict(Counter)
    reasons = defaultdict(Counter)
    contract_rows, failures = [], []
    original_tokens = 0
    span_token_sizes = [
        len(tokenizer.encode(s, add_special_tokens=False).ids) for s in priority.text
    ]
    ordered = train.contracts.sort_values(["n_chars", "contract"])
    sample_names = [
        ordered.iloc[round((len(ordered) - 1) * q)].contract for q in (0.1, 0.5, 0.9)
    ]
    sample_rows = []
    for index, contract in enumerate(ordered.itertuples(), 1):
        text = contract.text
        encoding = tokenizer.encode(text, add_special_tokens=False)
        original_tokens += len(encoding.ids)
        spans = by_contract.get(contract.contract, [])
        for name, overlap, hybrid in configs:
            chunks = (
                paragraph_chunks(text, tokenizer)
                if overlap is None
                else windows(text, encoding.offsets, tokenizer, overlap, hybrid)
            )
            validate_chunks(text, chunks)
            if overlap is not None:
                assert max(c[2] for c in chunks) <= LIMIT
            usable = [c for c in chunks if c[2] <= LIMIT]
            t = totals[name]
            t["chunks"] += len(chunks)
            t["over_budget_chunks"] += len(chunks) - len(usable)
            t["processed_tokens"] += sum(c[2] for c in chunks)
            t["max_chunk_tokens"] = max(
                t["max_chunk_tokens"], max(c[2] for c in chunks)
            )
            t["no_priority_overlap_chunks"] += sum(
                not any(a < s.end and s.start < b for s in spans)
                for a, b, _, _ in chunks
            )
            t["partial_only_label_chunks"] += sum(
                any(
                    any(
                        a < s.end and s.start < b and not (a <= s.start and s.end <= b)
                        for s in spans
                        if s.category == cat
                    )
                    and not any(
                        a <= s.start and s.end <= b for s in spans if s.category == cat
                    )
                    for cat in PRIORITY_CATEGORIES
                )
                for a, b, _, _ in chunks
            )
            reasons[name].update(c[3] for c in chunks)
            covered = 0
            for s in spans:
                any_full = any(a <= s.start and s.end <= b for a, b, _, _ in chunks)
                full = any(a <= s.start and s.end <= b for a, b, _, _ in usable)
                covered += full
                t["contained_spans_any_size"] += any_full
                t["contained_spans_within_budget"] += full
                c = categories[(name, s.category)]
                c["spans"] += 1
                c["contained_within_budget"] += full
                if not full:
                    failures.append(
                        {
                            "strategy": name,
                            "contract": s.contract,
                            "category": s.category,
                            "start": s.start,
                            "end": s.end,
                            "text": s.text,
                        }
                    )
            contract_rows.append(
                {
                    "strategy": name,
                    "contract": contract.contract,
                    "chunks": len(chunks),
                    "priority_spans": len(spans),
                    "contained_within_budget": covered,
                }
            )
            if contract.contract in sample_names:
                sample_rows.append(
                    {
                        "strategy": name,
                        "contract": contract.contract,
                        "words": len(text.split()),
                        "tokens": len(encoding.ids),
                        "priority_spans": len(spans),
                        "chunks": len(chunks),
                        "contained_within_budget": covered,
                        "first_chunk": {
                            "start": chunks[0][0],
                            "end": chunks[0][1],
                            "text": text[chunks[0][0] : chunks[0][1]],
                        },
                    }
                )
        if index % 50 == 0:
            print(f"Compared {index}/{len(ordered)} training contracts", flush=True)
    summary = []
    for name, _, _ in configs:
        row = {"strategy": name, **totals[name]}
        row["span_coverage_pct"] = round(
            100 * row["contained_spans_within_budget"] / len(priority), 3
        )
        row["token_work_multiplier"] = round(
            row["processed_tokens"] / original_tokens, 3
        )
        row["no_priority_overlap_pct"] = round(
            100 * row["no_priority_overlap_chunks"] / row["chunks"], 2
        )
        summary.append(row)
    per_category = [
        {
            "strategy": name,
            "category": cat,
            **counts,
            "coverage_pct": round(
                100 * counts["contained_within_budget"] / counts["spans"], 3
            ),
        }
        for (name, cat), counts in categories.items()
    ]
    metadata = {
        "source_commit": "1ce80db",
        "split": "train",
        "contracts": len(ordered),
        "priority_spans": len(priority),
        "categories": PRIORITY_CATEGORIES,
        "tokenizer": MODEL,
        "tokenizer_revision": REVISION,
        "tokenizers_version": version("tokenizers"),
        "tokenizer_sha256": hashlib.sha256(Path(token_path).read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(
            (DATA_DIR / "train_separate_questions.json").read_bytes()
        ).hexdigest(),
        "text_token_limit": LIMIT,
        "original_tokens": original_tokens,
        "priority_spans_above_limit": sum(n > LIMIT for n in span_token_sizes),
        "max_priority_span_tokens": max(span_token_sizes),
        "end_boundary_counts": dict(reasons),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame(per_category).to_csv(OUT / "per_category.csv", index=False)
    pd.DataFrame(contract_rows).to_csv(OUT / "per_contract.csv", index=False)
    for filename, payload in [
        ("metadata.json", metadata),
        ("uncovered_spans.json", failures),
        ("samples.json", sample_rows),
    ]:
        (OUT / filename).write_text(json.dumps(payload, indent=2) + "\n")
    print(pd.DataFrame(summary).to_string(index=False))
    print("Priority spans longer than budget:", metadata["priority_spans_above_limit"])
    print("Results:", OUT)


if __name__ == "__main__":
    main()
