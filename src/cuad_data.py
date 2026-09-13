"""
CUAD data loading for the Accenture contract review challenge.

Stage 1 output. Import this instead of re-parsing the JSON:

    from src.cuad_data import load_split

    train = load_split("train")
    train.spans        # one row per annotated clause span
    train.contracts    # one row per contract, text byte-identical to source
    train.grid         # every contract x category pair, INCLUDING the zeros

Design decision: contract text is kept byte-for-byte identical to the source JSON.
Every CUAD annotation is a character offset into that string, so any clean that
changes its length silently corrupts all 13,823 annotations. Cleaning is applied
to extracted span text only, in the `text_clean` column, with the original kept
in `text`.
"""

from __future__ import annotations

import itertools
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# repo_root/src/cuad_data.py -> repo_root/data
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INTERIM_DIR = DATA_DIR / "interim"

RAW_FILES = {
    "train": "train_separate_questions.json",
    "test": "test.json",
    "all": "CUADv1.json",
}

# Advisor-designated high-value categories, mapped onto CUAD's actual strings.
PRIORITY_CATEGORIES = [
    "Cap On Liability",
    "Uncapped Liability",
    "Governing Law",
    "Anti-Assignment",
    "Change Of Control",
    "Termination For Convenience",
    "Exclusivity",
    "Non-Compete",
    "Renewal Term",
    "Audit Rights",
]

REDACTION = re.compile(r"\[\s*\*+\s*\]")


@dataclass
class Split:
    """The three tables for one split."""
    name: str
    contracts: pd.DataFrame
    spans: pd.DataFrame
    grid: pd.DataFrame

    def __repr__(self):
        return (
            f"Split({self.name!r}: {len(self.contracts)} contracts, "
            f"{len(self.spans)} spans, {self.grid['category'].nunique()} categories)"
        )

    def priority(self) -> pd.DataFrame:
        """Spans limited to the 10 advisor-priority categories."""
        return self.spans[self.spans["category"].isin(PRIORITY_CATEGORIES)]


# ------------------------------------------------------------------ parsing

CATEGORY_SUFFIX = re.compile(r"_\d+$")


def category_from_id(qa_id: str) -> str:
    """
    CUAD ids are '{title}__{Category}'.

    train_separate_questions.json appends a numbered answer-slot suffix
    ('Parties_0' ... 'Parties_54'), which test.json does not. Strip it so
    both splits yield the same 41 category names.
    """
    cat = qa_id.split("__")[-1].strip()
    return CATEGORY_SUFFIX.sub("", cat)


def _iter_docs(raw):
    """
    Yield (title, context, qas) per contract.

    train_separate_questions.json repeats each contract's questions, so a title
    can appear with more than 41 qas. We merge on title and dedupe by qa id,
    which normalizes it back to the same 41 categories as test.json.
    """
    merged = {}
    for entry in raw["data"]:
        title = entry["title"].strip()
        for para in entry["paragraphs"]:
            ctx = para["context"]
            if title not in merged:
                merged[title] = {"context": ctx, "qas": {}}
            for qa in para["qas"]:
                merged[title]["qas"].setdefault(qa["id"], qa)

    for title, d in merged.items():
        yield title, d["context"], list(d["qas"].values())


def build_contracts(raw) -> pd.DataFrame:
    rows = [
        {"contract": title, "text": ctx, "n_chars": len(ctx)}
        for title, ctx, _ in _iter_docs(raw)
    ]
    return pd.DataFrame(rows)


def build_spans(raw) -> pd.DataFrame:
    rows = []
    for title, ctx, qas in _iter_docs(raw):
        for qa in qas:
            category = category_from_id(qa["id"])
            for a in qa["answers"]:
                start = a["answer_start"]
                text = a["text"]
                rows.append({
                    "contract": title,
                    "category": category,
                    "text": text,
                    "start": start,
                    "end": start + len(text),
                    "n_chars_span": len(text),
                    "rel_position": start / len(ctx),
                })
    spans = pd.DataFrame(rows)
    return spans.drop_duplicates(
        subset=["contract", "category", "start", "end"]
    ).reset_index(drop=True)


def build_grid(contracts: pd.DataFrame, spans: pd.DataFrame) -> pd.DataFrame:
    """
    Every contract x category pair, including the zeros.

    spans cannot answer 'which contracts are MISSING a liability cap' because
    absence is not a row. This table makes absence explicit, which the class
    imbalance analysis and the missing-clause risk signal both need.
    """
    titles = contracts["contract"].tolist()
    categories = sorted(spans["category"].unique())
    grid = pd.DataFrame(
        itertools.product(titles, categories), columns=["contract", "category"]
    )
    counts = spans.groupby(["contract", "category"]).size().rename("n_spans")
    grid = grid.merge(counts, on=["contract", "category"], how="left")
    grid["n_spans"] = grid["n_spans"].fillna(0).astype(int)
    grid["has_clause"] = grid["n_spans"] > 0
    return grid


# ------------------------------------------------------------------ cleaning

def clean_span_text(s: str) -> str:
    """
    Normalize extracted span text. Never applied to contract text.

    Deliberately does NOT lowercase or strip punctuation: legal phrasing carries
    signal ('IN NO EVENT SHALL', defined terms in quotes) and the brief says to
    preserve legal terminology. This only repairs SEC filing formatting damage.
    """
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def add_quality_flags(spans: pd.DataFrame) -> pd.DataFrame:
    spans = spans.copy()
    spans["text_clean"] = spans["text"].map(clean_span_text)
    spans["has_redaction"] = spans["text"].str.contains(REDACTION, regex=True)
    spans["is_very_short"] = spans["n_chars_span"] < 15
    spans["whitespace_ratio"] = (
        spans["text"].str.count(r"\s") / spans["n_chars_span"].clip(lower=1)
    ).round(3)
    return spans


# ------------------------------------------------------------------ checks

def verify_offsets(contracts: pd.DataFrame, spans: pd.DataFrame) -> list:
    """context[start:end] must reproduce the recorded span text, for every span."""
    lookup = dict(zip(contracts["contract"], contracts["text"]))
    return [
        (r.contract, r.category, r.start)
        for r in spans.itertuples()
        if lookup[r.contract][r.start:r.end] != r.text
    ]


# ------------------------------------------------------------------ pipeline

def build_split(name: str, raw_dir: Path | None = None) -> Split:
    """Parse one raw JSON file into the three tables."""
    raw_dir = Path(raw_dir) if raw_dir else DATA_DIR
    with open(raw_dir / RAW_FILES[name], encoding="utf-8") as f:
        raw = json.load(f)

    contracts = build_contracts(raw)
    spans = add_quality_flags(build_spans(raw))
    grid = build_grid(contracts, spans)

    bad = verify_offsets(contracts, spans)
    if bad:
        raise ValueError(f"{name}: {len(bad)} offset mismatches, first: {bad[:3]}")

    return Split(name, contracts, spans, grid)


def write_split(split: Split, out_dir: Path | None = None) -> Path:
    out_dir = Path(out_dir) if out_dir else INTERIM_DIR / split.name
    out_dir.mkdir(parents=True, exist_ok=True)
    split.contracts.to_parquet(out_dir / "contracts.parquet", index=False)
    split.spans.to_parquet(out_dir / "spans.parquet", index=False)
    split.grid.to_parquet(out_dir / "grid.parquet", index=False)
    return out_dir


def load_split(name: str, rebuild: bool = False) -> Split:
    """
    Load a split from parquet, building it from raw JSON if needed.

        train = load_split("train")
        test  = load_split("test")
    """
    if name not in RAW_FILES:
        raise KeyError(f"unknown split {name!r}, expected one of {list(RAW_FILES)}")

    d = INTERIM_DIR / name
    if rebuild or not (d / "spans.parquet").exists():
        split = build_split(name)
        write_split(split)
        return split

    return Split(
        name,
        pd.read_parquet(d / "contracts.parquet"),
        pd.read_parquet(d / "spans.parquet"),
        pd.read_parquet(d / "grid.parquet"),
    )


def load_all(rebuild: bool = False) -> dict:
    return {n: load_split(n, rebuild) for n in ("train", "test")}


def load_category_descriptions(raw_dir: Path | None = None) -> pd.DataFrame:
    """Load category_descriptions.csv and add a `category` column matching the JSON ids."""
    raw_dir = Path(raw_dir) if raw_dir else DATA_DIR
    df = pd.read_csv(raw_dir / "category_descriptions.csv")
    df.columns = ["category_raw", "description", "answer_format", "group"]
    for c in df.columns:
        df[c] = df[c].astype(str).str.replace(
            r"^(Category|Description|Answer Format|Group):\s*", "", regex=True
        ).str.strip()
    df["category"] = df["category_raw"].str.title()
    return df[["category", "category_raw", "description", "answer_format", "group"]]


def match_csv_to_json(cat_desc: pd.DataFrame, categories) -> pd.DataFrame:
    """Case-insensitive match of CSV category names onto the JSON ones."""
    json_lookup = {c.lower(): c for c in categories}
    out = cat_desc.copy()
    out["json_category"] = out["category_raw"].str.lower().map(json_lookup)
    return out


def check_leakage(train: Split, test: Split, n_chars: int = 3000) -> pd.DataFrame:
    """
    Contracts whose opening text is identical across train and test.

    We cannot re-split (the brief forbids it), but near-duplicates straddling the
    boundary inflate test scores and must be reported as a limitation.
    """
    tr = train.contracts.assign(split="train", sig=train.contracts["text"].str[:n_chars])
    te = test.contracts.assign(split="test", sig=test.contracts["text"].str[:n_chars])
    both = pd.concat([tr, te])
    dupes = both[both.duplicated("sig", keep=False)]
    return dupes[["contract", "split", "n_chars"]].sort_values("contract")


if __name__ == "__main__":
    for name in ("train", "test"):
        s = build_split(name)
        out = write_split(s)
        print(s, "->", out)
