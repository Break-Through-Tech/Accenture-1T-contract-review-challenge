"""Export unlabeled chunks for task #6; run with python -m scripts.export_chunks."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from src.chunking import (
    TOKENIZER_MODEL,
    TOKENIZER_REVISION,
    ChunkingConfig,
    chunk_contracts,
    load_tokenizer,
    validate_chunks,
)
from src.cuad_data import DATA_DIR, RAW_FILES, build_split


def export_split(split: str, output_dir: Path, *, tokenizer, raw_dir: Path = DATA_DIR):
    """Build from source, validate, write Parquet, and verify the saved table."""
    source = build_split(split, raw_dir=raw_dir)
    config = ChunkingConfig()
    chunks = chunk_contracts(
        source.contracts, split, tokenizer=tokenizer, config=config
    )
    checks = validate_chunks(
        source.contracts, chunks, split, tokenizer=tokenizer, config=config
    )
    target = output_dir / split
    target.mkdir(parents=True, exist_ok=True)
    chunk_path = target / "chunks.parquet"
    chunks.to_parquet(chunk_path, index=False)
    pd.testing.assert_frame_equal(chunks, pd.read_parquet(chunk_path))
    # Evidence is inspected only after boundaries have been fixed.
    bounds = {
        name: list(zip(group.start, group.end))
        for name, group in chunks.groupby("contract")
    }
    priority = source.priority()
    contained = sum(
        any(a <= span.start and span.end <= b for a, b in bounds.get(span.contract, []))
        for span in priority.itertuples(index=False)
    )
    metadata = {
        "split": split,
        "config": asdict(config),
        "tokenizer": TOKENIZER_MODEL,
        "tokenizer_revision": TOKENIZER_REVISION,
        "tokenizers_version": version("tokenizers"),
        "source_sha256": hashlib.sha256(
            (raw_dir / RAW_FILES[split]).read_bytes()
        ).hexdigest(),
        "chunks_sha256": hashlib.sha256(chunk_path.read_bytes()).hexdigest(),
        "validation": checks,
        "priority_spans": len(priority),
        "fully_contained_priority_spans": contained,
        "labeled": False,
    }
    (target / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument(
        "--output-dir", type=Path, default=DATA_DIR / "interim" / "chunks"
    )
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    tokenizer = load_tokenizer(local_files_only=args.local_files_only)
    result = export_split(args.split, args.output_dir, tokenizer=tokenizer)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
