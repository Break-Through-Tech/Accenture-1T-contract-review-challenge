"""Label exported chunks in task #6. Run with:
    uv run --locked python -B -m scripts.label_chunks --split train
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.cuad_data import DATA_DIR, PRIORITY_CATEGORIES, build_split
from src.labeling import label_arrays, label_chunk_table, validate_labels

EXPECTED_TRAIN = {
    "chunks": 14266,
    "priority_spans": 3241,
    "fully_contained_spans": 3229,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def markdown(report: dict) -> str:
    lines = [
        f"# Label validation report ({report['split']})",
        "",
        "| Check | Result |",
        "|---|---:|",
        f"| Chunks | {report['chunks']:,} |",
        f"| Priority annotation rows | {report['priority_spans']:,} |",
        f"| Fully contained annotations | {report['fully_contained_spans']:,} / {report['priority_spans']:,} |",
        f"| Not fully contained | {report['not_fully_contained_spans']:,} |",
        f"| Duplicate annotation rows | {report['duplicate_span_rows']:,} |",
        f"| Evidence rows | {report['evidence_rows']:,} |",
        f"| Chunks with no priority overlap | {report['chunks_without_priority_overlap']:,} ({report['chunks_without_priority_overlap_pct']}%) |",
        f"| Chunks with a partial-only (masked) category | {report['chunks_with_partial_only_category']:,} |",
        f"| Chunks with at least one positive label | {report['chunks_with_any_positive']:,} |",
        f"| Chunks with two or more positive labels | {report['chunks_with_multiple_positive_categories']:,} |",
        "",
        "| Category | Positive chunks | Masked (ambiguous) chunks |",
        "|---|---:|---:|",
    ]
    for category, counts in report["per_category"].items():
        lines.append(
            f"| {category} | {counts['positive_chunks']:,} | {counts['masked_chunks']:,} |"
        )
    lines += ["", "## Annotations not fully contained in any chunk", ""]
    if report["uncovered_spans"]:
        lines += ["| Contract | Category | Start | End |", "|---|---|---:|---:|"]
        lines += [
            f"| {u['contract']} | {u['category']} | {u['start']} | {u['end']} |"
            for u in report["uncovered_spans"]
        ]
    else:
        lines.append("None.")
    if report["reference_differences"]:
        lines += ["", "## Differences from task #5", ""]
        lines += [f"- {d}" for d in report["reference_differences"]]
    return "\n".join(lines) + "\n"


def run(
    split: str, chunk_dir: Path, output_dir: Path, raw_dir: Path = DATA_DIR
) -> dict:
    chunk_path = chunk_dir / split / "chunks.parquet"
    chunks = pd.read_parquet(chunk_path)
    spans = build_split(split, raw_dir=raw_dir).priority()
    labeled, evidence = label_chunk_table(chunks, spans)

    target = output_dir / split
    target.mkdir(parents=True, exist_ok=True)
    labeled_path, evidence_path = (
        target / "labeled_chunks.parquet",
        target / "evidence.parquet",
    )
    labeled.to_parquet(labeled_path, index=False)
    evidence.to_parquet(evidence_path, index=False)

    saved, saved_evidence = (
        pd.read_parquet(labeled_path),
        pd.read_parquet(evidence_path),
    )
    for a, b in zip(label_arrays(labeled), label_arrays(saved)):
        if not np.array_equal(a, b):
            raise ValueError("Labels changed in the Parquet round trip")
    pd.testing.assert_frame_equal(evidence, saved_evidence)
    report = validate_labels(chunks, spans, saved, saved_evidence, PRIORITY_CATEGORIES)

    differences = []
    if split == "train":
        for key, expected in EXPECTED_TRAIN.items():
            if report[key] != expected:
                differences.append(f"{key}: expected {expected:,}, got {report[key]:,}")
    report = {
        "split": split,
        "categories": list(PRIORITY_CATEGORIES),
        "chunks_sha256": sha256(chunk_path),
        "labeled_chunks_sha256": sha256(labeled_path),
        "evidence_sha256": sha256(evidence_path),
        "reference_differences": differences,
        **report,
    }
    (target / "label_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (target / "label_report.md").write_text(markdown(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument(
        "--chunk-dir", type=Path, default=DATA_DIR / "interim" / "chunks"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DATA_DIR / "interim" / "labels"
    )
    args = parser.parse_args()
    report = run(args.split, args.chunk_dir, args.output_dir)
    print(markdown(report))


if __name__ == "__main__":
    main()