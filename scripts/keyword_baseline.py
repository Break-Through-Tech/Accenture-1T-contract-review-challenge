"""Score the keyword/rule baseline for task #8. Run with:
    uv run --locked python -B -m scripts.keyword_baseline

Needs the task #6 outputs for both splits (see docs/keyword-baseline.md).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from scripts.tfidf_baseline import markdown
from src.cuad_data import DATA_DIR, PRIORITY_CATEGORIES
from src.keyword_baseline import RULES, evaluate, rule_table

REPO = Path(__file__).resolve().parent.parent


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(label_dir: Path, results_dir: Path) -> dict:
    paths = {s: label_dir / s / "labeled_chunks.parquet" for s in ("train", "test")}
    for split, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run scripts.export_chunks and "
                f"scripts.label_chunks with --split {split} first."
            )
    results_dir.mkdir(parents=True, exist_ok=True)
    rule_table().to_csv(results_dir / "rules.csv", index=False)

    out = {}
    for split, path in paths.items():
        labeled = pd.read_parquet(path)
        metrics, stats, _ = evaluate(labeled)
        metrics.to_csv(results_dir / f"{split}_per_category.csv", index=False)
        stats.to_csv(results_dir / f"{split}_rule_stats.csv", index=False)
        out[split] = {
            "metrics": metrics,
            "stats": stats,
            "contracts": int(labeled.contract.nunique()),
            "chunks": len(labeled),
        }

    metadata = {
        "categories": list(PRIORITY_CATEGORIES),
        "rules": len(RULES),
        "decision": "positive when at least one rule of the category matches",
        "rule_development_split": "train",
        "train_labeled_chunks_sha256": sha256(paths["train"]),
        "test_labeled_chunks_sha256": sha256(paths["test"]),
        **{
            s: {"contracts": out[s]["contracts"], "chunks": out[s]["chunks"]}
            for s in out
        },
    }
    (results_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return {"metadata": metadata, **{s: out[s]["metrics"] for s in out}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label-dir", type=Path, default=DATA_DIR / "interim" / "labels"
    )
    parser.add_argument(
        "--results-dir", type=Path, default=REPO / "docs" / "keyword-baseline-results"
    )
    args = parser.parse_args()
    out = run(args.label_dir, args.results_dir)
    print(json.dumps(out["metadata"], indent=2))
    print("\n## Training split (rules were written on this split)\n")
    print(markdown(out["train"]))
    print("\n## Official test split\n")
    print(markdown(out["test"]))


if __name__ == "__main__":
    main()
