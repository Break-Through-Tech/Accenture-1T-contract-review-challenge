"""Train and score the TF-IDF baseline for task #7. Run with:
    uv run --locked python -B -m scripts.tfidf_baseline

Needs the task #6 outputs for both splits (see docs/tfidf-baseline.md).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import joblib
import pandas as pd

from src.cuad_data import DATA_DIR, PRIORITY_CATEGORIES
from src.tfidf_baseline import DEFAULT_CONFIG, cross_validate, evaluate, fit

REPO = Path(__file__).resolve().parent.parent


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def markdown(table: pd.DataFrame) -> str:
    lines = [
        "| Category | Support | Precision | Recall | F1 | Avg. precision |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in table.itertuples(index=False):
        ap = "" if pd.isna(r.average_precision) else f"{r.average_precision:.3f}"
        lines.append(
            f"| {r.category} | {r.support:,} | {r.precision:.3f} | {r.recall:.3f} "
            f"| {r.f1:.3f} | {ap} |"
        )
    return "\n".join(lines)


def run(label_dir: Path, results_dir: Path, model_dir: Path, folds: int = 5) -> dict:
    paths = {s: label_dir / s / "labeled_chunks.parquet" for s in ("train", "test")}
    for split, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run scripts.export_chunks and "
                f"scripts.label_chunks with --split {split} first."
            )
    train, test = pd.read_parquet(paths["train"]), pd.read_parquet(paths["test"])
    shared = set(train.contract) & set(test.contract)
    if shared:
        raise ValueError(f"{len(shared)} contracts appear in both splits")

    config = DEFAULT_CONFIG
    cv = cross_validate(train, PRIORITY_CATEGORIES, config, n_splits=folds)
    model = fit(train, PRIORITY_CATEGORIES, config)
    test_metrics, test_scores = evaluate(model, test)

    results_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    cv.to_csv(results_dir / "train_cv_per_category.csv", index=False)
    test_metrics.to_csv(results_dir / "test_per_category.csv", index=False)
    model.top_terms().to_csv(results_dir / "top_terms.csv", index=False)
    joblib.dump(model, model_dir / "tfidf_baseline.joblib")
    predictions = test[["chunk_id", "contract"]].copy()
    for i, category in enumerate(PRIORITY_CATEGORIES):
        predictions[category] = test_scores[:, i].round(6)
    predictions.to_parquet(model_dir / "test_scores.parquet", index=False)

    metadata = {
        "config": config.to_dict(),
        "categories": list(PRIORITY_CATEGORIES),
        "scikit_learn_version": version("scikit-learn"),
        "train_labeled_chunks_sha256": sha256(paths["train"]),
        "test_labeled_chunks_sha256": sha256(paths["test"]),
        "train": {"contracts": int(train.contract.nunique()), "chunks": len(train)},
        "test": {"contracts": int(test.contract.nunique()), "chunks": len(test)},
        "vocabulary_size": len(model.vectorizer.vocabulary_),
        "cv_folds": folds,
        "cv_grouped_by": "contract",
    }
    (results_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return {"metadata": metadata, "cv": cv, "test": test_metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label-dir", type=Path, default=DATA_DIR / "interim" / "labels"
    )
    parser.add_argument(
        "--results-dir", type=Path, default=REPO / "docs" / "tfidf-baseline-results"
    )
    parser.add_argument(
        "--model-dir", type=Path, default=DATA_DIR / "interim" / "models"
    )
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    out = run(args.label_dir, args.results_dir, args.model_dir, args.folds)
    print(json.dumps(out["metadata"], indent=2))
    print("\n## Training split, 5-fold by contract\n")
    print(markdown(out["cv"]))
    print("\n## Official test split\n")
    print(markdown(out["test"]))


if __name__ == "__main__":
    main()
