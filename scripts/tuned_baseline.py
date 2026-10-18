"""Tune one decision threshold per category and re-score the TF-IDF baseline.

    uv run --locked python -B -m scripts.tuned_baseline

Task #9. Thresholds are chosen on out-of-fold predictions on the training split
and then frozen. The test split is scored once at those thresholds; it is never
used to choose them. Needs the task #6 outputs for both splits.

Writes the fold assignment to folds.csv so that October's transformer can be
cross-validated on exactly these folds instead of a fresh partition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from src.cuad_data import DATA_DIR, PRIORITY_CATEGORIES
from src.evaluation import (
    DEFAULT_THRESHOLD,
    THRESHOLD_GRID,
    contract_folds,
    score,
    threshold_map,
    tune_thresholds,
)
from src.labeling import label_arrays
from src.tfidf_baseline import DEFAULT_CONFIG, evaluate, fit, out_of_fold_proba

REPO = Path(__file__).resolve().parent.parent


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def markdown(table: pd.DataFrame) -> str:
    """Metrics table as Markdown, including the threshold each row used."""
    lines = [
        "| Category | Threshold | Support | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in table.itertuples(index=False):
        thr = "mixed" if pd.isna(r.threshold) else f"{r.threshold:.2f}"
        lines.append(
            f"| {r.category} | {thr} | {r.support:,} | {r.precision:.3f} "
            f"| {r.recall:.3f} | {r.f1:.3f} |"
        )
    return "\n".join(lines)


def load_splits(label_dir: Path) -> dict:
    paths = {s: label_dir / s / "labeled_chunks.parquet" for s in ("train", "test")}
    for split, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run scripts.export_chunks and "
                f"scripts.label_chunks with --split {split} first."
            )
    splits = {s: pd.read_parquet(p) for s, p in paths.items()}
    shared = set(splits["train"].contract) & set(splits["test"].contract)
    if shared:
        raise ValueError(f"{len(shared)} contracts appear in both splits")
    return {"paths": paths, **splits}


def run(label_dir: Path, results_dir: Path, model_dir: Path, folds: int = 5) -> dict:
    loaded = load_splits(label_dir)
    train, test = loaded["train"], loaded["test"]
    categories = list(PRIORITY_CATEGORIES)
    config = DEFAULT_CONFIG

    # Folds first, so the saved table is the one actually used below.
    fold_table = contract_folds(train, n_splits=folds)
    oof, fold_table = out_of_fold_proba(train, categories, config, folds, fold_table)
    labels, mask = label_arrays(train, len(categories))

    cv_default = score(labels, mask, oof, categories, DEFAULT_THRESHOLD)
    tuned = tune_thresholds(labels, mask, oof, categories, THRESHOLD_GRID)
    chosen = threshold_map(tuned)
    cv_tuned = score(labels, mask, oof, categories, chosen)

    # One model on the whole training split, scored twice: at the old shared
    # threshold and at the frozen tuned ones.
    model = fit(train, categories, config)
    test_default, _ = evaluate(model, test)
    test_tuned, _ = evaluate(model, test, thresholds=chosen)

    results_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    fold_table.to_csv(results_dir / "folds.csv", index=False)
    tuned.to_csv(results_dir / "thresholds.csv", index=False)
    cv_default.to_csv(results_dir / "train_cv_default.csv", index=False)
    cv_tuned.to_csv(results_dir / "train_cv_tuned.csv", index=False)
    test_default.to_csv(results_dir / "test_default.csv", index=False)
    test_tuned.to_csv(results_dir / "test_tuned.csv", index=False)

    # Out-of-fold scores are what a later model needs to tune its own
    # thresholds on the same folds, so keep them beside the model.
    oof_table = train[["chunk_id", "contract"]].copy()
    oof_table["fold"] = train["contract"].map(
        dict(zip(fold_table.contract, fold_table.fold))
    )
    for i, category in enumerate(categories):
        oof_table[category] = oof[:, i].round(6)
    oof_table.to_parquet(model_dir / "tfidf_oof_scores.parquet", index=False)

    def macro(table: pd.DataFrame, column: str = "f1") -> float:
        return float(table.loc[table.category == "macro avg", column].iloc[0])

    metadata = {
        "task": 9,
        "config": config.to_dict(),
        "categories": categories,
        "scikit_learn_version": version("scikit-learn"),
        "train_labeled_chunks_sha256": sha256(loaded["paths"]["train"]),
        "test_labeled_chunks_sha256": sha256(loaded["paths"]["test"]),
        "train": {"contracts": int(train.contract.nunique()), "chunks": len(train)},
        "test": {"contracts": int(test.contract.nunique()), "chunks": len(test)},
        "cv_folds": folds,
        "cv_grouped_by": "contract",
        "fold_chunk_counts": fold_table.groupby("fold").chunks.sum().to_dict(),
        "fold_contract_counts": fold_table.fold.value_counts().sort_index().to_dict(),
        "threshold_grid": {
            "start": float(THRESHOLD_GRID[0]),
            "stop": float(THRESHOLD_GRID[-1]),
            "step": 0.01,
            "size": int(THRESHOLD_GRID.size),
        },
        "tuned_on": "out-of-fold training predictions",
        "thresholds": {k: float(v) for k, v in chosen.items()},
        "macro_f1": {
            "train_cv_default": macro(cv_default),
            "train_cv_tuned": macro(cv_tuned),
            "test_default": macro(test_default),
            "test_tuned": macro(test_tuned),
        },
        "micro_f1": {
            "train_cv_default": float(
                cv_default.loc[cv_default.category == "micro avg", "f1"].iloc[0]
            ),
            "train_cv_tuned": float(
                cv_tuned.loc[cv_tuned.category == "micro avg", "f1"].iloc[0]
            ),
            "test_default": float(
                test_default.loc[test_default.category == "micro avg", "f1"].iloc[0]
            ),
            "test_tuned": float(
                test_tuned.loc[test_tuned.category == "micro avg", "f1"].iloc[0]
            ),
        },
    }
    (results_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return {
        "metadata": metadata,
        "folds": fold_table,
        "thresholds": tuned,
        "cv_default": cv_default,
        "cv_tuned": cv_tuned,
        "test_default": test_default,
        "test_tuned": test_tuned,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label-dir", type=Path, default=DATA_DIR / "interim" / "labels"
    )
    parser.add_argument(
        "--results-dir", type=Path, default=REPO / "docs" / "tuned-baseline-results"
    )
    parser.add_argument(
        "--model-dir", type=Path, default=DATA_DIR / "interim" / "models"
    )
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    out = run(args.label_dir, args.results_dir, args.model_dir, args.folds)
    print(json.dumps(out["metadata"], indent=2))
    print("\n## Chosen thresholds (out-of-fold training predictions)\n")
    print(out["thresholds"].to_string(index=False))
    print("\n## Training split, out-of-fold, tuned thresholds\n")
    print(markdown(out["cv_tuned"]))
    print("\n## Official test split, shared 0.5 threshold\n")
    print(markdown(out["test_default"]))
    print("\n## Official test split, tuned thresholds (the bar for October)\n")
    print(markdown(out["test_tuned"]))


if __name__ == "__main__":
    main()
