"""Shared scoring for every clause-detection model in this project.

Task #7 (TF-IDF), task #8 (keyword rules) and October's transformer have to be
compared on the same numbers, so the metric definitions, the decision thresholds
and the cross-validation folds live here rather than inside any one model.

Three things this module owns:

Thresholds
    A model produces a score per (chunk, category). Turning scores into decisions
    needs a threshold, and one shared threshold across ten categories is a
    compromise: it is too low for a category whose vocabulary is shared with
    another and too high for a category with few positive chunks. Every function
    here takes a threshold *per category* and a plain float still works.

Metrics
    Per-category precision, recall, F1 and average precision, plus macro and
    micro rows. Masked (chunk, category) pairs -- the partial-overlap cases from
    task #6 -- are excluded for that category only; the chunk still counts for
    the other nine.

Folds
    Cross-validation folds grouped by contract, because chunks from one contract
    overlap by 128 tokens and splitting by chunk would leak. The assignment is
    written to a file so that a later model can be scored on exactly these folds.

Tuning happens on out-of-fold training predictions. The test split is scored once,
at thresholds already frozen, and is never used to choose them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold

from src.cuad_data import PRIORITY_CATEGORIES

#: The single threshold used before task #9. Kept as the default so that the
#: untuned numbers in docs/tfidf-baseline.md stay reproducible.
DEFAULT_THRESHOLD = 0.5

#: Candidates tried by :func:`tune_thresholds`. Fixed, so a tuning run is
#: reproducible: 0.05 to 0.95 in steps of 0.01.
THRESHOLD_GRID = np.round(np.arange(0.05, 0.9501, 0.01), 2)

METRIC_COLUMNS = [
    "category",
    "threshold",
    "support",
    "scored_chunks",
    "masked_chunks",
    "predicted_positive",
    "precision",
    "recall",
    "f1",
    "average_precision",
]

TUNING_COLUMNS = [
    "category",
    "threshold",
    "support",
    "f1",
    "precision",
    "recall",
    "default_threshold",
    "default_f1",
    "f1_gain",
]

FOLD_COLUMNS = ["contract", "fold", "chunks"]


def resolve_thresholds(thresholds, categories=PRIORITY_CATEGORIES) -> np.ndarray:
    """Expand a threshold argument to one float per category.

    Accepts a single number (used for every category), a mapping from category
    name to number, or a sequence in ``categories`` order. A mapping must name
    every category: a silently missing category would be scored at a default
    that the caller did not ask for.
    """
    categories = list(categories)
    if isinstance(thresholds, (int, float, np.floating, np.integer)):
        values = np.full(len(categories), float(thresholds), dtype=float)
    elif isinstance(thresholds, pd.Series):
        return resolve_thresholds(thresholds.to_dict(), categories)
    elif isinstance(thresholds, dict):
        unknown = sorted(set(thresholds) - set(categories))
        if unknown:
            raise ValueError(f"thresholds name unknown categories: {unknown}")
        absent = [c for c in categories if c not in thresholds]
        if absent:
            raise ValueError(f"no threshold given for: {absent}")
        values = np.array([float(thresholds[c]) for c in categories], dtype=float)
    else:
        values = np.asarray(thresholds, dtype=float).ravel()
        if values.shape != (len(categories),):
            raise ValueError(
                f"expected {len(categories)} thresholds, got {values.size}"
            )
    if not np.isfinite(values).all():
        raise ValueError("thresholds must be finite")
    if (values < 0).any():
        raise ValueError("thresholds must not be negative")
    return values


def apply_thresholds(
    proba: np.ndarray, thresholds=DEFAULT_THRESHOLD, categories=PRIORITY_CATEGORIES
) -> np.ndarray:
    """Turn an (n_chunks, n_categories) score array into 0/1 predictions.

    A score exactly equal to its threshold counts as positive, matching the
    ``>=`` rule used by the baselines before task #9.
    """
    categories = list(categories)
    proba = np.asarray(proba, dtype=float)
    if proba.ndim != 2 or proba.shape[1] != len(categories):
        raise ValueError(
            f"expected scores with {len(categories)} columns, got {proba.shape}"
        )
    return (proba >= resolve_thresholds(thresholds, categories)).astype(np.int8)


def _check_shapes(labels, mask, proba, categories) -> None:
    shapes = {a.shape for a in (labels, mask, proba)}
    if len(shapes) != 1:
        raise ValueError(f"labels, mask and scores disagree in shape: {shapes}")
    if labels.shape[1] != len(categories):
        raise ValueError(
            f"expected {len(categories)} columns, got {labels.shape[1]}"
        )


def score(
    labels: np.ndarray,
    mask: np.ndarray,
    proba: np.ndarray,
    categories=PRIORITY_CATEGORIES,
    thresholds=DEFAULT_THRESHOLD,
) -> pd.DataFrame:
    """Per-category precision/recall/F1 plus macro and micro rows.

    ``thresholds`` is a float, a mapping or one value per category. The
    threshold used is reported in the table so that a saved result always says
    how its decisions were made. Masked pairs are excluded. Macro averages the
    ten category rows; micro pools every unmasked (chunk, category) pair, so a
    frequent category weighs more there.
    """
    categories = list(categories)
    labels = np.asarray(labels)
    mask = np.asarray(mask, dtype=bool)
    proba = np.asarray(proba, dtype=float)
    _check_shapes(labels, mask, proba, categories)
    values = resolve_thresholds(thresholds, categories)
    pred = apply_thresholds(proba, values, categories)

    rows = []
    for i, category in enumerate(categories):
        keep = mask[:, i]
        y, p = labels[keep, i], pred[keep, i]
        if len(y):
            precision, recall, f1, _ = precision_recall_fscore_support(
                y, p, average="binary", zero_division=0
            )
        else:
            precision = recall = f1 = float("nan")
        ap = average_precision_score(y, proba[keep, i]) if y.any() else float("nan")
        rows.append(
            {
                "category": category,
                "threshold": float(values[i]),
                "support": int(y.sum()),
                "scored_chunks": int(keep.sum()),
                "masked_chunks": int((~keep).sum()),
                "predicted_positive": int(p.sum()),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "average_precision": ap,
            }
        )
    table = pd.DataFrame(rows, columns=METRIC_COLUMNS)

    y, p = labels[mask], pred[mask]
    if len(y):
        precision, recall, f1, _ = precision_recall_fscore_support(
            y, p, average="binary", zero_division=0
        )
    else:
        precision = recall = f1 = float("nan")
    counts = {
        c: int(table[c].sum())
        for c in ("support", "scored_chunks", "masked_chunks", "predicted_positive")
    }
    # One threshold column cannot describe ten different thresholds, so the
    # average rows leave it empty rather than implying a shared value.
    shared = float(values[0]) if len(set(values.tolist())) == 1 else float("nan")
    macro = {
        "category": "macro avg",
        "threshold": shared,
        **counts,
        **{
            c: table[c].mean()
            for c in ("precision", "recall", "f1", "average_precision")
        },
    }
    micro = {
        "category": "micro avg",
        "threshold": shared,
        **counts,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "average_precision": float("nan"),
    }
    table = pd.concat([table, pd.DataFrame([macro, micro])], ignore_index=True)
    return table.round({"precision": 4, "recall": 4, "f1": 4, "average_precision": 4})


def tune_thresholds(
    labels: np.ndarray,
    mask: np.ndarray,
    proba: np.ndarray,
    categories=PRIORITY_CATEGORIES,
    grid=THRESHOLD_GRID,
    default: float = DEFAULT_THRESHOLD,
) -> pd.DataFrame:
    """Pick the F1-maximising threshold per category from ``proba``.

    Intended for out-of-fold predictions on the training split. Scoring the
    chosen thresholds on the same predictions is optimistic, because they were
    selected there; the honest number is the test split scored at these
    thresholds afterwards.

    Ties are broken towards the lower threshold. Equal F1 at two thresholds
    means the lower one trades precision for recall, and this project treats a
    missed clause as worse than one the reviewer dismisses.

    A category with no positive chunk keeps ``default``: there is nothing to
    maximise.
    """
    categories = list(categories)
    labels = np.asarray(labels)
    mask = np.asarray(mask, dtype=bool)
    proba = np.asarray(proba, dtype=float)
    _check_shapes(labels, mask, proba, categories)
    grid = np.asarray(grid, dtype=float)
    if grid.size == 0:
        raise ValueError("threshold grid is empty")
    if not np.all(np.diff(grid) > 0):
        raise ValueError("threshold grid must be strictly increasing")

    rows = []
    for i, category in enumerate(categories):
        keep = mask[:, i]
        y, p = labels[keep, i], proba[keep, i]
        base = precision_recall_fscore_support(
            y, (p >= default).astype(np.int8), average="binary", zero_division=0
        )
        if not y.any():
            rows.append(
                {
                    "category": category,
                    "threshold": float(default),
                    "support": 0,
                    "f1": float("nan"),
                    "precision": float("nan"),
                    "recall": float("nan"),
                    "default_threshold": float(default),
                    "default_f1": float("nan"),
                    "f1_gain": float("nan"),
                }
            )
            continue
        best = None
        for t in grid:
            precision, recall, f1, _ = precision_recall_fscore_support(
                y, (p >= t).astype(np.int8), average="binary", zero_division=0
            )
            # Strict > keeps the first, lowest threshold that reaches the best F1.
            if best is None or f1 > best[1]:
                best = (float(t), f1, precision, recall)
        threshold, f1, precision, recall = best
        rows.append(
            {
                "category": category,
                "threshold": threshold,
                "support": int(y.sum()),
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "default_threshold": float(default),
                "default_f1": base[2],
                "f1_gain": f1 - base[2],
            }
        )
    table = pd.DataFrame(rows, columns=TUNING_COLUMNS)
    return table.round(
        {"f1": 4, "precision": 4, "recall": 4, "default_f1": 4, "f1_gain": 4}
    )


def threshold_map(tuned: pd.DataFrame) -> dict:
    """``{category: threshold}`` from a :func:`tune_thresholds` table."""
    missing = {"category", "threshold"} - set(tuned.columns)
    if missing:
        raise ValueError(f"tuning table is missing columns: {sorted(missing)}")
    return dict(zip(tuned["category"], tuned["threshold"].astype(float)))


def contract_folds(labeled: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    """Assign every contract to one held-out fold.

    Grouping is by contract, never by chunk: overlapping chunks from one
    contract share 128 tokens of text, so a chunk-level split would put nearly
    the same words in both halves. ``GroupKFold`` does not shuffle, so the
    assignment depends only on the contracts present and their chunk counts --
    the same table every run, with no seed to record.

    Returns one row per contract with its fold and chunk count, sorted by
    contract. Write it next to a model's results and later models can be scored
    on exactly these folds instead of a fresh partition.
    """
    if "contract" not in labeled.columns:
        raise ValueError("labeled chunks need a 'contract' column")
    if labeled.empty:
        raise ValueError("labeled chunks are empty")
    contracts = labeled["contract"].to_numpy()
    n_contracts = len(np.unique(contracts))
    if n_splits < 2 or n_splits > n_contracts:
        raise ValueError(
            f"n_splits must be between 2 and the number of contracts ({n_contracts})"
        )
    fold_of = np.empty(len(labeled), dtype=np.int64)
    for fold, (_, held_rows) in enumerate(
        GroupKFold(n_splits=n_splits).split(labeled, groups=contracts)
    ):
        fold_of[held_rows] = fold
    rows = pd.DataFrame({"contract": contracts, "fold": fold_of})
    spread = rows.groupby("contract")["fold"].nunique()
    if (spread > 1).any():
        split = spread[spread > 1].index.tolist()
        raise ValueError(f"{len(split)} contracts were split across folds: {split[:3]}")
    table = rows.groupby("contract", as_index=False).agg(
        fold=("fold", "first"), chunks=("fold", "size")
    )
    return table[FOLD_COLUMNS].sort_values("contract", ignore_index=True)


def fold_of_rows(labeled: pd.DataFrame, folds: pd.DataFrame) -> np.ndarray:
    """Fold number for each chunk row, from a :func:`contract_folds` table.

    Use this to score a new model on saved folds. Every contract in ``labeled``
    must appear in ``folds``; a contract with no fold would otherwise be left
    out of training and scoring without saying so.
    """
    missing = {"contract", "fold"} - set(folds.columns)
    if missing:
        raise ValueError(f"folds table is missing columns: {sorted(missing)}")
    lookup = dict(zip(folds["contract"], folds["fold"].astype(int)))
    absent = sorted(set(labeled["contract"]) - set(lookup))
    if absent:
        raise ValueError(f"{len(absent)} contracts have no fold: {absent[:3]}")
    return labeled["contract"].map(lookup).to_numpy(dtype=np.int64)
