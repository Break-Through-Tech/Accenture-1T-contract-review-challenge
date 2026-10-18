"""TF-IDF multi-label baseline for task #7.

Input is the labeled chunk table from task #6 (``src.labeling``): one row per chunk
with ``text``, ``labels`` and ``label_mask`` in PRIORITY_CATEGORIES order.

One TF-IDF vocabulary is fitted on the training chunks and shared by all categories.
Each category then gets its own binary logistic regression (one-vs-rest), which is
what makes the model multi-label: a chunk can be positive for several categories.

Masked (chunk, category) pairs are the ambiguous partial-overlap cases from task #6.
They are left out of fitting and scoring for that category only; the chunk still
counts for every other category.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.cuad_data import PRIORITY_CATEGORIES
from src.evaluation import (
    DEFAULT_THRESHOLD,
    METRIC_COLUMNS,
    apply_thresholds,
    contract_folds,
    fold_of_rows,
    score,
)
from src.labeling import label_arrays

# METRIC_COLUMNS and score moved to src.evaluation in task #9 so that the
# keyword rules and October's transformer are scored by the same code. They are
# re-exported here because this module's own callers already import them.
__all__ = [
    "DEFAULT_CONFIG",
    "METRIC_COLUMNS",
    "BaselineConfig",
    "TfidfBaseline",
    "cross_validate",
    "evaluate",
    "fit",
    "out_of_fold_proba",
    "score",
]


@dataclass(frozen=True)
class BaselineConfig:
    """Fixed defaults, set before scoring. Do not tune on the test split."""

    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 2
    max_df: float = 0.95
    sublinear_tf: bool = True
    # Legal casing is not needed for bag-of-words; stop words are kept on purpose
    # because "not", "shall", "without" carry meaning in clauses.
    lowercase: bool = True
    C: float = 10.0
    class_weight: str | None = "balanced"
    # The shared starting threshold. Task #9 tunes one per category instead;
    # see docs/tuned-baseline.md.
    threshold: float = DEFAULT_THRESHOLD
    seed: int = 42

    def to_dict(self) -> dict:
        out = asdict(self)
        out["ngram_range"] = list(self.ngram_range)
        return out


DEFAULT_CONFIG = BaselineConfig()


@dataclass
class TfidfBaseline:
    """Fitted vectorizer plus one classifier per category."""

    vectorizer: TfidfVectorizer
    classifiers: list
    categories: list
    config: BaselineConfig

    def predict_proba(self, texts) -> np.ndarray:
        """Return an (n_chunks, n_categories) array of positive-class scores."""
        features = self.vectorizer.transform(texts)
        return np.column_stack(
            [clf.predict_proba(features)[:, 1] for clf in self.classifiers]
        )

    def predict(self, texts, thresholds=None) -> np.ndarray:
        """0/1 predictions, one column per category.

        ``thresholds`` takes a float, a mapping or one value per category and
        falls back to the single value in the config.
        """
        if thresholds is None:
            thresholds = self.config.threshold
        return apply_thresholds(self.predict_proba(texts), thresholds, self.categories)

    def top_terms(self, n: int = 15) -> pd.DataFrame:
        """Highest-weighted terms per category, for explaining the baseline."""
        vocab = self.vectorizer.get_feature_names_out()
        rows = []
        for category, clf in zip(self.categories, self.classifiers):
            weights = clf.coef_[0]
            for rank, i in enumerate(np.argsort(weights)[::-1][:n], start=1):
                rows.append(
                    {
                        "category": category,
                        "rank": rank,
                        "term": vocab[i],
                        "weight": round(float(weights[i]), 4),
                    }
                )
        return pd.DataFrame(rows, columns=["category", "rank", "term", "weight"])


def _arrays(labeled: pd.DataFrame, categories) -> tuple[np.ndarray, np.ndarray]:
    missing = {"text", "labels", "label_mask"} - set(labeled.columns)
    if missing:
        raise ValueError(f"labeled chunks are missing columns: {sorted(missing)}")
    return label_arrays(labeled, len(categories))


def fit(
    labeled: pd.DataFrame,
    categories=PRIORITY_CATEGORIES,
    config: BaselineConfig = DEFAULT_CONFIG,
) -> TfidfBaseline:
    """Fit the vocabulary on all chunks and one classifier per category."""
    categories = list(categories)
    labels, mask = _arrays(labeled, categories)
    vectorizer = TfidfVectorizer(
        ngram_range=config.ngram_range,
        min_df=config.min_df,
        max_df=config.max_df,
        sublinear_tf=config.sublinear_tf,
        lowercase=config.lowercase,
    )
    features = vectorizer.fit_transform(labeled["text"])
    classifiers = []
    for i, category in enumerate(categories):
        keep = mask[:, i]
        y = labels[keep, i]
        if len(np.unique(y)) < 2:
            raise ValueError(f"{category!r} needs positive and negative chunks")
        clf = LogisticRegression(
            C=config.C,
            class_weight=config.class_weight,
            solver="liblinear",
            random_state=config.seed,
        )
        classifiers.append(clf.fit(features[keep], y))
    return TfidfBaseline(vectorizer, classifiers, categories, config)


def evaluate(
    model: TfidfBaseline, labeled: pd.DataFrame, thresholds=None
) -> tuple[pd.DataFrame, np.ndarray]:
    """Score a fitted model on a labeled chunk table. Returns (metrics, scores).

    ``thresholds`` defaults to the single value in the model's config, so an
    untuned call reproduces the task #7 numbers. Pass a mapping from
    :func:`src.evaluation.tune_thresholds` to score at tuned thresholds.
    """
    labels, mask = _arrays(labeled, model.categories)
    proba = model.predict_proba(labeled["text"])
    if thresholds is None:
        thresholds = model.config.threshold
    return score(labels, mask, proba, model.categories, thresholds), proba


def out_of_fold_proba(
    labeled: pd.DataFrame,
    categories=PRIORITY_CATEGORIES,
    config: BaselineConfig = DEFAULT_CONFIG,
    n_splits: int = 5,
    folds: pd.DataFrame | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Predict every training chunk from a model that never saw its contract.

    Returns the (n_chunks, n_categories) score array and the fold table used.
    Pass ``folds`` to reuse a saved assignment; otherwise one is built with
    :func:`src.evaluation.contract_folds`.

    These scores are what thresholds are tuned on. Because each chunk is scored
    by a model fitted without its contract, the overlapping chunks of one
    contract cannot leak across the split.
    """
    categories = list(categories)
    if folds is None:
        folds = contract_folds(labeled, n_splits)
    fold_of = fold_of_rows(labeled, folds)
    proba = np.zeros((len(labeled), len(categories)), dtype=float)
    for fold in np.unique(fold_of):
        held = fold_of == fold
        model = fit(labeled.loc[~held], categories, config)
        proba[held] = model.predict_proba(labeled.loc[held, "text"])
    return proba, folds


def cross_validate(
    labeled: pd.DataFrame,
    categories=PRIORITY_CATEGORIES,
    config: BaselineConfig = DEFAULT_CONFIG,
    n_splits: int = 5,
    folds: pd.DataFrame | None = None,
    thresholds=None,
) -> pd.DataFrame:
    """Out-of-fold metrics on one split, with folds grouped by contract.

    Chunks from one contract overlap by 128 tokens, so splitting by chunk would
    leak. This is the number to use for any tuning; the test split stays
    untouched. Use :func:`out_of_fold_proba` when the scores themselves are
    needed, for instance to choose thresholds.
    """
    categories = list(categories)
    labels, mask = _arrays(labeled, categories)
    proba, _ = out_of_fold_proba(labeled, categories, config, n_splits, folds)
    if thresholds is None:
        thresholds = config.threshold
    return score(labels, mask, proba, categories, thresholds)
