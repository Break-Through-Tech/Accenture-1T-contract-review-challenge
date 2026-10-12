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
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold

from src.cuad_data import PRIORITY_CATEGORIES
from src.labeling import label_arrays

METRIC_COLUMNS = [
    "category",
    "support",
    "scored_chunks",
    "masked_chunks",
    "predicted_positive",
    "precision",
    "recall",
    "f1",
    "average_precision",
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
    threshold: float = 0.5
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

    def predict(self, texts) -> np.ndarray:
        return (self.predict_proba(texts) >= self.config.threshold).astype(np.int8)

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


def score(
    labels: np.ndarray,
    mask: np.ndarray,
    proba: np.ndarray,
    categories=PRIORITY_CATEGORIES,
    threshold: float = DEFAULT_CONFIG.threshold,
) -> pd.DataFrame:
    """Per-category precision/recall/F1 plus macro and micro rows.

    Masked pairs are excluded. Micro pools every unmasked (chunk, category) pair.
    """
    categories = list(categories)
    pred = (proba >= threshold).astype(np.int8)
    rows = []
    for i, category in enumerate(categories):
        keep = mask[:, i]
        y, p = labels[keep, i], pred[keep, i]
        precision, recall, f1, _ = precision_recall_fscore_support(
            y, p, average="binary", zero_division=0
        )
        ap = average_precision_score(y, proba[keep, i]) if y.any() else float("nan")
        rows.append(
            {
                "category": category,
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
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, p, average="binary", zero_division=0
    )
    counts = {
        c: int(table[c].sum())
        for c in ("support", "scored_chunks", "masked_chunks", "predicted_positive")
    }
    macro = {
        "category": "macro avg",
        **counts,
        **{
            c: table[c].mean()
            for c in ("precision", "recall", "f1", "average_precision")
        },
    }
    micro = {
        "category": "micro avg",
        **counts,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "average_precision": float("nan"),
    }
    table = pd.concat([table, pd.DataFrame([macro, micro])], ignore_index=True)
    return table.round({"precision": 4, "recall": 4, "f1": 4, "average_precision": 4})


def evaluate(
    model: TfidfBaseline, labeled: pd.DataFrame
) -> tuple[pd.DataFrame, np.ndarray]:
    """Score a fitted model on a labeled chunk table. Returns (metrics, scores)."""
    labels, mask = _arrays(labeled, model.categories)
    proba = model.predict_proba(labeled["text"])
    return score(labels, mask, proba, model.categories, model.config.threshold), proba


def cross_validate(
    labeled: pd.DataFrame,
    categories=PRIORITY_CATEGORIES,
    config: BaselineConfig = DEFAULT_CONFIG,
    n_splits: int = 5,
) -> pd.DataFrame:
    """Out-of-fold metrics on one split, with folds grouped by contract.

    Chunks from one contract overlap by 128 tokens, so splitting by chunk would leak.
    This is the number to use for any tuning; the test split stays untouched.
    """
    categories = list(categories)
    labels, mask = _arrays(labeled, categories)
    proba = np.zeros(labels.shape, dtype=float)
    groups = labeled["contract"].to_numpy()
    for train_rows, held_rows in GroupKFold(n_splits=n_splits).split(
        labeled, groups=groups
    ):
        model = fit(labeled.iloc[train_rows], categories, config)
        proba[held_rows] = model.predict_proba(labeled["text"].iloc[held_rows])
    return score(labels, mask, proba, categories, config.threshold)
