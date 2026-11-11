"""Class-imbalance strategies and how task #12 compares them.

Four arms train the same encoder on the same folds and differ in one thing:

=============  ==========================================================
bce            Plain masked BCE. The control: shows whether a fix helps at all.
pos_weight     BCE with positives weighted by negatives/positives, per category.
focal          Focal loss with the published defaults (gamma 2, alpha 0.25).
downsample     Plain BCE, with all-negative chunks subsampled in training only.
=============  ==========================================================

These strategies move predicted probabilities by very different amounts, so a shared
0.5 threshold would compare the size of that shift rather than the quality of the
ranking. Arms are therefore compared at per-category thresholds. Thresholds for a
held-out fold are tuned on the other folds only, so the score is not flattered by
choosing thresholds on the rows being scored.

The decision rule in :func:`choose_arm` was fixed before any result existed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from src.cuad_data import PRIORITY_CATEGORIES
from src.evaluation import apply_thresholds, score, threshold_map, tune_thresholds
from src.loss import masked_bce_with_logits, masked_focal_with_logits


@dataclass(frozen=True)
class Arm:
    name: str
    gamma: float | None = None  # focal loss
    alpha: float | None = None  # focal loss
    weight_positives: bool = False
    negatives_per_positive: float | None = None  # chunks; None keeps every negative


ARMS = {
    "bce": Arm("bce"),
    "pos_weight": Arm("pos_weight", weight_positives=True),
    "focal": Arm("focal", gamma=2.0, alpha=0.25),
    "downsample": Arm("downsample", negatives_per_positive=2.0),
}

#: Arms eligible to be picked, simplest first: fewest settings, no discarded data.
#: This order only settles arms the data cannot separate.
FIXES = ("pos_weight", "focal", "downsample")
CONTROL = "bce"

#: The four lowest F1 scores in the tuned TF-IDF baseline's cross-validation
#: (docs/tuned-baseline-results/train_cv_tuned.csv), named before any arm was run.
WEAK_CATEGORIES = (
    "Change Of Control",
    "Termination For Convenience",
    "Exclusivity",
    "Non-Compete",
)

#: Wider than src.evaluation.THRESHOLD_GRID, whose 0.05 floor could cut off a good
#: threshold for an arm such as focal loss that produces low probabilities.
ARM_THRESHOLD_GRID = np.round(np.arange(0.01, 0.9901, 0.01), 2)


def positive_weights(labels: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Negatives/positives per category over valid entries of the training rows."""
    labels, mask = np.asarray(labels), np.asarray(mask, dtype=bool)
    positives = ((labels == 1) & mask).sum(axis=0)
    negatives = ((labels == 0) & mask).sum(axis=0)
    if (positives == 0).any():
        raise ValueError("Every category needs a positive training entry for pos_weight")
    return negatives / positives


def make_loss(arm: Arm, labels: np.ndarray, mask: np.ndarray, device):
    """Return ``loss(logits, labels, label_mask)``; pass training rows only."""
    if arm.gamma is not None:
        return lambda logits, y, valid: masked_focal_with_logits(
            logits, y, valid, gamma=arm.gamma, alpha=arm.alpha
        )
    if arm.weight_positives:
        weights = torch.tensor(
            positive_weights(labels, mask), dtype=torch.float32, device=device
        )
        return lambda logits, y, valid: masked_bce_with_logits(logits, y, valid, weights)
    return masked_bce_with_logits


def epoch_rows(labels: np.ndarray, arm: Arm, seed: int, epoch: int) -> np.ndarray:
    """Training row positions for one epoch, sorted; the loader does the shuffling.

    A downsampling arm keeps every chunk with a positive label and draws a new sample
    of all-negative chunks each epoch, so more negatives are seen over a run while
    every epoch costs the same. Masked entries hold label 0, so a chunk with only
    partial evidence is in the negative pool; the mask still removes that category
    from its loss. Validation rows must never pass through here.
    """
    if arm.negatives_per_positive is None:
        return np.arange(len(labels))
    has_positive = np.asarray(labels).sum(axis=1) > 0
    positive, negative = np.flatnonzero(has_positive), np.flatnonzero(~has_positive)
    take = min(len(negative), round(arm.negatives_per_positive * len(positive)))
    drawn = np.random.default_rng([seed, epoch]).choice(negative, size=take, replace=False)
    return np.sort(np.concatenate([positive, drawn]))


def cross_fit_decisions(
    labels: np.ndarray,
    mask: np.ndarray,
    proba: np.ndarray,
    fold_of: np.ndarray,
    categories=PRIORITY_CATEGORIES,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Decide each fold's rows with thresholds tuned on the other folds' rows.

    Returns (0/1 decisions, a table of thresholds with one row per fold). Needs two
    or more folds, because a fold's thresholds must come from somewhere else.
    """
    categories = list(categories)
    folds = np.unique(fold_of)
    if len(folds) < 2:
        raise ValueError("Cross-fitting thresholds needs at least two folds")
    decisions = np.zeros(labels.shape, dtype=np.int8)
    rows = []
    for fold in folds:
        held, others = fold_of == fold, fold_of != fold
        tuned = tune_thresholds(
            labels[others], mask[others], proba[others], categories, grid=ARM_THRESHOLD_GRID
        )
        thresholds = threshold_map(tuned)
        decisions[held] = apply_thresholds(proba[held], thresholds, categories)
        rows.append({"fold": int(fold), **thresholds})
    return decisions, pd.DataFrame(rows)


def score_decisions(
    labels: np.ndarray, mask: np.ndarray, decisions: np.ndarray, proba: np.ndarray,
    categories=PRIORITY_CATEGORIES,
) -> pd.DataFrame:
    """Shared scoring for decisions already made; AP still comes from the scores."""
    table = score(labels, mask, decisions.astype(float), categories, thresholds=0.5)
    table["threshold"] = float("nan")  # thresholds differ by fold; see the thresholds table
    table["average_precision"] = score(labels, mask, proba, categories)["average_precision"]
    return table


def _f1_per_category(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> np.ndarray:
    denominator = 2 * tp + fp + fn
    return np.divide(2 * tp, denominator, out=np.zeros_like(denominator), where=denominator > 0)


def paired_bootstrap(
    labels: np.ndarray,
    mask: np.ndarray,
    decisions: dict[str, np.ndarray],
    contracts: np.ndarray,
    categories=PRIORITY_CATEGORIES,
    weak=WEAK_CATEGORIES,
    samples: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """95% intervals for every arm pair's gain in macro F1 and weak-category F1.

    Contracts are resampled, because a contract's chunks are not independent, and the
    same resample scores both arms, because the arms are paired on the same text.
    ``difference`` in a row is arm_a minus arm_b.
    """
    categories = list(categories)
    names, row_of = np.unique(contracts, return_inverse=True)
    valid_true = (labels == 1) & mask
    valid_false = (labels == 0) & mask
    counts = {}
    for arm, pred in decisions.items():
        tp, fp, fn = [np.zeros((len(names), len(categories))) for _ in range(3)]
        np.add.at(tp, row_of, (pred == 1) & valid_true)
        np.add.at(fp, row_of, (pred == 1) & valid_false)
        np.add.at(fn, row_of, (pred == 0) & valid_true)
        counts[arm] = (tp, fp, fn)

    draws = np.random.default_rng(seed).multinomial(
        len(names), np.full(len(names), 1 / len(names)), size=samples
    ).astype(float)
    weak_columns = [categories.index(c) for c in weak]

    def macro_scores(arm, weights):
        tp, fp, fn = (weights @ c for c in counts[arm])
        f1 = _f1_per_category(tp, fp, fn)
        return {"macro_f1": f1.mean(axis=-1), "weak_f1": f1[..., weak_columns].mean(axis=-1)}

    point = {arm: macro_scores(arm, np.ones(len(names))) for arm in counts}
    boot = {arm: macro_scores(arm, draws) for arm in counts}
    arms = list(counts)
    rows = []
    for i, a in enumerate(arms):
        for b in arms[i + 1:]:
            for metric in ("macro_f1", "weak_f1"):
                low, high = np.percentile(boot[a][metric] - boot[b][metric], [2.5, 97.5])
                rows.append({
                    "arm_a": a, "arm_b": b, "metric": metric,
                    "difference": float(point[a][metric] - point[b][metric]),
                    "ci_low": float(low), "ci_high": float(high),
                })
    return pd.DataFrame(rows)


def _clearly_better(pairs: pd.DataFrame, a: str, b: str, metric: str) -> bool:
    """True if arm ``a`` beats arm ``b``: the 95% interval excludes zero in a's favor."""
    if a == b:
        return False
    row = pairs.loc[(pairs.metric == metric)
                    & (((pairs.arm_a == a) & (pairs.arm_b == b))
                       | ((pairs.arm_a == b) & (pairs.arm_b == a)))]
    if len(row) != 1:
        raise ValueError(f"Expected one {metric} comparison of {a} and {b}, got {len(row)}")
    row = row.iloc[0]
    return bool(row.ci_low > 0 if row.arm_a == a else row.ci_high < 0)


def choose_arm(summary: pd.DataFrame, pairs: pd.DataFrame, fixes=FIXES, control=CONTROL) -> dict:
    """Apply the decision rule fixed before any arm was run.

    1. Keep the fix with the best macro F1, and any fix not clearly worse than it.
    2. Of those, keep the best on weak-category F1 and any not clearly worse than it.
    3. If several remain, the data cannot separate them: take the first in ``fixes``,
       which lists the simplest first.

    "Clearly" means the paired 95% interval excludes zero. The control is not eligible,
    because the task is to pick a fix, but the result states how the pick compares to
    it so that "no measurable gain" is reported rather than hidden.
    """
    table = summary.set_index("arm")
    leader = table.loc[list(fixes), "macro_f1"].idxmax()
    remaining = [f for f in fixes if not _clearly_better(pairs, leader, f, "macro_f1")]
    tied_on_macro = list(remaining)
    weak_leader = table.loc[remaining, "weak_f1"].idxmax()
    remaining = [f for f in remaining if not _clearly_better(pairs, weak_leader, f, "weak_f1")]
    pick = remaining[0]
    return {
        "pick": pick,
        "macro_f1_leader": leader,
        "tied_on_macro_f1": tied_on_macro,
        "weak_f1_leader": weak_leader,
        "tied_after_weak_f1": remaining,
        "decided_by": (
            "macro_f1" if len(tied_on_macro) == 1
            else "weak_f1" if len(remaining) == 1
            else "simplicity"
        ),
        "clearly_beats_control_macro_f1": _clearly_better(pairs, pick, control, "macro_f1"),
        "clearly_beats_control_weak_f1": _clearly_better(pairs, pick, control, "weak_f1"),
        "clearly_worse_than_control_macro_f1": _clearly_better(pairs, control, pick, "macro_f1"),
    }
