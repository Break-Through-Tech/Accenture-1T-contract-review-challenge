"""Compare TF-IDF and optional Task 13 Legal-BERT scores on the CUAD test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from scripts.tuned_baseline import load_splits
from src.cuad_data import DATA_DIR, PRIORITY_CATEGORIES, build_split
from src.evaluation import (
    DEFAULT_THRESHOLD,
    apply_thresholds,
    resolve_thresholds,
    score,
    threshold_map,
)
from src.labeling import label_arrays
from src.tfidf_baseline import fit

REPO = Path(__file__).resolve().parent.parent
CATEGORIES = PRIORITY_CATEGORIES


def pool_contract_scores(
    chunks: pd.DataFrame, probabilities: np.ndarray, contracts: pd.Index
) -> np.ndarray:
    """Take the maximum chunk score per contract and category in CUAD contract order."""
    if probabilities.shape != (len(chunks), len(CATEGORIES)):
        raise ValueError("Expected one score per chunk and priority category")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Chunk probabilities must be finite and in [0, 1]")
    pooled = pd.DataFrame(probabilities).groupby(chunks.contract.to_numpy()).max()
    if set(pooled.index) != set(contracts):
        raise ValueError("Predictions must cover exactly the official test contracts")
    return pooled.reindex(contracts).to_numpy()


def load_transformer_scores(
    predictions_path: Path, run_path: Path, contracts: pd.Index
) -> tuple[np.ndarray, dict]:
    """Read the final Task 13 export and return contract scores and frozen thresholds."""
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if run.get("task") != 13 or run.get("smoke") is not False or run.get("second_run") is not False:
        raise ValueError("Expected test_run.json from Task 13's official test run")
    predictions = pd.read_parquet(predictions_path)
    required = {"contract", "chunk_id", *CATEGORIES}
    if missing := required - set(predictions.columns):
        raise ValueError(f"Task 13 predictions are missing columns: {sorted(missing)}")
    ids = predictions["chunk_id"]
    if ids.isna().any() or ids.eq("").any() or ids.duplicated().any():
        raise ValueError("Task 13 predictions need a unique, nonempty ID for each chunk")
    if predictions["contract"].isna().any():
        raise ValueError("Task 13 predictions contain missing contract names")
    if run.get("test_chunks") != len(predictions) or run.get("test_contracts") != len(contracts):
        raise ValueError("Task 13 prediction counts disagree with the official run")
    thresholds = run.get("thresholds")
    if not isinstance(thresholds, dict) or set(thresholds) != set(CATEGORIES):
        raise ValueError("Task 13 run must include a threshold for each priority category")
    values = resolve_thresholds(thresholds, CATEGORIES)
    if (values > 1).any():
        raise ValueError("Task 13 thresholds must be in [0, 1]")
    probabilities = predictions[CATEGORIES].to_numpy(dtype=float)
    return pool_contract_scores(predictions, probabilities, contracts), thresholds


def contract_f1_intervals(
    labels: np.ndarray, decisions: dict[str, np.ndarray], samples: int
) -> pd.DataFrame:
    """Resample whole contracts and use the same draws for all configurations."""
    rng = np.random.default_rng(42)
    counts = rng.multinomial(len(labels), np.full(len(labels), 1 / len(labels)), size=samples)
    positive = labels.astype(bool)
    rows = []
    bootstraps = {}

    def f1(tp, denom):
        return np.divide(2 * tp, denom, out=np.zeros_like(denom, dtype=float), where=denom > 0)

    for name, predicted in decisions.items():
        tp = counts @ (positive & predicted.astype(bool))
        denom = counts @ (labels.astype(int) + predicted)
        by_category = f1(tp, denom)
        bootstraps[name] = np.column_stack([
            by_category, by_category.mean(axis=1), f1(tp.sum(axis=1), denom.sum(axis=1)),
        ])
    bootstraps["tuned_minus_default"] = bootstraps["tfidf_tuned"] - bootstraps["tfidf_default"]
    if "legal_bert" in bootstraps:
        for baseline in ("tfidf_default", "tfidf_tuned"):
            bootstraps[f"legal_bert_minus_{baseline}"] = bootstraps["legal_bert"] - bootstraps[baseline]
    for name, values in bootstraps.items():
        low, high = np.percentile(values, [2.5, 97.5], axis=0)
        for category, lo, hi in zip([*CATEGORIES, "macro avg", "micro avg"], low, high):
            rows.append({"comparison": name, "category": category, "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def write_pr_curves(
    labels: np.ndarray, scores_by_model: dict[str, np.ndarray], output: Path
) -> None:
    # The two TF-IDF configurations share scores and therefore share each PR curve.
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(2, 5, figsize=(15, 6), sharex=True, sharey=True, layout="constrained")
    rows = []
    colors = {"TF-IDF": "#245b92", "Legal-BERT": "#b65327"}
    for i, (category, ax) in enumerate(zip(CATEGORIES, axes.flat)):
        for model, probabilities in scores_by_model.items():
            precision, recall, _ = precision_recall_curve(labels[:, i], probabilities[:, i])
            rows.append(pd.DataFrame({
                "model": model, "category": category, "recall": recall, "precision": precision,
            }))
            ax.step(
                recall[::-1], precision[::-1], where="post", color=colors[model], label=model,
            )
        ax.axhline(
            labels[:, i].mean(), color="#71717a", linestyle="--", linewidth=1, label="Prevalence",
        )
        ax.set(title=category, xlim=(0, 1.02), ylim=(0, 1.02))
        ax.grid(alpha=0.2)
    handles, names = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="outside lower center", ncol=len(names), frameon=False)
    fig.suptitle("Precision–recall by clause type | CUAD test contracts")
    fig.supxlabel("Recall")
    fig.supylabel("Precision")
    fig.savefig(output / "contract_pr_curves.svg")
    plt.close(fig)
    pd.concat(rows, ignore_index=True).to_csv(output / "contract_pr_curves.csv", index=False)


def run(
    label_dir: Path,
    output: Path,
    samples: int = 2000,
    *,
    transformer_predictions: Path | None = None,
    transformer_run: Path | None = None,
) -> pd.DataFrame:
    if samples < 1:
        raise ValueError("Bootstrap sample count must be positive")
    if (transformer_predictions is None) != (transformer_run is None):
        raise ValueError("Provide both the Task 13 predictions and run metadata")
    splits = load_splits(label_dir)
    train, test = splits["train"], splits["test"]
    truth = build_split("test").grid.pivot(index="contract", columns="category", values="has_clause")
    truth = truth[CATEGORIES]
    if set(train.contract) & set(truth.index):
        raise ValueError("Training data contains official test contracts")
    transformer = None
    if transformer_predictions is not None and transformer_run is not None:
        transformer = load_transformer_scores(transformer_predictions, transformer_run, truth.index)
    model = fit(train)
    probabilities = model.predict_proba(test.text)
    contract_probs = pool_contract_scores(test, probabilities, truth.index)
    contract_labels = truth.to_numpy(dtype=np.int8)
    labels, mask = label_arrays(test)
    tuned = pd.read_csv(REPO / "docs" / "tuned-baseline-results" / "thresholds.csv")
    configs = {
        "tfidf_default": DEFAULT_THRESHOLD,
        "tfidf_tuned": threshold_map(tuned),
    }

    def metrics(y, valid, scores):
        return pd.concat(
            [score(y, valid, scores, thresholds=t).assign(configuration=name)
             for name, t in configs.items()],
            ignore_index=True,
        )

    chunk_table = metrics(labels, mask, probabilities)
    contract_mask = np.ones_like(contract_labels, dtype=bool)
    contract_table = metrics(contract_labels, contract_mask, contract_probs)
    decisions = {name: apply_thresholds(contract_probs, t) for name, t in configs.items()}
    curve_scores = {"TF-IDF": contract_probs}
    if transformer is not None:
        transformer_probs, thresholds = transformer
        transformer_table = score(
            contract_labels, contract_mask, transformer_probs, thresholds=thresholds,
        )
        contract_table = pd.concat(
            [contract_table, transformer_table.assign(configuration="legal_bert")], ignore_index=True,
        )
        decisions["legal_bert"] = apply_thresholds(transformer_probs, thresholds)
        curve_scores["Legal-BERT"] = transformer_probs
    contract_table = contract_table.rename(columns={
        "scored_chunks": "scored_contract_category_pairs",
        "masked_chunks": "masked_contract_category_pairs",
    })
    intervals = contract_f1_intervals(contract_labels, decisions, samples)

    output.mkdir(parents=True, exist_ok=True)
    chunk_table.to_csv(output / "chunk_metrics.csv", index=False)
    contract_table.to_csv(output / "contract_metrics.csv", index=False)
    intervals.to_csv(output / "contract_f1_intervals.csv", index=False)
    write_pr_curves(contract_labels, curve_scores, output)
    return contract_table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-dir", type=Path, default=DATA_DIR / "interim" / "labels")
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR / "interim" / "evaluation")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--transformer-predictions", type=Path, help="Task 13 test_predictions.parquet")
    parser.add_argument("--transformer-run", type=Path, help="Task 13 test_run.json")
    args = parser.parse_args()
    if (args.transformer_predictions is None) != (args.transformer_run is None):
        parser.error("--transformer-predictions and --transformer-run must be supplied together")
    table = run(
        args.label_dir,
        args.output_dir,
        args.bootstrap,
        transformer_predictions=args.transformer_predictions,
        transformer_run=args.transformer_run,
    )
    print(table[table.category.isin(["macro avg", "micro avg"])][
        ["configuration", "category", "precision", "recall", "f1"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
