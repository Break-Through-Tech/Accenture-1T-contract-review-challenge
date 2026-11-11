"""Task #13: final legal-bert training with pos_weight, CV thresholds, one test run.

Pieces used by scripts/final_training.py:

- train_encoder   training loop (pos_weight loss, AMP, linear warmup/decay)
- predict         probabilities for a set of rows
- sweep tables    out-of-fold scores per (learning rate, epochs) config
- select_config   pick a config with a rule fixed before any sweep run
- final thresholds per category, from out-of-fold predictions, with a stability check
- checkpoints     save/load encoder + head + tokenizer + thresholds

Settings from earlier tasks: encoder legal-bert (#11), pos_weight (#12), folds (#9).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, get_linear_schedule_with_warmup

from src.candidates import CANDIDATES, EncoderClassifier
from src.cuad_data import PRIORITY_CATEGORIES
from src.dataset import ChunkDataset
from src.evaluation import score, threshold_map, tune_thresholds
from src.imbalance import (
    ARM_THRESHOLD_GRID,
    ARMS,
    WEAK_CATEGORIES,
    cross_fit_decisions,
    make_loss,
    paired_bootstrap,
    score_decisions,
)
from src.labeling import label_arrays
from src.train import evaluate

ENCODER = "legal-bert"
ARM = "pos_weight"

# sweep grid, fixed before running
LEARNING_RATES = (2e-5, 3e-5)
MAX_EPOCHS = 3  # each sweep run is scored after every epoch up to this
BATCH_SIZE = 8
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
SEED = 42

# threshold stability flags (task #9 lesson: Non-Compete)
UNSTABLE_THRESHOLD_STD = 0.10
LOW_SUPPORT = 200


def config_name(learning_rate: float, epochs: int) -> str:
    return f"lr{learning_rate:g}-ep{epochs}"


# ------------------------------------------------------------------ training


def train_encoder(
    model,
    train_rows: pd.DataFrame,
    tokenizer,
    *,
    epochs: int,
    learning_rate: float,
    schedule_epochs: int | None = None,
    batch_size: int = BATCH_SIZE,
    weight_decay: float = WEIGHT_DECAY,
    warmup_ratio: float = WARMUP_RATIO,
    seed: int = SEED,
    device=None,
    max_steps: int | None = None,
    on_epoch_end=None,
) -> dict:
    """Train in place with the pos_weight loss; returns a history dict.

    schedule_epochs: length of the LR schedule. The sweep scores epoch k of a
    MAX_EPOCHS schedule, so the final run uses the same schedule and stops at k.
    on_epoch_end(epoch, model): called after each epoch (sweep saves predictions).
    """
    schedule_epochs = schedule_epochs or epochs
    if not 1 <= epochs <= schedule_epochs:
        raise ValueError("Need 1 <= epochs <= schedule_epochs")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp = device.type == "cuda"
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset = ChunkDataset(train_rows, tokenizer)
    labels, mask = label_arrays(train_rows)
    weights = make_loss(ARMS[ARM], labels, mask, device)  # weights from these rows only
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    steps_per_epoch = math.ceil(len(dataset) / batch_size)
    total = schedule_epochs * steps_per_epoch
    if max_steps:
        total = min(total, max_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(warmup_ratio * total), total)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    history, steps, skipped = [], 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(seed + epoch),
        )
        loss_sum, valid = 0.0, 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            count = int(batch["label_mask"].sum())
            if not count:
                continue  # all-masked batch: no AdamW step
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                logits = model(
                    input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
                ).logits
                loss = weights(logits, batch["labels"], batch["label_mask"])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=not amp)
            scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped += scaler.get_scale() < scale
            scheduler.step()
            steps += 1
            loss_sum += loss.item() * count
            valid += count
            if steps == 1 or steps % 100 == 0:
                print(f"epoch={epoch} step={steps}/{total} loss={loss.item():.5f}", flush=True)
            if steps >= total:
                break
        history.append({"epoch": epoch, "steps": steps, "train_loss": loss_sum / max(valid, 1)})
        if on_epoch_end is not None:
            on_epoch_end(epoch, model)
        if steps >= total:
            break
    if skipped > max(10, steps / 10):
        raise FloatingPointError(f"AMP skipped {skipped} of {steps} steps")
    return {
        "history": history, "optimizer_steps": steps, "amp_skipped_steps": int(skipped),
        "schedule_steps": total, "steps_per_epoch": steps_per_epoch, "mixed_precision": amp,
        "train_chunks": len(dataset), "positive_weights": weights_of(labels, mask),
    }


def weights_of(labels, mask) -> dict:
    from src.imbalance import positive_weights

    return dict(zip(PRIORITY_CATEGORIES, np.round(positive_weights(labels, mask), 3).tolist()))


def predict(model, rows: pd.DataFrame, tokenizer, *, batch_size=16, device=None):
    """(validation BCE, probabilities) in row order. Rows need labels and masks."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    loader = DataLoader(ChunkDataset(rows, tokenizer), batch_size=batch_size, shuffle=False)
    model.to(device)
    return evaluate(model, loader, device)


# ------------------------------------------------------------------ selection


def oof_matrix(rows: pd.DataFrame, fold_of: np.ndarray, predictions: dict[int, pd.DataFrame]):
    """Stack per-fold prediction tables into one (n_rows, 10) array aligned to rows."""
    position = pd.Series(np.arange(len(rows)), index=rows.chunk_id)
    proba = np.full((len(rows), len(PRIORITY_CATEGORIES)), np.nan)
    for fold, table in predictions.items():
        held = position.loc[table.chunk_id].to_numpy()
        if not np.array_equal(np.sort(held), np.flatnonzero(fold_of == fold)):
            raise ValueError(f"Fold {fold} predictions do not cover exactly that fold's chunks")
        proba[held] = np.array(table.probabilities.tolist())
    if np.isnan(proba).any():
        raise ValueError("Some training chunks have no out-of-fold prediction")
    return proba


def sweep_table(rows, fold_of, configs: dict[str, np.ndarray]):
    """Score every config with thresholds cross-fitted across folds (honest CV).

    configs: name -> out-of-fold probabilities. Returns (summary, decisions, per_category).
    """
    labels, mask = label_arrays(rows)
    summary, decisions, per_category = [], {}, {}
    for name, proba in configs.items():
        decided, _ = cross_fit_decisions(labels, mask, proba, fold_of)
        table = score_decisions(labels, mask, decided, proba).set_index("category")
        decisions[name], per_category[name] = decided, table
        lr, ep = name.removeprefix("lr").split("-ep")
        summary.append({
            "config": name, "learning_rate": float(lr), "epochs": int(ep),
            "macro_f1": table.f1["macro avg"],
            "weak_f1": table.f1[list(WEAK_CATEGORIES)].mean(),
            "macro_precision": table.precision["macro avg"],
            "macro_recall": table.recall["macro avg"],
            "macro_average_precision": table.average_precision["macro avg"],
            "micro_f1": table.f1["micro avg"],
        })
    return pd.DataFrame(summary), decisions, per_category


def select_config(summary: pd.DataFrame, pairs: pd.DataFrame) -> dict:
    """Rule fixed before the sweep ran.

    1. Leader = best cross-fitted macro F1.
    2. Keep configs not clearly worse than the leader (paired 95% CI includes 0).
    3. Of those, fewest epochs, then lowest learning rate (cheaper, less overfit risk).
    """
    table = summary.set_index("config")
    leader = table.macro_f1.idxmax()

    def clearly_worse(name):
        if name == leader:
            return False
        row = pairs.loc[(pairs.metric == "macro_f1") & (
            ((pairs.arm_a == leader) & (pairs.arm_b == name))
            | ((pairs.arm_a == name) & (pairs.arm_b == leader)))].iloc[0]
        return bool(row.ci_low > 0) if row.arm_a == leader else bool(row.ci_high < 0)

    tied = [n for n in table.index if not clearly_worse(n)]
    pick = min(tied, key=lambda n: (table.epochs[n], table.learning_rate[n]))
    return {
        "pick": pick, "learning_rate": float(table.learning_rate[pick]),
        "epochs": int(table.epochs[pick]), "leader": leader, "tied_with_leader": tied,
        "decided_by": "macro_f1" if len(tied) == 1 else "fewest_epochs_then_lowest_lr",
    }


def final_thresholds(rows, fold_of, proba) -> pd.DataFrame:
    """Per-category thresholds tuned on all out-of-fold rows, plus fold stability.

    Stability: thresholds re-tuned with each fold left out (same as cross-fitting).
    A wide spread or low support is flagged so the report can call it out.
    """
    labels, mask = label_arrays(rows)
    tuned = tune_thresholds(labels, mask, proba, grid=ARM_THRESHOLD_GRID)
    _, per_fold = cross_fit_decisions(labels, mask, proba, fold_of)
    spread = per_fold[list(PRIORITY_CATEGORIES)]
    tuned = tuned.set_index("category").loc[list(PRIORITY_CATEGORIES)].reset_index()
    tuned["fold_threshold_min"] = spread.min().to_numpy()
    tuned["fold_threshold_max"] = spread.max().to_numpy()
    tuned["fold_threshold_std"] = spread.std().round(4).to_numpy()
    tuned["flag"] = [
        "; ".join(f for f in (
            "unstable threshold" if std > UNSTABLE_THRESHOLD_STD else "",
            "low support" if support < LOW_SUPPORT else "",
        ) if f)
        for std, support in zip(tuned.fold_threshold_std, tuned.support)
    ]
    return tuned


def bootstrap_pairs(rows, decisions: dict[str, np.ndarray], samples=2000, seed=SEED):
    labels, mask = label_arrays(rows)
    return paired_bootstrap(
        labels, mask, decisions, rows.contract.to_numpy(), samples=samples, seed=seed
    )


# ------------------------------------------------------------------ checkpoints


def new_model(*, local_files_only=False) -> EncoderClassifier:
    candidate = CANDIDATES[ENCODER]
    return EncoderClassifier.from_pretrained(
        candidate, candidate.revision, local_files_only=local_files_only
    )


def save_checkpoint(model: EncoderClassifier, tokenizer, directory: Path, info: dict) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(directory / "encoder")
    torch.save(model.classifier.state_dict(), directory / "head.pt")
    tokenizer.save(str(directory / "tokenizer.json"))
    (directory / "classifier.json").write_text(json.dumps({
        "categories": list(PRIORITY_CATEGORIES), "dropout": model.dropout.p, **info,
    }, indent=2) + "\n")


def load_checkpoint(directory: Path) -> EncoderClassifier:
    directory = Path(directory)
    info = json.loads((directory / "classifier.json").read_text())
    if info["categories"] != list(PRIORITY_CATEGORIES):
        raise ValueError("Checkpoint category order differs from PRIORITY_CATEGORIES")
    encoder = AutoModel.from_pretrained(
        directory / "encoder", config=AutoConfig.from_pretrained(directory / "encoder")
    )
    model = EncoderClassifier(encoder, dropout=info["dropout"])
    model.classifier.load_state_dict(torch.load(directory / "head.pt", weights_only=True))
    model.eval()
    return model


def score_at(rows: pd.DataFrame, proba: np.ndarray, thresholds) -> pd.DataFrame:
    labels, mask = label_arrays(rows)
    return score(labels, mask, proba, thresholds=thresholds)


__all__ = [
    "ARM", "BATCH_SIZE", "ENCODER", "LEARNING_RATES", "MAX_EPOCHS", "SEED",
    "bootstrap_pairs", "config_name", "final_thresholds", "load_checkpoint", "new_model",
    "oof_matrix", "predict", "save_checkpoint", "score_at", "select_config",
    "sweep_table", "threshold_map", "train_encoder",
]
