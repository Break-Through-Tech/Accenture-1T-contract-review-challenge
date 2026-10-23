"""Train the CUAD classifier: python -m src.train --help."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from torch.utils.data import DataLoader
from transformers import BertConfig

from src.chunking import TOKENIZER_MODEL, TOKENIZER_REVISION, load_tokenizer
from src.cuad_data import PRIORITY_CATEGORIES
from src.dataset import ChunkDataset, split_by_fold
from src.evaluation import score
from src.labeling import label_arrays
from src.model import ContractClassifier, configure_labels, load_pretrained_model


@dataclass
class TrainingConfig:
    labels: Path = Path("data/interim/labels/train/labeled_chunks.parquet")
    folds: Path = Path("docs/tuned-baseline-results/folds.csv")
    output_dir: Path = Path("data/interim/models/transformer")
    validation_fold: int = 0
    epochs: int = 1
    batch_size: int = 2
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    seed: int = 42
    max_contracts: int | None = None
    max_steps: int | None = None
    tiny: bool = False
    local_files_only: bool = False
    device: str = "auto"
    cpu_threads: int = 2

    def validate(self):
        for name in ("epochs", "batch_size", "cpu_threads"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("max_contracts", "max_steps"):
            value = getattr(self, name)
            minimum = 2 if name == "max_contracts" else 1
            if value is not None and (type(value) is not int or value < minimum):
                raise ValueError(f"{name} must be at least {minimum}")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu, or cuda")


def select_contracts(rows, folds, *, validation_fold, count=None, seed=42):
    """Select whole contracts deterministically, with both sides represented.

    Selection uses identifiers and saved fold membership, never labels. Keep
    roughly one fifth for validation in a smoke subset; do not reassign folds.
    """
    train, validation = split_by_fold(rows, folds, validation_fold=validation_fold)
    if count is None:
        return train, validation
    if type(count) is not int or not 2 <= count <= rows.contract.nunique():
        raise ValueError("Contract count must be between 2 and the available count")
    rng = np.random.default_rng(seed)
    train_names = rng.permutation(sorted(train.contract.unique()))
    val_names = rng.permutation(sorted(validation.contract.unique()))
    n_val = min(len(val_names), max(1, math.ceil(count / 5)))
    n_val = max(n_val, count - len(train_names))
    n_train = count - n_val
    return (
        train.loc[train.contract.isin(train_names[:n_train])].reset_index(drop=True),
        validation.loc[validation.contract.isin(val_names[:n_val])].reset_index(drop=True),
    )


def evaluate(model, loader, device):
    """Return valid-entry-weighted BCE and scores in unshuffled loader order."""
    model.eval()
    total, valid = 0.0, 0
    probabilities = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch)
            if not torch.isfinite(output.loss) or not torch.isfinite(output.logits).all():
                raise FloatingPointError("Nonfinite validation loss or logits")
            count = int(batch["label_mask"].sum())
            total += output.loss.item() * count
            valid += count
            probabilities.append(output.logits.sigmoid().cpu().numpy())
    if not valid:
        raise ValueError("Validation has no valid labels")
    return total / valid, np.concatenate(probabilities)


def run_training(config: TrainingConfig, *, tokenizer=None) -> dict:
    config.validate()
    output_dir = Path(config.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("output_dir must be empty; choose a new directory for each run")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.cpu_threads)
    if config.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    device = torch.device(
        "cuda" if config.device == "auto" and torch.cuda.is_available()
        else "cpu" if config.device == "auto" else config.device
    )
    rows = pd.read_parquet(config.labels)
    folds = pd.read_csv(config.folds)
    train_rows, validation_rows = select_contracts(
        rows, folds, validation_fold=config.validation_fold,
        count=config.max_contracts, seed=config.seed,
    )
    tokenizer = tokenizer or load_tokenizer(local_files_only=config.local_files_only)
    train = ChunkDataset(train_rows, tokenizer)
    validation = ChunkDataset(validation_rows, tokenizer)
    if not train.label_mask.any():
        raise ValueError("Training has no valid labels")
    if not validation.label_mask.any():
        raise ValueError("Validation has no valid labels")
    train_loader = DataLoader(
        train, batch_size=config.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    validation_loader = DataLoader(validation, batch_size=config.batch_size, shuffle=False)
    if config.tiny:
        model = ContractClassifier(configure_labels(BertConfig(
            vocab_size=tokenizer.get_vocab_size(), hidden_size=32,
            num_hidden_layers=1, num_attention_heads=2, intermediate_size=64,
        )))
    else:
        model = load_pretrained_model(local_files_only=config.local_files_only)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    print(
        f"{device}: {train_rows.contract.nunique()} train contracts/{len(train)} chunks; "
        f"{validation_rows.contract.nunique()} validation contracts/{len(validation)} chunks; "
        f"mode={'tiny-random' if config.tiny else 'pretrained-bert'}", flush=True,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    history, steps, seen_chunks = [], 0, 0
    for epoch in range(config.epochs):
        model.train()
        total, valid, skipped = 0.0, 0, 0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            count = int(batch["label_mask"].sum())
            if not count:
                skipped += 1
                continue  # AdamW weight decay must not update an all-masked batch.
            optimizer.zero_grad(set_to_none=True)
            output = model(**batch)
            if not torch.isfinite(output.loss):
                raise FloatingPointError("Nonfinite training loss")
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            steps += 1
            seen_chunks += batch["labels"].shape[0]
            total += output.loss.item() * count
            valid += count
            if steps == 1 or steps % 10 == 0:
                print(f"epoch={epoch + 1} step={steps} loss={output.loss.item():.6f}", flush=True)
            if config.max_steps is not None and steps >= config.max_steps:
                break
        if not valid:
            raise ValueError("No valid training entries were processed")
        val_loss, probabilities = evaluate(model, validation_loader, device)
        history.append({
            "epoch": epoch + 1, "steps": steps, "train_loss": total / valid,
            "validation_loss": val_loss, "skipped_batches": skipped,
        })
        print(json.dumps(history[-1]), flush=True)
        if config.max_steps is not None and steps >= config.max_steps:
            break

    checkpoint = output_dir / "checkpoint"
    model.save_pretrained(checkpoint)
    tokenizer.save(str(checkpoint / "tokenizer.json"))
    # Validate the actual saved checkpoint before declaring the run successful.
    probe = next(iter(validation_loader))
    probe = {key: value.to(device) for key, value in probe.items()}
    model.eval()
    with torch.no_grad():
        expected = model(**probe).logits.cpu()
    del optimizer, model, output
    if device.type == "cuda":
        torch.cuda.empty_cache()
    restored = ContractClassifier.from_pretrained(checkpoint, local_files_only=True).to(device)
    restored.eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(**probe).logits.cpu(), expected)

    labels, mask = label_arrays(validation_rows)
    metrics = score(labels, mask, probabilities, thresholds=0.5)
    metrics.to_csv(output_dir / "validation_metrics.csv", index=False)
    predictions = validation_rows[["contract", "chunk_id"]].copy()
    predictions["probabilities"] = probabilities.tolist()
    predictions.to_parquet(output_dir / "validation_predictions.parquet", index=False)
    assignments = folds.loc[folds.contract.isin(pd.concat([
        train_rows.contract, validation_rows.contract
    ]))].copy()
    assignments["role"] = np.where(
        assignments.fold == config.validation_fold, "validation", "train"
    )
    assignments.to_csv(output_dir / "selected_contracts.csv", index=False)
    metadata = {
        "config": {key: str(value) if isinstance(value, Path) else value
                   for key, value in asdict(config).items()},
        "mode": "tiny-random" if config.tiny else "pretrained-bert",
        "base_model": TOKENIZER_MODEL, "base_revision": TOKENIZER_REVISION,
        "categories": PRIORITY_CATEGORIES, "device": str(device),
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        "labels_sha256": hashlib.sha256(Path(config.labels).read_bytes()).hexdigest(),
        "folds_sha256": hashlib.sha256(Path(config.folds).read_bytes()).hexdigest(),
        "train_contracts": train_rows.contract.nunique(), "train_chunks": len(train),
        "validation_contracts": validation_rows.contract.nunique(),
        "validation_chunks": len(validation), "processed_train_chunks": seen_chunks,
        "optimizer_steps": steps, "history": history, "threshold": 0.5,
        "checkpoint_reload_verified": True,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"PASS: checkpoint reload verified; results saved to {output_dir}", flush=True)
    return metadata


def main():
    defaults = TrainingConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("labels", "folds", "output_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, default=getattr(defaults, name))
    for name in ("validation_fold", "epochs", "batch_size", "seed", "cpu_threads", "max_contracts", "max_steps"):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=getattr(defaults, name))
    for name in ("learning_rate", "weight_decay"):
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=getattr(defaults, name))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--tiny", action="store_true", help="Tiny random BERT for pipeline checks only")
    parser.add_argument("--local-files-only", action="store_true")
    run_training(TrainingConfig(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
