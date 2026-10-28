"""Model selection experiment: python -m scripts.model_selection {count,train} --help.

count  Recount chunk lengths under every candidate tokenizer.
train  Short fine-tuning test of one candidate on a fixed subset of fold-0 training
       contracts, scored on the full validation fold. Official test data is never used.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from src.candidates import (
    CANDIDATES,
    MAX_LENGTH,
    EncoderClassifier,
    length_summary,
    load_backend_tokenizer,
    resolve_revision,
    token_lengths,
)
from src.cuad_data import PRIORITY_CATEGORIES
from src.dataset import ChunkDataset, split_by_fold
from src.evaluation import score, tune_thresholds
from src.labeling import label_arrays
from src.train import evaluate

DEFAULT_LABELS = Path("data/interim/labels/train/labeled_chunks.parquet")
DEFAULT_FOLDS = Path("docs/tuned-baseline-results/folds.csv")


def _fresh_dir(path: Path) -> Path:
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"{path} is not empty; choose a new directory for each run")
    path.mkdir(parents=True, exist_ok=True)
    return path


# --------------------------------------------------------------------- count


def run_count(args) -> None:
    rows = pd.read_parquet(args.labels)
    out = _fresh_dir(args.output_dir)
    positives = np.asarray(rows.labels.tolist()).sum(axis=1)
    lengths = rows[["contract", "chunk_id"]].copy()
    summary, meta = [], {}
    for key, candidate in CANDIDATES.items():
        revision = resolve_revision(candidate, local_files_only=args.local_files_only)
        tokenizer = load_backend_tokenizer(
            candidate, revision, local_files_only=args.local_files_only
        )
        counts = token_lengths(tokenizer, rows.text)
        lengths[f"len_{key}"] = counts
        stats = length_summary(counts, positives)
        summary.append({"candidate": key, **stats})
        meta[key] = {
            "model": candidate.model, "revision": revision,
            "vocab_size": tokenizer.get_vocab_size(),
        }
        print(f"{key}: max={stats['max']} over {MAX_LENGTH}: {stats['over_limit']} "
              f"({stats['over_limit_pct']}%)", flush=True)
    table = pd.DataFrame(summary)
    base = table.loc[table.candidate == "bert", "total_tokens"].iloc[0]
    table["tokens_vs_bert"] = (table.total_tokens / base).round(4)
    table.to_csv(out / "tokenizer_summary.csv", index=False)
    lengths.to_parquet(out / "chunk_lengths.parquet", index=False)
    (out / "tokenizer_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(table.to_string(index=False))


# --------------------------------------------------------------------- train


def select_subset(train_rows, count, seed):
    """Whole contracts, chosen from identifiers only, independent of the run seed."""
    names = sorted(train_rows.contract.unique())
    if count is None or count >= len(names):
        return train_rows
    keep = np.random.default_rng(seed).permutation(names)[:count]
    return train_rows.loc[train_rows.contract.isin(keep)].reset_index(drop=True)


def run_train(args) -> dict:
    candidate = CANDIDATES[args.candidate]
    out = _fresh_dir(args.output_dir / f"{args.candidate}-seed{args.seed}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = pd.read_parquet(args.labels)
    folds = pd.read_csv(args.folds)
    train_rows, validation_rows = split_by_fold(rows, folds, validation_fold=args.validation_fold)
    train_rows = select_subset(train_rows, args.train_contracts, args.subset_seed)

    # Same chunks for every candidate: keep only chunks that fit 512 tokens under ALL
    # tokenizers, so no model is advantaged or hurt by chunks another cannot take.
    lengths = pd.read_parquet(args.lengths).set_index("chunk_id")
    needed = pd.concat([train_rows.chunk_id, validation_rows.chunk_id])
    if not needed.isin(lengths.index).all():
        raise ValueError("--lengths is stale: rerun the count command on these labels")
    length_cols = [c for c in lengths.columns if c.startswith("len_")]
    fits = (lengths[length_cols].max(axis=1) <= MAX_LENGTH)
    dropped = {
        "train": int((~train_rows.chunk_id.map(fits)).sum()),
        "validation": int((~validation_rows.chunk_id.map(fits)).sum()),
    }
    train_rows = train_rows.loc[train_rows.chunk_id.map(fits)].reset_index(drop=True)
    validation_rows = validation_rows.loc[
        validation_rows.chunk_id.map(fits)
    ].reset_index(drop=True)

    revision = resolve_revision(candidate, local_files_only=args.local_files_only)
    tokenizer = load_backend_tokenizer(candidate, revision, local_files_only=args.local_files_only)
    train = ChunkDataset(train_rows, tokenizer)
    validation = ChunkDataset(validation_rows, tokenizer)
    if not train.label_mask.any() or not validation.label_mask.any():
        raise ValueError("Training and validation both need valid labels")
    train_loader = DataLoader(
        train, batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    validation_loader = DataLoader(validation, batch_size=args.batch_size * 2, shuffle=False)

    model = EncoderClassifier.from_pretrained(
        candidate, revision, local_files_only=args.local_files_only
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    total_steps = args.epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(args.warmup_ratio * total_steps), total_steps
    )
    print(f"{args.candidate} on {device}: {train_rows.contract.nunique()} train contracts/"
          f"{len(train)} chunks; {validation_rows.contract.nunique()} validation contracts/"
          f"{len(validation)} chunks; dropped for length {dropped}", flush=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    history, steps = [], 0
    for epoch in range(args.epochs):
        model.train()
        total, valid = 0.0, 0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            count = int(batch["label_mask"].sum())
            if not count:
                continue  # an all-masked batch must not trigger AdamW weight decay
            optimizer.zero_grad(set_to_none=True)
            output = model(**batch)
            if not torch.isfinite(output.loss):
                raise FloatingPointError("Nonfinite training loss")
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            steps += 1
            total += output.loss.item() * count
            valid += count
            if steps == 1 or steps % 25 == 0:
                print(f"epoch={epoch + 1} step={steps}/{total_steps} "
                      f"loss={output.loss.item():.5f}", flush=True)
        history.append({"epoch": epoch + 1, "steps": steps, "train_loss": total / max(valid, 1)})
    if device.type == "cuda":
        torch.cuda.synchronize()
    train_seconds = time.perf_counter() - started
    peak_gb = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else None

    started = time.perf_counter()
    validation_loss, probabilities = evaluate(model, validation_loader, device)
    eval_seconds = time.perf_counter() - started

    labels, mask = label_arrays(validation_rows)
    table = score(labels, mask, probabilities, thresholds=0.5).set_index("category")
    tuned = tune_thresholds(labels, mask, probabilities, PRIORITY_CATEGORIES)  # tuned ON validation: optimistic
    table.reset_index().to_csv(out / "per_category.csv", index=False)
    category_rows = table.drop(index=["macro avg", "micro avg"])
    predictions = validation_rows[["contract", "chunk_id"]].copy()
    predictions["probabilities"] = probabilities.tolist()
    predictions.to_parquet(out / "validation_predictions.parquet", index=False)

    result = {
        "candidate": args.candidate, "model": candidate.model, "revision": revision,
        "seed": args.seed, "subset_seed": args.subset_seed, "device": str(device),
        "parameters_millions": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
        "train_contracts": train_rows.contract.nunique(), "train_chunks": len(train),
        "validation_contracts": validation_rows.contract.nunique(),
        "validation_chunks": len(validation), "dropped_for_length": dropped,
        "epochs": args.epochs, "batch_size": args.batch_size,
        "learning_rate": args.learning_rate, "optimizer_steps": steps,
        "train_seconds": round(train_seconds, 1),
        "seconds_per_step": round(train_seconds / max(steps, 1), 3),
        "eval_seconds": round(eval_seconds, 1),
        "peak_gpu_gib": None if peak_gb is None else round(peak_gb, 2),
        "history": history, "validation_loss": validation_loss,
        "macro_average_precision": float(table.loc["macro avg", "average_precision"]),
        "macro_f1_at_0.5": float(table.loc["macro avg", "f1"]),
        "micro_f1_at_0.5": float(table.loc["micro avg", "f1"]),
        "macro_f1_best_threshold_on_validation": float(np.nanmean(tuned.f1)),
        "categories_with_positives": int((category_rows.support > 0).sum()),
    }
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2), flush=True)
    return result


# ----------------------------------------------------------------------- CLI


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    count = sub.add_parser("count", help="recount chunk lengths for all candidates")
    train = sub.add_parser("train", help="short fine-tuning test of one candidate")
    for p in (count, train):
        p.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
        p.add_argument("--output-dir", type=Path, required=True)
        p.add_argument("--local-files-only", action="store_true")
    train.add_argument("--candidate", choices=sorted(CANDIDATES), required=True)
    train.add_argument("--lengths", type=Path, required=True,
                       help="chunk_lengths.parquet written by the count command")
    train.add_argument("--folds", type=Path, default=DEFAULT_FOLDS)
    train.add_argument("--validation-fold", type=int, default=0)
    train.add_argument("--train-contracts", type=int, default=60,
                       help="training contracts to use; 0 means all in the fold")
    train.add_argument("--subset-seed", type=int, default=42)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--epochs", type=int, default=2)
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--learning-rate", type=float, default=2e-5)
    train.add_argument("--weight-decay", type=float, default=0.01)
    train.add_argument("--warmup-ratio", type=float, default=0.1)
    args = parser.parse_args()
    if args.command == "count":
        run_count(args)
    else:
        args.train_contracts = args.train_contracts or None
        run_train(args)


if __name__ == "__main__":
    main()
