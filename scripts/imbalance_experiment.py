"""Class-imbalance experiment for task #12: python -m scripts.imbalance_experiment --help.

prepare    Re-chunk and label the training split with legal-bert's tokenizer, into their
           own directories. Shared defaults and the baselines' chunks are unchanged.
train      Train one arm on one fold and save its held-out predictions. A finished run
           is skipped, so a notebook can resume after Colab disconnects.
summarize  Compare the arms on every fold they have all finished, and apply the decision
           rule in src.imbalance.choose_arm.

Official test data is never used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from torch.utils.data import DataLoader, Subset
from transformers import get_linear_schedule_with_warmup

from scripts.export_chunks import export_split
from scripts.label_chunks import run as label_split
from scripts.model_selection import select_subset
from src.candidates import (
    CANDIDATES,
    MAX_LENGTH,
    EncoderClassifier,
    load_backend_tokenizer,
    token_lengths,
)
from src.cuad_data import DATA_DIR, PRIORITY_CATEGORIES
from src.dataset import ChunkDataset, split_by_fold
from src.evaluation import fold_of_rows, score, threshold_map, tune_thresholds
from src.imbalance import (
    ARM_THRESHOLD_GRID,
    ARMS,
    CONTROL,
    FIXES,
    WEAK_CATEGORIES,
    choose_arm,
    cross_fit_decisions,
    epoch_rows,
    make_loss,
    paired_bootstrap,
    score_decisions,
)
from src.labeling import label_arrays
from src.train import evaluate

ENCODER = "legal-bert"
WORK_DIR = DATA_DIR / "interim" / "imbalance"
DEFAULT_LABELS = WORK_DIR / "labels" / "train" / "labeled_chunks.parquet"
DEFAULT_FOLDS = Path("docs/tuned-baseline-results/folds.csv")
BASELINE_CV = Path("docs/tuned-baseline-results/train_cv_tuned.csv")


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ------------------------------------------------------------------- prepare


def run_prepare(args) -> dict:
    candidate = CANDIDATES[ENCODER]
    tokenizer = load_backend_tokenizer(
        candidate, candidate.revision, local_files_only=args.local_files_only
    )
    chunks = export_split(
        "train", args.work_dir / "chunks", tokenizer=tokenizer,
        tokenizer_name=candidate.model, tokenizer_revision=candidate.revision,
    )
    report = label_split("train", args.work_dir / "chunks", args.work_dir / "labels")
    rows = pd.read_parquet(args.work_dir / "labels" / "train" / "labeled_chunks.parquet")
    longest = int(token_lengths(tokenizer, rows.text).max())
    if longest > MAX_LENGTH:
        raise ValueError(f"A chunk has {longest} tokens with special tokens; limit {MAX_LENGTH}")
    # Saved folds assign contracts, not chunks, so they still apply after re-chunking.
    fold_of_rows(rows, pd.read_csv(args.folds))
    result = {
        "encoder": candidate.model, "revision": candidate.revision,
        "chunks": len(rows), "chunks_sha256": chunks["chunks_sha256"],
        "contracts": int(rows.contract.nunique()), "longest_chunk_tokens": longest,
        "priority_spans": report["priority_spans"],
        "fully_contained_spans": report["fully_contained_spans"],
        "chunks_with_any_positive": report["chunks_with_any_positive"],
        "per_category": report["per_category"],
        "labels_sha256": report["labeled_chunks_sha256"],
    }
    print(json.dumps(result, indent=2))
    return result


# --------------------------------------------------------------------- train


def run_name(arm: str, fold: int, seed: int) -> str:
    return f"{arm}-fold{fold}-seed{seed}"


def run_train(args, *, tokenizer=None, model=None) -> dict:
    """Train one arm on one fold. ``tokenizer`` and ``model`` are for offline tests."""
    arm = ARMS[args.arm]
    out = args.output_dir / run_name(args.arm, args.validation_fold, args.seed)
    if (out / "result.json").exists():
        print(f"SKIP: {out.name} already finished", flush=True)
        return json.loads((out / "result.json").read_text())
    if out.exists():  # interrupted before result.json: nothing in it can be trusted
        shutil.rmtree(out)
    out.mkdir(parents=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = pd.read_parquet(args.labels)
    folds = pd.read_csv(args.folds)
    train_rows, validation_rows = split_by_fold(rows, folds, validation_fold=args.validation_fold)
    train_rows = select_subset(train_rows, args.train_contracts, args.subset_seed)
    validation_rows = select_subset(validation_rows, args.validation_contracts, args.subset_seed)
    smoke = bool(args.train_contracts or args.validation_contracts or args.max_steps)

    candidate = CANDIDATES[ENCODER]
    if tokenizer is None:
        tokenizer = load_backend_tokenizer(
            candidate, candidate.revision, local_files_only=args.local_files_only
        )
    train = ChunkDataset(train_rows, tokenizer)
    validation = ChunkDataset(validation_rows, tokenizer)
    labels, mask = label_arrays(train_rows)
    if model is None:
        model = EncoderClassifier.from_pretrained(
            candidate, candidate.revision, local_files_only=args.local_files_only
        )
    model.to(device)
    loss_fn = make_loss(arm, labels, mask, device)
    validation_loader = DataLoader(validation, batch_size=args.batch_size * 2, shuffle=False)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    # Every epoch of an arm has the same size, so the schedule is known in advance.
    epoch_size = len(epoch_rows(labels, arm, args.seed, 0))
    total_steps = args.epochs * math.ceil(epoch_size / args.batch_size)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(args.warmup_ratio * total_steps), total_steps
    )
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    print(f"{out.name} on {device}: {train_rows.contract.nunique()} train contracts, "
          f"{epoch_size}/{len(train)} chunks per epoch; {validation_rows.contract.nunique()} "
          f"validation contracts, {len(validation)} chunks; {total_steps} steps", flush=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    history, steps, skipped_steps = [], 0, 0
    for epoch in range(args.epochs):
        model.train()
        chosen = epoch_rows(labels, arm, args.seed, epoch).tolist()
        loader = DataLoader(
            Subset(train, chosen), batch_size=args.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(args.seed + epoch),
        )
        total, valid = 0.0, 0
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            count = int(batch["label_mask"].sum())
            if not count:
                continue  # an all-masked batch must not trigger AdamW weight decay
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                logits = model(
                    input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
                ).logits
                loss = loss_fn(logits, batch["labels"], batch["label_mask"])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # With AMP the scaler itself skips steps with nonfinite gradients and lowers
            # the scale, which is normal early on; too many skips fail the run below.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=not amp)
            scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped_steps += scaler.get_scale() < scale
            scheduler.step()
            steps += 1
            total += loss.item() * count
            valid += count
            if steps == 1 or steps % 100 == 0:
                print(f"epoch={epoch + 1} step={steps}/{total_steps} loss={loss.item():.5f}",
                      flush=True)
            if steps >= total_steps:
                break
        history.append({"epoch": epoch + 1, "steps": steps, "train_loss": total / max(valid, 1)})
        if steps >= total_steps:
            break
    if skipped_steps > max(10, steps / 10):
        raise FloatingPointError(f"AMP skipped {skipped_steps} of {steps} steps")
    train_seconds = time.perf_counter() - started
    peak_gb = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else None

    # The model's own loss is always plain BCE, so this is comparable across arms.
    validation_bce, probabilities = evaluate(model, validation_loader, device)
    validation_labels, validation_mask = label_arrays(validation_rows)
    table = score(validation_labels, validation_mask, probabilities, thresholds=0.5)
    table.to_csv(out / "per_category_at_0.5.csv", index=False)
    predictions = validation_rows[["contract", "chunk_id"]].copy()
    predictions["probabilities"] = probabilities.tolist()
    predictions.to_parquet(out / "validation_predictions.parquet", index=False)

    macro = table.set_index("category").loc["macro avg"]
    result = {
        "arm": args.arm, "validation_fold": args.validation_fold, "seed": args.seed,
        "smoke": smoke, "encoder": candidate.model, "revision": candidate.revision,
        "device": str(device), "mixed_precision": amp,
        "epochs": args.epochs, "batch_size": args.batch_size,
        "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "train_contracts": int(train_rows.contract.nunique()), "train_chunks": len(train),
        "chunks_per_epoch": epoch_size,
        "validation_contracts": int(validation_rows.contract.nunique()),
        "validation_chunks": len(validation),
        "optimizer_steps": steps, "amp_skipped_steps": int(skipped_steps),
        "train_seconds": round(train_seconds, 1),
        "seconds_per_chunk": round(train_seconds / max(steps * args.batch_size, 1), 4),
        "peak_gpu_gib": None if peak_gb is None else round(peak_gb, 2),
        "history": history, "validation_bce": validation_bce,
        "macro_average_precision": float(macro.average_precision),
        "macro_f1_at_0.5": float(macro.f1),
        "labels_sha256": sha256(args.labels), "folds_sha256": sha256(args.folds),
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
    }
    # Written last and moved into place: its presence is what marks the run finished.
    temporary = out / "result.json.tmp"
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(out / "result.json")
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2), flush=True)
    return result


# ----------------------------------------------------------------- summarize


def load_runs(runs_dirs: list[Path], seed: int) -> dict[tuple[str, int], tuple[dict, pd.DataFrame]]:
    """Read finished runs from one or more directories (e.g. several Colab accounts)."""
    runs = {}
    for directory in runs_dirs:
        for path in sorted(Path(directory).glob(f"*-seed{seed}/result.json")):
            result = json.loads(path.read_text())
            if result["smoke"]:
                raise ValueError(f"{path} is a smoke run and cannot be compared")
            key = (result["arm"], result["validation_fold"])
            if key in runs:
                raise ValueError(f"Two runs for {key}; keep one of them")
            runs[key] = (result, pd.read_parquet(path.parent / "validation_predictions.parquet"))
    if not runs:
        raise ValueError("No finished runs were found")
    return runs


def run_summarize(args) -> dict:
    runs = load_runs(args.runs_dir, args.seed)
    rows = pd.read_parquet(args.labels)
    folds = pd.read_csv(args.folds)
    expected_labels = sha256(args.labels)
    for key, (result, _) in runs.items():
        if result["labels_sha256"] != expected_labels or result["folds_sha256"] != sha256(args.folds):
            raise ValueError(f"Run {key} used different labels or folds than this summary")

    arms = list(ARMS)
    finished = {fold for _, fold in runs}
    complete = sorted(f for f in finished if all((arm, f) in runs for arm in arms))
    ignored = sorted(key for key in runs if key[1] not in complete)
    if len(complete) < 2:
        raise ValueError(f"Need at least two folds finished by every arm; have {complete}")

    # Rows of the finished folds, in a fixed order that every arm is aligned to.
    fold_of = fold_of_rows(rows, folds)
    keep = np.isin(fold_of, complete)
    rows, fold_of = rows.loc[keep].reset_index(drop=True), fold_of[keep]
    labels, mask = label_arrays(rows)
    contracts = rows.contract.to_numpy()
    positions = pd.Series(np.arange(len(rows)), index=rows.chunk_id)

    summary, per_category, per_fold, threshold_tables, decisions = [], [], [], [], {}
    categories = list(PRIORITY_CATEGORIES)
    for arm in arms:
        proba = np.full(labels.shape, np.nan)
        for fold in complete:
            predictions = runs[(arm, fold)][1]
            held = positions.loc[predictions.chunk_id].to_numpy()
            if not np.array_equal(np.sort(held), np.flatnonzero(fold_of == fold)):
                raise ValueError(f"{arm} fold {fold} does not cover exactly that fold's chunks")
            proba[held] = np.array(predictions.probabilities.tolist())
        decided, thresholds = cross_fit_decisions(labels, mask, proba, fold_of)
        decisions[arm] = decided
        table = score_decisions(labels, mask, decided, proba)
        at_half = score(labels, mask, proba, thresholds=0.5)
        # Tuned and scored on the same rows, as task #9 did. Optimistic: only for a rough
        # comparison with that baseline, never for choosing an arm.
        tuned = tune_thresholds(labels, mask, proba, grid=ARM_THRESHOLD_GRID)
        in_sample = score(labels, mask, proba, thresholds=threshold_map(tuned))
        table.insert(0, "arm", arm)
        table["f1_at_0.5"] = at_half.f1
        table["f1_thresholds_tuned_in_sample"] = in_sample.f1
        table["mean_threshold"] = thresholds[categories].mean().reindex(table.category).to_numpy()
        per_category.append(table)
        thresholds.insert(0, "arm", arm)
        threshold_tables.append(thresholds)

        by_name = table.set_index("category")
        for fold in complete:
            held = fold_of == fold
            fold_table = score_decisions(labels[held], mask[held], decided[held], proba[held])
            f1 = fold_table.set_index("category").f1
            per_fold.append({
                "arm": arm, "fold": fold, "macro_f1": f1["macro avg"],
                "weak_f1": f1[list(WEAK_CATEGORIES)].mean(),
                "macro_average_precision":
                    fold_table.set_index("category").average_precision["macro avg"],
            })
        results = [runs[(arm, fold)][0] for fold in complete]
        summary.append({
            "arm": arm,
            "macro_f1": by_name.f1["macro avg"],
            "weak_f1": by_name.f1[list(WEAK_CATEGORIES)].mean(),
            "macro_precision": by_name.precision["macro avg"],
            "macro_recall": by_name.recall["macro avg"],
            "macro_average_precision": by_name.average_precision["macro avg"],
            "macro_f1_at_0.5": by_name["f1_at_0.5"]["macro avg"],
            "macro_f1_thresholds_tuned_in_sample":
                by_name["f1_thresholds_tuned_in_sample"]["macro avg"],
            "validation_bce_mean": float(np.mean([r["validation_bce"] for r in results])),
            "train_minutes_mean": round(float(np.mean([r["train_seconds"] for r in results])) / 60, 1),
        })
    summary = pd.DataFrame(summary)
    per_fold = pd.DataFrame(per_fold)
    spread = per_fold.groupby("arm")[["macro_f1", "weak_f1"]].std()
    summary["macro_f1_fold_std"] = summary.arm.map(spread.macro_f1)
    summary["weak_f1_fold_std"] = summary.arm.map(spread.weak_f1)
    pairs = paired_bootstrap(labels, mask, decisions, contracts, samples=args.bootstrap)
    decision = choose_arm(summary, pairs)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.round(4).to_csv(args.output_dir / "arm_summary.csv", index=False)
    pd.concat(per_category).round(4).to_csv(args.output_dir / "per_category.csv", index=False)
    per_fold.round(4).to_csv(args.output_dir / "per_fold.csv", index=False)
    pd.concat(threshold_tables).to_csv(args.output_dir / "thresholds.csv", index=False)
    pairs.round(4).to_csv(args.output_dir / "pairwise_bootstrap.csv", index=False)
    any_result = next(iter(runs.values()))[0]
    metadata = {
        "task": 12, "decision": decision,
        "folds_compared": complete, "folds_expected": sorted(folds.fold.unique().tolist()),
        "runs_ignored_incomplete_fold": [list(key) for key in ignored],
        "seed": args.seed, "bootstrap_samples": args.bootstrap,
        "fixes": list(FIXES), "control": CONTROL, "weak_categories": list(WEAK_CATEGORIES),
        "contracts": int(rows.contract.nunique()), "chunks": len(rows),
        "encoder": any_result["encoder"], "revision": any_result["revision"],
        "epochs": any_result["epochs"], "batch_size": any_result["batch_size"],
        "learning_rate": any_result["learning_rate"],
        "labels_sha256": expected_labels, "folds_sha256": sha256(args.folds),
        "official_test_split_used": False,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(summary.round(4).to_string(index=False))
    print(pairs.round(4).to_string(index=False))
    print(json.dumps(decision, indent=2))
    return metadata


# ----------------------------------------------------------------------- CLI


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="re-chunk and label with legal-bert's tokenizer")
    prepare.add_argument("--work-dir", type=Path, default=WORK_DIR)
    prepare.add_argument("--folds", type=Path, default=DEFAULT_FOLDS)
    prepare.add_argument("--local-files-only", action="store_true")

    train = sub.add_parser("train", help="train one arm on one fold")
    train.add_argument("--arm", choices=sorted(ARMS), required=True)
    train.add_argument("--validation-fold", type=int, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    train.add_argument("--folds", type=Path, default=DEFAULT_FOLDS)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--subset-seed", type=int, default=42)
    train.add_argument("--epochs", type=int, default=2)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--learning-rate", type=float, default=2e-5)
    train.add_argument("--weight-decay", type=float, default=0.01)
    train.add_argument("--warmup-ratio", type=float, default=0.1)
    train.add_argument("--local-files-only", action="store_true")
    # Any of these marks the run as a smoke test, which summarize refuses to compare.
    train.add_argument("--train-contracts", type=int, default=None)
    train.add_argument("--validation-contracts", type=int, default=None)
    train.add_argument("--max-steps", type=int, default=None)

    summarize = sub.add_parser("summarize", help="compare finished runs and pick an arm")
    summarize.add_argument("--runs-dir", type=Path, nargs="+", required=True)
    summarize.add_argument("--output-dir", type=Path, required=True)
    summarize.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    summarize.add_argument("--folds", type=Path, default=DEFAULT_FOLDS)
    summarize.add_argument("--seed", type=int, default=42)
    summarize.add_argument("--bootstrap", type=int, default=2000)

    args = parser.parse_args()
    {"prepare": run_prepare, "train": run_train, "summarize": run_summarize}[args.command](args)


if __name__ == "__main__":
    main()
