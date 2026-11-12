"""Task #13 final training: python -m scripts.final_training --help.

prepare  Re-chunk + label train AND test with legal-bert's tokenizer (data/interim/final/).
sweep    Train one (learning rate, fold) with pos_weight; save held-out predictions after
         every epoch. Finished runs are skipped, so Colab can resume after a disconnect.
select   Score every (learning rate, epochs) config on 5-fold CV, pick one with the fixed
         rule, tune per-category thresholds on its out-of-fold predictions.
final    Train on all 408 training contracts with the picked config; save the checkpoint.
test     Score the official test split ONCE at the frozen thresholds. Locked after one run.

Only `test` touches the test split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers

from scripts.export_chunks import export_split
from scripts.label_chunks import run as label_split
from scripts.model_selection import select_subset
from src.candidates import CANDIDATES, MAX_LENGTH, load_backend_tokenizer, token_lengths
from src.cuad_data import DATA_DIR, PRIORITY_CATEGORIES
from src.dataset import split_by_fold
from src.evaluation import fold_of_rows
from src.final_training import (
    ARM,
    BATCH_SIZE,
    ENCODER,
    MAX_EPOCHS,
    SEED,
    bootstrap_pairs,
    config_name,
    final_thresholds,
    load_checkpoint,
    new_model,
    oof_matrix,
    predict,
    save_checkpoint,
    score_at,
    select_config,
    sweep_table,
    threshold_map,
    train_encoder,
)
from src.labeling import label_arrays

WORK_DIR = DATA_DIR / "interim" / "final"
FOLDS = Path("docs/tuned-baseline-results/folds.csv")
RESULTS = Path("docs/final-model-results")
# task #12 trained on these exact legal-bert training labels
TASK12_LABELS_SHA256 = "3c48a0174374c23b590559669bf57efa7f95dba373a6888b9b64084a22525b4e"


def labels_path(work_dir: Path, split: str) -> Path:
    return Path(work_dir) / "labels" / split / "labeled_chunks.parquet"


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tokenizer_for(args, tokenizer=None):
    if tokenizer is not None:
        return tokenizer
    candidate = CANDIDATES[ENCODER]
    return load_backend_tokenizer(
        candidate, candidate.revision, local_files_only=args.local_files_only
    )


def write_json(path: Path, data: dict) -> None:
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str) + "\n")
    tmp.replace(path)  # atomic: presence = finished


# ------------------------------------------------------------------ prepare


def run_prepare(args, *, tokenizer=None) -> dict:
    tokenizer = tokenizer_for(args, tokenizer)
    candidate = CANDIDATES[ENCODER]
    out = {}
    for split in ("train", "test"):
        export_split(
            split, args.work_dir / "chunks", tokenizer=tokenizer, raw_dir=args.raw_dir,
            tokenizer_name=candidate.model, tokenizer_revision=candidate.revision,
        )
        report = label_split(split, args.work_dir / "chunks", args.work_dir / "labels",
                             raw_dir=args.raw_dir)
        rows = pd.read_parquet(labels_path(args.work_dir, split))
        longest = int(token_lengths(tokenizer, rows.text).max())
        if longest > MAX_LENGTH:
            raise ValueError(f"{split}: a chunk has {longest} tokens; limit {MAX_LENGTH}")
        out[split] = {
            "contracts": int(rows.contract.nunique()), "chunks": len(rows),
            "longest_chunk_tokens": longest,
            "priority_spans": report["priority_spans"],
            "fully_contained_spans": report["fully_contained_spans"],
            "labels_sha256": sha256(labels_path(args.work_dir, split)),
        }
    train_rows = pd.read_parquet(labels_path(args.work_dir, "train"))
    fold_of_rows(train_rows, pd.read_csv(args.folds))  # every contract has a fold
    out["train_matches_task12"] = out["train"]["labels_sha256"] == TASK12_LABELS_SHA256
    if not out["train_matches_task12"]:
        print("WARNING: training labels differ from task #12's; check the tokenizer revision")
    print(json.dumps(out, indent=2))
    return out


# -------------------------------------------------------------------- sweep


def run_name(learning_rate: float, fold: int, seed: int) -> str:
    return f"lr{learning_rate:g}-fold{fold}-seed{seed}"


def run_sweep(args, *, tokenizer=None, model=None) -> dict:
    out = args.output_dir / run_name(args.learning_rate, args.validation_fold, args.seed)
    if (out / "result.json").exists():
        print(f"SKIP: {out.name} already finished", flush=True)
        return json.loads((out / "result.json").read_text())
    if out.exists():
        shutil.rmtree(out)  # interrupted run: start clean
    out.mkdir(parents=True)

    rows = pd.read_parquet(args.labels)
    train_rows, val_rows = split_by_fold(
        rows, pd.read_csv(args.folds), validation_fold=args.validation_fold
    )
    train_rows = select_subset(train_rows, args.train_contracts, args.seed)
    val_rows = select_subset(val_rows, args.validation_contracts, args.seed)
    smoke = bool(args.train_contracts or args.validation_contracts or args.max_steps)
    tokenizer = tokenizer_for(args, tokenizer)
    model = model or new_model(local_files_only=args.local_files_only)
    epoch_scores = []

    def save_epoch(epoch, trained):
        bce, proba = predict(trained, val_rows, tokenizer, batch_size=args.batch_size * 2)
        table = val_rows[["contract", "chunk_id"]].copy()
        table["probabilities"] = proba.tolist()
        table.to_parquet(out / f"epoch{epoch}_predictions.parquet", index=False)
        macro = score_at(val_rows, proba, 0.5).set_index("category").loc["macro avg"]
        epoch_scores.append({"epoch": epoch, "validation_bce": bce,
                             "macro_average_precision": float(macro.average_precision)})
        print(json.dumps(epoch_scores[-1]), flush=True)

    started = time.perf_counter()
    trained = train_encoder(
        model, train_rows, tokenizer, epochs=args.epochs, schedule_epochs=args.epochs,
        learning_rate=args.learning_rate, batch_size=args.batch_size, seed=args.seed,
        max_steps=args.max_steps, on_epoch_end=save_epoch,
    )
    result = {
        "learning_rate": args.learning_rate, "validation_fold": args.validation_fold,
        "seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size,
        "smoke": smoke, "encoder": CANDIDATES[ENCODER].model,
        "revision": CANDIDATES[ENCODER].revision, "arm": ARM,
        "train_contracts": int(train_rows.contract.nunique()),
        "validation_contracts": int(val_rows.contract.nunique()),
        "validation_chunks": len(val_rows), "epoch_scores": epoch_scores,
        "train_minutes": round((time.perf_counter() - started) / 60, 1),
        "labels_sha256": sha256(args.labels), "folds_sha256": sha256(args.folds),
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        **trained,
    }
    write_json(out / "result.json", result)
    print(f"DONE: {out.name} in {result['train_minutes']} min", flush=True)
    return result


# ------------------------------------------------------------------- select


def load_sweep(runs_dirs, seed, labels_sha, folds_sha):
    runs = {}
    for directory in runs_dirs:
        for path in sorted(Path(directory).glob(f"lr*-fold*-seed{seed}/result.json")):
            result = json.loads(path.read_text())
            if result["smoke"]:
                raise ValueError(f"{path} is a smoke run; keep smoke runs in another folder")
            if result["labels_sha256"] != labels_sha or result["folds_sha256"] != folds_sha:
                raise ValueError(f"{path} used different labels or folds")
            key = (result["learning_rate"], result["validation_fold"])
            if key in runs:
                raise ValueError(f"Two runs for lr={key[0]} fold={key[1]}; keep one")
            runs[key] = (result, path.parent)
    if not runs:
        raise ValueError("No finished sweep runs found")
    return runs


def run_select(args) -> dict:
    rows = pd.read_parquet(args.labels)
    folds = pd.read_csv(args.folds)
    fold_of = fold_of_rows(rows, folds)
    all_folds = sorted(np.unique(fold_of).tolist())
    runs = load_sweep(args.runs_dir, args.seed, sha256(args.labels), sha256(args.folds))

    configs, missing = {}, []
    for lr in sorted({lr for lr, _ in runs}):
        if any((lr, f) not in runs for f in all_folds):
            missing.append(lr)  # thresholds need out-of-fold scores for every chunk
            continue
        epochs = min(runs[(lr, f)][0]["epochs"] for f in all_folds)
        for ep in range(1, epochs + 1):
            preds = {f: pd.read_parquet(runs[(lr, f)][1] / f"epoch{ep}_predictions.parquet")
                     for f in all_folds}
            configs[config_name(lr, ep)] = oof_matrix(rows, fold_of, preds)
    if not configs:
        raise ValueError(f"No learning rate has all {len(all_folds)} folds; missing {missing}")

    summary, decisions, per_category = sweep_table(rows, fold_of, configs)
    pairs = bootstrap_pairs(rows, decisions, samples=args.bootstrap, seed=args.seed)
    decision = select_config(summary, pairs)
    thresholds = final_thresholds(rows, fold_of, configs[decision["pick"]])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rounded = summary.round({c: 4 for c in summary.columns if c != "learning_rate"})
    rounded.to_csv(args.output_dir / "sweep_summary.csv", index=False)  # keep lr unrounded
    pairs.round(4).to_csv(args.output_dir / "sweep_pairwise_bootstrap.csv", index=False)
    per_category[decision["pick"]].round(4).reset_index().to_csv(
        args.output_dir / "cv_per_category.csv", index=False)
    thresholds.round(4).to_csv(args.output_dir / "thresholds.csv", index=False)
    oof = rows[["contract", "chunk_id"]].copy()
    oof["probabilities"] = configs[decision["pick"]].tolist()
    oof.to_parquet(args.runs_dir[0] / "oof_predictions_selected.parquet", index=False)
    selection = {
        "task": 13, "decision": decision, "arm": ARM,
        "encoder": CANDIDATES[ENCODER].model, "revision": CANDIDATES[ENCODER].revision,
        "batch_size": next(iter(runs.values()))[0]["batch_size"],
        "schedule_epochs": max(r["epochs"] for r, _ in runs.values()),
        "seed": args.seed, "folds": all_folds,
        "learning_rates_compared": sorted({float(c.removeprefix("lr").split("-ep")[0])
                                           for c in configs}),
        "learning_rates_incomplete": missing,
        "thresholds": threshold_map(thresholds),
        "flagged_categories": {r.category: r.flag for r in thresholds.itertuples() if r.flag},
        "labels_sha256": sha256(args.labels), "folds_sha256": sha256(args.folds),
        "official_test_split_used": False,
    }
    write_json(args.output_dir / "selection.json", selection)
    print(rounded.to_string(index=False))
    print(json.dumps(decision, indent=2))
    print(thresholds[["category", "threshold", "f1", "fold_threshold_std", "flag"]].to_string(index=False))
    return selection


# -------------------------------------------------------------------- final


def run_final(args, *, tokenizer=None, model=None) -> dict:
    selection = json.loads(args.selection.read_text())
    if selection["labels_sha256"] != sha256(args.labels):
        raise ValueError("Training labels differ from the ones the selection used")
    out = args.output_dir
    if (out / "result.json").exists():
        print(f"SKIP: final model already trained in {out}", flush=True)
        return json.loads((out / "result.json").read_text())
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    rows = pd.read_parquet(args.labels)
    rows = select_subset(rows, args.train_contracts, selection["seed"])
    smoke = bool(args.train_contracts or args.max_steps)
    tokenizer = tokenizer_for(args, tokenizer)
    model = model or new_model(local_files_only=args.local_files_only)
    decision = selection["decision"]
    started = time.perf_counter()
    trained = train_encoder(
        model, rows, tokenizer, epochs=decision["epochs"],
        schedule_epochs=selection["schedule_epochs"],  # same schedule the sweep scored
        learning_rate=decision["learning_rate"], batch_size=selection["batch_size"],
        seed=selection["seed"], max_steps=args.max_steps,
    )
    info = {"thresholds": selection["thresholds"], "learning_rate": decision["learning_rate"],
            "epochs": decision["epochs"], "schedule_epochs": selection["schedule_epochs"],
            "encoder": selection["encoder"], "revision": selection["revision"], "smoke": smoke}
    save_checkpoint(model, tokenizer, out / "checkpoint", info)
    reloaded = load_checkpoint(out / "checkpoint")  # check the saved files reproduce the model
    probe = rows.head(4)
    _, before = predict(model, probe, tokenizer)
    _, after = predict(reloaded, probe, tokenizer)
    if not np.allclose(before, after, atol=1e-4):
        raise ValueError("Reloaded checkpoint gives different predictions")
    result = {
        **info, "train_contracts": int(rows.contract.nunique()),
        "train_minutes": round((time.perf_counter() - started) / 60, 1),
        "labels_sha256": sha256(args.labels), "checkpoint_reload_verified": True,
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        **trained,
    }
    write_json(out / "result.json", result)
    print(f"DONE: final model saved to {out / 'checkpoint'}", flush=True)
    return result


# --------------------------------------------------------------------- test


def run_test(args, *, tokenizer=None) -> dict:
    lock = args.results_dir / "test_run.json"
    if lock.exists() and not args.allow_second_test_run:
        raise SystemExit(
            f"The test split was already scored ({lock}). One test run only: report that one."
        )
    final = json.loads((args.checkpoint.parent / "result.json").read_text())
    selection = json.loads(args.selection.read_text())
    smoke = bool(final.get("smoke") or args.test_contracts)
    if smoke and not args.smoke_ok:
        raise SystemExit("Smoke checkpoint or test subset: not reportable (pass --smoke-ok)")
    if smoke and Path(args.results_dir).resolve() == RESULTS.resolve():
        raise SystemExit("Smoke test runs must not write to docs/final-model-results")
    if final["thresholds"] != selection["thresholds"]:
        raise ValueError("Checkpoint thresholds differ from selection.json")

    rows = pd.read_parquet(args.labels)
    if not rows.split.eq("test").all():
        raise ValueError("--labels must be the test split")
    rows = select_subset(rows, args.test_contracts, SEED)
    model = load_checkpoint(args.checkpoint)
    tokenizer = tokenizer_for(args, tokenizer)
    bce, proba = predict(model, rows, tokenizer, batch_size=args.batch_size)
    thresholds = selection["thresholds"]
    tuned = score_at(rows, proba, thresholds)
    at_half = score_at(rows, proba, 0.5)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tuned.round(4).to_csv(args.results_dir / "test_per_category.csv", index=False)
    at_half.round(4).to_csv(args.results_dir / "test_at_0.5.csv", index=False)
    predictions = rows[["contract", "chunk_id"]].copy()
    for i, category in enumerate(PRIORITY_CATEGORIES):
        predictions[category] = proba[:, i].round(6)
    predictions.to_parquet(args.output_dir / "test_predictions.parquet", index=False)
    labels, mask = label_arrays(rows)
    macro = tuned.set_index("category")
    record = {
        "task": 13, "scored_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "checkpoint": str(args.checkpoint), "smoke": smoke,
        "test_contracts": int(rows.contract.nunique()), "test_chunks": len(rows),
        "valid_label_entries": int(mask.sum()), "positive_entries": int((labels * mask).sum()),
        "macro_f1": float(macro.f1["macro avg"]), "micro_f1": float(macro.f1["micro avg"]),
        "macro_average_precision": float(macro.average_precision["macro avg"]),
        "macro_f1_at_0.5": float(at_half.set_index("category").f1["macro avg"]),
        "test_bce": bce, "thresholds": thresholds,
        "labels_sha256": sha256(args.labels),
        "second_run": lock.exists(),
    }
    write_json(lock, record)
    print(tuned.round(3).to_string(index=False))
    print(json.dumps({k: v for k, v in record.items() if k != "thresholds"}, indent=2))
    return record


# ---------------------------------------------------------------------- CLI


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p, split="train"):
        p.add_argument("--labels", type=Path, default=labels_path(WORK_DIR, split))
        p.add_argument("--local-files-only", action="store_true")

    p = sub.add_parser("prepare", help="legal-bert chunks + labels for train and test")
    p.add_argument("--work-dir", type=Path, default=WORK_DIR)
    p.add_argument("--raw-dir", type=Path, default=DATA_DIR)
    p.add_argument("--folds", type=Path, default=FOLDS)
    p.add_argument("--local-files-only", action="store_true")

    p = sub.add_parser("sweep", help="one learning rate on one fold")
    common(p)
    p.add_argument("--learning-rate", type=float, required=True)
    p.add_argument("--validation-fold", type=int, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--folds", type=Path, default=FOLDS)
    p.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--train-contracts", type=int)  # smoke only
    p.add_argument("--validation-contracts", type=int)  # smoke only
    p.add_argument("--max-steps", type=int)  # smoke only

    p = sub.add_parser("select", help="pick config + thresholds from the sweep")
    common(p)
    p.add_argument("--runs-dir", type=Path, nargs="+", required=True)
    p.add_argument("--output-dir", type=Path, default=RESULTS)
    p.add_argument("--folds", type=Path, default=FOLDS)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--bootstrap", type=int, default=2000)

    p = sub.add_parser("final", help="train on all training contracts")
    common(p)
    p.add_argument("--selection", type=Path, default=RESULTS / "selection.json")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--train-contracts", type=int)  # smoke only
    p.add_argument("--max-steps", type=int)  # smoke only

    p = sub.add_parser("test", help="score the test split once")
    common(p, split="test")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--selection", type=Path, default=RESULTS / "selection.json")
    p.add_argument("--results-dir", type=Path, default=RESULTS)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--test-contracts", type=int)  # smoke only
    p.add_argument("--smoke-ok", action="store_true")
    p.add_argument("--allow-second-test-run", action="store_true")

    args = parser.parse_args()
    {"prepare": run_prepare, "sweep": run_sweep, "select": run_select,
     "final": run_final, "test": run_test}[args.command](args)


if __name__ == "__main__":
    main()
