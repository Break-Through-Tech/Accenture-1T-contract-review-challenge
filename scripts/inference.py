"""Task #18 inference + clause register: python -m scripts.inference --help.

predict   Chunk the test contracts with legal-bert's tokenizer and run the final model
          checkpoint from task #13 on them (needs the weights and a GPU, so Colab).
register  Apply the CV thresholds, merge overlapping positive windows into one clause,
          and write the clause register: the full one to --output-dir (Drive), a few
          contracts as a sample plus a check against the gold clauses to docs/.

register defaults to the probabilities task #13 saved from the same model on the same
chunks, so it runs on a laptop without the weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.candidates import CANDIDATES, load_backend_tokenizer
from src.chunking import chunk_contracts
from src.cuad_data import DATA_DIR, PRIORITY_CATEGORIES, build_split
from src.final_training import ENCODER, load_checkpoint
from src.inference import add_evidence, merge_windows, predict_chunks, score_register

TASK13 = Path("docs/final-model-results")
TASK13_PREDICTIONS = TASK13 / "test_predictions.parquet"
RESULTS = Path("docs/inference-results")
SAMPLE_CONTRACTS = 3
SEED = 42


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_predict(args, *, tokenizer=None, model=None) -> dict:
    if tokenizer is None:
        candidate = CANDIDATES[ENCODER]
        tokenizer = load_backend_tokenizer(
            candidate, candidate.revision, local_files_only=args.local_files_only
        )
    contracts = build_split("test", raw_dir=args.raw_dir).contracts
    chunks = chunk_contracts(contracts, "test", tokenizer=tokenizer)
    model = model if model is not None else load_checkpoint(args.checkpoint)
    proba = predict_chunks(model, chunks, tokenizer, batch_size=args.batch_size)
    predictions = chunks[["contract", "chunk_id"]].copy()
    for i, category in enumerate(PRIORITY_CATEGORIES):
        predictions[category] = proba[:, i].round(6)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(args.output, index=False)

    report = {"contracts": int(chunks.contract.nunique()), "chunks": len(chunks),
              "output": str(args.output)}
    if args.compare and Path(args.compare).exists():
        # same model on the same chunks should reproduce task #13's saved probabilities
        saved = pd.read_parquet(args.compare).set_index("chunk_id")
        report["same_chunks_as_task13"] = set(saved.index) == set(predictions.chunk_id)
        if report["same_chunks_as_task13"]:
            ours = predictions.set_index("chunk_id").loc[saved.index, PRIORITY_CATEGORIES]
            difference = np.abs(saved[PRIORITY_CATEGORIES].to_numpy() - ours.to_numpy())
            report["max_difference_from_task13"] = float(difference.max())
    print(json.dumps(report, indent=2))
    return report


def run_register(args) -> dict:
    predictions = pd.read_parquet(args.predictions)
    thresholds = json.loads(args.selection.read_text())["thresholds"]
    split = build_split("test", raw_dir=args.raw_dir)
    if set(predictions.contract) - set(split.contracts.contract):
        raise ValueError("Predictions have contracts that are not in the test split")

    register = add_evidence(merge_windows(predictions, thresholds), split.contracts)
    check = score_register(register, split.spans, split.contracts.contract)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    register.to_parquet(args.output_dir / "clause_register.parquet", index=False)
    register.to_csv(args.output_dir / "clause_register.csv", index=False)

    positive_windows = sum(
        int((predictions[c] >= thresholds[c]).sum()) for c in PRIORITY_CATEGORIES
    )
    summary = {
        "task": 18,
        "predictions": Path(args.predictions).as_posix(), "predictions_sha256": sha256(args.predictions),
        "thresholds": thresholds,
        "test_contracts": int(split.contracts.contract.nunique()),
        "chunks": len(predictions),
        "positive_windows": positive_windows,
        "register_entries": len(register),
        "windows_merged_away": positive_windows - len(register),
        "contracts_with_an_entry": int(register.contract.nunique()),
        "macro": check.iloc[-1].drop("category").to_dict(),
    }
    args.results_dir.mkdir(parents=True, exist_ok=True)
    check.to_csv(args.results_dir / "register_check.csv", index=False)
    contracts = pd.Series(sorted(register.contract.unique()))
    sample = contracts.sample(min(SAMPLE_CONTRACTS, len(contracts)), random_state=SEED)
    register.loc[register.contract.isin(sample)].to_csv(
        args.results_dir / "clause_register_sample.csv", index=False
    )
    (args.results_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(check.to_string(index=False))
    print(json.dumps({k: v for k, v in summary.items() if k != "thresholds"}, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("predict", help="run the final checkpoint on the test contracts")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--compare", type=Path, default=TASK13_PREDICTIONS)
    p.add_argument("--raw-dir", type=Path, default=DATA_DIR)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--local-files-only", action="store_true")

    p = sub.add_parser("register", help="thresholds + merged windows -> clause register")
    p.add_argument("--predictions", type=Path, default=TASK13_PREDICTIONS)
    p.add_argument("--selection", type=Path, default=TASK13 / "selection.json")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--results-dir", type=Path, default=RESULTS)
    p.add_argument("--raw-dir", type=Path, default=DATA_DIR)

    args = parser.parse_args()
    {"predict": run_predict, "register": run_register}[args.command](args)


if __name__ == "__main__":
    main()
