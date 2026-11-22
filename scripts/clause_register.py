"""Task #18 clause register: python -m scripts.clause_register --help.

Default: build the register for the 102 official test contracts from Task #13's
test predictions + frozen thresholds (no GPU needed), compare it with CUAD gold
spans, and write the summary, a 5-contract sample and the full register (parquet)
to docs/clause-register-results/. A JSONL copy goes to data/interim/clause_register/.

New contracts: --contracts-json FILE --checkpoint DIR chunks them with the
legal-bert tokenizer and runs the saved model first (needs the Task #13 weights).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from src.clause_register import CATEGORIES, build_register, match_gold, score_chunks
from src.cuad_data import DATA_DIR, build_split

RESULTS = Path("docs/final-model-results")
OUT_DOCS = Path("docs/clause-register-results")
OUT_FULL = DATA_DIR / "interim" / "clause_register"
SAMPLE_CONTRACTS = 5


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_thresholds(run_path: Path) -> dict:
    run = json.loads(Path(run_path).read_text())
    if run.get("task") != 13 or run.get("smoke") or run.get("second_run"):
        raise ValueError("Expected test_run.json from Task #13's single official test run")
    return run["thresholds"]


def write_full(register: pd.DataFrame, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    register.to_parquet(out / f"{name}.parquet", index=False)
    with open(out / f"{name}.jsonl", "w", encoding="utf-8") as fh:
        for rec in register.to_dict("records"):
            rec = {k: (None if pd.isna(v) else v) if not isinstance(v, (list, dict)) else v
                   for k, v in rec.items()}
            fh.write(json.dumps(rec, ensure_ascii=False, default=int) + "\n")


def run_test_split(args) -> dict:
    predictions = pd.read_parquet(args.predictions)
    thresholds = frozen_thresholds(args.run)
    split = build_split("test", raw_dir=args.raw_dir)
    register = build_register(predictions, split.contracts, thresholds, split="test")

    positives = pd.Series({c: int((predictions[c] >= thresholds[c]).sum()) for c in CATEGORIES})
    summary = match_gold(register, split.spans, positives)
    args.out_docs.mkdir(parents=True, exist_ok=True)
    summary.round(4).to_csv(args.out_docs / "summary.csv", index=False)

    # per-contract roll-up: what Task #19 ranks
    per_contract = (register.groupby("contract")
                    .agg(clauses=("clause_id", "size"),
                         categories=("category", lambda s: ";".join(sorted(set(s)))),
                         needs_review=("needs_review", "sum"))
                    .reindex(split.contracts.contract)
                    .reset_index())
    per_contract = per_contract.fillna({"clauses": 0, "categories": "", "needs_review": 0})
    per_contract[["clauses", "needs_review"]] = per_contract[["clauses", "needs_review"]].astype(int)
    per_contract.to_csv(args.out_docs / "per_contract.csv", index=False)

    sample_names = sorted(register.contract.unique())[: args.sample_contracts]
    sample = register[register.contract.isin(sample_names)].drop(columns=["chunk_ids"])
    sample.to_csv(args.out_docs / "clause_register_sample.csv", index=False)

    write_full(register, args.out_full, "test_clause_register")
    # full register is small (<1 MB parquet): also commit it so Task #19 can read it
    register.to_parquet(args.out_docs / "test_clause_register.parquet", index=False)
    meta = {
        "task": 18, "split": "test",
        "contracts": int(split.contracts.contract.nunique()),
        "chunks_scored": len(predictions), "clauses": len(register),
        "positive_chunks": int(positives.sum()),
        "thresholds": thresholds, "review_margin": 0.10,
        "model_scores": "Task #13 official test export (legal-bert + pos_weight, 3 epochs)",
        "predictions_sha256": sha256(args.predictions),
        "test_run_sha256": sha256(args.run),
        "sample_contracts": sample_names,
        "full_register": "docs/clause-register-results/test_clause_register.parquet",
        "full_register_jsonl": "data/interim/clause_register/test_clause_register.jsonl (not in Git)",
    }
    (args.out_docs / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(summary.round(3).to_string(index=False))
    print(json.dumps({k: v for k, v in meta.items() if k != "thresholds"}, indent=2))
    return meta


def run_new_contracts(args) -> None:
    from tokenizers import Tokenizer

    from src.chunking import chunk_contracts

    data = json.loads(Path(args.contracts_json).read_text())
    contracts = pd.DataFrame(data)  # [{"contract": name, "text": full text}, ...]
    if {"contract", "text"} - set(contracts.columns):
        raise ValueError("contracts JSON must be a list of {contract, text}")
    tokenizer = Tokenizer.from_file(str(Path(args.checkpoint) / "tokenizer.json"))
    chunks = chunk_contracts(contracts, args.split_name, tokenizer=tokenizer)
    predictions = score_chunks(chunks, args.checkpoint)
    thresholds = frozen_thresholds(args.run)
    register = build_register(predictions, contracts, thresholds, split=args.split_name)
    write_full(register, args.out_full, f"{Path(args.contracts_json).stem}_clause_register")
    print(f"{len(register)} clauses from {contracts.contract.nunique()} contracts -> {args.out_full}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions", type=Path, default=RESULTS / "test_predictions.parquet")
    p.add_argument("--run", type=Path, default=RESULTS / "test_run.json")
    p.add_argument("--raw-dir", type=Path, default=DATA_DIR)
    p.add_argument("--out-docs", type=Path, default=OUT_DOCS)
    p.add_argument("--out-full", type=Path, default=OUT_FULL)
    p.add_argument("--sample-contracts", type=int, default=SAMPLE_CONTRACTS)
    p.add_argument("--contracts-json", type=Path, help="new contracts: [{contract, text}]")
    p.add_argument("--checkpoint", type=Path, help="Task #13 final_model/checkpoint")
    p.add_argument("--split-name", default="new")
    args = p.parse_args()
    if args.contracts_json:
        if not args.checkpoint:
            p.error("--contracts-json needs --checkpoint")
        run_new_contracts(args)
    else:
        run_test_split(args)


if __name__ == "__main__":
    main()
