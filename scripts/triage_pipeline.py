"""Task #19 end-to-end pipeline: python -m scripts.triage_pipeline --help.

contract JSON -> chunks -> model -> clause register -> clause risk -> contract triage

Test split (default, no GPU): runs twice and compares the two rankings.
  --source gold   CUAD gold spans as the clause register (no model)
  --source model  Task #18 register built from Task #13's official test scores
  --source both   (default) both, plus gold-vs-model agreement

New contracts: --contracts-json FILE --checkpoint DIR chunks them with the legal-bert
tokenizer, runs the saved Task #13 model, builds the register and triages it.

Outputs go to docs/triage-results/ (clause risk without the long evidence windows,
contract triage tables, comparison, metadata) and data/interim/triage/ (full clause
risk with evidence, not in Git).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.clause_register import build_register
from src.cuad_data import DATA_DIR, build_split
from src.triage import (
    LEVEL_WEIGHT,
    MISSING_CAP_POINTS,
    cap_candidates,
    clause_level_agreement,
    compare_rankings,
    gold_register,
    score_register,
    triage,
)

RESULTS = Path("docs/final-model-results")
OUT_DOCS = Path("docs/triage-results")
OUT_FULL = DATA_DIR / "interim" / "triage"

# readable clause table: category > evidence > risk level > rationale (> triage, per contract)
CLAUSE_VIEW = [
    "contract", "category", "clause_id", "source", "start", "end",
    "excerpt_start", "excerpt_end", "evidence_excerpt", "confidence",
    "risk_score", "risk_level", "adjustment", "trigger_text", "rationale",
    "needs_review", "review_reasons",
]


def frozen_thresholds(run_path: Path) -> dict:
    run = json.loads(Path(run_path).read_text())
    if run.get("task") != 13 or run.get("smoke") or run.get("second_run"):
        raise ValueError("Expected test_run.json from Task #13's single official test run")
    return run["thresholds"]


def clause_view(scored: pd.DataFrame, max_chars: int = 600) -> pd.DataFrame:
    ev = scored.excerpt_text.where(scored.excerpt_text.str.strip().astype(bool),
                                   scored.evidence_text)
    out = scored.assign(evidence_excerpt=ev.str.slice(0, max_chars).str.split().str.join(" "))
    return out[CLAUSE_VIEW]


def write(scored, ranked, name, out_docs: Path, out_full: Path) -> None:
    out_docs.mkdir(parents=True, exist_ok=True)
    out_full.mkdir(parents=True, exist_ok=True)
    clause_view(scored).to_csv(out_docs / f"{name}_clause_risk.csv", index=False)
    ranked.to_csv(out_docs / f"{name}_contract_triage.csv", index=False)
    scored.to_parquet(out_full / f"{name}_clause_risk.parquet", index=False)


def run_gold(split) -> tuple[pd.DataFrame, pd.DataFrame]:
    scored = score_register(gold_register(split.spans, split.contracts))
    return scored, triage(scored, split.contracts.contract)


def run_model(split, predictions, thresholds) -> tuple[pd.DataFrame, pd.DataFrame]:
    register = build_register(predictions, split.contracts, thresholds, split=split.name)
    scored = score_register(register)
    ranked = triage(scored, split.contracts.contract,
                    cap_candidate_contracts=cap_candidates(predictions, thresholds))
    return scored, ranked


def summarize(scored, ranked) -> dict:
    return {
        "clauses": len(scored),
        "clause_levels": scored.risk_level.value_counts().to_dict(),
        "clauses_for_review": int(scored.needs_review.sum()),
        "contracts": len(ranked),
        "contracts_with_high": int((ranked.n_high > 0).sum()),
        "missing_cap_penalty": int((ranked.missing_cap_points > 0).sum()),
        "cap_unresolved": int(ranked.cap_status.str.startswith("unresolved").sum()),
        "triage_score_median": float(ranked.triage_score.median()),
        "triage_score_max": float(ranked.triage_score.max()),
        "top5": ranked.head(5)[["rank", "contract", "triage_score", "top_risks"]]
                .to_dict("records"),
    }


def run_test_split(args) -> None:
    split = build_split("test", raw_dir=args.raw_dir)
    meta = {"task": 19, "split": "test", "level_weight": LEVEL_WEIGHT,
            "missing_cap_points": MISSING_CAP_POINTS, "runs": {}}
    out = {}
    if args.source in ("gold", "both"):
        out["gold"] = run_gold(split)
    if args.source in ("model", "both"):
        predictions = pd.read_parquet(args.predictions)
        out["model"] = run_model(split, predictions, frozen_thresholds(args.run))
    for name, (scored, ranked) in out.items():
        write(scored, ranked, name, args.out_docs, args.out_full)
        meta["runs"][name] = summarize(scored, ranked)
    if len(out) == 2:
        cmp = compare_rankings(out["gold"][1], out["model"][1])
        agree = clause_level_agreement(out["gold"][0], out["model"][0])
        agree.round(4).to_csv(args.out_docs / "clause_level_agreement.csv", index=False)
        meta["gold_vs_model"] = cmp
        print(agree.round(3).to_string(index=False))
    (args.out_docs / "metadata.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    for name, (_, ranked) in out.items():
        print(f"\n== {name}: top 10 ==")
        print(ranked.head(10)[["rank", "triage_score", "n_high", "cap_status", "contract"]]
              .to_string(index=False))
    print(json.dumps({k: v for k, v in meta.items() if k != "runs"}, indent=2))


def run_new_contracts(args) -> None:
    from tokenizers import Tokenizer

    from src.chunking import chunk_contracts
    from src.clause_register import score_chunks

    contracts = pd.DataFrame(json.loads(Path(args.contracts_json).read_text()))
    if {"contract", "text"} - set(contracts.columns):
        raise ValueError("contracts JSON must be a list of {contract, text}")
    tokenizer = Tokenizer.from_file(str(Path(args.checkpoint) / "tokenizer.json"))
    chunks = chunk_contracts(contracts, args.split_name, tokenizer=tokenizer)
    predictions = score_chunks(chunks, args.checkpoint)
    thresholds = frozen_thresholds(args.run)
    register = build_register(predictions, contracts, thresholds, split=args.split_name)
    scored = score_register(register)
    ranked = triage(scored, contracts.contract,
                    cap_candidate_contracts=cap_candidates(predictions, thresholds))
    name = Path(args.contracts_json).stem
    write(scored, ranked, name, args.out_full, args.out_full)  # new data stays out of docs/
    print(ranked[["rank", "triage_score", "worst_level", "top_risks", "contract"]]
          .to_string(index=False))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["gold", "model", "both"], default="both")
    p.add_argument("--predictions", type=Path, default=RESULTS / "test_predictions.parquet")
    p.add_argument("--run", type=Path, default=RESULTS / "test_run.json")
    p.add_argument("--raw-dir", type=Path, default=DATA_DIR)
    p.add_argument("--out-docs", type=Path, default=OUT_DOCS)
    p.add_argument("--out-full", type=Path, default=OUT_FULL)
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
