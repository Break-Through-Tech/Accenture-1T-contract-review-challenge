"""Task #19: roll clause risk up into one contract-level triage score and rank contracts.

Pipeline (one row per clause -> one row per contract):

1. Clause register: Task #18 model output (``source="model"``) or CUAD gold spans
   (``gold_register``, ``source="cuad"``), same columns either way.
2. Clause risk: ``src.risk.score_clause`` on the excerpt when there is one, else the
   full evidence window (Task #18 found excerpts land on the real clause 93% of the time
   and keep wording from neighbouring clauses from firing red flags).
3. Contract triage score:

       triage = sum over categories of (worst clause score in that category x level weight)
                + missing-cap penalty (0 or 0.5, rubric section 3)

   Taking the worst clause per category stops a contract with three overlapping Cap On
   Liability spans from outscoring one with a single uncapped-liability carve-out.
   High clauses are weighted up so that one serious clause is not outweighed by a few
   routine ones. The weights are an initial assumption for advisor validation.
4. Rank: triage score (desc), then number of High clauses, then name.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.clause_register import CATEGORIES, REGISTER_COLUMNS, merge_windows
from src.risk import score_clause

LEVEL_WEIGHT = {"Low": 1.0, "Medium": 1.0, "High": 1.5}
MISSING_CAP_POINTS = 0.5  # rubric section 3
CAP = "Cap On Liability"
CAP_CANDIDATE_FLOOR = 0.5  # model chunk prob in [floor, threshold) = unresolved cap candidate
LEVEL_ORDER = {"Low": 1, "Medium": 2, "High": 3}

# a damages exclusion alone leaves the cap check unresolved (rubric section 3)
DAMAGES_EXCLUSION = re.compile(
    r"(consequential|indirect|incidental|special|punitive|exemplary)\s+damages", re.IGNORECASE
)

CLAUSE_RISK_COLUMNS = [
    "base_points", "adjustment", "adjustment_points", "trigger_text",
    "risk_score", "risk_level", "needs_review", "review_reasons", "rationale",
]


# ------------------------------------------------------------------ clause register from gold


def gold_register(spans: pd.DataFrame, contracts: pd.DataFrame) -> pd.DataFrame:
    """CUAD gold spans as a clause register (same columns as Task #18's model register).

    Overlapping or touching spans of the same contract and category become one clause,
    exactly as model chunks are merged, so both sources count clauses the same way.
    """
    text_of = dict(zip(contracts.contract, contracts.text))
    spans = spans[spans.category.isin(CATEGORIES)]
    rows = []
    for (contract, category), g in spans.groupby(["contract", "category"], sort=True):
        text = text_of[contract]
        for k, (s, e, _) in enumerate(merge_windows(g.start.to_numpy(), g.end.to_numpy())):
            rows.append({
                "contract": contract, "category": category,
                "clause_id": f"{contract}|{category}|{k}",
                "start": int(s), "end": int(e), "evidence_text": text[s:e],
                "excerpt_start": None, "excerpt_end": None, "excerpt_text": "",
                "excerpt_rules": "", "confidence": np.nan, "threshold": np.nan,
                "review_boundary": np.nan, "needs_review": False, "review_reason": "",
                "n_chunks": 0, "chunk_ids": "", "source": "cuad",
            })
    reg = pd.DataFrame(rows, columns=REGISTER_COLUMNS)
    for col in ("excerpt_start", "excerpt_end"):
        reg[col] = reg[col].astype("Int64")
    return reg.sort_values(["contract", "start", "category"], ignore_index=True)


# ------------------------------------------------------------------ clause risk


def scoring_text(row) -> str:
    """Excerpt when Task #18 located one, else the full evidence window."""
    ex = row.get("excerpt_text")  # row: dict or pandas Series
    if isinstance(ex, str) and ex.strip():
        return ex
    return row["evidence_text"]


def score_register(register: pd.DataFrame) -> pd.DataFrame:
    """Add rubric risk columns to a clause register (one row per clause)."""
    out = register.copy()
    # Task #18's needs_review is the confidence check only; the scorer redoes it with
    # the same boundary and adds the High / invalid-input reasons.
    out = out.rename(columns={"needs_review": "low_confidence", "review_reason": "register_note"})
    results = []
    for row in out.to_dict("records"):
        conf = row.get("confidence")
        thr = row.get("threshold")
        r = score_clause(
            row["category"], scoring_text(row),
            None if pd.isna(conf) else conf,
            threshold=None if pd.isna(thr) else thr,
            source=row.get("source", "model"),
        )
        results.append({
            "base_points": r.base_points, "adjustment": r.adjustment or "",
            "adjustment_points": r.adjustment_points, "trigger_text": r.trigger_text,
            "risk_score": r.score, "risk_level": r.level, "needs_review": r.needs_review,
            "review_reasons": "; ".join(r.review_reasons), "rationale": r.rationale,
        })
    risk = pd.DataFrame(results, columns=CLAUSE_RISK_COLUMNS, index=out.index)
    risk["risk_score"] = risk["risk_score"].astype("Int64")
    risk["base_points"] = risk["base_points"].astype("Int64")
    return pd.concat([out, risk], axis=1)


# ------------------------------------------------------------------ missing cap


def missing_cap_check(
    clauses: pd.DataFrame,
    *,
    expect_cap: bool | None = True,
    processed_in_full: bool = True,
    cap_candidate: bool = False,
) -> tuple[float, str, bool]:
    """Rubric section 3 for one contract: (points, status, needs_review).

    ``clauses`` are that contract's scored clauses. ``cap_candidate`` marks an
    unresolved cap candidate the register did not keep (e.g. a model chunk just under
    the Cap On Liability threshold).
    """
    if expect_cap is None:
        return 0.0, "cap expectation unknown", False
    if not expect_cap:
        return 0.0, "cap not expected", False
    if (clauses.category == CAP).any():
        return 0.0, "cap present", False
    if not processed_in_full:
        return 0.0, "unresolved: contract not processed in full", True
    if cap_candidate:
        return 0.0, "unresolved: cap candidate below threshold", True
    texts = clauses.apply(scoring_text, axis=1) if len(clauses) else pd.Series(dtype=str)
    if texts.map(lambda t: bool(DAMAGES_EXCLUSION.search(t))).any():
        return 0.0, "unresolved: damages exclusion without a cap", True
    return MISSING_CAP_POINTS, "Expected liability cap not found. Check the full contract.", False


def cap_candidates(predictions: pd.DataFrame, thresholds: dict, floor=CAP_CANDIDATE_FLOOR) -> set:
    """Contracts with a model chunk scoring in [floor, threshold) for Cap On Liability."""
    p = predictions[CAP]
    return set(predictions.loc[(p >= floor) & (p < thresholds[CAP]), "contract"])


# ------------------------------------------------------------------ contract triage


def triage(
    scored: pd.DataFrame,
    contracts,
    *,
    expect_cap: bool | None = True,
    cap_candidate_contracts=frozenset(),
    level_weight=LEVEL_WEIGHT,
) -> pd.DataFrame:
    """One row per contract (all of ``contracts``, even those with no clauses), ranked."""
    names = sorted(set(contracts))
    by_contract = {c: g for c, g in scored.groupby("contract")}
    empty = scored.iloc[0:0]
    rows = []
    for name in names:
        g = by_contract.get(name, empty)
        ok = g[g.risk_score.notna()]
        per_cat = []
        for category, cg in ok.groupby("category"):
            worst = cg.loc[cg.risk_score.astype(int).idxmax()]
            per_cat.append((category, int(worst.risk_score), worst.risk_level,
                            int(worst.risk_score) * level_weight[worst.risk_level], worst))
        clause_points = sum(p for *_, p, _ in per_cat)
        cap_pts, cap_status, cap_review = missing_cap_check(
            g, expect_cap=expect_cap, cap_candidate=name in cap_candidate_contracts)
        levels = g.risk_level.dropna()
        top = sorted(per_cat, key=lambda x: (-x[3], x[0]))[:3]
        rows.append({
            "contract": name,
            "triage_score": round(clause_points + cap_pts, 2),
            "clause_points": round(clause_points, 2),
            "missing_cap_points": cap_pts,
            "cap_status": cap_status,
            "n_clauses": len(g),
            "n_categories": len(per_cat),
            "n_high": int((levels == "High").sum()),
            "n_medium": int((levels == "Medium").sum()),
            "n_low": int((levels == "Low").sum()),
            "worst_level": max(levels, key=LEVEL_ORDER.get) if len(levels) else "None",
            "clauses_for_review": int(g.needs_review.sum()),
            "needs_review": bool(g.needs_review.any() or cap_review),
            "categories": ";".join(c for c, *_ in sorted(per_cat)),
            "top_risks": " | ".join(f"{c} {lvl} ({s})" for c, s, lvl, _, _ in top),
            "top_rationale": top[0][4].rationale if top else "",
        })
    out = pd.DataFrame(rows)
    out = out.sort_values(["triage_score", "n_high", "contract"],
                          ascending=[False, False, True], ignore_index=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


# ------------------------------------------------------------------ gold vs model


def compare_rankings(gold: pd.DataFrame, model: pd.DataFrame, top_k=(10, 20)) -> dict:
    """How closely the model-driven triage ranking tracks the gold-span ranking."""
    m = gold[["contract", "rank", "triage_score", "worst_level"]].merge(
        model[["contract", "rank", "triage_score", "worst_level"]],
        on="contract", suffixes=("_gold", "_model"))
    out = {
        "contracts": len(m),
        "spearman_triage_score": float(m.triage_score_gold.corr(m.triage_score_model,
                                                                method="spearman")),
        "pearson_triage_score": float(m.triage_score_gold.corr(m.triage_score_model)),
        "mean_abs_rank_diff": float((m.rank_gold - m.rank_model).abs().mean()),
        "worst_level_agreement": float((m.worst_level_gold == m.worst_level_model).mean()),
    }
    for k in top_k:
        g = set(m.nsmallest(k, "rank_gold").contract)
        md = set(m.nsmallest(k, "rank_model").contract)
        out[f"top{k}_overlap"] = len(g & md) / k
    return out


def clause_level_agreement(gold_scored: pd.DataFrame, model_scored: pd.DataFrame) -> pd.DataFrame:
    """For model clauses that overlap a gold clause of the same category, do the levels agree?"""
    g_by = {k: v for k, v in gold_scored.groupby(["contract", "category"])}
    rows = []
    for r in model_scored.itertuples(index=False):
        g = g_by.get((r.contract, r.category))
        if g is None:
            continue
        hit = g[(g.start < r.end) & (g.end > r.start)]
        if hit.empty or pd.isna(r.risk_score):
            continue
        gs = int(hit.risk_score.max())
        rows.append({"category": r.category, "model_level": r.risk_level,
                     "gold_level": hit.loc[hit.risk_score.astype(int).idxmax()].risk_level,
                     "model_score": int(r.risk_score), "gold_score": gs})
    df = pd.DataFrame(rows, columns=["category", "model_level", "gold_level",
                                     "model_score", "gold_score"])
    if df.empty:
        return df
    table = (df.assign(agree=df.model_level == df.gold_level,
                       model_higher=df.model_score > df.gold_score,
                       model_lower=df.model_score < df.gold_score)
               .groupby("category")
               .agg(matched=("agree", "size"), level_agreement=("agree", "mean"),
                    model_higher=("model_higher", "mean"), model_lower=("model_lower", "mean"))
               .reindex(CATEGORIES).reset_index())
    total = {"category": "all", "matched": len(df),
             "level_agreement": float((df.model_level == df.gold_level).mean()),
             "model_higher": float((df.model_score > df.gold_score).mean()),
             "model_lower": float((df.model_score < df.gold_score).mean())}
    return pd.concat([table, pd.DataFrame([total])], ignore_index=True)
