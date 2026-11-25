"""Tests for Task #19: clause scoring (score_clause) and contract triage."""

import math

import pandas as pd
import pytest

from src.clause_register import REGISTER_COLUMNS
from src.risk import review_boundary, score_clause
from src.triage import (
    MISSING_CAP_POINTS,
    cap_candidates,
    compare_rankings,
    gold_register,
    missing_cap_check,
    score_register,
    triage,
)

# ------------------------------------------------------------------ score_clause


def test_rubric_example_termination_without_cause_is_medium():
    r = score_clause("Termination For Convenience",
                     "Either party may terminate without cause on 60 days' written notice.",
                     source="cuad")
    assert (r.base_points, r.adjustment, r.score, r.level) == (
        2, "active_termination_right", 3, "Medium")
    assert not r.needs_review
    assert "without cause" in r.trigger_text


def test_only_highest_adjustment_counts():
    r = score_clause(
        "Termination For Convenience",
        "Customer may terminate immediately and may terminate for convenience at any time.",
        source="cuad")
    assert (r.adjustment, r.adjustment_points, r.score, r.level) == (
        "immediate_termination", 2, 4, "High")
    assert r.needs_review and "High risk" in r.review_reasons


def test_uncapped_carve_out_scores_six_high():
    r = score_clause("Uncapped Liability",
                     "The limitation shall not apply to fraud or gross negligence.", source="cuad")
    assert (r.score, r.level) == (6, "High")


def test_adjustment_for_other_category_is_ignored():
    r = score_clause("Governing Law", "sole discretion; without prior notice", source="cuad")
    assert (r.adjustment, r.score, r.level) == (None, 0, "Low")
    assert "no red flag" in r.rationale


@pytest.mark.parametrize(("category", "evidence", "reason"), [
    ("Most Favored Nation", "Some text", "unknown category"),
    ("Governing Law", "   ", "empty evidence"),
])
def test_unscorable_clause_is_unset_and_reviewed(category, evidence, reason):
    r = score_clause(category, evidence, source="cuad")
    assert r.score is None and r.level is None and r.needs_review
    assert r.review_reasons == (reason,)


def test_confidence_routing_equal_boundary_clears():
    t = 0.80
    b = review_boundary(t)
    assert b == pytest.approx(0.82)
    assert not score_clause("Governing Law", "laws of Delaware", b, threshold=t).needs_review
    low = score_clause("Governing Law", "laws of Delaware", 0.81, threshold=t)
    assert low.needs_review and low.level == "Low"  # confidence never changes the level
    assert low.review_reasons[0].startswith("low confidence")


@pytest.mark.parametrize(("conf", "thr"), [(None, 0.8), (0.9, None), (math.nan, 0.8), (1.2, 0.8)])
def test_invalid_confidence_or_threshold_goes_to_review(conf, thr):
    r = score_clause("Governing Law", "laws of Delaware", conf, threshold=thr)
    assert r.needs_review and "invalid" in r.review_reasons[0]
    assert r.score == 0


def test_cuad_source_skips_confidence_check():
    assert not score_clause("Audit Rights", "may audit the books", None,
                            source="cuad").needs_review


# ------------------------------------------------------------------ register helpers


def make_register(rows):
    base = {c: None for c in REGISTER_COLUMNS}
    base.update(excerpt_text="", source="cuad", confidence=math.nan, threshold=math.nan,
                needs_review=False, review_reason="")
    return pd.DataFrame([{**base, **r} for r in rows], columns=REGISTER_COLUMNS)


def test_gold_register_merges_overlapping_spans_and_keeps_offsets():
    contracts = pd.DataFrame({"contract": ["A"], "text": ["0123456789" * 3]})
    spans = pd.DataFrame({
        "contract": ["A", "A", "A", "A"],
        "category": ["Audit Rights", "Audit Rights", "Audit Rights", "Parties"],
        "start": [0, 5, 20, 0], "end": [8, 10, 25, 3],
    })
    reg = gold_register(spans, contracts)
    assert list(zip(reg.start, reg.end)) == [(0, 10), (20, 25)]  # non-priority dropped
    assert list(reg.evidence_text) == ["0123456789", "01234"]
    assert set(reg.source) == {"cuad"}


def test_score_register_prefers_excerpt_over_window():
    reg = make_register([{
        "contract": "A", "category": "Termination For Convenience", "start": 0, "end": 10,
        "evidence_text": ("Customer may terminate immediately for breach. "
                          "Either party may terminate for convenience."),
        "excerpt_text": "Either party may terminate for convenience.",
    }])
    scored = score_register(reg)
    assert scored.loc[0, "adjustment"] == "active_termination_right"
    assert scored.loc[0, "risk_score"] == 3


# ------------------------------------------------------------------ missing cap


def clauses(*cats, text="x"):
    return make_register([{"contract": "A", "category": c, "evidence_text": text} for c in cats])


def test_missing_cap_adds_half_point_once():
    pts, status, review = missing_cap_check(clauses("Governing Law", "Audit Rights"))
    assert (pts, review) == (MISSING_CAP_POINTS, False)
    assert status.startswith("Expected liability cap not found")


@pytest.mark.parametrize(("kwargs", "status", "review"), [
    ({"expect_cap": False}, "cap not expected", False),
    ({"expect_cap": None}, "cap expectation unknown", False),
    ({"processed_in_full": False}, "unresolved: contract not processed in full", True),
    ({"cap_candidate": True}, "unresolved: cap candidate below threshold", True),
])
def test_missing_cap_zero_cases(kwargs, status, review):
    assert missing_cap_check(clauses("Governing Law"), **kwargs) == (0.0, status, review)


def test_cap_present_gives_zero():
    assert missing_cap_check(clauses("Cap On Liability"))[:2] == (0.0, "cap present")


def test_damages_exclusion_leaves_cap_unresolved():
    c = clauses("Uncapped Liability", text="Neither party is liable for consequential damages.")
    assert missing_cap_check(c) == (0.0, "unresolved: damages exclusion without a cap", True)


def test_cap_candidates_window():
    preds = pd.DataFrame({"contract": ["A", "B", "C"], "Cap On Liability": [0.49, 0.6, 0.95]})
    assert cap_candidates(preds, {"Cap On Liability": 0.93}) == {"B"}


# ------------------------------------------------------------------ triage


def scored_fixture():
    def row(contract, category, start, text):
        return {"contract": contract, "category": category, "start": start,
                "end": start + 5, "evidence_text": text}

    return score_register(make_register([
        # A: one High (6) uncapped carve-out, no cap -> 6 * 1.5 + 0.5
        row("A", "Uncapped Liability", 0, "Liability for fraud is unlimited."),
        # B: Medium clauses (duplicate category counted once) + cap present -> 2+2+2+1
        row("B", "Anti-Assignment", 0, "may not assign"),
        row("B", "Anti-Assignment", 9, "may not assign"),
        row("B", "Exclusivity", 20, "exclusive"),
        row("B", "Non-Compete", 30, "not compete"),
        row("B", "Cap On Liability", 40, "capped at fees"),
    ]))


def test_triage_scores_worst_clause_per_category_plus_cap_penalty():
    ranked = triage(scored_fixture(), ["A", "B", "C"]).set_index("contract")
    assert ranked.loc["A", "triage_score"] == pytest.approx(6 * 1.5 + 0.5)
    assert ranked.loc["A", "missing_cap_points"] == 0.5
    assert ranked.loc["B", "triage_score"] == pytest.approx(2 + 2 + 2 + 1)
    assert ranked.loc["B", "n_clauses"] == 5 and ranked.loc["B", "n_categories"] == 4
    # a contract with no clauses is still ranked and gets only the cap penalty
    assert ranked.loc["C", "triage_score"] == 0.5 and ranked.loc["C", "worst_level"] == "None"


def test_triage_ranks_one_serious_clause_above_routine_ones():
    ranked = triage(scored_fixture(), ["A", "B", "C"])
    assert list(ranked.contract) == ["A", "B", "C"]
    assert list(ranked["rank"]) == [1, 2, 3]
    assert ranked.loc[0, "needs_review"]  # High clause
    assert "Uncapped Liability High (6)" in ranked.loc[0, "top_risks"]


def test_triage_ties_break_by_high_count_then_name():
    reg = make_register([
        {"contract": n, "category": "Governing Law", "start": 0, "end": 1, "evidence_text": "law"}
        for n in ["Z", "Y"]
    ])
    ranked = triage(score_register(reg), ["Z", "Y"], expect_cap=False)
    assert list(ranked.contract) == ["Y", "Z"]


def test_compare_rankings_identical_is_perfect():
    ranked = triage(scored_fixture(), ["A", "B", "C"])
    cmp = compare_rankings(ranked, ranked, top_k=(2,))
    assert cmp["spearman_triage_score"] == pytest.approx(1.0)
    assert cmp["mean_abs_rank_diff"] == 0 and cmp["top2_overlap"] == 1.0
