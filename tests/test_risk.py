"""Unit coverage for the Task #17 rule-based clause risk scorer."""

import pytest

from src.cuad_data import PRIORITY_CATEGORIES
from src.risk import ADJUSTMENTS, BASE, level_for, score_clause


EXPECTED_BASE = {
    "Cap On Liability": 1,
    "Uncapped Liability": 4,
    "Governing Law": 0,
    "Anti-Assignment": 2,
    "Change Of Control": 2,
    "Termination For Convenience": 2,
    "Exclusivity": 2,
    "Non-Compete": 2,
    "Renewal Term": 1,
    "Audit Rights": 1,
}


@pytest.mark.parametrize("category", PRIORITY_CATEGORIES)
def test_every_supported_category_scores_its_base_without_red_flags(category):
    result = score_clause(category, "The parties will cooperate in good faith.")

    assert result["score"] == EXPECTED_BASE[category]
    assert result["risk_level"] == level_for(EXPECTED_BASE[category])
    assert f"{category} base risk" in result["rationale"]


def test_base_points_cover_exactly_supported_categories():
    assert BASE == EXPECTED_BASE
    assert set(BASE) == set(PRIORITY_CATEGORIES)


@pytest.mark.parametrize(
    ("category", "evidence", "expected_score", "expected_level", "signal"),
    [
        ("Cap On Liability", "The limitation shall not apply to fraud.", 3, "Medium", "fraud"),
        ("Uncapped Liability", "Liability for fraud is unlimited.", 6, "High", "unlimited"),
        ("Termination For Convenience", "Customer may terminate immediately without prior notice.", 4, "High", "immediately"),
        ("Termination For Convenience", "Either party may terminate without cause on 60 days' notice.", 3, "Medium", "without cause"),
        ("Renewal Term", "The term automatically renews and may not be cancelled.", 3, "Medium", "automatically renews"),
        ("Anti-Assignment", "The Company may withhold consent in its sole discretion.", 3, "Medium", "sole discretion"),
        ("Change Of Control", "Approval of a change of control is in the other party's sole discretion.", 3, "Medium", "sole discretion"),
        ("Audit Rights", "Customer may inspect records without prior notice.", 2, "Medium", "without prior notice"),
    ],
)
def test_each_rubric_red_flag_returns_exact_score_and_explains_trigger(
    category, evidence, expected_score, expected_level, signal
):
    result = score_clause(category, evidence)

    assert result["score"] == expected_score
    assert result["risk_level"] == expected_level
    assert signal.casefold() in result["rationale"].casefold()


@pytest.mark.parametrize("score, expected", [(0, "Low"), (1, "Low"), (2, "Medium"), (3, "Medium"), (4, "High"), (6, "High")])
def test_risk_level_cutoff_boundaries(score, expected):
    assert level_for(score) == expected


def test_multiple_distinct_signals_apply_only_highest_adjustment():
    text = "Customer may terminate immediately without notice and for convenience at any time."
    result = score_clause("Termination For Convenience", text)

    assert result["score"] == 4  # base 2 + highest adjustment 2, not +2 and +1
    assert "immediate effect" in result["rationale"].casefold()


def test_overlapping_liability_patterns_are_counted_once():
    result = score_clause(
        "Uncapped Liability",
        "Liability is unlimited and the limitation shall not apply to fraud.",
    )

    assert result["score"] == 6
    assert result["rationale"].count("+2") == 1


@pytest.mark.parametrize(
    ("category", "evidence", "expected"),
    [
        ("Cap On Liability", "Liability remains subject to the agreed cap.", 1),
        ("Termination For Convenience", "May terminate for convenience on 60 days' notice.", 3),
        ("Renewal Term", "Automatically renews unless either party gives notice.", 1),
        ("Termination For Convenience", "Upon termination without cause, the parties return materials.", 2),
        ("Audit Rights", "Customer may audit the books upon 10 days' notice.", 1),
    ],
)
def test_non_triggering_or_contextual_phrasing_does_not_add_points(category, evidence, expected):
    assert score_clause(category, evidence)["score"] == expected


def test_matching_is_case_insensitive_and_tolerates_whitespace_differences():
    result = score_clause(
        "Termination For Convenience",
        "CUSTOMER MAY\n\tTERMINATE for its convenience immediately.",
    )

    assert result["score"] == 4
    assert "immediate effect" in result["rationale"].casefold()


@pytest.mark.parametrize("category", [None, "", "Not A CUAD Category", 3])
def test_invalid_categories_return_unset_score_and_level(category):
    result = score_clause(category, "A clause with otherwise valid text.")

    assert result["score"] is None
    assert result["risk_level"] is None
    assert "review required" in result["rationale"]


@pytest.mark.parametrize("evidence", [None, "", " \n\t ", 12])
def test_empty_or_invalid_evidence_returns_unset_score_and_level(evidence):
    result = score_clause("Governing Law", evidence)

    assert result["score"] is None
    assert result["risk_level"] is None
    assert "review required" in result["rationale"]


def test_confidence_never_changes_risk_points_including_boundary_values():
    evidence = "This Agreement is governed by the laws of New York."
    results = [score_clause("Governing Law", evidence, confidence=c) for c in (None, 0, 0.8, 0.82, 1)]

    assert {result["score"] for result in results} == {0}
    assert all(result["risk_level"] == "Low" for result in results)


@pytest.mark.parametrize("confidence", [-0.01, 1.01, float("nan"), "high", object()])
def test_invalid_confidence_does_not_corrupt_risk_score(confidence):
    result = score_clause("Governing Law", "The laws of Delaware shall govern.", confidence)

    # This function only calculates risk points. Confidence validation/routing
    # requires a frozen category threshold and is outside its current API.
    assert result["score"] == 0
    assert result["risk_level"] == "Low"


def test_rationale_names_base_adjustment_and_the_matched_words():
    result = score_clause(
        "Termination For Convenience",
        "Either party may terminate without cause on 60 days' written notice.",
    )

    assert "Termination For Convenience base risk: 2" in result["rationale"]
    assert "+1" in result["rationale"]
    assert "may terminate without cause" in result["rationale"].casefold()
    assert "3 point(s), Medium risk" in result["rationale"]


@pytest.mark.xfail(
    strict=True,
    reason="Current liability regex matches 'unlimited' even when expressly negated.",
)
def test_expressly_negated_unlimited_liability_does_not_add_points():
    result = score_clause(
        "Cap On Liability", "Liability is not unlimited under this Agreement."
    )

    assert result["score"] == BASE["Cap On Liability"]


@pytest.mark.xfail(
    strict=True,
    reason="Current audit regex does not tie 'without notice' to audit access.",
)
def test_unrelated_no_notice_phrase_does_not_increase_audit_risk():
    result = score_clause(
        "Audit Rights",
        "Audits are available annually. The agreement may terminate without prior notice.",
    )

    assert result["score"] == BASE["Audit Rights"]


def test_repeated_identical_calls_are_deterministic():
    args = (
        "Audit Rights",
        "Customer may inspect the records without prior notice.",
        0.95,
    )

    assert score_clause(*args) == score_clause(*args)


def test_adjustment_metadata_matches_rubric():
    expected_points = {
        "liability_carve_out": 2,
        "immediate_termination": 2,
        "renewal_no_opt_out": 2,
        "sole_discretion": 1,
        "active_termination_right": 1,
        "audit_without_notice": 1,
    }

    assert {rule.name: rule.points for rule in ADJUSTMENTS} == expected_points
    assert all(rule.categories and rule.why for rule in ADJUSTMENTS)
    assert all(rule.categories <= set(PRIORITY_CATEGORIES) for rule in ADJUSTMENTS)
