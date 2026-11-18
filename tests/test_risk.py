"""Tests for the rule definitions currently present in ``src.risk``.

The module currently exposes the rubric constants, level mapping, and compiled
adjustment patterns. End-to-end score, rationale, and confidence routing tests
can be added once a public clause-scoring function exists.
"""

import pytest

from src.cuad_data import PRIORITY_CATEGORIES
from src.risk import ADJUSTMENTS, BASE, level_for


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
def test_base_points_match_rubric_for_every_priority_category(category):
    assert BASE[category] == EXPECTED_BASE[category]


def test_base_points_cover_exactly_the_priority_categories():
    assert set(BASE) == set(PRIORITY_CATEGORIES)


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0, "Low"),
        (1, "Low"),
        (2, "Medium"),
        (3, "Medium"),
        (4, "High"),
        (6, "High"),
    ],
)
def test_risk_level_threshold_boundaries(score, expected):
    assert level_for(score) == expected


@pytest.mark.parametrize(
    ("adjustment_name", "category", "text"),
    [
        (
            "liability_carve_out",
            "Uncapped Liability",
            "Liability for fraud is unlimited.",
        ),
        (
            "liability_carve_out",
            "Cap On Liability",
            "The limitation shall not apply to gross negligence.",
        ),
        (
            "immediate_termination",
            "Termination For Convenience",
            "Customer may terminate this agreement immediately.",
        ),
        (
            "immediate_termination",
            "Termination For Convenience",
            "Customer may terminate without prior notice.",
        ),
        (
            "renewal_no_opt_out",
            "Renewal Term",
            "The agreement automatically renews and may not be cancelled.",
        ),
        (
            "sole_discretion",
            "Anti-Assignment",
            "Consent may be withheld in the sole discretion of the Company.",
        ),
        (
            "active_termination_right",
            "Termination For Convenience",
            "Either party may terminate without cause on 60 days' written notice.",
        ),
        (
            "audit_without_notice",
            "Audit Rights",
            "Customer may inspect the records without prior notice.",
        ),
    ],
)
def test_each_adjustment_pattern_matches_representative_contract_wording(
    adjustment_name, category, text
):
    adjustment = next(rule for rule in ADJUSTMENTS if rule.name == adjustment_name)

    assert category in adjustment.categories
    assert adjustment.regex.search(text)


@pytest.mark.parametrize(
    ("adjustment_name", "text"),
    [
        ("liability_carve_out", "Liability remains subject to the agreed cap."),
        (
            "immediate_termination",
            "Customer may terminate for convenience on 60 days' notice.",
        ),
        (
            "renewal_no_opt_out",
            "The agreement automatically renews unless either party gives notice.",
        ),
        (
            "active_termination_right",
            "Upon termination without cause, the parties shall return materials.",
        ),
        ("audit_without_notice", "Customer may audit the books upon 10 days' notice."),
    ],
)
def test_adjustment_patterns_skip_representative_non_triggering_context(
    adjustment_name, text
):
    adjustment = next(rule for rule in ADJUSTMENTS if rule.name == adjustment_name)

    assert not adjustment.regex.search(text)


def test_adjustments_use_case_insensitive_and_flexible_whitespace_matching():
    termination = next(
        rule for rule in ADJUSTMENTS if rule.name == "active_termination_right"
    )
    audit = next(rule for rule in ADJUSTMENTS if rule.name == "audit_without_notice")

    assert termination.regex.search("CUSTOMER MAY\n\tTERMINATE for its convenience")
    assert audit.regex.search("Audit access WITHOUT\n  PRIOR notice")


def test_a_clause_can_match_multiple_distinct_adjustments():
    text = (
        "Customer may terminate immediately without notice and may terminate "
        "for convenience at any time."
    )
    applicable = {
        rule.name for rule in ADJUSTMENTS
        if "Termination For Convenience" in rule.categories and rule.regex.search(text)
    }

    assert applicable == {"immediate_termination", "active_termination_right"}


def test_overlapping_liability_phrasings_are_both_recognized_as_one_adjustment_rule():
    liability = next(rule for rule in ADJUSTMENTS if rule.name == "liability_carve_out")
    text = "Liability is unlimited and the limitation shall not apply to fraud."

    # One adjustment object represents this rubric signal. A scorer should apply
    # its points once even when more than one alternative in the regex matches.
    assert liability.regex.search(text)
    assert sum(rule.name == "liability_carve_out" for rule in ADJUSTMENTS) == 1


def test_adjustment_metadata_is_complete_and_points_follow_rubric():
    expected_points = {
        "liability_carve_out": 2,
        "immediate_termination": 2,
        "renewal_no_opt_out": 2,
        "sole_discretion": 1,
        "active_termination_right": 1,
        "audit_without_notice": 1,
    }

    assert {rule.name: rule.points for rule in ADJUSTMENTS} == expected_points
    assert all(rule.categories for rule in ADJUSTMENTS)
    assert all(rule.why for rule in ADJUSTMENTS)


def test_all_adjustments_reference_only_supported_categories():
    for rule in ADJUSTMENTS:
        assert rule.categories <= set(PRIORITY_CATEGORIES)


def test_unrelated_evidence_fires_no_adjustment():
    text = "The parties will meet quarterly to review implementation progress."

    assert not [rule.name for rule in ADJUSTMENTS if rule.regex.search(text)]
