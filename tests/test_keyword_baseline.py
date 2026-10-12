import numpy as np
import pandas as pd
import pytest

from src.cuad_data import PRIORITY_CATEGORIES
from src.keyword_baseline import (
    RULES,
    Rule,
    category_scores,
    evaluate,
    explain,
    predict,
    rule_hits,
    rule_table,
)

CAT = {c: i for i, c in enumerate(PRIORITY_CATEGORIES)}

# (category, text that should fire it). Short paraphrases of CUAD phrasing.
POSITIVE = [
    ("Cap On Liability", "In no event shall either party be liable for lost profits."),
    ("Cap On Liability", "neither party is responsible for consequential damages"),
    ("Cap On Liability", "Licensor's total liability shall not exceed the fees paid."),
    (
        "Uncapped Liability",
        "Except for breach of Section 9, in no event shall either party be liable.",
    ),
    ("Governing Law", "This Agreement shall be governed by the laws of Delaware."),
    ("Governing Law", "The laws of the State of New York shall govern this Agreement."),
    ("Governing Law", "construed in accordance with the laws of England"),
    ("Anti-Assignment", "Neither party may assign this Agreement."),
    ("Anti-Assignment", "may not be assigned without the prior written consent"),
    ("Change Of Control", "In the event of a Change of Control of Supplier"),
    ("Change Of Control", "upon a merger of Company, Licensor may terminate"),
    ("Termination For Convenience", "Customer may terminate for convenience."),
    (
        "Termination For Convenience",
        "Either party may terminate this Agreement at any time on 30 days notice.",
    ),
    ("Exclusivity", "Distributor shall be the exclusive distributor in the Territory."),
    ("Exclusivity", "Company shall purchase exclusively from Supplier."),
    (
        "Non-Compete",
        "Consultant shall not, directly or indirectly, compete with Company.",
    ),
    ("Renewal Term", "This Agreement shall automatically renew for one-year terms."),
    ("Renewal Term", "renewed for successive one (1) year periods"),
    ("Audit Rights", "Licensor may audit Licensee's books upon notice."),
    ("Audit Rights", "the right to inspect the books and records of Supplier"),
]

# (category, text that should NOT fire it): the known false-positive phrasings.
NEGATIVE = [
    ("Exclusivity", "a non-exclusive, worldwide license"),
    ("Exclusivity", "a non- exclusive license"),
    ("Exclusivity", "a non exclusive license"),
    ("Exclusivity", "a non‑exclusive license"),
    ("Exclusivity", "the courts of Delaware shall have exclusive jurisdiction"),
    ("Exclusivity", "this is the sole and exclusive remedy of Customer"),
    ("Exclusivity", "fees are exclusive of taxes"),
    ("Exclusivity", "shall remain the exclusive property of Licensor"),
    ("Cap On Liability", "including but not limited to marketing materials"),
    (
        "Termination For Convenience",
        "Upon termination of this Agreement for any reason, Distributor shall cease use.",
    ),
    ("Governing Law", "comply with all applicable law"),
    ("Audit Rights", "the audited financial statements of the Company"),
]


@pytest.mark.parametrize("category,text", POSITIVE)
def test_rules_fire_on_typical_phrasing(category, text):
    assert predict([text])[0, CAT[category]] == 1


@pytest.mark.parametrize("category,text", NEGATIVE)
def test_rules_skip_known_false_positives(category, text):
    assert predict([text])[0, CAT[category]] == 0


def test_whitespace_and_case_are_tolerated():
    text = "THIS AGREEMENT SHALL BE\n\n   GOVERNED    BY THE\tLAWS OF Texas"
    assert predict([text])[0, CAT["Governing Law"]] == 1


def test_every_category_has_rules_and_names_are_unique():
    table = rule_table()
    assert set(table.category) == set(PRIORITY_CATEGORIES)
    assert not table.duplicated(["category", "rule"]).any()
    assert table.rationale.str.len().gt(0).all()


def test_scores_count_distinct_rules_per_category():
    text = (
        "This Agreement shall be governed by the laws of Ohio and construed in "
        "accordance with the laws of Ohio."
    )
    scores = category_scores(rule_hits([text]))
    assert scores[0, CAT["Governing Law"]] == 2
    assert scores.sum() == 2


def test_explain_reports_offsets_into_original_text():
    text = "Preamble.  Neither party may assign this Agreement."
    out = explain(text)
    row = out[out.rule == "no_party_assigns"].iloc[0]
    assert text[row.start : row.end] == row.match
    assert row.category == "Anti-Assignment"


def test_explain_empty_text_has_columns():
    out = explain("")
    assert out.empty and list(out.columns) == [
        "category",
        "rule",
        "start",
        "end",
        "match",
    ]


def test_unknown_rule_category_is_rejected():
    bad = (Rule("Indemnification", "x", r"indemnif", "not a priority category"),)
    with pytest.raises(ValueError, match="unknown categories"):
        category_scores(np.zeros((1, 1), dtype=bool), bad)


def labeled(rows):
    """rows: (text, labels dict, masked categories)."""
    out = []
    for i, (text, positive, masked) in enumerate(rows):
        labels = [int(c in positive) for c in PRIORITY_CATEGORIES]
        mask = [c not in masked for c in PRIORITY_CATEGORIES]
        out.append(
            {
                "contract": f"c{i}",
                "chunk_id": str(i),
                "text": text,
                "labels": labels,
                "label_mask": mask,
            }
        )
    return pd.DataFrame(out)


def test_evaluate_scores_and_masks():
    law = "This Agreement shall be governed by the laws of Ohio."
    data = labeled(
        [
            (law, {"Governing Law"}, set()),  # true positive
            ("Payment is due in 30 days.", {"Governing Law"}, set()),  # false negative
            (law, set(), {"Governing Law"}),  # masked: would be a false positive
            ("Delivery on Monday.", set(), set()),  # true negative
        ]
    )
    metrics, stats, scores = evaluate(data)
    row = metrics.set_index("category").loc["Governing Law"]
    assert (row.precision, row.recall) == (1.0, 0.5)
    assert (row.scored_chunks, row.masked_chunks) == (3, 1)
    assert scores.shape == (4, len(PRIORITY_CATEGORIES))
    rule = stats.set_index("rule").loc["governed_by_laws"]
    assert (rule.fired, rule.true_positive) == (1, 1)
    assert len(stats) == len(RULES)
