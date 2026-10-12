"""Keyword/rule-based baseline for task #8.

Every category has a short list of hand-written regular expressions built from legal
phrasing seen in the training annotations. A chunk is predicted positive for a category
when at least one of that category's rules matches. Nothing is fitted, so every
prediction can be traced to the rule and the exact text that fired it.

Patterns are written in lowercase with single spaces; at compile time each space
becomes ``\\s+`` (CUAD text has irregular SEC-filing whitespace) and matching ignores
case. Matches are offsets into the original chunk text.

Input and scoring follow the task #7 baseline: the labeled chunk table from task #6,
with masked (ambiguous) chunk-category pairs excluded per category.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.cuad_data import PRIORITY_CATEGORIES
from src.labeling import label_arrays
from src.tfidf_baseline import score

# Generic building blocks, reused across categories.
ANY = r"[\s\S]"  # any character, including newlines
LIABLE = r"liab(le|ility|ilities)"
MUST_NOT = r"(shall|will|may|must) not|cannot|can not"


def near(a: str, b: str, window: int) -> str:
    """``a`` and ``b`` within ``window`` characters of each other, in either order."""
    return rf"(?:{a}){ANY}{{0,{window}}}?(?:{b})|(?:{b}){ANY}{{0,{window}}}?(?:{a})"


@dataclass(frozen=True)
class Rule:
    category: str
    name: str
    pattern: str
    rationale: str
    regex: re.Pattern = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        compiled = re.compile(self.pattern.replace(" ", r"\s+"), re.IGNORECASE)
        object.__setattr__(self, "regex", compiled)


def _rules(category: str, *specs: tuple[str, str, str]) -> list[Rule]:
    return [Rule(category, name, pattern, why) for name, pattern, why in specs]


RULES: tuple[Rule, ...] = tuple(
    _rules(
        "Cap On Liability",
        (
            "in_no_event_liable",
            rf"\bin no event\b{ANY}{{0,120}}?\b{LIABLE}",
            "Standard opening of a limitation clause: 'In no event shall X be liable'.",
        ),
        (
            "excluded_damage_types",
            (
                rf"\b(indirect|incidental|consequential|special|punitive|exemplary)\b"
                rf"{ANY}{{0,80}}?\bdamages\b"
            ),
            "Exclusion of consequential/indirect damages, the most common cap form.",
        ),
        (
            "amount_ceiling",
            near(
                rf"\b{LIABLE}\b",
                r"\b(shall|will|to) (not|in no event) exceed\b|\bcapped at\b|"
                r"\blimited to (the|an) (amount|sum|aggregate|total)\b",
                150,
            ),
            "Liability tied to a ceiling: 'shall not exceed', 'limited to the amount'.",
        ),
    )
    + _rules(
        "Uncapped Liability",
        (
            "cap_with_carve_out",
            near(
                r"\bin no event\b|\blimitations? (of|on) liability\b|"
                r"\b(consequential|indirect) damages\b|\b(aggregate|total) liability\b",
                r"\bexcept (for|with respect to|in the case of|as)\b|\bexcluding\b|"
                r"\b(shall|will|does|do) not apply\b|\bother than\b|"
                r"\bnot limit\b|\bnotwithstanding\b",
                250,
            ),
            (
                "CUAD marks a cap as uncapped for carved-out breaches: "
                "'except for breach of confidentiality, in no event ...'."
            ),
        ),
    )
    + _rules(
        "Governing Law",
        (
            "governed_by_laws",
            rf"\bgovern(ed|s|ing)?\b{ANY}{{0,100}}?\blaws? of\b",
            "'governed by the laws of the State of ...'.",
        ),
        (
            "laws_govern",
            rf"\blaws? of\b{ANY}{{0,80}}?\bgovern(ed|s|ing)?\b",
            "Reversed order: 'The laws of Delaware shall govern'.",
        ),
        (
            "construed_under_laws",
            rf"\b(construed|interpreted|enforced)\b{ANY}{{0,80}}?\blaws? of\b",
            "'construed in accordance with the laws of ...'.",
        ),
        (
            "governing_law_heading",
            r"\bgoverning law\b",
            "Section heading for the clause.",
        ),
    )
    + _rules(
        "Anti-Assignment",
        (
            "assign_needs_consent",
            near(
                r"\bassign(s|ed|ment|able)?\b",
                r"\bconsent\b",
                200,
            ),
            "Assignment conditioned on the other party's consent.",
        ),
        (
            "may_not_assign",
            rf"\b({MUST_NOT})\b{ANY}{{0,40}}?\b(assign|transfer|delegate)\w*",
            "'shall not assign, transfer or delegate'.",
        ),
        (
            "no_party_assigns",
            (
                rf"\b(neither party|no party)\b{ANY}{{0,80}}?\bassign\w*|"
                r"\b(not|non-?) ?assignable\b"
            ),
            "'Neither party may assign ...' / 'not assignable'.",
        ),
        (
            "void_assignment",
            rf"\b(purported|attempted) (assignment|transfer)\b{ANY}{{0,80}}?\b(void|null)\b",
            "'Any purported assignment in violation hereof is void'.",
        ),
    )
    + _rules(
        "Change Of Control",
        (
            "change_of_control",
            r"\bchanges? (of|in) (the )?(control|ownership)\b",
            "The defined term itself.",
        ),
        (
            "merger_triggers_right",
            near(
                r"\b(merger|merges?|consolidat\w*|acqui(red|sition|res))\b",
                r"\bterminat\w*",
                150,
            ),
            "Merger or acquisition triggering a termination right.",
        ),
    )
    + _rules(
        "Termination For Convenience",
        (
            "terminate_without_cause",
            near(
                r"\b(may|can|right to|entitled to|option to) terminate\b|"
                r"\bmay be terminated\b",
                r"\bfor (its )?convenience\b|\bfor any (reason|or no reason)\b|"
                r"\bwithout cause\b",
                120,
            ),
            (
                "A termination right 'for convenience', 'without cause', 'for any reason'. "
                "Requires 'may terminate' or similar, because 'upon termination for any "
                "reason' is survival language."
            ),
        ),
        (
            "terminate_at_any_time",
            near(r"\b(may|can|right to) terminate\b", r"\bat any time\b", 80),
            "'may terminate this Agreement at any time upon notice'.",
        ),
    )
    + _rules(
        "Exclusivity",
        (
            "exclusive",
            (
                # CUAD writes "non-exclusive" also as "non- exclusive", "non exclusive"
                # and with soft or non-breaking hyphens.
                r"(?<!non[-\s\xad‐‑])(?<!non[-\xad‐‑]\s)"
                r"(?<!and\s)(?<!regulatory\s)\bexclusiv(e|ely|ity)\b"
                r"(?! (of|property|jurisdiction|remed(y|ies)|venue|forum|owners?(hip)?|"
                r"own|control|statement|benefit|by|and|to|agreement)\b)"
            ),
            (
                "Exclusive dealing or rights. Skips 'non-exclusive', 'sole and exclusive', "
                "'exclusive jurisdiction/remedy/property/owner', 'exclusive of taxes'."
            ),
        ),
        (
            "sole_supplier",
            (
                r"\bsole (and exclusive )?(supplier|distributor|provider|source|reseller|"
                r"agent|licensee|manufacturer|representative)\b"
            ),
            "Sole supplier/distributor appointment.",
        ),
    )
    + _rules(
        "Non-Compete",
        (
            "non_compete_term",
            r"\bnon-?compet\w*|\bnot (to )?compete\b|\bcompete with\b",
            "The covenant named directly.",
        ),
        (
            "restricted_competition",
            near(
                r"\bcompet(e|es|ing|itive|itor|itors)\b",
                r"\b(shall|will|may) not\b|\bagrees? not\b|\bdirectly or indirectly\b|"
                r"\brefrain\b",
                120,
            ),
            "A prohibition near competing products, businesses or competitors.",
        ),
    )
    + _rules(
        "Renewal Term",
        (
            "automatic_renewal",
            r"\bautomatic(ally)? (be )?(renew|extend)\w*|\bauto-?renew\w*|\bevergreen\b",
            "Automatic renewal or extension.",
        ),
        (
            "renewal_period",
            r"\brenewal (terms?|periods?)\b",
            "Defined renewal term or period.",
        ),
        (
            "successive_periods",
            near(
                r"\b(renew|extend)\w*",
                r"\b(successive|additional|further) \S+( \S+)? (years?|months?|"
                r"periods?|terms?)\b",
                100,
            ),
            "'renewed for successive one-year periods'.",
        ),
    )
    + _rules(
        "Audit Rights",
        (
            "audit",
            r"\baudit(s|ing|or|ors)?\b",
            (
                "The word itself (not 'audited', which mostly refers to financial "
                "statements)."
            ),
        ),
        (
            "inspect_records",
            near(
                r"\b(inspect|examin|review)\w*",
                r"\b(books|records|facilit(y|ies)|premises|accounts)\b",
                100,
            ),
            "Right to inspect or examine books, records or facilities.",
        ),
        (
            "access_to_records",
            rf"\baccess to\b{ANY}{{0,60}}?\b(books|records|accounts)\b",
            "Access to books and records.",
        ),
    )
)


def rule_table(rules=RULES) -> pd.DataFrame:
    """The rules as a table, for the documentation."""
    return pd.DataFrame(
        [
            {
                "category": r.category,
                "rule": r.name,
                "pattern": r.pattern,
                "rationale": r.rationale,
            }
            for r in rules
        ]
    )


def rule_hits(texts, rules=RULES) -> np.ndarray:
    """Boolean (n_texts, n_rules) array: does each rule match each text?"""
    texts = list(texts)
    hits = np.zeros((len(texts), len(rules)), dtype=bool)
    for j, rule in enumerate(rules):
        for i, text in enumerate(texts):
            hits[i, j] = rule.regex.search(text) is not None
    return hits


def category_scores(
    hits: np.ndarray, rules=RULES, categories=PRIORITY_CATEGORIES
) -> np.ndarray:
    """(n_texts, n_categories) count of distinct rules that fired per category.

    The prediction is ``score >= 1``. The count only serves to rank chunks for
    average precision; it is not a calibrated probability.
    """
    categories = list(categories)
    unknown = {r.category for r in rules} - set(categories)
    if unknown:
        raise ValueError(f"rules refer to unknown categories: {sorted(unknown)}")
    scores = np.zeros((hits.shape[0], len(categories)), dtype=float)
    for j, rule in enumerate(rules):
        scores[:, categories.index(rule.category)] += hits[:, j]
    return scores


def predict(texts, rules=RULES, categories=PRIORITY_CATEGORIES) -> np.ndarray:
    """(n_texts, n_categories) 0/1 predictions."""
    scores = category_scores(rule_hits(texts, rules), rules, categories)
    return (scores >= 1).astype(np.int8)


def explain(text: str, rules=RULES) -> pd.DataFrame:
    """Every rule that fires on ``text``, with the first match's offsets and text."""
    rows = []
    for rule in rules:
        m = rule.regex.search(text)
        if m:
            rows.append(
                {
                    "category": rule.category,
                    "rule": rule.name,
                    "start": m.start(),
                    "end": m.end(),
                    "match": m.group(0),
                }
            )
    return pd.DataFrame(rows, columns=["category", "rule", "start", "end", "match"])


def evaluate(
    labeled: pd.DataFrame, rules=RULES, categories=PRIORITY_CATEGORIES
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Score the rules on a labeled chunk table.

    Returns (per-category metrics, per-rule statistics, category scores).
    """
    categories = list(categories)
    labels, mask = label_arrays(labeled, len(categories))
    hits = rule_hits(labeled["text"], rules)
    scores = category_scores(hits, rules, categories)
    metrics = score(labels, mask, scores, categories, threshold=1)

    rows = []
    for j, rule in enumerate(rules):
        i = categories.index(rule.category)
        keep = mask[:, i]
        fired = hits[keep, j]
        y = labels[keep, i].astype(bool)
        only = fired & (scores[keep, i] == 1)  # this rule alone fired
        rows.append(
            {
                "category": rule.category,
                "rule": rule.name,
                "fired": int(fired.sum()),
                "true_positive": int((fired & y).sum()),
                "precision": (fired & y).sum() / fired.sum() if fired.any() else np.nan,
                "recall": (fired & y).sum() / y.sum() if y.any() else np.nan,
                "only_rule_fired": int(only.sum()),
                "only_rule_true_positive": int((only & y).sum()),
            }
        )
    stats = pd.DataFrame(rows).round({"precision": 4, "recall": 4})
    return metrics, stats, scores
