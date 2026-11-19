"""Rule-based clause risk scorer for Task #17.

Implements docs/risk-rubric.md: base points by category, plus the single highest
red-flag adjustment found in the evidence, mapped to Low/Medium/High. Model
confidence never changes the score; it only routes clauses to human review.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.keyword_baseline import ANY, near

# Section 1 of the rubric: base risk by clause type
BASE = {
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

# Score -> level (rubric: 0-1 Low, 2-3 Medium, 4-6 High)
def level_for(score: int) -> str:
    if score >= 4:
        return "High"
    if score >= 2:
        return "Medium"
    return "Low"


def score_clause(category: str, evidence: str, confidence: float | None = None) -> dict:
    """Score one clause using the Task #17 rubric.

    Confidence is accepted for callers that also have a model prediction, but
    never changes risk points. The rubric's confidence review boundary requires
    the model's frozen category threshold, which is intentionally not guessed
    here.

    Invalid categories or empty/non-string evidence return unset score and level
    so callers can route the clause for human review.
    """
    if not isinstance(category, str) or category not in BASE:
        return {
            "risk_level": None,
            "score": None,
            "rationale": f"Unknown category {category!r}; review required.",
        }
    if not isinstance(evidence, str) or not evidence.strip():
        return {
            "risk_level": None,
            "score": None,
            "rationale": f"{category}: evidence is empty or invalid; review required.",
        }

    base = BASE[category]
    rationale = [f"{category} base risk: {base} point(s)."]

    # The rubric adds only the highest applicable adjustment once per clause.
    # Keep the actual matched phrase in the explanation so a reviewer can
    # verify why it fired.
    matches = []
    for adjustment in ADJUSTMENTS:
        if category not in adjustment.categories:
            continue
        match = adjustment.regex.search(evidence)
        if match:
            matches.append((adjustment, match.group(0)))

    if matches:
        adjustment, trigger = max(matches, key=lambda item: item[0].points)
        rationale.append(
            f"+{adjustment.points} for {adjustment.why} Trigger: {trigger!r}."
        )
    else:
        adjustment = None

    score = base + (adjustment.points if adjustment else 0)
    rationale.append(f"Total: {score} point(s), {level_for(score)} risk.")

    # Confidence is an independent review signal, not a risk point. Since this
    # API has no frozen category threshold, do not infer a review boundary.
    if confidence is not None:
        rationale.append("Model confidence does not change risk points.")

    return {"risk_level": level_for(score), "score": score, "rationale": " ".join(rationale)}


@dataclass(frozen=True)
class Adjustment:
    name: str
    points: int
    categories: frozenset[str]
    pattern: str
    why: str
    regex: re.Pattern = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(
            self, "regex", re.compile(self.pattern.replace(" ", r"\s+"), re.IGNORECASE)
        )


TERMINATE = r"terminat(e|ion|ing)"
MAY = r"(may|can|shall have the right to|is entitled to|reserves the right to)"

# Section 2 of the rubric: red flags. Only the highest one that matches counts.
ADJUSTMENTS: tuple[Adjustment, ...] = (
    Adjustment(
        "liability_carve_out", 2,
        frozenset({"Uncapped Liability", "Cap On Liability"}),
        rf"unlimited liability|liability{ANY}{{0,40}}?unlimited"
        rf"|nothing{ANY}{{0,120}}?(shall|will) (limit|exclude|restrict)"
        rf"|shall not apply to{ANY}{{0,80}}?(fraud|gross negligence|willful|indemnif)",
        "Removes a liability limit for a stated claim or makes liability unlimited.",
    ),
    Adjustment(
        "immediate_termination", 2,
        frozenset({"Termination For Convenience"}),
        near(rf"{MAY} {TERMINATE}", r"immediately|immediate effect|without (prior |any )?notice", 80),
        "Termination for convenience with no notice or immediate effect.",
    ),
    Adjustment(
        "renewal_no_opt_out", 2,
        frozenset({"Renewal Term"}),
        near(r"automatic(ally)? renew", r"(may not|shall not|cannot|can not) (be )?(cancel|terminat|opt)", 150),
        "Automatic renewal that expressly bars cancelling before renewal.",
    ),
    Adjustment(
        "sole_discretion", 1,
        frozenset({"Anti-Assignment", "Change Of Control", "Termination For Convenience"}),
        r"sole (and absolute )?discretion",
        "A party decides in its sole discretion on assignment, change of control or termination.",
    ),
    Adjustment(
        "active_termination_right", 1,
        frozenset({"Termination For Convenience"}),
        rf"{MAY} {TERMINATE}{ANY}{{0,100}}?(without cause|for (its |their )?convenience|for any reason|at any time)",
        "An active right to terminate without cause or for convenience.",
    ),
    Adjustment(
        "audit_without_notice", 1,
        frozenset({"Audit Rights"}),
        r"without (prior |advance |any )?notice",
        "Audit access without prior notice.",
    ),
)
