"""Rule-based clause risk scorer for Task #17.

Implements docs/risk-rubric.md: base points by category, plus the single highest
red-flag adjustment found in the evidence, mapped to Low/Medium/High. Model
confidence never changes the score; it only routes clauses to human review.
"""

from __future__ import annotations

import math
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

# ------------------------------------------------------------------ clause scoring

REVIEW_MARGIN = 0.10  # rubric section 4


@dataclass(frozen=True)
class ClauseRisk:
    """Scorer output for one clause (rubric "Scorer handoff")."""

    category: str
    source: str
    base_points: int | None
    adjustment: str | None
    adjustment_points: int
    trigger_text: str
    score: int | None
    level: str | None
    needs_review: bool
    review_reasons: tuple[str, ...]
    rationale: str


def _finite_unit(x) -> bool:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return False
    return math.isfinite(x) and 0.0 <= x <= 1.0


def review_boundary(threshold: float) -> float:
    return round(float(threshold) + REVIEW_MARGIN * (1 - float(threshold)), 6)


def best_adjustment(category: str, text: str) -> tuple[Adjustment | None, str]:
    """Highest-point red flag for this category that matches ``text`` (ties: rubric order)."""
    best, trigger = None, ""
    for adj in ADJUSTMENTS:
        if category not in adj.categories:
            continue
        m = adj.regex.search(text)
        if m and (best is None or adj.points > best.points):
            best, trigger = adj, " ".join(m.group(0).split())
    return best, trigger


def score_clause(
    category: str,
    evidence: str,
    confidence: float | None = None,
    *,
    threshold: float | None = None,
    source: str = "model",
) -> ClauseRisk:
    """Score one clause with the Task #16 rubric.

    ``source`` is "model" or "cuad". Model clauses need ``confidence`` and the frozen
    category ``threshold`` for the confidence check; CUAD annotations skip it.
    Confidence never changes the score, only whether the clause goes to review.
    """
    reasons: list[str] = []
    text = evidence or ""
    if category not in BASE or not text.strip():
        why = "unknown category" if category not in BASE else "empty evidence"
        return ClauseRisk(category, source, None, None, 0, "", None, None, True, (why,),
                          f"{category}: not scored ({why}); needs human review.")

    base = BASE[category]
    adj, trigger = best_adjustment(category, text)
    points = adj.points if adj else 0
    score = base + points
    level = level_for(score)

    if level == "High":
        reasons.append("High risk")
    if source != "cuad":
        if not (_finite_unit(confidence) and _finite_unit(threshold)):
            reasons.append("missing or invalid confidence/threshold")
        elif float(confidence) < review_boundary(threshold):
            reasons.append(
                f"low confidence ({float(confidence):.3f} < boundary {review_boundary(threshold):.3f})"
            )

    parts = [f"{category} base {base}"]
    if adj:
        parts.append(f'+{points} {adj.name} ("{trigger[:120]}"): {adj.why}')
    else:
        parts.append("no red flag found")
    rationale = f"{'; '.join(parts)}. Total {score} = {level}."
    if reasons:
        rationale += f" Review: {', '.join(reasons)}."
    return ClauseRisk(category, source, base, adj.name if adj else None, points, trigger,
                      score, level, bool(reasons), tuple(reasons), rationale)
