"""Inspect deterministic risk-rule behavior on representative CUAD gold spans.

Run with ``python -m scripts.validate_risk``. Gold annotations do not contain
risk levels or model confidence. For this rule-only exercise the script passes
confidence=1.0 as a documented placeholder; confidence contributes zero risk
points and is not interpreted as a model probability. The output is evidence
for rule behavior, not classification accuracy against gold risk labels.
"""

from __future__ import annotations

import argparse
import json

from src.cuad_data import PRIORITY_CATEGORIES, build_split
from src.risk import ADJUSTMENTS, score_clause


def validate(max_per_category: int = 3) -> list[dict]:
    """Score a deterministic, evenly spaced sample from each gold category."""
    if max_per_category < 1:
        raise ValueError("max_per_category must be at least 1")

    # Parse the checked-in CUAD JSON directly; this does not load the model or
    # create/modify derived dataset files.
    split = build_split("all")
    spans = split.priority().sort_values(["category", "contract", "start"])
    records = []
    for category in PRIORITY_CATEGORIES:
        group = spans.loc[spans.category == category].reset_index(drop=True)
        if group.empty:
            continue
        count = min(max_per_category, len(group))
        # Evenly spaced positions make the selection stable while spanning the
        # sorted category data rather than taking only its first contract.
        positions = sorted({round(i * (len(group) - 1) / max(count - 1, 1)) for i in range(count)})
        for position in positions:
            row = group.iloc[position]
            evidence = row["text"]
            triggered = [
                {
                    "name": rule.name,
                    "points": rule.points,
                    "matched_text": match.group(0),
                }
                for rule in ADJUSTMENTS
                if category in rule.categories
                if (match := rule.regex.search(evidence))
            ]
            # CUAD supplies human clause spans, not confidence. 1.0 is an
            # explicit placeholder only; the scorer correctly assigns it zero
            # risk points and this script does not run a model.
            result = score_clause(category, evidence, confidence=1.0)
            records.append(
                {
                    "contract": row["contract"],
                    "category": category,
                    "evidence_text": evidence,
                    "triggered_signals": triggered,
                    "score": result["score"],
                    "risk_level": result["risk_level"],
                    "rationale": result["rationale"],
                    "review_note": (
                        "Rule hit; verify operative right and context manually."
                        if triggered
                        else "No configured red-flag rule matched this span."
                    ),
                }
            )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-per-category", type=int, default=3)
    args = parser.parse_args()
    records = validate(args.max_per_category)
    print(json.dumps(records, ensure_ascii=False, indent=2))
    print(
        json.dumps(
            {
                "sampled_spans": len(records),
                "categories_covered": sorted({record["category"] for record in records}),
                "spans_with_rule_hits": sum(bool(r["triggered_signals"]) for r in records),
                "gold_risk_labels_available": False,
                "risk_accuracy_claimed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
