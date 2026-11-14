# Risk rubric

Use this first-pass rubric to prioritize clauses for review. The points reflect
the potential impact visible in the evidence. The reviewer can revise the
assessment after checking the parties, commercial terms, and referenced sections.
The initial weights and cutoffs are project assumptions for advisor validation.

Tasks [17](https://github.com/Break-Through-Tech/Accenture-1T-contract-review-challenge/issues/25)
and [19](https://github.com/Break-Through-Tech/Accenture-1T-contract-review-challenge/issues/27)
use this rubric for clause scoring and contract triage, respectively.

## 1. Base risk by clause type

| CUAD category | Points | What a reviewer looks for |
|---|---:|---|
| Cap On Liability | 1 | The amount, who it covers, and exceptions to the cap. |
| Uncapped Liability | 4 | Exposure outside the cap and which claims it covers. |
| Governing Law | 0 | The stated law and any supplied jurisdiction requirements. |
| Anti-Assignment | 2 | Consent, transfer restrictions, and exceptions. |
| Change Of Control | 2 | Consent or termination rights triggered by a transaction. |
| Termination For Convenience | 2 | Who may terminate, notice, and transition time. |
| Exclusivity | 2 | Who is restricted and the duration or scope of the restriction. |
| Non-Compete | 2 | Restricted activity, duration, geography, and affected parties. |
| Renewal Term | 1 | Renewal length and the window for opting out. |
| Audit Rights | 1 | Access scope, frequency, and notice. |

## 2. Red flags in the evidence

Add the **highest applicable** adjustment from the table. Count it once per
clause so several phrases describing the same right do not inflate the score.

| Points | When to apply |
|---|---|
| +2 | The passage expressly removes a liability limit for a stated claim or provides unlimited liability. |
| +2 | The passage expressly grants termination for convenience without prior notice or with immediate effect. |
| +2 | The passage makes renewal automatic and expressly bars opting out or canceling before renewal. |
| +1 | A party may decide in its sole discretion whether to allow assignment, transfer, or a change of control, or may terminate at its sole discretion. |
| +1 | The passage grants an active right to terminate without cause or for convenience, subject to notice or with notice terms unstated. |
| +1 | The passage grants audit access without prior notice. |
| 0 | The evidence supports none of these conditions. |

Match phrases case insensitively and ignore spacing differences for matching.
Tie the phrase to the operative right in its sentence or provision. Check who
holds that right and whether an exception, negation, or condition limits it.
For example, "may terminate without cause" can support the +1 rule. "Upon
termination without cause" by itself describes an event. "Without notice" must
modify the right being scored. If the excerpt omits a material notice period
or points to a missing exception, flag the need for context and score the
conditions supported by the excerpt. Absence of a notice phrase gives no points
for a no-notice rule.

Use the [keyword baseline](keyword-baseline.md) to identify candidate phrases.
A category match is only a starting point for checking the rule's conditions.

| Base + adjustment | Level |
|---:|---|
| 0–1 | Low |
| 2–3 | Medium |
| 4–6 | High |

The clause rationale should name the base category and points, cite the words
that triggered any adjustment, and state any review reason. If the category
falls outside the table or the evidence is empty, mark the item for review and
leave the score and level unset.

## 3. Missing expected clause

For the first version, support one expectation, **Cap On Liability**. Add
**0.5 points once per contract** to the contract triage score if the review
profile expects a monetary cap, the contract was processed in full, and the
available clause register has no evidence or unresolved candidate for a cap.
Report "Expected liability cap not found. Check the full contract."

Add **0 points** when a cap is present, when no cap is expected, or when the
expectation is unknown. If processing was incomplete or a cap candidate is
unresolved, add 0 points and flag the check for review. A damages exclusion by
itself leaves the monetary cap check unresolved if the passage may contain or
refer to a cap that needs review. This signal changes the contract triage
score. Each detected clause keeps its own score.

## 4. Confidence and human review

Model confidence carries **0 risk points**. It determines whether a reviewer
should check the detection. For a category threshold `t` fixed during
cross-validation, set the review boundary to `t + 0.10 × (1 - t)`. A model
score below the boundary goes to review. This covers scores below the
detection cutoff if they are passed to the scorer and the first tenth of the
remaining range above it. The category threshold and score must both be finite
values in [0, 1]. If either is missing or invalid for a model result, route it
to review. Compare the score and boundary using decimal values as exported. A score
equal to the review boundary clears the confidence check.

The margin is an initial routing choice. Positive class weighting can shift
model scores, so treat the margin as proximity to the cutoff. CUAD annotations
are a separate evidence source and skip the confidence check. Flag High
clauses and any excerpt with unresolved material context for human review
regardless of source or score.

## Scorer handoff

For each clause, carry its category, original evidence and offsets, and source
(CUAD or model). Report the base points, highest matching adjustment and its
triggering text, total score, risk level, and rationale. Keep the review flag
and reasons separate from the risk level. Use the frozen category cutoff from
the model run for the confidence check. Keep the missing-cap check and its
0 or 0.5 point adjustment on the contract record.

| Condition | Score | Human review |
|---|---|---|
| Recognized category and sufficient evidence | Base plus highest supported adjustment | If High or flagged by the confidence check |
| Recognized category with a missing referenced section or material context | Base plus any adjustment supported by the visible text | Yes, with the context needed |
| Unknown category or empty evidence | Unset | Yes |
| CUAD annotation with sufficient evidence | Base plus highest supported adjustment | If High |

## Examples

| Evidence | Result |
|---|---|
| A termination clause permits termination without cause on 60 days' written notice. | 2 base + 1 for the active right = 3, Medium. |
| A termination clause expressly permits immediate termination for convenience without notice. | 2 base + 2 for immediate termination = 4, High. |
| An uncapped-liability clause expressly removes a limit for fraud. | 4 base + 2 for the carve-out = 6, High. The rationale names the covered claim. |
| A governing-law clause has score 0.82 with a category threshold of 0.80. | 0 risk points, Low. A score of 0.82 sits on the review boundary. A lower score triggers review without changing the level. |
