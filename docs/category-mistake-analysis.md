# Sampled mistake analysis by category and recommendations for November

## Scope
This note samples the actual test-set mistakes of the final clause classifier, groups them into recurring error families, explains the Non-Compete CV-vs-test gap, and recommends follow-up work for November. It closes the analysis task that depended on the final test run (Task #13, issue #17).

## Evidence and method
Unlike an impression-based review, every number here is reconstructed from the committed outputs:

- Predictions: [docs/final-model-results/test_predictions.parquet](./final-model-results/test_predictions.parquet)
(per-chunk probabilities for all 10 categories, 3,029 test chunks).
- Gold labels: rebuilt from [data/interim/test/spans.parquet](../data/interim/test/spans.parquet) using the Task #6 rule — a chunk is **positive** for a category when a gold span of that category is *fully contained* in the chunk, and **masked** (excluded from scoring) when a span only partially overlaps (see [docs/chunking-labels.md](./chunking-labels.md)).
- Thresholds: the frozen per-category thresholds from [docs/final-model-results/selection.json](./final-model-results/selection.json).

Applying the frozen thresholds to the gold labels reproduces every per-category precision, recall and F1 in [docs/final-model-results/test_per_category.csv](./final-model-results/test_per_category.csv) and [docs/final-model.md](./final-model.md) exactly, so the false positives and false negatives sampled below are the real ones, not estimates. Analysis is per chunk, matching how the model is scored.

## Error counts per category (test split, frozen thresholds)
FP = model predicted the category but no full gold span is present; FN = a full gold span is present but the model did not predict it. "FP with nothing related" counts FP chunks that contain **no** priority-category gold at all — the model fired on language the annotator did not mark.

| Category | Support | F1 | FP | FN | FP with nothing related |
|---|---:|---:|---:|---:|---:|
| Cap On Liability | 105 | 0.850 | 14 | 17 | 10 |
| Uncapped Liability | 16 | 0.759 | 2 | 5 | 0 |
| Governing Law | 126 | 0.935 | 14 | 3 | 12 |
| Anti-Assignment | 129 | 0.857 | 15 | 21 | 8 |
| Change Of Control | 56 | 0.720 | 8 | 20 | 5 |
| Termination For Convenience | 43 | 0.753 | 10 | 11 | 4 |
| Exclusivity | 71 | 0.694 | 25 | 20 | 4 |
| Non-Compete | 53 | 0.764 | 15 | 11 | 3 |
| Renewal Term | 32 | 0.829 | 9 | 3 | 1 |
| Audit Rights | 90 | 0.866 | 7 | 16 | 6 |

## Executive summary
Most mistakes are not random. Grouping the sampled errors gives four recurring families, ordered here by how much of the error mass they actually explain:

1. **Semantic overlap inside clause families** — the single largest driver. Restrictive-covenant categories (Exclusivity, Non-Compete) and assignment/control categories (Anti-Assignment, Change Of Control) trade errors with each other and with near-neighbours that are *not* in the 10-label set (Rofr/Rofo/Rofn, Minimum Commitment, License Grant).
2. **Cap vs Uncapped Liability confusion** — small in volume but systematic: every Uncapped error involves a co-located Cap span.
3. **Annotation scope / repeated mentions** — the model fires on real category language that the gold set did not mark, or misses one of several co-located mentions.
4. **Negation and exception handling** — a smaller, cross-cutting contributor, mostly in the liability family.

A note on what is **not** a leading cause: clauses physically cut off at a chunk boundary are already handled by the labeling pipeline (partial-overlap spans are masked out of scoring — only 100 of 3,029 chunks are masked for any category). The scored false negatives are overwhelmingly chunks that *fully contain* the gold span but still got a low probability, so chunk truncation is a real risk to guard against going forward but is not the dominant observed error. This corrects an earlier framing of cut-off clauses as the most common failure mode.

## Pattern 1: Semantic overlap inside clause families (largest driver)
The clearest real pattern. Categories that share legal vocabulary trade false positives and false negatives, and several of the strongest confusers are categories outside the 10-label set.

Measured confusion partners (other gold present in the category's FP chunks):

- **Exclusivity** (25 FP — the most of any category) fires on **Rofr/Rofo/Rofn (7)**, Non-Compete, Joint IP Ownership and Covenant-Not-To-Sue chunks. Its 20 FNs co-occur most with License Grant (6) and Competitive Restriction Exception.
- **Non-Compete** (15 FP) fires on **Exclusivity (6)**, License Grant and Rofr/Rofo/Rofn chunks; its FNs co-occur with Exclusivity and Minimum Commitment.
- **Anti-Assignment** (15 FP / 21 FN — the most FNs of any category) confuses with Non-Transferable License, Change Of Control and Rofr/Rofo/Rofn; its FNs sit in dense Affiliate-License chunks.
- **Change Of Control** (20 FN) misses spans buried in dense multi-label chunks that also carry Anti-Assignment (6) and License Grant (4) gold.

Real examples:
- Non-Compete predicted on an **Exclusivity** clause: *"4.17 PPI as Exclusive Provider. During the Term … EKR shall purchase all [Products] …"*.
- Exclusivity predicted on a **Rofr** clause: *"Excite will offer Sponsor the right of first refusal to negotiate …"*.

### Why this matters
These are the categories that already carry the lowest F1 (Exclusivity 0.694, Change Of Control 0.720, Termination For Convenience 0.753), so the overlap directly caps macro F1.

### Recommendation
- Treat the restrictive-covenant family (Exclusivity / Non-Compete / Rofr-Rofo-Rofn) and the assignment-control family (Anti-Assignment / Change Of Control) as deliberate disambiguation targets, not independent heads.
- Several top confusers (Rofr/Rofo/Rofn, Minimum Commitment, License Grant) are out-of-scope categories. Consider training on the full CUAD label set and reporting the 10, so the model can tell a neighbour apart instead of being forced to choose among 10.

## Pattern 2: Cap vs Uncapped Liability confusion
Low volume (Uncapped has only 16 test positives) but completely systematic: **all** of Uncapped Liability's errors involve a co-located Cap On Liability span. Both Uncapped false positives land on chunks whose gold is Cap, and all 5 Uncapped false negatives are chunks that also carry a Cap gold — the model sees the dominant cap language and predicts Cap, suppressing Uncapped.

- Cap On Liability: *"liability shall not exceed fees paid"*, *"aggregate liability capped at X"*.
- Uncapped Liability: *"liability is not limited"*, *"unlimited liability"*, carve-outs that restore full liability for fraud, gross negligence or IP infringement.

### Why this matters
This pair maps directly onto legal and financial risk severity, so a Cap/Uncapped swap is a high-impact error even though the raw count is small. Uncapped's 0.759 F1 rests on only 13 predictions, so it is also noisy.

### Recommendation
- Add a focused Cap-vs-Uncapped tie-break: when both are plausible in a chunk, apply phrase rules for cap language ("cap", "not exceed", "maximum liability", "aggregate liability") vs unlimited language ("not limited", "unlimited", "shall not apply to", "nothing … shall limit").
- Give special weight to carve-out language that restores liability (fraud, gross negligence, IP), which is the usual Uncapped trigger the model misses.

## Pattern 3: Annotation scope and repeated mentions
Some sampled "mistakes" are the model finding category language the gold set did not mark. This shows up as false positives with **no related gold at all**: Governing Law (12 of 14 FPs), Cap On Liability (10 of 14) and Anti-Assignment (8 of 15). It also shows up as missed secondary mentions when a category appears several times in a contract but only one span is annotated.

Real examples:
- Change Of Control predicted on a **definition** of "Change of Control" (*"… Voting Stock … fifty percent (50%) or more of the total voting power …"*) that was not annotated as the operative clause.

### Why this matters
These inflate the false-positive count and make some errors look worse than they are, especially for high-recall categories (Governing Law recall 0.976).

### Recommendation
- Manually review a sample of the "nothing related" false positives per category to separate genuine model errors from annotation gaps.
- Deduplicate to contract level before reporting operational metrics, so a missed secondary mention of an already-detected category is not double-counted.

## Pattern 4: Negation and exception handling
A smaller, cross-cutting contributor, concentrated in the liability family. The model keys on a trigger word and misses that a negation or exception inverts its meaning.

Real examples:
- A **warranty disclaimer** predicted as Cap On Liability: *"THE PARTIES EXPRESSLY EXCLUDE ANY IMPLIED WARRANTIES OF FITNESS …"* (no liability cap present).
- A liability **carve-out** predicted as Cap: *"Nothing in this Clause shall exclude or restrict the liability of either Party arising out of fraud …"* (this is closer to uncapped).

### Recommendation
- Add light negation/exception awareness around liability triggers ("nothing … shall limit", "shall not apply to", "except as", "notwithstanding"), and distinguish warranty-disclaimer language from liability-cap language.

## Explaining Non-Compete's CV-vs-test gap
Non-Compete scores markedly higher on the test split than in cross-validation, and the gap is visible on both F1 and the threshold-independent average precision:

| Metric | CV (out-of-fold) | Test |
|---|---:|---:|
| F1 | 0.617 | 0.764 |
| Average precision | 0.529 | 0.813 |
| Positive chunks (support) | 197 | 53 |

(Source: [cv_per_category.csv](./final-model-results/cv_per_category.csv) and [test_per_category.csv](./final-model-results/test_per_category.csv).)

The most telling number is **average precision**, which jumps 0.53 → 0.81. AP does not depend on the chosen threshold, so this is not a threshold artifact — the test contracts genuinely rank Non-Compete chunks more cleanly than the CV folds do. Combined with low support (53 test positives),
this points to a **distributional effect**: the test split happens to contain clearer, more explicit Non-Compete language, while the validation folds contain more ambiguous or heterogeneous phrasing.

This reading is consistent with:
- the same direction of gap in the TF-IDF baseline (Task #9), i.e. it is a property of the split, not of legal-bert;
- Non-Compete's low support, which makes the category sensitive to contract mix;
- `final-model.md` flagging Non-Compete as low-support and deferring it to Task #15.

In short, the split appears favorable for Non-Compete on test — the model did not suddenly improve; the test contracts are easier to classify for that category. Because support is low, treat the 0.764 as optimistic and report it with its CV number alongside.

## Recommendations for November
1. **Target the overlap families, not individual heads.**
   - Disambiguate Exclusivity / Non-Compete / Rofr-Rofo-Rofn and Anti-Assignment / Change Of Control explicitly.
   - Evaluate training on the full CUAD label set (reporting the 10) so near-neighbours like Rofr/Rofo/Rofn and Minimum Commitment stop being misread as in-scope categories.

2. **Add legal-logic post-processing for the liability family.**
   - Cap vs Uncapped tie-break with phrase rules.
   - Negation/exception and warranty-disclaimer checks around liability triggers.

3. **Run an annotation-quality pass on the weakest categories.**
   - Review the "nothing related" false positives for Governing Law, Cap On Liability and Anti-Assignment to separate annotation gaps from model errors.
   - Prioritize Change Of Control, Exclusivity and Non-Compete, which also carry the lowest F1.

4. **Guard against chunk truncation even though it is not the top observed error.**
   - Keep paragraph-aware chunking and adequate overlap so trigger phrases and their exceptions stay in one chunk; re-check the masked-chunk rate if chunking changes.

5. **Use a focused manual-review set.**
   - Sample 20–30 high-risk FPs and FNs for Exclusivity, Change Of Control, Anti-Assignment and Non-Compete, label each with the four families above, and feed the counts into the next model and rule iteration.

6. **Calibrate probabilities before using them as risk scores.**
   - `pos_weight` inflates probabilities (hence thresholds up to 0.95); thresholds fix the decisions, not the scores. Calibrate before surfacing probabilities to reviewers.

7. **Treat legal explainability as a first-class requirement.**
   - Surface the clause evidence behind each prediction, so a reviewer can tell a genuine Cap/Uncapped call from an overlap or annotation artifact.

## Final conclusion
The dominant issue is recurring legal-language complexity — semantic overlap between related clause families, cap-vs-uncapped semantics, annotation scope, and negation — not broad random error, and not (as first thought) chunk truncation, which the labeling pipeline already handles. The highest-value work for November is targeted disambiguation and legal-logic post-processing for the specific confused families, plus an annotation-quality pass on the weakest categories, rather than broad retraining. This directly addresses the errors that drive legal-risk decisions and review
prioritization.
