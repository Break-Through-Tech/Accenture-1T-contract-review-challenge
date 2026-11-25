# Task #19: Triage score + end-to-end pipeline

**Status:** done for the 102 official test contracts, from both gold spans and the model register.

One command runs the whole chain:

```
contract JSON > chunks > model > clause register > clause risk > contract triage
```

and produces **clause category > evidence > risk level > rationale > triage score**.

## Run it

```bash
# test split, gold spans AND model register, plus the comparison (~45 s on CPU, no GPU)
uv run --locked python -B -m scripts.triage_pipeline

# only one source
uv run --locked python -B -m scripts.triage_pipeline --source gold
uv run --locked python -B -m scripts.triage_pipeline --source model

# new contracts ([{"contract": ..., "text": ...}]) through the saved Task #13 model
uv run --locked python -B -m scripts.triage_pipeline --contracts-json my_contracts.json \
    --checkpoint path/to/final_model/checkpoint
```

## Pipeline

| Step | Code | What happens |
|---|---|---|
| 1. Chunks + model | `src.chunking`, `src.clause_register.score_chunks` | Legal-bert chunking and the Task #13 model. On the test split, Task #13's official export is used instead of re-scoring (see Task #18). |
| 2. Clause register | `src.clause_register.build_register` / `src.triage.gold_register` | Model: Task #18 register. Gold: CUAD spans in the 10 categories, with overlapping spans of the same category merged the same way model chunks are, so both sources count clauses the same way. |
| 3. Clause risk | `src.risk.score_clause`, `src.triage.score_register` | Task #16 rubric: base points + the single highest red flag, mapped to Low/Medium/High, with a rationale citing the triggering words. Scores the Task #18 excerpt when there is one, else the full window. |
| 4. Review flags | `src.risk.score_clause` | High clauses; model clauses whose confidence is below `t + 0.10 × (1 − t)` (equal clears); missing or invalid confidence/threshold; unknown category or empty evidence (score left unset). CUAD clauses skip the confidence check. |
| 5. Missing cap | `src.triage.missing_cap_check` | +0.5 once per contract when a cap is expected, the contract was processed in full, and no Cap On Liability clause or unresolved candidate exists. 0 points + review when a model chunk scored in [0.5, threshold) for Cap On Liability, or when a clause excludes consequential/indirect damages without a cap. |
| 6. Triage | `src.triage.triage` | One row per contract, ranked (below). |

### Triage score

```
triage = Σ over categories ( worst clause score in that category × level weight )
         + missing-cap penalty (0 or 0.5)

level weight: Low 1.0, Medium 1.0, High 1.5
```

- **Worst clause per category, not every clause.** Gold data often has several spans of
  the same category (Cap On Liability has 118 spans in 44 contracts). Summing all of them
  would rank contracts by how verbose they are rather than by how risky.
- **High clauses get weighted up** so one uncapped-liability carve-out (6 × 1.5 = 9) is not
  outweighed by a few routine Medium clauses. The 1.5 is an initial assumption, like the
  rubric weights, and should be checked against the advisor (Lipika) rankings from Task #16.
- **Ranking:** triage score (descending), then number of High clauses, then contract name.
- Contracts with no detected clauses are still ranked (they get only the cap penalty).

### Contract triage columns

| Column | Meaning |
|---|---|
| `rank`, `triage_score` | Position and score |
| `clause_points`, `missing_cap_points`, `cap_status` | The two parts of the score, and why the cap penalty did or did not apply |
| `n_clauses`, `n_categories`, `n_high`, `n_medium`, `n_low`, `worst_level` | Clause counts |
| `clauses_for_review`, `needs_review` | Review routing (clause flags or an unresolved cap check) |
| `categories`, `top_risks`, `top_rationale` | What drove the score, for the reviewer |

## Results on the 102 test contracts

| | Gold spans | Model register |
|---|---:|---:|
| Clauses | 731 | 497 |
| Low / Medium / High | 339 / 376 / 16 | 240 / 238 / 19 |
| Clauses flagged for review | 16 | 22 |
| Contracts with a High clause | 13 | 16 |
| Missing-cap penalty applied | 58 | 55 |
| Cap check unresolved | 0 | 1 |
| Median / max triage score | 5.5 / 25.0 | 5.5 / 23.0 |

### Does the model-driven ranking match the gold ranking?

| Metric | Value |
|---|---:|
| Spearman correlation of triage scores | **0.89** |
| Top-10 overlap | **8 / 10** |
| Top-20 overlap | 13 / 20 |
| Mean absolute rank difference | 8.5 places |
| Same worst level (Low/Medium/High) | 84% |

For model clauses that overlap a gold clause of the same category (421 pairs), the risk
level agrees **98%** of the time. The only real disagreement is **Termination For
Convenience (76%)**: in 6 of 25 pairs the model scores higher, because its excerpt or
window also includes "may terminate ... immediately" wording that usually belongs to a
termination-for-breach provision next door, which fires the +2 no-notice rule.

### What the numbers say

- **Switching from gold spans to the model barely changes who reviewers open first.** Eight of
  the top 10 contracts are the same and the scores correlate at 0.89.
- **Base points drive most of the score.** Red flags fire on only ~4% of clauses
  (mostly Termination For Convenience). The Task #17 patterns are deliberately narrow; widening
  them (e.g. more "unlimited liability" wordings) is the main lever for a more discriminating
  triage.
- **The missing-cap signal is common.** Over half the test contracts have no Cap On Liability
  clause, so the 0.5 penalty affects mostly the low end of the ranking, as the rubric intended.
- **Known gaps:**
  - The scorer does not check negation or who holds the right (rubric section 2 asks for this).
  - The damages-exclusion check only sees the 10 priority categories, not the whole contract.
  - The triage weights are unvalidated until the advisor rankings come back.

## Files

| File | Contents | In Git |
|---|---|---|
| `docs/triage-results/{gold,model}_contract_triage.csv` | Ranked contracts, 102 rows each | Yes |
| `docs/triage-results/{gold,model}_clause_risk.csv` | One row per clause: category, offsets, evidence excerpt (600 chars), confidence, risk score/level, red flag, rationale, review reasons | Yes |
| `docs/triage-results/clause_level_agreement.csv` | Gold vs model risk level per category | Yes |
| `docs/triage-results/metadata.json` | Weights, run summaries, gold-vs-model metrics | Yes |
| `data/interim/triage/*_clause_risk.parquet` | Same clause tables with full evidence windows | No |

## Checks

```bash
uv run --locked python -B -m pytest -q -p no:cacheprovider tests/test_triage.py tests/test_risk.py
uv run --locked ruff check --no-cache src/risk.py src/triage.py scripts/triage_pipeline.py tests/test_triage.py
```

`tests/test_triage.py` (26 tests) covers the clause scorer (rubric examples, highest
adjustment only, unscorable clauses, confidence routing with the equal-boundary rule, invalid
inputs, CUAD skipping the confidence check), gold span merging, excerpt-over-window scoring,
every missing-cap branch, the triage formula, ranking and tie-breaks, and the ranking
comparison. The full suite (288 tests) passes.
