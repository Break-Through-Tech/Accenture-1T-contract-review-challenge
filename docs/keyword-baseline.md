# Task #8: Keyword/rule baseline

A transparent baseline: hand-written regular expressions built from legal phrasing,
with no fitted parameters. Every prediction can be traced to a named rule and the exact
words that triggered it. It sits beside the [TF-IDF baseline](tfidf-baseline.md) as a
second floor for the October transformer, and as a reference for what plain search
terms already achieve.

## Run it

The rules read the labeled chunk tables from task #6. Build them for both splits first
(outputs are ignored by Git):

```bash
uv sync --locked
uv run --locked python -B -m scripts.export_chunks --split train
uv run --locked python -B -m scripts.label_chunks  --split train
uv run --locked python -B -m scripts.export_chunks --split test
uv run --locked python -B -m scripts.label_chunks  --split test
uv run --locked python -B -m scripts.keyword_baseline
```

The last command takes about a minute and writes to `docs/keyword-baseline-results/`:

- `rules.csv`: every rule with its full pattern and rationale
- `train_per_category.csv`, `test_per_category.csv`: per-category metrics
- `train_rule_stats.csv`, `test_rule_stats.csv`: how often each rule fires and how often it is right
- `metadata.json`: input hashes and counts

To see why a chunk was flagged:

```python
from src.keyword_baseline import explain
explain(chunk_text)  # category, rule, start, end, matched text
```

## Method

| Step | Choice |
|---|---|
| Input | Chunk `text`, exactly as exported in task #6 |
| Matching | Case-insensitive regular expressions; a space in a pattern matches any run of whitespace, because CUAD keeps SEC-filing spacing and line breaks |
| Decision | A chunk is positive for a category if at least one of that category's rules matches |
| Score for average precision | Number of distinct rules of the category that matched (a ranking, not a probability) |
| Masking and metrics | Same as task #7: masked (chunk, category) pairs are excluded; scoring reuses `src.tfidf_baseline.score` |
| Categories | The ten in `PRIORITY_CATEGORIES` |

### How the rules were chosen

1. Candidate phrases came from the most frequent three-word phrases in the **training**
   annotations of each category, plus standard contract wording.
2. The rules were scored on the training chunks. Rules whose matches were mostly wrong
   were tightened or removed by inspecting training false positives. This took two
   rounds, and the changes are listed under [Development history](#development-history).
3. The rules were then frozen and the test split was scored. A code review afterwards
   found that the `exclusive` rule missed two spellings of "non-exclusive" that its
   description says it skips. That bug was fixed and both splits were rescored; no rule
   was changed because of test results.

Training scores are therefore optimistic: the rules were adjusted while looking at them.
The test split was not used to write or change any rule.

## The rules

26 rules. `rules.csv` and [src/keyword_baseline.py](../src/keyword_baseline.py) hold the
exact patterns. "Near" means within the given number of characters, in either order.

| Category | Rule | Matches |
|---|---|---|
| Cap On Liability | `in_no_event_liable` | "in no event" followed within 120 characters by "liable/liability" |
| | `excluded_damage_types` | indirect, incidental, consequential, special, punitive or exemplary ... "damages" |
| | `amount_ceiling` | "liability" near (150) "shall not exceed", "capped at", "limited to the amount/sum/aggregate/total" |
| Uncapped Liability | `cap_with_carve_out` | liability-limit language ("in no event", "limitation of liability", "consequential damages", "aggregate liability") near (250) an exception: "except for", "excluding", "shall not apply", "other than", "notwithstanding" |
| Governing Law | `governed_by_laws` | "govern(ed/s/ing)" followed within 100 characters by "law(s) of" |
| | `laws_govern` | "law(s) of" followed by "govern" ("The laws of Delaware shall govern") |
| | `construed_under_laws` | "construed/interpreted/enforced" ... "law(s) of" |
| | `governing_law_heading` | "governing law" |
| Anti-Assignment | `assign_needs_consent` | "assign/assignment/assignable" near (200) "consent" |
| | `may_not_assign` | "shall/will/may/must not", "cannot" followed by assign/transfer/delegate |
| | `no_party_assigns` | "neither party / no party" ... "assign"; "not assignable", "non-assignable" |
| | `void_assignment` | "purported/attempted assignment" ... "void/null" |
| Change Of Control | `change_of_control` | "change of/in control", "change of/in ownership" |
| | `merger_triggers_right` | merger/consolidation/acquisition near (150) "terminat..." |
| Termination For Convenience | `terminate_without_cause` | "may terminate", "right to terminate", "may be terminated" near (120) "for convenience", "for any reason", "without cause" |
| | `terminate_at_any_time` | "may/can/right to terminate" near (80) "at any time" |
| Exclusivity | `exclusive` | "exclusive/exclusively/exclusivity", excluding "non-exclusive" (also written "non- exclusive" and "non exclusive" in CUAD), "sole and exclusive", "regulatory exclusivity", and "exclusive" followed by of, property, jurisdiction, remedy, venue, forum, owner(ship), control, and similar |
| | `sole_supplier` | "sole (and exclusive) supplier/distributor/provider/source/reseller/agent/licensee/..." |
| Non-Compete | `non_compete_term` | "non-compete", "not (to) compete", "compete with" |
| | `restricted_competition` | compete/competing/competitive/competitor near (120) "shall/will/may not", "agree(s) not", "directly or indirectly", "refrain" |
| Renewal Term | `automatic_renewal` | "automatically renew/extend", "auto-renew", "evergreen" |
| | `renewal_period` | "renewal term(s)/period(s)" |
| | `successive_periods` | renew/extend near (100) "successive/additional/further ... year(s)/month(s)/period(s)/term(s)" |
| Audit Rights | `audit` | "audit", "audits", "auditing", "auditor(s)" (not "audited") |
| | `inspect_records` | inspect/examine/review near (100) books, records, facilities, premises, accounts |
| | `access_to_records` | "access to" ... books/records/accounts |

## Results on the official test split

102 contracts, 3,096 chunks. Support is the number of positive chunks.

| Category | Support | Precision | Recall | F1 | Avg. precision |
|---|---:|---:|---:|---:|---:|
| Cap On Liability | 107 | 0.675 | 0.523 | 0.590 | 0.429 |
| Uncapped Liability | 17 | 0.210 | 0.471 | 0.291 | 0.102 |
| Governing Law | 116 | 0.763 | 0.914 | 0.831 | 0.827 |
| Anti-Assignment | 134 | 0.726 | 0.791 | 0.757 | 0.710 |
| Change Of Control | 58 | 0.329 | 0.448 | 0.380 | 0.170 |
| Termination For Convenience | 44 | 0.655 | 0.432 | 0.520 | 0.330 |
| Exclusivity | 68 | 0.235 | 0.765 | 0.360 | 0.196 |
| Non-Compete | 53 | 0.394 | 0.736 | 0.513 | 0.387 |
| Renewal Term | 31 | 0.561 | 0.742 | 0.639 | 0.535 |
| Audit Rights | 90 | 0.444 | 0.789 | 0.568 | 0.471 |
| **Macro average** | 718 | 0.499 | 0.661 | 0.545 | 0.416 |
| **Micro average** | 718 | 0.489 | 0.705 | 0.577 | |

## Results on the training split

408 contracts, 14,266 chunks. The rules were written on this split, so these numbers
are optimistic.

| Category | Support | Precision | Recall | F1 | Avg. precision |
|---|---:|---:|---:|---:|---:|
| Cap On Liability | 532 | 0.759 | 0.592 | 0.665 | 0.532 |
| Uncapped Liability | 150 | 0.294 | 0.547 | 0.382 | 0.165 |
| Governing Law | 495 | 0.806 | 0.952 | 0.873 | 0.900 |
| Anti-Assignment | 512 | 0.661 | 0.834 | 0.738 | 0.723 |
| Change Of Control | 181 | 0.376 | 0.602 | 0.463 | 0.254 |
| Termination For Convenience | 236 | 0.680 | 0.487 | 0.568 | 0.347 |
| Exclusivity | 315 | 0.296 | 0.832 | 0.436 | 0.250 |
| Non-Compete | 198 | 0.439 | 0.475 | 0.456 | 0.276 |
| Renewal Term | 210 | 0.594 | 0.662 | 0.626 | 0.547 |
| Audit Rights | 425 | 0.469 | 0.864 | 0.608 | 0.560 |
| **Macro average** | 3,254 | 0.537 | 0.685 | 0.582 | 0.456 |
| **Micro average** | 3,254 | 0.529 | 0.732 | 0.614 | |

## Comparison with the TF-IDF baseline (test F1)

| Category | Keyword | TF-IDF | Keyword recall | TF-IDF recall |
|---|---:|---:|---:|---:|
| Cap On Liability | 0.590 | 0.736 | 0.523 | 0.664 |
| Uncapped Liability | 0.291 | 0.667 | 0.471 | 0.941 |
| Governing Law | 0.831 | 0.868 | 0.914 | 0.931 |
| Anti-Assignment | 0.757 | 0.788 | 0.791 | 0.761 |
| Change Of Control | 0.380 | 0.442 | 0.448 | 0.328 |
| Termination For Convenience | 0.520 | 0.555 | 0.432 | 0.636 |
| Exclusivity | 0.360 | 0.511 | 0.765 | 0.500 |
| Non-Compete | 0.513 | 0.725 | 0.736 | 0.623 |
| Renewal Term | 0.639 | 0.824 | 0.742 | 0.903 |
| Audit Rights | 0.568 | 0.720 | 0.789 | 0.656 |
| **Macro average** | **0.545** | **0.683** | 0.661 | 0.694 |

## What the numbers say

- TF-IDF beats the rules on every category's F1. The gap is smallest where the wording
  is fixed: Governing Law (0.83 vs 0.87), Anti-Assignment (0.76 vs 0.79) and Termination
  For Convenience (0.52 vs 0.55). Much of the TF-IDF result for these categories can be
  reproduced with a handful of readable phrases.
- The rules have **higher recall than TF-IDF** on Exclusivity, Change Of Control,
  Non-Compete, Audit Rights and Anti-Assignment, at much lower precision. The keywords
  are present but also appear in chunks that CUAD did not label.
- Precision problems come from words that are necessary but not sufficient. In test,
  `exclusive` fired on 218 chunks and 52 were positive; `audit` fired on 118 and 56 were
  positive. "Exclusive" also describes licences, ownership and remedies; "audit" also
  describes financial statements and auditors.
- Uncapped Liability is the hardest category for rules (test F1 0.29). Its clauses use
  the same words as Cap On Liability, and whether a carve-out actually removes the cap
  depends on sentence structure that a nearby-words rule cannot read.
- Change Of Control is weak (0.38). The literal "change of control" matched 60 test
  chunks but only 24 positives. The phrase often appears in definitions or in
  anti-assignment exceptions that CUAD did not label as this category.
- Non-Compete scores higher on test (0.51) than on train (0.46). As with TF-IDF, the
  test number for this category moves a lot between splits; do not read it as a gain.
- On training data, the rules score 0.456 on Non-Compete against 0.295 for TF-IDF
  cross-validation. Short covenant phrasing ("shall not ... compete") is easier to write
  as a rule than to learn from 198 positive chunks.

## Limitations

- **Presence, not meaning.** A rule detects that a phrase is present. It does not know
  who the obligation binds, whether it is negated, or whether a later sentence overrides
  it. "Licensee may assign this Agreement without Licensor's consent" triggers the
  Anti-Assignment consent rule even though it grants the right to assign.
- **Fixed window sizes.** "Near" rules use character windows (80 to 250). A clause whose
  trigger and qualifier are farther apart is missed; unrelated sentences that happen to
  be close can match.
- **Headings, tables of contents and definitions.** The phrase "Governing Law" in a table
  of contents or the definition of "Change of Control" fires the rule without the clause
  itself. A rule for the "Limitation of Liability" heading was dropped for this reason.
- **Overlap between categories.** Cap On Liability and Uncapped Liability share
  vocabulary, and so do Anti-Assignment and Change Of Control, and Exclusivity and
  Non-Compete. Rules cannot separate them reliably.
- **Wording outside the list.** Paraphrases, unusual drafting, other jurisdictions'
  terms and OCR or filing artifacts are missed. Redacted text (`[***]`) can remove the
  amount or trigger a rule needs.
- **Development bias.** The rules were adjusted on the training split. Their training
  scores are optimistic, and a new contract collection may need new rules.
- **Chunk level only.** As in task #7, scores are per chunk. Five of the 737 test
  annotations are not fully inside any chunk, so no chunk-level method can be credited
  for them.
- **Ranking is coarse.** The score counts matching rules, so most chunks tie at 1.
  Average precision is reported for comparability but is low by construction.

## Development history

All changes were made on the training split before test was scored.

| Round | Train macro F1 | Change |
|---|---:|---|
| Initial draft | 0.495 | 32 rules from frequent training phrases |
| 1 | 0.563 | Governing Law heading limited to "governing law"; with "applicable law" included it was the only rule to fire on 1,183 chunks, 30 of them positive. Cap heading rule removed; "limited to" restricted to amounts ("not limited to" is common). Change Of Control: two rules with precision 0.11 and 0.18 removed; merger rule restricted to termination rights. "exclusive" excludes property/jurisdiction/remedy/owner/of and similar. |
| 2 | 0.580 | Termination For Convenience requires a termination right ("may terminate"), because "upon termination for any reason" is survival language. Three rules that were almost never right (`unlimited`, `all_requirements`, `no_other_third_parties`) removed. |
| 3 (frozen) | 0.582 | Bug fix after code review: "non-exclusive" written with a space after the hyphen (46 times in CUAD) or without a hyphen (8 times) was still matched by `exclusive`. Test numbers in this document are from the frozen rules. |

## Verification

```bash
uv run --locked python -B -m pytest -q -p no:cacheprovider
uv run --locked ruff check --no-cache src/keyword_baseline.py scripts/keyword_baseline.py tests/test_keyword_baseline.py
```

Tests check typical phrasing for every category, the known false-positive phrasings
each rule is meant to skip, whitespace and case handling, explanation offsets, and
masked scoring.
