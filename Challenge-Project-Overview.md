
---

# Contract Review Challenge

**Company / Org:** Accenture  
**Challenge Advisor:** Lipika Mukherjee, lipika.mukherjee@accenture.com  
**AI Studio Coach**: Shweta Malabade, shweta.malabade@breakthroughtech.org  \
**Program:** Break Through Tech AI Studio - Fall 2026  

---

## 🏢 About Accenture
Accenture is a leading global professional services company that provides a broad range of services and solutions in strategy, consulting, technology, and operations. 

---

## 🎯 The Challenge
### Project Summary
In this project, you will use real-world commercial contracts from the CUAD dataset (510 contracts, 41 expert-annotated clause categories) and NLP techniques including chunk-based multi-label classification with fine-tuned transformer encoders, paired with an explainable rule-based risk-scoring layer, to build a pipeline that automatically detects key clauses, flags them as Low/Medium/High risk, and rolls these up into a contract-level triage score. This will help our company address the bottleneck legal and procurement teams face when manually reviewing tens of thousands of contracts a year to find the small number of clauses that carry meaningful risk, enabling reviewers to prioritize which contracts to open first.

### Success Criteria
Success has two tracks:
- For clause detection: per-category precision/recall/F1 clearly beating the baseline (accuracy is misleading under CUAD's imbalance), with error analysis on where the model struggles.   
- For risk scoring: since there are no ground-truth labels, success means strong Spearman correlation and bucket agreement between the model's risk rankings and the advisor's hand-ranked clauses, plus a sensitivity analysis showing the High/Medium boundary is stable.

Overall, a successful December outcome is a working end-to-end pipeline producing risk-scored clause registers the advisor finds plausible and useful, a clean documented repo, and a final report covering results, limitations, and estimated reviewer time saved — an auditable triage tool the advisor would actually trust, not a black box.

### Stretch Goals
Stretch goals include span extraction, a trained risk model benchmarked against the rule-based baseline, LLM-generated clause explanations, broader category coverage, a Streamlit/Gradio demo, and an active-learning loop using advisor/model disagreements. These extend modeling or usability without affecting core deliverables.

### Project Milestones
Use these milestones to guide your work. Your team will create a GitHub Projects board to track tasks within each milestone.
| Month | Milestone | Key Activities |
|-------|-----------|----------------|
| **September** | Data Preparation & Baseline Modeling | Clean and split the CUAD data, run EDA on class imbalance, build a chunking strategy, and establish a TF-IDF/keyword baseline with per-category metrics. |
| **October** | Transformer Model Development | Fine-tune a transformer encoder for multi-label clause classification, address class imbalance, evaluate with per-category precision/recall/F1, and conduct error analysis. |
| **November** | Risk Scoring & End-to-End Validation | Build and calibrate the four-signal risk-scoring layer, assemble the end-to-end pipeline, and validate risk rankings against advisor-labeled examples. |

> **Note for the team:** Please create a GitHub Projects board in this repository to break these milestones into weekly tasks. Go to the **Projects** tab → **New project** → Choose **Board** → Add columns for each month.

---

## 📊 Dataset
**Name and Source:** CUAD Dataset (Contract Understanding Atticus Dataset)  
**Format:** JSON, Raw Text/PDF  
**Size:** under 1gb  
**Location:** https://github.com/TheAtticusProject/cuad  

### Key Details
- `CUADv1.json` contains 510 commercial contracts and 13,823 annotated answer spans across 41 contract-review categories.
- Use the prepared JSON files: `train_separate_questions.json` contains 408 contracts and `test.json` contains 102 contracts. These are the **official** train/test splits released by The Atticus Project — split at the contract level (not by individual clause) to prevent data leakage, and directly comparable to the results in the original CUAD paper. Do not re-split the data yourselves.
- Contract text is already available in each JSON document's `paragraphs[].context` field, with clause questions in `paragraphs[].qas[]` and labeled spans in `paragraphs[].qas[].answers[]`. **Do not parse raw PDFs for this project.**
- `category_descriptions.csv` provides the name, description, answer format, and group for each of the 41 categories.
- Contracts vary substantially in length, so teams should develop a chunking strategy, preserve important legal terminology during cleaning, and account for class imbalance.

| Dataset / Source | Purpose in Project | Format | Access |
|---|---|---|---|
| **CUAD Category Descriptions** | Defines the 41 clause categories and provides guidance on what each category represents. Useful for building the label mapping and understanding the classification task. | CSV | [CUAD GitHub Repository](https://github.com/TheAtticusProject/cuad) |
| **CUAD Dataset – Hugging Face** | Provides a machine-learning-friendly way to load CUAD directly into Python and Hugging Face workflows.| Hugging Face Dataset | [CUAD on Hugging Face](https://huggingface.co/datasets/theatticusproject/cuad-qa) |

> ⚠️ **Note on Hugging Face naming:** use `theatticusproject/cuad-qa` specifically. The similarly named `theatticusproject/cuad` (no `-qa`) is a different, unstructured repository containing only documentation text — it is **not** usable contract data.

### Working Dataset Expectations

* **Primary Dataset:** Use the CUAD (Contract Understanding Atticus Dataset), containing 510 commercial contracts and 41 expert-annotated clause categories.
* **Initial Scope:** Start with approximately 50–100 contracts for data exploration, preprocessing, and baseline development before expanding to the full dataset.
* **Data Exploration:** Analyze contract length, clause frequency, category distribution, and potential class imbalance.
* **Preprocessing:** Clean and standardize contract text while preserving relevant clause boundaries and annotations.
* **Chunking:** Develop a chunking strategy that allows long contracts to be processed by transformer models while retaining sufficient context.
* **Data Splits:** Create contract-level training, validation, and test sets to prevent data leakage.
* **Classification Labels:** Use CUAD’s 41 clause categories as the initial multi-label classification targets.
* **Evidence Retention:** Preserve the relevant text span for each detected clause so predictions can be explained and reviewed.
* **Risk Scoring:** Develop a separate, explainable rule-based layer to assign **Low/Medium/High** risk, since CUAD does not provide risk labels.
* **Contract-Level Triage:** Aggregate clause-level risk scores into an overall contract risk/triage score.
* **Reproducibility:** Document the dataset version, preprocessing, data splits, assumptions, and methodology in GitHub.
* **Final Output:** The pipeline should produce **clause category → evidence → risk level → rationale → contract-level triage score**.

### Known Preprocessing and Data Risks
* Normalize contract text consistently, including formatting, whitespace, headers, and page breaks while preserving meaningful legal language.
* Standardize contract IDs, clause labels, annotation spans, and document metadata across all source files.
* Handle long contracts carefully: chunk text without separating important clause context or splitting relevant annotations incorrectly.
* Expect domain and annotation variability: CUAD contracts and expert annotations may differ in structure and language, which can affect model performance on new or unseen contracts.

---

## 🛠️ Suggested Approach

**ML Problem Type:** NLP / Multi-label Classification / Information Extraction / Explainable Risk Scoring

**Recommended Libraries:** Hugging Face Transformers, PyTorch, scikit-learn, pandas, NumPy, Hugging Face Datasets

**Suggested Pipeline:** Contract Text → Preprocessing & Chunking → Multi-label Clause Classification → Evidence Extraction → Rule-based Risk Scoring → Contract-level Triage Score

**Evaluation Metrics:** Precision, Recall, and F1-Score for clause classification; High-Risk Recall and Precision@K for contract triage; basic error analysis for risk-scoring results.

**Development Environment:** Google Colab for model training and experiments; VS Code and Jupyter Notebooks for development and analysis.

---

## 📚 Resources to Get Started

The following resources will help your team understand the problem space and potential technical approaches for this project:

**Background Reading:**
- CUAD Dataset Paper: https://arxiv.org/abs/2103.06268
- Contract Understanding Atticus Dataset GitHub: https://github.com/TheAtticusProject/cuad

**Technical Tutorials:**
- Hugging Face NLP Course: https://huggingface.co/learn/nlp-course
- Hugging Face Transformers Documentation: https://huggingface.co/docs/transformers

**Code Examples:**
- CUAD GitHub Repository: https://github.com/TheAtticusProject/cuad
- Hugging Face Text Classification Examples: https://github.com/huggingface/transformers/tree/main/examples

**Other:**
- •	Attention Is All You Need (Transformer Paper)
  •	Practical Legal NLP examples on Hugging Face

**Suggested Pipeline:**

CUAD Contracts → Preprocessing → Chunking → Multi-label Clause Classification → Evidence Extraction → Rule-based Risk Scoring → Contract-level Triage Score → Explainable Risk Report

## Recommended Modeling Approach

- **Establish a Baseline:** Build a simple TF-IDF/keyword-based classifier before using transformer models.
- **Train the Model:** Fine-tune a transformer encoder for multi-label classification across the 41 CUAD clause categories.
- **Preserve Evidence:** Retain the relevant contract text/span for each detected clause to support explainability.
- **Add Risk Scoring:** Develop an explainable rule-based layer that assigns **Low / Medium / High** risk based on detected clause characteristics.
- **Calculate Triage Score:** Aggregate clause-level risks into an overall **contract-level triage score** to help prioritize contracts for review.
- **Evaluate Performance:** Measure clause classification using **Precision, Recall, and F1 Score**, with emphasis on identifying high-risk clauses.
- **Perform Error Analysis:** Review false positives, false negatives, rare clause categories, and difficult contract language to identify opportunities for improvement.

*Feel free to explore beyond these,and share anything interesting you find with me!*

---

## Evaluation Metrics

| **Component** | **Metric** | **Purpose** |
|---|---|---|
| Clause Classification | **Precision** | Of the clauses the model identifies, how many are correct? |
| Clause Classification | **Recall** | Of the relevant clauses in the contract, how many does the model find? |
| Clause Classification | **F1 Score** | Combines Precision and Recall into one overall classification score. |
| Risk Scoring | **Accuracy** | How often does the system correctly assign Low, Medium, or High risk? |
| Contract Triage | **High-Risk Recall** | Of the contracts that should receive priority review, how many does the system successfully flag? |

### Primary Evaluation

Team should focus primarily on **Precision, Recall, and F1 Score** when evaluating clause classification.

For the final risk-triage pipeline, **High-Risk Recall** is especially important because the goal is to avoid missing contracts that may require additional legal or procurement review.

Team should also perform a simple **error analysis** by reviewing examples of:
- Incorrectly identified clauses
- Missed clauses
- Incorrect risk assignments
- Difficult or ambiguous contract language

The goal is not only to report a score, but to understand **where the model works well, where it fails, and why**.

---

## 🤝 How We'll Work Together

**Official check-ins:** During our biweekly 45-minute AI Studio Lab Section meeting block (2nd and 4th week of every month)

 **Other ways to reach out to me with questions:** 
* Email: lipika.mukherjee@accenture.com
  Please include all teammates and the AI Studio Coach.
  
* Discord: Use the team's assigned Break Through Tech channel.
  
* Additional meetings:
  Request through email if additional guidance is needed.

* Response expectation:
  I will aim to respond within 48 hours.



**Recommended free coding / collaboration tools**
* Google Colab
• GitHub Projects
• GitHub Issues
• VS Code
• Jupyter Notebooks

---

## Coach Questions - Advisor Guidance

**Data & Format:**
  Use the CUAD JSON files as the source of truth but reshape them into a chunk-based multi-label classification format for model training. Each contract chunk should be labeled with one or more applicable clause categories. The CUAD train/test split may be reused if the split is preserved at the contract level. Chunks from the same contract should not appear in both train and test sets to avoid data leakage.

**Clause Category Scope:**
  For the initial scope, focus on approximately 10 high-value clause categories rather than all 41. Suggested priority categories are: Limitation of Liability, Indemnification, Termination Rights, Confidentiality, Governing Law, Dispute Resolution, Assignment, Change of Control, Auto-Renewal, and Exclusivity or Non-Compete / Non-Solicit obligations. This list can be refined during advisor check-ins.

**Risk Scoring:**
  Risk should be based on legal, financial, operational, or compliance exposure. Examples of risky signals include missing or unlimited liability cap, broad indemnification, one-sided termination rights, vague language, auto-renewal without clear notice, restrictive assignment/change-of-control terms, unfavorable governing law, or missing expected clauses.

For this challenge, the team should define a simple Low/Medium/High rubric:

Low: Standard clause with limited business impact.
Medium: Clause contains unclear, non-standard, or potentially unfavorable language.
High: Clause may create significant legal, financial, operational, or compliance exposure and should be prioritized for human review.
Missing expected clauses may count as risk signals, but should be handled carefully because “not detected” may not always mean “truly missing.” Risk signals should be weighted rather than treated equally. Higher-weight items include unlimited liability, broad indemnification, unfavorable termination, and missing liability cap.

**Advisor Reference Ranking:**
  
  The team can select a small sample of 20–30 clauses that may be hand-ranked by the advisor as Low, Medium, or High to help the team compare model-generated risk scores against advisor judgment. This should be used as a lightweight validation aid, not a full ground-truth dataset.

**Modeling / Implementation:**
  Google Colab free tier should be sufficient for the initial scope if the team uses a smaller model, limits the number of clause categories, and starts with a baseline approach. Paid compute is not required unless the team chooses larger models or extensive experiments.

For long contracts, use paragraph-based chunking where possible, with a maximum token limit and optional overlapping windows. If paragraphs are too long, split by sentence or fixed token length. The team should document the chunking method and ensure it preserves enough context for clause classification.

## 🚀 Getting Started

Read this overview and list your open questions before our first team meeting.

1. **Review this overview document** and note any questions for our first meeting: Understand the project goals, technical approach, dataset expectations, milestones.
2. **Begin reviewing the JSON dataset** in the [data folder](data/cuad) Explore the 41 clause categories and identify the initial subset of contracts you will use for data exploration and baseline development.
3. **Read the GitHub Projects documentation** [here](https://docs.github.com/en/issues/planning-and-tracking-with-projects/learning-about-projects/about-projects)
4. **Prepare Open Questions:** Record questions, assumptions, and areas where you need clarification before the first team meeting.
5. **Document Your Decisions:** Keep important technical decisions and findings in GitHub Issues or project documentation so the entire team can follow the project's progress.
I’m excited to work with you!

---

## ❓ Questions?

Please bring any questions to our first meeting during the week of August 24th (Break Through Tech’s Bridge to Studio - Session C). 
