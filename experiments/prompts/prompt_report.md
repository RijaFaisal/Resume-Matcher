Prompt Evaluation Report — Milestone 2 (D1)
1. Overview

This report summarizes the evaluation of three prompting strategies used for the Resume–Job Matching LLM task:

Zero-Shot Prompting

Few-Shot Prompting (k = 3)

Advanced Prompting (Chain-of-Thought + Meta-Prompting)

All strategies were tested on a 20-item evaluation set (eval.jsonl).
We evaluate performance using:

Quantitative metrics: accuracy, BLEU, ROUGE-L, embedding cosine similarity

Qualitative human rubric: factuality, helpfulness

MLflow logging: all metrics + artifacts saved to prompt_eval experiment.

2. Prompt Structures
2.1 Zero-Shot Prompt

(From zero_shot.txt)

System instructs model to act as an expert recruiter.

User query provides candidate + job description.

Output required: match score (High/Medium/Low) + short justification.

No examples included.

Strength: Fast, simple.
Weakness: Often generic, inconsistent reasoning.

2.2 Few-Shot Prompt (k = 3)

(From few_shot_k3.txt)

Includes three example (Q → A) pairs:

Example of High match

Example of Low match

Example of Medium match

The model receives examples before the real query.

Strength:
Learns correct labeling—highest accuracy.

Weakness:
Sensitive to example choice; small hallucinations appear.

2.3 Advanced Prompt (CoT + Meta Prompt)

(From cot_meta.txt)

Meta-prompt defines:

Model persona

Output rules

Step-by-step reasoning

Scoring standards

Explicit Chain-of-Thought enforced (“Think step-by-step”)

Strength:
Most detailed, interpretable reasoning.

Weakness:
Long responses sometimes drift from High/Medium/Low classification.

3. Quantitative Evaluation (Improved & Detailed)

We evaluated each prompting strategy across 20 test cases using accuracy, label-level statistics, and semantic metrics (cosine similarity).
The predictions were compared against the human-defined ground truth labels: High, Medium, Low.

A. Accuracy Scores
Strategy	Accuracy
Few-Shot	0.75
Zero-Shot	0.60
Advanced (CoT / Meta)	0.55

Interpretation:

Few-shot prompting is the strongest strategy overall.
It benefits from concrete examples that guide the model toward correct label boundaries.

Zero-shot is surprisingly competitive, showing decent generalization without examples.

CoT/Meta underperforms despite being more “advanced.”
This happens because CoT produces long, thoughtful reasoning, but our task is simple classification, where longer reasoning introduces noise and inconsistency.

B. Label Distribution

(How often each strategy predicts High/Medium/Low)

Few-shot: Balanced across labels → reflects controlled, example-driven behavior

Zero-shot: Tends to over-predict Medium, meaning it's cautious and non-committal

Advanced: Over-predicts High because reflective CoT reasoning inflates confidence

This demonstrates that reasoning-heavy prompts can distort probability calibration.

C. Semantic Similarity (Cosine Similarity)

Even when labels mismatch, we checked whether the meaning of the output is close to the ground truth using embedding similarity.

Few-shot embeddings aligned best with intended label meaning

Zero-shot was close but showed drift in ambiguous cases

CoT/Meta had the lowest similarity due to overly verbose reasoning

This reinforces that more reasoning ≠ better classification for this task.

4. Qualitative Evaluation (Human-in-the-loop)

We manually scored each model output on:

Factuality (1–5): Does the explanation match real candidate–job alignment?

Helpfulness (1–5): Is the explanation clear, concise, and actionable?

Clarity (1–5): Does the model directly answer the question?

Findings:
Strategy	Factuality	Helpfulness	Clarity	Notes
Few-Shot	 Highest	 Highest	 Best	Clear, controlled, avoids hallucinations
Zero-Shot	Medium	Medium	Medium	Sometimes vague, inconsistent justification
CoT / Meta	Mixed	Lower	Lower	Explanations are long, sometimes overthink simple decisions

Few-shot outputs were the easiest for humans to judge as “correct.”

CoT/Meta produced well-written but often unnecessary reasoning that did not improve correctness.

5. MLflow Logging Summary

All runs were successfully logged, check the csv file called responses and evaluation_results:

G:\Resume-Matcher\experiments\prompts\results\responses.csv
G:\Resume-Matcher\experiments\prompts\results\evaluation_results.csv
G:\Resume-Matcher\experiments\prompts\results\human_rubric_template.csv

Logged items include:

Metrics:

{strategy}_accuracy

{strategy}_bleu

{strategy}_rouge

{strategy}_embed_score

Artifacts:

responses.csv

human_rubric_template.csv

quant_summary.json

*_classif.json reports

MLflow UI was verified to list all runs successfully.

6. Key Insights & Failure Cases

A. Few-shot prompting is the best overall strategy

It provides structure and examples

Reduces randomness

Keeps the model within correct label boundaries

Produces the highest accuracy and semantic alignment

B. Zero-shot is decent but unstable

Works fine for clear cases (e.g., High or Low)

Fails on borderline cases where examples would help

Often defaults to “Medium” when uncertain

C. CoT / Meta is not effective for short classification tasks

Chain-of-thought encourages long reasoning but classification requires short decisions

Longer reasoning introduces more opportunities for incorrect assumptions

The model becomes too confident, misclassifying ambiguous profiles as "High"

D. Advanced prompting helps only when the task is reasoning-heavy

Not for:

classification

similarity scoring

short decision-making

This is a classic case of overprompting making results worse.

E. Semantic similarity revealed hidden alignment issues

Even when labels mismatched:

Few-shot was semantically closest

Zero-shot mildly close

CoT/Meta drifted significantly (topic drift, extra assumptions)

F. Human scoring revealed that verbosity ≠ quality

Humans preferred:

Short

Direct

Justified
explanations over long chain-of-thought paragraphs.

7. Final Ranking of Prompting Strategies
Rank	Strategy	Reason
Few-Shot (k=3)	Best accuracy + best human scores	
zero-Shot	Good speed, moderate consistency	
Advanced CoT/Meta	Good reasoning but unstable classification