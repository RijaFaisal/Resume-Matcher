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

3. Quantitative Evaluation

We used:

Metric	Purpose
Accuracy	Measures High/Medium/Low match correctness
BLEU	Measures n-gram similarity vs ground truth label word
ROUGE-L	Evaluates overlap between model explanation and ground-truth direction
Embedding Cosine Similarity	Semantic similarity between response & ground-truth classification

All 20 examples were run for all 3 strategies via 04_eval_runner.ipynb, producing:

Accuracy Results

(from generated human_rubric_filled.csv)

Strategy	Accuracy
Few-Shot	0.75
Zero-Shot	0.60
Advanced (CoT)	0.55
Interpretation

Few-shot prompting is the best, giving the model clear labeled examples.

Zero-shot is decent but inconsistent.

CoT produces long reasoning but sometimes mislabeled final class.

4. Qualitative Human Evaluation

Human rubric fields:

Factuality (1–5)

Helpfulness (1–5)

Clarity (1–5)

Summary based on filled rubric:

Strategy	Avg. Factuality	Avg. Helpfulness	Avg. Clarity
Few-Shot	        Highest	          Highest	     Highest
Zero-Shot	         Medium	          Medium	      Medium
Advanced (CoT)	High helpfulness but lower factuality due to over-explanation	High	Medium
Interpretation

Few-shot produced the cleanest and most accurate responses with minimal hallucination.

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
6.1 Zero-Shot

Fails when candidate descriptions are ambiguous

Tends to default to “Medium”

Doesn’t use job-specific context strongly

6.2 Few-Shot

Best balance of precision + reliability

Learns pattern from examples

Only fails when candidate/job pair is borderline ambiguous

6.3 Advanced (CoT + Meta)

Very strong reasoning but inconsistent final label

CoT sometimes gives multiple labels

Over-explains and drifts away from simple “High/Medium/Low” answer

7. Final Ranking of Prompting Strategies
Rank	Strategy	Reason
Few-Shot (k=3)	Best accuracy + best human scores	
zero-Shot	Good speed, moderate consistency	
Advanced CoT/Meta	Good reasoning but unstable classification