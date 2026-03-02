"""
validator.py — Independent answer-grading layer.

Uses mistral as a lightweight judge to evaluate the pipeline's answer
on four auditing-relevant dimensions.  Invoked only on user request from
the Streamlit dashboard ("Validate This Answer" button).

Dimensions
----------
  faithfulness    — Is every claim in the answer supported by the context?
  relevance       — Does the answer actually address the original question?
  completeness    — Are all important facts from the context included?
  hallucination   — Does the answer introduce facts not in the context?

Each dimension is scored 1–5 (5 = best).
A pass/fail flag is set if the total weighted score ≥ threshold (default 14/20).
"""

import json
import re
from typing import Dict

import ollama

from src.config import REASONING_MODEL as VALIDATOR_MODEL  # same model, strict grading prompt


# ── Prompt ────────────────────────────────────────────────

_JUDGE_SYSTEM = (
    "You are a strict audit quality assurance evaluator. "
    "You assess AI-generated answers objectively based solely on the provided "
    "question and reference context. Be rigorous and concise."
)

_JUDGE_PROMPT = """AUDITOR QUESTION:
{question}

REFERENCE CONTEXT (retrieved from the official report):
{context}

AI-GENERATED ANSWER:
{answer}

Evaluate the answer on the following 4 dimensions. Score each 1-5 (5 = best):

1. FAITHFULNESS (1-5): Is every claim in the answer directly supported by the context?
2. RELEVANCE (1-5): Does the answer directly and completely address the question?
3. COMPLETENESS (1-5): Does the answer cover all key facts present in the context?
4. HALLUCINATION RISK (1-5, where 5 = no hallucinations, 1 = severe hallucinations):
   Does the answer avoid introducing facts not present in the context?

Also write a short CRITIQUE (2-3 sentences) summarising strengths and weaknesses.

Respond ONLY with valid JSON:
{{
  "faithfulness":      {{"score": <1-5>, "comment": "<brief comment>"}},
  "relevance":         {{"score": <1-5>, "comment": "<brief comment>"}},
  "completeness":      {{"score": <1-5>, "comment": "<brief comment>"}},
  "hallucination_risk":{{"score": <1-5>, "comment": "<brief comment>"}},
  "overall_critique":  "<2-3 sentences>",
  "pass": <true/false>
}}"""


# ── Main Validation Function ──────────────────────────────

def validate_answer(
    question: str,
    answer  : str,
    context : str,
    pass_threshold: int = 14,
) -> Dict:
    """
    Grade the AI answer against the reference context.

    Parameters
    ----------
    question        : original auditor query
    answer          : pipeline's final_answer
    context         : summarized_context from the pipeline state
    pass_threshold  : minimum total score (out of 20) to mark as PASS

    Returns
    -------
    dict with keys: faithfulness, relevance, completeness, hallucination_risk,
                    overall_critique, pass, total_score, error (if any)
    """
    prompt = _JUDGE_PROMPT.format(
        question=question,
        context =context[:3000],   # cap to stay within context window
        answer  =answer[:2000],
    )

    try:
        response = ollama.chat(
            model=VALIDATOR_MODEL,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user"  , "content": prompt},
            ],
            options={"temperature": 0},
        )
        raw   = response["message"]["content"].strip()
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in validator response")

        result = json.loads(match.group(0))

        # Compute total score
        dims = ["faithfulness", "relevance", "completeness", "hallucination_risk"]
        total = sum(
            result.get(d, {}).get("score", 0) for d in dims
        )
        result["total_score"] = total

        # Override pass/fail with numeric threshold
        result["pass"] = total >= pass_threshold

        result["error"] = None
        return result

    except Exception as e:
        return {
            "faithfulness"     : {"score": 0, "comment": "Evaluation failed"},
            "relevance"        : {"score": 0, "comment": "Evaluation failed"},
            "completeness"     : {"score": 0, "comment": "Evaluation failed"},
            "hallucination_risk":{"score": 0, "comment": "Evaluation failed"},
            "overall_critique" : "Validation could not be completed.",
            "total_score"      : 0,
            "pass"             : False,
            "error"            : str(e),
        }
