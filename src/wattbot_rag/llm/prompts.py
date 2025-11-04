from __future__ import annotations

from typing import Literal


QuestionType = Literal["bool", "range", "category", "numeric"]


def classify_question_type(q: str) -> QuestionType:
    ql = q.lower().strip()
    if (
        "true" in ql
        or "false" in ql
        or ql.startswith(("is ", "are ", "does ", "do ", "can ", "should "))
    ):
        return "bool"
    if "range" in ql or "between" in ql:
        return "range"
    if "which" in ql or "what is the name" in ql or "who " in ql:
        return "category"
    return "numeric"


def build_wattbot_prompt(
    question: str,
    unit: str,
    qtype: QuestionType,
    context_text: str,
) -> str:
    return f"""
You are an expert research assistant for the WattBot 2025 competition.

Your task:
Given a question, its expected answer unit, and a context extracted from scientific
papers (including text, tables, and OCR from figures), you must answer ONLY using
the information from the context.

You MUST output a single JSON object with the following fields:

{{
  "answer": "...",               // short natural-language answer, or "is_blank"
  "answer_value": "...",         // machine-friendly value (number / range / True / False / category / "is_blank")
  "answer_unit": "...",          // usually same as the given Answer_unit; use "is_blank" if unanswerable
  "ref_id": ["...", "..."],      // list of doc_id you used (e.g. ["paper_001", "paper_002"])
  "ref_url": ["...", "..."],     // leave as [] or ["N/A"]; it will be filled in by the code
  "supporting_material": "...",  // short quote or description from the context
  "explanation": "..."           // short explanation of how you got the answer
}}

Rules:
- If the question truly cannot be answered from the context, then:
  - "answer" should be "Unable to answer with confidence based on the provided documents."
  - "answer_value", "answer_unit", "ref_id", "ref_url", "supporting_material" and "explanation"
    must all be exactly "is_blank".
- For boolean questions, answer_value MUST be "True" or "False" (or "is_blank").
- For numeric questions, answer_value should be a single number string (e.g. "3.5", "175").
- For range questions, answer_value should be like "[low,high]" (e.g. "[3,5]" or "[1.0,2.5]").


Now answer the following question.

Question: {question}
Expected answer unit: {unit}
Question type: {qtype}

Context:
{context_text}

Remember: respond with ONLY a valid JSON object, nothing else.
""".strip()

