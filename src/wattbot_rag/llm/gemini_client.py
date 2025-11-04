from __future__ import annotations

import json
from typing import Dict, List

import google.generativeai as genai

from ..config import get_google_api_key
from .prompts import build_wattbot_prompt, classify_question_type


DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


def configure_gemini(model_name: str = DEFAULT_GEMINI_MODEL) -> str:
    """
    Configure the Gemini client using the GOOGLE_API_KEY environment variable.

    Returns the model name actually used.
    """

    api_key = get_google_api_key()
    genai.configure(api_key=api_key)
    return model_name


def generate_answer_with_gemini(
    row: Dict,
    retrieval_results: List[Dict],
    doc_id_to_url: Dict[str, str],
    model_name: str = DEFAULT_GEMINI_MODEL,
) -> Dict:
    """
    Call Gemini to generate an answer based on retrieval results.

    Notes:
    - If GOOGLE_API_KEY is not set, this function will raise.
    - Currently provided as an example; the CLI does not invoke it directly yet.
    """

    if not retrieval_results:
        raise RuntimeError("No retrieval results provided to Gemini.")

    model_name = configure_gemini(model_name)
    model = genai.GenerativeModel(model_name)

    qid = row["id"]
    question = row["question"]
    unit = row.get("answer_unit", "is_blank")
    qtype = classify_question_type(question)

    # 簡單將前幾個 chunks 拼接成 context
    context_blocks = []
    used_doc_ids: List[str] = []
    for r in retrieval_results[:4]:
        did = r["doc_id"]
        used_doc_ids.append(did)
        context_blocks.append(f"[DOC_ID: {did}]\n{r['content']}")
    context_text = "\n\n".join(context_blocks)

    prompt = build_wattbot_prompt(question, unit, qtype, context_text)

    resp = model.generate_content(
        prompt,
        generation_config={
            "response_mime_type": "application/json",
            "temperature": 0.0,
        },
    )
    raw_text = resp.text
    js = json.loads(raw_text)

    # 補上必要欄位與 URL
    final_ref_id = js.get("ref_id") or used_doc_ids
    if isinstance(final_ref_id, str):
        try:
            if final_ref_id.startswith("["):
                final_ref_id = json.loads(final_ref_id)
            else:
                final_ref_id = [final_ref_id]
        except Exception:  # noqa: BLE001
            final_ref_id = [final_ref_id]

    final_ref_url = [doc_id_to_url.get(did, "N/A") for did in final_ref_id]
    js["ref_id"] = final_ref_id
    js["ref_url"] = final_ref_url
    js["id"] = qid
    js["question"] = question

    if "supporting_material" not in js:
        js["supporting_material"] = "is_blank"
    if "explanation" not in js:
        js["explanation"] = "is_blank"

    return js

