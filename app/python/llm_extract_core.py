# llm_extract_core.py (patch)

from __future__ import annotations
import json
import re
from typing import List, Dict, Optional, Any
from ollama import Client
import os

DEFAULT_TIMEOUT = 120.0


SYSTEM_PROMPT = """You are an information extraction assistant.

You will receive:
- A context note
- The full document text (or a chunk)
- A JSON list of questions with ids

Return ONLY valid JSON, and ONLY this structure:
{
  "answers": {
    "<id>": {"value": <string or null>, "confidence": <float 0.0-1.0>},
    ...
  }
}

Rules:
- Use only evidence from the document text.
- If not present / not inferable, set value to null and confidence to 0.0.
- confidence is your certainty that the extracted value is correct (0.0 = no evidence, 1.0 = exact match found).
- Do not add extra keys.
- No markdown, no explanation.
"""

def extract_json_object(text: str) -> Optional[str]:
    """Pull a JSON object out of markdown/codefences/prose."""
    if not text:
        return None

    # 1) Code fence: ```json { ... } ```
    m = re.search(r"```(?:json)?\s*({.*?})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # 2) First balanced {...} block, respecting strings/escapes
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_str = False
    esc = False

    for i in range(start, len(text)):
        ch = text[i]

        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        # not in string
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1].strip()

    return None


def safe_load_json_from_model(text: str) -> Dict[str, Any]:
    snippet = extract_json_object(text)
    if not snippet:
        return {}
    try:
        obj = json.loads(snippet)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _repair_to_json(model: str, bad_text: str, timeout: float = DEFAULT_TIMEOUT) -> Dict[str, Any]:
    repair_system = "You output ONLY valid JSON. No markdown. No extra text."
    client = Client(timeout=timeout)
    resp = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": repair_system},
            {"role": "user", "content": f"Fix this into valid JSON only:\n\n{bad_text}"},
        ],
        options={"temperature": 0.0},
    )
    content = getattr(getattr(resp, "message", None), "content", None) or ""
    parsed = safe_load_json_from_model(content)
    return parsed if isinstance(parsed, dict) else {}


def _normalize_question_for_model(q: str) -> str:
    # keeps user-visible key unchanged elsewhere; this is only for the model prompt
    q2 = (q or "").strip()
    q2 = re.sub(r"\s+", " ", q2)
    q2 = q2.rstrip(":").strip()
    return q2


def _extract_field(raw: Any) -> Dict[str, Any]:
    """Normalize a single answer entry to {value, confidence}."""
    if isinstance(raw, dict):
        value = raw.get("value")
        confidence = raw.get("confidence", 0.0)
    else:
        # Old-style plain string answer — treat as full confidence if non-null
        value = raw
        confidence = 0.0 if raw is None else 1.0

    if isinstance(value, str):
        value = value.strip() or None
    elif value is not None:
        value = str(value).strip() or None

    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    return {"value": value, "confidence": confidence}


def answer_questions_json(
    *,
    model: str,
    document_text: str,
    questions: List[str],
    context_note: str = "The information might be spread out through the entire document.",
    temperature: float = 0.1,
    timeout: float = DEFAULT_TIMEOUT,
) -> Dict[str, Dict[str, Any]]:
    orig_questions = [q.strip() for q in questions if q and q.strip()]

    if not orig_questions:
        return {}

    # Build an ID map so punctuation (like trailing :) never breaks key matching.
    q_items = []
    id_to_key: Dict[str, str] = {}
    for i, q in enumerate(orig_questions, start=1):
        qid = f"q{i}"
        id_to_key[qid] = q
        q_items.append({"id": qid, "question": _normalize_question_for_model(q)})

    user_prompt = (
        f"Context note:\n{context_note}\n\n"
        f"Questions:\n{json.dumps(q_items, ensure_ascii=False, indent=2)}\n\n"
        "Document text:\n<doc>\n"
        f"{document_text}\n"
        "</doc>\n"
        "\nReturn JSON now."
    )
    client = Client(timeout=timeout)
    resp = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": temperature},
    )

    content = getattr(getattr(resp, "message", None), "content", None) or ""
    parsed = safe_load_json_from_model(content)
    if not isinstance(parsed, dict):
        parsed = _repair_to_json(model, content, timeout=timeout)

    # --- flexible answer extraction ---
    answers: Dict[str, Any] = {}

    if isinstance(parsed, dict):
        if isinstance(parsed.get("answers"), dict):
            answers = parsed["answers"]
        elif any(k.startswith("q") and k[1:].isdigit() for k in parsed.keys()):
            answers = parsed

    # Remap back to the original question strings as keys.
    out: Dict[str, Dict[str, Any]] = {}
    for qid, key in id_to_key.items():
        raw = answers.get(qid)
        if raw is None:
            raw = answers.get(id_to_key[qid])
        if raw is None:
            raw = answers.get(id_to_key[qid].rstrip(":").strip())
        out[key] = _extract_field(raw)

    return out


def _chunk_text(s: str, max_chars: int = 12000, overlap: int = 800) -> list[str]:
    s = (s or "").replace("\r\n", "\n").replace("\r", "\n")
    chunks = []
    i = 0
    while i < len(s):
        j = min(len(s), i + max_chars)
        chunks.append(s[i:j])
        if j == len(s):
            break
        i = max(0, j - overlap)
    return chunks


def answer_questions_json_chunked(
    *,
    model: str,
    document_text: str,
    questions: List[str],
    context_note: str = "The information might be spread out through the entire document.",
    max_chars: int = 12000,
    overlap: int = 800,
    timeout: float = DEFAULT_TIMEOUT,
) -> Dict[str, Dict[str, Any]]:
    qs = [q.strip() for q in questions if q and q.strip()]
    if not qs:
        return {}

    merged: Dict[str, Dict[str, Any]] = {q: {"value": None, "confidence": 0.0} for q in qs}
    chunks = _chunk_text(document_text, max_chars=max_chars, overlap=overlap)

    for idx, ch in enumerate(chunks, start=1):
        partial = answer_questions_json(
            model=model,
            document_text=f"[CHUNK {idx}/{len(chunks)}]\n{ch}",
            questions=qs,
            context_note=context_note + " Answer only if the information is present in this chunk.",
            temperature=0.1,
            timeout=timeout,
        )
        for q in qs:
            new_entry = partial.get(q, {"value": None, "confidence": 0.0})
            if new_entry.get("confidence", 0.0) > merged[q].get("confidence", 0.0):
                merged[q] = new_entry

    return merged

