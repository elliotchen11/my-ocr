"""
Simulates clicking "Run extraction now" in the Streamlit UI.
Hardcoded parameters mirror the UI defaults.
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from llm_extract_core import answer_questions_json, answer_questions_json_chunked

# ---------------------------------------------------------------------------
# Hardcoded parameters (mirrors UI defaults)
# ---------------------------------------------------------------------------

MODEL = "gpt-oss"
OCR_MODEL = "mistral-small3.2"

QUESTIONS = [
    "What is the patient's full name?",
    "What is the date of the document?",
    "What is the document type?",
]

CONTEXT_NOTE = "This is a medical document. Extract the requested fields accurately."

DOCUMENT_TEXT = """
PATIENT RECORD

Name: John A. Smith
Date of Birth: 1985-04-22
Document Date: March 10 2025
Document Type: Discharge Summary

The patient was admitted on March 7 2025 and discharged on March 10 2025
following treatment for acute appendicitis. All vitals are stable at discharge.
"""

# Chunking settings (mirrors UI sliders)
FORCE_CHUNKING = False
TOKEN_THRESHOLD = 8000
CHUNK_CHARS = 12000
OVERLAP_CHARS = 800


def estimate_tokens(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return len(text) // 4


def run_extraction():
    tok_est = estimate_tokens(DOCUMENT_TEXT)
    use_chunked = FORCE_CHUNKING or (tok_est > TOKEN_THRESHOLD)

    print(f"Model        : {MODEL}")
    print(f"Token est.   : {tok_est}")
    print(f"Use chunked  : {use_chunked}")
    print(f"Questions    : {QUESTIONS}")
    print()

    if use_chunked:
        raw = answer_questions_json_chunked(
            model=MODEL,
            document_text=DOCUMENT_TEXT,
            questions=QUESTIONS,
            context_note=CONTEXT_NOTE,
            max_chars=CHUNK_CHARS,
            overlap=OVERLAP_CHARS,
        )
    else:
        raw = answer_questions_json(
            model=MODEL,
            document_text=DOCUMENT_TEXT,
            questions=QUESTIONS,
            context_note=CONTEXT_NOTE,
        )

    print("Extraction result:")
    print(json.dumps(raw, indent=2))
    return raw


if __name__ == "__main__":
    run_extraction()
