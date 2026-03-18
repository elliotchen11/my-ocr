from __future__ import annotations

import json
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.python.llm_extract_core import answer_questions_json, answer_questions_json_chunked

router = APIRouter(prefix="/api/runs", tags=["runs"])

DATA_ROOT = Path(__file__).resolve().parents[3] / "app" / "data" / "projects"
TOKEN_THRESHOLD = 25_000


# ---- Helpers ----

def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def now_display() -> str:
    return datetime.now(timezone.utc).strftime("%-Y-%m-%d %-I:%M %p")


def new_run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}_{secrets.token_hex(5)}"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, obj) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def get_project_root(project_id: str) -> Path:
    root = DATA_ROOT / project_id
    if not root.is_dir():
        raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")
    return root


def append_audit(audit_path: Path, entry: dict) -> None:
    """Append one entry to audit.json. Never raises."""
    try:
        try:
            data = json.loads(audit_path.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                data = []
        except Exception:
            data = []
        data.append(entry)
        audit_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def estimate_tokens(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, int(len(text) / 4))


def normalize_answers(result: Any, questions: list[str]) -> dict[str, Any]:
    """Normalize extraction result to {question: {value, confidence}} for all questions."""
    null_entry: dict[str, Any] = {"value": None, "confidence": 0.0}
    if not isinstance(result, dict):
        return {q: null_entry for q in questions}
    out: dict[str, Any] = {}
    for q in questions:
        entry = result.get(q, null_entry)
        if isinstance(entry, dict) and "value" in entry:
            out[q] = {
                "value": entry.get("value"),
                "confidence": float(entry.get("confidence", 0.0)),
            }
        else:
            out[q] = null_entry
    return out


def build_context_note(base: str, standard: dict | None, structure_text: str | None) -> str:
    parts = []
    if base.strip():
        parts.append(base.strip())
    if structure_text and structure_text.strip():
        parts.append("File-type structure / layout notes:\n" + structure_text.strip())
    if standard and isinstance(standard.get("fields"), list):
        lines = ["Data standard (fields + definitions):"]
        for f in standard["fields"]:
            fn = str(f.get("fieldName") or f.get("field") or "").strip()
            if not fn:
                continue
            desc = str(f.get("description") or f.get("definition") or "").strip()
            dtype = str(f.get("dataType") or "").strip()
            line = f"- {fn}: {desc}"
            if dtype:
                line += f" [{dtype}]"
            lines.append(line)
        parts.append("\n".join(lines))
    return "\n\n".join(parts).strip()


# ---- Step function ----

def step_extraction(
    doc_text: str,
    questions: list[str],
    model: str,
    ctx_full: str,
    force_chunking: bool,
    token_threshold: int,
    chunk_chars: int,
    overlap_chars: int,
) -> tuple[dict[str, Any], bool]:
    """
    Run LLM extraction on document text.
    Returns (normalized_answers, use_chunked).
    Raises on failure so the caller can record the error.
    """
    tok_est = estimate_tokens(doc_text)
    use_chunked = force_chunking or (tok_est > token_threshold)

    if use_chunked:
        raw = answer_questions_json_chunked(
            model=model,
            document_text=doc_text,
            questions=questions,
            context_note=ctx_full,
            max_chars=chunk_chars,
            overlap=overlap_chars,
        )
    else:
        raw = answer_questions_json(
            model=model,
            document_text=doc_text,
            questions=questions,
            context_note=ctx_full,
        )

    return normalize_answers(raw, questions), use_chunked


# ---- Request model ----

class StandardField(BaseModel):
    field: str = ""
    fieldName: str = ""
    definition: str = ""
    description: str = ""
    dataType: str = ""
    required: bool = False
    format: str = ""
    allowed_values: str = ""


class StandardSchema(BaseModel):
    fields: list[StandardField] = []
    version: int | None = None


class RunExtractRequest(BaseModel):
    project_id: str
    file_ids: list[str]
    model: str = "gpt-oss"
    standard: StandardSchema | None = None  # Inline standard JSON (same schema as standard_v1.json)
    structure: str | None = None  # Direct text content of the structure file
    context_note: str = ""
    force_chunking: bool = False
    token_threshold: int = TOKEN_THRESHOLD
    chunk_chars: int = 12_000
    overlap_chars: int = 800


# ---- Endpoint ----

@router.post("")
def run_extract(body: RunExtractRequest):
    root = get_project_root(body.project_id)
    audit_path = root / "audit.json"

    try:
        manifest = read_json(root / "project.json", {})
        file_index = {f["id"]: f for f in manifest.get("files", []) if f.get("id")}

        missing = [fid for fid in body.file_ids if fid not in file_index]
        if missing:
            raise HTTPException(status_code=404, detail=f"File IDs not found in project: {missing}")

        standard: dict | None = body.standard.model_dump() if body.standard else None

        structure_text: str | None = body.structure

        questions: list[str] = []
        if standard and isinstance(standard.get("fields"), list):
            questions = [
                str(f.get("fieldName") or f.get("field") or "").strip()
                for f in standard["fields"]
                if str(f.get("fieldName") or f.get("field") or "").strip()
            ]
        if not questions:
            raise HTTPException(status_code=400, detail="No questions to extract. Provide a standard with fields.")

        ctx_full = build_context_note(body.context_note, standard, structure_text)

        run_id = new_run_id()
        run_record: dict = {
            "run_id": run_id,
            "created_at": now_iso(),
            "project_id": body.project_id,
            "model": body.model,
            "context_note": ctx_full,
            "standard": body.standard.model_dump() if body.standard else None,
            "structure": body.structure,
            "files": [],
            "outputs": {},
            "params": {
                "force_chunking": body.force_chunking,
                "token_threshold": body.token_threshold,
                "chunk_chars": body.chunk_chars,
                "overlap_chars": body.overlap_chars,
            },
        }

        errors: list[str] = []

        for fid in body.file_ids:
            file_record = file_index[fid]
            file_name = file_record.get("fileName", "")
            text_path = root / "text" / f"{fid}.txt"
            preview_paths: list[str] = []

            # Collect existing preview paths if available
            preview_dir = root / "previews" / fid
            if preview_dir.is_dir():
                preview_paths = sorted(str(p) for p in preview_dir.iterdir() if p.suffix.lower() == ".png")

            if not text_path.is_file():
                run_record["outputs"][fid] = {q: {"value": None, "confidence": 0.0} for q in questions}
                errors.append(f"{fid}: no text layer found — run OCR first via POST /api/ocr")
                run_record["files"].append({
                    "file_id": fid,
                    "fileName": file_name,
                    "token_estimate": None,
                    "chunked": None,
                    "preview_count": len(preview_paths),
                    "preview_paths": preview_paths,
                })
                continue

            # Extract fields from text
            doc_text = text_path.read_text(encoding="utf-8", errors="replace")
            tok_est = estimate_tokens(doc_text)
            append_audit(audit_path, {"ts": now_iso(), "action": "start step_extraction", "project_id": body.project_id, "file_id": fid})
            try:
                answers, use_chunked = step_extraction(
                    doc_text=doc_text,
                    questions=questions,
                    model=body.model,
                    ctx_full=ctx_full,
                    force_chunking=body.force_chunking,
                    token_threshold=body.token_threshold,
                    chunk_chars=body.chunk_chars,
                    overlap_chars=body.overlap_chars,
                )
            except Exception as e:
                append_audit(audit_path, {"ts": now_iso(), "action": "complete step_extraction", "project_id": body.project_id, "file_id": fid, "status": "failed", "error": str(e)})
                run_record["outputs"][fid] = {q: {"value": None, "confidence": 0.0} for q in questions}
                errors.append(f"{fid}: extraction failed — {e}")
                run_record["files"].append({
                    "file_id": fid,
                    "fileName": file_name,
                    "token_estimate": tok_est,
                    "chunked": None,
                    "preview_count": len(preview_paths),
                    "preview_paths": preview_paths,
                })
                continue
            append_audit(audit_path, {"ts": now_iso(), "action": "complete step_extraction", "project_id": body.project_id, "file_id": fid, "status": "successful"})

            run_record["outputs"][fid] = answers
            run_record["files"].append({
                "file_id": fid,
                "fileName": file_name,
                "token_estimate": tok_est,
                "chunked": use_chunked,
                "preview_count": len(preview_paths),
                "preview_paths": preview_paths,
            })

        runs_dir = root / "runs"
        ensure_dir(runs_dir)
        write_json(runs_dir / f"run_{run_id}.json", run_record)

        manifest["lastModified"] = now_display()
        write_json(root / "project.json", manifest)

        if errors:
            run_record["errors"] = errors

        append_audit(audit_path, {
            "ts": now_iso(),
            "action": "run.save",
            "project_id": body.project_id,
            "run_id": run_id,
            "file_ids": body.file_ids,
            "status": "successful",
        })

        flat_outputs = [
            {"question": q, "answer": v["value"], "confidence": v["confidence"]}
            for answers in run_record["outputs"].values()
            for q, v in answers.items()
        ]

        return JSONResponse(content={
            "run_id": run_id,
            "project_id": body.project_id,
            "model": body.model,
            "file_ids": body.file_ids,
            "outputs": flat_outputs,
        })

    except Exception as e:
        append_audit(audit_path, {
            "ts": now_iso(),
            "action": "run.save",
            "project_id": body.project_id,
            "status": "failed",
            "error": str(e),
        })
        raise
