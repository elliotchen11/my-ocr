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
from app.python.convert_to_img import convert_pdf2img
from app.python.ocr import ocr_image, DEFAULT_PROMPT, DEFAULT_TIMEOUT, IMAGE_EXTS

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
    if not isinstance(result, dict):
        return {q: None for q in questions}
    answers = result.get("answers") if isinstance(result.get("answers"), dict) else result
    if not isinstance(answers, dict):
        return {q: None for q in questions}
    out: dict[str, Any] = {}
    for i, q in enumerate(questions):
        if q in answers:
            out[q] = answers[q]
        elif str(i) in answers:
            out[q] = answers[str(i)]
        else:
            out[q] = None
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


# ---- Step functions ----

def step_convert_to_img(pdf_path: Path, out_dir: Path, file_id: str, zoom: float = 2.0) -> list[str]:
    """Convert a PDF to page images. Returns list of image paths."""
    ensure_dir(out_dir)
    return convert_pdf2img(
        input_file=str(pdf_path),
        out_dir=str(out_dir),
        base_override=file_id,
        zoom=zoom,
    )


def step_create_preview(
    pdf_path: Path,
    file_name: str,
    preview_dir: Path,
    file_id: str,
) -> tuple[list[str], str | None]:
    """
    Generate preview images for a PDF file.
    Returns (preview_paths, error_message).
    """
    if not pdf_path.is_file() or not file_name.lower().endswith(".pdf"):
        return [], None
    try:
        preview_paths = step_convert_to_img(pdf_path, preview_dir, file_id)
        return preview_paths, None
    except Exception as e:
        return [], f"preview generation failed — {e}"


def step_ocr(
    pdf_path: Path,
    preview_paths: list[str],
    text_path: Path,
    ocr_model: str,
    ocr_prompt: str,
    ocr_timeout: float,
) -> list[str]:
    """
    Run OCR on preview images (PDF) or the file directly (image).
    Writes text to text_path incrementally (one page at a time).
    Returns list of per-page error messages.
    Skips if text_path already exists.
    """
    if text_path.is_file():
        return []

    images_to_ocr: list[str] = []
    if preview_paths:
        images_to_ocr = preview_paths
    elif pdf_path.is_file() and pdf_path.suffix.lower() in IMAGE_EXTS:
        images_to_ocr = [str(pdf_path)]

    if not images_to_ocr:
        return []

    ensure_dir(text_path.parent)
    ocr_errors: list[str] = []
    with text_path.open("w", encoding="utf-8") as fh:
        for i, img_path in enumerate(images_to_ocr, start=1):
            try:
                page_text = ocr_image(
                    model=ocr_model,
                    image_path=Path(img_path),
                    prompt=ocr_prompt,
                    timeout=ocr_timeout,
                ).strip()
                fh.write(f"===== PAGE {i} =====\n{page_text}\n\n")
                fh.flush()
            except Exception as e:
                ocr_errors.append(f"page {i} ({img_path}): {e}")
                fh.write(f"===== PAGE {i} OCR FAILED =====\n{e}\n\n")
                fh.flush()
    return ocr_errors


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

class RunExtractRequest(BaseModel):
    project_id: str
    file_ids: list[str]
    model: str = "gpt-oss"
    ocr_model: str = "mistral-small3.2"
    ocr_prompt: str = DEFAULT_PROMPT
    ocr_timeout: float = DEFAULT_TIMEOUT
    standard_id: str | None = None
    structure_name: str | None = None
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

        standard: dict | None = None
        if body.standard_id:
            std_path = root / "data_standards" / f"{body.standard_id}.json"
            if not std_path.is_file():
                raise HTTPException(status_code=404, detail=f"Standard not found: {body.standard_id}")
            standard = read_json(std_path, {})

        structure_text: str | None = None
        if body.structure_name:
            struct_path = root / "structures" / body.structure_name
            if not struct_path.is_file():
                raise HTTPException(status_code=404, detail=f"Structure not found: {body.structure_name}")
            structure_text = struct_path.read_text(encoding="utf-8")

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
            "standard_id": body.standard_id,
            "structure_name": body.structure_name,
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
            pdf_path = root / "files" / file_name

            # Step 1: generate preview images (also used for OCR)
            append_audit(audit_path, {"ts": now_iso(), "action": "start step_create_preview", "project_id": body.project_id, "file_id": fid})
            preview_paths, preview_err = step_create_preview(
                pdf_path=pdf_path,
                file_name=file_name,
                preview_dir=root / "previews" / fid,
                file_id=fid,
            )
            if preview_err:
                errors.append(f"{fid}: {preview_err}")
            append_audit(audit_path, {"ts": now_iso(), "action": "complete step_create_preview", "project_id": body.project_id, "file_id": fid, "preview_count": len(preview_paths), "error": preview_err})

            # Step 2: OCR images → text file
            text_path = root / "text" / f"{fid}.txt"
            append_audit(audit_path, {"ts": now_iso(), "action": "start step_ocr", "project_id": body.project_id, "file_id": fid})
            ocr_errs = step_ocr(
                pdf_path=pdf_path,
                preview_paths=preview_paths,
                text_path=text_path,
                ocr_model=body.ocr_model,
                ocr_prompt=body.ocr_prompt,
                ocr_timeout=body.ocr_timeout,
            )
            if ocr_errs:
                errors.extend(f"{fid}: OCR — {e}" for e in ocr_errs)
            append_audit(audit_path, {"ts": now_iso(), "action": "complete step_ocr", "project_id": body.project_id, "file_id": fid, "status": "failed" if ocr_errs else "successful", "errors": ocr_errs})

            if not text_path.is_file():
                run_record["outputs"][fid] = {q: None for q in questions}
                errors.append(f"{fid}: no text layer found at {text_path}")
                run_record["files"].append({
                    "file_id": fid,
                    "fileName": file_name,
                    "token_estimate": None,
                    "chunked": None,
                    "preview_count": len(preview_paths),
                    "preview_paths": preview_paths,
                })
                continue

            # Step 3: extract fields from text
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
                run_record["outputs"][fid] = {q: None for q in questions}
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

        return JSONResponse(content=run_record)

    except Exception as e:
        append_audit(audit_path, {
            "ts": now_iso(),
            "action": "run.save",
            "project_id": body.project_id,
            "status": "failed",
            "error": str(e),
        })
        raise
