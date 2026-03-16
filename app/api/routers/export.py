from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.python.llm_extract_core import answer_questions_json, answer_questions_json_chunked

router = APIRouter(prefix="/api/exports", tags=["exports"])

DATA_ROOT = Path(__file__).resolve().parents[3] / "app" / "data" / "projects"
TOKEN_THRESHOLD = 25_000


# ---- Helpers ----

def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def find_run(run_id: str) -> tuple[Path, dict]:
    """Search all projects for a run file. Returns (run_path, run_record)."""
    for project_dir in DATA_ROOT.iterdir():
        if not project_dir.is_dir() or not project_dir.name.startswith("p_"):
            continue
        run_path = project_dir / "runs" / f"run_{run_id}.json"
        if run_path.is_file():
            try:
                record = json.loads(run_path.read_text(encoding="utf-8"))
            except Exception:
                raise HTTPException(status_code=500, detail=f"Failed to read run file: {run_id}")
            return run_path, record
    raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")


def save_run(run_path: Path, record: dict) -> None:
    run_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")


def load_standard(project_root: Path, standard_id: str) -> dict:
    std_path = project_root / "data_standards" / f"{standard_id}.json"
    if not std_path.is_file():
        return {}
    try:
        return json.loads(std_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def get_questions(run_record: dict, project_root: Path) -> list[str]:
    standard_id = run_record.get("standard_id")
    if standard_id:
        standard = load_standard(project_root, standard_id)
        fields = standard.get("fields") or []
        questions = [str(f.get("fieldName") or "").strip() for f in fields if str(f.get("fieldName") or "").strip()]
        if questions:
            return questions
    # Fall back to keys from first output
    outputs = run_record.get("outputs") or {}
    if outputs:
        first = next(iter(outputs.values()))
        if isinstance(first, dict):
            return list(first.keys())
    return []


def completeness(outputs: dict, questions: list[str]) -> float:
    if not questions:
        return 0.0
    filled = sum(1 for q in questions if outputs.get(q) not in (None, "", "null"))
    return filled / len(questions)


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


# ---- Request models ----

class SaveReviewRequest(BaseModel):
    outputs: dict[str, Any] | None = None      # updated field values
    status: str = "unverified"                  # unverified | verified | flagged
    note: str = ""


class SaveTextRequest(BaseModel):
    text: str


class RerunRequest(BaseModel):
    model: str
    save_text_first: bool = True
    force_chunking: bool = False
    token_threshold: int = TOKEN_THRESHOLD
    chunk_chars: int = 12_000
    overlap_chars: int = 800


# ---- Image export endpoint ----

@router.get("/images/{run_id}")
def export_img(run_id: str):
    """Return a ZIP archive of all preview images produced by convert_to_img for the given run."""
    _, run_record = find_run(run_id)

    image_files: list[Path] = []
    for file_entry in run_record.get("files", []):
        for p in file_entry.get("preview_paths", []):
            img = Path(p)
            if img.is_file():
                image_files.append(img)

    if not image_files:
        raise HTTPException(status_code=404, detail="No preview images found for this run.")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for img_path in image_files:
            arcname = Path(img_path.parent.name) / img_path.name
            zf.write(img_path, arcname)

    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{run_id}_images.zip"'},
    )


# ---- Review endpoints ----

@router.get("/reviews/{project_id}")
def get_review(
    project_id: str,
    run_id: str | None = Query(default=None, description="Run ID to review. Defaults to the most recent run."),
):
    """
    Return a comprehensive review object for a project: run, file metadata, previews, text, and extraction outputs.
    Defaults to the most recent run; pass run_id to select a specific one.
    """
    project_root = DATA_ROOT / project_id
    if not project_root.is_dir():
        raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")

    # Load project manifest for file metadata
    manifest_path = project_root / "project.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    except Exception:
        manifest = {}
    manifest_files = {f["id"]: f for f in manifest.get("files", []) if f.get("id")}

    # Resolve run: use specified run_id or pick the most recent by filename
    runs_dir = project_root / "runs"
    if not runs_dir.is_dir() or not any(runs_dir.iterdir()):
        raise HTTPException(status_code=404, detail="No runs found for this project.")

    all_run_files = sorted(runs_dir.glob("run_*.json"))
    available_runs = [p.stem[len("run_"):] for p in all_run_files]

    if run_id:
        run_path = runs_dir / f"run_{run_id}.json"
        if not run_path.is_file():
            raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    else:
        run_path = all_run_files[-1]
        run_id = run_path.stem[len("run_"):]

    try:
        run_record = json.loads(run_path.read_text(encoding="utf-8"))
    except Exception:
        raise HTTPException(status_code=500, detail=f"Failed to read run file: {run_id}")

    questions = get_questions(run_record, project_root)
    outputs: dict = run_record.get("outputs") or {}
    validation: dict = run_record.get("validation") or {}
    run_files_meta = {fe["file_id"]: fe for fe in run_record.get("files", []) if fe.get("file_id")}

    # Build per-file review objects
    files_out = []
    for fid, fe in run_files_meta.items():
        # File metadata from project manifest
        file_obj = manifest_files.get(fid, {})

        # Preview images
        previews_dir = project_root / "previews" / fid
        preview_paths = sorted(str(p) for p in previews_dir.glob("*.png")) if previews_dir.is_dir() else fe.get("preview_paths", [])

        # Text layer
        text_path = project_root / "text" / f"{fid}.txt"
        text_content = text_path.read_text(encoding="utf-8", errors="replace") if text_path.is_file() else ""

        # Extraction outputs + validation
        out_dict = outputs.get(fid) or {}
        val = validation.get(fid) or {"status": "unverified", "note": ""}

        files_out.append({
            "file": {
                "file_id": fid,
                "fileName": fe.get("fileName", file_obj.get("fileName", "")),
                "status": file_obj.get("status", ""),
                "uploadedAt": file_obj.get("uploadedAt", ""),
                "uploadedBy": file_obj.get("uploadedBy", ""),
                "token_estimate": fe.get("token_estimate"),
                "chunked": fe.get("chunked"),
            },
            "previews": preview_paths,
            "text": text_content,
            "extraction": {
                "outputs": out_dict,
                "completeness_pct": int(completeness(out_dict, questions) * 100),
            },
            "validation": val,
        })

    total = len(files_out)
    validations = [f["validation"].get("status", "unverified") for f in files_out]

    return {
        "project_id": project_id,
        "run": {
            "run_id": run_id,
            "created_at": run_record.get("created_at"),
            "model": run_record.get("model"),
            "standard_id": run_record.get("standard_id"),
            "structure_name": run_record.get("structure_name"),
            "context_note": run_record.get("context_note"),
            "params": run_record.get("params"),
            "errors": run_record.get("errors", []),
        },
        "available_runs": available_runs,
        "questions": questions,
        "summary": {
            "total": total,
            "unverified": validations.count("unverified"),
            "verified": validations.count("verified"),
            "flagged": validations.count("flagged"),
        },
        "files": files_out,
    }


@router.patch("/reviews/{project_id}/files/{file_id}")
def save_review(project_id: str, file_id: str, body: SaveReviewRequest, run_id: str = Query(...)):
    """Save field edits and validation status/note for a file in a run."""
    if body.status not in ("unverified", "verified", "flagged"):
        raise HTTPException(status_code=400, detail="status must be unverified, verified, or flagged.")
    if body.status == "flagged" and not (body.note or "").strip():
        raise HTTPException(status_code=400, detail="A note is required when flagging.")

    run_path, run_record = find_run(run_id)

    outputs: dict = run_record.setdefault("outputs", {})
    if file_id not in outputs:
        raise HTTPException(status_code=404, detail=f"File {file_id} not found in run {run_id}.")

    before = dict(outputs.get(file_id) or {})
    if body.outputs is not None:
        outputs[file_id] = body.outputs

    validation: dict = run_record.setdefault("validation", {})
    validation[file_id] = {
        "status": body.status,
        "note": body.note,
        "updated_at": now_iso(),
    }

    changed_fields = [k for k in (body.outputs or {}) if str(before.get(k)) != str((body.outputs or {}).get(k))]
    audit = run_record.setdefault("audit", [])
    audit.append({
        "ts": now_iso(),
        "action": "review.save",
        "file_id": file_id,
        "status": body.status,
        "changed_fields": changed_fields,
    })

    save_run(run_path, run_record)
    return {"run_id": run_id, "file_id": file_id, "validation": validation[file_id], "outputs": outputs[file_id]}


@router.put("/reviews/{project_id}/files/{file_id}/text")
def save_text_layer(project_id: str, file_id: str, body: SaveTextRequest, run_id: str = Query(...)):
    """Save (or overwrite) the extracted text layer for a file."""
    run_path, run_record = find_run(run_id)
    project_root = run_path.parents[1]

    outputs: dict = run_record.get("outputs") or {}
    if file_id not in outputs:
        raise HTTPException(status_code=404, detail=f"File {file_id} not found in run {run_id}.")

    text_path = project_root / "text" / f"{file_id}.txt"
    ensure_dir(text_path.parent)
    text_path.write_text(body.text or "", encoding="utf-8")

    audit = run_record.setdefault("audit", [])
    audit.append({"ts": now_iso(), "action": "file.text_layer_saved", "file_id": file_id})
    save_run(run_path, run_record)

    return {"run_id": run_id, "file_id": file_id, "text_path": str(text_path)}


@router.post("/reviews/{project_id}/files/{file_id}/rerun")
def rerun_extraction(project_id: str, file_id: str, body: RerunRequest, run_id: str = Query(...)):
    """Re-run field extraction for a single file using its current text layer."""
    run_path, run_record = find_run(run_id)
    project_root = run_path.parents[1]

    outputs: dict = run_record.get("outputs") or {}
    if file_id not in outputs:
        raise HTTPException(status_code=404, detail=f"File {file_id} not found in run {run_id}.")

    text_path = project_root / "text" / f"{file_id}.txt"
    if not text_path.is_file():
        raise HTTPException(status_code=404, detail=f"No text layer found for file {file_id}.")

    doc_text = text_path.read_text(encoding="utf-8", errors="replace")
    questions = get_questions(run_record, project_root)
    if not questions:
        raise HTTPException(status_code=400, detail="No questions to extract. Run has no associated standard.")

    ctx = (run_record.get("context_note") or "").strip()
    tok_est = estimate_tokens(doc_text)
    use_chunked = body.force_chunking or (tok_est > body.token_threshold)

    try:
        if use_chunked:
            raw = answer_questions_json_chunked(
                model=body.model,
                document_text=doc_text,
                questions=questions,
                context_note=ctx,
                max_chars=body.chunk_chars,
                overlap=body.overlap_chars,
            )
        else:
            raw = answer_questions_json(
                model=body.model,
                document_text=doc_text,
                questions=questions,
                context_note=ctx,
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {e}")

    new_outputs = normalize_answers(raw, questions)
    run_record.setdefault("outputs", {})[file_id] = new_outputs

    # Reset validation to unverified after rerun
    run_record.setdefault("validation", {})[file_id] = {
        "status": "unverified",
        "note": f"Re-run at {now_iso()} with model={body.model}",
        "updated_at": now_iso(),
    }

    # Update file metadata in run
    for fe in run_record.get("files", []):
        if fe.get("file_id") == file_id:
            fe["token_estimate"] = tok_est
            fe["chunked"] = use_chunked
            fe["rerun_at"] = now_iso()
            fe["rerun_model"] = body.model
            break

    run_record.setdefault("audit", []).append({
        "ts": now_iso(),
        "action": "extraction.rerun",
        "file_id": file_id,
        "model": body.model,
        "chunked": use_chunked,
    })

    save_run(run_path, run_record)

    return {
        "run_id": run_id,
        "file_id": file_id,
        "model": body.model,
        "chunked": use_chunked,
        "token_estimate": tok_est,
        "outputs": new_outputs,
        "validation": run_record["validation"][file_id],
    }
