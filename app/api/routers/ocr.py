from __future__ import annotations

import base64
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx
from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.python.convert_to_img import convert_pdf2img
from app.python.ocr import ocr_image, DEFAULT_PROMPT, DEFAULT_TIMEOUT, IMAGE_EXTS

router = APIRouter(prefix="/api/ocr", tags=["ocr"])

DATA_ROOT = Path(__file__).resolve().parents[3] / "app" / "data" / "projects"


# ---- Helpers ----

def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


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


# ---- Step functions ----

def step_convert_to_img(pdf_path: Path, out_dir: Path, file_id: str, zoom: float = 2.0) -> list[str]:
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
            p = Path(img_path)
            if not p.is_file() or p.stat().st_size == 0:
                msg = f"image file missing or empty: {img_path}"
                ocr_errors.append(f"page {i}: {msg}")
                fh.write(f"===== PAGE {i} OCR FAILED =====\n{msg}\n\n")
                fh.flush()
                continue
            try:
                page_text = ocr_image(
                    model=ocr_model,
                    image_path=p,
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


# ---- Request model ----

class RunOCRRequest(BaseModel):
    project_id: str
    file_ids: list[str]
    ocr_model: str = "mistral-small3.2"
    ocr_prompt: str = DEFAULT_PROMPT
    ocr_timeout: float = DEFAULT_TIMEOUT
    force_rerun: bool = False
    callback: str | None = None  # Full URL (http/https) to POST results to when done

    from pydantic import field_validator

    @field_validator("callback")
    @classmethod
    def callback_must_be_absolute(cls, v: str | None) -> str | None:
        if v and not v.startswith(("http://", "https://")):
            raise ValueError("callback must be an absolute URL starting with http:// or https://")
        return v


# ---- Background worker ----

def _run_ocr_job(job_id: str, body: RunOCRRequest) -> None:
    root = DATA_ROOT / body.project_id
    audit_path = root / "audit.json"

    manifest = read_json(root / "project.json", {})
    file_index = {f["id"]: f for f in manifest.get("files", []) if f.get("id")}

    results: list[dict] = []

    for fid in body.file_ids:
        file_record = file_index.get(fid)
        if not file_record:
            results.append({"project_id": body.project_id, "file_id": fid, "error": "file not found"})
            continue

        file_name = file_record.get("fileName", "")
        pdf_path = root / "files" / file_name
        text_path = root / "text" / f"{fid}.txt"

        if body.force_rerun and text_path.is_file():
            text_path.unlink()

        append_audit(audit_path, {"ts": now_iso(), "action": "start step_create_preview", "job_id": job_id, "project_id": body.project_id, "file_id": fid})
        preview_paths, preview_err = step_create_preview(
            pdf_path=pdf_path,
            file_name=file_name,
            preview_dir=root / "previews" / fid,
            file_id=fid,
        )
        append_audit(audit_path, {"ts": now_iso(), "action": "complete step_create_preview", "job_id": job_id, "project_id": body.project_id, "file_id": fid, "preview_count": len(preview_paths), "error": preview_err})

        skipped = text_path.is_file()
        append_audit(audit_path, {"ts": now_iso(), "action": "start step_ocr", "job_id": job_id, "project_id": body.project_id, "file_id": fid, "skipped": skipped})
        ocr_errs = step_ocr(
            pdf_path=pdf_path,
            preview_paths=preview_paths,
            text_path=text_path,
            ocr_model=body.ocr_model,
            ocr_prompt=body.ocr_prompt,
            ocr_timeout=body.ocr_timeout,
        )
        append_audit(audit_path, {"ts": now_iso(), "action": "complete step_ocr", "job_id": job_id, "project_id": body.project_id, "file_id": fid, "skipped": skipped, "status": "failed" if ocr_errs else "successful", "errors": ocr_errs})

        file_contents: list[dict] = []
        if text_path.is_file():
            file_contents.append({
                "contentType": "text/plain",
                "contentBase64": base64.b64encode(text_path.read_bytes()).decode("utf-8"),
            })
        for preview_path in preview_paths:
            p = Path(preview_path)
            if p.is_file():
                file_contents.append({
                    "contentType": "image/png",
                    "contentBase64": base64.b64encode(p.read_bytes()).decode("utf-8"),
                })

        results.append({
            "project_id": body.project_id,
            "file_id": fid,
            "ocr_skipped": skipped,
            "ocr_errors": ocr_errs,
            "preview_error": preview_err,
            "files": file_contents,
        })

    if body.callback:
        payload = {"job_id": job_id, "status": "completed", "results": results}
        try:
            with httpx.Client(timeout=30) as client:
                client.post(body.callback, json=payload)
        except Exception as e:
            append_audit(audit_path, {"ts": now_iso(), "action": "callback_failed", "job_id": job_id, "project_id": body.project_id, "error": str(e)})


# ---- Endpoint ----

@router.post("")
def run_ocr(body: RunOCRRequest, background_tasks: BackgroundTasks):
    root = get_project_root(body.project_id)

    manifest = read_json(root / "project.json", {})
    file_index = {f["id"]: f for f in manifest.get("files", []) if f.get("id")}

    missing = [fid for fid in body.file_ids if fid not in file_index]
    if missing:
        raise HTTPException(status_code=404, detail=f"File IDs not found in project: {missing}")

    # If a callback URL is provided, run in background and return immediately
    if body.callback:
        job_id = str(uuid.uuid4())
        background_tasks.add_task(_run_ocr_job, job_id, body)
        return JSONResponse(content={"job_id": job_id, "status": "queued"})

    # No callback — run synchronously and return results directly
    job_id = str(uuid.uuid4())
    results: list[dict] = []
    audit_path = root / "audit.json"

    for fid in body.file_ids:
        file_record = file_index[fid]
        file_name = file_record.get("fileName", "")
        pdf_path = root / "files" / file_name
        text_path = root / "text" / f"{fid}.txt"

        if body.force_rerun and text_path.is_file():
            text_path.unlink()

        append_audit(audit_path, {"ts": now_iso(), "action": "start step_create_preview", "project_id": body.project_id, "file_id": fid})
        preview_paths, preview_err = step_create_preview(
            pdf_path=pdf_path,
            file_name=file_name,
            preview_dir=root / "previews" / fid,
            file_id=fid,
        )
        append_audit(audit_path, {"ts": now_iso(), "action": "complete step_create_preview", "project_id": body.project_id, "file_id": fid, "preview_count": len(preview_paths), "error": preview_err})

        skipped = text_path.is_file()
        append_audit(audit_path, {"ts": now_iso(), "action": "start step_ocr", "project_id": body.project_id, "file_id": fid, "skipped": skipped})
        ocr_errs = step_ocr(
            pdf_path=pdf_path,
            preview_paths=preview_paths,
            text_path=text_path,
            ocr_model=body.ocr_model,
            ocr_prompt=body.ocr_prompt,
            ocr_timeout=body.ocr_timeout,
        )
        append_audit(audit_path, {"ts": now_iso(), "action": "complete step_ocr", "project_id": body.project_id, "file_id": fid, "skipped": skipped, "status": "failed" if ocr_errs else "successful", "errors": ocr_errs})

        file_contents: list[dict] = []
        if text_path.is_file():
            file_contents.append({
                "contentType": "text/plain",
                "contentBase64": base64.b64encode(text_path.read_bytes()).decode("utf-8"),
            })
        for preview_path in preview_paths:
            p = Path(preview_path)
            if p.is_file():
                file_contents.append({
                    "contentType": "image/png",
                    "contentBase64": base64.b64encode(p.read_bytes()).decode("utf-8"),
                })

        results.append({
            "project_id": body.project_id,
            "file_id": fid,
            "ocr_skipped": skipped,
            "ocr_errors": ocr_errs,
            "preview_error": preview_err,
            "files": file_contents,
        })

    return JSONResponse(content=results)
