from __future__ import annotations

import json
import secrets
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/api/documents", tags=["documents"])

DATA_ROOT = Path(__file__).resolve().parents[3] / "app" / "data" / "projects"


# ---- Helpers ----

def now_display() -> str:
    return datetime.now(timezone.utc).strftime("%-Y-%m-%d %-I:%M %p")


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def new_file_id() -> str:
    return "f_" + secrets.token_hex(7)[1:]  # 13 hex chars


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


# ---- Endpoint ----

@router.post("/upload", status_code=201)
async def upload_pdf(
    file: UploadFile = File(...),
    project_id: str = Query(..., description="Project ID (e.g. 'p_6c696af63208a8')"),
    uploaded_by: str = Query("", description="Name of the uploader"),
):
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    project_root = DATA_ROOT / project_id
    if not project_root.is_dir():
        raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")

    files_dir = project_root / "files"
    ensure_dir(files_dir)

    # Save PDF using original filename
    stored_path = files_dir / filename
    stored_path.write_bytes(await file.read())

    # Build file metadata
    file_id = new_file_id()
    now = now_display()
    file_record = {
        "id": file_id,
        "fileName": filename,
        "fileType": "application/pdf",
        "uploadedBy": uploaded_by,
        "uploadedAt": now,
        "lastModifiedBy": uploaded_by,
        "lastModifiedAt": now,
        "status": "Uploaded",
        "flag": False,
    }

    # Write sidecar metadata file
    write_json(files_dir / f"{file_id}.json", file_record)

    # Append to project.json files array
    project_manifest = read_json(project_root / "project.json", {})
    project_manifest.setdefault("files", []).append({
        "id": file_id,
        "fileName": filename,
        "lastModified": now,
        "status": "Uploaded",
    })
    project_manifest["lastModified"] = now
    write_json(project_root / "project.json", project_manifest)

    audit_path = project_root / "audit.json"
    audit = read_json(audit_path, [])
    audit.append({
        "ts": now_iso(),
        "action": "file.upload",
        "project_id": project_id,
        "file_id": file_id,
        "fileName": filename,
        "uploadedBy": uploaded_by,
        "status": "successful",
    })
    write_json(audit_path, audit)

    return JSONResponse(status_code=201, content=file_record)
