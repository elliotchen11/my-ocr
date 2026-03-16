from __future__ import annotations

import json
import secrets
import shutil
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

router = APIRouter(prefix="/api/projects", tags=["projects"])

DATA_ROOT = Path(__file__).resolve().parents[3] / "app" / "data" / "projects"


# ---- Helpers ----

def now_display() -> str:
    """Return timestamp in the format used by project.json: '2026-03-12 3:25 PM'"""
    return datetime.now(timezone.utc).strftime("%-Y-%m-%d %-I:%M %p")


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def new_project_id() -> str:
    return "p_" + secrets.token_hex(7)


def new_structure_id() -> str:
    return "s_" + secrets.token_hex(6)[1:]  # 11 hex chars, matching schema


def new_standard_id() -> str:
    return "ds_" + secrets.token_hex(6)  # 12 hex chars, matching schema


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


def append_audit(audit_path: Path, entry: dict) -> None:
    """Append one entry to audit.json, creating it if absent. Never raises."""
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


def ensure_project_layout(root: Path) -> None:
    for subdir in ("files", "structures", "data_standards", "profiles", "previews", "text", "runs"):
        ensure_dir(root / subdir)


def get_project_root(project_id: str) -> Path:
    root = DATA_ROOT / project_id
    if not root.is_dir():
        raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")
    return root


# ---- Request models ----

class CreateProjectRequest(BaseModel):
    displayName: str
    codeName: str = ""
    createdBy: str = ""
    createdByEmail: str = ""


class UpdateProjectRequest(BaseModel):
    displayName: str | None = None
    codeName: str | None = None


class AddStructureRequest(BaseModel):
    name: str
    description: str = ""
    createdBy: str = ""
    content: str


class StandardField(BaseModel):
    fieldName: str
    description: str = ""
    dataType: str = ""
    goldenExample: str = ""
    location: str = ""


class AddStandardRequest(BaseModel):
    name: str
    createdBy: str = ""
    fields: list[StandardField] = []


# ---- Endpoints ----

@router.get("")
def get_all_projects():
    ensure_dir(DATA_ROOT)
    projects = []
    for p in sorted(DATA_ROOT.iterdir()):
        if not p.is_dir() or not p.name.startswith("p_"):
            continue
        manifest = read_json(p / "project.json", {})
        projects.append({
            "id": manifest.get("id", p.name),
            "displayName": manifest.get("displayName", ""),
            "codeName": manifest.get("codeName", ""),
            "createdBy": manifest.get("createdBy", ""),
            "lastModified": manifest.get("lastModified"),
        })
    return projects


@router.get("/{project_id}")
def get_project_by_id(project_id: str):
    root = get_project_root(project_id)
    return read_json(root / "project.json", {})


@router.post("", status_code=201)
def create_project(body: CreateProjectRequest):
    if not body.displayName.strip():
        raise HTTPException(status_code=400, detail="displayName is required.")

    project_id = new_project_id()
    root = DATA_ROOT / project_id
    ensure_project_layout(root)

    manifest = {
        "id": project_id,
        "displayName": body.displayName.strip(),
        "codeName": body.codeName.strip(),
        "createdBy": body.createdBy.strip(),
        "createdByEmail": body.createdByEmail.strip(),
        "lastModified": now_display(),
        "files": [],
        "structures": [],
        "data_standards": [],
        "text": [],
    }
    write_json(root / "project.json", manifest)

    audit_entry = {
        "ts": now_iso(),
        "action": "project.created",
        "project_id": project_id,
        "status": "successful",
    }
    write_json(root / "audit.json", [audit_entry])

    return JSONResponse(status_code=201, content=manifest)


@router.patch("/{project_id}")
def update_project(project_id: str, body: UpdateProjectRequest):
    root = get_project_root(project_id)
    manifest = read_json(root / "project.json", {})

    if body.displayName is not None:
        manifest["displayName"] = body.displayName.strip()
    if body.codeName is not None:
        manifest["codeName"] = body.codeName.strip()

    manifest["lastModified"] = now_display()
    write_json(root / "project.json", manifest)

    audit_path = root / "audit.json"
    audit = read_json(audit_path, [])
    audit.append({
        "ts": now_iso(),
        "action": "project.updated",
        "project_id": project_id,
        "status": "successful",
    })
    write_json(audit_path, audit)

    return manifest


@router.post("/{project_id}/structures", status_code=201)
def add_structure(project_id: str, body: AddStructureRequest):
    name = (body.name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="Structure name is required.")

    root = get_project_root(project_id)
    audit_path = root / "audit.json"

    try:
        structures_dir = root / "structures"
        ensure_dir(structures_dir)

        file_path = structures_dir / name
        file_path.write_text(body.content or "", encoding="utf-8")

        now = now_display()
        structure_record = {
            "id": new_structure_id(),
            "name": name,
            "description": body.description.strip(),
            "createdBy": body.createdBy.strip(),
            "createdAt": now,
            "lastModified": now,
        }

        manifest = read_json(root / "project.json", {})
        manifest.setdefault("structures", []).append(structure_record)
        manifest["lastModified"] = now
        write_json(root / "project.json", manifest)

        append_audit(audit_path, {
            "ts": now_iso(),
            "action": "structure.add",
            "project_id": project_id,
            "structure_id": structure_record["id"],
            "name": name,
            "status": "successful",
        })

        return JSONResponse(status_code=201, content=structure_record)

    except Exception as e:
        append_audit(audit_path, {
            "ts": now_iso(),
            "action": "structure.add",
            "project_id": project_id,
            "name": name,
            "status": "failed",
            "error": str(e),
        })
        raise


@router.post("/{project_id}/data-standards", status_code=201)
def add_standard(project_id: str, body: AddStandardRequest):
    name = (body.name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="Standard name is required.")

    root = get_project_root(project_id)
    audit_path = root / "audit.json"

    try:
        standards_dir = root / "data_standards"
        ensure_dir(standards_dir)

        now = now_display()
        standard_id = new_standard_id()

        standard_record = {
            "id": standard_id,
            "name": name,
            "createdBy": body.createdBy.strip(),
            "createdAt": now,
            "lastModified": now,
            "fields": [f.model_dump() for f in body.fields],
        }

        write_json(standards_dir / f"{standard_id}.json", standard_record)

        manifest = read_json(root / "project.json", {})
        manifest.setdefault("data_standards", []).append({
            "id": standard_id,
            "name": name,
            "createdBy": body.createdBy.strip(),
            "createdAt": now,
            "lastModified": now,
        })
        manifest["lastModified"] = now
        write_json(root / "project.json", manifest)

        append_audit(audit_path, {
            "ts": now_iso(),
            "action": "standard.add",
            "project_id": project_id,
            "standard_id": standard_id,
            "name": name,
            "status": "successful",
        })

        return JSONResponse(status_code=201, content=standard_record)

    except Exception as e:
        append_audit(audit_path, {
            "ts": now_iso(),
            "action": "standard.add",
            "project_id": project_id,
            "name": name,
            "status": "failed",
            "error": str(e),
        })
        raise


@router.delete("/{project_id}", status_code=204)
def delete_project(project_id: str):
    root = (DATA_ROOT / project_id).resolve()
    base = DATA_ROOT.resolve()

    if root == base or base not in root.parents:
        raise HTTPException(status_code=400, detail="Refusing to delete outside projects root.")

    get_project_root(project_id)  # raises 404 if not found

    try:
        shutil.rmtree(root)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {e}")
