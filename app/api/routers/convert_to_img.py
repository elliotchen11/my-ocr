from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from app.python.convert_to_img import convert_pdf2img

router = APIRouter(prefix="/api/convert", tags=["convert"])


@dataclass
class ProjectPaths:
    root: Path
    manifest: Path
    audit: Path
    files_manifest: Path
    files_dir: Path
    text_dir: Path
    previews_dir: Path
    standards_dir: Path
    structures_dir: Path
    runs_dir: Path
    queries_dir: Path
    exports_dir: Path


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _preview_dir_for_file(pp: ProjectPaths, file_id: str) -> Path:
    return pp.previews_dir / file_id


def list_preview_images(pp: ProjectPaths, file_record: dict) -> list[Path]:
    file_id = str(file_record.get("file_id") or "").strip()
    if not file_id:
        return []
    d = _preview_dir_for_file(pp, file_id)
    if not d.is_dir():
        return []

    def page_key(p: Path) -> tuple[int, str]:
        m = re.search(r"(?i)_page(\d+)\b", p.stem)
        return (int(m.group(1)) if m else 0, p.name)

    return sorted(d.glob("*.png"), key=page_key)


def build_pdf_previews(
    pp: ProjectPaths,
    file_record: dict,
    *,
    zoom: float,
    rotate: int,
    max_pages: int,
    force: bool = False,
) -> tuple[list[Path], str | None]:
    """
    Render PDF pages to PNGs using convert_to_img.py (PyMuPDF) and store under project/previews/<file_id>/.
    Returns: (image_paths, error)
    """
    stored_path = Path(str(file_record.get("stored_path") or ""))
    if not stored_path.is_file() or stored_path.suffix.lower() != ".pdf":
        return [], "Not a PDF file."

    file_id = str(file_record.get("file_id") or "").strip()
    if not file_id:
        return [], "Missing file_id."

    out_dir = _preview_dir_for_file(pp, file_id)
    ensure_dir(out_dir)

    existing = list_preview_images(pp, file_record)
    if existing and not force:
        return existing, None

    # Clear previous previews if forcing rebuild
    if force and out_dir.is_dir():
        for p in out_dir.glob("*.png"):
            try:
                p.unlink()
            except Exception:
                pass

    # Limit pages (convert_pdf2img expects 1-based page numbers)
    pages = list(range(1, max_pages + 1)) if max_pages and max_pages > 0 else None

    try:
        outputs = convert_pdf2img(
            input_file=str(stored_path),
            out_dir=str(out_dir),
            base_override=file_id,
            pages=pages,
            zoom=float(zoom),
            rotate=int(rotate),
        )
        # convert_pdf2img returns list[str]
        imgs = [Path(x) for x in outputs if x]
        imgs = [p for p in imgs if p.is_file()]
        imgs = sorted(imgs, key=lambda p: p.name)
        return imgs, None if imgs else "Rendered 0 preview images."
    except Exception as e:
        return [], f"Failed to render PDF previews: {e}"


@router.post("/pdf-previews")
async def pdf_previews_endpoint(
    file: UploadFile = File(...),
    zoom: float = Query(2.0, gt=0),
    rotate: int = Query(0),
    max_pages: int = Query(0, ge=0),
    project_root: str = Query(..., description="Absolute path to the project root directory"),
    file_id: str = Query(..., description="Unique file identifier"),
    force: bool = Query(False),
):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    root = Path(project_root)
    if not root.is_dir():
        raise HTTPException(status_code=400, detail=f"Project root not found: {project_root}")

    pp = ProjectPaths(
        root=root,
        manifest=root / "manifest.json",
        audit=root / "audit.jsonl",
        files_manifest=root / "files.json",
        files_dir=root / "files",
        text_dir=root / "text",
        previews_dir=root / "previews",
        standards_dir=root / "standards",
        structures_dir=root / "structures",
        runs_dir=root / "runs",
        queries_dir=root / "queries",
        exports_dir=root / "exports",
    )

    stored_path = pp.files_dir / file_id / (file.filename or "upload.pdf")
    ensure_dir(stored_path.parent)
    stored_path.write_bytes(await file.read())

    file_record = {"file_id": file_id, "stored_path": str(stored_path)}

    imgs, err = build_pdf_previews(pp, file_record, zoom=zoom, rotate=rotate, max_pages=max_pages, force=force)
    if err and not imgs:
        raise HTTPException(status_code=422, detail=err)

    return JSONResponse({"pages": len(imgs), "preview_paths": [str(p) for p in imgs]})
