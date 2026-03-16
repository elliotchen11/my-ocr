# app_streamlit_llm_extract.py
# Run: streamlit run app_streamlit_llm_extract.py
#
# Local-only workflow (Option A):
# - PDFs: render to images via PyMuPDF (convert_to_img.py) -> OCR via Ollama vision (ocr.py) -> merged text layer
# - Images: OCR via Ollama vision (ocr.py)
# - TXT: used directly
# - Review tab: shows stored PDF page images (previews) alongside extracted text + fields for validation

from __future__ import annotations

import json
import os
import re
import uuid
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from llm_extract_core import answer_questions_json, answer_questions_json_chunked
from azure_authentication import authenticate_streamlit_user, logout

# --- User-provided local OCR module ---
# Expecting: ocr.py with ocr_image(model, image_path, prompt) + DEFAULT_PROMPT
try:
    from ocr import ocr_image, DEFAULT_PROMPT  # type: ignore
except Exception:
    ocr_image = None  # type: ignore
    DEFAULT_PROMPT = "Extract all readable text from this image. Return plain text only."

# --- User-provided PDF->images module ---
# Expecting: convert_to_img.py with convert_pdf2img(...)
try:
    from convert_to_img import convert_pdf2img  # type: ignore
except Exception:
    convert_pdf2img = None  # type: ignore


# ---------------- Paths ----------------
DATA_ROOT = Path("./data")
PROJECTS_ROOT = DATA_ROOT / "streamlit_projects"


# ---------------- Utilities ----------------
def now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def safe_slug(name: str) -> str:
    name = (name or "").strip().lower()
    name = re.sub(r"[^a-z0-9]+", "-", name)
    name = re.sub(r"-{2,}", "-", name).strip("-")
    return name or "project"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def append_jsonl(path: Path, record: dict) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def short_id() -> str:
    return uuid.uuid4().hex[:10]


def estimate_tokens(text: str) -> int:
    text = text or ""
    try:
        import tiktoken  # type: ignore

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, int(len(text) / 4))


def parse_ctx_from_show(show_text: str) -> int | None:
    t = (show_text or "").lower()
    hits = [int(x) for x in re.findall(r"\b(2048|4096|8192|16384|32768|65536|131072)\b", t)]
    return max(hits) if hits else None


def get_text_path(file_record: dict) -> Path | None:
    """
    Return a valid Path to the extracted text layer if it exists and is a file.
    Prevents Path("") -> "." causing IsADirectoryError.
    """
    raw = (file_record or {}).get("text_path")
    if not raw:
        return None
    raw = str(raw).strip()
    if raw in ("", ".", "./"):
        return None
    try:
        p = Path(raw)
        return p if p.is_file() else None
    except Exception:
        return None


def is_pdf(file_record: dict) -> bool:
    p = str((file_record or {}).get("stored_path") or "")
    return p.lower().endswith(".pdf")


def is_image(file_record: dict) -> bool:
    p = str((file_record or {}).get("stored_path") or "").lower()
    return any(p.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"])


# ---------------- Project storage ----------------
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


def project_paths(slug: str) -> ProjectPaths:
    root = PROJECTS_ROOT / slug
    return ProjectPaths(
        root=root,
        manifest=root / "project.json",
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


def ensure_project_layout(pp: ProjectPaths) -> None:
    ensure_dir(pp.root)
    ensure_dir(pp.files_dir)
    ensure_dir(pp.text_dir)
    ensure_dir(pp.previews_dir)
    ensure_dir(pp.standards_dir)
    ensure_dir(pp.structures_dir)
    ensure_dir(pp.runs_dir)
    ensure_dir(pp.queries_dir)
    ensure_dir(pp.exports_dir)


def list_projects() -> list[tuple[str, str]]:
    ensure_dir(PROJECTS_ROOT)
    out: list[tuple[str, str]] = []
    for p in PROJECTS_ROOT.iterdir():
        if not p.is_dir():
            continue
        slug = p.name
        man = read_json(p / "project.json", {})
        display = man.get("name") or slug
        out.append((slug, str(display)))
    out.sort(key=lambda x: x[1].lower())
    return out


def create_project(name: str) -> str:
    slug0 = safe_slug(name)
    slug = slug0
    i = 2
    while (PROJECTS_ROOT / slug).exists():
        slug = f"{slug0}-{i}"
        i += 1

    pp = project_paths(slug)
    ensure_project_layout(pp)

    manifest = {
        "name": name.strip() or slug,
        "slug": slug,
        "created_at": now_iso(),
        "picklists": {"location": ["(add...)"], "name": ["(add...)"]},
    }
    write_json(pp.manifest, manifest)
    write_json(pp.files_manifest, {"files": []})
    append_jsonl(pp.audit, {"ts": now_iso(), "action": "project.create", "project": slug, "name": manifest["name"]})
    return slug


def delete_project(slug: str) -> tuple[bool, str]:
    """Delete a project folder (dangerous). Returns (ok, message)."""
    slug = (slug or "").strip()
    if not slug:
        return False, "No project selected."
    root = (PROJECTS_ROOT / slug).resolve()
    base = PROJECTS_ROOT.resolve()
    if root == base or base not in root.parents:
        return False, "Refusing to delete outside project root."
    if not root.exists() or not root.is_dir():
        return False, "Project folder not found."
    try:
        shutil.rmtree(root)
        return True, f"Deleted project: {slug}"
    except Exception as e:
        return False, f"Delete failed: {e}"


def load_project_manifest(pp: ProjectPaths) -> dict:
    m = read_json(pp.manifest, {})
    if not m:
        m = {
            "name": pp.root.name,
            "slug": pp.root.name,
            "created_at": now_iso(),
            "picklists": {"location": ["(add...)"], "name": ["(add...)"]},
        }
        write_json(pp.manifest, m)
    return m


def save_project_manifest(pp: ProjectPaths, manifest: dict) -> None:
    write_json(pp.manifest, manifest)
    append_jsonl(pp.audit, {"ts": now_iso(), "action": "project.update_manifest", "project": pp.root.name})


def load_files_manifest(pp: ProjectPaths) -> dict:
    m = read_json(pp.files_manifest, {"files": []})
    if "files" not in m or not isinstance(m["files"], list):
        m = {"files": []}
    return m


def save_files_manifest(pp: ProjectPaths, files_manifest: dict) -> None:
    write_json(pp.files_manifest, files_manifest)


def update_file_record_in_manifest(pp: ProjectPaths, file_id: str, updates: dict) -> None:
    """Update a file record in files.json (in-place) and persist."""
    try:
        mf = load_files_manifest(pp)
        files = mf.get('files', [])
        for rec in files:
            if str(rec.get('file_id')) == str(file_id):
                rec.update(updates or {})
                break
        mf['files'] = files
        save_files_manifest(pp, mf)
    except Exception:
        return



def _preview_dir_for_file(pp: ProjectPaths, file_id: str) -> Path:
    return pp.previews_dir / file_id


def list_preview_images(pp: ProjectPaths, file_record: dict) -> list[Path]:
    """
    Returns existing preview images (PNG) for a PDF, sorted by page.
    """
    file_id = str(file_record.get("file_id") or "").strip()
    if not file_id:
        return []
    d = _preview_dir_for_file(pp, file_id)
    if not d.is_dir():
        return []
    imgs = sorted(d.glob("*.png"))
    # Prefer _pageNNN naming from convert_to_img.py; still works for any .png
    def page_key(p: Path) -> tuple[int, str]:
        m = re.search(r"(?i)_page(\d+)\b", p.stem)
        n = int(m.group(1)) if m else 0
        return (n, p.name)
    return sorted(imgs, key=page_key)


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
    if convert_pdf2img is None:
        return [], "convert_to_img.py is not importable (convert_pdf2img missing)."

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


def ocr_images_with_ollama(ocr_model: str, images: list[Path], prompt: str) -> str | None:
    """
    OCR each image with user-provided ocr.py (ocr_image) and merge into one text blob.
    """
    if ocr_image is None:
        return None
    if not images:
        return None

    parts: list[str] = []
    for img in images:
        m = re.search(r"(?i)_page(\d+)\b", img.stem)
        page = int(m.group(1)) if m else 0
        txt = (ocr_image(model=ocr_model, image_path=img, prompt=prompt) or "").strip()  # type: ignore

        if page > 0:
            parts.append(f"\n\n===== PAGE {page} ({img.name}) =====\n{txt}\n")
        else:
            parts.append(f"\n\n===== IMAGE ({img.name}) =====\n{txt}\n")

    merged = "".join(parts).strip()
    return merged if merged else None


def extract_pdf_text_pymupdf(pdf_path: Path) -> str | None:
    """
    Optional fast path: extract embedded text from a text-native PDF using PyMuPDF.
    Returns None if PyMuPDF isn't available or PDF yields no useful text.
    """
    try:
        import fitz  # type: ignore
    except Exception:
        return None

    if not pdf_path.is_file():
        return None

    try:
        doc = fitz.open(str(pdf_path))
        try:
            pieces: list[str] = []
            for i in range(doc.page_count):
                page = doc.load_page(i)
                t = (page.get_text("text") or "").strip()
                if t:
                    pieces.append(t)
            out = "\n\n".join(pieces).strip()
            # heuristic: if it's too short, treat as scanned
            return out if len(out) >= 300 else None
        finally:
            doc.close()
    except Exception:
        return None


def extract_text_local_only(
    pp: ProjectPaths,
    file_record: dict,
    *,
    ocr_model: str,
    ocr_prompt: str,
    pdf_zoom: float,
    pdf_rotate: int,
    pdf_max_pages: int,
    build_previews: bool,
) -> tuple[str | None, str, str | None]:
    """
    Returns: (text_or_none, method, error)
    Methods: txt | pymupdf_text | ollama_ocr_pdf | ollama_ocr_img | unsupported
    """
    stored_path = Path(str(file_record.get("stored_path") or ""))
    if not stored_path.is_file():
        return None, "unsupported", "Stored file path missing or not a file."

    suf = stored_path.suffix.lower()

    if suf == ".txt":
        try:
            return stored_path.read_text(encoding="utf-8", errors="replace"), "txt", None
        except Exception as e:
            return None, "txt", f"Failed to read txt: {e}"

    if suf == ".pdf":
        # 1) Fast embedded text (optional, local)
        t = extract_pdf_text_pymupdf(stored_path)
        if t:
            # Still build previews if requested (for validation)
            if build_previews:
                _imgs, _err = build_pdf_previews(
                    pp, file_record, zoom=pdf_zoom, rotate=pdf_rotate, max_pages=pdf_max_pages, force=False
                )
                # preview failure shouldn't fail extraction
                if _imgs:
                    file_record["preview_images_count"] = len(_imgs)
                    file_record["preview_dir"] = str(_preview_dir_for_file(pp, str(file_record.get("file_id") or "")))
            return t, "pymupdf_text", None

        # 2) OCR via Ollama vision (requires previews/images)
        if ocr_image is None:
            return None, "ollama_ocr_pdf", "ocr.py is not importable (ocr_image missing)."

        # Build previews (or temp images) to OCR
        imgs: list[Path] = []
        if build_previews:
            imgs, err = build_pdf_previews(
                pp, file_record, zoom=pdf_zoom, rotate=pdf_rotate, max_pages=pdf_max_pages, force=False
            )
            if err and not imgs:
                return None, "ollama_ocr_pdf", err
        else:
            # If not building previews, we still need images to OCR (render into previews dir anyway, but delete later is messy).
            # For simplicity, use previews dir regardless; it keeps the workflow consistent.
            imgs, err = build_pdf_previews(
                pp, file_record, zoom=pdf_zoom, rotate=pdf_rotate, max_pages=pdf_max_pages, force=False
            )
            if err and not imgs:
                return None, "ollama_ocr_pdf", err

        if imgs:
            file_record["preview_images_count"] = len(imgs)
            file_record["preview_dir"] = str(_preview_dir_for_file(pp, str(file_record.get("file_id") or "")))

        t = ocr_images_with_ollama(ocr_model, imgs, ocr_prompt)
        return (t, "ollama_ocr_pdf", None) if t else (None, "ollama_ocr_pdf", "OCR produced no text.")

    if suf in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"]:
        if ocr_image is None:
            return None, "ollama_ocr_img", "ocr.py is not importable (ocr_image missing)."
        t = ocr_images_with_ollama(ocr_model, [stored_path], ocr_prompt)
        return (t, "ollama_ocr_img", None) if t else (None, "ollama_ocr_img", "OCR produced no text.")

    return None, "unsupported", f"Unsupported file type: {suf}"


def ensure_text_layer_for_file(
    pp: ProjectPaths,
    file_record: dict,
    *,
    ocr_model: str,
    ocr_prompt: str,
    pdf_zoom: float,
    pdf_rotate: int,
    pdf_max_pages: int,
    build_previews: bool,
) -> dict:
    """
    Ensure file_record has a text layer (text/<file_id>.txt). Updates file_record in-place-ish and returns it.
    """
    file_id = str(file_record.get("file_id") or "").strip()
    if not file_id:
        file_record["status"] = "failed"
        file_record["error"] = "Missing file_id."
        return file_record

    ensure_project_layout(pp)
    out_text_path = pp.text_dir / f"{file_id}.txt"

    text, method, err = extract_text_local_only(
        pp,
        file_record,
        ocr_model=ocr_model,
        ocr_prompt=ocr_prompt,
        pdf_zoom=pdf_zoom,
        pdf_rotate=pdf_rotate,
        pdf_max_pages=pdf_max_pages,
        build_previews=build_previews,
    )

    if text and text.strip():
        out_text_path.write_text(text, encoding="utf-8")
        file_record["text_path"] = str(out_text_path)
        file_record["status"] = "ready"
        file_record["error"] = None
        file_record["text_method"] = method
        file_record["processed_at"] = now_iso()
        append_jsonl(
            pp.audit,
            {"ts": now_iso(), "action": "file.text_layer_built", "project": pp.root.name, "file_id": file_id, "method": method},
        )
    else:
        file_record["status"] = "failed"
        file_record["error"] = err or "Text layer build failed."
        file_record["text_method"] = method
        file_record["processed_at"] = now_iso()
        append_jsonl(
            pp.audit,
            {
                "ts": now_iso(),
                "action": "file.text_layer_failed",
                "project": pp.root.name,
                "file_id": file_id,
                "method": method,
                "error": file_record["error"],
            },
        )

    return file_record


# ---------------- Standards / structures / runs ----------------
def list_standard_versions(pp: ProjectPaths) -> list[Path]:
    """Return standard version files, newest first (vN descending)."""
    ensure_project_layout(pp)
    files = [p for p in pp.standards_dir.glob("standard_v*.json") if p.is_file()]

    def _ver(path: Path) -> int:
        m = re.search(r"standard_v(\d+)\.json$", path.name)
        return int(m.group(1)) if m else 0

    return sorted(files, key=_ver, reverse=True)


def next_standard_version(pp: ProjectPaths) -> int:
    vs = []
    for p in list_standard_versions(pp):
        m = re.search(r"standard_v(\d+)\.json$", p.name)
        if m:
            vs.append(int(m.group(1)))
    return (max(vs) + 1) if vs else 1


def load_standard(pp: ProjectPaths, path: Path) -> dict:
    return read_json(path, {})


def save_standard(pp: ProjectPaths, standard: dict) -> Path:
    v = next_standard_version(pp)
    path = pp.standards_dir / f"standard_v{v}.json"
    standard = {**standard, "version": v, "saved_at": now_iso()}
    write_json(path, standard)
    append_jsonl(pp.audit, {"ts": now_iso(), "action": "standard.save", "project": pp.root.name, "version": v})
    return path


def list_structure_versions(pp: ProjectPaths) -> list[Path]:
    """Return structure version files, newest first (vN descending)."""
    ensure_project_layout(pp)
    files = [p for p in pp.structures_dir.glob("structure_v*.txt") if p.is_file()]

    def _ver(path: Path) -> int:
        m = re.search(r"structure_v(\d+)\.txt$", path.name)
        return int(m.group(1)) if m else 0

    return sorted(files, key=_ver, reverse=True)


def next_structure_version(pp: ProjectPaths) -> int:
    vs = []
    for p in list_structure_versions(pp):
        m = re.search(r"structure_v(\d+)\.txt$", p.name)
        if m:
            vs.append(int(m.group(1)))
    return (max(vs) + 1) if vs else 1


def save_structure(pp: ProjectPaths, structure_text: str) -> Path:
    v = next_structure_version(pp)
    path = pp.structures_dir / f"structure_v{v}.txt"
    ensure_dir(path.parent)
    path.write_text(structure_text or "", encoding="utf-8")
    append_jsonl(pp.audit, {"ts": now_iso(), "action": "structure.save", "project": pp.root.name, "version": v})
    return path


def list_runs(pp: ProjectPaths) -> list[Path]:
    ensure_project_layout(pp)
    return sorted(pp.runs_dir.glob("run_*.json"), reverse=True)


def count_runs_using_standard(pp: ProjectPaths, standard_filename: str) -> int:
    """How many saved runs reference this standard file name."""
    if not standard_filename:
        return 0
    cnt = 0
    for rp in list_runs(pp):
        r = read_json(rp, {})
        if isinstance(r, dict) and r.get("standard_file") == standard_filename:
            cnt += 1
    return cnt


def count_runs_using_structure(pp: ProjectPaths, structure_filename: str) -> int:
    """How many saved runs reference this structure file name."""
    if not structure_filename:
        return 0
    cnt = 0
    for rp in list_runs(pp):
        r = read_json(rp, {})
        if isinstance(r, dict) and r.get("structure_file") == structure_filename:
            cnt += 1
    return cnt


def save_run(pp: ProjectPaths, run_record: dict) -> Path:
    run_id = run_record.get("run_id") or f"{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_{short_id()}"
    run_record["run_id"] = run_id
    path = pp.runs_dir / f"run_{run_id}.json"
    write_json(path, run_record)
    append_jsonl(pp.audit, {"ts": now_iso(), "action": "run.save", "project": pp.root.name, "run_id": run_id})
    return path


def load_run(path: Path) -> dict:
    return read_json(path, {})


def save_query(pp: ProjectPaths, query_name: str, payload: dict) -> Path:
    qid = f"{safe_slug(query_name)}_{short_id()}"
    path = pp.queries_dir / f"query_{qid}.json"
    payload = {**payload, "query_id": qid, "name": query_name.strip() or qid, "saved_at": now_iso()}
    write_json(path, payload)
    append_jsonl(pp.audit, {"ts": now_iso(), "action": "query.save", "project": pp.root.name, "query_id": qid})
    return path


def list_queries(pp: ProjectPaths) -> list[Path]:
    ensure_project_layout(pp)
    return sorted(pp.queries_dir.glob("query_*.json"))


# ---------------- Extraction helpers ----------------
def build_context_note(base: str, standard: dict | None, structure_text: str | None) -> str:
    parts = []
    if base.strip():
        parts.append(base.strip())

    if structure_text and structure_text.strip():
        parts.append("File-type structure / layout notes:\n" + structure_text.strip())

    if standard and isinstance(standard.get("fields"), list):
        lines = ["Data standard (fields + definitions):"]
        for f in standard["fields"]:
            fn = str(f.get("field") or "").strip()
            if not fn:
                continue
            desc = str(f.get("definition") or "").strip()
            req = bool(f.get("required", False))
            fmt = str(f.get("format") or "").strip()
            line = f"- {fn}: {desc}"
            if req:
                line += " (REQUIRED)"
            if fmt:
                line += f" [format: {fmt}]"
            lines.append(line)
        parts.append("\n".join(lines))

    return "\n\n".join(parts).strip()


def normalize_answers(result: Any, questions: list[str]) -> dict[str, Any]:
    """
    Handles a few possible shapes:
    - {"answers": {"0": "...", "1": "..."}}
    - {"answers": {"question text": "..."}}
    - {"0": "..."} or {"question text": "..."}
    - llm_extract_core.py returns {question -> answer}
    """
    if isinstance(result, dict):
        ans = result.get("answers") if isinstance(result.get("answers"), dict) else result
        if not isinstance(ans, dict):
            return {}
        out: dict[str, Any] = {}
        for i, q in enumerate(questions):
            if q in ans:
                out[q] = ans.get(q)
                continue
            k = str(i)
            if k in ans:
                out[q] = ans.get(k)
                continue
            out[q] = None
        return out
    return {q: None for q in questions}


def compute_validation_issues(fields: list[dict], answers_by_field: dict[str, Any]) -> dict:
    issues = {"missing_required": [], "format_errors": []}
    for f in fields:
        name = str(f.get("field") or "").strip()
        if not name:
            continue
        val = answers_by_field.get(name)
        sval = "" if val is None else str(val).strip()

        if bool(f.get("required", False)) and not sval:
            issues["missing_required"].append(name)

        fmt = str(f.get("format") or "").strip()
        if fmt and sval:
            if any(tok in fmt for tok in ["^", "$", "\\d", "\\w", "(", ")", "[", "]", "{", "}", "|", "+", "*", "?"]):
                try:
                    if re.search(fmt, sval) is None:
                        issues["format_errors"].append(name)
                except re.error:
                    pass
    return issues


def completeness_ratio(fields: list[dict], answers_by_field: dict[str, Any]) -> float:
    names = [str(f.get("field") or "").strip() for f in fields if str(f.get("field") or "").strip()]
    if not names:
        return 0.0
    filled = 0
    for n in names:
        v = answers_by_field.get(n)
        if v is None:
            continue
        if str(v).strip():
            filled += 1
    return filled / len(names)


# ---------------- Ingest ----------------
def add_file_record(pp: ProjectPaths, filename: str, meta: dict, raw_bytes: bytes, ext: str) -> dict:
    file_id = short_id()
    stored_name = f"{file_id}_{safe_slug(Path(filename).stem)}{ext}"
    stored_path = pp.files_dir / stored_name

    ensure_project_layout(pp)
    stored_path.write_bytes(raw_bytes)

    # If it's a .txt, treat as extracted text layer immediately.
    text_path = pp.text_dir / f"{file_id}.txt"
    status = "uploaded"
    err = None

    if ext.lower() == ".txt":
        try:
            text_path.write_text(raw_bytes.decode("utf-8", errors="replace"), encoding="utf-8")
            status = "ready"
        except Exception as e:
            status = "failed"
            err = f"Failed to decode txt: {e}"

    record = {
        "file_id": file_id,
        "filename": filename,
        "stored_path": str(stored_path),
        "text_path": str(text_path) if text_path.is_file() else None,
        "uploaded_at": now_iso(),
        "status": status,
        "error": err,
        "metadata": meta,
        "text_method": None,
        "preview_dir": None,
        "preview_images_count": 0,
    }

    mf = load_files_manifest(pp)
    mf["files"].append(record)
    save_files_manifest(pp, mf)

    append_jsonl(pp.audit, {"ts": now_iso(), "action": "file.add", "project": pp.root.name, "file_id": file_id, "filename": filename})
    return record


# ---------------- UI ----------------
ensure_dir(DATA_ROOT)
ensure_dir(PROJECTS_ROOT)

st.set_page_config(page_title="Project OCR/Text → LLM Extract → Review", layout="wide")

# ---------------- Authentication ----------------
azure_user = authenticate_streamlit_user()
if azure_user is None:
    st.stop()

st.title("Project OCR/Text → LLM Extract → Review")

# Sidebar — show signed-in user and logout button
st.sidebar.markdown(
    f"**Signed in as:** {azure_user.get('name', azure_user.get('preferred_username', 'Unknown'))}"
)
if st.sidebar.button("Sign out"):
    logout()
    st.rerun()

# Sidebar navigation
page = st.sidebar.radio(
    "Navigate",
    ["Projects", "Ingest", "Data Standard", "Structure", "Run Extraction", "Review", "Export"],
    index=0,
)

# ---- Project selection (sticky) ----
st.sidebar.markdown("---")
projects = list_projects()
proj_labels = ["(none)"] + [f"{name}  —  {slug}" for slug, name in projects]
label_to_slug = {f"{name}  —  {slug}": slug for slug, name in projects}

if "active_project_slug" not in st.session_state:
    st.session_state.active_project_slug = projects[0][0] if projects else None

selected_label = st.sidebar.selectbox(
    "Active project",
    options=proj_labels,
    index=(proj_labels.index(next((lab for lab in proj_labels if st.session_state.active_project_slug and lab.endswith(st.session_state.active_project_slug)), "(none)"))),
)

active_slug = label_to_slug.get(selected_label)
if active_slug:
    st.session_state.active_project_slug = active_slug

active_slug = st.session_state.active_project_slug
pp: ProjectPaths | None = project_paths(active_slug) if active_slug else None
manifest = load_project_manifest(pp) if pp else None

# ---- Ollama model picker (global) ----
st.sidebar.markdown("---")
st.sidebar.subheader("Models")

@st.cache_data(ttl=10)
def list_ollama_models() -> list[str]:
    try:
        import subprocess
        out = subprocess.check_output(["ollama", "list"], text=True, stderr=subprocess.STDOUT)
    except Exception:
        return []
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    if not lines:
        return []
    if "NAME" in lines[0].upper():
        lines = lines[1:]
    models = []
    for ln in lines:
        name = ln.split()[0]
        if name:
            models.append(name)
    seen, uniq = set(), []
    for m in models:
        if m not in seen:
            seen.add(m)
            uniq.append(m)
    return uniq

@st.cache_data(ttl=60)
def ollama_show_raw(model: str) -> str:
    try:
        import subprocess
        out = subprocess.check_output(["ollama", "show", model], text=True, stderr=subprocess.STDOUT)
        return out.strip()
    except Exception as e:
        return f"Failed to run: ollama show {model}\n{e}"

def infer_tags_from_show(model: str, show_text: str) -> list[str]:
    n = (model or "").lower()
    t = (show_text or "").lower()
    tags: list[str] = []
    if "vision" in n or "vision" in t or "vl" in n:
        tags.append("vision")
    if "embed" in n or "embedding" in t:
        tags.append("embeddings")
    if "code" in n or "coder" in n or "code" in t:
        tags.append("code")
    if "instruct" in n or "instruct" in t:
        tags.append("instruct")
    if "json" in t or "structured" in t or "function" in t:
        tags.append("structured/json")
    m = re.search(r"[:\-]([0-9]+)b\b", n)
    if m:
        b = int(m.group(1))
        tags.append("fast" if b <= 4 else ("balanced" if b <= 9 else "quality"))
    ctx_nums = [int(x) for x in re.findall(r"\b(8192|16384|32768|65536|131072)\b", t)]
    if ctx_nums and max(ctx_nums) >= 32768:
        tags.append("long-doc")
    return tags or ["general"]

@st.cache_data(ttl=60)
def model_meta(model: str) -> dict:
    show_text = ollama_show_raw(model)
    tags = infer_tags_from_show(model, show_text)
    return {"show": show_text, "tags": tags}

models = list_ollama_models()
annotate = st.sidebar.toggle("Annotate model dropdown", value=True)
if st.sidebar.button("Refresh models"):
    st.cache_data.clear()
    models = list_ollama_models()

if not models:
    st.sidebar.warning("No Ollama models detected (ollama list failed).")
    model = st.sidebar.text_input("LLM model (manual)", value="gemma3")
    meta = {"tags": ["unknown"], "show": ""}
else:
    default_model = "gemma3" if "gemma3" in models else models[0]
    if annotate:
        def _fmt(m: str) -> str:
            mm = model_meta(m)
            return f"{m}  —  {', '.join(mm['tags'])}"
        model = st.sidebar.selectbox("LLM model", models, index=models.index(default_model), format_func=_fmt)
    else:
        model = st.sidebar.selectbox("LLM model", models, index=models.index(default_model))
    meta = model_meta(model)
    st.sidebar.caption(f"Tags: **{', '.join(meta['tags'])}**")
    with st.sidebar.expander("ollama show"):
        st.sidebar.code(meta["show"], language="text")

# ---- OCR controls (local only) ----
st.sidebar.markdown("---")
st.sidebar.subheader("OCR (Local)")

if ocr_image is None:
    st.sidebar.error("ocr.py not found / not importable (ocr_image missing).")
if convert_pdf2img is None:
    st.sidebar.error("convert_to_img.py not found / not importable (convert_pdf2img missing).")

ocr_model = st.sidebar.selectbox(
    "OCR model (Ollama vision)",
    options=models if models else [model],
    index=(models.index(model) if models and model in models else 0),
)

ocr_prompt = st.sidebar.text_area("OCR prompt", value=DEFAULT_PROMPT, height=80)

pdf_zoom = st.sidebar.slider("PDF render zoom", min_value=1.0, max_value=4.0, value=2.0, step=0.25)
pdf_rotate = st.sidebar.selectbox("PDF rotate", options=[0, 90, 180, 270], index=0)
pdf_max_pages = st.sidebar.number_input("Max PDF pages to render/store", min_value=1, max_value=500, value=60, step=10)

# ---- Common extraction controls (global) ----
st.sidebar.markdown("---")
st.sidebar.subheader("Extraction controls")
force_chunking = st.sidebar.toggle("Force chunking mode", value=False)
TOKEN_THRESHOLD = st.sidebar.number_input(
    "Auto-switch to chunking above (estimated tokens)",
    min_value=5_000,
    max_value=120_000,
    value=25_000,
    step=5_000,
)
default_context_note = st.sidebar.text_area(
    "Context note (default)",
    value="Extract only from evidence in the document text. If not present, return null.",
    height=90,
)

# ---------------- Pages ----------------
if page == "Projects":
    st.subheader("Projects")

    c1, c2 = st.columns([1, 1])
    with c1:
        new_name = st.text_input("Create a new project", placeholder="e.g., FIA Batch Feb 2026")
    with c2:
        if st.button("Create project", type="primary", disabled=not bool(new_name.strip())):
            slug = create_project(new_name.strip())
            st.session_state.active_project_slug = slug
            st.success(f"Created project: {new_name.strip()} ({slug})")
            st.rerun()

    st.markdown("---")
    if not projects:
        st.info("No projects yet. Create one above.")
    else:
        rows = []
        for slug, name in projects:
            man = read_json(project_paths(slug).manifest, {})
            rows.append({"Project": name, "Slug": slug, "Created": man.get("created_at")})
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        st.markdown("---")
        st.subheader("Delete a project")
        st.warning("This permanently deletes the entire project folder (files, text layers, standards, structures, runs, exports).")
        del_slugs = [s for s, _ in projects]
        if del_slugs:
            name_by_slug = {s: n for s, n in projects}
            del_slug = st.selectbox(
                "Project to delete",
                options=del_slugs,
                index=del_slugs.index(active_slug) if active_slug in del_slugs else 0,
                format_func=lambda s: f"{name_by_slug.get(s, s)}  —  {s}",
            )
            confirm = st.text_input("Type the project slug to confirm", value="", key="confirm_delete_project")
            if st.button("Delete project permanently", type="primary", disabled=(confirm.strip() != (del_slug or ""))):
                ok, msg = delete_project(del_slug)
                if ok:
                    # If we just deleted the active project, switch to another (or none)
                    if st.session_state.get("active_project_slug") == del_slug:
                        remaining = list_projects()
                        st.session_state.active_project_slug = remaining[0][0] if remaining else None
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
        else:
            st.info("No projects available to delete.")

    if active_slug and pp and manifest:
        st.markdown("---")
        st.subheader("Project picklists (metadata)")
        pick = manifest.get("picklists", {})
        locs = pick.get("location", [])
        names = pick.get("name", [])

        colA, colB = st.columns(2)
        with colA:
            st.write("Location picklist")
            locs_edit = st.text_area("One per line", value="\n".join(locs) if locs else "", height=140, key="locs_edit")
        with colB:
            st.write("Name picklist")
            names_edit = st.text_area("One per line", value="\n".join(names) if names else "", height=140, key="names_edit")

        if st.button("Save picklists"):
            manifest["picklists"] = {
                "location": [x.strip() for x in locs_edit.splitlines() if x.strip()],
                "name": [x.strip() for x in names_edit.splitlines() if x.strip()],
            }
            save_project_manifest(pp, manifest)
            st.success("Saved picklists.")
            st.rerun()

elif page == "Ingest":
    st.subheader("Ingest (Upload files + metadata)")

    if not pp or not active_slug:
        st.warning("Select or create a project first (Projects page).")
    else:
        ensure_project_layout(pp)
        mf = load_files_manifest(pp)

        st.markdown("### Existing files")
        files = mf.get("files", [])
        if not files:
            st.info("No files uploaded yet.")
        else:
            df = pd.DataFrame(
                [
                    {
                        "file_id": f.get("file_id"),
                        "filename": f.get("filename"),
                        "type": Path(str(f.get("stored_path") or "")).suffix.lower(),
                        "status": f.get("status"),
                        "text_method": f.get("text_method"),
                        "preview_images": f.get("preview_images_count", 0),
                        "uploaded_at": f.get("uploaded_at"),
                        "location": (f.get("metadata") or {}).get("location"),
                        "name": (f.get("metadata") or {}).get("name"),
                        "date": (f.get("metadata") or {}).get("date"),
                    }
                    for f in files
                ]
            )
            st.dataframe(df, width="stretch", hide_index=True)

        st.markdown("---")
        st.markdown("### Upload new files")
        pick = (manifest or {}).get("picklists", {})
        loc_opts = pick.get("location") or []
        name_opts = pick.get("name") or []

        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            meta_location = st.selectbox("Location", options=loc_opts if loc_opts else ["(none)"])
        with c2:
            meta_name = st.selectbox("Name", options=name_opts if name_opts else ["(none)"])
        with c3:
            meta_date = st.date_input("Date")

        uploads = st.file_uploader(
            "Upload documents (.txt, .pdf, images)",
            type=["txt", "pdf", "png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"],
            accept_multiple_files=True,
        )

        if st.button("Add to project", type="primary", disabled=not uploads):
            added = 0
            for up in uploads or []:
                ext = "." + up.name.split(".")[-1].lower() if "." in up.name else ""
                meta = {"location": meta_location, "name": meta_name, "date": str(meta_date)}
                add_file_record(pp, up.name, meta, up.getvalue(), ext)
                added += 1
            st.success(f"Added {added} file(s) to project.")
            st.rerun()

elif page == "Data Standard":
    st.subheader("Data Standard (fields + definitions + rules)")

    if not pp or not active_slug:
        st.warning("Select or create a project first.")
    else:
        ensure_project_layout(pp)

        versions = list_standard_versions(pp)
        latest = versions[0] if versions else None
        latest_std = load_standard(pp, latest) if latest else {"fields": []}

        st.markdown("### Edit standard (then save as a new version)")
        seed_rows = latest_std.get("fields") if isinstance(latest_std.get("fields"), list) else []
        if not seed_rows:
            seed_rows = [
                {"field": "agreement_id", "definition": "Unique identifier of the agreement", "required": True, "format": "", "allowed_values": ""},
                {"field": "agreement_date", "definition": "Date of the agreement", "required": False, "format": "", "allowed_values": ""},
            ]

        df = pd.DataFrame(seed_rows)
        for col in ["field", "definition", "format", "allowed_values"]:
            if col not in df.columns:
                df[col] = ""
        if "required" not in df.columns:
            df["required"] = False

        edited = st.data_editor(
            df,
            width="stretch",
            num_rows="dynamic",
            column_config={
                "field": st.column_config.TextColumn(required=True),
                "definition": st.column_config.TextColumn(width="large"),
                "required": st.column_config.CheckboxColumn(),
                "format": st.column_config.TextColumn(help="Optional. Regex encouraged for MVP."),
                "allowed_values": st.column_config.TextColumn(help="Optional. Comma-separated."),
            },
        )

        if st.button("Save as new standard version", type="primary"):
            fields = []
            for _, row in edited.iterrows():
                fn = str(row.get("field") or "").strip()
                if not fn:
                    continue
                fields.append(
                    {
                        "field": fn,
                        "definition": str(row.get("definition") or "").strip(),
                        "required": bool(row.get("required", False)),
                        "format": str(row.get("format") or "").strip(),
                        "allowed_values": str(row.get("allowed_values") or "").strip(),
                    }
                )
            saved_path = save_standard(pp, {"fields": fields})
            st.success(f"Saved {saved_path.name}")
            st.rerun()

        st.markdown("---")
        st.markdown("### Existing versions")
        if not versions:
            st.info("No saved standards yet.")
        else:
            for p in versions[:10]:
                st.write(p.name)
        st.markdown("---")
        with st.expander("Delete standard versions"):
            st.warning("Deleting versions can break old runs that reference them. (Runs will keep working, but you won't be able to re-select the deleted version in the UI.)")
            if not versions:
                st.info("No standard versions to delete.")
            else:
                allow_latest = st.checkbox(
                    "Allow deleting the latest version (risky)",
                    value=False,
                    key="std_delete_allow_latest",
                )
                candidates = versions if allow_latest else versions[1:]
                if not candidates:
                    st.info("Only one version exists; nothing to delete.")
                else:
                    opts: list[str] = []
                    label_to_path: dict[str, Path] = {}

                    for p in candidates:
                        meta0 = read_json(p, {}) if p.exists() else {}
                        v0 = meta0.get("version")
                        if not v0:
                            m0 = re.search(r"standard_v(\d+)\.json$", p.name)
                            v0 = int(m0.group(1)) if m0 else "?"
                        saved0 = meta0.get("saved_at") or meta0.get("created_at") or ""
                        used0 = count_runs_using_standard(pp, p.name)
                        extra = []
                        if saved0:
                            extra.append(f"saved {saved0}")
                        extra.append(f"used by {used0} run(s)")
                        label = f"{p.name}  (v{v0}; " + ", ".join(extra) + ")"
                        opts.append(label)
                        label_to_path[label] = p

                    sel = st.multiselect("Select standard versions to delete", options=opts, default=[], key="std_delete_sel")
                    if st.button("Delete selected standard versions", type="secondary", disabled=not bool(sel), key="std_delete_btn"):
                        deleted: list[str] = []
                        errors: list[str] = []
                        for lab in sel:
                            fp = label_to_path.get(lab)
                            if not fp:
                                continue
                            try:
                                if fp.exists():
                                    fp.unlink()
                                    deleted.append(fp.name)
                            except Exception as e:
                                errors.append(f"{fp.name}: {e}")

                        if errors:
                            st.error("Some deletes failed:\n" + "\n".join(errors))
                        if deleted:
                            append_jsonl(pp.audit, {"ts": now_iso(), "action": "standard.delete", "project": pp.root.name, "files": deleted})
                            st.success(f"Deleted {len(deleted)} standard version(s).")
                            st.rerun()


elif page == "Structure":
    st.subheader("File-Type Structure (layout hints)")

    if not pp or not active_slug:
        st.warning("Select or create a project first.")
    else:
        ensure_project_layout(pp)

        versions = list_structure_versions(pp)
        latest = versions[0] if versions else None
        latest_text = latest.read_text(encoding="utf-8") if latest and latest.exists() else ""

        structure_text = st.text_area(
            "Describe common sections, headers, tables, patterns, etc.",
            value=latest_text,
            height=260,
        )

        if st.button("Save as new structure version", type="primary"):
            saved_path = save_structure(pp, structure_text)
            st.success(f"Saved {saved_path.name}")
            st.rerun()

        st.markdown("---")
        st.markdown("### Existing versions")
        if not versions:
            st.info("No saved structures yet.")
        else:
            for p in versions[:10]:
                st.write(p.name)
        st.markdown("---")
        with st.expander("Delete structure versions"):
            st.warning("Deleting versions can break old runs that reference them. (Runs will keep working, but you won't be able to re-select the deleted version in the UI.)")
            if not versions:
                st.info("No structure versions to delete.")
            else:
                allow_latest = st.checkbox(
                    "Allow deleting the latest version (risky)",
                    value=False,
                    key="struct_delete_allow_latest",
                )
                candidates = versions if allow_latest else versions[1:]
                if not candidates:
                    st.info("Only one version exists; nothing to delete.")
                else:
                    opts: list[str] = []
                    label_to_path: dict[str, Path] = {}

                    for p in candidates:
                        # structure files are plain text, so saved_at isn't embedded; infer version from filename
                        m0 = re.search(r"structure_v(\d+)\.txt$", p.name)
                        v0 = int(m0.group(1)) if m0 else "?"
                        used0 = count_runs_using_structure(pp, p.name)
                        label = f"{p.name}  (v{v0}; used by {used0} run(s))"
                        opts.append(label)
                        label_to_path[label] = p

                    sel = st.multiselect("Select structure versions to delete", options=opts, default=[], key="struct_delete_sel")
                    if st.button("Delete selected structure versions", type="secondary", disabled=not bool(sel), key="struct_delete_btn"):
                        deleted: list[str] = []
                        errors: list[str] = []
                        for lab in sel:
                            fp = label_to_path.get(lab)
                            if not fp:
                                continue
                            try:
                                if fp.exists():
                                    fp.unlink()
                                    deleted.append(fp.name)
                            except Exception as e:
                                errors.append(f"{fp.name}: {e}")

                        if errors:
                            st.error("Some deletes failed:\n" + "\n".join(errors))
                        if deleted:
                            append_jsonl(pp.audit, {"ts": now_iso(), "action": "structure.delete", "project": pp.root.name, "files": deleted})
                            st.success(f"Deleted {len(deleted)} structure version(s).")
                            st.rerun()


elif page == "Run Extraction":
    st.subheader("Run Extraction (creates a versioned Run)")

    if not pp or not active_slug:
        st.warning("Select or create a project first.")
    else:
        mf = load_files_manifest(pp)
        files = mf.get("files", [])
        if not files:
            st.info("Upload files first (Ingest page).")
        else:
            st.markdown("### Choose files")
            df = pd.DataFrame(
                [
                    {
                        "file_id": f.get("file_id"),
                        "filename": f.get("filename"),
                        "type": Path(str(f.get("stored_path") or "")).suffix.lower(),
                        "status": f.get("status"),
                        "text_method": f.get("text_method"),
                        "preview_images": f.get("preview_images_count", 0),
                    }
                    for f in files
                ]
            )
            st.dataframe(df, width="stretch", hide_index=True)

            ids = [f["file_id"] for f in files if f.get("file_id")]
            selected_ids = st.multiselect("Select file_ids to include in this run", options=ids, default=ids[: min(5, len(ids))])

            st.markdown("---")
            st.markdown("### Choose Standard + Structure versions")
            std_versions = list_standard_versions(pp)
            struct_versions = list_structure_versions(pp)

            std_choice = st.selectbox("Data Standard version", options=["(none)"] + [p.name for p in std_versions], index=1 if std_versions else 0)
            struct_choice = st.selectbox("Structure version", options=["(none)"] + [p.name for p in struct_versions], index=1 if struct_versions else 0)

            standard = load_standard(pp, pp.standards_dir / std_choice) if std_choice != "(none)" else None
            structure_text = (pp.structures_dir / struct_choice).read_text(encoding="utf-8") if struct_choice != "(none)" else None

            use_data_standard = st.toggle("Use Data Standard fields (recommended)", value=True)
            if use_data_standard and standard and isinstance(standard.get("fields"), list) and standard["fields"]:
                fields = standard["fields"]
                questions = [str(f.get("field") or "").strip() for f in fields if str(f.get("field") or "").strip()]
            else:
                st.caption("Fallback: custom questions (one per line).")
                questions_text = st.text_area("Questions", value="What is the agreement ID?\nWhat is the agreement date?", height=120)
                questions = [q.strip() for q in questions_text.splitlines() if q.strip()]
                fields = []

            context_note = st.text_area("Context note (override for this run)", value=default_context_note, height=90)
            ctx_full = build_context_note(context_note, standard if use_data_standard else None, structure_text)

            chunk_chars = 12000
            overlap_chars = 800
            with st.expander("Chunking settings"):
                chunk_chars = st.slider("Chunk size (chars)", 4000, 30000, 12000, step=1000)
                overlap_chars = st.slider("Overlap (chars)", 0, 5000, 800, step=100)

            st.markdown("---")
            auto_process_missing = st.toggle("Auto-build missing text layers (PDF/images) before extraction", value=True)
            build_previews = st.toggle("Store PDF page images for Review validation", value=True)
            force_rebuild_previews = st.toggle("Force rebuild PDF previews (selected files only)", value=False)

            run_btn = st.button("Run extraction now", type="primary", disabled=(not selected_ids or not questions))

            if run_btn:
                run_record = {
                    "run_id": None,
                    "created_at": now_iso(),
                    "model": model,
                    "context_note": ctx_full,
                    "standard_version": standard.get("version") if standard else None,
                    "structure_version": None,
                    "standard_file": std_choice if std_choice != "(none)" else None,
                    "structure_file": struct_choice if struct_choice != "(none)" else None,
                    "files": [],
                    "outputs": {},
                    "validation": {},
                    "audit": [],
                    "params": {
                        "force_chunking": force_chunking,
                        "token_threshold": int(TOKEN_THRESHOLD),
                        "chunk_chars": int(chunk_chars),
                        "overlap_chars": int(overlap_chars),
                        "ocr_model": ocr_model,
                        "pdf_zoom": float(pdf_zoom),
                        "pdf_rotate": int(pdf_rotate),
                        "pdf_max_pages": int(pdf_max_pages),
                        "build_previews": bool(build_previews),
                    },
                }

                
                if struct_choice != "(none)":
                    m = re.search(r"structure_v(\d+)\.txt$", struct_choice)
                    run_record["structure_version"] = int(m.group(1)) if m else None

                file_map = {f["file_id"]: f for f in files if f.get("file_id")}
                errors: list[str] = []

                with st.spinner("Running extraction…"):
                    for fid in selected_ids:
                        f = file_map.get(fid)
                        if not f:
                            continue

                        # Ensure previews if requested + forcing rebuild
                        if build_previews and force_rebuild_previews and is_pdf(f):
                            imgs, perr = build_pdf_previews(
                                pp, f, zoom=float(pdf_zoom), rotate=int(pdf_rotate), max_pages=int(pdf_max_pages), force=True
                            )
                            if imgs:
                                f["preview_dir"] = str(_preview_dir_for_file(pp, fid))
                                f["preview_images_count"] = len(imgs)
                            if perr:
                                # Don't fail run; just capture
                                errors.append(f"{fid}: {perr}")

                        # Auto-build missing text layers
                        if auto_process_missing and not get_text_path(f):
                            try:
                                ensure_text_layer_for_file(
                                    pp,
                                    f,
                                    ocr_model=ocr_model,
                                    ocr_prompt=ocr_prompt,
                                    pdf_zoom=float(pdf_zoom),
                                    pdf_rotate=int(pdf_rotate),
                                    pdf_max_pages=int(pdf_max_pages),
                                    build_previews=bool(build_previews),
                                )
                            except Exception as e:
                                f["status"] = "failed"
                                f["error"] = f"Text layer build exception: {e}"
                                errors.append(f"{fid}: {e}")

                        text_path = get_text_path(f)
                        if not text_path:
                            run_record["outputs"][fid] = {q: None for q in questions}
                            run_record["validation"][fid] = {
                                "status": "unverified",
                                "note": f"No extracted text layer available. status={f.get('status')} error={f.get('error')} method={f.get('text_method')}",
                                "issues": {"missing_required": [], "format_errors": []},
                            }
                            run_record["files"].append(
                                {
                                    "file_id": fid,
                                    "filename": f.get("filename"),
                                    "metadata": f.get("metadata"),
                                    "token_estimate": None,
                                    "chunked": None,
                                    "text_method": f.get("text_method"),
                                }
                            )
                            continue

                        doc_text = text_path.read_text(encoding="utf-8", errors="replace")

                        tok_est = estimate_tokens(doc_text)
                        ctx = parse_ctx_from_show(meta.get("show", ""))
                        ctx_threshold = int(ctx * 0.8) if ctx else None
                        effective_threshold = int(TOKEN_THRESHOLD)
                        if ctx_threshold:
                            effective_threshold = min(effective_threshold, ctx_threshold)

                        auto_chunk = tok_est > effective_threshold
                        use_chunked = force_chunking or auto_chunk

                        if use_chunked:
                            raw = answer_questions_json_chunked(
                                model=model,
                                document_text=doc_text,
                                questions=questions,
                                context_note=ctx_full,
                                max_chars=int(chunk_chars),
                                overlap=int(overlap_chars),
                            )
                        else:
                            raw = answer_questions_json(
                                model=model,
                                document_text=doc_text,
                                questions=questions,
                                context_note=ctx_full,
                            )

                        answers_by_question = normalize_answers(raw, questions)
                        answers_by_field = {q: answers_by_question.get(q) for q in questions}
                        issues = compute_validation_issues(fields, answers_by_field) if fields else {"missing_required": [], "format_errors": []}

                        run_record["outputs"][fid] = answers_by_field
                        run_record["validation"][fid] = {"status": "unverified", "note": "", "issues": issues}

                        run_record["files"].append(
                            {
                                "file_id": fid,
                                "filename": f.get("filename"),
                                "metadata": f.get("metadata"),
                                "token_estimate": tok_est,
                                "chunked": bool(use_chunked),
                                "text_method": f.get("text_method"),
                            }
                        )

                # Persist updated file records back to files.json
                mf2 = load_files_manifest(pp)
                by_id = {x.get("file_id"): x for x in mf2.get("files", []) if x.get("file_id")}
                for fid, rec in file_map.items():
                    if fid in by_id:
                        by_id[fid] = rec
                mf2["files"] = list(by_id.values())
                save_files_manifest(pp, mf2)

                run_path = save_run(pp, run_record)
                st.success(f"Saved run: {run_path.name}")

                if errors:
                    with st.expander("Non-fatal issues during run"):
                        for e in errors[:200]:
                            st.write("- " + e)

                st.rerun()

elif page == "Review":
    st.subheader("Review")

    if not pp or not active_slug:
        st.warning("Select or create a project first.")
    else:
        runs = list_runs(pp)
        if not runs:
            st.info("No runs yet. Create one in 'Run Extraction'.")
        else:
            run_choice = st.selectbox("Select a run version", options=[p.name for p in runs], index=0)
            run = load_run(pp.runs_dir / run_choice)

            mode = st.radio("Mode", ["Dave (Reader/Analyst)", "Bob (Validator)"], horizontal=True)

            mf = load_files_manifest(pp)
            files = {f["file_id"]: f for f in mf.get("files", []) if f.get("file_id")}
            outputs: dict = run.get("outputs", {}) or {}
            validation: dict = run.get("validation", {}) or {}

            # Field list
            standard_file = run.get("standard_file")
            fields_list: list[str] = []
            std_fields: list[dict] = []
            if standard_file and standard_file != "(none)":
                std = load_standard(pp, pp.standards_dir / standard_file)
                if isinstance(std.get("fields"), list):
                    std_fields = std.get("fields") or []
                    fields_list = [str(x.get("field") or "").strip() for x in std_fields if str(x.get("field") or "").strip()]
            if not fields_list and outputs:
                first = next(iter(outputs.values()))
                if isinstance(first, dict):
                    fields_list = list(first.keys())

            # Summary table
            rows = []
            for fid, f in files.items():
                out = outputs.get(fid) or {}
                val = validation.get(fid) or {"status": "unverified", "note": "", "issues": {}}
                issues = val.get("issues") or {}
                miss = issues.get("missing_required") or []
                ferr = issues.get("format_errors") or []
                comp = completeness_ratio([{"field": k} for k in fields_list], out if isinstance(out, dict) else {})
                rows.append(
                    {
                        "file_id": fid,
                        "filename": f.get("filename"),
                        "type": Path(str(f.get("stored_path") or "")).suffix.lower(),
                        "status": f.get("status"),
                        "text_method": f.get("text_method"),
                        "preview_images": f.get("preview_images_count", 0),
                        "validation_status": val.get("status", "unverified"),
                        "missing_required": len(miss),
                        "format_errors": len(ferr),
                        "completeness_%": int(comp * 100),
                    }
                )
            df = pd.DataFrame(rows)

            if mode.startswith("Dave"):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Files", len(df))
                c2.metric("Ready", int((df["status"] == "ready").sum()) if "status" in df.columns else 0)
                c3.metric("Avg completeness", f"{int(df['completeness_%'].mean()) if len(df) else 0}%")
                c4.metric("Needs attention", int(((df["missing_required"] > 0) | (df["format_errors"] > 0)).sum()))
                df_view = df.sort_values(["missing_required", "format_errors", "completeness_%"], ascending=[False, False, True])
            else:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Unverified", int((df["validation_status"] == "unverified").sum()))
                c2.metric("Flagged", int((df["validation_status"] == "flagged").sum()))
                c3.metric("Verified", int((df["validation_status"] == "verified").sum()))
                c4.metric("Rule issues", int(((df["missing_required"] > 0) | (df["format_errors"] > 0)).sum()))
                df_view = df[df["validation_status"] != "verified"].copy()
                if df_view.empty:
                    df_view = df.copy()
                df_view = df_view.sort_values(["validation_status", "missing_required", "format_errors"], ascending=[True, False, False])

            st.dataframe(df_view, width="stretch", hide_index=True)

            st.markdown("---")
            fid_options = df_view["file_id"].tolist() if len(df_view) else df["file_id"].tolist()
            fid_choice = st.selectbox("Open a file", options=fid_options, index=0 if fid_options else 0)

            if fid_choice:
                f = files.get(fid_choice, {})
                out = outputs.get(fid_choice) or {}
                val = validation.get(fid_choice) or {"status": "unverified", "note": "", "issues": {}}

                left, right = st.columns([1.35, 1], gap="large")
                with left:
                    st.markdown("### Document")
                    st.write(f"**{f.get('filename')}**")
                    st.caption(f"Metadata: {f.get('metadata')}")

                    stored_path = Path(str(f.get("stored_path") or ""))

                    # PDF images (previews) for validation
                    if stored_path.suffix.lower() == ".pdf":
                        imgs = list_preview_images(pp, f)
                        if imgs:
                            st.caption(f"PDF page images: {len(imgs)}")
                            page = st.slider("Page", min_value=1, max_value=len(imgs), value=1, key=f"page_{fid_choice}")
                            st.image(str(imgs[page - 1]), caption=imgs[page - 1].name)
                        else:
                            st.info("No PDF page images stored yet. Re-run extraction with 'Store PDF page images' enabled (or force rebuild).")

                    # Image file
                    elif stored_path.is_file() and is_image(f):
                        st.image(str(stored_path), caption=stored_path.name)
                    st.markdown("### Fields")
                    if not isinstance(out, dict):
                        out = {}

                    field_rows = (
                        [{"field": k, "value": ("" if out.get(k) is None else str(out.get(k)))} for k in fields_list]
                        if fields_list
                        else [{"field": k, "value": "" if out.get(k) is None else str(out.get(k))} for k in out.keys()]
                    )
                    fdf = pd.DataFrame(field_rows)
                    edited = st.data_editor(
                        fdf,
                        width="stretch",
                        column_config={
                            "field": st.column_config.TextColumn(disabled=True),
                            "value": st.column_config.TextColumn(),
                        },
                        hide_index=True,
                    )
                    
                with right:
                    # Extracted text layer (editable + saveable)
                    text_path = get_text_path(f)
                    doc_text = ""
                    if text_path:
                        try:
                            doc_text = text_path.read_text(encoding="utf-8", errors="replace")
                        except Exception as e:
                            st.error(f"Failed to read text layer: {e}")
                            doc_text = ""
                    else:
                        st.warning("No extracted text layer available for this file yet. You can paste/edit text below and save it.")

                    too_big = len(doc_text) > 60000
                    edit_full = st.toggle(
                        "Edit full text (recommended if you want to save/rerun; may be slow for huge docs)",
                        value=(not too_big),
                        key=f"edit_full_text__{run_choice}__{fid_choice}",
                    )

                    if doc_text and (not edit_full) and too_big:
                        st.caption(f"Showing first 60,000 / {len(doc_text):,} characters. Turn on 'Edit full text' to save/rerun full content.")
                        shown_text = doc_text[:60000]
                    else:
                        shown_text = doc_text

                    edited_doc_text = st.text_area(
                        "Extracted text layer (editable)",
                        value=shown_text,
                        key=f"doc_text_edit__{run_choice}__{fid_choice}",
                        height=320,
                    )

                    save_text_btn = st.button(
                        "Save text layer",
                        key=f"save_text__{run_choice}__{fid_choice}",
                        disabled=(doc_text != "" and too_big and not edit_full),
                    )

                    if save_text_btn:
                        try:
                            out_text_path = pp.text_dir / f"{fid_choice}.txt"
                            ensure_dir(out_text_path.parent)
                            out_text_path.write_text((edited_doc_text or ""), encoding="utf-8")

                            update_file_record_in_manifest(
                                pp,
                                fid_choice,
                                {
                                    "text_path": str(out_text_path),
                                    "status": "ready",
                                    "error": None,
                                    "text_method": "manual_edit",
                                    "processed_at": now_iso(),
                                },
                            )
                            append_jsonl(
                                pp.audit,
                                {
                                    "ts": now_iso(),
                                    "action": "file.text_layer_manual_edit",
                                    "project": pp.root.name,
                                    "file_id": fid_choice,
                                },
                            )
                            st.success("Saved text layer.")
                        except Exception as e:
                            st.error(f"Failed to save text layer: {e}")

                    st.markdown("---")
                    st.markdown("### Validation")
                    status = st.selectbox(
                        "Status",
                        ["unverified", "verified", "flagged"],
                        index=["unverified", "verified", "flagged"].index(val.get("status", "unverified")),
                    )
                    note = st.text_area("Flag/Review note", value=val.get("note", ""), height=90)

                    issues = val.get("issues") or {}
                    miss = issues.get("missing_required") or []
                    ferr = issues.get("format_errors") or []
                    if miss or ferr:
                        st.warning(f"Issues detected — missing_required: {miss} | format_errors: {ferr}")
                    else:
                        st.success("No rule issues detected (based on current standard rules in this run).")

                    # --- Re-run extraction using the current edited text layer ---
                    with st.expander("Re-run extraction for this file"):
                        ctx_for_rerun = (run.get("context_note") or default_context_note or "").strip()
                        questions_for_rerun = fields_list if fields_list else (list(out.keys()) if isinstance(out, dict) else [])

                        run_model = str(run.get("model") or "").strip() or model
                        available_models = models if models else [run_model]
                        default_idx = available_models.index(run_model) if run_model in available_models else 0

                        rerun_model = st.selectbox(
                            "Extraction model",
                            options=available_models,
                            index=default_idx,
                            key=f"rerun_model__{run_choice}__{fid_choice}",
                        )

                        persist_text_first = st.checkbox(
                            "Save the edited text layer before re-running",
                            value=True,
                            key=f"persist_text_before_rerun__{run_choice}__{fid_choice}",
                        )

                        disable_rerun = (doc_text != "" and len(doc_text) > 60000 and not st.session_state.get(f"edit_full_text__{run_choice}__{fid_choice}", False))
                        if disable_rerun:
                            st.info("Turn on 'Edit full text' on the left to re-run for very large documents.")

                        rerun_now = st.button(
                            "Re-run extraction now (overwrites this file's outputs in this run)",
                            type="primary",
                            key=f"rerun_now__{run_choice}__{fid_choice}",
                            disabled=(not questions_for_rerun or disable_rerun),
                        )

                        if rerun_now:
                            try:
                                text_to_use = edited_doc_text or ""

                                if persist_text_first:
                                    out_text_path = pp.text_dir / f"{fid_choice}.txt"
                                    ensure_dir(out_text_path.parent)
                                    out_text_path.write_text(text_to_use, encoding="utf-8")
                                    update_file_record_in_manifest(
                                        pp,
                                        fid_choice,
                                        {
                                            "text_path": str(out_text_path),
                                            "status": "ready",
                                            "error": None,
                                            "text_method": "manual_edit" if doc_text else "manual_entry",
                                            "processed_at": now_iso(),
                                        },
                                    )
                                    append_jsonl(
                                        pp.audit,
                                        {
                                            "ts": now_iso(),
                                            "action": "file.text_layer_saved_before_rerun",
                                            "project": pp.root.name,
                                            "file_id": fid_choice,
                                        },
                                    )

                                params = run.get("params") or {}
                                force_chunk = bool(params.get("force_chunking", False))
                                token_threshold = int(params.get("token_threshold", TOKEN_THRESHOLD))
                                chunk_chars = int(params.get("chunk_chars", 12000))
                                overlap_chars = int(params.get("overlap_chars", 800))

                                tok_est = estimate_tokens(text_to_use)
                                rerun_meta = model_meta(rerun_model)
                                ctx = parse_ctx_from_show((rerun_meta or {}).get("show", ""))
                                ctx_threshold = int(ctx * 0.8) if ctx else None
                                effective_threshold = int(token_threshold)
                                if ctx_threshold:
                                    effective_threshold = min(effective_threshold, ctx_threshold)
                                auto_chunk = tok_est > effective_threshold
                                use_chunked = force_chunk or auto_chunk

                                if use_chunked:
                                    raw = answer_questions_json_chunked(
                                        model=rerun_model,
                                        document_text=text_to_use,
                                        questions=questions_for_rerun,
                                        context_note=ctx_for_rerun,
                                        max_chars=chunk_chars,
                                        overlap=overlap_chars,
                                    )
                                else:
                                    raw = answer_questions_json(
                                        model=rerun_model,
                                        document_text=text_to_use,
                                        questions=questions_for_rerun,
                                        context_note=ctx_for_rerun,
                                    )

                                answers_by_question = normalize_answers(raw, questions_for_rerun)
                                answers_by_field = {q: answers_by_question.get(q) for q in questions_for_rerun}

                                issues2 = compute_validation_issues(std_fields, answers_by_field) if std_fields else {"missing_required": [], "format_errors": []}

                                outputs[fid_choice] = answers_by_field
                                validation[fid_choice] = {
                                    "status": "unverified",
                                    "note": f"Re-run on {now_iso()} with model={rerun_model}",
                                    "issues": issues2,
                                    "updated_at": now_iso(),
                                    "rerun_model": rerun_model,
                                }

                                run_files = run.get("files") or []
                                updated = False
                                for rf in run_files:
                                    if str(rf.get("file_id")) == str(fid_choice):
                                        rf["token_estimate"] = tok_est
                                        rf["chunked"] = bool(use_chunked)
                                        rf["model"] = rerun_model
                                        rf["rerun_at"] = now_iso()
                                        updated = True
                                        break
                                if not updated:
                                    run_files.append(
                                        {
                                            "file_id": fid_choice,
                                            "filename": f.get("filename"),
                                            "metadata": f.get("metadata"),
                                            "token_estimate": tok_est,
                                            "chunked": bool(use_chunked),
                                            "model": rerun_model,
                                            "rerun_at": now_iso(),
                                        }
                                    )
                                run["files"] = run_files

                                run["outputs"] = outputs
                                run["validation"] = validation
                                run["audit"] = run.get("audit") or []
                                run["audit"].append(
                                    {
                                        "ts": now_iso(),
                                        "action": "extraction.rerun",
                                        "file_id": fid_choice,
                                        "model": rerun_model,
                                        "chunked": bool(use_chunked),
                                    }
                                )

                                run_path = pp.runs_dir / run_choice
                                write_json(run_path, run)
                                append_jsonl(
                                    pp.audit,
                                    {
                                        "ts": now_iso(),
                                        "action": "run.rerun_file",
                                        "project": pp.root.name,
                                        "run_id": run.get("run_id"),
                                        "file_id": fid_choice,
                                        "model": rerun_model,
                                        "chunked": bool(use_chunked),
                                    },
                                )

                                st.success("Re-ran extraction for this file and updated the run.")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Re-run failed: {e}")

                    # --- Save field edits + validation (manual) ---
                    save_btn = st.button("Save changes", type="primary")
                    if save_btn:
                        new_out = dict(out)
                        for _, r in edited.iterrows():
                            k = str(r["field"]).strip()
                            v = str(r["value"]).strip()
                            new_out[k] = v if v else None

                        if status == "flagged" and not note.strip():
                            st.error("Flagged requires a note/reason.")
                        else:
                            before = outputs.get(fid_choice) or {}
                            outputs[fid_choice] = new_out
                            validation[fid_choice] = {"status": status, "note": note, "issues": issues, "updated_at": now_iso()}
                            run["outputs"] = outputs
                            run["validation"] = validation

                            run["audit"] = run.get("audit") or []
                            run["audit"].append(
                                {
                                    "ts": now_iso(),
                                    "action": "review.save",
                                    "file_id": fid_choice,
                                    "changed_fields": [k for k in new_out.keys() if str(before.get(k)) != str(new_out.get(k))],
                                    "status": status,
                                }
                            )

                            run_path = pp.runs_dir / run_choice
                            write_json(run_path, run)
                            append_jsonl(
                                pp.audit,
                                {"ts": now_iso(), "action": "run.update_file", "project": pp.root.name, "run_id": run.get("run_id"), "file_id": fid_choice, "status": status, "note": note},
                            )
                            st.success("Saved.")
                            st.rerun()

elif page == "Export":
    st.subheader("Export (CSV / JSON)")

    if not pp or not active_slug:
        st.warning("Select or create a project first.")
    else:
        runs = list_runs(pp)
        if not runs:
            st.info("No runs yet.")
        else:
            run_choice = st.selectbox("Select a run version to export", options=[p.name for p in runs], index=0)
            run = load_run(pp.runs_dir / run_choice)

            scope = st.selectbox("Scope", ["all", "verified only", "flagged only", "unverified only"], index=0)

            outputs: dict = run.get("outputs", {}) or {}
            validation: dict = run.get("validation", {}) or {}

            mf = load_files_manifest(pp)
            files = {f["file_id"]: f for f in mf.get("files", []) if f.get("file_id")}

            rows = []
            for fid, out in outputs.items():
                v = validation.get(fid) or {"status": "unverified", "note": ""}
                stt = v.get("status", "unverified")

                if scope == "verified only" and stt != "verified":
                    continue
                if scope == "flagged only" and stt != "flagged":
                    continue
                if scope == "unverified only" and stt != "unverified":
                    continue

                f = files.get(fid) or {}
                meta = f.get("metadata") or {}

                row = {
                    "project": pp.root.name,
                    "run_id": run.get("run_id"),
                    "file_id": fid,
                    "filename": f.get("filename"),
                    "location": meta.get("location"),
                    "name": meta.get("name"),
                    "date": meta.get("date"),
                    "validation_status": stt,
                    "validation_note": v.get("note", ""),
                }
                if isinstance(out, dict):
                    row.update(out)
                rows.append(row)

            if not rows:
                st.warning("No rows to export for the selected scope.")
            else:
                df = pd.DataFrame(rows)
                st.dataframe(df, width="stretch", hide_index=True)

                export_id = f"{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_{short_id()}"
                base_name = f"export_{export_id}"

                csv_bytes = df.to_csv(index=False).encode("utf-8")
                csv_path = pp.exports_dir / f"{base_name}.csv"
                csv_path.write_bytes(csv_bytes)

                json_obj = {
                    "project": pp.root.name,
                    "run_id": run.get("run_id"),
                    "scope": scope,
                    "exported_at": now_iso(),
                    "rows": rows,
                }
                json_path = pp.exports_dir / f"{base_name}.json"
                write_json(json_path, json_obj)

                append_jsonl(pp.audit, {"ts": now_iso(), "action": "export.create", "project": pp.root.name, "run_id": run.get("run_id"), "scope": scope, "export_id": export_id})

                c1, c2 = st.columns(2)
                with c1:
                    st.download_button("Download CSV", data=csv_bytes, file_name=csv_path.name, mime="text/csv")
                    st.caption(f"Saved to: {csv_path}")
                with c2:
                    st.download_button("Download JSON", data=json.dumps(json_obj, indent=2).encode("utf-8"), file_name=json_path.name, mime="application/json")
                    st.caption(f"Saved to: {json_path}")

else:
    st.info("Pick a page from the sidebar.")



