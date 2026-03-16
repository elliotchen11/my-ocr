# app_streamlit_llm_extract.py
# Run: streamlit run app_streamlit_llm_extract.py
#
# Local-first workflow:
# - Upload PDFs/images/txt into a Project
# - Build a text layer:
#     * TXT: read file
#     * PDF: try embedded text via PyMuPDF (fitz)
#            else render pages via PyMuPDF (your convert_to_img.py) -> OCR via Ollama (your ocr.py)
#     * Images: OCR via Ollama (your ocr.py)
# - Extract fields via llm_extract_core.py (Ollama)
# - Review in 2 modes: Dave (Reader) / Bob (Validator)
# - Export CSV/JSON
#
# IMPORTANT:
# - This file DOES NOT use poppler-utils (pdftoppm/pdftotext).
# - PDF->image rendering uses your convert_to_img.py (PyMuPDF/fitz).
# - OCR uses your ocr.py (Ollama vision).
# - Extraction uses your llm_extract_core.py (Ollama).

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from llm_extract_core import answer_questions_json, answer_questions_json_chunked

# ---- Your local PDF->images converter (PyMuPDF) ----
# File provided by you: convert_to_img.py
try:
    from convert_to_img import convert_pdf2img  # type: ignore
except Exception:
    convert_pdf2img = None  # type: ignore

# ---- Your local Ollama OCR (vision) ----
# File provided by you: ocr.py
try:
    from ocr import ocr_image, DEFAULT_PROMPT  # type: ignore
except Exception:
    ocr_image = None  # type: ignore
    DEFAULT_PROMPT = "Extract all readable text from this image. Return plain text only."

# ---- Optional: PyMuPDF for extracting embedded text from PDFs ----
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None  # type: ignore


# ---------------- Paths ----------------
DATA_ROOT = Path("/home/doanm/ocr-automate/data")
PROJECTS_ROOT = DATA_ROOT / "streamlit_projects"


# ---------------- Small utilities ----------------
def now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_slug(name: str) -> str:
    name = (name or "").strip().lower()
    name = re.sub(r"[^a-z0-9]+", "-", name)
    name = re.sub(r"-{2,}", "-", name).strip("-")
    return name or "project"


def short_id() -> str:
    return uuid.uuid4().hex[:10]


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


# ---------------- Ollama model list + show (CLI) ----------------
@st.cache_data(ttl=10)
def list_ollama_models() -> list[str]:
    import subprocess

    try:
        out = subprocess.check_output(["ollama", "list"], text=True, stderr=subprocess.STDOUT)
    except Exception:
        return []

    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    if not lines:
        return []
    if "NAME" in lines[0].upper():
        lines = lines[1:]

    models: list[str] = []
    for ln in lines:
        name = ln.split()[0]
        if name:
            models.append(name)

    seen: set[str] = set()
    uniq: list[str] = []
    for m in models:
        if m not in seen:
            seen.add(m)
            uniq.append(m)
    return uniq


@st.cache_data(ttl=60)
def ollama_show_raw(model: str) -> str:
    import subprocess

    try:
        out = subprocess.check_output(["ollama", "show", model], text=True, stderr=subprocess.STDOUT)
        return out.strip()
    except subprocess.CalledProcessError as e:
        return (e.output or "").strip() or f"Failed to run: ollama show {model}"
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

    m = re.search(r"[:\\-]([0-9]+)b\\b", n)
    if m:
        b = int(m.group(1))
        tags.append("fast" if b <= 4 else ("balanced" if b <= 9 else "quality"))

    ctx_nums = [int(x) for x in re.findall(r"\\b(8192|16384|32768|65536|131072)\\b", t)]
    if ctx_nums and max(ctx_nums) >= 32768:
        tags.append("long-doc")

    if not tags:
        tags.append("general")

    seen: set[str] = set()
    out: list[str] = []
    for x in tags:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


@st.cache_data(ttl=60)
def model_meta(model: str) -> dict:
    show_text = ollama_show_raw(model)
    tags = infer_tags_from_show(model, show_text)
    return {"show": show_text, "tags": tags}


# ---------------- Project storage ----------------
@dataclass
class ProjectPaths:
    root: Path
    manifest: Path
    audit: Path
    files_manifest: Path
    files_dir: Path
    text_dir: Path
    standards_dir: Path
    structures_dir: Path
    runs_dir: Path
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
        standards_dir=root / "standards",
        structures_dir=root / "structures",
        runs_dir=root / "runs",
        exports_dir=root / "exports",
    )


def ensure_project_layout(pp: ProjectPaths) -> None:
    ensure_dir(pp.root)
    ensure_dir(pp.files_dir)
    ensure_dir(pp.text_dir)
    ensure_dir(pp.standards_dir)
    ensure_dir(pp.structures_dir)
    ensure_dir(pp.runs_dir)
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


# ---------------- Text layer extraction (LOCAL ONLY) ----------------
def extract_pdf_text_via_pymupdf(pdf_path: Path) -> str | None:
    if fitz is None or not pdf_path.is_file():
        return None
    try:
        doc = fitz.open(str(pdf_path))
        try:
            parts: list[str] = []
            for page in doc:
                txt = page.get_text("text")  # type: ignore
                if txt:
                    parts.append(txt)
            out = "\n".join(parts).strip()
            return out if out else None
        finally:
            doc.close()
    except Exception:
        return None


def ocr_images_with_ollama(ocr_model: str, images: list[Path], prompt: str) -> str | None:
    if ocr_image is None:
        return None
    if not images:
        return None

    merged_parts: list[str] = []
    for img in images:
        m = re.search(r"(?i)_page(?P<page>\\d+)\\b", img.stem)
        page = int(m.group("page")) if m else 0
        try:
            txt = (ocr_image(model=ocr_model, image_path=img, prompt=prompt) or "").strip()
        except Exception as e:
            txt = f"[OCR FAILED] {e}"

        if page > 0:
            merged_parts.append(f"\\n\\n===== PAGE {page} ({img.name}) =====\\n{txt}\\n")
        else:
            merged_parts.append(f"\\n\\n===== IMAGE ({img.name}) =====\\n{txt}\\n")

    out = "".join(merged_parts).strip()
    return out if out else None


def extract_text_layer_local(
    stored_path: Path,
    *,
    ocr_model: str,
    ocr_prompt: str,
    pdf_zoom: float = 2.0,
) -> tuple[str | None, str, str | None]:
    if not stored_path.is_file():
        return None, "unsupported", "Stored file path missing or not a file."

    suf = stored_path.suffix.lower()

    if suf == ".txt":
        try:
            return stored_path.read_text(encoding="utf-8", errors="replace"), "txt", None
        except Exception as e:
            return None, "txt", f"Failed to read txt: {e}"

    if suf == ".pdf":
        t = extract_pdf_text_via_pymupdf(stored_path)
        if t and len(t) >= 200:
            return t, "pdf_pymupdf_text", None

        if convert_pdf2img is None:
            return None, "pdf_ollama_ocr", "convert_to_img.py not importable (convert_pdf2img missing)."
        if ocr_image is None:
            return None, "pdf_ollama_ocr", "ocr.py not importable (ocr_image missing)."

        import tempfile
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            try:
                img_files = convert_pdf2img(
                    input_file=str(stored_path),
                    out_dir=str(out_dir),
                    base_override=stored_path.stem,
                    zoom=float(pdf_zoom),
                )
            except Exception as e:
                return None, "pdf_ollama_ocr", f"PDF->image conversion failed: {e}"

            images = [Path(p) for p in img_files if p and Path(p).is_file()]
            images.sort(key=lambda p: p.name)

            t2 = ocr_images_with_ollama(ocr_model, images, ocr_prompt)
            if t2 and len(t2) >= 10:
                return t2, "pdf_ollama_ocr", None

        return None, "pdf_ollama_ocr", "OCR produced no text (scanned/low quality?)"

    if suf in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}:
        if ocr_image is None:
            return None, "image_ollama_ocr", "ocr.py not importable (ocr_image missing)."
        t = ocr_images_with_ollama(ocr_model, [stored_path], ocr_prompt)
        if t and len(t) >= 1:
            return t, "image_ollama_ocr", None
        return None, "image_ollama_ocr", "OCR produced no text."

    return None, "unsupported", f"Unsupported file type: {suf}"


def ensure_text_layer_for_file(
    pp: ProjectPaths,
    file_record: dict,
    *,
    ocr_model: str,
    ocr_prompt: str,
    pdf_zoom: float,
) -> dict:
    stored = Path(file_record.get("stored_path") or "")
    file_id = file_record.get("file_id")

    if not file_id:
        file_record["status"] = "failed"
        file_record["error"] = "Missing file_id."
        return file_record

    ensure_project_layout(pp)
    out_text_path = pp.text_dir / f"{file_id}.txt"

    text, method, err = extract_text_layer_local(
        stored,
        ocr_model=ocr_model,
        ocr_prompt=ocr_prompt,
        pdf_zoom=pdf_zoom,
    )

    if text and text.strip():
        out_text_path.write_text(text, encoding="utf-8")
        file_record["text_path"] = str(out_text_path)
        file_record["status"] = "ready"
        file_record["error"] = None
        file_record["text_method"] = method
        file_record["processed_at"] = now_iso()
        append_jsonl(pp.audit, {"ts": now_iso(), "action": "file.text_layer_built", "project": pp.root.name, "file_id": file_id, "method": method})
    else:
        file_record["status"] = "failed"
        file_record["error"] = err or "Text layer build failed."
        file_record["text_method"] = method
        file_record["processed_at"] = now_iso()
        append_jsonl(
            pp.audit,
            {"ts": now_iso(), "action": "file.text_layer_failed", "project": pp.root.name, "file_id": file_id, "method": method, "error": file_record["error"]},
        )

    return file_record


# ---------------- Standards / structure / runs ----------------
def list_standard_versions(pp: ProjectPaths) -> list[Path]:
    ensure_project_layout(pp)
    return sorted(pp.standards_dir.glob("standard_v*.json"))


def next_standard_version(pp: ProjectPaths) -> int:
    vs: list[int] = []
    for p in list_standard_versions(pp):
        m = re.search(r"standard_v(\\d+)\\.json$", p.name)
        if m:
            vs.append(int(m.group(1)))
    return (max(vs) + 1) if vs else 1


def load_standard(path: Path) -> dict:
    return read_json(path, {})


def save_standard(pp: ProjectPaths, standard: dict) -> Path:
    v = next_standard_version(pp)
    path = pp.standards_dir / f"standard_v{v}.json"
    standard = {**standard, "version": v, "saved_at": now_iso()}
    write_json(path, standard)
    append_jsonl(pp.audit, {"ts": now_iso(), "action": "standard.save", "project": pp.root.name, "version": v})
    return path


def list_structure_versions(pp: ProjectPaths) -> list[Path]:
    ensure_project_layout(pp)
    return sorted(pp.structures_dir.glob("structure_v*.txt"))


def next_structure_version(pp: ProjectPaths) -> int:
    vs: list[int] = []
    for p in list_structure_versions(pp):
        m = re.search(r"structure_v(\\d+)\\.txt$", p.name)
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


def save_run(pp: ProjectPaths, run_record: dict) -> Path:
    run_id = run_record.get("run_id") or f"{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_{short_id()}"
    run_record["run_id"] = run_id
    path = pp.runs_dir / f"run_{run_id}.json"
    write_json(path, run_record)
    append_jsonl(pp.audit, {"ts": now_iso(), "action": "run.save", "project": pp.root.name, "run_id": run_id})
    return path


def load_run(path: Path) -> dict:
    return read_json(path, {})


def build_context_note(base: str, standard: dict | None, structure_text: str | None) -> str:
    parts: list[str] = []
    if base.strip():
        parts.append(base.strip())

    if structure_text and structure_text.strip():
        parts.append("File-type structure / layout notes:\\n" + structure_text.strip())

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
        parts.append("\\n".join(lines))

    return "\\n\\n".join(parts).strip()


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
            if any(tok in fmt for tok in ["^", "$", "\\\\d", "\\\\w", "(", ")", "[", "]", "{", "}", "|", "+", "*", "?"]):
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


# ---------------- Add / ingest files ----------------
def add_file_record(pp: ProjectPaths, filename: str, meta: dict, raw_bytes: bytes, ext: str) -> dict:
    file_id = short_id()
    stored_name = f"{file_id}_{safe_slug(Path(filename).stem)}{ext}"
    stored_path = pp.files_dir / stored_name

    ensure_project_layout(pp)
    stored_path.write_bytes(raw_bytes)

    record = {
        "file_id": file_id,
        "filename": filename,
        "stored_path": str(stored_path),
        "text_path": None,
        "uploaded_at": now_iso(),
        "status": "uploaded",
        "error": None,
        "metadata": meta,
        "text_method": None,
        "processed_at": None,
    }

    mf = load_files_manifest(pp)
    mf["files"].append(record)
    save_files_manifest(pp, mf)

    append_jsonl(pp.audit, {"ts": now_iso(), "action": "file.add", "project": pp.root.name, "file_id": file_id, "filename": filename})
    return record


# ---------------- UI ----------------
ensure_dir(DATA_ROOT)
ensure_dir(PROJECTS_ROOT)

st.set_page_config(page_title="Project OCR/Text → LLM Extract → Review (Dave/Bob)", layout="wide")
st.title("Project OCR/Text → LLM Extract → Review (Dave/Bob)")

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

default_label = "(none)"
if st.session_state.active_project_slug:
    for lab in proj_labels:
        if lab.endswith(st.session_state.active_project_slug):
            default_label = lab
            break

selected_label = st.sidebar.selectbox("Active project", options=proj_labels, index=proj_labels.index(default_label), key="active_project_select")
active_slug = label_to_slug.get(selected_label)
if active_slug:
    st.session_state.active_project_slug = active_slug

active_slug = st.session_state.active_project_slug
pp: ProjectPaths | None = project_paths(active_slug) if active_slug else None
manifest = load_project_manifest(pp) if pp else None

# ---- Model picker (global) ----
st.sidebar.markdown("---")
st.sidebar.subheader("LLM Model (Extraction)")
models = list_ollama_models()
annotate = st.sidebar.toggle("Annotate model dropdown with tags", value=True, key="annotate_models")
if st.sidebar.button("Refresh models", key="refresh_models"):
    st.cache_data.clear()
    models = list_ollama_models()

if not models:
    st.sidebar.warning("No Ollama models detected.")
    model = st.sidebar.text_input("Ollama model (manual)", value="gemma3", key="manual_model")
    meta = {"tags": ["unknown"], "show": ""}
else:
    default_model = "gemma3" if "gemma3" in models else models[0]
    if annotate:
        def _fmt(m: str) -> str:
            mm = model_meta(m)
            return f"{m}  —  {', '.join(mm['tags'])}"
        model = st.sidebar.selectbox("Ollama model", models, index=models.index(default_model), format_func=_fmt, key="model_select")
    else:
        model = st.sidebar.selectbox("Ollama model", models, index=models.index(default_model), key="model_select_plain")
    meta = model_meta(model)
    st.sidebar.caption(f"Tags: **{', '.join(meta['tags'])}**")
    with st.sidebar.expander("ollama show"):
        st.sidebar.code(meta["show"], language="text")

# ---- OCR settings (PDF/images) ----
st.sidebar.markdown("---")
st.sidebar.subheader("OCR (PDF / images)")

ocr_model = st.sidebar.selectbox(
    "OCR model (Ollama vision)",
    options=models if models else [model],
    index=(models.index(model) if models and model in models else 0),
    key="ocr_model_select",
)
ocr_prompt = st.sidebar.text_area("OCR prompt", value=DEFAULT_PROMPT, height=80, key="ocr_prompt")
pdf_zoom = st.sidebar.slider("PDF render zoom (PyMuPDF)", min_value=1.0, max_value=4.0, value=2.0, step=0.25, key="pdf_zoom")

if ocr_image is None:
    st.sidebar.warning("ocr.py not importable (ocr_image missing). OCR for PDFs/images won't work.")
if convert_pdf2img is None:
    st.sidebar.warning("convert_to_img.py not importable (convert_pdf2img missing). PDF→image conversion won't work.")
if fitz is None:
    st.sidebar.warning("PyMuPDF (fitz) not importable. PDF embedded-text extraction won't work.")

# ---- Extraction controls ----
st.sidebar.markdown("---")
st.sidebar.subheader("Extraction controls")
force_chunking = st.sidebar.toggle("Force chunking mode", value=False, key="force_chunking")
TOKEN_THRESHOLD = st.sidebar.number_input(
    "Auto-switch to chunking above (estimated tokens)",
    min_value=5_000,
    max_value=120_000,
    value=25_000,
    step=5_000,
    key="token_threshold",
)
default_context_note = st.sidebar.text_area(
    "Context note (default)",
    value="Extract only from evidence in the document text. If not present, return null.",
    height=90,
    key="default_context_note",
)

# ---------------- Pages ----------------
if page == "Projects":
    st.subheader("Projects")

    c1, c2 = st.columns([1, 1])
    with c1:
        new_name = st.text_input("Create a new project", placeholder="e.g., FIA M&A Batch Feb 2026", key="new_project_name")
    with c2:
        if st.button("Create project", type="primary", disabled=not bool(new_name.strip()), key="create_project_btn"):
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

    if active_slug and pp and manifest:
        st.markdown("---")
        st.subheader("Project picklists (metadata)")
        st.caption("These power the upload metadata picklists.")
        pick = manifest.get("picklists", {})
        locs = pick.get("location", [])
        names = pick.get("name", [])

        colA, colB = st.columns(2)
        with colA:
            st.write("Location picklist")
            locs_edit = st.text_area("One per line", value="\\n".join(locs) if locs else "", height=140, key="locs_edit")
        with colB:
            st.write("Name picklist")
            names_edit = st.text_area("One per line", value="\\n".join(names) if names else "", height=140, key="names_edit")

        if st.button("Save picklists", key="save_picklists_btn"):
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
        files = mf.get("files", [])

        st.markdown("### Existing files")
        if not files:
            st.info("No files uploaded yet.")
        else:
            df = pd.DataFrame(
                [
                    {
                        "file_id": f.get("file_id"),
                        "filename": f.get("filename"),
                        "status": f.get("status"),
                        "text_method": f.get("text_method"),
                        "error": f.get("error"),
                        "uploaded_at": f.get("uploaded_at"),
                        "processed_at": f.get("processed_at"),
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
            meta_location = st.selectbox("Location", options=loc_opts if loc_opts else ["(none)"], key="upload_location")
        with c2:
            meta_name = st.selectbox("Name", options=name_opts if name_opts else ["(none)"], key="upload_name")
        with c3:
            meta_date = st.date_input("Date", key="upload_date")

        uploads = st.file_uploader(
            "Upload documents (.txt / .pdf / images). Text layers will be built later (Run Extraction).",
            type=["txt", "pdf", "png", "jpg", "jpeg", "webp", "tif", "tiff"],
            accept_multiple_files=True,
            key="uploads",
        )

        if st.button("Add to project", type="primary", disabled=not uploads, key="add_to_project_btn"):
            added = 0
            for up in uploads or []:
                ext = "." + up.name.split(".")[-1].lower() if "." in up.name else ""
                meta = {"location": meta_location, "name": meta_name, "date": str(meta_date)}
                add_file_record(pp, up.name, meta, up.getvalue(), ext)
                added += 1
            st.success(f"Added {added} file(s) to project.")
            st.rerun()

        st.markdown("---")
        st.markdown("### Build text layers (optional)")
        st.caption("Same as 'Run Extraction' auto-build — useful to pre-OCR everything.")
        if st.button("Build text layers for ALL uploaded files", type="secondary", key="build_text_layers_all"):
            mf = load_files_manifest(pp)
            files = mf.get("files", [])
            if not files:
                st.info("No files to process.")
            else:
                prog = st.progress(0, text="Building text layers…")
                for i, rec in enumerate(files, start=1):
                    rec = ensure_text_layer_for_file(pp, rec, ocr_model=ocr_model, ocr_prompt=ocr_prompt, pdf_zoom=float(pdf_zoom))
                    files[i - 1] = rec
                    prog.progress(i / len(files), text=f"Processed {i}/{len(files)} — {rec.get('filename')}")
                mf["files"] = files
                save_files_manifest(pp, mf)
                st.success("Done. Text layers updated.")
                st.rerun()

elif page == "Data Standard":
    st.subheader("Data Standard (fields + definitions + rules)")

    if not pp or not active_slug:
        st.warning("Select or create a project first.")
    else:
        ensure_project_layout(pp)
        versions = list_standard_versions(pp)
        latest = versions[0] if versions else None
        latest_std = load_standard(latest) if latest else {"fields": []}

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
            key="standard_editor",
        )

        if st.button("Save as new standard version", type="primary", key="save_standard_btn"):
            fields: list[dict] = []
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
            placeholder="e.g., Parties section near top; Dates in header; Payment terms in Section 3; Table 'Schedule A' contains totals...",
            key="structure_text",
        )

        if st.button("Save as new structure version", type="primary", key="save_structure_btn"):
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
            selectable = [
                {
                    "file_id": f.get("file_id"),
                    "filename": f.get("filename"),
                    "status": f.get("status"),
                    "text_method": f.get("text_method"),
                    "error": f.get("error"),
                    "location": (f.get("metadata") or {}).get("location"),
                    "name": (f.get("metadata") or {}).get("name"),
                    "date": (f.get("metadata") or {}).get("date"),
                }
                for f in files
            ]
            st.dataframe(pd.DataFrame(selectable), width="stretch", hide_index=True)

            ids = [f["file_id"] for f in files if f.get("file_id")]
            selected_ids = st.multiselect("Select file_ids to include in this run", options=ids, default=ids[: min(5, len(ids))], key="selected_ids")

            st.markdown("---")
            st.markdown("### Choose Standard + Structure versions")
            std_versions = list_standard_versions(pp)
            struct_versions = list_structure_versions(pp)

            std_choice = st.selectbox("Data Standard version", options=["(none)"] + [p.name for p in std_versions], index=1 if std_versions else 0, key="std_choice")
            struct_choice = st.selectbox("Structure version", options=["(none)"] + [p.name for p in struct_versions], index=1 if struct_versions else 0, key="struct_choice")

            standard = load_standard(pp.standards_dir / std_choice) if std_choice != "(none)" else None
            structure_text = (pp.structures_dir / struct_choice).read_text(encoding="utf-8") if struct_choice != "(none)" else None

            use_data_standard = st.toggle("Use Data Standard fields (recommended)", value=True, key="use_data_standard")
            if use_data_standard and standard and isinstance(standard.get("fields"), list) and standard["fields"]:
                fields = standard["fields"]
                questions = [str(f.get("field") or "").strip() for f in fields if str(f.get("field") or "").strip()]
            else:
                st.caption("Fallback: custom questions (one per line).")
                questions_text = st.text_area("Questions", value="What is the agreement ID?\\nWhat is the agreement date?", height=120, key="questions_text")
                questions = [q.strip() for q in questions_text.splitlines() if q.strip()]
                fields = []

            context_note = st.text_area("Context note (override for this run)", value=default_context_note, height=90, key="context_note_override")
            ctx_full = build_context_note(context_note, standard if use_data_standard else None, structure_text)

            st.markdown("---")
            st.markdown("### Processing options")
            auto_process_missing = st.toggle("Auto-build missing text layers (PDF/images)", value=True, key="auto_process_missing")

            chunk_chars = 12000
            overlap_chars = 800
            with st.expander("Chunking settings"):
                chunk_chars = st.slider("Chunk size (chars)", 4000, 30000, 12000, step=1000, key="chunk_chars")
                overlap_chars = st.slider("Overlap (chars)", 0, 5000, 800, step=100, key="overlap_chars")

            run_btn = st.button("Run extraction now", type="primary", disabled=(not selected_ids or not questions), key="run_extraction_btn")

            if run_btn:
                file_map = {f["file_id"]: f for f in files if f.get("file_id")}
                run_record: dict[str, Any] = {
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
                        "force_chunking": bool(force_chunking),
                        "token_threshold": int(TOKEN_THRESHOLD),
                        "chunk_chars": int(chunk_chars),
                        "overlap_chars": int(overlap_chars),
                        "auto_process_missing": bool(auto_process_missing),
                        "ocr_model": ocr_model,
                        "pdf_zoom": float(pdf_zoom),
                    },
                }
                if struct_choice != "(none)":
                    m = re.search(r"structure_v(\\d+)\\.txt$", struct_choice)
                    run_record["structure_version"] = int(m.group(1)) if m else None

                status_box = st.status("Running extraction…", expanded=True)
                updated_any_manifest = False

                for idx, fid in enumerate(selected_ids, start=1):
                    rec = file_map.get(fid)
                    if not rec:
                        continue

                    status_box.write(f"**{idx}/{len(selected_ids)}** — {rec.get('filename')}")

                    text_path = get_text_path(rec)
                    if not text_path and auto_process_missing:
                        rec2 = ensure_text_layer_for_file(pp, rec, ocr_model=ocr_model, ocr_prompt=ocr_prompt, pdf_zoom=float(pdf_zoom))
                        file_map[fid] = rec2
                        rec = rec2
                        updated_any_manifest = True
                        text_path = get_text_path(rec)

                    if not text_path:
                        run_record["outputs"][fid] = {q: None for q in questions}
                        run_record["validation"][fid] = {
                            "status": "unverified",
                            "note": f"No extracted text layer. status={rec.get('status')} error={rec.get('error')} method={rec.get('text_method')}",
                            "issues": {"missing_required": [], "format_errors": []},
                        }
                        run_record["files"].append({"file_id": fid, "filename": rec.get("filename"), "metadata": rec.get("metadata"), "token_estimate": None, "chunked": None})
                        continue

                    try:
                        doc_text = text_path.read_text(encoding="utf-8", errors="replace")
                    except Exception as e:
                        run_record["outputs"][fid] = {q: None for q in questions}
                        run_record["validation"][fid] = {"status": "unverified", "note": f"Failed to read text layer: {e}", "issues": {"missing_required": [], "format_errors": []}}
                        run_record["files"].append({"file_id": fid, "filename": rec.get("filename"), "metadata": rec.get("metadata"), "token_estimate": None, "chunked": None})
                        continue

                    tok_est = estimate_tokens(doc_text)
                    ctx = parse_ctx_from_show(meta.get("show", ""))
                    ctx_threshold = int(ctx * 0.8) if ctx else None
                    effective_threshold = int(TOKEN_THRESHOLD)
                    if ctx_threshold:
                        effective_threshold = min(effective_threshold, ctx_threshold)

                    auto_chunk = tok_est > effective_threshold
                    use_chunked = bool(force_chunking or auto_chunk)

                    try:
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
                    except Exception as e:
                        run_record["outputs"][fid] = {q: None for q in questions}
                        run_record["validation"][fid] = {"status": "unverified", "note": f"LLM extraction failed: {e}", "issues": {"missing_required": [], "format_errors": []}}
                        run_record["files"].append({"file_id": fid, "filename": rec.get("filename"), "metadata": rec.get("metadata"), "token_estimate": tok_est, "chunked": use_chunked})
                        continue

                    answers_by_field = raw if isinstance(raw, dict) else {q: None for q in questions}
                    issues = compute_validation_issues(fields, answers_by_field) if fields else {"missing_required": [], "format_errors": []}

                    run_record["outputs"][fid] = answers_by_field
                    run_record["validation"][fid] = {"status": "unverified", "note": "", "issues": issues}
                    run_record["files"].append({"file_id": fid, "filename": rec.get("filename"), "metadata": rec.get("metadata"), "token_estimate": tok_est, "chunked": use_chunked})

                if updated_any_manifest:
                    mf2 = load_files_manifest(pp)
                    idx_by_id = {x.get("file_id"): i for i, x in enumerate(mf2.get("files", [])) if x.get("file_id")}
                    for fid, rec in file_map.items():
                        if fid in idx_by_id:
                            mf2["files"][idx_by_id[fid]] = rec
                    save_files_manifest(pp, mf2)

                run_path = save_run(pp, run_record)
                status_box.update(label=f"Saved run: {run_path.name}", state="complete", expanded=False)
                st.success(f"Saved run: {run_path.name}")
                st.rerun()

elif page == "Review":
    st.subheader("Review (mode-driven layout: Dave / Bob)")

    if not pp or not active_slug:
        st.warning("Select or create a project first.")
    else:
        runs = list_runs(pp)
        if not runs:
            st.info("No runs yet. Create one in 'Run Extraction'.")
        else:
            run_choice = st.selectbox("Select a run version", options=[p.name for p in runs], index=0, key="run_choice_review")
            run = load_run(pp.runs_dir / run_choice)

            mode = st.radio("Mode", ["Dave (Reader/Analyst)", "Bob (Validator)"], horizontal=True, key="review_mode")

            mf = load_files_manifest(pp)
            files = {f["file_id"]: f for f in mf.get("files", []) if f.get("file_id")}
            outputs: dict = run.get("outputs", {}) or {}
            validation: dict = run.get("validation", {}) or {}

            standard_file = run.get("standard_file")
            fields_list: list[str] = []
            if standard_file and standard_file != "(none)":
                std = load_standard(pp.standards_dir / standard_file)
                if isinstance(std.get("fields"), list):
                    fields_list = [str(x.get("field") or "").strip() for x in std["fields"] if str(x.get("field") or "").strip()]
            if not fields_list and outputs:
                first = next(iter(outputs.values()))
                if isinstance(first, dict):
                    fields_list = list(first.keys())

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
                        "status": f.get("status"),
                        "text_method": f.get("text_method"),
                        "location": (f.get("metadata") or {}).get("location"),
                        "name": (f.get("metadata") or {}).get("name"),
                        "date": (f.get("metadata") or {}).get("date"),
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
            if df_view.empty:
                st.info("No files to review in this view.")
            else:
                fid_choice = st.selectbox("Open a file", options=df_view["file_id"].tolist(), index=0, key="fid_choice")

                f = files.get(fid_choice, {})
                out = outputs.get(fid_choice) or {}
                val = validation.get(fid_choice) or {"status": "unverified", "note": "", "issues": {}}

                left, right = st.columns([1.2, 1], gap="large")
                with left:
                    st.markdown("### Document")
                    st.write(f"**{f.get('filename')}**")
                    st.caption(f"Metadata: {f.get('metadata')} | text_method: {f.get('text_method')} | status: {f.get('status')}")

                    text_path = get_text_path(f)
                    if text_path:
                        doc_text = text_path.read_text(encoding="utf-8", errors="replace")
                        st.text_area("Extracted text layer", value=doc_text[:20000], height=420, key="doc_text_preview")
                    else:
                        st.warning("No extracted text layer available for this file.")

                with right:
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
                        column_config={"field": st.column_config.TextColumn(disabled=True), "value": st.column_config.TextColumn()},
                        hide_index=True,
                        key="fields_editor",
                    )

                    st.markdown("---")
                    st.markdown("### Validation")
                    status = st.selectbox("Status", ["unverified", "verified", "flagged"], index=["unverified", "verified", "flagged"].index(val.get("status", "unverified")), key="validation_status_select")
                    note = st.text_area("Flag/Review note", value=val.get("note", ""), height=90, key="validation_note")

                    issues = val.get("issues") or {}
                    miss = issues.get("missing_required") or []
                    ferr = issues.get("format_errors") or []
                    if miss or ferr:
                        st.warning(f"Issues detected — missing_required: {miss} | format_errors: {ferr}")
                    else:
                        st.success("No rule issues detected (based on the standard rules used for this run).")

                    save_btn = st.button("Save changes", type="primary", key="save_review_btn")
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
                                {"ts": now_iso(), "action": "review.save", "file_id": fid_choice, "changed_fields": [k for k in new_out.keys() if str(before.get(k)) != str(new_out.get(k))], "status": status}
                            )

                            run_path = pp.runs_dir / run_choice
                            write_json(run_path, run)
                            append_jsonl(pp.audit, {"ts": now_iso(), "action": "run.update_file", "project": pp.root.name, "run_id": run.get("run_id"), "file_id": fid_choice, "status": status, "note": note})
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
            run_choice = st.selectbox("Select a run version to export", options=[p.name for p in runs], index=0, key="run_choice_export")
            run = load_run(pp.runs_dir / run_choice)

            scope = st.selectbox("Scope", ["all", "verified only", "flagged only", "unverified only"], index=0, key="export_scope")

            outputs: dict = run.get("outputs", {}) or {}
            validation: dict = run.get("validation", {}) or {}

            mf = load_files_manifest(pp)
            files = {f["file_id"]: f for f in mf.get("files", []) if f.get("file_id")}

            rows: list[dict] = []
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
                ensure_dir(csv_path.parent)
                csv_path.write_bytes(csv_bytes)

                json_obj = {"project": pp.root.name, "run_id": run.get("run_id"), "scope": scope, "exported_at": now_iso(), "rows": rows}
                json_path = pp.exports_dir / f"{base_name}.json"
                write_json(json_path, json_obj)

                append_jsonl(pp.audit, {"ts": now_iso(), "action": "export.create", "project": pp.root.name, "run_id": run.get("run_id"), "scope": scope, "export_id": export_id})

                c1, c2 = st.columns(2)
                with c1:
                    st.download_button("Download CSV", data=csv_bytes, file_name=csv_path.name, mime="text/csv", key="dl_csv")
                    st.caption(f"Saved to: {csv_path}")
                with c2:
                    st.download_button("Download JSON", data=json.dumps(json_obj, indent=2).encode("utf-8"), file_name=json_path.name, mime="application/json", key="dl_json")
                    st.caption(f"Saved to: {json_path}")

else:
    st.info("Pick a page from the sidebar.")
