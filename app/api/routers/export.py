from __future__ import annotations

import base64
import csv
import io
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/api/exports", tags=["exports"])

DATA_ROOT = Path(__file__).resolve().parents[3] / "app" / "data" / "projects"


# ---- Helpers ----

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


# ---- JSON export endpoint ----

@router.get("/json/{run_id}")
def export_json(run_id: str):
    """Return a JSON export of the run: file name, model, and per-question answers with confidence."""
    _, run_record = find_run(run_id)

    model = run_record.get("model", "")
    file_meta = {fe["file_id"]: fe for fe in run_record.get("files", []) if fe.get("file_id")}
    outputs: dict = run_record.get("outputs") or {}

    files_out = []
    for fid, answers in outputs.items():
        fe = file_meta.get(fid, {})
        file_name = fe.get("fileName", fid)
        fields = [
            {
                "question": q,
                "answer": v.get("value") if isinstance(v, dict) else v,
                "confidence": v.get("confidence") if isinstance(v, dict) else None,
            }
            for q, v in answers.items()
        ]
        files_out.append({"fileName": file_name, "fields": fields})

    return JSONResponse(content={"run_id": run_id, "model": model, "files": files_out})


# ---- CSV export endpoint ----

@router.get("/csv/{run_id}")
def export_csv(run_id: str):
    """Return a CSV export of the run: one row per file×question with answer and confidence."""
    _, run_record = find_run(run_id)

    model = run_record.get("model", "")
    file_meta = {fe["file_id"]: fe for fe in run_record.get("files", []) if fe.get("file_id")}
    outputs: dict = run_record.get("outputs") or {}

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["fileName", "model", "question", "answer", "confidence"])

    for fid, answers in outputs.items():
        file_name = file_meta.get(fid, {}).get("fileName", fid)
        for q, v in answers.items():
            answer = v.get("value") if isinstance(v, dict) else v
            confidence = v.get("confidence") if isinstance(v, dict) else None
            writer.writerow([file_name, model, q, answer, confidence])

    csv_bytes = buf.getvalue().encode("utf-8")
    return JSONResponse(content={
        "run_id": run_id,
        "filename": f"{run_id}.csv",
        "contentType": "text/csv",
        "contentBase64": base64.b64encode(csv_bytes).decode("utf-8"),
    })
