import json
import re
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from app.api.routers import convert_to_img, document_file, export, models, project, run_extract

DATA_ROOT = Path(__file__).resolve().parent / "data" / "projects"

# Maps URL path patterns to audit action names
_AUDIT_ACTIONS = {
    r"^/api/projects/(?P<project_id>p_[^/]+)/structures": "structure.add",
    r"^/api/projects/(?P<project_id>p_[^/]+)/data-standards": "standard.add",
    r"^/api/documents/upload": None,  # project_id comes from query param
}


def _try_write_validation_audit(request: Request, error_detail: str) -> None:
    try:
        path = request.url.path
        project_id = None
        action = None

        for pattern, act in _AUDIT_ACTIONS.items():
            m = re.match(pattern, path)
            if m:
                action = act
                groups = m.groupdict()
                project_id = groups.get("project_id") or request.query_params.get("project_id")
                break

        if not project_id or not action:
            return

        audit_path = DATA_ROOT / project_id / "audit.json"
        if not audit_path.parent.is_dir():
            return

        try:
            data = json.loads(audit_path.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                data = []
        except Exception:
            data = []

        data.append({
            "ts": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "action": action,
            "project_id": project_id,
            "status": "failed",
            "error": error_detail,
        })
        audit_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass

app = FastAPI()

app.include_router(models.router)
app.include_router(convert_to_img.router)
app.include_router(document_file.router)
app.include_router(project.router)
app.include_router(run_extract.router)
app.include_router(export.router)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    detail = str(exc.errors())
    _try_write_validation_audit(request, detail)
    return JSONResponse(status_code=422, content={"success": False, "error": "Validation error", "detail": exc.errors()})


@app.exception_handler(Exception)
async def generic_exception_handler(_request: Request, _exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": "Internal server error"},
    )


@app.get("/health")
async def health_check():
    return {"success": True, "message": "API is running"}



