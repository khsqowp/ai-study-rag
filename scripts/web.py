import json
import os
import shutil
import subprocess
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from qdrant_client import QdrantClient

from answer import build_prompt, call_gemini, retrieve, stream_gemini
from index_pdfs import SUPPORTED_EXTENSIONS, collection_for_project


_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


app = FastAPI(title="AI Sandbox RAG")


class AskRequest(BaseModel):
    query: str
    project: str = "inbox"
    limit: int = 12
    show_context: bool = True
    web_search: bool = False


class ReindexRequest(BaseModel):
    project: str = "inbox"
    recreate: bool = True


class ProjectCreateRequest(BaseModel):
    name: str


class FolderCreateRequest(BaseModel):
    project: str
    path: str = ""
    name: str


DATA_ROOT = Path("/sandbox/data")


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def _safe_name(name: str) -> str:
    cleaned = name.strip()
    if not cleaned or cleaned in {".", ".."} or "/" in cleaned or "\\" in cleaned:
        raise HTTPException(status_code=400, detail="Invalid name")
    return cleaned


def _safe_project_path(project: str, relative_path: str = "") -> Path:
    project = _safe_name(project)
    root = DATA_ROOT.resolve()
    base = (root / project).resolve()
    if not _is_relative_to(base, root):
        raise HTTPException(status_code=400, detail="Invalid project")
    rel = relative_path.strip().strip("/")
    target = (base / rel).resolve() if rel else base
    if not _is_relative_to(target, base):
        raise HTTPException(status_code=400, detail="Invalid path")
    return target


def _relative_to_project(project: str, path: Path) -> str:
    base = (DATA_ROOT.resolve() / project).resolve()
    rel = path.resolve().relative_to(base)
    return "" if str(rel) == "." else str(rel)


def project_names() -> list[str]:
    root = DATA_ROOT
    if not root.exists():
        return []
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def ensure_project(project: str) -> str:
    if project not in project_names():
        raise HTTPException(status_code=404, detail=f"Unknown project: {project}")
    return project


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return Path("/sandbox/scripts/web_index.html").read_text(encoding="utf-8")


@app.get("/api/status")
def status(project: str = "") -> dict:
    all_projects = project_names()
    if not all_projects:
        return {"project": None, "collection": None, "points_count": 0, "status": "no_projects", "projects": []}
    if not project or project not in all_projects:
        project = all_projects[0]
    collection = collection_for_project(project)
    client = QdrantClient(url=os.environ["QDRANT_URL"])
    if not client.collection_exists(collection):
        return {
            "project": project,
            "collection": collection,
            "points_count": 0,
            "status": "missing",
            "projects": all_projects,
        }
    collection_info = client.get_collection(collection)
    return {
        "project": project,
        "collection": collection,
        "points_count": collection_info.points_count,
        "status": str(collection_info.status),
        "projects": all_projects,
    }


@app.get("/api/files")
def list_files(project: str, path: str = "") -> dict:
    project = ensure_project(project)
    target = _safe_project_path(project, path)
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {path}")
    if not target.is_dir():
        raise HTTPException(status_code=400, detail="Path is not a directory")

    items = []
    for child in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
        stat = child.stat()
        items.append(
            {
                "name": child.name,
                "path": _relative_to_project(project, child),
                "type": "directory" if child.is_dir() else "file",
                "size": stat.st_size,
                "mtime": stat.st_mtime,
                "supported": child.is_dir() or child.suffix.lower() in SUPPORTED_EXTENSIONS,
            }
        )
    return {
        "project": project,
        "path": _relative_to_project(project, target),
        "items": items,
    }


@app.post("/api/projects")
def create_project(request: ProjectCreateRequest) -> dict:
    name = _safe_name(request.name)
    target = (DATA_ROOT.resolve() / name).resolve()
    if not _is_relative_to(target, DATA_ROOT.resolve()):
        raise HTTPException(status_code=400, detail="Invalid project")
    try:
        target.mkdir(parents=False, exist_ok=False)
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=f"Project already exists: {name}") from exc
    return {"project": name}


@app.post("/api/files/folder")
def create_folder(request: FolderCreateRequest) -> dict:
    project = ensure_project(request.project)
    parent = _safe_project_path(project, request.path)
    if not parent.exists() or not parent.is_dir():
        raise HTTPException(status_code=404, detail="Parent folder not found")
    name = _safe_name(request.name)
    target = (parent / name).resolve()
    if not _is_relative_to(target, (DATA_ROOT.resolve() / project).resolve()):
        raise HTTPException(status_code=400, detail="Invalid folder")
    try:
        target.mkdir(parents=False, exist_ok=False)
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=f"Folder already exists: {name}") from exc
    return {"path": _relative_to_project(project, target)}


@app.post("/api/files/upload")
def upload_files(
    project: str = Form(...),
    path: str = Form(""),
    files: list[UploadFile] = File(...),
) -> dict:
    project = ensure_project(project)
    target_dir = _safe_project_path(project, path)
    if not target_dir.exists() or not target_dir.is_dir():
        raise HTTPException(status_code=404, detail="Upload folder not found")
    uploaded = []
    for upload in files:
        filename = _safe_name(upload.filename or "")
        target = (target_dir / filename).resolve()
        if not _is_relative_to(target, (DATA_ROOT.resolve() / project).resolve()):
            raise HTTPException(status_code=400, detail=f"Invalid filename: {filename}")
        with target.open("wb") as out:
            shutil.copyfileobj(upload.file, out)
        uploaded.append({"name": filename, "path": _relative_to_project(project, target)})
    return {"uploaded": uploaded}


@app.post("/api/ask")
def ask(request: AskRequest) -> dict:
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is required")
    project = ensure_project(request.project)

    contexts = retrieve(query, request.limit, max(50, request.limit * 6), project)
    if not contexts:
        raise HTTPException(status_code=404, detail="No relevant context found")

    prompt = build_prompt(query, contexts, request.web_search)
    try:
        answer, web_sources = call_gemini(prompt, request.web_search)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return {
        "answer": answer,
        "contexts": contexts if request.show_context else [],
        "web_sources": web_sources,
    }


@app.post("/api/ask/stream")
def ask_stream(request: AskRequest) -> StreamingResponse:
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is required")
    project = ensure_project(request.project)

    contexts = retrieve(query, request.limit, max(50, request.limit * 6), project)
    if not contexts:
        raise HTTPException(status_code=404, detail="No relevant context found")

    prompt = build_prompt(query, contexts, request.web_search)

    def generate():
        yield f"data: {json.dumps({'type': 'contexts', 'contexts': contexts})}\n\n"
        try:
            for event in stream_gemini(prompt, request.web_search):
                if event["type"] == "token":
                    yield f"data: {json.dumps({'type': 'token', 'text': event['text']})}\n\n"
                elif event["type"] == "grounding":
                    yield f"data: {json.dumps({'type': 'web_sources', 'sources': event['sources']})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'detail': str(exc)})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


def _run_reindex_job(job_id: str, project: str, recreate: bool) -> None:
    command = [
        "python",
        "/sandbox/scripts/index_pdfs.py",
        "--project",
        project,
        "--skip-unchanged",
    ]
    if recreate:
        command.append("--recreate")

    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    def _read_stderr():
        assert proc.stderr
        proc.stderr.read()  # drain stderr (tqdm progress bars) to prevent pipe deadlock

    stderr_thread = threading.Thread(target=_read_stderr, daemon=True)
    stderr_thread.start()

    files: list[dict] = []  # [{filename, source, status, chunks, error}]
    state: dict = {
        "status": "running",
        "total_files": 0,
        "total_deleted": 0,
        "current_file_index": 0,
        "current_file": None,
        "files_done": 0,
        "files_failed": 0,
        "files_skipped": 0,
        "files_deleted": 0,
        "total_chunks": 0,
        "files": files,
        "ok": None,
    }
    with _jobs_lock:
        _jobs[job_id] = state

    assert proc.stdout
    for raw in proc.stdout:
        line = raw.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue

        event = ev.get("event")
        with _jobs_lock:
            job = _jobs[job_id]
            if event == "start":
                job["total_files"] = ev["total_files"]
                job["total_deleted"] = ev["total_deleted"]
            elif event == "migration":
                job["migration"] = ev["message"]
            elif event == "no_changes":
                job["no_changes"] = True
            elif event == "no_files":
                job["no_files"] = True
            elif event == "delete":
                job["files_deleted"] = ev["index"]
            elif event == "file_start":
                job["current_file_index"] = ev["file_index"]
                job["current_file"] = ev["filename"]
                files.append({
                    "file_index": ev["file_index"],
                    "filename": ev["filename"],
                    "source": ev["source"],
                    "status": "indexing",
                    "chunks": None,
                    "error": None,
                })
            elif event == "file_done":
                job["files_done"] = ev["file_index"] - job["files_failed"] - job["files_skipped"]
                job["total_chunks"] += ev["chunks"]
                job["files_done"] += 1
                for f in files:
                    if f["file_index"] == ev["file_index"]:
                        f["status"] = "done"
                        f["chunks"] = ev["chunks"]
                        break
                # recount cleanly
                job["files_done"] = sum(1 for f in files if f["status"] == "done")
            elif event == "file_error":
                for f in files:
                    if f["file_index"] == ev["file_index"]:
                        f["status"] = "error"
                        f["error"] = ev["error"]
                        break
                job["files_failed"] = sum(1 for f in files if f["status"] == "error")
            elif event == "file_skip":
                for f in files:
                    if f["file_index"] == ev["file_index"]:
                        f["status"] = "skipped"
                        f["error"] = ev.get("reason")
                        break
                job["files_skipped"] = sum(1 for f in files if f["status"] == "skipped")
            elif event == "done":
                job["total_chunks"] = ev["total_chunks"]
                job["files_done"] = ev["files_done"]
                job["files_failed"] = ev["files_failed"]
                job["files_skipped"] = ev["files_skipped"]
                job["files_deleted"] = ev["files_deleted"]

    proc.wait()
    stderr_thread.join()

    with _jobs_lock:
        job = _jobs[job_id]
        no_changes = job.get("no_changes", False)
        no_files = job.get("no_files", False)
        ok = proc.returncode == 0
        if no_changes or no_files:
            ok = True
        job["status"] = "completed" if ok else "failed"
        job["ok"] = ok
        job["returncode"] = proc.returncode


@app.post("/api/reindex")
def reindex(request: ReindexRequest) -> dict:
    project = ensure_project(request.project)
    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {"status": "running"}
    threading.Thread(
        target=_run_reindex_job, args=(job_id, project, request.recreate), daemon=True
    ).start()
    return {"job_id": job_id, "status": "running"}


@app.get("/api/reindex/status/{job_id}")
def reindex_status(job_id: str) -> dict:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}")
    return job
