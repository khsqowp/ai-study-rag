import json
import os
import subprocess
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from qdrant_client import QdrantClient

from answer import build_prompt, call_gemini, retrieve, stream_gemini
from index_pdfs import collection_for_project


_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


app = FastAPI(title="AI Sandbox RAG")


class AskRequest(BaseModel):
    query: str
    project: str = "inbox"
    limit: int = 12
    show_context: bool = True


class ReindexRequest(BaseModel):
    project: str = "inbox"
    recreate: bool = True


def project_names() -> list[str]:
    root = Path("/sandbox/data")
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def ensure_project(project: str) -> str:
    if project not in project_names():
        raise HTTPException(status_code=404, detail=f"Unknown project: {project}")
    return project


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return Path("/sandbox/scripts/web_index.html").read_text(encoding="utf-8")


@app.get("/api/status")
def status(project: str = "inbox") -> dict:
    project = ensure_project(project)
    collection = collection_for_project(project)
    client = QdrantClient(url=os.environ["QDRANT_URL"])
    if not client.collection_exists(collection):
        return {
            "project": project,
            "collection": collection,
            "points_count": 0,
            "status": "missing",
            "projects": project_names(),
        }
    collection_info = client.get_collection(collection)
    return {
        "project": project,
        "collection": collection,
        "points_count": collection_info.points_count,
        "status": str(collection_info.status),
        "projects": project_names(),
    }


@app.post("/api/ask")
def ask(request: AskRequest) -> dict:
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is required")
    project = ensure_project(request.project)

    contexts = retrieve(query, request.limit, max(50, request.limit * 6), project)
    if not contexts:
        raise HTTPException(status_code=404, detail="No relevant context found")

    prompt = build_prompt(query, contexts)
    try:
        answer = call_gemini(prompt)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return {
        "answer": answer,
        "contexts": contexts if request.show_context else [],
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

    prompt = build_prompt(query, contexts)

    def generate():
        yield f"data: {json.dumps({'type': 'contexts', 'contexts': contexts})}\n\n"
        try:
            for chunk in stream_gemini(prompt):
                yield f"data: {json.dumps({'type': 'token', 'text': chunk})}\n\n"
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
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    with _jobs_lock:
        _jobs[job_id] = {
            "status": "completed" if result.returncode == 0 else "failed",
            "ok": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }


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
