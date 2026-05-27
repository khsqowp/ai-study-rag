from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from reindex_jobs import get_reindex_job, start_reindex


app = FastAPI(title="AI Sandbox Indexer")


class ReindexRequest(BaseModel):
    project: str = "inbox"
    recreate: bool = True


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/api/reindex")
def reindex(request: ReindexRequest) -> dict:
    return start_reindex(request.project, request.recreate)


@app.get("/api/reindex/status/{job_id}")
def reindex_status(job_id: str) -> dict:
    job = get_reindex_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}")
    return job
