import json
import subprocess
import threading
import uuid


_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


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

    def _read_stderr() -> None:
        assert proc.stderr
        proc.stderr.read()

    stderr_thread = threading.Thread(target=_read_stderr, daemon=True)
    stderr_thread.start()

    files: list[dict] = []
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
                job["total_chunks"] += ev["chunks"]
                for f in files:
                    if f["file_index"] == ev["file_index"]:
                        f["status"] = "done"
                        f["chunks"] = ev["chunks"]
                        break
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


def start_reindex(project: str, recreate: bool) -> dict:
    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {"status": "running"}
    threading.Thread(target=_run_reindex_job, args=(job_id, project, recreate), daemon=True).start()
    return {"job_id": job_id, "status": "running"}


def get_reindex_job(job_id: str) -> dict | None:
    with _jobs_lock:
        return _jobs.get(job_id)
