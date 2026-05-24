import os
import queue
import subprocess
import threading
import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers.polling import PollingObserver

from settings import get_data_root

DEBOUNCE_SECONDS = 30

_index_queue: queue.Queue = queue.Queue()
_pending: dict[str, float] = {}
_pending_lock = threading.Lock()


def project_names() -> list[str]:
    configured = os.environ.get("PROJECT_NAME", "").strip()
    if configured and configured != "*":
        return [configured]
    root = get_data_root()
    if not root.exists():
        return []
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def run_index(project: str) -> None:
    try:
        subprocess.run(
            [
                "python",
                "/sandbox/scripts/index_pdfs.py",
                "--project",
                project,
                "--skip-unchanged",
            ],
            check=False,
        )
    except Exception as exc:
        print(f"[watcher] indexing error for {project!r}: {exc}", flush=True)


class ProjectEventHandler(FileSystemEventHandler):
    def __init__(self, project: str) -> None:
        self.project = project

    def _maybe_enqueue(self) -> None:
        with _pending_lock:
            _pending[self.project] = time.monotonic()

    def on_created(self, event) -> None:
        self._maybe_enqueue()

    def on_modified(self, event) -> None:
        self._maybe_enqueue()

    def on_moved(self, event) -> None:
        self._maybe_enqueue()

    def on_deleted(self, event) -> None:
        self._maybe_enqueue()


def _debounce_worker() -> None:
    while True:
        time.sleep(5)
        now = time.monotonic()
        with _pending_lock:
            ready = [p for p, t in _pending.items() if now - t >= DEBOUNCE_SECONDS]
            for p in ready:
                del _pending[p]
        for project in ready:
            print(f"[watcher] enqueuing {project!r}", flush=True)
            _index_queue.put(project)


def _index_worker() -> None:
    while True:
        project = _index_queue.get()
        print(f"[watcher] indexing {project!r}", flush=True)
        run_index(project)
        _index_queue.task_done()


_watched_projects: set[tuple[str, str]] = set()
_watched_lock = threading.Lock()


def _watch_project(observer: PollingObserver, project: str) -> None:
    root = get_data_root()
    project_dir = root / project
    if not project_dir.is_dir():
        return
    watch_key = (str(root), project)
    with _watched_lock:
        if watch_key in _watched_projects:
            return
        _watched_projects.add(watch_key)
    handler = ProjectEventHandler(project)
    observer.schedule(handler, str(project_dir), recursive=True)
    print(f"[watcher] watching {project_dir}", flush=True)
    _index_queue.put(project)


def _new_project_scanner(observer: PollingObserver) -> None:
    while True:
        time.sleep(60)
        current_root = get_data_root()
        with _watched_lock:
            stale = [key for key in _watched_projects if key[0] != str(current_root) or not (current_root / key[1]).is_dir()]
            for key in stale:
                _watched_projects.remove(key)
        for project in project_names():
            with _watched_lock:
                already = (str(get_data_root()), project) in _watched_projects
            if not already:
                print(f"[watcher] new project detected: {project!r}", flush=True)
                _watch_project(observer, project)


def main() -> None:
    observer = PollingObserver(timeout=10)
    for project in project_names():
        _watch_project(observer, project)

    threading.Thread(target=_debounce_worker, daemon=True).start()
    threading.Thread(target=_index_worker, daemon=True).start()

    observer.start()
    threading.Thread(target=_new_project_scanner, args=(observer,), daemon=True).start()

    print("[watcher] started", flush=True)
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
