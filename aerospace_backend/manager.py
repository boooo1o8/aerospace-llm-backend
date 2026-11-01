import os
import subprocess
import threading
import time

from .core import ServiceError, exclusive, matches, now, process_info, stop_group


class Manager:
    """Single API manager; detached per-run workers survive API restarts."""

    def __init__(self, settings):
        self.settings, self.db = settings, settings.db
        self.halt = threading.Event()
        self.thread = threading.Thread(target=self.loop, name="execution-manager", daemon=True)
        self.children = []

    def start(self):
        self.thread.start()

    def stop(self):
        self.halt.set()
        self.thread.join(timeout=15)
        # Keep Popen handles alive/reaped while the hosting process remains alive
        # (notably TestClient restarts). Do not interrupt detached execution.
        children, self.children = self.children[:], []
        if children:
            threading.Thread(target=lambda: [child.wait() for child in children], daemon=True,
                             name="execution-reaper").start()

    def loop(self):
        while not self.halt.is_set():
            for process in self.children[:]:
                if process.poll() is not None:
                    self.children.remove(process)
            try:
                for run in self.db.active():
                    self.tick(run)
            except Exception as exc:
                # Retry transient database/filesystem errors; never fabricate success.
                print("Execution manager error:", type(exc).__name__, flush=True)
            self.halt.wait(0.3)

    def tick(self, run):
        id_ = run["run_id"]
        if run["_worker_pid"] and matches(run["_worker_pid"], run["_worker_start"], id_):
            return
        # The engine inherits the run/GPU leases. Reap a verified orphan before
        # attempting the run lease, otherwise a dead worker would block recovery.
        if run["_worker_pid"] and run["_engine_pid"]:
            marker = str(self.settings.root / "runs" / id_ / "engine.json")
            stop_group(run["_engine_pid"], run["_engine_start"], marker)
        try:
            with exclusive(self.settings.root / "locks" / (id_ + ".lock")):
                if run["_engine_pid"]:
                    marker = str(self.settings.root / "runs" / id_ / "engine.json")
                    stop_group(run["_engine_pid"], run["_engine_start"], marker)
                if run["_worker_pid"]:
                    # A worker died. It is unsafe to silently restart partial training.
                    def interrupt(r):
                        if r["status"] not in {"succeeded", "failed", "cancelled"}:
                            cancelled = r["status"] == "cancel_requested"
                            r.update(status="cancelled" if cancelled else "failed", phase="interrupted", finished_at=now())
                            r["failure"] = None if cancelled else {"code": "worker_interrupted", "message": "Execution worker exited unexpectedly; create a new run to retry", "request_id": r["_request_id"]}
                    self.db.update(id_, interrupt)
                    self.db.log(id_, "Recovered interrupted worker; owned engine stopped and execution marked terminal", "WARNING")
                    return
                if run["status"] == "cancel_requested":
                    self.db.update(id_, lambda r: r.update(status="cancelled", phase="cancelled", finished_at=now()))
                    return
                folder = self.settings.root / "runs" / id_
                folder.mkdir(exist_ok=True, mode=0o700)
                with (folder / "worker.log").open("ab") as log:
                    process = subprocess.Popen([self.settings.data["python"], "-m", "aerospace_backend.worker", str(self.settings.path), id_],
                        cwd=str(__import__("pathlib").Path(__file__).resolve().parents[1]),
                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                        env=os.environ.copy())
                self.children.append(process)
                info = process_info(process.pid)
                if info:
                    def remember(r):
                        if not r["_worker_pid"]:
                            r.update(_worker_pid=process.pid, _worker_start=info["start"])
                    self.db.update(id_, remember)
        except ServiceError as exc:
            if exc.code != "resource_busy":
                raise
