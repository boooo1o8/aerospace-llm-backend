"""Start/stop this account's loopback execution API; never signal unrelated PIDs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aerospace_backend.core import Settings, exclusive, matches, process_info, stop_group


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["start", "stop", "status", "restart"])
    parser.add_argument("--config", default=os.environ.get("AEROSPACE_CONFIG"))
    args = parser.parse_args()
    if not args.config:
        parser.error("--config or AEROSPACE_CONFIG is required")
    os.umask(0o077)
    settings = Settings(args.config)
    pidfile = settings.root / "api-process.json"
    host, port = "127.0.0.1", settings.data.get("port", 8010)
    url = f"http://{host}:{port}"
    marker = "aerospace_backend.app:create_app"
    def current():
        try:
            p = json.loads(pidfile.read_text())
            return p if matches(p["pid"], p["start"], marker) else None
        except (OSError, ValueError):
            return None
    def healthy():
        try:
            with urllib.request.urlopen(url + "/healthz", timeout=2) as response:
                return json.load(response).get("status") == "ok"
        except Exception:
            return False
    with exclusive(settings.root / "locks/service.lock"):
        process = current()
        if args.command == "status":
            result = {"running": bool(process), "healthy": healthy(), "url": url, "pid": process["pid"] if process else None}
            print(json.dumps(result))
            return 0 if result["running"] and result["healthy"] else 1
        if args.command in {"stop", "restart"} and process:
            stop_group(process["pid"], process["start"], marker, grace=10)
            pidfile.unlink(missing_ok=True)
            process = None
        if args.command == "stop":
            print("API stopped; detached executions continue and can be queried after restart")
            return 0
        if process:
            print("Already running", url)
            return 0 if healthy() else 1
        env = os.environ.copy()
        env["AEROSPACE_CONFIG"] = str(settings.path)
        with (settings.root / "api.log").open("ab") as log:
            child = subprocess.Popen([settings.data["python"], "-m", "uvicorn", marker, "--factory",
                                      "--host", host, "--port", str(port), "--workers", "1", "--no-access-log"],
                cwd=str(Path(__file__).resolve().parents[1]), env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        info = process_info(child.pid)
        if not info:
            raise RuntimeError("API process exited before startup")
        pidfile.write_text(json.dumps({"pid": child.pid, "start": info["start"]}))
        for _ in range(40):
            if child.poll() is not None:
                raise RuntimeError("API exited; inspect private api.log")
            if healthy():
                print("Started", url, "PID", child.pid)
                return 0
            time.sleep(0.25)
        raise RuntimeError("API did not become healthy; inspect private api.log")


if __name__ == "__main__":
    sys.exit(main())
