from __future__ import annotations

import base64
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import re
import signal
import sqlite3
import subprocess
import time
import uuid

import fcntl
from jsonschema import Draft202012Validator

TERMINAL = {"succeeded", "failed", "cancelled"}
IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
PUBLIC_RUN = {"run_id", "business_task_id", "kind", "status", "phase", "created_at",
              "started_at", "finished_at", "progress", "metrics", "artifact_ids",
              "output_model_id", "result_artifact_id", "failure", "status_url"}


def now():
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix):
    return prefix + "_" + uuid.uuid4().hex


def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(data):
    return hashlib.sha256(data).hexdigest()


class ServiceError(Exception):
    def __init__(self, status, code, message, details=None):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message
        self.details = details or {}


def invalid(message, details=None):
    raise ServiceError(422, "invalid_request", message, details)


def validate_id(value):
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        invalid("Invalid identifier")
    return value


def finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        invalid("NaN and Infinity are not valid input values")
    if isinstance(value, dict):
        for v in value.values():
            finite(v)
    if isinstance(value, list):
        for v in value:
            finite(v)


class Settings:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.data = json.loads(self.path.read_text(encoding="utf-8"))
        for key in ("data_root", "model_root", "credentials_file", "cursor_key_file", "python"):
            if not Path(self.data[key]).is_absolute():
                raise ValueError(key + " must be an absolute path")
        self.root = Path(self.data["data_root"]).resolve()
        self.models = Path(self.data["model_root"]).resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.models.mkdir(parents=True, exist_ok=True, mode=0o700)
        for folder in ("files", "runs", "locks", "artifacts"):
            (self.root / folder).mkdir(exist_ok=True, mode=0o700)
        self.credentials = json.loads(Path(self.data["credentials_file"]).read_text())
        self.cursor_key = Path(self.data["cursor_key_file"]).read_bytes().strip()
        if len(self.cursor_key) < 32:
            raise ValueError("cursor key must have at least 32 bytes")
        if self.data.get("test_engine") and os.environ.get("AEROSPACE_TESTING") != "1":
            raise ValueError("test engine is disabled outside explicit test mode")
        self.max_upload = self.data.get("max_upload_bytes", 8 * 1024 * 1024)
        self.contract = json.loads((Path(__file__).resolve().parents[1] / "contracts/v1/execution.openapi.json").read_text(encoding="utf-8"))
        self.db = Database(self.root / "state.sqlite3")

    def validate(self, name, value):
        finite(value)
        schema = {"$ref": "#/components/schemas/" + name, "components": self.contract["components"]}
        errors = sorted(Draft202012Validator(schema).iter_errors(value), key=lambda e: str(e.path))
        if errors:
            invalid("Request does not match " + name, {"fields": ["/".join(map(str, e.path)) for e in errors[:8]]})

    def authenticate(self, token, project, actor):
        validate_id(project)
        validate_id(actor)
        digest = sha(token.encode())
        for entry in self.credentials["services"]:
            if hmac.compare_digest(digest, entry["token_sha256"]):
                if project not in entry["projects"] or actor not in entry["actors"]:
                    raise ServiceError(403, "forbidden", "Caller is not authorized for this project or actor")
                return {"service": entry["service_id"], "project": project, "actor": actor}
        raise ServiceError(401, "unauthenticated", "Invalid service credential")


class Database:
    def __init__(self, path):
        self.path = str(path)
        with self.connect() as con:
            con.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS objects(kind TEXT,id TEXT PRIMARY KEY,project TEXT,payload TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS object_scope ON objects(kind,project);
            CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY,project TEXT,payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS idempotency(scope TEXT,key TEXT,hash TEXT,run_id TEXT,PRIMARY KEY(scope,key));
            CREATE TABLE IF NOT EXISTS logs(seq INTEGER PRIMARY KEY AUTOINCREMENT,run_id TEXT,stamp TEXT,level TEXT,message TEXT);
            CREATE INDEX IF NOT EXISTS log_run ON logs(run_id,seq);
            """)

    @contextmanager
    def connect(self, write=False):
        con = sqlite3.connect(self.path, timeout=30)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA busy_timeout=30000")
        if write:
            con.execute("BEGIN IMMEDIATE")
        try:
            yield con
            con.commit()
        except BaseException:
            con.rollback()
            raise
        finally:
            con.close()

    def put(self, kind, id_, project, value, con=None):
        if con is None:
            with self.connect(write=True) as tx:
                return self.put(kind, id_, project, value, tx)
        con.execute("INSERT INTO objects VALUES(?,?,?,?)", (kind, id_, project, dumps(value)))

    def get(self, kind, id_, project, con=None):
        if con is None:
            with self.connect() as tx:
                return self.get(kind, id_, project, tx)
        row = con.execute("SELECT payload FROM objects WHERE kind=? AND id=? AND project IN (?, '*')", (kind, id_, project)).fetchone()
        if not row:
            raise ServiceError(404, "not_found", "Resource is not available in this project")
        return json.loads(row[0])

    def objects(self, kind, project):
        with self.connect() as con:
            return [json.loads(r[0]) for r in con.execute("SELECT payload FROM objects WHERE kind=? AND project IN (?, '*') ORDER BY id", (kind, project))]

    def run(self, id_, project=None, con=None):
        if con is None:
            with self.connect() as tx:
                return self.run(id_, project, tx)
        row = con.execute("SELECT project,payload FROM runs WHERE id=?", (id_,)).fetchone()
        if not row or (project is not None and row[0] != project):
            raise ServiceError(404, "not_found", "Run is not available in this project")
        return json.loads(row[1])

    def update(self, id_, mutate):
        with self.connect(write=True) as con:
            run = self.run(id_, con=con)
            mutate(run)
            con.execute("UPDATE runs SET payload=? WHERE id=?", (dumps(run), id_))
            return run

    def active(self, con=None):
        if con is None:
            with self.connect() as tx:
                return self.active(tx)
        return [r for row in con.execute("SELECT payload FROM runs") if (r := json.loads(row[0]))["status"] not in TERMINAL]

    def log(self, id_, message, level="INFO"):
        with self.connect(write=True) as con:
            con.execute("INSERT INTO logs(run_id,stamp,level,message) VALUES(?,?,?,?)", (id_, now(), level, message[:8000]))


def public_run(run):
    return {k: v for k, v in run.items() if k in PUBLIC_RUN}


def public_object(obj):
    return {k: v for k, v in obj.items() if not k.startswith("_")}


def read_file(settings, file):
    data = Path(file["_path"]).read_bytes()
    if sha(data) != file["sha256"]:
        raise ServiceError(409, "checksum_mismatch", "Registered input file has changed")
    return data


def read_records(data, format_):
    try:
        records = json.loads(data.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        invalid("Dataset must be a UTF-8 JSON array")
    finite(records)
    if not isinstance(records, list) or not 1 <= len(records) <= 100000:
        invalid("Dataset must contain 1 to 100000 records")
    ids, signatures = set(), set()
    allowed = {"instruction", "input", "output", "system"} if format_ == "alpaca" else {"question_id", "prompt", "system", "answer", "question_type"}
    required = ("instruction", "output") if format_ == "alpaca" else ("question_id", "prompt")
    for index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) - allowed:
            invalid("Unsupported record fields", {"record": index})
        for field in required:
            if not isinstance(record.get(field), str) or not record[field].strip():
                invalid("Missing or empty required text", {"record": index, "field": field})
        if any(not isinstance(v, str) or len(v) > 32768 for v in record.values()):
            invalid("Record text must be a string of at most 32768 characters", {"record": index})
        if format_ == "question_set_v1":
            validate_id(record["question_id"])
            if record["question_id"] in ids:
                invalid("Duplicate question_id", {"record": index})
            ids.add(record["question_id"])
            signature = record["prompt"].strip()
        else:
            signature = record["instruction"].strip() + "\n" + record.get("input", "").strip()
        digest = sha(signature.encode())
        if digest in signatures:
            invalid("Duplicate input in the same split", {"record": index})
        signatures.add(digest)
    return records, signatures


def cursor_encode(settings, project, run_id, seq):
    body = base64.urlsafe_b64encode(dumps([project, run_id, seq]).encode()).rstrip(b"=")
    return (body + b"." + hmac.new(settings.cursor_key, body, hashlib.sha256).hexdigest().encode()).decode()


def cursor_decode(settings, value, project, run_id, integer=True):
    if not value:
        return 0
    try:
        body, signature = value.encode().split(b".")
        if not hmac.compare_digest(hmac.new(settings.cursor_key, body, hashlib.sha256).hexdigest().encode(), signature):
            raise ValueError()
        scope, id_, seq = json.loads(base64.urlsafe_b64decode(body + b"=" * (-len(body) % 4)))
        if (scope, id_) != (project, run_id):
            raise ValueError()
        if integer and (type(seq) is not int or seq < 0):
            raise ValueError()
        if not integer and (not isinstance(seq, str) or not IDENTIFIER.fullmatch(seq)):
            raise ValueError()
        return seq
    except (ValueError, TypeError, UnicodeError):
        invalid("Invalid log cursor")


@contextmanager
def exclusive(path):
    with open(path, "a+b") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ServiceError(503, "resource_busy", "Execution resource is already reserved")
        try:
            yield handle
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def check_gpu(settings, assignment):
    if assignment["node_id"] != settings.data["node_id"] or assignment["gpu_ids"] != [settings.data["gpu_id"]]:
        raise ServiceError(403, "resource_not_allowed", "Only the configured single GPU is available to this service")
    if settings.data.get("test_engine"):
        return
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.free", "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True, timeout=5)
        rows = [r.split(",") for r in result.stdout.strip().splitlines()]
        selected = next(r for r in rows if r[0].strip() == settings.data["gpu_id"])
        gpu_uuid, free = selected[1].strip(), int(selected[2])
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True, timeout=5)
        occupied = any(r.split(",")[0].strip() == gpu_uuid for r in apps.stdout.splitlines())
        if occupied or free < settings.data.get("min_free_gpu_mib", 7000):
            raise ServiceError(503, "gpu_busy", "Configured GPU is occupied or has insufficient free memory", {"gpu_id": settings.data["gpu_id"], "free_mib": free})
    except ServiceError:
        raise
    except (OSError, subprocess.SubprocessError, StopIteration, ValueError):
        raise ServiceError(503, "gpu_unavailable", "Cannot verify configured GPU availability")


def process_info(pid):
    try:
        proc = Path("/proc") / str(pid)
        text = (proc / "stat").read_text()
        fields = text[text.rfind(")") + 2:].split()
        if fields[0] == "Z" or proc.stat().st_uid != os.getuid():
            return None
        return {"start": fields[19], "args": (proc / "cmdline").read_bytes().split(b"\0")}
    except (OSError, ValueError):
        return None


def matches(pid, start, marker):
    info = process_info(pid)
    return bool(info and info["start"] == str(start) and marker.encode() in info["args"])


def stop_group(pid, start, marker, grace=5):
    if not matches(pid, start, marker):
        return
    if os.getpgid(pid) != pid:
        raise RuntimeError("Refusing to signal an unowned process group")
    os.killpg(pid, signal.SIGTERM)
    deadline = time.monotonic() + grace
    while matches(pid, start, marker) and time.monotonic() < deadline:
        time.sleep(0.1)
    if matches(pid, start, marker):
        os.killpg(pid, signal.SIGKILL)
    deadline = time.monotonic() + 5
    while matches(pid, start, marker) and time.monotonic() < deadline:
        time.sleep(0.1)
    if matches(pid, start, marker):
        raise RuntimeError("Engine process did not stop")


def scrub(settings, text):
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    text = re.sub(r"(?i)(bearer\s+)[^\s\"']+", r"\1[REDACTED]", text)
    text = re.sub(r"(?i)((?:api[_-]?key|password|access[_-]?token)\s*[:=]\s*)[^\s,]+", r"\1[REDACTED]", text)
    for key in ("data_root", "model_root", "llamafactory_root"):
        if settings.data.get(key):
            text = text.replace(settings.data[key], "[managed-storage]")
    return text
