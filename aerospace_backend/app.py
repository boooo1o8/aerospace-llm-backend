from __future__ import annotations

from contextlib import asynccontextmanager
import hashlib
import json
import os
from pathlib import Path
import sqlite3

from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from starlette.exceptions import HTTPException

from . import __version__
from .core import (Settings, ServiceError, TERMINAL, check_gpu, cursor_decode, cursor_encode,
                   dumps, exclusive, invalid, new_id, now, public_object, public_run,
                   read_file, read_records, sha, validate_id)
from .manager import Manager


def create_app(config_path=None):
    settings = Settings(config_path or os.environ["AEROSPACE_CONFIG"])
    db = settings.db

    @asynccontextmanager
    async def lifespan(app):
        with exclusive(settings.root / "locks/api.lock"):
            for model in settings.data["models"]:
                value = {"model_id": model["model_id"], "name": model["name"], "kind": "base",
                         "artifact_ids": [], "supported_methods": ["lora"], "max_context_tokens": 1024,
                         "base_model_id": None, "source_training_run_id": None,
                         "_path": model["path"], "_template": model["template"]}
                try:
                    db.put("model", value["model_id"], "*", value)
                except sqlite3.IntegrityError:
                    existing = db.get("model", value["model_id"], "*")
                    if existing != value:
                        raise RuntimeError("Registered base model changed; use a new model ID")
            manager = Manager(settings)
            app.state.manager = manager
            manager.start()
            try:
                yield
            finally:
                # Detached workers own their leases and continue across API restarts.
                manager.stop()

    app = FastAPI(title="航空测控后端执行服务", version=__version__, lifespan=lifespan)
    app.state.settings = settings

    @app.middleware("http")
    async def request_id(request, call_next):
        request.state.request_id = new_id("req")
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["Cache-Control"] = "no-store"
        return response

    def error_response(request, status, code, message, details=None):
        return JSONResponse(status_code=status, content={"error": {"code": code, "message": message,
            "request_id": request.state.request_id, "details": details or {}}})

    @app.exception_handler(ServiceError)
    async def service_error(request, exc):
        return error_response(request, exc.status, exc.code, exc.message, exc.details)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        return error_response(request, 422, "invalid_request", "Missing or invalid request fields",
                              {"fields": ["/".join(map(str, e["loc"])) for e in exc.errors()[:8]]})

    @app.exception_handler(HTTPException)
    async def http_error(request, exc):
        return error_response(request, exc.status_code, "http_error", str(exc.detail))

    @app.exception_handler(Exception)
    async def unexpected_error(request, exc):
        # No filesystem paths, credentials or exception input in the HTTP response.
        return error_response(request, 500, "internal_error", "Internal execution service error")

    def context(request: Request):
        authorization = request.headers.get("Authorization", "")
        if not authorization.startswith("Bearer ") or not authorization[7:]:
            raise ServiceError(401, "unauthenticated", "Service Bearer credential is required")
        project, actor = request.headers.get("X-Project-ID"), request.headers.get("X-Actor-ID")
        if not project or not actor:
            invalid("X-Project-ID and X-Actor-ID are required")
        return settings.authenticate(authorization[7:], project, actor)

    @app.get("/healthz", include_in_schema=False)
    def health():
        return {"status": "ok", "version": __version__}

    prefix = "/internal/v1"

    @app.get(prefix + "/capabilities")
    def capabilities(ctx=Depends(context)):
        return {"contract_version": __version__, "supported_methods": ["lora"],
                "max_upload_bytes": settings.max_upload, "max_log_page_size": 500,
                "idempotency_retention_seconds": 86400,
                "supported_dataset_formats": ["alpaca", "question_set_v1"],
                "native_inference_model_ids": settings.data.get("native_inference_model_ids", [])}

    @app.post(prefix + "/files", status_code=201)
    async def upload(file: UploadFile = File(...), purpose: str = Form(...), ctx=Depends(context)):
        if purpose not in {"source_document", "training_data", "evaluation_data", "prediction_data"}:
            invalid("Unsupported file purpose")
        filename = (file.filename or "upload").replace("\\", "/").split("/")[-1]
        filename = "".join(c for c in filename if ord(c) >= 32)[:160] or "upload"
        file_id, size, digest = new_id("file"), 0, hashlib.sha256()
        target = settings.root / "files" / file_id
        try:
            with target.open("xb") as handle:
                while chunk := await file.read(65536):
                    size += len(chunk)
                    if size > settings.max_upload:
                        raise ServiceError(413, "upload_too_large", "File exceeds the configured upload limit")
                    digest.update(chunk)
                    handle.write(chunk)
            if not size:
                invalid("Empty file")
            target.chmod(0o400)
            value = {"file_id": file_id, "purpose": purpose, "filename": filename,
                     "content_type": file.content_type or "application/octet-stream", "size_bytes": size,
                     "sha256": digest.hexdigest(), "created_at": now(), "_path": str(target)}
            db.put("file", file_id, ctx["project"], value)
            return public_object(value)
        except BaseException:
            target.unlink(missing_ok=True)
            raise
        finally:
            await file.close()

    @app.post(prefix + "/dataset-versions", status_code=201)
    def dataset_create(body: dict, ctx=Depends(context)):
        settings.validate("DatasetVersionCreate", body)
        if body["format"] not in {"alpaca", "question_set_v1"}:
            invalid("This implementation supports alpaca and question_set_v1 only")
        if len(body["version"]) > 128 or len(body.get("description", "")) > 2000:
            invalid("Dataset version or description is too long")
        if body["format"] == "question_set_v1" and "train" in body["splits"]:
            invalid("Question sets are for validation/test prediction, not training")
        splits, seen = {}, set()
        for split, id_ in body["splits"].items():
            item = db.get("file", id_, ctx["project"])
            allowed = {"training_data"} if split == "train" else {"evaluation_data", "prediction_data"}
            if item["purpose"] not in allowed:
                invalid("File purpose does not match split", {"split": split})
            records, signatures = read_records(read_file(settings, item), body["format"])
            if seen & signatures:
                invalid("Identical question/input appears in multiple splits")
            seen.update(signatures)
            splits[split] = {"file_id": id_, "sha256": item["sha256"], "sample_count": len(records)}
        value = {"dataset_version_id": new_id("data"), "dataset_id": body["dataset_id"],
                 "version": body["version"], "format": body["format"], "splits": splits, "created_at": now()}
        with db.connect(write=True) as con:
            existing = con.execute("SELECT payload FROM objects WHERE kind='dataset' AND project=?", (ctx["project"],))
            if any((r["dataset_id"], r["version"]) == (body["dataset_id"], body["version"]) for row in existing if (r := json.loads(row[0]))):
                raise ServiceError(409, "version_exists", "Dataset version is immutable; select a new version")
            db.put("dataset", value["dataset_version_id"], ctx["project"], value, con)
        return value

    @app.get(prefix + "/dataset-versions/{dataset_version_id}")
    def dataset_get(dataset_version_id: str, ctx=Depends(context)):
        return db.get("dataset", dataset_version_id, ctx["project"])

    @app.get(prefix + "/models")
    def models(cursor: str | None = None, limit: int = 100, ctx=Depends(context)):
        if not 1 <= limit <= 1000:
            invalid("Model page limit must be between 1 and 1000")
        after = cursor_decode(settings, cursor, ctx["project"], "model-catalog", integer=False) if cursor else ""
        items = [v for v in db.objects("model", ctx["project"]) if not v.get("source_training_run_id")
                 or db.run(v["source_training_run_id"], ctx["project"])["status"] == "succeeded"]
        items = [v for v in items if v["model_id"] > after]
        more = len(items) > limit
        items = items[:limit]
        return {"items": [public_object(v) for v in items],
                "next_cursor": cursor_encode(settings, ctx["project"], "model-catalog", items[-1]["model_id"]) if more else None}

    def validate_training(body, project):
        model = db.get("model", body["base_model_id"], project)
        dataset = db.get("dataset", body["dataset_version_id"], project)
        if model["kind"] != "base" or body["method"] != "lora":
            invalid("Only LoRA training from a registered base model is supported")
        if dataset["format"] != "alpaca" or "train" not in dataset["splits"]:
            invalid("Training requires an Alpaca dataset with a train split")
        p = body["hyperparameters"]
        limits = {"max_steps": (1, 1000), "num_train_epochs": (0.01, 20), "cutoff_len": (32, 1024),
                  "per_device_train_batch_size": (1, 2), "gradient_accumulation_steps": (1, 64),
                  "lora_rank": (1, 32), "learning_rate": (1e-8, 0.01)}
        for field, (lo, hi) in limits.items():
            if field in p and not lo <= p[field] <= hi:
                invalid("Parameter exceeds first-version resource limits", {"field": field, "minimum": lo, "maximum": hi})
        if p["compute_dtype"] != "float16" or p["attention"] != "sdpa" or "quantization_bit" in p:
            invalid("This GPU configuration currently permits FP16 + SDPA LoRA without quantization")
        # Bound epoch-only jobs by their computed optimizer-step upper bound.
        if "max_steps" not in p:
            import math
            steps = math.ceil(math.ceil(dataset["splits"]["train"]["sample_count"] / p["per_device_train_batch_size"]) / p["gradient_accumulation_steps"]) * p["num_train_epochs"]
            if steps > 1000:
                invalid("Epoch request exceeds 1000 update steps; use a bounded max_steps")
        for split in dataset["splits"].values():
            read_file(settings, db.get("file", split["file_id"], project))

    def validate_prediction(body, project):
        model = db.get("model", body["model_id"], project)
        if model.get("source_training_run_id") and db.run(model["source_training_run_id"], project)["status"] != "succeeded":
            raise ServiceError(409, "model_not_ready", "Model source execution has not completed successfully")
        dataset = db.get("dataset", body["dataset_version_id"], project)
        if dataset["format"] != "question_set_v1" or body["split"] not in dataset["splits"]:
            invalid("Prediction requires a question_set_v1 dataset containing the requested split")
        if body["generation"]["max_new_tokens"] > 256:
            invalid("max_new_tokens must be at most 256")
        split = dataset["splits"][body["split"]]
        read_file(settings, db.get("file", split["file_id"], project))

    def create_run(kind, body, request, ctx):
        settings.validate("TrainingCreate" if kind == "training" else "PredictionCreate", body)
        key = request.headers.get("Idempotency-Key", "")
        if not key or len(key) > 128 or any(ord(c) < 33 or ord(c) > 126 for c in key):
            invalid("A printable Idempotency-Key of 1 to 128 characters is required")
        scope = dumps([ctx["service"], ctx["project"], ctx["actor"], kind])
        fingerprint = sha(dumps(body).encode())
        with db.connect(write=True) as con:
            old = con.execute("SELECT hash,run_id FROM idempotency WHERE scope=? AND key=?", (scope, key)).fetchone()
            if old:
                if old[0] != fingerprint:
                    raise ServiceError(409, "idempotency_conflict", "This idempotency key belongs to a different request")
                run = db.run(old[1], ctx["project"], con)
            else:
                # Validate references and parameters even when resources are occupied.
                (validate_training if kind == "training" else validate_prediction)(body, ctx["project"])
                if db.active(con):
                    raise ServiceError(429, "execution_busy", "One execution is already active; business queueing belongs to the middle tier")
                with exclusive(settings.root / "locks/gpu.lock"):
                    check_gpu(settings, body["resource_assignment"])
                id_ = new_id("train" if kind == "training" else "predict")
                run = {"run_id": id_, "business_task_id": body["business_task_id"], "kind": kind,
                       "status": "accepted", "phase": "accepted", "created_at": now(), "started_at": None,
                       "finished_at": None, "progress": {}, "metrics": {}, "artifact_ids": [],
                       "output_model_id": None, "result_artifact_id": None, "failure": None,
                       "status_url": prefix + ("/training-runs/" if kind == "training" else "/prediction-runs/") + id_,
                       "_project": ctx["project"], "_actor": ctx["actor"], "_request": body,
                       "_request_id": request.state.request_id, "_worker_pid": None, "_worker_start": None,
                       "_engine_pid": None, "_engine_start": None}
                con.execute("INSERT INTO runs VALUES(?,?,?)", (id_, ctx["project"], dumps(run)))
                con.execute("INSERT INTO idempotency VALUES(?,?,?,?)", (scope, key, fingerprint, id_))
        return JSONResponse(status_code=202, content=public_run(run), headers={"Location": run["status_url"]})

    @app.post(prefix + "/training-runs", status_code=202)
    def training_create(body: dict, request: Request, ctx=Depends(context)):
        return create_run("training", body, request, ctx)

    @app.post(prefix + "/prediction-runs", status_code=202)
    def prediction_create(body: dict, request: Request, ctx=Depends(context)):
        return create_run("prediction", body, request, ctx)

    def scoped_run(id_, kind, project):
        run = db.run(id_, project)
        if run["kind"] != kind:
            raise ServiceError(404, "not_found", "Run type does not match this endpoint")
        return run

    @app.get(prefix + "/training-runs/{run_id}")
    def training_get(run_id: str, ctx=Depends(context)):
        return public_run(scoped_run(run_id, "training", ctx["project"]))

    @app.get(prefix + "/prediction-runs/{run_id}")
    def prediction_get(run_id: str, ctx=Depends(context)):
        return public_run(scoped_run(run_id, "prediction", ctx["project"]))

    @app.post(prefix + "/training-runs/{run_id}/cancel", status_code=202)
    def cancel(run_id: str, ctx=Depends(context)):
        scoped_run(run_id, "training", ctx["project"])
        def change(run):
            if run["phase"] == "publishing":
                raise ServiceError(409, "finalizing", "Engine has finished; artifact publication cannot be cancelled")
            if run["status"] in {"succeeded", "failed"}:
                raise ServiceError(409, "terminal_state", "Completed execution cannot be cancelled")
            if run["status"] != "cancelled":
                run["status"] = "cancel_requested"
        return public_run(db.update(run_id, change))

    @app.get(prefix + "/training-runs/{run_id}/logs")
    def logs(run_id: str, cursor: str | None = None, limit: int = 100, ctx=Depends(context)):
        scoped_run(run_id, "training", ctx["project"])
        if not 1 <= limit <= 500:
            invalid("Log page limit must be between 1 and 500")
        seq = cursor_decode(settings, cursor, ctx["project"], run_id)
        with db.connect() as con:
            rows = con.execute("SELECT seq,stamp,level,message FROM logs WHERE run_id=? AND seq>? ORDER BY seq LIMIT ?", (run_id, seq, limit + 1)).fetchall()
        entries = [{"sequence": r[0], "timestamp": r[1], "level": r[2], "message": r[3]} for r in rows[:limit]]
        return {"run_id": run_id, "entries": entries,
                "next_cursor": cursor_encode(settings, ctx["project"], run_id, entries[-1]["sequence"] if entries else seq),
                "has_more": len(rows) > limit}

    @app.get(prefix + "/artifacts/{artifact_id}/content")
    def artifact(artifact_id: str, ctx=Depends(context)):
        item = db.get("artifact", artifact_id, ctx["project"])
        path = Path(item["_path"])
        if not path.is_file() or path.stat().st_size != item["size_bytes"]:
            raise ServiceError(409, "artifact_unavailable", "Artifact is missing or has changed")
        # Files are immutable and only service-controlled paths are registered.
        return FileResponse(path, filename=item["filename"], media_type=item["content_type"],
                            headers={"X-Content-SHA256": item["sha256"]})

    # Publish the maintained contract, not the generic dict signatures above.
    app.openapi = lambda: settings.contract
    return app
