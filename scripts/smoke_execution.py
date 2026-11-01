"""Real HTTP/GPU validation using engineering fixtures, never domain evaluation."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid
import zipfile

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.generate_demo import generate

ACTIVE_CONTEXT = {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--key-file", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--project", default="demo-project")
    parser.add_argument("--actor", default="demo-user")
    parser.add_argument("--probe-busy", action="store_true")
    parser.add_argument("--restart-api", action="store_true")
    args = parser.parse_args()
    report = {"engineering_demo_only": True, "checks": {}, "started_at": time.time()}
    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    fixtures = generate(path.parent / "demo-data")
    suffix = uuid.uuid4().hex[:12]
    report["validation_tag"] = suffix
    ACTIVE_CONTEXT.update(config=args.config, tag=suffix)
    headers = {"Authorization": "Bearer " + Path(args.key_file).read_text().strip(),
               "X-Project-ID": args.project, "X-Actor-ID": args.actor}
    client = httpx.Client(base_url=args.base_url, headers=headers, timeout=20)
    prefix = "/internal/v1"
    def check(name, ok, detail=None):
        report["checks"][name] = {"passed": bool(ok), "detail": detail}
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        if not ok:
            raise RuntimeError(name + " failed: " + str(detail))
        print("PASS", name, flush=True)
    def upload(filename, purpose):
        with (fixtures / filename).open("rb") as handle:
            response = client.post(prefix + "/files", data={"purpose": purpose}, files={"file": (filename, handle, "application/json")})
        response.raise_for_status()
        return response.json()["file_id"]
    def wait(run, status=None, timeout=240):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            r = client.get(run["status_url"])
            r.raise_for_status()
            result = r.json()
            if status and result["status"] == status:
                return result
            if result["status"] in {"succeeded", "failed", "cancelled"}:
                return result
            time.sleep(0.5)
        raise RuntimeError("Execution timed out")
    check("health", client.get("/healthz").status_code == 200)
    no_auth = httpx.get(args.base_url + prefix + "/models")
    check("unauthenticated_401", no_auth.status_code == 401)
    denied = client.get(prefix + "/models", headers={"X-Project-ID": "not-authorized"})
    check("unauthorized_project_403", denied.status_code == 403)
    models = client.get(prefix + "/models").json()["items"]
    check("model_catalog", any(m["model_id"] == "qwen2.5-1.5b-instruct" for m in models))
    for name in ("invalid_missing_output.json", "invalid_empty_text.json", "invalid_long_text.json"):
        file_id = upload(name, "training_data")
        r = client.post(prefix + "/dataset-versions", json={"dataset_id": "bad-" + suffix, "version": name,
                         "format": "alpaca", "splits": {"train": file_id}})
        check(name + "_rejected", r.status_code == 422)
    train, val = upload("train.json", "training_data"), upload("validation.json", "evaluation_data")
    dataset_body = {"dataset_id": "engineering-demo-" + suffix, "version": "1", "format": "alpaca", "splits": {"train": train, "validation": val}}
    response = client.post(prefix + "/dataset-versions", json=dataset_body)
    check("dataset_registered", response.status_code == 201, response.status_code)
    dataset = response.json()
    report["dataset_version_id"] = dataset["dataset_version_id"]
    check("dataset_query", client.get(prefix + "/dataset-versions/" + dataset["dataset_version_id"]).json() == dataset)
    check("immutable_version_409", client.post(prefix + "/dataset-versions", json=dataset_body).status_code == 409)
    body = {"business_task_id": "engineering-" + suffix, "base_model_id": "qwen2.5-1.5b-instruct",
            "dataset_version_id": dataset["dataset_version_id"], "method": "lora",
            "hyperparameters": {"learning_rate": 0.0001, "max_steps": 8, "cutoff_len": 256,
                "per_device_train_batch_size": 1, "gradient_accumulation_steps": 4, "lora_rank": 8,
                "lora_dropout": 0.05, "compute_dtype": "float16", "attention": "sdpa", "seed": 42},
            "resource_assignment": {"node_id": "node01", "gpu_ids": ["0"]}}
    key = "engineering-" + suffix
    r = client.post(prefix + "/training-runs", json=body, headers={"Idempotency-Key": key})
    if args.probe_busy:
        check("busy_gpu_503_without_launch", r.status_code == 503 and r.json()["error"]["code"] == "gpu_busy", r.status_code)
        report["completed_at"] = time.time()
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    check("training_accepted", r.status_code == 202, r.status_code)
    run = r.json()
    report["training_run_id"] = run["run_id"]
    replay = client.post(prefix + "/training-runs", json=body, headers={"Idempotency-Key": key})
    check("idempotent_same_run", replay.status_code == 202 and replay.json()["run_id"] == run["run_id"])
    check("single_execution_429", client.post(prefix + "/training-runs", json=body, headers={"Idempotency-Key": key + "-other"}).status_code == 429)
    running = wait(run, "running")
    check("running_observed", running["status"] == "running", running)
    if args.restart_api:
        subprocess.run([sys.executable, str(Path(__file__).with_name("service.py")), "restart", "--config", args.config], check=True)
        after = client.get(run["status_url"])
        check("api_restart_retains_run", after.status_code == 200 and after.json()["run_id"] == run["run_id"])
    finished = wait(run)
    report["training_result"] = finished
    check("real_lora_succeeded", finished["status"] == "succeeded", finished)
    logs = client.get(run["status_url"] + "/logs", params={"limit": 2}).json()
    check("incremental_logs", bool(logs["entries"]) and bool(logs["next_cursor"]))
    follow = client.get(run["status_url"] + "/logs", params={"cursor": logs["next_cursor"]}).json()
    check("log_cursor_advances", not follow["entries"] or follow["entries"][0]["sequence"] > logs["entries"][-1]["sequence"])
    response = client.get(prefix + "/artifacts/" + finished["artifact_ids"][0] + "/content")
    check("adapter_download_checksum", response.status_code == 200 and hashlib.sha256(response.content).hexdigest() == response.headers.get("X-Content-SHA256"))
    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_:
        evidence = json.loads(zip_.read("adapter-verification.json"))
    report["adapter_verification"] = evidence
    check("finite_and_updated_adapter", evidence["finite_weights"] and evidence["updated_lora_b_matrices"] > 0, evidence)
    models = client.get(prefix + "/models").json()["items"]
    check("output_model_registered", any(m["model_id"] == finished["output_model_id"] for m in models))
    questions = upload("questions.json", "prediction_data")
    r = client.post(prefix + "/dataset-versions", json={"dataset_id": "questions-" + suffix, "version": "1", "format": "question_set_v1", "splits": {"test": questions}})
    r.raise_for_status()
    question_dataset = r.json()["dataset_version_id"]
    r = client.post(prefix + "/prediction-runs", headers={"Idempotency-Key": key + "-predict"}, json={
        "business_task_id": "predict-" + suffix, "model_id": finished["output_model_id"], "dataset_version_id": question_dataset,
        "split": "test", "generation": {"max_new_tokens": 64, "temperature": 0, "seed": 42},
        "resource_assignment": body["resource_assignment"]})
    check("prediction_accepted", r.status_code == 202, r.status_code)
    prediction = wait(r.json())
    report["prediction_result"] = prediction
    check("new_adapter_prediction_succeeded", prediction["status"] == "succeeded", prediction)
    r = client.get(prefix + "/artifacts/" + prediction["result_artifact_id"] + "/content")
    records = [json.loads(line) for line in r.text.splitlines()]
    report["prediction_records"] = records
    check("per_question_output", len(records) == 2 and all(v["raw_output"] and v["error"] is None for v in records), records)
    check("question_ids_preserved", [v["question_id"] for v in records] == ["demo_q1", "demo_q2"])
    # A long real run is cancelled while its engine process is alive.
    body["hyperparameters"]["max_steps"] = 1000
    r = client.post(prefix + "/training-runs", json=body, headers={"Idempotency-Key": key + "-cancel"})
    check("cancel_run_accepted", r.status_code == 202, r.status_code)
    cancel_run = r.json()
    report["cancel_run_id"] = cancel_run["run_id"]
    from aerospace_backend.core import Settings, exclusive, process_info
    settings = Settings(args.config)
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        private = settings.db.run(cancel_run["run_id"])
        if private["_engine_pid"] and private["progress"].get("current_step", 0) >= 1:
            break
        time.sleep(0.1)
    check("cancel_engine_started", bool(private["_engine_pid"]))
    check("cancel_after_real_update", private["progress"].get("current_step", 0) >= 1)
    r = client.post(cancel_run["status_url"] + "/cancel")
    check("cancel_requested", r.status_code == 202)
    cancelled = wait(cancel_run)
    report["cancelled_run"] = cancelled
    check("cancelled_terminal", cancelled["status"] == "cancelled", cancelled)
    check("cancelled_engine_exited", process_info(private["_engine_pid"]) is None)
    with exclusive(settings.root / "locks/gpu.lock"):
        pass
    check("gpu_lease_released", True)
    report["completed_at"] = time.time()
    report["all_passed"] = all(v["passed"] for v in report["checks"].values())
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    client.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        # Do not leave long engineering-validation runs active on an assertion or HTTP failure.
        if ACTIVE_CONTEXT:
            from aerospace_backend.core import Settings
            db = Settings(ACTIVE_CONTEXT["config"]).db
            tasks = {"engineering-" + ACTIVE_CONTEXT["tag"], "predict-" + ACTIVE_CONTEXT["tag"]}
            for run in db.active():
                if run["business_task_id"] in tasks:
                    db.update(run["run_id"], lambda r: r.update(status="cancel_requested"))
            deadline = time.monotonic() + 30
            while any(r["business_task_id"] in tasks for r in db.active()) and time.monotonic() < deadline:
                time.sleep(0.2)
