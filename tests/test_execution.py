"""HTTP/subprocess integration tests. The test engine never accesses a GPU/model."""
import concurrent.futures
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
import unittest
import zipfile

os.environ["AEROSPACE_TESTING"] = "1"
from fastapi.testclient import TestClient

from aerospace_backend.app import create_app
from aerospace_backend.core import Settings, exclusive, process_info, read_records


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="aerospace-test-")
        self.root = Path(self.temp.name)
        token = "unit-test-key-not-a-real-credential"
        credentials = {"services": [{"service_id": "test", "token_sha256": hashlib.sha256(token.encode()).hexdigest(),
                                     "projects": ["alpha", "beta"], "actors": ["tester"]}]}
        (self.root / "credentials.json").write_text(json.dumps(credentials))
        (self.root / "cursor.key").write_text("c" * 64)
        self.config = {"data_root": str(self.root / "data"), "model_root": str(self.root / "models"),
                       "credentials_file": str(self.root / "credentials.json"), "cursor_key_file": str(self.root / "cursor.key"),
                       "python": sys.executable, "node_id": "node01", "gpu_id": "0", "test_engine": "normal",
                       "max_upload_bytes": 100000,
                       "models": [{"model_id": "base", "name": "Fixture", "path": str(self.root / "base"), "template": "qwen"}]}
        self.path = self.root / "config.json"
        self.save_config()
        self.headers = {"Authorization": "Bearer " + token, "X-Project-ID": "alpha", "X-Actor-ID": "tester"}
        self.app = create_app(self.path)
        self.client = TestClient(self.app, headers=self.headers)
        self.client.__enter__()

    def save_config(self):
        self.path.write_text(json.dumps(self.config))

    def tearDown(self):
        # Leave no detached test processes behind, including after a failed assertion.
        for run in self.app.state.settings.db.active():
            self.app.state.settings.db.update(run["run_id"], lambda r: r.update(status="cancel_requested"))
        deadline = time.monotonic() + 12
        while self.app.state.settings.db.active() and time.monotonic() < deadline:
            time.sleep(0.1)
        self.client.__exit__(None, None, None)
        self.temp.cleanup()

    def upload(self, records, purpose="training_data", headers=None):
        response = self.client.post("/internal/v1/files", data={"purpose": purpose},
            files={"file": ("../../demo.json", json.dumps(records, ensure_ascii=False).encode(), "application/json")}, headers=headers)
        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(response.json()["filename"], "demo.json")
        return response.json()["file_id"]

    def dataset(self):
        train = self.upload([{"instruction": "test train", "output": "answer"}])
        val = self.upload([{"instruction": "test validation", "output": "answer"}], "evaluation_data")
        r = self.client.post("/internal/v1/dataset-versions", json={"dataset_id": "demo", "version": "1",
            "format": "alpaca", "splits": {"train": train, "validation": val}})
        self.assertEqual(r.status_code, 201, r.text)
        self.app.state.settings.validate("DatasetVersion", r.json())
        return r.json()["dataset_version_id"]

    def body(self):
        return {"business_task_id": "task", "base_model_id": "base", "dataset_version_id": self.dataset(), "method": "lora",
                "hyperparameters": {"learning_rate": 0.0001, "max_steps": 8, "cutoff_len": 256,
                    "per_device_train_batch_size": 1, "gradient_accumulation_steps": 4, "compute_dtype": "float16", "attention": "sdpa"},
                "resource_assignment": {"node_id": "node01", "gpu_ids": ["0"]}}

    def start(self, body=None, key="first"):
        r = self.client.post("/internal/v1/training-runs", json=body or self.body(), headers={"Idempotency-Key": key})
        self.assertEqual(r.status_code, 202, r.text)
        self.app.state.settings.validate("Run", r.json())
        return r.json()

    def wait(self, run, statuses=None, timeout=20):
        statuses = statuses or {"succeeded", "failed", "cancelled"}
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            r = self.client.get(run["status_url"])
            self.assertEqual(r.status_code, 200, r.text)
            if r.json()["status"] in statuses:
                return r.json()
            time.sleep(0.1)
        self.fail("Execution did not reach expected state")

    def test_auth_and_scope(self):
        self.assertEqual(self.client.get("/openapi.json").json()["servers"][0]["url"], "/")
        self.assertEqual(self.client.get("/internal/v1/models", headers={"Authorization": ""}).status_code, 401)
        self.assertEqual(self.client.get("/internal/v1/models", headers={"X-Project-ID": "other"}).status_code, 403)
        self.assertEqual(self.client.get("/internal/v1/models", headers={"X-Actor-ID": "other"}).status_code, 403)
        response = self.client.get("/internal/v1/models")
        self.assertNotIn("_path", response.text)
        self.app.state.settings.validate("ModelPage", response.json())
        self.app.state.settings.validate("Capabilities", self.client.get("/internal/v1/capabilities").json())

    def test_data_validation_immutability_and_cross_project(self):
        data_id = self.dataset()
        self.assertEqual(self.client.get("/internal/v1/dataset-versions/" + data_id, headers={"X-Project-ID": "beta"}).status_code, 404)
        original = self.client.get("/internal/v1/dataset-versions/" + data_id).json()
        repeat = {"dataset_id": "demo", "version": "1", "format": "alpaca", "splits": {"train": original["splits"]["train"]["file_id"]}}
        self.assertEqual(self.client.post("/internal/v1/dataset-versions", json=repeat).status_code, 409)
        for record in ({"instruction": "missing"}, {"instruction": "", "output": "x"}, {"instruction": "a" * 32769, "output": "x"}):
            file_id = self.upload([record])
            repeat.update(version=file_id, splits={"train": file_id})
            self.assertEqual(self.client.post("/internal/v1/dataset-versions", json=repeat).status_code, 422)
        same = original["splits"]["train"]["file_id"]
        # Correct purpose but identical content must not cross train/evaluation splits.
        duplicate = self.upload([{"instruction": "test train", "output": "different answer"}], "evaluation_data")
        repeat.update(version="overlap", splits={"train": same, "validation": duplicate})
        self.assertEqual(self.client.post("/internal/v1/dataset-versions", json=repeat).status_code, 422)
        r = self.client.post("/internal/v1/files", data={"purpose": "training_data"}, files={"file": ("large", b"x" * 100001)})
        self.assertEqual(r.status_code, 413)

    def test_idempotency_concurrency_and_parameter_validation(self):
        body = self.body()
        def create(_):
            return self.client.post("/internal/v1/training-runs", json=body, headers={"Idempotency-Key": "same"})
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            responses = list(pool.map(create, range(4)))
        self.assertTrue(all(r.status_code == 202 for r in responses), [r.text for r in responses])
        self.assertEqual(len({r.json()["run_id"] for r in responses}), 1)
        changed = copy.deepcopy(body)
        changed["hyperparameters"]["max_steps"] = 9
        self.assertEqual(self.client.post("/internal/v1/training-runs", json=changed, headers={"Idempotency-Key": "same"}).status_code, 409)
        changed["hyperparameters"]["shell"] = "should never be executed"
        self.assertEqual(self.client.post("/internal/v1/training-runs", json=changed, headers={"Idempotency-Key": "other"}).status_code, 422)
        self.assertEqual(self.client.post("/internal/v1/training-runs", json=body, headers={"Idempotency-Key": "other"}).status_code, 429)
        self.wait(responses[0].json())
        replay = create(0)
        self.assertEqual(replay.json()["status"], "succeeded")

    def test_success_artifacts_logs_and_prediction(self):
        run = self.wait(self.start())
        self.assertEqual(run["status"], "succeeded", run)
        self.assertIsNotNone(run["output_model_id"])
        catalog = self.client.get("/internal/v1/models", params={"limit": 1}).json()
        self.assertEqual(len(catalog["items"]), 1)
        self.assertTrue(catalog["next_cursor"])
        second = self.client.get("/internal/v1/models", params={"cursor": catalog["next_cursor"], "limit": 1}).json()
        self.assertNotEqual(catalog["items"][0]["model_id"], second["items"][0]["model_id"])
        self.assertEqual(self.client.get("/internal/v1/models", params={"cursor": "bad"}).status_code, 422)
        self.assertEqual(run["progress"]["current_step"], 20)
        self.app.state.settings.validate("Run", run)
        page = self.client.get(run["status_url"] + "/logs", params={"limit": 1}).json()
        self.assertTrue(page["has_more"])
        self.app.state.settings.validate("LogPage", page)
        next_page = self.client.get(run["status_url"] + "/logs", params={"cursor": page["next_cursor"], "limit": 500}).json()
        self.assertGreater(next_page["entries"][0]["sequence"], page["entries"][0]["sequence"])
        self.assertEqual(self.client.get(run["status_url"] + "/logs", params={"cursor": page["next_cursor"] + "x"}).status_code, 422)
        artifact = self.client.get("/internal/v1/artifacts/" + run["artifact_ids"][0] + "/content")
        self.assertEqual(artifact.status_code, 200)
        self.assertEqual(hashlib.sha256(artifact.content).hexdigest(), artifact.headers["x-content-sha256"])
        with zipfile.ZipFile(io.BytesIO(artifact.content)) as zip_:
            self.assertIn("adapter_model.safetensors", zip_.namelist())
        self.assertEqual(self.client.get("/internal/v1/artifacts/" + run["artifact_ids"][0] + "/content", headers={"X-Project-ID": "beta"}).status_code, 404)
        self.assertEqual(self.client.post(run["status_url"] + "/cancel").status_code, 409)
        file_id = self.upload([{"question_id": "q1", "prompt": "Question"}], "prediction_data")
        dataset = self.client.post("/internal/v1/dataset-versions", json={"dataset_id": "questions", "version": "1", "format": "question_set_v1", "splits": {"test": file_id}}).json()
        response = self.client.post("/internal/v1/prediction-runs", headers={"Idempotency-Key": "prediction"}, json={
            "business_task_id": "p1", "model_id": run["output_model_id"], "dataset_version_id": dataset["dataset_version_id"],
            "split": "test", "generation": {"max_new_tokens": 32}, "resource_assignment": {"node_id": "node01", "gpu_ids": ["0"]}})
        self.assertEqual(response.status_code, 202, response.text)
        prediction = self.wait(response.json())
        self.assertEqual(prediction["status"], "succeeded", prediction)
        result = self.client.get("/internal/v1/artifacts/" + prediction["result_artifact_id"] + "/content")
        self.app.state.settings.validate("PredictionRecord", json.loads(result.text))

    def test_cancel_releases_process_and_gpu_lease(self):
        run = self.start()
        self.wait(run, {"running"})
        deadline = time.monotonic() + 5
        while not (private := self.app.state.settings.db.run(run["run_id"]))["_engine_pid"] and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertIsNotNone(private["_engine_pid"])
        self.assertEqual(self.client.post(run["status_url"] + "/cancel").status_code, 202)
        self.assertEqual(self.wait(run)["status"], "cancelled")
        self.assertIsNone(process_info(private["_engine_pid"]))
        time.sleep(0.1)
        with exclusive(self.app.state.settings.root / "locks/gpu.lock"):
            pass

    def test_api_restart_preserves_active_execution(self):
        run = self.start()
        self.wait(run, {"running"})
        self.client.__exit__(None, None, None)
        self.app = create_app(self.path)
        self.client = TestClient(self.app, headers=self.headers)
        self.client.__enter__()
        self.assertEqual(self.wait(run)["status"], "succeeded")
        models = self.client.get("/internal/v1/models").json()["items"]
        self.assertEqual(sum(m["kind"] == "adapter" for m in models), 1)

    def test_killed_worker_reaps_orphan_engine(self):
        run = self.start()
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            private = self.app.state.settings.db.run(run["run_id"])
            if private["_engine_pid"]:
                break
            time.sleep(0.05)
        self.assertIsNotNone(private["_engine_pid"])
        os.kill(private["_worker_pid"], signal.SIGKILL)
        failed = self.wait(run)
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["failure"]["code"], "worker_interrupted")
        self.assertIsNone(process_info(private["_engine_pid"]))
        with exclusive(self.app.state.settings.root / "locks/gpu.lock"):
            pass

    def test_nonfinite_metric_fails_without_publishing_model(self):
        self.config["test_engine"] = "nan"
        self.save_config()
        run = self.wait(self.start())
        self.assertEqual(run["status"], "failed")
        self.assertEqual(run["failure"]["code"], "non_finite_metric")
        self.assertIsNone(run["metrics"].get("latest_loss"))
        self.assertIsNone(run["output_model_id"])

    def test_engine_failure_and_checksum_tampering(self):
        self.config["test_engine"] = "fail"
        self.save_config()
        body = self.body()
        run = self.wait(self.start(body))
        self.assertEqual(run["status"], "failed")
        self.assertEqual(run["failure"]["code"], "engine_failed")
        dataset = self.app.state.settings.db.get("dataset", body["dataset_version_id"], "alpha")
        file = self.app.state.settings.db.get("file", dataset["splits"]["train"]["file_id"], "alpha")
        path = Path(file["_path"])
        path.chmod(0o600)
        path.write_text("changed")
        response = self.client.post("/internal/v1/training-runs", json=body, headers={"Idempotency-Key": "tamper"})
        self.assertEqual(response.status_code, 409)


if __name__ == "__main__":
    unittest.main(verbosity=2)
