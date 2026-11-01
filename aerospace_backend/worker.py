"""Detached execution owner. The API may restart without stopping this process."""
from __future__ import annotations

import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import zipfile

from .core import (Settings, ServiceError, TERMINAL, check_gpu, dumps, exclusive,
                   new_id, now, process_info, read_file, read_records, scrub, sha, stop_group)


class Worker:
    def __init__(self, settings, run_id):
        self.s, self.db, self.id = settings, settings.db, run_id
        self.folder = settings.root / "runs" / run_id
        self.folder.mkdir(exist_ok=True, mode=0o700)
        self.engine = None
        self.engine_start = None
        self.spec_path = self.folder / "engine.json"
        self.signal_cancel = False
        self.offset = 0
        self.partial = ""
        self.seen_metrics = None

    def log(self, message, level="INFO"):
        message = scrub(self.s, message)
        self.db.log(self.id, message, level)
        with (self.folder / "execution.log").open("a", encoding="utf-8") as handle:
            handle.write(now() + " " + level + " " + message + "\n")

    def signal(self, *_):
        self.signal_cancel = True

    def cancellation(self):
        return self.signal_cancel or self.db.run(self.id)["status"] == "cancel_requested"

    def finish(self, status, failure=None):
        def change(run):
            if run["status"] in TERMINAL:
                return
            if run["status"] == "cancel_requested" or self.signal_cancel:
                run.update(status="cancelled", phase="cancelled", failure=None, finished_at=now())
            else:
                run.update(status=status, phase=status, failure=failure, finished_at=now())
        return self.db.update(self.id, change)

    def prepare(self, run):
        body, project = run["_request"], run["_project"]
        dataset = self.db.get("dataset", body["dataset_version_id"], project)
        provenance = {"run_id": self.id, "kind": run["kind"], "business_task_id": body["business_task_id"],
                      "request": body, "dataset": dataset, "created_at": run["created_at"],
                      "engine": "llamafactory", "source_revision": self.s.data.get("llamafactory_revision")}
        (self.folder / "provenance.json").write_text(dumps(provenance), encoding="utf-8")
        if run["kind"] == "training":
            model = self.db.get("model", body["base_model_id"], project)
            data_dir = self.folder / "dataset"
            data_dir.mkdir(exist_ok=True)
            info = {}
            for split, entry in dataset["splits"].items():
                if split not in {"train", "validation"}:
                    continue
                uploaded = self.db.get("file", entry["file_id"], project)
                data = read_file(self.s, uploaded)
                records, _ = read_records(data, dataset["format"])
                records = [{"instruction": row["instruction"], "output": row["output"],
                            "input": row.get("input", ""), "system": row.get("system", "")} for row in records]
                (data_dir / (split + ".json")).write_text(dumps(records), encoding="utf-8")
                info[split] = {"file_name": split + ".json", "formatting": "alpaca",
                    "columns": {"prompt": "instruction", "query": "input", "response": "output", "system": "system"}}
            (data_dir / "dataset_info.json").write_text(dumps(info), encoding="utf-8")
            p = body["hyperparameters"]
            config = {"model_name_or_path": model["_path"], "trust_remote_code": False, "stage": "sft",
                "do_train": True, "finetuning_type": "lora", "lora_target": "all",
                "lora_rank": p.get("lora_rank", 8), "lora_dropout": p.get("lora_dropout", 0.05),
                "dataset": "train", "dataset_dir": str(data_dir), "template": model["_template"],
                "cutoff_len": p["cutoff_len"], "preprocessing_num_workers": 1, "dataloader_num_workers": 0,
                "output_dir": str(self.folder / "output"), "overwrite_output_dir": False,
                "logging_steps": 1, "save_strategy": "no", "report_to": "none", "plot_loss": False,
                "per_device_train_batch_size": p["per_device_train_batch_size"], "per_device_eval_batch_size": 1,
                "gradient_accumulation_steps": p["gradient_accumulation_steps"], "learning_rate": p["learning_rate"],
                "lr_scheduler_type": "cosine", "warmup_ratio": 0.1, "fp16": True, "bf16": False,
                "flash_attn": "sdpa", "optim": "adamw_torch", "gradient_checkpointing": True,
                "upcast_layernorm": True, "upcast_lmhead_output": True, "seed": p.get("seed", 42)}
            if "max_steps" in p:
                config["max_steps"] = p["max_steps"]
            else:
                config["num_train_epochs"] = p["num_train_epochs"]
            if "validation" in info:
                config.update(do_eval=True, eval_dataset="validation", eval_strategy="steps",
                              eval_steps=min(4, p.get("max_steps", 4)))
            spec = {"kind": "training", "train_config": config}
        else:
            model = self.db.get("model", body["model_id"], project)
            base = self.db.get("model", model["base_model_id"], project) if model["kind"] == "adapter" else model
            entry = dataset["splits"][body["split"]]
            records, _ = read_records(read_file(self.s, self.db.get("file", entry["file_id"], project)), dataset["format"])
            (self.folder / "questions.json").write_text(dumps(records), encoding="utf-8")
            config = {"model_name_or_path": base["_path"], "template": base["_template"], "finetuning_type": "lora",
                      "infer_backend": "huggingface", "infer_dtype": "float16", "flash_attn": "sdpa",
                      "trust_remote_code": False, "upcast_layernorm": True, "upcast_lmhead_output": True}
            if model["kind"] == "adapter":
                config["adapter_name_or_path"] = model["_path"]
            spec = {"kind": "prediction", "infer_config": config, "generation": body["generation"],
                    "questions": str(self.folder / "questions.json"), "results": str(self.folder / "predictions.jsonl")}
            self.db.update(self.id, lambda r: r["progress"].update(total_items=len(records), completed_items=0))
        spec.update(run_id=self.id, runtime_config=str(self.s.path), owner_pid=os.getpid(),
                    start_ack=str(self.folder / "engine-start-ack"))
        if self.s.data.get("test_engine"):
            spec["test_engine"] = self.s.data["test_engine"]
        self.spec_path.write_text(dumps(spec), encoding="utf-8")
        return spec

    def drain(self, final=False):
        path = self.folder / "engine.log"
        if path.exists():
            with path.open("rb") as handle:
                handle.seek(self.offset)
                data = handle.read(1024 * 1024 if final else 65536)
                self.offset = handle.tell()
            text = self.partial + data.decode("utf-8", errors="replace").replace("\r", "\n")
            lines = text.split("\n")
            self.partial = lines.pop()
            for line in lines:
                if line.strip():
                    self.log(line[:8000])
            if final and self.partial:
                self.log(self.partial[:8000])
                self.partial = ""

    def metrics(self, spec):
        if spec["kind"] == "prediction":
            result = self.folder / "prediction-progress.json"
            if result.exists():
                try:
                    data = json.loads(result.read_text())
                    self.db.update(self.id, lambda r: r["progress"].update(completed_items=data["completed_items"]))
                except json.JSONDecodeError:
                    pass
            return
        path = self.folder / "output/trainer_log.jsonl"
        if not path.exists():
            return
        with path.open("rb") as handle:
            handle.seek(max(0, path.stat().st_size - 65536))
            lines = handle.read().decode("utf-8", errors="replace").splitlines()
        latest, merged = None, {}
        for line in lines:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            for key in ("loss", "eval_loss", "grad_norm"):
                if key in data and isinstance(data[key], (float, int)) and not math.isfinite(data[key]):
                    raise ServiceError(500, "non_finite_metric", "Training produced a non-finite metric")
            if "loss" in data:
                merged["latest_loss"] = data["loss"]
            if "eval_loss" in data:
                merged["eval_loss"] = data["eval_loss"]
            latest = data
        if latest:
            def change(run):
                for src, dest in (("current_steps", "current_step"), ("total_steps", "total_steps"), ("epoch", "epoch")):
                    if src in latest:
                        run["progress"][dest] = latest[src]
                run["metrics"].update(merged)
            self.db.update(self.id, change)

    def artifact(self, path, content_type):
        id_ = new_id("artifact")
        value = {"artifact_id": id_, "filename": path.name, "content_type": content_type,
                 "size_bytes": path.stat().st_size, "sha256": sha(path.read_bytes()), "_path": str(path)}
        self.db.put("artifact", id_, self.db.run(self.id)["_project"], value)
        self.db.update(self.id, lambda r: r["artifact_ids"].append(id_))
        return id_

    def publish(self, spec):
        run = self.db.run(self.id)
        if spec["kind"] == "training":
            output = self.folder / "output"
            if not (output / "adapter_model.safetensors").is_file() or not (output / "adapter_config.json").is_file():
                raise ServiceError(500, "missing_adapter", "Engine did not produce a usable adapter")
            final = self.s.models / self.id
            final.mkdir(mode=0o700)
            # Keep the final adapter on SSD; checkpoints and working inputs stay on /data.
            for path in output.iterdir():
                if path.is_file() and path.suffix in {".json", ".safetensors", ".model", ".txt", ".jinja"}:
                    shutil.copy2(path, final / path.name)
            archive = final / "adapter.zip"
            with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zip_:
                for path in final.iterdir():
                    if path.is_file() and path != archive:
                        zip_.write(path, path.name)
            artifact = self.artifact(archive, "application/zip")
            base = self.db.get("model", run["_request"]["base_model_id"], run["_project"])
            model_id = new_id("model")
            self.db.put("model", model_id, run["_project"], {"model_id": model_id,
                "name": "LoRA " + self.id, "kind": "adapter", "base_model_id": base["model_id"],
                "source_training_run_id": self.id, "artifact_ids": [artifact], "supported_methods": [],
                "max_context_tokens": base["max_context_tokens"], "_path": str(final), "_template": base["_template"]})
            self.db.update(self.id, lambda r: r.update(output_model_id=model_id))
        else:
            path = self.folder / "predictions.jsonl"
            id_ = self.artifact(path, "application/x-ndjson")
            self.db.update(self.id, lambda r: r.update(result_artifact_id=id_))

    def execute(self):
        signal.signal(signal.SIGTERM, self.signal)
        signal.signal(signal.SIGINT, self.signal)
        with (self.s.root / "locks" / (self.id + ".lock")).open("a+b") as run_lock:
            fcntl.flock(run_lock, fcntl.LOCK_EX)
            run = self.db.run(self.id)
            if run["status"] in TERMINAL or (run["started_at"] and run["_worker_pid"] != os.getpid()):
                return
            info = process_info(os.getpid())
            self.db.update(self.id, lambda r: r.update(_worker_pid=os.getpid(), _worker_start=info["start"]))
            try:
                with exclusive(self.s.root / "locks/gpu.lock") as gpu_lock:
                    if self.cancellation():
                        self.finish("cancelled")
                        return
                    check_gpu(self.s, run["_request"]["resource_assignment"])
                    def running(r):
                        if r["status"] != "cancel_requested":
                            r.update(status="running", phase="preparing", started_at=now())
                    self.db.update(self.id, running)
                    self.log("Execution started; inputs and resource assignment are fixed")
                    spec = self.prepare(run)
                    if self.cancellation():
                        self.finish("cancelled")
                        return
                    env = os.environ.copy()
                    for name in list(env):
                        if any(s in name.upper() for s in ("TOKEN", "PASSWORD", "API_KEY", "SECRET")) or name.startswith(("MASTER_", "RANK", "LOCAL_RANK", "WORLD_SIZE")):
                            env.pop(name, None)
                    env.update(CUDA_VISIBLE_DEVICES=self.s.data["gpu_id"], HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                               TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4", PYTHONUNBUFFERED="1")
                    with (self.folder / "engine.log").open("ab") as log:
                        # Both leases are inherited by the engine. A killed worker cannot expose its GPU as free.
                        self.engine = subprocess.Popen([self.s.data["python"], "-m", "aerospace_backend.engine", str(self.spec_path)],
                            cwd=str(Path(__file__).resolve().parents[1]), env=env, stdin=subprocess.DEVNULL,
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                            pass_fds=(gpu_lock.fileno(), run_lock.fileno()))
                    self.engine_start = process_info(self.engine.pid)["start"]
                    self.db.update(self.id, lambda r: r.update(phase="training" if spec["kind"] == "training" else "predicting",
                        _engine_pid=self.engine.pid, _engine_start=self.engine_start))
                    # The engine cannot use the GPU until its identity is durable.
                    Path(spec["start_ack"]).touch(mode=0o600)
                    while self.engine.poll() is None:
                        self.drain()
                        self.metrics(spec)
                        if self.cancellation():
                            stop_group(self.engine.pid, self.engine_start, str(self.spec_path))
                            self.engine.wait(timeout=10)
                            break
                        time.sleep(0.25)
                    self.drain(final=True)
                    if self.cancellation():
                        self.log("Engine stopped; cancellation completed")
                        self.finish("cancelled")
                        return
                    self.metrics(spec)
                    if self.engine.returncode != 0:
                        tail = (self.folder / "engine.log").read_bytes()[-16000:].decode("utf-8", errors="replace")
                        code = "out_of_memory" if "out of memory" in tail.lower() else "engine_failed"
                        raise ServiceError(500, code, "Execution engine exited unsuccessfully; inspect the sanitized execution log")
                    def publishing(r):
                        if r["status"] == "cancel_requested":
                            raise ServiceError(409, "cancelled", "Cancellation requested before publication")
                        r["phase"] = "publishing"
                    self.db.update(self.id, publishing)
                    self.publish(spec)
                    self.log("Execution completed; model/results are registered")
                    self.artifact(self.folder / "provenance.json", "application/json")
                    self.artifact(self.folder / "execution.log", "text/plain; charset=utf-8")
                    self.finish("succeeded")
            except BaseException as exc:
                if self.engine is not None:
                    stop_group(self.engine.pid, self.engine_start, str(self.spec_path))
                    self.engine.wait(timeout=10)
                code = exc.code if isinstance(exc, ServiceError) else "execution_failed"
                message = exc.message if isinstance(exc, ServiceError) else "Execution failed; inspect the sanitized execution log"
                self.log(code + ": " + message, "ERROR")
                self.finish("failed", {"code": code, "message": message, "request_id": run["_request_id"]})


def main():
    os.umask(0o077)
    settings = Settings(sys.argv[1])
    Worker(settings, sys.argv[2]).execute()


if __name__ == "__main__":
    main()
