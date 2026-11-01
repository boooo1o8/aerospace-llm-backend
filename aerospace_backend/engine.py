"""GPU engine subprocess; no web server or credentials are loaded here."""
import json
import math
import os
from pathlib import Path
import sys
import time


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def test_engine(spec):
    # Explicitly isolated from production; used by subprocess integration tests.
    if os.environ.get("AEROSPACE_TESTING") != "1":
        raise RuntimeError("Test engine is disabled")
    mode = spec["test_engine"]
    if mode == "fail":
        raise RuntimeError("Deliberate engine failure")
    if spec["kind"] == "training":
        output = Path(spec["train_config"]["output_dir"])
        output.mkdir(exist_ok=True)
        for i in range(20):
            loss = float("nan") if mode == "nan" else 2 - i / 100
            with (output / "trainer_log.jsonl").open("a") as handle:
                handle.write(json.dumps({"current_steps": i + 1, "total_steps": 20, "loss": loss}) + "\n")
            print("Fixture update", i + 1, flush=True)
            time.sleep(0.15)
        (output / "adapter_model.safetensors").write_bytes(b"TEST FIXTURE; NOT A MODEL")
        write_json(output / "adapter_config.json", {"test_fixture": True})
    else:
        rows = json.loads(Path(spec["questions"]).read_text())
        with open(spec["results"], "w") as handle:
            for i, item in enumerate(rows):
                handle.write(json.dumps({"question_id": item["question_id"], "raw_output": "fixture",
                    "parse_status": "not_attempted", "usage": {"prompt_tokens": 1, "completion_tokens": 1}, "error": None}) + "\n")
                write_json(Path(spec["results"]).parent / "prediction-progress.json", {"completed_items": i + 1})


def train(spec):
    from llamafactory.train.tuner import run_exp
    run_exp(args=spec["train_config"])
    import torch
    from safetensors import safe_open
    output = Path(spec["train_config"]["output_dir"])
    changed = 0
    with safe_open(output / "adapter_model.safetensors", framework="pt", device="cpu") as weights:
        for key in weights.keys():
            tensor = weights.get_tensor(key)
            if not torch.isfinite(tensor).all():
                raise RuntimeError("Non-finite adapter weights")
            if "lora_B" in key and torch.count_nonzero(tensor):
                changed += 1
    if not changed:
        raise RuntimeError("Adapter did not receive a nonzero update")
    write_json(output / "adapter-verification.json", {"finite_weights": True, "updated_lora_b_matrices": changed})
    for file in ("train_results.json", "eval_results.json"):
        if (output / file).exists():
            values = json.loads((output / file).read_text())
            if any(isinstance(v, float) and not math.isfinite(v) for v in values.values()):
                raise RuntimeError("Non-finite final training metric")


def predict(spec):
    import torch
    from transformers import set_seed
    from llamafactory.chat import ChatModel
    set_seed(spec["generation"].get("seed", 42))
    model = ChatModel(spec["infer_config"])
    records = json.loads(Path(spec["questions"]).read_text(encoding="utf-8"))
    generation = spec["generation"]
    with open(spec["results"], "w", encoding="utf-8") as handle:
        for i, item in enumerate(records):
            record = {"question_id": item["question_id"], "raw_output": None, "parse_status": "not_attempted",
                      "usage": {"prompt_tokens": 0, "completion_tokens": 0}, "error": None}
            try:
                # Leave generation space inside this service's first-version context cap.
                tokens = model.engine.tokenizer.encode(item["prompt"] + item.get("system", ""))
                if len(tokens) + generation["max_new_tokens"] + 64 > 1024:
                    raise ValueError("input_too_long")
                temperature = generation.get("temperature", 0)
                answer = model.chat([{"role": "user", "content": item["prompt"]}], system=item.get("system"),
                    max_new_tokens=generation["max_new_tokens"], do_sample=temperature > 0,
                    temperature=max(temperature, 0.01))[0]
                record.update(raw_output=answer.response_text,
                              usage={"prompt_tokens": answer.prompt_length, "completion_tokens": answer.response_length})
            except torch.OutOfMemoryError:
                raise
            except Exception as exc:
                record["error"] = {"code": "input_too_long" if str(exc) == "input_too_long" else "prediction_failed",
                                   "message": "Input exceeds context limit" if str(exc) == "input_too_long" else "This question failed during generation"}
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            write_json(Path(spec["results"]).parent / "prediction-progress.json", {"completed_items": i + 1})
            print("Completed question", item["question_id"], flush=True)


if __name__ == "__main__":
    spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    deadline = time.monotonic() + 10
    while not Path(spec["start_ack"]).exists():
        if os.getppid() != spec["owner_pid"] or time.monotonic() > deadline:
            raise RuntimeError("Owner disappeared before engine registration")
        time.sleep(0.05)
    if spec.get("test_engine"):
        test_engine(spec)
    elif spec["kind"] == "training":
        train(spec)
    else:
        predict(spec)
