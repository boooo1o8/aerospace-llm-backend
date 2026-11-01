"""Generate explicitly labeled engineering fixtures; no aerospace knowledge claims."""
import argparse
import json
from pathlib import Path


def generate(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    files = {
        "train.json": [{"instruction": f"计算 {i} 加 2，只返回数字。", "input": "", "output": str(i + 2)} for i in range(1, 33)],
        "validation.json": [{"instruction": f"计算 {i} 加 3，只返回数字。", "input": "", "output": str(i + 3)} for i in range(101, 109)],
        "questions.json": [{"question_id": "demo_q1", "prompt": "计算17加25，只返回数字。", "answer": "42"},
                           {"question_id": "demo_q2", "prompt": "用一句话说明你能帮助做什么。"}],
        "invalid_missing_output.json": [{"instruction": "演示缺少答案字段"}],
        "invalid_empty_text.json": [{"instruction": "", "output": "示例"}],
        "invalid_long_text.json": [{"instruction": "长" * 32769, "output": "示例"}]
    }
    for name, value in files.items():
        (folder / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    (folder / "README.md").write_text("# 工程验证演示数据\n\n仅用于验证接口、训练执行和错误处理，不用于证明航天领域能力或业务成绩。\n", encoding="utf-8")
    return folder


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    generate(parser.parse_args().directory)
