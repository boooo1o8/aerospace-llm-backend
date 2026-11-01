"""Create account-private configuration and a scoped service key, without printing it."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    parser.add_argument("--project", default="demo-project")
    parser.add_argument("--actor", default="demo-user")
    parser.add_argument("--model-path", type=Path, default=Path.home() / "models/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--llamafactory-root", type=Path, default=Path.home() / "projects/LLaMAFactory")
    parser.add_argument("--data-root", type=Path, default=Path.home() / ".local/share/aerospace-backend")
    parser.add_argument("--model-root", type=Path, default=Path.home() / "models/aerospace-backend")
    args = parser.parse_args()
    os.umask(0o077)
    folder = Path(args.directory).resolve()
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    paths = [folder / n for n in ("config.json", "credentials.json", "service.key", "cursor.key")]
    if any(path.exists() for path in paths):
        raise SystemExit("Configuration already exists; refusing to replace credentials")
    config = json.loads((Path(__file__).resolve().parents[1] / "config.example.json").read_text())
    key = secrets.token_urlsafe(48)
    config.update(credentials_file=str(paths[1]), cursor_key_file=str(paths[3]))
    config.update(python=sys.executable, data_root=str(args.data_root.resolve()),
                  model_root=str(args.model_root.resolve()),
                  llamafactory_root=str(args.llamafactory_root.resolve()))
    config["models"][0]["path"] = str(args.model_path.resolve())
    credentials = {"services": [{"service_id": "middle-demo", "token_sha256": hashlib.sha256(key.encode()).hexdigest(),
                                 "projects": [args.project], "actors": [args.actor]}]}
    for path, contents in zip(paths, [json.dumps(config, indent=2), json.dumps(credentials, indent=2), key + "\n", secrets.token_hex(32) + "\n"]):
        with path.open("x") as handle:
            handle.write(contents)
        path.chmod(0o600)
    print("Created private configuration in", folder, "(credentials not printed)")


if __name__ == "__main__":
    main()
