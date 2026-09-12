"""Нарезка web/gpt2_124m.onnx на части < лимита GitHub (100MB/файл).

Части собираются в браузере в ArrayBuffer (web/main.js), затем
ort.InferenceSession.create(buf). После нарезки в web/config.json
добавляется ключ "model_chunks" — список имён частей; он же может быть
заменён на абсолютные URL (например, GitHub Release assets).

Использование:
  ./venv/bin/python scripts/split_model.py            # части по 90 МБ в web/model/
  ./venv/bin/python scripts/split_model.py --undo     # убрать части и ключ из config
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WEB = REPO / "web"
MODEL = WEB / "gpt2_124m.onnx"
CHUNK_MB = 90


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--size", type=int, default=CHUNK_MB, help="размер части, МБ")
    p.add_argument("--undo", action="store_true", help="удалить части и ключ model_chunks")
    args = p.parse_args()

    config_path = WEB / "config.json"
    config = json.loads(config_path.read_text())

    if args.undo:
        for f in WEB.glob("model/gpt2_124m.part-*"):
            f.unlink()
        model_dir = WEB / "model"
        if model_dir.exists() and not any(model_dir.iterdir()):
            model_dir.rmdir()
        config.pop("model_chunks", None)
        config_path.write_text(json.dumps(config, indent=2))
        print("части удалены, model_chunks убран из config.json")
        return

    if not MODEL.exists():
        sys.exit(f"нет {MODEL} — сначала scripts/export_onnx.py")

    out_dir = WEB / "model"
    out_dir.mkdir(exist_ok=True)
    chunk = args.size * 2**20
    data = MODEL.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    parts = []
    for i, off in enumerate(range(0, len(data), chunk)):
        name = f"gpt2_124m.part-{i:03d}"
        (out_dir / name).write_bytes(data[off:off + chunk])
        parts.append(f"model/{name}")
        print(f"{name}: {min(chunk, len(data) - off) / 2**20:.1f} MiB")

    config["model_sha256"] = digest
    config["model_size"] = len(data)
    config["model_chunks"] = parts
    config_path.write_text(json.dumps(config, indent=2))
    print(f"config.json: model_chunks ({len(parts)} частей), sha256 {digest[:16]}...")


if __name__ == "__main__":
    main()
