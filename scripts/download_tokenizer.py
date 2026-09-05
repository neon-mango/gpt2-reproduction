"""Скачивает официальные файлы GPT-2 BPE (encoder.json + vocab.bpe).

Источники (первый доступный):
  1. открытый бакет OpenAI (как в оригинальном релизе gpt-2),
  2. зеркало на Hugging Face (openai-community/gpt2).

Файлы кладутся в data/tokenizer/ и далее используются всеми скриптами.
"""

import sys
from pathlib import Path

import requests
from tqdm import tqdm

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "data" / "tokenizer"

SOURCES = [
    # (имя файла, [url, ...])
    (
        "encoder.json",
        [
            "https://openaipublic.blob.core.windows.net/gpt-2/models/124M/encoder.json",
            "https://huggingface.co/openai-community/gpt2/resolve/main/vocab.json",
        ],
    ),
    (
        "vocab.bpe",
        [
            "https://openaipublic.blob.core.windows.net/gpt-2/models/124M/vocab.bpe",
            "https://huggingface.co/openai-community/gpt2/resolve/main/merges.txt",
        ],
    ),
]


def download(fname: str, urls: list[str]) -> Path:
    out = OUT_DIR / fname
    if out.exists() and out.stat().st_size > 0:
        print(f"{out} уже существует, пропускаю")
        return out
    last_err: Exception | None = None
    for url in urls:
        try:
            print(f"Скачиваю {url} ...")
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                tmp = out.with_suffix(out.suffix + ".tmp")
                with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=fname) as bar:
                    for chunk in r.iter_content(chunk_size=1 << 16):
                        f.write(chunk)
                        bar.update(len(chunk))
            tmp.replace(out)
            return out
        except Exception as e:  # noqa: BLE001 — пробуем следующий источник
            last_err = e
            print(f"  не удалось: {e}", file=sys.stderr)
    raise RuntimeError(f"Не удалось скачать {fname}: {last_err}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for fname, urls in SOURCES:
        download(fname, urls)
    print(f"Готово: {OUT_DIR}")


if __name__ == "__main__":
    main()
