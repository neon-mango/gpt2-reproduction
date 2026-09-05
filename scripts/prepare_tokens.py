"""Токенизация корпуса официальным GPT-2 BPE в бинарники train.bin / val.bin.

Формат: плотный uint16-массив (макс. id 50256 < 65535), читается через
np.memmap без загрузки в RAM. Документы разделяются <|endoftext|> (50256),
как в оригинальном пайплайне OpenAI.

Валидация: последние --val-docs документов корпуса (не пересекаются с train).
Токенизация параллелится по процессам, результаты пишутся в порядке документов.
Подготовка возобновляемая: готовые .bin не пересчитываются, кодирование
продолжается с места остановки.
"""

import argparse
import gzip
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from gpt2rep.tokenizer import GPT2Tokenizer

CHUNK_DOCS = 512  # документов на задачу воркера

_STATE = {}  # состояние воркера (инициализируется один раз на процесс)


def _worker_init(tokenizer_dir: str) -> None:
    _STATE["tok"] = GPT2Tokenizer.load(tokenizer_dir)


def _encode_chunk(task: tuple[bool, list[str]]) -> tuple[bool, int, bytes]:
    is_val, texts = task
    tok = _STATE["tok"]
    eot = tok.eot_token
    out: list[int] = []
    for t in texts:
        out.extend(tok.encode(t))
        out.append(eot)
    return is_val, len(texts), np.asarray(out, dtype=np.uint16).tobytes()


def iter_docs(raw_paths: list[Path]):
    for path in raw_paths:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)["text"]


def count_docs(raw_paths: list[Path]) -> int:
    n = 0
    for path in raw_paths:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            n += sum(1 for _ in f)
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=REPO / "data" / "raw" / "openwebtext")
    parser.add_argument("--out-dir", type=Path, default=REPO / "data" / "tokens")
    parser.add_argument("--tokenizer-dir", type=Path, default=REPO / "data" / "tokenizer")
    parser.add_argument("--val-docs", type=int, default=5_000,
                        help="документов в валидации (берутся с конца корпуса)")
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    args = parser.parse_args()

    raw_paths = sorted(args.raw_dir.glob("*.jsonl.gz"))
    if not raw_paths:
        sys.exit(f"Нет файлов {args.raw_dir}/docs-*.jsonl.gz — сначала запустите download_data.py")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = GPT2Tokenizer.load(args.tokenizer_dir)

    total_docs = count_docs(raw_paths)
    val_docs = min(args.val_docs, max(0, total_docs - 1))
    train_docs = total_docs - val_docs

    train_path = args.out_dir / "train.bin"
    val_path = args.out_dir / "val.bin"
    done_train = train_path.stat().st_size // 2 if train_path.exists() else 0
    done_val = val_path.stat().st_size // 2 if val_path.exists() else 0
    skip_docs = done_train + done_val
    print(f"Документов: {total_docs} (train {train_docs}, val {val_docs}); "
          f"уже закодировано: train {done_train}, val {done_val}")
    if skip_docs >= total_docs:
        print("Всё уже токенизировано")
    else:
        meta_path = args.out_dir / "meta.json"
        if skip_docs == 0:
            # фиксируем план первого прогона; при возобновлении сверяемся с ним
            meta_path.write_text(json.dumps({
                "total_docs": total_docs, "train_docs": train_docs, "val_docs": val_docs,
            }))
        else:
            if not meta_path.exists():
                sys.exit("meta.json не найден, а .bin есть — корпус менялся? Удалите "
                         f"{train_path} и {val_path} и запустите prepare_tokens заново")
            plan = json.loads(meta_path.read_text())
            if plan.get("total_docs") != total_docs:
                sys.exit(f"meta.json зафиксировал {plan['total_docs']} документов, а сейчас "
                         f"{total_docs} — корпус докачивался. Удалите {train_path}, {val_path} "
                         "и meta.json, затем запустите prepare_tokens заново")
            if plan.get("train_docs") != train_docs:
                train_docs = plan["train_docs"]
                val_docs = plan["val_docs"]

        seen = skip_docs

        def task_iter():
            """Порождает задачи (is_val, chunk) с корректным сплитом train/val."""
            nonlocal seen
            batch: list[str] = []
            batch_is_val = False
            for text in iter_docs(raw_paths):
                if seen < skip_docs:
                    seen += 1
                    continue
                is_val = seen >= train_docs
                seen += 1
                if batch and is_val != batch_is_val:
                    # граница train/val внутри батча — отдаём батч как есть
                    yield batch_is_val, batch
                    batch = []
                batch_is_val = is_val
                batch.append(text)
                if len(batch) >= CHUNK_DOCS:
                    yield batch_is_val, batch
                    batch = []
            if batch:
                yield batch_is_val, batch

        with mp.get_context("spawn").Pool(
            args.workers, initializer=_worker_init, initargs=(str(args.tokenizer_dir),)
        ) as pool:
            from tqdm import tqdm

            bar = tqdm(total=total_docs - skip_docs, desc="tokenize", unit="doc")
            with open(train_path, "ab") as ftrain, open(val_path, "ab") as fval:
                # imap сохраняет порядок задач и распараллеливает кодирование
                for is_val, n_docs, payload in pool.imap(_encode_chunk, task_iter(), chunksize=1):
                    (fval if is_val else ftrain).write(payload)
                    bar.update(n_docs)
            bar.close()

    final_train = train_path.stat().st_size // 2
    final_val = val_path.stat().st_size // 2
    meta_path = args.out_dir / "meta.json"
    plan = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    plan.update({
        "train_tokens": final_train, "val_tokens": final_val,
        "vocab_size": tokenizer.vocab_size,
    })
    meta_path.write_text(json.dumps(plan, indent=2))
    print(f"train: {final_train} токенов ({train_path}), val: {final_val} токенов ({val_path})")


if __name__ == "__main__":
    main()
