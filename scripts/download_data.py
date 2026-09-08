"""Скачивание OpenWebText — открытая реконструкция WebText из статьи GPT-2.

Оригинальный WebText не опубликован, поэтому стандарт замены — датасет
openwebtext (https://huggingface.co/datasets/openwebtext): ~8M документов,
~38GB текста, собранных по той же идее (исходящие ссылки с Reddit 3+ karma).

Режимы:
  --max-docs N   подвыборка первых N документов (быстрый первый прогон),
  --full         весь датасет (~38GB текста на диск, часы на скачивание).

Документы пишутся в data/raw/openwebtext/ шардами docs-XXXXX.jsonl.gz
({"text": ...} на строку). Скачивание возобновляемо: готовые шарды пропускаются.
"""

import argparse
import gzip
import json
import os
import sys
import zlib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

OUT_DIR = REPO / "data" / "raw" / "openwebtext"
DOCS_PER_SHARD = 10_000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-docs", type=int, default=500_000,
                        help="сколько документов скачать (0 = как --full)")
    parser.add_argument("--full", action="store_true",
                        help="скачать весь датасет (~8M документов)")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--shard-size", type=int, default=DOCS_PER_SHARD)
    args = parser.parse_args()
    limit = None if (args.full or args.max_docs == 0) else args.max_docs

    from datasets import load_dataset
    from tqdm import tqdm

    args.out_dir.mkdir(parents=True, exist_ok=True)

    shard_idx = 0
    total = 0
    while (args.out_dir / f"docs-{shard_idx:05d}.jsonl.gz").exists():
        shard_idx += 1
    # пересчитываем, сколько документов уже скачано; хвостовой шард мог быть
    # повреждён (обрыв в середине записи: EOFError/zlib.error/OSError) —
    # чиним циклом, битых хвостов может быть несколько
    in_last = 0
    while shard_idx > 0:
        last = args.out_dir / f"docs-{shard_idx - 1:05d}.jsonl.gz"
        try:
            with gzip.open(last, "rt", encoding="utf-8") as f:
                in_last = sum(1 for _ in f)
            break
        except (EOFError, OSError, zlib.error, ValueError):
            print(f"{last.name} повреждён — удаляю, шард будет перекачан")
            last.unlink()
            shard_idx -= 1
            in_last = 0
    total = shard_idx * args.shard_size + in_last
    if total:
        print(f"Найден готовый прогресс: {total} документов, продолжаю с шарда {shard_idx}")

    if limit is not None and total >= limit:
        print(f"Лимит {limit} документов уже скачан")
        return

    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    # пропускаем уже скачанное (стрики идут в детерминированном порядке)
    if total:
        ds = ds.skip(total)

    # шард пишется в .tmp и атомарно переименовывается при закрытии: обрыв
    # в середине записи не оставит битого файла под финальным именем
    final_path = args.out_dir / f"docs-{shard_idx:05d}.jsonl.gz"
    tmp_path = final_path.with_name(final_path.name + ".tmp")
    fout = gzip.open(tmp_path, "wt", encoding="utf-8")
    n_in_shard = 0
    pbar = tqdm(initial=total, desc="openwebtext", unit="doc")
    try:
        for row in ds:
            text = row["text"]
            if not text or not text.strip():
                continue
            fout.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            total += 1
            n_in_shard += 1
            pbar.update(1)
            if n_in_shard >= args.shard_size:
                fout.close()
                os.replace(tmp_path, final_path)
                shard_idx += 1
                final_path = args.out_dir / f"docs-{shard_idx:05d}.jsonl.gz"
                tmp_path = final_path.with_name(final_path.name + ".tmp")
                fout = gzip.open(tmp_path, "wt", encoding="utf-8")
                n_in_shard = 0
            if limit is not None and total >= limit:
                break
    finally:
        fout.close()
        if n_in_shard > 0:
            # частичный хвостовой шард — валидный gzip, resume досчитает строки
            os.replace(tmp_path, final_path)
        else:
            tmp_path.unlink(missing_ok=True)
        pbar.close()

    print(f"Скачано {total} документов -> {args.out_dir}")


if __name__ == "__main__":
    main()
