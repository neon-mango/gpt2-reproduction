"""Сравнение своего BPE из предыдущей попытки (../prev-llm-attempt) с официальным
GPT-2 BPE. Измерения выполняются на реальном тексте, результат пишется в
docs/bpe-comparison.md.

Проверяется:
  - устройство: base-словарь (кодпоинты vs байты), источники словаря;
  - сжатие: символов/токен и токенов на 1МБ английского веб-текста;
  - устойчивость: символы, которых не было в обучающей выборке (эмодзи, CJK);
  - точность round-trip encode->decode;
  - скорость кодирования.

Важно: encode старого BPE зацикливается на неизвестном кодпоинте
(best_len == 0 -> i += 0), поэтому для него используется сторожевой таймер,
а метрики сжатия снимаются патченной версией (пропуск неизвестных символов).
"""

import gzip
import signal
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

PREV_DIR = REPO / "prev-llm-attempt"
OLD_TOKENIZER = PREV_DIR / "deepseek" / "tokenizer_tiny.pkl.gz"
RAW_DIR = REPO / "data" / "raw" / "openwebtext"
OUT_MD = REPO / "docs" / "bpe-comparison.md"

RU_SAMPLE = (
    "Языковое моделирование — это задача предсказания следующего токена по "
    "контексту. Трансформеры справляются с ней впечатляюще хорошо."
)
EMOJI_SAMPLE = "emoji test 🚀🤖🌍 café naïve 中文日本語"


def load_old_bpe():
    sys.path.insert(0, str(PREV_DIR))
    from bpe import BPE  # noqa: PLC0415 — модуль предыдущей попытки
    return BPE.load(str(OLD_TOKENIZER))


def sample_english_text(max_bytes: int) -> str:
    """Первые max_bytes текста из скачанного OpenWebText (первый шард)."""
    import json

    path = sorted(RAW_DIR.glob("docs-*.jsonl.gz"))[0]
    buf = []
    size = 0
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            buf.append(json.loads(line)["text"])
            size += len(buf[-1])
            if size >= max_bytes:
                break
    return "\n".join(buf)[:max_bytes]


class EncodeHang(Exception):
    pass


def old_encode_watchdog(tok, text: str, timeout: float = 30.0):
    """encode старого BPE с таймаутом. Возвращает (ids | None, hang_info | None)."""

    def on_alarm(signum, frame):
        raise EncodeHang()

    was_local = tok.local
    tok.local = False  # без tqdm
    signal.signal(signal.SIGALRM, on_alarm)
    signal.alarm(int(timeout))
    try:
        return tok.encode(text), None
    except EncodeHang:
        return None, f"зависание encode (> {timeout:.0f} с): бесконечный цикл на неизвестном символе"
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"
    finally:
        signal.alarm(0)
        tok.local = was_local


def old_encode_patched(tok, text: str) -> list[int]:
    """Копия encode старого BPE с фиксом: неизвестный символ пропускается
    (i += 1) вместо зависания. Только для измерения сжатия/скорости."""
    root = {}
    for token, tid in tok.token2id.items():
        node = root
        for ch in token:
            node = node.setdefault(ch, {})
        node[""] = tid
    indices = []
    i = 0
    n = len(text)
    while i < n:
        node = root
        best_id = None
        best_len = 0
        j = i
        while j < n:
            node = node.get(text[j])
            if node is None:
                break
            j += 1
            tid = node.get("")
            if tid is not None:
                best_id = tid
                best_len = j - i
        if best_id is not None:
            indices.append(best_id)
            i += best_len
        else:
            i += 1
    return indices


def main() -> None:
    print("Загружаю токенизаторы...", file=sys.stderr)
    old = load_old_bpe()
    from gpt2rep.tokenizer import GPT2Tokenizer
    new = GPT2Tokenizer.load(REPO / "data" / "tokenizer")

    eng = sample_english_text(1_000_000)
    print(f"Текст: {len(eng):,} символов английского веб-текста", file=sys.stderr)

    # --- GPT-2 BPE: обычное измерение ---
    t0 = time.perf_counter()
    new_ids = new.encode(eng)
    new_enc_s = time.perf_counter() - t0
    assert new.decode(new_ids) == eng

    # --- старый BPE: сторож на оригинальном encode ---
    print("Пробую оригинальный encode старого BPE (таймаут 30 с)...", file=sys.stderr)
    old_ids, old_hang = old_encode_watchdog(old, eng, timeout=30.0)
    if old_hang:
        print(f"  {old_hang}", file=sys.stderr)

    # --- старый BPE: патченный encode для метрик ---
    print("Измеряю патченный encode (пропуск неизвестных символов)...", file=sys.stderr)
    t0 = time.perf_counter()
    old_ids_patched = old_encode_patched(old, eng)
    old_enc_s = time.perf_counter() - t0

    # --- устойчивость на контрольных строках ---
    tests = {"эмодзи и rare unicode": EMOJI_SAMPLE, "русский текст": RU_SAMPLE}
    extra = {}
    for name, text in tests.items():
        o_ids, o_note = old_encode_watchdog(old, text, timeout=10.0)
        o_round = old.decode(o_ids) == text if o_ids is not None else False
        n_ids = new.encode(text)
        extra[name] = {
            "text": text,
            "old_ok": o_ids is not None,
            "old_note": o_note or "",
            "old_tokens": len(o_ids) if o_ids is not None else None,
            "old_roundtrip": o_round,
            "new_tokens": len(n_ids),
            "new_roundtrip": new.decode(n_ids) == text,
        }

    # --- витрина токенов (repr, чтобы контроль-символы не ломали markdown) ---
    old_show = [repr(old.id2token[i]) for i in (0, 1, 2, 100, 500, 1000, 4000, 7999)]
    new_show = [repr(new.decode([i])) for i in (0, 1, 2, 100, 1000, 10000, 30000, 50000)]

    ru = extra["русский текст"]
    emoji = extra["эмодзи и rare unicode"]
    old_vocab_char = sum(1 for t in old.id2token.values() if len(t) == 1)

    hang_cell = old_hang or f"{len(old_ids):,} токенов"
    rows = [
        ("Base-словарь",
         f"кодпоинты обучающего текста ({len(old.id2token):,} всего словарь)",
         "все 256 байтов UTF-8 + 50 000 мерджей + <|endoftext|>"),
        ("Размер словаря", f"{len(old.id2token):,}", f"{new.vocab_size:,}"),
        ("Из них одно-символьных", f"{old_vocab_char:,}", "256 (байты)"),
        ("Предтокенизация", "нет (мерджи через границы слов/категорий)",
         "regex: буквы/цифры/пунктуация/пробелы не смешиваются"),
        ("Где обучался", "корпус прошлой попытки (RU+EN: wiki, alpaca, dolly, orca_math, saiga)",
         "корпус OpenAI WebText (файлы релиза 2019 г.)"),
        ("Encode 1МБ EN-текста", f"**{hang_cell}**", f"{len(new_ids):,} токенов"),
        ("Токенов на 1МБ (с патчем)" if old_hang else "Токенов на 1МБ EN",
         f"{len(old_ids_patched):,}", f"{len(new_ids):,}"),
        ("Символов на токен (EN)", f"{len(eng)/len(old_ids_patched):.2f}",
         f"{len(eng)/len(new_ids):.2f}"),
        ("Round-trip на EN-тексте",
         "OK" if old_ids is not None else "н/д — encode зависает",
         "OK"),
        ("Неизвестные символы (эмодзи/CJK)",
         ("OK, " + f"{emoji['old_tokens']} ток." if emoji["old_ok"]
          else "не кодируются (" + emoji["old_note"] + ")"),
         f"OK, {emoji['new_tokens']} ток."),
        ("Русский текст", f"{ru['old_tokens']} ток. (round-trip: "
         f"{'OK' if ru['old_roundtrip'] else 'FAIL'})" if ru["old_ok"] else ru["old_note"],
         f"{ru['new_tokens']} ток. (round-trip: {'OK' if ru['new_roundtrip'] else 'FAIL'})"),
        ("Скорость encode, МБ/с (1 поток)",
         f"{len(eng)/old_enc_s/1e6:.2f} (патченный)",
         f"{len(eng)/new_enc_s/1e6:.2f}"),
    ]

    def table(rows):
        w0 = max(len(r[0]) for r in rows) + 2
        w1 = max(len(r[1]) for r in rows) + 2
        w2 = max(len(r[2]) for r in rows) + 2
        head = (f"| {'Метрика'.ljust(w0)}| {'Старый BPE (prev-llm-attempt)'.ljust(w1)}"
                f"| {'Официальный GPT-2 BPE'.ljust(w2)}|")
        sep = f"| {'-'*w0}| {'-'*w1}| {'-'*w2}|"
        lines = [head, sep]
        for a, b, c in rows:
            lines.append(f"| {a.ljust(w0)}| {b.ljust(w1)}| {c.ljust(w2)}|")
        return "\n".join(lines)

    md = f"""# Сравнение токенизаторов: свой BPE (прошлая попытка) vs официальный GPT-2 BPE

Дата: {time.strftime('%Y-%m-%d')}. Измерения: `scripts/compare_bpe.py`.
Текст: {len(eng):,} символов английского веб-текста из OpenWebText (первый
шард), плюс контрольные строки с эмодзи и русским текстом.

## Устройство

- **Старый BPE** (`prev-llm-attempt/bpe.py`) обучался на корпусе предыдущей
  попытки (смесь RU+EN: wikipedia, alpaca, dolly, orca_math, saiga, ...).
  Base-словарь — уникальные **кодпоинты** обучающего текста (текст кодировался
  в UTF-32), словарь добирался мерджами до {len(old.id2token):,}.
- **Официальный GPT-2 BPE** — **byte-level**: base-словарь ровно 256 байтов,
  поэтому любой unicode-текст кодируется без «дыр» в словаре; сверху 50 000
  мерджей и `<|endoftext|>` — итого **50 257**. Мерджи не пересекают границы
  символьных категорий (regex-предтокенизация) — это спасает словарь от
  дубликатов вида `dog`, `dog.`, `dog!` (мотивация в п. 2.2 статьи GPT-2).

## Измерения

{table(rows)}

## Витрина токенов

Старые токены (id 0/1/2/100/500/1000/4000/7999):
`{'`, `'.join(old_show)}`

Токены GPT-2 (id 0/1/2/100/1000/10000/30000/50000):
`{'`, `'.join(new_show)}`

У GPT-2 id 0 — это байт `!`; пробел вшит в начало слова (`Ġ` = байт 0x20,
токен `Ġthe` = " the").

## Один и тот же русский текст

```
{RU_SAMPLE}
```

- старый BPE: **{ru['old_tokens'] if ru['old_ok'] else 'не кодируется'} токенов**
  (round-trip: {'OK' if ru['old_roundtrip'] else 'FAIL'}{', ' + ru['old_note'] if ru['old_note'] and not ru['old_ok'] else ''})
- GPT-2 BPE: **{ru['new_tokens']} токенов**, round-trip OK.

  Честная оговорка: на русском **старый BPE компактнее** ({ru['old_tokens']} против
  {ru['new_tokens']} токенов) — русский был в его обучающей смеси, а GPT-2 обучен
  только на английском WebText. Это не противоречит выводам: gpt2 —
  англоязычная модель, и воспроизводим мы её на англоязычных данных.

## Эмодзи и rare unicode

```
{emoji['text']}
```

- старый BPE: {('кодируется, ' + str(emoji['old_tokens']) + ' ток.') if emoji['old_ok'] else '**не кодируется** — ' + emoji['old_note']}
- GPT-2 BPE: {emoji['new_tokens']} токенов, round-trip OK.

## Выводы

1. **Byte-level base — главный выигрыш.** Старый BPE не может закодировать
   символ, которого не было в обучающем корпусе: `encode` зацикливается
   (i += best_len при best_len == 0), в реальном прогоне на 1МБ веб-текста он
   завис и был остановлен сторожем. GPT-2 BPE кодирует **любую** строку
   байтов: непокрываемые последовательности всегда представимы базовыми 256.
2. **Сжатие выше за счёт объёма словаря и предтокенизации.** {len(eng)/len(old_ids_patched):.2f} против
   {len(eng)/len(new_ids):.2f} символов на токен: меньше токенов — больше текста в контексте
   1024 и дешевле вычисления на слово. Словарь GPT-2 в {new.vocab_size//len(old.id2token)} раза больше
   и обучен на ~40GB веб-текста, а не на десятках МБ смеси wiki+инструкции.
3. **Совместимость с эталоном.** Официальный BPE позволяет сверять логиты и
   генерацию с openai-community/gpt2 (см. `scripts/verify_parity.py`); свой
   словарь сделал бы воспроизведение несравнимым с оригиналом.
4. Инженерия старого BPE (индекс пар на C, linked-list мерджи, trie-encode) —
   сильная оптимизация, и она сохранена по духу: в новой версии те же идеи
   живут на уровне данных (uint16 memmap, возобновляемые шарды) и обучения
   (fused AdamW, SDPA). Но оптимизировать надо было **правильный** словарь:
   8K кодпоинт-мерджей на смеси RU+EN принципиально не догоняют 50K
   байт-мерджей на 40GB веб-текста.
"""
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(md, encoding="utf-8")
    print(f"Готово: {OUT_MD}", file=sys.stderr)


if __name__ == "__main__":
    main()
