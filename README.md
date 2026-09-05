# gpt2-reproduction

Воспроизведение [openai-community/gpt2](https://huggingface.co/openai-community/gpt2)
(124M) по статье [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf):
своя реализация архитектуры, официального byte-level BPE и пайплайна обучения
на OpenWebText. Всё с нуля, без `transformers` в рантайме обучения.

**Корректность реализации уже доказана**: с эталонными весами OpenAI наша
модель выдаёт логиты **побитово идентичные** HF gpt2, токенизатор
идентичен, жадная генерация совпадает до символа — `scripts/verify_parity.py`.

Анализ предыдущей попытки (что было не так): [docs/analysis-previous-attempt.md](docs/analysis-previous-attempt.md).
Сравнение старого и нового токенизаторов: [docs/bpe-comparison.md](docs/bpe-comparison.md).

## Быстрый старт

```bash
# 1. Окружение (Python 3.14; torch ставится с CUDA, работает и на CPU)
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
# если у видеокарты старый драйвер: pip install torch --index-url https://download.pytorch.org/whl/cu126

# 2. Токенизатор GPT-2 (encoder.json + vocab.bpe, ~1.5MB)
./venv/bin/python scripts/download_tokenizer.py

# 3. (опционально, разово) Проверка паритета с эталоном HF (~550MB скачиваний)
./venv/bin/python scripts/verify_parity.py

# 4. Данные: подвыборка OpenWebText (по умолчанию 500k документов ~ 920MB gz)
./venv/bin/python scripts/download_data.py            # --full для всего корпуса (~8M документов)

# 5. Токенизация в train.bin/val.bin (uint16, читаются как memmap)
./venv/bin/python scripts/prepare_tokens.py

# 6. Обучение GPT-2 124M (значения по умолчанию = статья; под 10GB VRAM)
./venv/bin/python train.py

# 7. Генерация
./venv/bin/python generate.py --prompt "The meaning of life is" --temperature 0.8 --top-k 50
```

На CPU всё проверяется на уменьшенной модели, например:

```bash
./venv/bin/python train.py --n-layer 2 --n-embd 128 --n-head 2 \
  --batch-size 4 --grad-accum 2 --max-iters 30 --eval-iters 3
./venv/bin/python generate.py --ckpt checkpoints/run1/best.pt --tokens 50
```

## Возобновляемость: каждый этап продолжается с места остановки

Как и в предыдущей версии, любой этап можно прервать и запустить заново —
продолжение автоматическое, с обработкой битых хвостов:

| Этап | Артефакты | Как продолжается |
|---|---|---|
| `download_tokenizer.py` | `data/tokenizer/*.json,bpe` | пропускает существующие файлы |
| `download_data.py` | `data/raw/openwebtext/docs-*.jsonl.gz` | шарды по 10k документов; пересчитывает прогресс; обрыв середины записи (битый gzip) удаляется и перекачивается |
| `prepare_tokens.py` | `data/tokens/train.bin, val.bin` | дописывает в конец .bin, пропуская уже закодированное; план прогона фиксируется в `meta.json` (при докачке корпуса попросит перетокенизировать) |
| `train.py` | `checkpoints/run1/last.pt` | модель + **оптимизатор** + номер шага + состояние ГПСЧ + best_val; resume по умолчанию включён (`--no-resume` чтобы начать заново); чекпоинты атомарные (`.tmp` + rename) |
| `train.py` (best) | `checkpoints/run1/best.pt` | лучшая модель по val_loss — для генерации |
| `verify_parity.py`, `compare_bpe.py` | отчёты | идемпотентны, можно запускать повторно |

## Гиперпараметры

Модель — точно по статье (Table 2) и HF-конфигу gpt2:

| Параметр | Значение | Комментарий |
|---|---|---|
| n_layer / n_embd / n_head | 12 / 768 / 12 | 124 439 808 параметров, ровно как у эталона |
| n_positions (контекст) | 1024 | |
| vocab | 50 257 | официальный byte-level BPE OpenAI |
| активация | GELU (tanh) | `gelu_new` |
| нормализация | pre-LN + финальный ln_f, eps 1e-5 | |
| tying | lm_head = wte | |
| dropout | 0.1 (attn/resid/embd) | |
| init | N(0, 0.02), c_proj ÷ √(2·n_layer) | «modified initialization» из статьи |

Обучение — по статье, с задокументированными отклонениями (см. docstring
`train.py`): батч 512×1024 = 524 288 токенов/шаг (через gradient accumulation),
lr 2.5e-4, warmup 2000, cosine decay до 6%, grad clip 1.0. Отклонения: AdamW
(β2=0.95, wd 0.1 на матрицах — практика nanoGPT/GPT-3, стабильнее plain Adam),
bf16 autocast (иначе 10GB VRAM не хватит), окна данных случайные на каждый шаг
(а не скользящее окно с шагом 1, как было раньше).

## Запуск на 3080 (когда будет подключена) — пошагово

Все команды ниже выполнять из корня репозитория, окружение уже настроено
(venv + torch с CUDA). Каждый шаг опирается на артефакты предыдущего;
прерывания безопасны — всё возобновляется (см. таблицу выше).

```bash
# 0. Проверить, что карта видна и torch её использует
nvidia-smi
./venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# если CUDA недоступна при живом nvidia-smi — драйвер старее сборки torch,
# тогда: ./venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu126

# 1. Smoke-тест на GPU: 30 шагов 124M, заодно замер tok/s (~3-5 минут)
./venv/bin/python train.py --out-dir checkpoints/smoke --batch-size 8 --grad-accum 64 \
  --max-iters 30 --warmup-steps 10 --eval-interval 10 --eval-iters 5 --no-resume --log-interval 10
# если OOM на bf16: --batch-size 4 --grad-accum 128

# 2. Основной прогон: 2.5B токенов (Chinchilla-оптимум для 124M), ~1 сутки
./venv/bin/python train.py --batch-size 8 --grad-accum 64 --max-tokens 2.5e9 --compile
#    --compile: первый шаг занимает пару минут, дальше шаги быстрее.
#    Следить за прогрессом (в другом терминале):
tail -f checkpoints/run1/train.log

# 3. Прервали/упали/выключили машину — просто запустить ту же команду ещё раз:
#    обучение продолжится с checkpoints/run1/last.pt (шаг, оптимизатор, ГПСЧ).

# 4. Сгенерировать из лучшего чекпоинта
./venv/bin/python generate.py --prompt "The meaning of life is" --temperature 0.8 --top-k 50
./venv/bin/python generate.py --ckpt checkpoints/run1/last.pt --tokens 400

# 5. (Опционально) Полный масштаб: докачать весь OpenWebText (~8M документов,
#    часы скачивания) и перетокенизировать, затем прогон ~8-9B токенов (~3 дня)
./venv/bin/python scripts/download_data.py --full
rm data/tokens/train.bin data/tokens/val.bin data/tokens/meta.json
./venv/bin/python scripts/prepare_tokens.py
./venv/bin/python train.py --batch-size 8 --grad-accum 64 --max-tokens 8e9 --compile --out-dir checkpoints/run_full
```

Замечания:
- smoke из п. 1 пишет в отдельный `checkpoints/smoke` и не влияет на основной
  прогон; основной прогресс живёт в `checkpoints/run1` (`train.log`, `last.pt`,
  `best.pt`).
- бюджет прогонов:

| Прогон | Токенов | Время (оценка) | Что получится |
|---|---|---|---|
| smoke (п. 1) | — | минуты | замер tok/s, проверка пайплайна на GPU |
| Chinchilla (п. 2) | 2.5B | ~1 сутки | связный веб-текст |
| масштаб WebText (п. 5) | ~8–9B | ~2.5–4 дня | близко к качеству оригинала (оригинал underfit WebText) |

- скорость записи ~30–50K tok/s на 124M в bf16; в логе печатаются tok/s,
  s/шаг и ETA. Скачанной подвыборки (560M токенов) уже хватает на Chinchilla-прогон
  — это ~4.5 прохода по ней (без дедупликации — терпимо, но корпус `--full`
  ~9B токенов лучше, единый проход без повторов).


## Структура репозитория

```
gpt2rep/
  tokenizer.py     # byte-level BPE GPT-2 с нуля (regex + merges + cache)
  model.py         # архитектура GPT-2 (pre-LN, GELU, tying, scaled init, SDPA)
scripts/
  download_tokenizer.py  # официальные encoder.json + vocab.bpe
  download_data.py       # OpenWebText: подвыборка или --full, возобновляемо
  prepare_tokens.py      # токенизация корпуса в train.bin/val.bin, параллельно
  verify_parity.py       # сверка токенизатора и логитов с эталоном HF
  compare_bpe.py         # измерения: старый BPE vs GPT-2 BPE -> docs/
train.py           # обучение (resume, AMP, cosine LR, grad clip, чекпоинты)
generate.py        # сэмплирование (temperature/top-k/top-p/greedy)
docs/
  analysis-previous-attempt.md  # почему прошлая попытка была хуже
  bpe-comparison.md             # сравнение токенизаторов с измерениями
prev-llm-attempt -> /home/valeriy/custom-llm  # локальная ссылка на прошлую попытку
```

## Примечания

- `transformers` нужен только `verify_parity.py`; обучение и генерация его не
  используют.
- WebText не опубликован; стандартная открытая замена — OpenWebText
  (та же идея: исходящие ссылки Reddit 3+ karma). Полный корпус ~8M документов
  (~38GB текста), качается стримингом shardами.
- Оригинальные файлы токенизатора берутся из открытого бакета OpenAI
  (зеркало — HF), поэтому последовательность токенов совпадает с эталоном.
