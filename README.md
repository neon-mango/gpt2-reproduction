# gpt2-reproduction

Воспроизведение [openai-community/gpt2](https://huggingface.co/openai-community/gpt2)
(124M) по статье [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf):
своя реализация архитектуры, официального byte-level BPE и пайплайна обучения
на OpenWebText. Всё с нуля, без `transformers` в рантайме обучения.

**Корректность реализации доказана**: с эталонными весами OpenAI наша
модель выдаёт логиты **побитово идентичные** HF gpt2, токенизатор
идентичен, жадная генерация совпадает до символа — `scripts/verify_parity.py`.

## Что получилось

Обучение завершено на полном OpenWebText. Измерения на валидации
(5.6M токенов, одинаковый протокол для обеих моделей, `scripts/eval_model.py`):

| Модель | val_loss | Перплексия | Обучающие данные |
|---|---|---|---|
| **Наша** (124M, из чекпоинта) | **3.15** | **~23** | полный OWT |
| Эталон HF gpt2 (124M) | 3.13 | ~23 | WebText (не опубликован) |

Разрыв с оригиналом — в пределах ~2% перплексии; архитектура при этом
побитово совпадает с эталоном. Модель можно **запустить в браузере**
(раздел ниже).

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

Прогресс в терминале: два tqdm-бара как в предыдущей попытке — `train`
(обновление на каждом микро-шаге, postfix: loss / lr / tok/s) и `valid`
(по батчам оценки). Строки логов печатаются через `tqdm.write` и не ломают
отрисовку; всё то же дублируется в `train.log`. При перенаправлении вывода
(nohup, пайпы) бары отключаются автоматически; принудительно — `--no-bar`.

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

Обучение — по статье с несколькими отклонениями, полный список ниже.

## Отклонения от оригинала (полный список)

**Что отклонений НЕ содержит** (и чем это подтверждено):

- Архитектура: размеры, pre-LN + ln_f, GELU, tying, dropout, eps LN, init —
  ровно как в статье и HF-конфиге. Подтверждение: с эталонными весами OpenAI
  модель выдаёт **побитово те же логиты**, что HF gpt2 (`verify_parity.py`),
  и жадная генерация совпадает до символа.
- Токенизатор — официальные файлы OpenAI без изменений.
- Эффективный батч — 512×1024 = 524 288 токенов/шаг, как в статье.
- Chunked cross-entropy и gradient accumulation — не отклонение, а
  организация памяти: математически тот же loss и те же градиенты
  (у OpenAI — 512 последовательностей на 256 TPU-ядрах, у нас —
  микро-батчи на одной карте; см. историю в git и docstring `model.py`).

**Реальные отклонения и почему:**

| Что | В оригинале (статья/код OpenAI) | Здесь | Почему |
|---|---|---|---|
| Корпус | WebText, ~40GB (~9B токенов) — не опубликован | OpenWebText (открытая реконструкция той же идеи: ссылки с Reddit 3+ karma) | WebText недоступен; OWT — стандартная замена |
| Бюджет токенов | весь WebText, 124M остался underfit | первый прогон 2.5B (Chinchilla для 124M); полный ~8–9B опционально (п. 5 секции «Запуск на 3080») | 2.5B ≈ 1.5–2 суток на 3080; полные ~9B ≈ 5–6 дней |
| Оптимизатор | Adam (β2=0.999 по умолчанию TF) | AdamW: β2=0.95, weight decay 0.1 на матрицах | практика nanoGPT/GPT-3; стабильнее на малых батчах, лучше генерализация |
| LR-расписание | lr 2.5e-4 «вручную подобран»; детали в статье не раскрыты | warmup 2000 шагов + cosine до 6% от пика | стандартная реконструкция (nanoGPT); сам lr 2.5e-4 — из статьи |
| Точность | fp32 (TF, TPU) | bf16 autocast + TF32, CE-сумма в fp32 | fp32 не влезает в 10GB VRAM; стандарт для обучения LLM |

**Что НЕ отклоняется, но выглядит непривычно:** микро-батч 4×1024 с
накоплением 128 (в статье батч целиком, но у нас одна карта); случайные окна
данных вместо последовательной подачи документов (статистически эквивалентно);
dropout 0.1 оставлен как в статье, хотя nanoGPT для этого же датасета
использовал 0.0.

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

# 1. Smoke-тест на GPU: 15 шагов 124M дефолтной командой (~6 минут),
#    заодно замер tok/s и пик VRAM (в конце лога)
./venv/bin/python train.py --out-dir checkpoints/smoke --max-iters 15 \
  --warmup-steps 5 --eval-interval 15 --eval-iters 5 --no-resume --log-interval 5

# 2. Основной прогон: 2.5B токенов (Chinchilla-оптимум для 124M), ~1.5-2 суток
./venv/bin/python train.py --max-tokens 2.5e9 --compile
#    значения по умолчанию (batch 4 × accum 128 × ctx 1024) уже подобраны
#    под 10GB VRAM: пик ~5.8 GiB, эффективный батч = статье (512×1024).
#    batch 8 проверен — НЕ влезает (OOM в backward), не поднимайте.
#    --compile: первый шаг занимает пару минут, дальше шаги быстрее.
#    Следить за прогрессом (в другом терминале):
tail -f checkpoints/run1/train.log

# 3. Прервали (Ctrl+C), упали, выключили машину — просто запустить ту же команду ещё раз:
#    обучение продолжится с checkpoints/run1/last.pt (шаг, оптимизатор, ГПСЧ).
#    Прогресс между сохранениями теряется не более eval-interval шагов (250).

# 4. Сгенерировать из лучшего чекпоинта / оценить качество
./venv/bin/python generate.py "The meaning of life is" --temperature 0.8 --top-k 50
./venv/bin/python generate.py --ckpt checkpoints/run1/last.pt --tokens 400
./venv/bin/python scripts/eval_model.py --ckpt checkpoints/run1/best.pt   # val_loss + перплексия + сэмплы
./venv/bin/python scripts/eval_model.py --no-samples --tokens 5000000     # только цифры по всей валидации

# 5. (Опционально) Полный масштаб: докачать весь OpenWebText (~8M документов,
#    часы скачивания) и перетокенизировать (~4-5 ч), затем прогон ~8-9B токенов (~2 суток)
./venv/bin/python scripts/download_data.py --full
rm data/tokens/train.bin data/tokens/val.bin data/tokens/meta.json
./venv/bin/python scripts/prepare_tokens.py
./venv/bin/python train.py --max-tokens 8e9 --compile --out-dir checkpoints/run_full
```

Замечания:
- smoke из п. 1 пишет в отдельный `checkpoints/smoke` и не влияет на основной
  прогон; основной прогресс живёт в `checkpoints/run1` (`train.log`, `last.pt`,
  `best.pt`).
- бюджет прогонов (замерено на 3080: с `--compile` ~48K tok/s, без ~20K;
  пик ~6 GiB VRAM):

| Прогон | Токенов | Время (по замеру) | Что получится |
|---|---|---|---|
| smoke (п. 1) | — | ~6 мин | проверка пайплайна, замер tok/s |
| Chinchilla (п. 2) | 2.5B | ~14 ч (с `--compile`) | связный веб-текст |
| масштаб WebText (п. 5) | ~8–9B | ~2 суток (с `--compile`) | близко к качеству оригинала (оригинал underfit WebText) |

- подвыборка (560M токенов) уже скачана и токенизирована; Chinchilla-прогон —
  это ~4.5 прохода по ней (без дедупликации — терпимо, но корпус `--full`
  ~9B токенов лучше: единый проход без повторов).


## Запуск в браузере

Модель экспортируется в ONNX и работает целиком на стороне клиента
(onnxruntime-web: WebGPU с фолбэком на WASM), токенизатор — собственный
JS-порт (`web/bpe.js`), совпадающий с каноничными ID GPT-2.

```bash
# 1. Однократно: экспорт best.pt в ONNX + проверка графа (~3-5 минут)
./venv/bin/pip install onnx onnxruntime onnxscript
./venv/bin/python scripts/export_onnx.py
#    -> web/gpt2_124m.onnx (fp16, ~310 МБ) + копии encoder.json/vocab.bpe
#    проверка: префилл и декодирование сверяются с torch (fp32-граф: max|Δ| ~ 1e-5)

# 2. Раздать статически и открыть в браузере (Chrome/Edge — WebGPU):
./venv/bin/python -m http.server 8000
#    открыть http://localhost:8000/web/
```

Генерация инкрементальная (KV-кэш внутри ONNX-графа, состояние передаётся
через вход `state`), скорость — десятки токенов/с на WebGPU. Для доступности
из интернета достаточно любого статического хостинга: `web/` не имеет
сборочных зависимостей, модель и файлы токенизатора кладутся рядом.

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
  eval_model.py          # val_loss/перплексия чекпоинта + сэмплы
  export_onnx.py         # чекпоинт -> web/gpt2_124m.onnx (браузерный инференс)
web/                     # демо в браузере: index.html, bpe.js (BPE на JS), main.js
train.py           # обучение (resume, AMP, cosine LR, grad clip, чекпоинты)
generate.py        # продолжение текста (temperature/top-k/top-p/greedy, stdin)
docs/
  analysis-previous-attempt.md  # почему прошлая попытка была хуже
  bpe-comparison.md             # сравнение токенизаторов с измерениями
prev-llm-attempt -> /home/valeriy/custom-llm  # локальная ссылка на прошлую попытку
```

## Примечания

- **О переобучении на подвыборке.** На первых 560M токенов val_loss начинает
  дрейфовать вверх после ~7 проходов (наблюдалось: best 3.3333 на шаге 7750,
  затем 3.39–3.44). Лечение — полный корпус (9B токенов ≈ 1 проход), а не
  увеличение dropout: в статье у всех моделей dropout 0.1, и 124M на полном
  WebText **недообучался**, а не переобучался. Лучшая модель подвыборки
  сохраняется в `best.pt`, её качество: `scripts/eval_model.py`.
- `transformers` нужен только `verify_parity.py`; обучение и генерация его не
  используют.
- WebText не опубликован; стандартная открытая замена — OpenWebText
  (та же идея: исходящие ссылки Reddit 3+ karma). Полный корпус ~8M документов
  (~38GB текста), качается стримингом shardами.
- Оригинальные файлы токенизатора берутся из открытого бакета OpenAI
  (зеркало — HF), поэтому последовательность токенов совпадает с эталоном.

## Благодарности

В разработке этого репозитория помогала [GLM](https://z.ai) — модель
GLM-5.3-Flash в режиме max (Z.ai).
