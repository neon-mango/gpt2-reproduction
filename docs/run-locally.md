# Запуск самостоятельно

Полный цикл: окружение → данные → обучение → генерация → браузерное демо.
Все этапы возобновляемые: прерывание в любом месте безопасно, продолжение —
повторным запуском той же команды.

Требования: Python ≥3.12, NVIDIA GPU ≥10GB VRAM (проверено на RTX 3080),
~40 ГБ диска на полный корпус, Node.js ≥20 для веб-фронтенда.

## 1. Окружение

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
# для старых драйверов: pip install torch --index-url https://download.pytorch.org/whl/cu126
```

## 2. Токенизатор

```bash
./venv/bin/python scripts/download_tokenizer.py
# -> data/tokenizer/encoder.json + vocab.bpe (официальные файлы OpenAI)
```

## 3. Данные (OpenWebText)

```bash
./venv/bin/python scripts/download_data.py          # подвыборка 500k документов (~920MB, ~10 мин)
./venv/bin/python scripts/download_data.py --full   # весь корпус ~8M документов (часы)
```

## 4. Токенизация

```bash
./venv/bin/python scripts/prepare_tokens.py          # ~20 мин на 560M токенов (7 воркеров)
# -> data/tokens/train.bin, val.bin (uint16, memmap) + meta.json
```

Валидация — последние 5 000 документов корпуса. Прогресс и возобновление —
в консоли; закодированное не пересчитывается.

## 5. Обучение

Значения по умолчанию соответствуют статье и подобраны под 10GB VRAM
(микро-батч 4 × накопление 128 × контекст 1024 = эффективный батч статьи
512×1024; пик ~6 GiB).

```bash
# smoke-тест (~6 минут): проверка пайплайна + замер tok/s
./venv/bin/python train.py --out-dir checkpoints/smoke --max-iters 15 \
  --warmup-steps 5 --eval-interval 15 --no-resume --log-interval 5

# основной прогон (с --compile ~49K tok/s, шаг ~10.7 с):
./venv/bin/python train.py --compile --max-tokens 2.5e9      # Chinchilla, ~14 ч
./venv/bin/python train.py --compile --max-tokens 8e9        # масштаб WebText, ~2 суток
```

- прерывание (Ctrl+C) и восстановление: просто запустить команду ещё раз —
  продолжение с `checkpoints/run1/last.pt` (веса + оптимизатор + шаг + ГПСЧ);
  теряется не более 250 шагов;
- прогресс: tqdm-бары (train/valid) и `tail -f checkpoints/run1/train.log`;
- batch 8 × accum 64 на 10GB не влезает (OOM) — не поднимайте микро-батч.

Бюджеты (замерено на 3080):

| Прогон | Токенов | Время |
|---|---|---|
| smoke | — | ~6 мин |
| Chinchilla | 2.5B | ~14 ч |
| масштаб WebText | ~8–9B | ~2 суток |

## 6. Оценка и генерация

```bash
./venv/bin/python scripts/eval_model.py                  # val_loss + перплексия + сэмплы
./venv/bin/python scripts/eval_model.py --no-samples --tokens 5000000
./venv/bin/python generate.py "Once upon a time" --tokens 300
echo "Продолжи этот текст" | ./venv/bin/python generate.py -
```

Для сверки с эталоном (необязательно): `scripts/verify_parity.py` скачает
оригинальные веса gpt2 и докажет побитовое совпадение логитов.

## 7. Браузерное демо

```bash
./venv/bin/pip install onnx onnxruntime onnxscript
./venv/bin/python scripts/export_onnx.py      # best.pt -> web/public/gpt2_124m.onnx
./venv/bin/python scripts/split_model.py      # части <100MB + model_chunks в config.json

cd web
npm install
npm run dev      # http://localhost:5173
npm run build    # прод-сборка в dist/
```

Экспорт сам проверяет граф: fp32-копия сверяется с torch (max|Δ| ~ 1e-5),
fp16 — по argmax/top-5.

## 8. Публикация на GitHub Pages

Модель нарезается на части <100 МБ, коммитится через **Git LFS**
(`web/public/model/`), страница собирается GitHub Actions
(`.github/workflows/deploy-pages.yml`, checkout с `lfs: true`) и отдаёт
части с того же домена — CORS не нужен.

```bash
git lfs install   # однократно
git add web/public/model web/public/config.json web/public/encoder.json web/public/vocab.bpe
git commit -m "model weights" && git push
```

Settings → Pages → Source: **GitHub Actions**. При пуше в master сайт
соберётся и опубликуется автоматически. В `web/public/config.json`:
`model_chunks` — относительные пути частей (можно заменить на абсолютные
URL любого статического хостинга с CORS), `model_sha256` — контроль целостности.

## Структура репозитория

```
gpt2rep/                 # ядро: tokenizer.py, model.py (архитектура + ONNX-обёртка)
scripts/                 # download_tokenizer, download_data, prepare_tokens,
                         # verify_parity, eval_model, export_onnx, split_model
web/                     # React + drawably фронтенд, JS-токенизатор, деплой-воркфлоу
train.py                 # обучение
generate.py              # генерация (CLI)
```

## Заметки

- Уменьшенная модель для проверки на CPU:
  `train.py --n-layer 2 --n-embd 128 --n-head 2 --batch-size 4 --grad-accum 2 --max-iters 30`.
- `transformers` нужен только `verify_parity.py`.
- На маленьком корпусе (560M токенов) val_loss растёт после ~7 проходов —
  берите полный корпус: 124M на ~10B токенов недообучается, как в статье.

---

Назад к [README](../README.md) · [Обзор проекта](overview.md)
