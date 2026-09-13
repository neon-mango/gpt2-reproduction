# gpt2-reproduction

Воспроизведение [openai-community/gpt2](https://huggingface.co/openai-community/gpt2)
(124M) по статье [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf):
своя реализация архитектуры, официального byte-level BPE и пайплайна обучения
на OpenWebText. Всё с нуля, без `transformers` в рантайме обучения.

> ## 🚀 [Попробовать в браузере](https://neon-mango.github.io/gpt2-reproduction/)
>
> Модель (326 МБ, ONNX fp16) целиком загружается на сторону клиента и
> генерирует текст локально — WebGPU, при отсутствии — WASM. Ничего
> устанавливать не нужно.

## Результат

| Модель | val_loss | Перплексия | Данные обучения |
|---|---|---|---|
| Наша, финальный чекпоинт (20 000 шагов ≈ 10.5B токенов) | **3.141** | **23.1** | полный OWT |
| Наша, `best.pt` (шаг 17 250) | 3.153 | 23.4 | полный OWT |
| Эталон HF gpt2 (124M) | 3.133 | 23.0 | WebText (не опубликован) |

Единый протокол измерений — 5.6M токенов OWT-валидации
(`scripts/eval_model.py`). Совпадение с оригиналом в пределах ~1%
перплексии. Архитектура совпадает **строго**: с эталонными весами OpenAI
наша модель выдаёт логиты, побитово идентичные HF gpt2, жадная генерация
совпадает до символа (`scripts/verify_parity.py`). Подробности и честные
оговорки — в [docs/overview.md](docs/overview.md).

## Документация

| Хочу... | Документ |
|---|---|
| просто познакомиться с проектом | [docs/overview.md](docs/overview.md) — что сделано, как проверялось, отклонения от оригинала |
| запустить и обучить модель сам | [docs/run-locally.md](docs/run-locally.md) — окружение, данные, обучение на одной карте, браузер, публикация |

## Структура репозитория

```
gpt2rep/                 # ядро: tokenizer.py (byte-level BPE), model.py (GPT-2)
scripts/                 # скачивание данных/токенизатора, токенизация,
                         # верификация, ONNX-экспорт, нарезка модели
web/                     # фронтенд: React + Material UI (Vite), инференс ORT-web
train.py / generate.py   # обучение и генерация (CLI)
docs/                    # документация и разборы
```

## Благодарности

В разработке этого репозитория помогала [GLM](https://z.ai) — модель
GLM-5.3-Flash в режиме max (Z.ai).
