"""Проверка корректности воспроизведения: сверка с эталонным openai-community/gpt2.

Скачивает model.safetensors (~548 МБ) в data/hf_reference/ и проверяет:
  1. Токенизация: наш GPT2Tokenizer vs transformers.GPT2Tokenizer.
  2. Логиты: наша модель с эталонными весами vs transformers.GPT2LMHeadModel.
Запускать не обязательно для обучения — это контроль качества реализации.
"""

import sys
from pathlib import Path

import requests
import torch
from tqdm import tqdm

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from gpt2rep.model import GPT2LMHeadModel
from gpt2rep.tokenizer import GPT2Tokenizer

HF_URL = "https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors"
REF_DIR = REPO / "data" / "hf_reference"

TEST_TEXTS = [
    "Hello, my name is",
    "The quick brown fox jumps over the lazy dog. " * 5,
    "В 1750 году французский физик предложил измерить...",
    "def fibonacci(n):\n    if n < 2:\n        return n\n",
    "Mixed 123 numbers and — dashes… émojis 🚀🤖 and tabs\t\tnewlines\n\n",
    "<|endoftext|>The meaning of life is",
]


def hf_encode(hf_tok, text: str) -> list[int]:
    # HF по умолчанию распознаёт <|endoftext|> как спец-токен; оригинальный
    # OpenAI-энкодер кодирует его как обычный текст. split_special_tokens
    # включает «сырое» поведение оригинала.
    try:
        return hf_tok.encode(text, split_special_tokens=True)
    except TypeError:
        return hf_tok.encode(text)


def download_weights() -> Path:
    out = REF_DIR / "model.safetensors"
    if out.exists() and out.stat().st_size > 0:
        return out
    REF_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Скачиваю {HF_URL}")
    with requests.get(HF_URL, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        tmp = out.with_suffix(".tmp")
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc="model.safetensors") as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.replace(out)
    return out


def main() -> None:
    from transformers import GPT2LMHeadModel as HFModel
    from transformers import GPT2Tokenizer as HFTokenizer

    weights = download_weights()

    print("== 1. Токенизатор ==")
    tok = GPT2Tokenizer.load(REPO / "data" / "tokenizer")
    hf_tok = HFTokenizer.from_pretrained("openai-community/gpt2")
    for text in TEST_TEXTS:
        a, b = tok.encode(text), hf_encode(hf_tok, text)
        status = "OK " if a == b else "FAIL"
        print(f"  [{status}] {text[:40]!r}: {len(a)} токенов")
        assert a == b, f"расхождение токенизации:\n{a}\n{b}"
    print("  токенизаторы идентичны")

    print("== 2. Логиты ==")
    ours = GPT2LMHeadModel.from_hf_safetensors(str(weights))
    ours.eval()
    hf = HFModel.from_pretrained("openai-community/gpt2")
    hf.eval()

    ids = torch.tensor([tok.encode(TEST_TEXTS[1])])
    with torch.no_grad():
        ours_logits, _ = ours(ids)
        hf_logits = hf(ids).logits
    diff = (ours_logits - hf_logits).abs().max().item()
    hf_scale = hf_logits.abs().max().item()
    print(f"  max|Δlogits| = {diff:.3e} (масштаб логитов ~{hf_scale:.1f})")
    assert diff < 1e-3, "логиты расходятся с эталоном"

    # генерация должна совпадать жадным поиском
    ctx = torch.tensor([tok.encode("The meaning of life is")])
    with torch.no_grad():
        g1 = ours.generate(ctx.clone(), 20, temperature=0.0)
        g2 = hf.generate(ctx.clone(), max_new_tokens=20, do_sample=False)
    s1 = tok.decode(g1[0].tolist())
    s2 = hf_tok.decode(g2[0])
    print(f"  greedy: {s1!r}")
    print(f"  HF:     {s2!r}")
    assert s1 == s2, "жадная генерация разошлась с HF"

    print("ПАРИТЕТ С ЭТАЛОНОМ ПОДТВЕРЖДЁН")


if __name__ == "__main__":
    main()
