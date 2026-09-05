"""GPT-2 byte-level BPE токенизатор, реализованный с нуля.

Использует официальные файлы OpenAI (encoder.json + vocab.bpe, 50257 токенов),
алгоритм и предтокенизация повторены по https://github.com/openai/gpt-2.
"""

import json
from functools import lru_cache
from pathlib import Path

import regex as re


@lru_cache()
def bytes_to_unicode() -> dict[int, str]:
    """Биективное отображение байтов 0..255 в печатные unicode-символы.

    Печатные ASCII и ряд латинских символов отображаются сами в себя,
    остальные байты — в кодовые точки 256+n. Так любой байт представления
    текста в UTF-8 становится видимым символом и участвует в BPE-мерджах.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


def get_pairs(word: tuple[str, ...]) -> set[tuple[str, str]]:
    """Множество соседних пар символов слова."""
    return {(word[i], word[i + 1]) for i in range(len(word) - 1)}


class GPT2Tokenizer:
    def __init__(self, encoder: dict[str, int], bpe_merges: list[tuple[str, str]]):
        self.encoder = encoder
        self.decoder = {v: k for k, v in encoder.items()}
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache: dict[str, str] = {}
        # Предтокенизация GPT-2: не даёт BPE мерджить через границы категорий
        # (буквы/цифры/пунктуация/пробелы), пробел приклеивается к следующему слову.
        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )
        self.errors = "replace"
        self.eot_token = encoder["<|endoftext|>"]
        self.vocab_size = len(encoder)

    @classmethod
    def load(cls, tokenizer_dir: str | Path) -> "GPT2Tokenizer":
        tokenizer_dir = Path(tokenizer_dir)
        with open(tokenizer_dir / "encoder.json", encoding="utf-8") as f:
            encoder = json.load(f)
        merges: list[tuple[str, str]] = []
        with open(tokenizer_dir / "vocab.bpe", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                first, second = line.split()
                merges.append((first, second))
        return cls(encoder, merges)

    def save(self, tokenizer_dir: str | Path) -> None:
        tokenizer_dir = Path(tokenizer_dir)
        tokenizer_dir.mkdir(parents=True, exist_ok=True)
        with open(tokenizer_dir / "encoder.json", "w", encoding="utf-8") as f:
            json.dump(self.encoder, f, ensure_ascii=False)
        with open(tokenizer_dir / "vocab.bpe", "w", encoding="utf-8") as f:
            f.write("#version: 0.2\n")
            for a, b in self.bpe_ranks:
                f.write(f"{a} {b}\n")

    def bpe(self, token: str) -> str:
        """Слитие пар символов одного предтокена в порядке приоритета мерджей."""
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        if len(word) < 2:
            return token
        pairs = get_pairs(word)
        while True:
            bigram = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word: list[str] = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                new_word.extend(word[i:j])
                i = j
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = get_pairs(word)
        result = " ".join(word)
        self.cache[token] = result
        return result

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for token in self.pat.findall(text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            ids.extend(self.encoder[t] for t in self.bpe(token).split(" "))
        return ids

    def decode(self, ids: list[int]) -> str:
        text = "".join(self.decoder[int(i)] for i in ids)
        data = bytearray(self.byte_decoder[c] for c in text)
        return data.decode("utf-8", errors=self.errors)
