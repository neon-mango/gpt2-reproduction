"""Архитектура GPT-2, воспроизводящая huggingface.co/openai-community/gpt2 (124M).

Отличия от предыдущей попытки (см. docs/analysis-previous-attempt.md):
  - Pre-LN: LayerNorm ВНУТРИ ветки перед attention/FFN + финальный ln_f
    (в прошлой версии был post-LN как в GPT-1);
  - GELU (tanh-аппроксимация) вместо ReLU;
  - weight tying: lm_head = wte (общие веса, как в оригинале);
  - инициализация N(0, 0.02), веса c_proj масштабируются 1/sqrt(2*n_layer)
    ("modified initialization" из статьи, п. 2.3);
  - dropout: attn/resid/embd по 0.1, attn-dropout реализован внутри SDPA.

Гиперпараметры 124M (Table 2 статьи): 12 слоёв, d_model=768, 12 голов,
контекст 1024, словарь 50257. Число параметров — ровно 124 439 808.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def new_gelu(x: torch.Tensor) -> torch.Tensor:
    """GELU из GPT-2 (tanh-аппроксимация, activation_function=gelu_new в HF)."""
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


@dataclass
class GPT2Config:
    vocab_size: int = 50257
    n_positions: int = 1024
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    n_inner: int | None = None  # None -> 4*n_embd
    attn_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    embd_pdrop: float = 0.1
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02

    def __post_init__(self):
        if self.n_inner is None:
            self.n_inner = 4 * self.n_embd


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.scale = self.head_dim**-0.5  # scale_attn_weights=True в HF gpt2
        # c_attn в HF — Conv1D, математически эквивалентен Linear(768, 2304)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = config.attn_pdrop
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        # Flash Attention: causal-маска и attn-dropout внутри fused-ядра
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=True,
            scale=self.scale,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))

    def forward_cached(self, x: torch.Tensor, pk: torch.Tensor | None, pv: torch.Tensor | None,
                       past_len=None):
        """Инкрементальный путь для генерации/ONNX-экспорта (без SDPA —
        обычный matmul+softmax, безопасно экспортируется; dropout выключен,
        т.к. путь только для eval).

        pk/pv: кэш ключей/значений (B, n_head, P, head_dim) или None (prefill).
        past_len: длина кэша (int или 0-dim int64 тензор — для динамического
        ONNX-экспорта); если None, берётся из формы pk.
        Возвращает (выход attention, новый кэш)."""
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        if pk is not None:
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        if past_len is None:
            past_len = k.size(2) - T
        att = (q @ k.transpose(-2, -1)) * self.scale
        # токен на позиции past_len+i видит ключи 0..past_len+i.
        # seq_len берётся тензором (Shape->Gather), иначе в трассировке ONNX
        # длина запечётся константой из dummy-примера
        seq_len = torch._shape_as_tensor(x)[1]
        q_pos = torch.arange(past_len, past_len + seq_len, device=x.device)
        kv_idx = torch.arange(past_len + seq_len, device=x.device)
        mask = kv_idx[None, :] <= q_pos[:, None]
        att = att.masked_fill(~mask, float("-inf")).softmax(dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y), (k, v)


class MLP(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_inner)
        self.c_proj = nn.Linear(config.n_inner, config.n_embd)
        self.drop = nn.Dropout(config.resid_pdrop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.c_proj(new_gelu(self.c_fc(x))))


class Block(nn.Module):
    """Pre-LN блок GPT-2: x += attn(ln_1(x)); x += mlp(ln_2(x))."""

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

    def forward_cached(self, x: torch.Tensor, pk: torch.Tensor | None, pv: torch.Tensor | None,
                       past_len=None):
        y, present = self.attn.forward_cached(self.ln_1(x), pk, pv, past_len)
        x = x + y
        x = x + self.mlp(self.ln_2(x))
        return x, present


class GPT2Model(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.apply(self._init_weights)
        # "Modified initialization which accounts for the accumulation on the
        # residual path with model depth" (статья, п. 2.3): масштаб 1/sqrt(N),
        # N = 2*n_layer — число residual-путей (attention + mlp в каждом блоке).
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=config.initializer_range / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        assert T <= self.config.n_positions, f"T={T} > n_positions={self.config.n_positions}"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos))
        for block in self.h:
            x = block(x)
        return self.ln_f(x)

    def forward_cached(self, idx: torch.Tensor, past: torch.Tensor | None = None,
                       past_len=None):
        """Инкрементальный путь (генерация/ONNX): возвращает (h, new_past).

        idx — НОВЫЕ токены (B, T); past — состояние (n_layer, 2, B, n_head,
        P, head_dim) или None. Поддерживается prefill (P=0) и декодирование
        (T=1). Позиции берутся со смещением past_len (int или 0-dim int64
        тензор; по умолчанию — из формы past)."""
        B, T = idx.shape
        if past_len is None:
            past_len = 0 if past is None else past.shape[4]
        # seq_len тензором — динамический Range в ONNX (см. forward_cached)
        seq_len = torch._shape_as_tensor(idx)[1]
        pos = torch.arange(past_len, past_len + seq_len, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos))
        presents = []
        for i, block in enumerate(self.h):
            pk = pv = None
            if past is not None:
                pk, pv = past[i, 0], past[i, 1]
            x, present = block.forward_cached(x, pk, pv, past_len)
            presents.append(torch.stack(present))
        return self.ln_f(x), torch.stack(presents)


class GPT2LMHeadModel(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying (как в оригинале): lm_head.weight is wte.weight
        self.tie_weights()

    def tie_weights(self) -> None:
        self.lm_head.weight = self.transformer.wte.weight

    # Кусок логитов (chunk, 50257) во float32 весит chunk×50257×4 байт:
    # при 2048 токенов это ~0.4GB вместо (B×T)×50257×4 ≈ 1.6GB+ на батч 8×1024.
    # Именно полный тензор логитов+softmax буферы съедали VRAM (см. OOM),
    # поэтому loss считается по кускам последовательности.
    LOSS_CHUNK_TOKENS = 1024

    @staticmethod
    def _chunk_loss(h: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """logits+cross_entropy для одного куска токенов; вызывается под autocast."""
        logits = F.linear(h, w)
        return F.cross_entropy(logits.float(), y)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """idx: (B, T) int64.

        Без targets возвращает (logits, None) — полный тензор логитов.
        С targets возвращает (None, loss): logits+CE считаются кусками по
        LOSS_CHUNK_TOKENS токенов; в режиме обучения каждый кусок оборачивается
        в checkpoint, поэтому в backward логиты пересчитываются и в памяти
        между шагами их полная версия не хранится вовсе.
        """
        hidden = self.transformer(idx)
        if targets is None:
            return self.lm_head(hidden), None
        B, T, C = hidden.shape
        h = hidden.reshape(B * T, C)
        y = targets.reshape(B * T)
        chunk = self.LOSS_CHUNK_TOKENS
        loss_sum = torch.zeros((), device=hidden.device, dtype=torch.float32)
        for i in range(0, B * T, chunk):
            hc, yc = h[i:i + chunk], y[i:i + chunk]
            if self.training:
                # use_reentrant=False корректно работает под autocast
                part = torch.utils.checkpoint.checkpoint(
                    self._chunk_loss, hc, yc, self.lm_head.weight, use_reentrant=False
                )
            else:
                part = self._chunk_loss(hc, yc, self.lm_head.weight)
            loss_sum = loss_sum + part * yc.numel()
        return None, loss_sum / (B * T)

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:  # wte не вычитаем из-за tying; вычитаем только wpe
            n -= self.transformer.wpe.weight.numel()
        return n

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        eot_token: int | None = None,
    ) -> torch.Tensor:
        """Сэмплирование. top_k/top_p — усечение хвоста распределения
        (в оригинальной статье — top-k random sampling)."""
        self.eval()
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.config.n_positions:])
            scores = logits[:, -1, :]
            if temperature <= 0:  # greedy
                idx = torch.cat((idx, scores.argmax(dim=-1, keepdim=True)), dim=1)
                continue
            scores = scores / temperature
            if top_k is not None:
                k = min(top_k, scores.size(-1))
                thresh = torch.topk(scores, k, dim=-1).values[..., -1, None]
                scores = scores.masked_fill(scores < thresh, float("-inf"))
            if top_p is not None:
                sorted_scores, sorted_idx = torch.sort(scores, descending=True, dim=-1)
                probs = sorted_scores.softmax(dim=-1)
                cum = probs.cumsum(dim=-1)
                # всегда оставляем минимум один токен, даже если top_p мал
                mask = cum - probs >= top_p
                scores = scores.masked_fill(mask.scatter(-1, sorted_idx, mask), float("-inf"))
            next_id = torch.multinomial(scores.softmax(dim=-1), 1)
            idx = torch.cat((idx, next_id), dim=1)
            if eot_token is not None and (next_id == eot_token).all():
                break
        return idx

    # ------------------------------------------------------------------
    # Загрузка эталонных весов HF (model.safetensors из openai-community/gpt2)
    # ------------------------------------------------------------------
    @classmethod
    def from_hf_safetensors(cls, path: str, device: str = "cpu") -> "GPT2LMHeadModel":
        from safetensors.torch import load_file

        sd = load_file(path, device="cpu")
        model = cls(GPT2Config())
        remapped = {}
        for k, v in sd.items():
            # легаси-буфер causal-маски из TF-чекпоинта ("h.N.attn.bias"), не параметры
            if k.endswith("attn.bias") and "c_attn" not in k:
                continue
            if k in ("wte.weight", "wpe.weight", "ln_f.weight", "ln_f.bias"):
                new_k = "transformer." + k
                new_v = v
            elif k == "lm_head.weight":
                new_k, new_v = k, v  # будет перетянут через tying
            else:
                # "h.0.attn.c_attn.weight" -> "transformer.h.0.attn.c_attn.weight";
                # Conv1D в HF хранит вес как (in, out), у nn.Linear — (out, in).
                new_k = "transformer." + k
                new_v = v.T.contiguous() if k.endswith("weight") and ".c_" in k else v
            remapped[new_k] = new_v
        missing, unexpected = model.load_state_dict(remapped, strict=False)
        # lm_head tied к wte — в чекпоинте HF его нет, это нормально
        assert not [m for m in missing if m != "lm_head.weight"], f"missing: {missing}"
        assert not unexpected, f"unexpected: {unexpected}"
        model.tie_weights()
        return model.to(device)


def gpt2_124m() -> GPT2LMHeadModel:  # то, что на HF зовётся openai-community/gpt2
    return GPT2LMHeadModel(GPT2Config())


def gpt2_355m() -> GPT2LMHeadModel:  # gpt2-medium
    return GPT2LMHeadModel(GPT2Config(n_embd=1024, n_layer=24, n_head=16))


def gpt2_774m() -> GPT2LMHeadModel:  # gpt2-large
    return GPT2LMHeadModel(GPT2Config(n_embd=1280, n_layer=36, n_head=20))


def gpt2_1542m() -> GPT2LMHeadModel:  # gpt2-xl
    return GPT2LMHeadModel(GPT2Config(n_embd=1600, n_layer=48, n_head=25))


class OnnxWrapper(nn.Module):
    """Обёртка для ONNX-экспорта (браузерный инференс через onnxruntime-web).

    Входы:  ids      — (batch, seq) int64, новые токены;
            state    — (n_layer, 2, batch, n_head, past, head_dim); для
                       prefill past=0 (тензоры нулевой длины);
            past_len — 0-dim int64, длина кэша (0 для prefill). Отдельный
                       вход, чтобы длина была динамической в графе.
    Выходы: logits — логиты ПОСЛЕДНЕГО токена (batch, vocab) во float32;
            new_state — состояние с дописанными k/v.
    """

    def __init__(self, model: GPT2LMHeadModel):
        super().__init__()
        self.model = model

    def forward(self, idx: torch.Tensor, state: torch.Tensor, past_len: torch.Tensor):
        h, new_state = self.model.transformer.forward_cached(idx, state, past_len)
        logits = self.model.lm_head(h[:, -1:, :])  # (B, 1, V)
        return logits.squeeze(1).float(), new_state
