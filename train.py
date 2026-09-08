"""Обучение GPT-2 (124M по умолчанию) на токенах из data/tokens/*.bin.

Гиперпараметры по умолчанию повторяют GPT-2 из статьи (Radford et al., 2019):
батч 512×1024 токенов на шаг (через gradient accumulation), lr 2.5e-4,
warmup 2000 шагов, cosine decay, dropout 0.1, grad clip 1.0.

Практические отклонения (не влияют на архитектуру):
  - AdamW (β2=0.95, weight decay 0.1 на матрицах) вместо Adam — стандарт
    де-факто для тренировки GPT-2-класса (nanoGPT/GPT-3), стабилен без тюнинга;
  - bf16/fp16 autocast — иначе 10GB VRAM не хватит;
  - окна данных выбираются случайными смещениями, а не скользящим окном
    с шагом 1 (в прошлой попытке соседние примеры пересекались на 99%,
    тратя почти весь шаг на повтор).

Возобновление: чекпоинт last.pt (модель+оптимизатор+шаг+состояния ГПСЧ)
пишется атомарно; при запуске обучение продолжается с него автоматически.
"""

import argparse
import dataclasses
import glob
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

# до первого импорта torch: меньше фрагментации VRAM при кусочном CE
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from tqdm import tqdm

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from gpt2rep.model import GPT2Config, GPT2LMHeadModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # модель
    p.add_argument("--n-layer", type=int, default=12)
    p.add_argument("--n-embd", type=int, default=768)
    p.add_argument("--n-head", type=int, default=12)
    p.add_argument("--n-positions", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)
    # данные
    p.add_argument("--data-dir", type=Path, default=REPO / "data" / "tokens")
    p.add_argument("--train-file", type=Path, default=None)
    p.add_argument("--val-file", type=Path, default=None)
    # оптимизация
    p.add_argument("--batch-size", type=int, default=4,
                   help="последовательностей на микро-шаг (4 — для 10GB VRAM)")
    p.add_argument("--grad-accum", type=int, default=128,
                   help="микро-шагов на шаг оптимизатора; "
                        "batch*accum*seq_len = токенов на шаг (в статье 512*1024)")
    p.add_argument("--max-iters", type=int, default=20_000)
    p.add_argument("--max-tokens", type=float, default=None,
                   help="если задано — перекрывает max-iters (max-tokens / токенов на шаг)")
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--min-lr-ratio", type=float, default=0.06)
    p.add_argument("--warmup-steps", type=int, default=2_000)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--grad-clip", type=float, default=1.0)
    # служебное
    p.add_argument("--out-dir", type=Path, default=REPO / "checkpoints" / "run1")
    p.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    p.add_argument("--compile", action="store_true", help="torch.compile (дольше старт, быстрее шаги)")
    p.add_argument("--eval-interval", type=int, default=250)
    p.add_argument("--eval-iters", type=int, default=20, help="батчей на одну оценку val")
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--no-resume", action="store_true", help="не продолжать с last.pt")
    p.add_argument("--no-bar", action="store_true",
                   help="без прогресс-баров tqdm (иначе они и так отключаются "
                        "автоматически, если вывод не в терминал)")
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def get_lr_frac(step: int, warmup: int, max_iters: int, min_ratio: float) -> float:
    """Линейный warmup, затем cosine decay до min_ratio (nanoGPT-конвенция)."""
    if step < warmup:
        return (step + 1) / max(1, warmup)
    if step >= max_iters:
        return min_ratio
    t = (step - warmup) / max(1, max_iters - warmup)
    return min_ratio + 0.5 * (1 - min_ratio) * (1 + math.cos(math.pi * t))


def load_token_bin(path: Path) -> np.memmap:
    return np.memmap(path, dtype=np.uint16, mode="r")


def get_batch(data: np.memmap, batch_size: int, block_size: int, device: str,
              generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """B случайных окон по block_size токенов; x — окно, y — то же со сдвигом на 1."""
    hi = len(data) - block_size - 1
    assert hi > 0, f"файл слишком мал: {len(data)} токенов < {block_size + 2}"
    ix = torch.randint(hi, (batch_size,), generator=generator)
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    if device.startswith("cuda"):
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    return x, y


def save_ckpt(path: Path, payload: dict) -> None:
    """Атомарное сохранение: .tmp + rename, чтобы SIGTERM не оставил битый файл."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def find_resume(out_dir: Path) -> Path | None:
    cands = glob.glob(str(out_dir / "iter_*.pt")) + glob.glob(str(out_dir / "last.pt"))
    if not cands:
        return None
    last = max((Path(c).stat().st_mtime, Path(c)) for c in cands)[1]
    print(f"[resume] продолжаю с {last.name}")
    return last


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    is_cuda = device.startswith("cuda")
    if is_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # --- данные ---
    train_file = args.train_file or (args.data_dir / "train.bin")
    val_file = args.val_file or (args.data_dir / "val.bin")
    train_data = load_token_bin(train_file)
    val_data = load_token_bin(val_file)
    vocab_size = 50257
    print(f"train: {len(train_data):,} токенов, val: {len(val_data):,} токенов")

    # --- модель ---
    # --- конфиг: из CLI, при возобновлении — из чекпоинта (чтобы переопределить CLI) ---
    ckpt_path = None if args.no_resume else find_resume(args.out_dir)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False) if ckpt_path else None
    if ckpt is not None and "config" in ckpt:
        config = GPT2Config(**ckpt["config"])
    else:
        config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=args.n_positions,
            n_embd=args.n_embd,
            n_layer=args.n_layer,
            n_head=args.n_head,
            attn_pdrop=args.dropout,
            resid_pdrop=args.dropout,
            embd_pdrop=args.dropout,
        )
    # синхронизируем аргументы CLI, зависящие от конфига
    args.n_positions = config.n_positions
    model = GPT2LMHeadModel(config)
    n_params = model.num_params()
    print(f"модель: {n_params:,} параметров ({n_params/1e6:.0f}M), device={device}")
    model = model.to(device)

    if args.compile and is_cuda:
        model = torch.compile(model)

    # --- точность ---
    if args.dtype == "auto":
        amp_dtype = torch.bfloat16 if is_cuda and torch.cuda.is_bf16_supported() else (
            torch.float16 if is_cuda else torch.float32)
    else:
        amp_dtype = getattr(torch, args.dtype)
    use_scaler = amp_dtype == torch.float16
    amp_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
        if is_cuda and amp_dtype != torch.float32 else nullcontext()
    )
    scaler = torch.amp.GradScaler(enabled=use_scaler)
    print(f"точность: {amp_dtype}, scaler={use_scaler}")

    # --- оптимизатор (decay только на матрицах; LN и bias без него) ---
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (decay if p.dim() >= 2 else no_decay).append(p)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, args.beta2), eps=1e-8, fused=is_cuda)

    tokens_per_step = args.batch_size * args.grad_accum * args.n_positions
    max_iters = args.max_iters
    if args.max_tokens is not None:
        max_iters = math.ceil(args.max_tokens / tokens_per_step)
    print(f"токенов на шаг: {args.batch_size}×{args.grad_accum}×{args.n_positions} = "
          f"{tokens_per_step:,}; шагов: {max_iters} (~{max_iters * tokens_per_step/1e9:.2f}B токенов)")

    # --- восстановление состояния ---
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(args.out_dir / "train.log", "a")
    step = 0
    best_val = float("inf")

    if ckpt is not None:
        raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        raw_model.load_state_dict(ckpt["model"])
        if ckpt.get("optimizer"):
            optimizer.load_state_dict(ckpt["optimizer"])
        step = ckpt.get("step", 0)
        best_val = ckpt.get("best_val", float("inf"))
        rng = ckpt.get("rng") or {}
        # rng-состояния должны быть CPU ByteTensor, а чекпоинт загружен с
        # map_location=cuda — возвращаем на CPU перед set_rng_state
        rng_t = rng.get("torch")
        if rng_t is not None:
            torch.set_rng_state(rng_t.cpu().to(torch.uint8).contiguous())
        rng_c = rng.get("cuda")
        if is_cuda and rng_c is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in rng_c])
        print(f"[resume] шаг {step}, best_val {best_val:.4f}")

    # --- прогресс-бары: как в предыдущей попытке (train + valid, с postfix) ---
    bars_active = (not args.no_bar) and sys.stderr.isatty()
    train_bar = val_bar = None
    if bars_active:
        # shutil.get_terminal_size, а не внутренний запрос tqdm: при нулевом
        # размере окна (некоторые pty/IDE) tqdm 4.70 молча рисует пустой бар
        import shutil
        ncols = shutil.get_terminal_size().columns or None
        train_bar = tqdm(total=float(max_iters), initial=float(step),
                         desc=f"train {n_params/1e6:.0f}M", position=0,
                         leave=False, ncols=ncols, unit="step")
        val_bar = tqdm(total=float(args.eval_iters), desc="valid", position=1,
                       leave=False, ncols=ncols, unit="batch")

    def log(msg: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        line = f"[{stamp}] {msg}"
        if bars_active:
            tqdm.write(line)  # не ломает отрисовку баров
        else:
            print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    # --- оценка ---
    val_gen = torch.Generator().manual_seed(args.seed)

    @torch.no_grad()
    def evaluate() -> float:
        model.eval()
        losses = torch.zeros(args.eval_iters)
        if val_bar is not None:
            val_bar.reset()
        for k in range(args.eval_iters):
            x, y = get_batch(val_data, args.batch_size, args.n_positions, device, val_gen)
            with amp_ctx:
                _, loss = model(x, y)
            losses[k] = loss.detach().float()
            if val_bar is not None:
                val_bar.update(1)
                val_bar.set_postfix(loss=f"{losses[:k + 1].mean().item():.4f}")
        model.train()
        return losses.mean().item()

    # --- обучающий цикл ---
    batch_gen = torch.Generator().manual_seed(args.seed + 1)
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    flops_per_token = 6 * raw_model.num_params(non_embedding=True) + \
        12 * config.n_layer * config.n_embd * config.n_positions
    t0 = time.time()
    cur_lr = args.lr
    log(f"старт: max_iters={max_iters}, lr={args.lr}, warmup={args.warmup_steps}")

    while step < max_iters:
        # --- валидация/сэмпл/чекпоинт раз в eval-interval ---
        if step > 0 and step % args.eval_interval == 0:
            val_loss = evaluate()
            improved = val_loss < best_val
            if improved:
                best_val = val_loss
            log(f"step {step}: val_loss {val_loss:.4f}"
                + (f" (лучшая, сохраняю best.pt)" if improved else f" (best {best_val:.4f})"))
            if improved:
                save_ckpt(args.out_dir / "best.pt", {
                    "model": raw_model.state_dict(), "config": dataclasses.asdict(config),
                    "step": step, "val_loss": val_loss, "args": vars(args),
                })
            # качественный сэмпл для мониторинга
            with torch.no_grad():
                ctx = torch.tensor([[50256]], device=device)  # <|endoftext|>
                out = raw_model.generate(ctx, max_new_tokens=120, temperature=0.8, top_k=50)
            try:
                from gpt2rep.tokenizer import GPT2Tokenizer
                text = GPT2Tokenizer.load(REPO / "data" / "tokenizer").decode(out[0].tolist())
                log("sample: " + text.replace("\n", " ")[:200])
            except Exception as e:  # noqa: BLE001 — сэмпл не должен ронять обучение
                log(f"(сэмпл не удался: {e})")
            save_ckpt(args.out_dir / "last.pt", {
                "model": raw_model.state_dict(), "optimizer": optimizer.state_dict(),
                "config": dataclasses.asdict(config),
                "step": step, "best_val": best_val, "args": vars(args),
                "rng": {"torch": torch.get_rng_state(),
                        "cuda": torch.cuda.get_rng_state_all() if is_cuda else None},
            })

        # --- шаг с градиентным накоплением ---
        optimizer.zero_grad(set_to_none=True)
        loss_acc = torch.zeros((), device=device)   # среднее по шагу (для лога)
        loss_sum = torch.zeros((), device=device)   # честная сумма loss микро-шагов (для бара)
        last_post = time.time()
        tokens_since = 0
        for micro in range(args.grad_accum):
            x, y = get_batch(train_data, args.batch_size, args.n_positions, device, batch_gen)
            with amp_ctx:
                _, loss = model(x, y)
                loss_sum += loss.detach()
                loss = loss / args.grad_accum
            scaler.scale(loss).backward()
            loss_acc += loss.detach()
            tokens_since += args.batch_size * args.n_positions
            if train_bar is not None:
                # точная позиция с учётом накопления float-дробей
                target = step + (micro + 1) / args.grad_accum
                train_bar.update(target - train_bar.n)
                now = time.time()
                if now - last_post >= 2.0 or micro == args.grad_accum - 1:
                    tps = tokens_since / max(now - last_post, 1e-6)
                    train_bar.set_postfix(loss=f"{(loss_sum / (micro + 1)).item():.4f}",
                                          lr=f"{cur_lr:.2e}", tok_s=f"{tps/1e3:.0f}K")
                    tokens_since = 0
                    last_post = now
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        step += 1
        cur_lr = args.lr * get_lr_frac(step, args.warmup_steps, max_iters, args.min_lr_ratio)
        for group in optimizer.param_groups:
            group["lr"] = cur_lr
        lr = cur_lr

        if step % args.log_interval == 0 or step == max_iters:
            dt = time.time() - t0
            tps = tokens_per_step * args.log_interval / dt
            loss_val = loss_acc.item()
            log(f"step {step}/{max_iters} train_loss {loss_val:.4f} lr {lr:.2e} "
                f"{tps/1e3:.0f}K tok/s {dt/args.log_interval:.2f}s/шаг "
                f"осталось ~{(max_iters - step) * dt / args.log_interval / 3600:.1f}ч")
            t0 = time.time()

    save_ckpt(args.out_dir / "last.pt", {
        "model": raw_model.state_dict(), "optimizer": optimizer.state_dict(),
        "config": dataclasses.asdict(config),
        "step": step, "best_val": best_val, "args": vars(args),
    })
    final_loss = evaluate()
    log(f"готово: step {step}, val_loss {final_loss:.4f}, best {best_val:.4f}")
    if is_cuda:
        peak = torch.cuda.max_memory_allocated() / 2**30
        log(f"пик VRAM (PyTorch): {peak:.2f} GiB")
    if train_bar is not None:
        train_bar.close()
    if val_bar is not None:
        val_bar.close()
    log_file.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[прервано] обучение продолжится с последнего last.pt — "
              "запустите ту же команду ещё раз", file=sys.stderr)
        sys.exit(130)
