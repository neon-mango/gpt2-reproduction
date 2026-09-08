"""Оценка чекпоинта: val_loss и перплексия на валидации + контрольные сэмплы.

В отличие от оценки внутри train.py (20 батчей, ~82K токенов — шум ±0.01),
здесь окно последовательных непересекающихся фрагментов задаётся --tokens.

Примеры:
  ./venv/bin/python scripts/eval_model.py                     # best.pt прогона
  ./venv/bin/python scripts/eval_model.py --ckpt checkpoints/run1/last.pt
  ./venv/bin/python scripts/eval_model.py --tokens 5000000    # вся валидация
  ./venv/bin/python scripts/eval_model.py --no-samples        # только цифры
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from gpt2rep.model import GPT2Config, GPT2LMHeadModel
from gpt2rep.tokenizer import GPT2Tokenizer


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=Path, default=REPO / "checkpoints" / "run1" / "best.pt")
    p.add_argument("--val-file", type=Path, default=REPO / "data" / "tokens" / "val.bin")
    p.add_argument("--tokens", type=int, default=2_000_000,
                   help="сколько валидационных токенов оценить")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--no-samples", action="store_true")
    p.add_argument("--temperature", type=float, default=0.8)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = GPT2LMHeadModel(GPT2Config(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    recorded = ckpt.get("val_loss", ckpt.get("best_val"))
    print(f"чекпоинт {args.ckpt}: шаг {ckpt.get('step', '?')}, "
          f"записанный val_loss {recorded if recorded is None else round(recorded, 4)}")

    data = np.memmap(args.val_file, dtype=np.uint16, mode="r")
    T = model.config.n_positions
    B = args.batch_size
    n_windows = min((len(data) - 1) // T, args.tokens // T)
    n_batches = n_windows // B
    if n_batches == 0:
        sys.exit("валидации не хватает даже на один батч")

    # последовательные непересекающиеся окна: детерминированная оценка
    loss_sum = torch.zeros((), device=device)
    n_tok = 0
    with torch.no_grad():
        for k in range(n_batches):
            starts = torch.arange(k * B, (k + 1) * B) * T
            x = torch.stack([torch.from_numpy(data[s:s + T].astype(np.int64)) for s in starts])
            y = torch.stack([torch.from_numpy(data[s + 1:s + 1 + T].astype(np.int64)) for s in starts])
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast("cuda", torch.bfloat16, enabled=device.startswith("cuda")):
                _, loss = model(x, y)
            loss_sum += loss.float() * (B * T)
            n_tok += B * T
            if (k + 1) % 25 == 0:
                cur = (loss_sum / n_tok).item()
                print(f"  {n_tok:,}/{n_batches * B * T:,} токенов: loss {cur:.4f}", flush=True)

    avg = (loss_sum / n_tok).item()
    print(f"\nИТОГ: val_loss {avg:.4f} | перплексия {math.exp(avg):.2f} | "
          f"токенов оценено: {n_tok:,}")

    if not args.no_samples:
        tok = GPT2Tokenizer.load(REPO / "data" / "tokenizer")
        for prompt in ["The meaning of life is", "Breaking news:", "Once upon a time"]:
            ctx = torch.tensor([tok.encode(prompt)], device=device)
            out = model.generate(ctx, max_new_tokens=150, temperature=args.temperature,
                                 top_k=50, eot_token=tok.eot_token)
            print(f"\n--- {prompt!r}\n" + tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
