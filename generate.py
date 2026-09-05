"""Генерация текста обученной моделью.

Примеры:
  python generate.py --prompt "The meaning of life is"
  python generate.py --ckpt checkpoints/run1/best.pt --temperature 0.8 --top-k 50
  python generate.py --greedy --tokens 200
"""

import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from gpt2rep.model import GPT2Config, GPT2LMHeadModel
from gpt2rep.tokenizer import GPT2Tokenizer


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=Path, default=REPO / "checkpoints" / "run1" / "best.pt",
                   help="чекпоинт обучения (best.pt / last.pt / iter_N.pt)")
    p.add_argument("--hf-safetensors", type=Path, default=None,
                   help="вместо своего чекпоинта загрузить эталон (model.safetensors)")
    p.add_argument("--prompt", type=str, default="The meaning of life is")
    p.add_argument("--tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--greedy", action="store_true", help="арgmах вместо сэмплирования")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = GPT2Tokenizer.load(REPO / "data" / "tokenizer")

    if args.hf_safetensors:
        model = GPT2LMHeadModel.from_hf_safetensors(str(args.hf_safetensors), device=device)
    else:
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        model = GPT2LMHeadModel(GPT2Config(**ckpt["config"]))
        model.load_state_dict(ckpt["model"])
        print(f"загружен {args.ckpt} (шаг {ckpt.get('step', '?')}, "
              f"val_loss {ckpt.get('val_loss', ckpt.get('best_val', '?'))})")
    model = model.to(device).eval()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    ctx = torch.tensor([tok.encode(args.prompt)], device=device)
    out = model.generate(
        ctx,
        max_new_tokens=args.tokens,
        temperature=0.0 if args.greedy else args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        eot_token=tok.eot_token,
    )
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
