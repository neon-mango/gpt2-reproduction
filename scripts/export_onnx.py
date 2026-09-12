"""Экспорт чекпоинта GPT-2 в ONNX для запуска в браузере (onnxruntime-web).

Экспортируется обёртка OnnxWrapper: (ids, state, past_len) ->
(логиты последнего токена fp32, новое состояние KV-кэша). Префилл — вызов с
состоянием нулевой длины, декодирование — по одному токену с растущим кэшем.

Проверка экспорта двухэтапная:
  1) временная fp32-копия сверяется с torch жёстко (max|Δ| <= 0.05) —
     доказывает корректность самого графа;
  2) итоговый файл (обычно fp16) сверяется по argmax и overlap top-5 —
     накопления fp16 дают разброс абсолютных значений при том же выборе
     токенов.

Использование:
  ./venv/bin/pip install onnx onnxruntime onnxscript
  ./venv/bin/python scripts/export_onnx.py                       # best.pt -> fp16
  ./venv/bin/python scripts/export_onnx.py --dtype fp32 --ckpt checkpoints/run1/last.pt
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from gpt2rep.model import GPT2Config, GPT2LMHeadModel, OnnxWrapper

WEB = REPO / "web"


def export_model(wrapper: OnnxWrapper, out: Path, dtype: str, device: str) -> None:
    n_layer, n_head, head_dim = wrapper.model.config.n_layer, \
        wrapper.model.config.n_head, wrapper.model.config.n_embd // wrapper.model.config.n_head
    dummy_ids = torch.randint(0, wrapper.model.config.vocab_size, (1, 8),
                              dtype=torch.int64, device=device)
    dummy_state = torch.zeros(n_layer, 2, 1, n_head, 0, head_dim, device=device,
                              dtype=torch.float16 if dtype == "fp16" else torch.float32)
    dummy_len = torch.zeros((), dtype=torch.int64, device=device)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (dummy_ids, dummy_state, dummy_len),
        str(out),
        input_names=["ids", "state", "past_len"],
        output_names=["logits", "new_state"],
        dynamic_axes={
            "ids": {0: "batch", 1: "seq"},
            "state": {2: "batch", 4: "past"},
            "past_len": {},
            "logits": {0: "batch"},
            "new_state": {2: "batch", 4: "past"},
        },
        opset_version=17,
        dynamo=False,  # легаси-экспортёр: честно соблюдает opset 17
    )
    # легаси-экспортёр оставляет модуль в train-режиме — без этого
    # последующая сверка с torch пойдёт с активным dropout
    wrapper.eval()
    print(f"экспортировано: {out} ({out.stat().st_size / 2**20:.0f} MiB)")


def verify(path: Path, wrapper: OnnxWrapper, tight: bool) -> None:
    """prefill + 3 шага декодирования: ONNX против torch.

    tight=True: жёсткий допуск по абсолютной ошибке (fp32-граф).
    tight=False: argmax и overlap top-5 обязаны совпадать (fp16-допуск)."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    torch_dtype = wrapper.model.transformer.wte.weight.dtype
    n_layer, n_head, head_dim = wrapper.model.config.n_layer, \
        wrapper.model.config.n_head, wrapper.model.config.n_embd // wrapper.model.config.n_head
    vocab = wrapper.model.config.vocab_size

    torch.manual_seed(0)
    prompt = torch.randint(0, vocab - 1, (1, 6), dtype=torch.int64) + 1
    state = torch.zeros(n_layer, 2, 1, n_head, 0, head_dim, dtype=torch_dtype)
    past_len = torch.zeros((), dtype=torch.int64)

    def ort_run(ids, st, pl):
        outs = sess.run(None, {
            "ids": ids.numpy().astype(np.int64),
            "state": st.numpy(),
            "past_len": np.asarray(pl, dtype=np.int64).reshape(()),
        })
        return torch.from_numpy(outs[0]), torch.from_numpy(outs[1])

    def top5(t: torch.Tensor) -> set[int]:
        return set(torch.topk(t.flatten(), 5).indices.tolist())

    def check(name, a: torch.Tensor, b: torch.Tensor) -> None:
        if tight:
            diff = (a - b).abs().max().item()
            print(f"  {name}: max|Δ| = {diff:.5f} (допуск 0.05) {'OK' if diff <= 0.05 else 'FAIL'}")
            assert diff <= 0.05
        else:
            same_argmax = a.argmax().item() == b.argmax().item()
            overlap = len(top5(a) & top5(b))
            scale = b.abs().max().item()
            diff = (a - b).abs().max().item()
            print(f"  {name}: max|Δ| = {diff:.3f} (шкала ±{scale:.0f}), "
                  f"argmax {'OK' if same_argmax else 'FAIL'}, top-5 {overlap}/5")
            assert same_argmax and overlap == 5, f"{name}: ONNX разошёлся с torch"

    with torch.no_grad():
        t_logits, t_state = wrapper(prompt, state, past_len)
        o_logits, o_state = ort_run(prompt, state, past_len)
        check("префилл logits", o_logits, t_logits.squeeze(1).float())

        cur_len = prompt.shape[1]
        for step in range(3):
            next_tok = t_logits.argmax().view(1, 1).to(torch.int64)
            pl = torch.tensor(cur_len, dtype=torch.int64)
            t_logits, t_state = wrapper(next_tok, t_state, pl)
            o_logits, o_state = ort_run(next_tok, o_state, pl)
            check(f"decode {step + 1} logits", o_logits, t_logits.squeeze(1).float())
            cur_len += 1
    print(f"проверка {path.name}: {'строгая' if tight else 'argmax/top-5'} пройдена")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=Path, default=REPO / "checkpoints" / "run1" / "best.pt")
    p.add_argument("--out", type=Path, default=WEB / "public" / "gpt2_124m.onnx")
    p.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16",
                   help="fp16 вдвое меньше и быстрее на WebGPU")
    p.add_argument("--skip-verify", action="store_true")
    args = p.parse_args()

    # экспорт полностью на CPU: трассировке всё равно, а так не будет
    # рассинхронизации устройств между весами и dummy-входами
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    config = GPT2Config(**ckpt["config"])
    model = GPT2LMHeadModel(config)
    model.load_state_dict(ckpt["model"])
    model.eval()
    wrapper = OnnxWrapper(model)
    print(f"чекпоинт {args.ckpt} (шаг {ckpt.get('step', '?')}), dtype {args.dtype}")

    if not args.skip_verify:
        if args.dtype == "fp16":
            tmp = args.out.with_name(args.out.stem + "_fp32check.onnx")
            export_model(wrapper, tmp, "fp32", "cpu")
            verify(tmp, wrapper, tight=True)
            tmp.unlink()
            wrapper.half()
        export_model(wrapper, args.out, args.dtype, "cpu")
        verify(args.out, wrapper, tight=False)
    else:
        if args.dtype == "fp16":
            wrapper.half()
        export_model(wrapper, args.out, args.dtype, "cpu")

    # файлы для веб-страницы
    WEB.mkdir(parents=True, exist_ok=True)
    for f in ("encoder.json", "vocab.bpe"):
        shutil.copy(REPO / "data" / "tokenizer" / f, WEB / "public" / f)
    # ключи release/repo не перетираем: их редактирует пользователь
    # (тег пинает модель к коммиту фронтенда)
    config_path = WEB / "public" / "config.json"
    merged = json.loads(config_path.read_text()) if config_path.exists() else {}
    merged.update({
        "dtype": args.dtype,
        "n_layer": config.n_layer, "n_head": config.n_head,
        "head_dim": config.n_embd // config.n_head,
        "vocab_size": config.vocab_size, "n_positions": config.n_positions,
        "ckpt_step": ckpt.get("step"),
    })
    merged.setdefault("release", "latest")
    merged.setdefault("repo", "")
    config_path.write_text(json.dumps(merged, indent=2))
    print("скопированы encoder.json, vocab.bpe и записан web/public/config.json")
    print("скопированы encoder.json, vocab.bpe и записан web/config.json")


if __name__ == "__main__":
    main()
