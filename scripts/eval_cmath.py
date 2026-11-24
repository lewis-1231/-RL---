"""CMath 评测：贪心解码 + judge 判分，支持 base / adapter（SFT 或 GRPO）。

评测纪律（务必遵守）：
  - 选 checkpoint / 调参 → 只用 cmath_dev.jsonl（--split dev）
  - cmath_test.jsonl 每个模型只允许评一次，且只评最终选定的那个 checkpoint

用法：
  python scripts/eval_cmath.py --model Qwen/Qwen3-1.7B --split dev --tag base
  python scripts/eval_cmath.py --model Qwen/Qwen3-1.7B \
      --adapter out/grpo/checkpoint-500 --split test --tag grpo_ckpt500
"""
import argparse
import csv
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from judge import answers_equal, extract_boxed  # noqa: E402

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYSTEM = "你是一个严谨的数学解题助手。请一步步推理，并把最终答案放在 \\boxed{} 中。"


def load_rows(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def summarize_buckets(rows, flags, field):
    """按 grade / reasoning_step 分桶统计。flags 与 rows 一一对应。"""
    buckets = {}
    for r, ok in zip(rows, flags):
        k = r.get(field)
        if k is None:
            continue
        c, n = buckets.get(k, (0, 0))
        buckets[k] = (c + int(ok), n + 1)
    if not buckets:
        return
    print(f"[by {field}]")
    for k in sorted(buckets):
        c, n = buckets[k]
        print(f"  {k}: {c / max(1, n):.4f}  (n={n})")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    _local = os.path.join(PROJECT, "models", "qwen3-1.7b")
    ap.add_argument("--model", default=_local if os.path.isdir(_local) else "Qwen/Qwen3-1.7B",
                    help="基模路径；默认本地 modelscope 下载目录，离线优先")
    ap.add_argument("--adapter", default=None, help="LoRA adapter 目录（GRPO checkpoint / SFT final）")
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--tag", required=True, help="结果文件命名，如 base / sft / grpo_ckpt500")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=768)
    args = ap.parse_args()

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
        print(f"[adapter] {args.adapter}", flush=True)
    model.eval()

    rows = load_rows(os.path.join(PROJECT, "data", f"cmath_{args.split}.jsonl"))
    if args.limit:
        rows = rows[: args.limit]
    print(f"[data] {args.split}: {len(rows)} rows", flush=True)

    out_path = os.path.join(PROJECT, "results", f"cmath_{args.split}_{args.tag}.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    t0, n_correct = time.time(), 0
    flags = []
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["question", "golden", "pred_boxed", "pred_raw_head", "correct", "output_tail"])
        for i in range(0, len(rows), args.batch):
            chunk = rows[i : i + args.batch]
            prompts = [
                tok.apply_chat_template(
                    [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": r["question"]}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False,
                )
                for r in chunk
            ]
            enc = tok(prompts, return_tensors="pt", padding=True, padding_side="left").to(model.device)
            gen = model.generate(
                **enc, max_new_tokens=args.max_new, do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
            outs = tok.batch_decode(gen[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
            for r, o in zip(chunk, outs):
                pred = extract_boxed(o)
                ok = answers_equal(pred, r["answer"])
                flags.append(ok)
                n_correct += int(ok)
                w.writerow([r["question"], r["answer"], pred or "", o.strip()[:120],
                            int(ok), o[-300:].replace("\n", " ")])
            if (i // args.batch) % 5 == 0:
                n = len(flags)
                print(f"[{n}/{len(rows)}] acc={n_correct / max(1, n):.4f}  {time.time() - t0:.0f}s", flush=True)

    n = len(flags)
    print(f"\n[RESULT] tag={args.tag} split={args.split} n={n} "
          f"acc={n_correct / max(1, n):.4f} elapsed={time.time() - t0:.0f}s -> {out_path}", flush=True)
    summarize_buckets(rows, flags, "grade")
    summarize_buckets(rows, flags, "reasoning_step")


if __name__ == "__main__":
    main()
