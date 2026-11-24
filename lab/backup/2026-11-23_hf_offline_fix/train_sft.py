"""SFT 对照臂（可选，1 小时级）：手写 HF Trainer，loss 只算 completion 部分。

用法：
  python scripts/train_sft.py --smoke                    # 64 条冒烟，验链路
  python scripts/train_sft.py                            # 5k 条 × 2 epoch，约 1 小时
"""
import argparse
import json
import os
import sys

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAX_LEN = 1024
SYSTEM = "你是一个严谨的数学解题助手。请一步步推理，并把最终答案放在 \\boxed{} 中。"


class SftJsonl(Dataset):
    def __init__(self, path, tokenizer):
        self.tok = tokenizer
        self.rows = [json.loads(l) for l in open(path, encoding="utf-8")]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": r["problem"]},
        ]
        prompt = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        return {"prompt": prompt, "completion": r["solution"]}


def collate(batch, tokenizer):
    input_ids_all, labels_all, attn_all = [], [], []
    for x in batch:
        prompt_ids = tokenizer(x["prompt"], add_special_tokens=False).input_ids
        comp_ids = tokenizer(x["completion"] + tokenizer.eos_token, add_special_tokens=False).input_ids
        ids = (prompt_ids + comp_ids)[:MAX_LEN]
        labels = ([-100] * len(prompt_ids) + comp_ids)[:MAX_LEN]
        pad = MAX_LEN - len(ids)
        input_ids_all.append(ids + [tokenizer.pad_token_id] * pad)
        labels_all.append(labels + [-100] * pad)
        attn_all.append([1] * len(ids) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids_all),
        "labels": torch.tensor(labels_all),
        "attention_mask": torch.tensor(attn_all),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--train-file", default=os.path.join(PROJECT, "data", "train.jsonl"))
    ap.add_argument("--output-dir", default=os.path.join(PROJECT, "out", "sft"))
    ap.add_argument("--n-train", type=int, default=5000)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = [json.loads(l) for l in open(args.train_file, encoding="utf-8")]
    if args.smoke:
        rows = rows[:64]
    else:
        rows = rows[: args.n_train]
    tmp = args.train_file.replace(".jsonl", "_used.jsonl")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    peft_config = LoraConfig(
        task_type="CAUSAL_LM", r=args.lora_rank, lora_alpha=2 * args.lora_rank,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    model = get_peft_model(model, peft_config)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=1,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        report_to=["tensorboard"],
        seed=42,
        remove_unused_columns=False,
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=SftJsonl(tmp, tokenizer),
        data_collator=lambda b: collate(b, tokenizer),
    )
    trainer.train()
    model.save_pretrained(os.path.join(args.output_dir, "final_adapter"))
    print("[done] adapter saved to", os.path.join(args.output_dir, "final_adapter"), flush=True)


if __name__ == "__main__":
    main()
