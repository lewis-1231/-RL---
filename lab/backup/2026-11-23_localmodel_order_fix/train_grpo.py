"""GRPO 训练（TRL GRPOTrainer + LoRA）。

用法：
  python scripts/train_grpo.py --smoke                 # 20 步冒烟，标定单步耗时
  python scripts/train_grpo.py --max-steps 500         # 正式训练
  python scripts/train_grpo.py --max-steps 500 --vllm  # vLLM colocate 加速 rollout
  python scripts/train_grpo.py --max-steps 500 --resume auto  # 断点续训

要点：
  - 奖励 = 答案判分（1.0/0.0）+ boxed 格式分（0.2），判分逻辑与 eval_cmath.py 共用 judge.py
  - LoRA rank 默认 32：RL 需要探索半径，rank 16 可能学不动（见 README 坑 4）
  - save_total_limit=None：保留全部 checkpoint，防止最优点被轮转删除（上一个项目的教训）
"""
import argparse
import inspect
import os
import sys

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from judge import answers_equal, extract_boxed  # noqa: E402

LOCAL_MODEL = os.path.join(PROJECT, "models", "qwen3-1.7b")  # setup.sh 的 modelscope 下载目录
# 离线优先：本地有就用本地，避免 transformers 去连 huggingface.co（大陆云不可达）
BASE_MODEL = LOCAL_MODEL if os.path.isdir(LOCAL_MODEL) else "Qwen/Qwen3-1.7B"
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT, "out", "grpo")

FORMAT_REWARD = 0.2
CORRECT_REWARD = 1.0


def _filter_kwargs(cls, kwargs):
    """版本兼容：按当前类签名过滤不支持的参数（README 坑 5，TRL GRPOConfig 跨版本增删）。"""
    valid = inspect.signature(cls.__init__).parameters
    ok = {k: v for k, v in kwargs.items() if k in valid}
    dropped = sorted(set(kwargs) - set(ok))
    if dropped:
        print(f"[compat] {cls.__name__} 不支持，已丢弃: {dropped}", flush=True)
    return ok


def build_prompt(problem: str, tokenizer) -> str:
    messages = [
        {"role": "system", "content": "你是一个严谨的数学解题助手。请一步步推理，并把最终答案放在 \\boxed{} 中。"},
        {"role": "user", "content": problem},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def correctness_reward(prompts, completions, answer=None, **kwargs):
    """答案判分。answer 列由 GRPOTrainer 通过 dataset 透传进来。"""
    rewards = []
    for comp, gold in zip(completions, answer):
        pred = extract_boxed(comp)
        rewards.append(CORRECT_REWARD if answers_equal(pred, gold) else 0.0)
    return rewards


def format_reward(prompts, completions, **kwargs):
    return [FORMAT_REWARD if extract_boxed(c) is not None else 0.0 for c in completions]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=BASE_MODEL, help="基座或 SFT 后的模型路径")
    ap.add_argument("--train-file", default=os.path.join(PROJECT, "data", "train.jsonl"))
    ap.add_argument("--output-dir", default=OUT_DIR)
    ap.add_argument("--smoke", action="store_true", help="20 步冒烟")
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--per-device-batch", type=int, default=64, help="每步生成的 completion 数（须为 num_generations 整数倍）")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=0.0, help="KL 系数，可验证奖励下可取 0")
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--max-completion-len", type=int, default=512)
    ap.add_argument("--max-prompt-len", type=int, default=384)
    ap.add_argument("--vllm", action="store_true", help="vLLM colocate 加速 rollout")
    ap.add_argument("--resume", default=None, help="'auto' 自动找最新 checkpoint")
    args = ap.parse_args()

    max_steps = 20 if args.smoke else args.max_steps
    save_steps = 100 if not args.smoke else 20

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset("json", data_files=args.train_file, split="train")
    ds = ds.shuffle(seed=42)

    def add_prompt(x):
        return {"prompt": build_prompt(x["problem"], tokenizer), "answer": x["answer"]}

    ds = ds.map(add_prompt, remove_columns=[c for c in ds.column_names if c not in ("prompt", "answer")])
    if args.smoke:
        ds = ds.select(range(min(64, len(ds))))

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    model.config.use_cache = False

    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=args.lora_rank,
        lora_alpha=2 * args.lora_rank,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    extra = {}
    if args.vllm:
        extra.update(use_vllm=True, vllm_mode="colocate", vllm_gpu_memory_utilization=0.25)

    cfg = GRPOConfig(**_filter_kwargs(GRPOConfig, dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=1,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_len,
        max_completion_length=args.max_completion_len,
        learning_rate=args.lr,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=10,
        beta=args.beta,
        temperature=1.0,
        top_p=1.0,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=5,
        save_steps=save_steps,
        save_total_limit=None,           # 上一个项目的教训：最优点不能被轮转删掉
        max_steps=max_steps,
        report_to=["tensorboard"],
        seed=42,
        **extra,
    )))

    trainer = GRPOTrainer(
        model=model,
        args=cfg,
        reward_funcs=[correctness_reward, format_reward],
        train_dataset=ds,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    resume_path = None
    if args.resume == "auto":
        cands = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")] if os.path.isdir(args.output_dir) else []
        if cands:
            resume_path = os.path.join(args.output_dir, max(cands, key=lambda d: int(d.split("-")[-1])))
            print(f"[resume] from {resume_path}", flush=True)

    trainer.train(resume_from_checkpoint=resume_path)
    trainer.save_model(os.path.join(args.output_dir, "final_adapter"))
    print("[done] adapter saved to", os.path.join(args.output_dir, "final_adapter"), flush=True)


if __name__ == "__main__":
    main()
