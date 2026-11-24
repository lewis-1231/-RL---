# GRPO 强化学习后训练（Qwen3-1.7B 数学推理）

上一个项目（qwen3-4b-math-sft）的续集：SFT 在强基座上是负增益（CMath 0.909 → 0.878），
根因定位为过拟合，结论指向"用可验证奖励的 RL 替代继续调 SFT 超参"。本项目把这一步补上。

**核心实验：同任务三臂对照**

| 臂 | 模型 | 说明 |
|---|---|---|
| A | Qwen3-1.7B-Instruct-2507 base | 基线 |
| B | base + LoRA SFT（5k × 2 epoch） | 复现 SFT 负增益 |
| C | base + LoRA GRPO（可验证奖励） | 主实验 |

评测口径：CMath dev 600（选 checkpoint / 观察）→ CMath test（最终汇报，每臂只评一次）。
奖励 = 答案判分 1.0 + `\boxed{}` 格式分 0.2，判分逻辑与评测共用 `scripts/judge.py`（口径一致性是防 reward hacking 的底线）。

## 目录

```
data/       train.jsonl（GRPO/SFT 训练）、cmath_dev/test.jsonl（评测）、heldout.jsonl（训练侧留出）
scripts/
  judge.py          共用判分器（boxed 提取 + 数值容差匹配）
  build_dataset.py  NuminaMath 清洗分层抽样 + CMath 规范化
  train_grpo.py     GRPO 主脚本（TRL GRPOTrainer + LoRA r32）
  train_sft.py      SFT 对照臂（loss 只算 completion）
  eval_cmath.py     评测（贪心解码，自动按 grade/reasoning_step 分桶）
  setup.sh          环境一键准备
configs/    （超参均以 argparse 暴露，暂无 yaml）
results/    逐题明细 csv
```

## 执行顺序

```bash
bash scripts/setup.sh                                   # 依赖 + 数据 + 基座，含链路自检

# 1) 冒烟标定：跑 20 步，掐表记单步耗时，线性外推总时长
python scripts/train_grpo.py --smoke

# 2) SFT 对照臂（约 1 小时）
python scripts/train_sft.py

# 3) GRPO 正式训练（vLLM 加速约 3-4.5h；不加 --vllm 则 8-12h，靠断点续训）
python scripts/train_grpo.py --max-steps 500 --vllm
# 中断后续训：
python scripts/train_grpo.py --max-steps 500 --vllm --resume auto

# 4) 评测三臂（dev 评所有 checkpoint 选点；test 只评最终选定者）
python scripts/eval_cmath.py --model Qwen/Qwen3-1.7B-Instruct-2507 --split dev --tag base
python scripts/eval_cmath.py --model Qwen/Qwen3-1.7B-Instruct-2507 --adapter out/sft/final_adapter --split dev --tag sft
python scripts/eval_cmath.py --model Qwen/Qwen3-1.7B-Instruct-2507 --adapter out/grpo/checkpoint-500 --split dev --tag grpo_ckpt500
# 终报（test，一次）：
python scripts/eval_cmath.py --model Qwen/Qwen3-1.7B-Instruct-2507 --adapter <最终checkpoint> --split test --tag <臂名>_final
```

## 三条硬纪律

1. **test 只评一次**。选点、调参一律用 dev；test 上的数字是汇报数字，评第二次就废了。
2. **save_total_limit=None**。全部 checkpoint 保留，最优点不许被轮转删（上一个项目踩过）。
3. **判分口径唯一**。奖励函数和评测脚本共用 judge.py，改动即两处同时改。

## 已知坑（提前踩好）

1. **组内全对/全错 → 优势全 0**：太简单的题学不到东西。看 reward 日志方差，长期全 0 的题源要过滤。
2. **长度塌缩/膨胀**：盯 completion 长度曲线，异常膨胀通常伴随 reward hacking。
3. **8 小时会话上限**：HF 路线 500 步约 8-12h，必须 `--resume auto`；vLLM 路线单会话内完成。实例重启后系统盘重置，pip 装的东西会丢（setup.sh 重跑即可），/mnt/workspace 持久化。**任何时候不要删除实例。**
4. **LoRA rank**：RL 依赖探索，rank 16 可能学不动；默认 32，曲线纹丝不动先升 rank 再查别的。
5. **TRL 版本漂移**：GRPOConfig 参数在版本间有增删。若报未知参数，先查 `python -c "from trl import GRPOConfig; print(GRPOConfig)"`，对应删改；smoke 通过 = 链路确认，再跑正式。
6. **vLLM colocate 显存**：`vllm_gpu_memory_utilization=0.25` 是保守值，OOM 就降 num_generations 到 4，或去掉 --vllm 退回 HF 路线。
7. **CMath 数据结构**：evalscope/cmath 的字段名以实际下载为准，build_dataset.py 已做多字段兼容；若 test 拆分未识别，手工检查 data/cmath/ 后按文件名归位。

## 汇报格式（跑完后填）

| 模型 | CMath test acc | Δ vs base | 备注 |
|---|---|---|---|
| base 1.7B | | | |
| + LoRA SFT | | | 预期复现负增益或持平 |
| + LoRA GRPO | | | 主实验 |
