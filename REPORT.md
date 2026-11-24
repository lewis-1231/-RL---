# Qwen3-1.7B 的 GRPO 后训练实验（CMath 评测）：一个负结果

> **一句话结论**：在单张 A10（24GB）上用 500 步 GRPO + LoRA r32 训练 Qwen3-1.7B，**未能在 CMath 上取得统计显著的提升**。dev 最好成绩 +1.0pp、test 仅 +0.36pp（多对 4 道题，McNemar 精确检验 p=0.71）。训练终点（500 步）反而比基座低 2.7pp。根因是评测基准接近饱和 + 训练/评测分布错位，而非 GRPO 本身失效。

---

## 1. 实验动机

验证一个常见假设：**可验证奖励的强化学习（GRPO）能否让小参数量模型（1.7B）在某些能力上超过"基座模型 + 提示词"的表现**。

设计上是单变量对照：同一基座、同一评测集、同一解码配置，唯一变化是有没有做 RL 后训练。

## 2. 实验设置

### 2.1 环境

| 项 | 配置 |
|---|---|
| 硬件 | 阿里云 PAI-DSW 交互式建模，单卡 **NVIDIA A10（24GB，可用 22.07GB）** |
| 镜像 | `pai/modelscope:1.40.1-pytorch2.13.0-gpu-py312-cu130-ubuntu22.04` |
| 关键依赖 | transformers（新版，`torch_dtype` 已弃用）、TRL（GRPOTrainer）、PyTorch 2.x + CUDA 13 |
| 实际耗时 | 500 步 ≈ 5.4 小时（vLLM 未启用，见 §5），单步 27-41s |

### 2.2 基座模型

| 项 | 取值 | 说明 |
|---|---|---|
| 模型 | `Qwen/Qwen3-1.7B` | 原版混合思考模型 |
| thinking 模式 | **强制关闭**（`enable_thinking=False`） | 关闭后 completion 预算才不会被思考 token 吃光；训练与评测口径一致 |

> 设计初稿写的是 `Qwen3-1.7B-Instruct-2507`，实测该模型**在任何公开 hub 上都不存在**（2507 refresh 只发布 4B/30B-A3B/235B-A22B）。改用原版 + 强制非思考，是 1.7B 规格下最接近原意图的口径。

### 2.3 数据

| 用途 | 数据集 | 规模 | 说明 |
|---|---|---|---|
| 训练 | NuminaMath（清洗后） | 5,000 题 | 从 859,594 行原始数据清洗出 586,441 行后按 seed=42 抽样 |
| 训练内 held-out | NuminaMath | 500 题 | 监控训练分布内过拟合 |
| 评测 dev | CMath | 600 题 | **所有选点/调参只在这里做** |
| 评测 test | CMath | 1,098 题 | 只评最终选定的模型，各评一次 |

### 2.4 训练配置

| 参数 | 取值 |
|---|---|
| 算法 | GRPO（TRL `GRPOTrainer`） |
| LoRA | r=32，可训参数占比 1.99%（~34M） |
| 每步生成数 `num_generations` | 8 |
| 每卡 batch | 8（= 1 个 prompt 组 × 8 生成） |
| 训练步数 | 500 |
| 学习率 | 1e-5，constant + warmup |
| KL 系数 `beta` | 0.0（不加载 ref model，可验证奖励下省显存） |
| 采样温度 / top_p | 1.0 / 1.0 |
| 最大 prompt / completion | prompt 不截断（新版 TRL 移除该参数）/ **768** |
| 精度 | bf16 |
| 显存优化 | gradient checkpointing + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |

### 2.5 奖励函数

```
reward = 0.2 × format_reward + 1.0 × correctness_reward
```

- `format_reward`：输出是否包含 `\boxed{...}`（0/1）
- `correctness_reward`：`\boxed{}` 内答案与标准答案数值等价（0/1）
- 优势在每组 8 个生成内部做归一化，无需 value model

### 2.6 评测口径

| 项 | 配置 |
|---|---|
| 解码 | **贪心**（temperature=0），`max_new_tokens=768` |
| 判分 | 抽取 `\boxed{}` 内容，数值比对（`judge.py`） |
| 分层 | 按年级（grade 1-6）、推理步数（reasoning_step 1-5）分桶 |
| 产物 | 逐题 CSV（question / golden / pred_boxed / pred_raw_head / correct / output_tail） |

**样本纪律**：dev 上反复评估、选 checkpoint；test 只在选定后评一次，且不因 test 结果回头改选点。

## 3. 结果

### 3.1 dev（600 题，用于选点）

| 模型 | dev acc | 与 base 差 | McNemar p |
|---|---|---|---|
| base（Qwen3-1.7B，非思考） | 0.8583 | — | — |
| GRPO checkpoint-100 | 0.8667 | +0.8pp | — |
| **GRPO checkpoint-200（dev 最优，选定）** | **0.8683** | **+1.0pp（+6 题）** | 0.44 |
| GRPO checkpoint-300 | 0.8633 | +0.5pp | — |
| GRPO final（500 步） | 0.8317 | **−2.7pp（−16 题）** | — |

### 3.2 test（1,098 题，各评一次）

| 模型 | test acc | 与 base 差 | 配对检验 |
|---|---|---|---|
| base | 0.8816 | — | — |
| **GRPO checkpoint-200** | **0.8852** | **+0.36pp（+4 题）** | McNemar：base 错→对 35 题，base 对→错 31 题，**双侧精确 p = 0.71** |

**统计结论：两组预测在 test 上无显著差异（p=0.71）。** 用配对检验而不是比较两个独立比例，是因为同一批题、同一基座，配对信息更有力。

### 3.3 分桶表现（ckpt-200，dev 终评跑打印）

| 维度 | 趋势 |
|---|---|
| by grade（1→6 年级） | 0.90 → 0.71，单调递减，符合难度直觉 |
| by reasoning_step（1→5 步） | 0.92 → 0.63，单调递减 |

分桶结构正常，说明判分链路没有系统性错误、也不是格式 hack 造成的虚高——模型是真的在解题。

## 4. 结论与分析

### 4.1 为什么没有提升

1. **基准饱和**。基座在 CMath dev 上已有 85.8%，剩余可提升空间本就有限；600 题里 1pp 只有 6 道题，模型必须解决"系统性难题"才能体现差距，而 500 步 GRPO 做不到。
2. **训练/评测分布错位**。训练用 NuminaMath（含竞赛题、奖励均值 ~0.5，明显偏难），评测用 CMath（小学 1-6 年级应用题）。在难题上学到的策略迁移不到简单题上，表现为 reward 上升但 dev 不动。
3. **后 200 步轻微负优化**。500 步终点比 base 低 2.7pp，而 checkpoint-200 是峰值——说明出现过早收敛后的漂移（reward 在 0.44-0.56 箱体震荡，优势信号已耗尽，梯度噪声开始主导）。

### 4.2 这是一个有价值的负结果

- 说明"小模型 + 短程 GRPO"不是万灵药，**基准选择和分布对齐比算法本身更决定成败**
- 提供了完整的可复现管线与逐题证据，任何想复现/反驳的人都能直接跑
- 记录的全部工程坑（§6）对后续在云端跑 RL 的人有直接价值

### 4.3 下一步该怎么改

| 方向 | 具体做法 |
|---|---|
| 换更难基准 | MATH-500 / AIME 样本，基座分数 30-60% 区间才有提升空间 |
| 对齐训练分布 | 用 CMath 训练集（或同分布合成题）做 RL，而不是 NuminaMath |
| 拉长训练 | 2000-5000 步 + 中途多次 dev 选点，观察是否越过平台期 |
| 加 SFT 起点 | 先 SFT 再 GRPO（RL 从更强的策略出发），本实验的 SFT 臂未完成 |
| 换更强奖励 | 增加过程奖励（推理步正确性）或 length penalty，缓解格式奖励饱和 |

## 5. 工程记录：vLLM 为什么没用上

镜像自带 vLLM 0.27.1，而当时 TRL 声明只支持 0.10.2-0.12.0，colocate 模式存在 API 断裂风险，**主动放弃 vLLM，走 HF generate 路线**（500 步 5.4 小时，可接受）。若坚持用 vLLM，需先 `pip install "vllm<0.13"` 并验证与镜像 CUDA/PyTorch 的兼容性。

## 6. 踩坑清单（全部已修，patch 在 `lab/patches/`）

| # | 问题 | 现象 | 修复 |
|---|---|---|---|
| 0 | 基座模型名不存在 | `Qwen3-1.7B-Instruct-2507` 下载 404 | 换 `Qwen3-1.7B` + `enable_thinking=False` |
| 1 | 训练脚本连 HuggingFace | 大陆云不可达，加载卡死 | 本地路径优先（`models/qwen3-1.7b` 存在即用），`HF_ENDPOINT` 兜底 |
| 2 | TRL/transformers 版本漂移 | `TrainingArguments`/`GRPOConfig` 参数被移除，TypeError | 加 `_filter_kwargs` 兼容层，按签名过滤不支持的参数并打印 `[compat]` |
| 3 | CMath 字段名假设错误 | dev/test 解析出 0 行 | 答案字段补 `golden`（真实字段名） |
| 4 | SFT 数据契约缺字段 | `KeyError: 'solution'` | 落盘时保留 solution 文本 |
| 5 | 变量定义顺序错误 | `NameError`（LOCAL_MODEL 早于 PROJECT） | 调整定义顺序 |
| 6 | `--resume auto` 解析崩溃 | 目录名 `checkpoint-20_smoke` 触发 `int()` 异常 | 正则只匹配纯数字后缀 |
| 7 | 显存 OOM（fp32 logits） | batch 16 × 768 token 要 6.96GB | 降 batch（16→8）、completion 512→768 权衡；公式：`batch × seq × vocab × 4 byte` |

**显存估算口诀**（Qwen3 15 万词表）：大头永远是 `batch × seq × vocab × 4`。A10 上 512/16 ≈ 9.9GB、768/16 ≈ 6.96GB（配合其他占用会爆）、768/8 ≈ 3.5GB（可用）。经验上真实占用比公式高 25%（缓存与碎片），留够余量再发车。

## 7. 复现

```bash
# 0. 环境（DSW 单卡 A10，PyTorch 镜像）
bash scripts/setup.sh                 # 装依赖、拉模型与数据、构建 jsonl、SFT smoke 自检

# 1. 冒烟（20 步，确认 reward 非 0、不 OOM）
python scripts/train_grpo.py --smoke --per-device-batch 8 --max-completion-len 768

# 2. 正式训练（nohup 防断线）
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup python scripts/train_grpo.py \
  --max-steps 500 --per-device-batch 8 --max-completion-len 768 \
  > logs/grpo_$(date +%m%d_%H%M).log 2>&1 &

# 3. dev 选点（base + 各 checkpoint）
python scripts/eval_cmath.py --model models/qwen3-1.7b --split dev --tag base
for ck in out/grpo/checkpoint-100 out/grpo/checkpoint-200 out/grpo/checkpoint-300; do
  python scripts/eval_cmath.py --model models/qwen3-1.7b --adapter $ck --split dev --tag dev_$(basename $ck)
done

# 4. test 终评（只评选定点，各一次）
python scripts/eval_cmath.py --model models/qwen3-1.7b --split test --tag test_base
python scripts/eval_cmath.py --model models/qwen3-1.7b --adapter out/grpo/checkpoint-200 --split test --tag test_grpo_ckpt200
```

## 8. 仓库内容

```
scripts/                     训练/评测/数据构建脚本（含全部修复）
grpo_results/results/        逐题评测 CSV（dev 5 个点 + test 2 个点）
grpo_results/logs/           训练与评测日志（reward 曲线原始证据）
lab/patches/                 7 个修复的 unified diff（可 patch -R 回滚）
lab/backup/                  修复前原文件备份
```

**未入库**：LoRA 权重（`adapter_model.safetensors` 134MB，超出 GitHub 单文件 100MB 限制）、基座模型、训练数据。权重可用上面命令重新训练复现，或另行分发。

## 9. 环境与复现声明

- 所有数字来自本仓库 `grpo_results/results/*.csv`，可用 CSV 重新计算
- 训练日志为原始 nohup 输出，未做编辑
- 评测使用贪心解码，结果确定性可复现（同环境同版本下）
