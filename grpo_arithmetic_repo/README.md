# GRPO Arithmetic — 8GB 显存上的 Qwen2-0.5B 可验证奖励强化学习

用 GRPO（Group Relative Policy Optimization）在单张消费级 8GB 显卡上对 `Qwen2-0.5B-Instruct`
做强化学习训练。奖励全部来自**可程序化验证的规则**（无需奖励模型、无需人工标注），
目标是让模型同时学会两件事：按 `<answer>` 格式输出，以及把算术题算对。

项目包含完整闭环：难度探测 → GRPO 训练（全量 / LoRA 双方案）→ 同评估集复测 → 配对对比与曲线分析。

## 结果一览

评估集固定为 6 个难度 × 50 题，`seed=42`，每题 1 次 greedy 解码 + 8 次温度采样；
基线与训练后使用完全相同的题目，可逐题配对比较。

| 难度 | 是否进训练集 | 格式率 基线→全量→LoRA | greedy 正确率 基线→全量→LoRA | pass@8 基线→全量→LoRA |
|------|:---:|---|---|---|
| L1 个位数加法 | — | 0.00 → 0.96 → 1.00 | 0.98 → 1.00 → 1.00 | 1.00 → 1.00 → 1.00 |
| L2 两位数加减 | √ | 0.02 → 1.00 → 1.00 | 0.76 → 0.98 → 1.00 | 0.90 → 1.00 → 1.00 |
| L3 三位数加减 | √ | 0.06 → 0.96 → 1.00 | 0.42 → 0.88 → 0.90 | 0.84 → 0.98 → 0.96 |
| L4 表内乘法 | — | 0.00 → 0.84 → 1.00 | 0.62 → 1.00 → 1.00 | 0.96 → 1.00 → 1.00 |
| L5 两位×一位 | √ | 0.00 → 0.98 → 1.00 | 0.20 → 0.88 → 0.96 | 0.62 → 0.94 → 0.98 |
| L6 两位×两位 | — | 0.08 → 0.98 → 1.00 | 0.08 → 0.22 → 0.24 | 0.18 → 0.42 → 0.38 |

训练成本（200 优化步，每步 4 个 prompt × 8 条采样）：

| 方案 | 训练耗时 | GPU 峰值显存 | 可训练部分 | 学习率 |
|------|:---:|:---:|:---:|:---:|
| 全量微调 | 617.9 s | 5.46 GB | 全部参数 | 2e-6 |
| LoRA（r=16, q/k/v/o_proj） | 607.2 s | 3.36 GB | 仅 adapter | 2e-4 |

三条可验证的结论：

1. **格式行为可以零成本学会**：所有难度（含未训练的 L1/L4/L6）格式率从 0~0.08 提升到 0.84~1.00。
2. **能力提升发生在模型能力边界内**：训练集内的 L5 从 0.20 → 0.88（4.4×），
   未训练的 L4 从 0.62 → 1.00（泛化），而超出边界的两两位数乘法 L6 仅 0.08 → 0.22。
3. **LoRA 在本场景性价比更高**：显存降低 38%（3.36 vs 5.46 GB）、耗时相当，
   正确率与格式率均略优于全量；代价是策略熵更低（0.011 vs 0.081），探索更早枯竭。

## 目录结构

```
grpo_arithmetic_repo/
├── src/
│   ├── probe_baseline.py       # 基线探测 / 训练后评估（同一脚本，--model 切换）
│   ├── train_grpo.py           # GRPO 训练（TRL GRPOTrainer + 复合规则奖励）
│   ├── compare_results.py      # 三方对比表 + 样例对照 + 训练曲线
│   ├── test_general_dialog.py  # 通用对话能力抽查（检查灾难性遗忘）
│   └── trl_compat.py           # trl 0.21 × transformers 5.x 兼容补丁（必须先导入）
├── outputs/                    # 真实运行产物（指标 JSON + 训练曲线）
├── logs/                       # 4 次运行的完整命令行输出（仅隐去本机绝对路径）
├── requirements.txt
├── README.md
├── ARCHITECTURE.md             # 架构设计、参数选型、工程问题定位过程
└── USAGE.md                    # 环境准备、四步流程、接口与 FAQ
```

## 快速开始

```bash
pip install -r requirements.txt
# 将 Qwen2-0.5B-Instruct 放到仓库上三级目录的 pretrain_models/ 下
#（或修改 src/probe_baseline.py 与 src/train_grpo.py 顶部的 MODEL_PATH）

python src/probe_baseline.py                    # 1. 基线探测（6 难度 × 50 题，约 70 秒）
python src/train_grpo.py                        # 2. GRPO 全量训练 200 步（约 10 min）
python src/probe_baseline.py --model outputs/grpo_ckpt \
       --out outputs/post_train_probe.json --seed 42    # 3. 同评估集复测
python src/compare_results.py                   # 4. 对比表 + 训练曲线
```

显存不足时把第 2 步换成 `python src/train_grpo.py --lora`，第 3 步改用
`--model outputs/grpo_lora_ckpt --out outputs/post_train_probe_lora.json`
（脚本会自动识别 adapter 并挂载到基座）。

## 数据口径

- 所有指标来自 `outputs/*.json` 与 `logs/*.log` 的真实运行记录，未做估算或手工修饰。
- 评估题的 greedy 准确率使用宽松口径（输出中最后一个数字正确即算对）；
  `pass@8` 为 8 条采样中至少 1 条正确。
- 训练/评估均为单次运行结果，`temperature=1.0` 采样存在随机性，重复运行会有数个百分点波动，
  结论看趋势而非个位数。

详细设计与实现说明见 [ARCHITECTURE.md](ARCHITECTURE.md)，运行与调试见 [USAGE.md](USAGE.md)。
