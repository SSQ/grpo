# USAGE

## 1. 环境准备

```bash
pip install -r requirements.txt
```

| 依赖 | 版本 | 用途 |
|------|------|------|
| `transformers` | 5.5.3 | 模型与 tokenizer 加载（与 trl 0.21 的兼容问题由 `src/trl_compat.py` 处理） |
| `trl` | 0.21.0 | `GRPOTrainer` |
| `peft` | 0.15.0 | LoRA（`--lora` 与 adapter 加载） |
| `accelerate` | 1.5.2 | Trainer 后端 |
| `datasets` | — | 训练集封装 |
| `matplotlib` | — | 训练曲线 |

另外需要安装与本机 CUDA 匹配的 PyTorch（`requirements.txt` 未锁定 torch 版本）。

**基座模型**：`Qwen2-0.5B-Instruct`。默认路径为仓库上三级目录下的
`pretrain_models/Qwen2-0.5B-Instruct`，可在 `src/probe_baseline.py`、
`src/train_grpo.py`、`src/test_general_dialog.py` 顶部的 `MODEL_PATH` 修改。

**硬件**：在 RTX 2070（8GB）上验证。全量微调峰值 5.46 GB，LoRA 峰值 3.36 GB；
显存更小的机器直接使用 `--lora`。

## 2. 四步流程

### Step 1 基线探测

```bash
python src/probe_baseline.py              # 6 难度 × 50 题，K=8
python src/probe_baseline.py --quick      # 每难度 10 题，快速验证
```

每个难度执行两种解码：greedy 1 条（确定性能力 + 格式遵循），温度 1.0 采样 8 条
（pass@8 与组内方差）。输出：

- `greedy_format_rate`：输出 `<answer>N</answer>` 的比例
- `greedy_loose_acc` / `greedy_strict_acc`：宽松 / 严格口径的 greedy 正确率
- `pass@8` / `loose_pass@8`：8 条采样中至少 1 条正确
- `informative_group_rate`：组内有对有错（0 < 正确数 < 8）的 prompt 比例，
  决定该难度能否被 GRPO 有效学习

实测参考：L3 的 loose informative rate 为 0.80，L5 为 0.62，L6 仅 0.18；
全部难度的格式率在 0.00~0.08。结果写入 `outputs/baseline_probe.json`，
六个难度累计耗时约 70 秒。

### Step 2 GRPO 训练

```bash
python src/train_grpo.py                          # 全量微调，200 步
python src/train_grpo.py --max_steps 3 --tag smoke # 冒烟测试
python src/train_grpo.py --lora                   # LoRA（r=16，q/k/v/o_proj）
python src/train_grpo.py --log_completions        # 打印每步真实采样，便于调试
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `--max_steps` | 200 | 优化步数；每步 4 个 prompt × 8 条采样 |
| `--n_prompts` | 1000 | 训练集条数（L3 50% / L5 25% / L2 25%） |
| `--lr` | 2e-6 | 全量学习率；`--lora` 时自动使用 2e-4 |
| `--lora` | 关 | 切换 LoRA |
| `--tag` | 空 | 输出目录与日志文件名后缀，用于区分实验 |

每 5 步打印一次指标：两个奖励分量、`frac_reward_zero_std`、策略熵、clip 比例、
补全长度。运行结束输出 checkpoint、训练日志与 GPU 峰值显存：

- `outputs/grpo_ckpt/`（或 `outputs/grpo_lora_ckpt/`）
- `outputs/train_log.json`（或 `train_log_lora.json`）

实测全量 200 步耗时 617.9 s、峰值 5.46 GB；LoRA 200 步耗时 607.2 s、峰值 3.36 GB。
健康标志：前 20 步 `rewards/reward_correct/mean` 从 0.475 升到 0.88 附近。

### Step 3 训练后评估

```bash
# 全量 checkpoint
python src/probe_baseline.py --model outputs/grpo_ckpt \
       --out outputs/post_train_probe.json --seed 42

# LoRA checkpoint（脚本检测到 adapter_config.json 后自动加载基座并挂载 adapter）
python src/probe_baseline.py --model outputs/grpo_lora_ckpt \
       --out outputs/post_train_probe_lora.json --seed 42
```

`--seed 42` 必须与基线一致，否则评估题目不同，前后不可配对比较。
基线与两次复测的 GPU 峰值显存分别为 1.54 GB / 1.49 GB / 1.49 GB。

### Step 4 对比分析

```bash
python src/compare_results.py
```

输出三方对比表（基线 / 全量 / LoRA 的格式率、greedy 正确率、pass@8）、
训练集内难度的逐题样例对照，并生成 `outputs/figures/train_curves.png`
（三面板：奖励分量 / 退化组比例 / 策略熵，全量与 LoRA 叠加）。
若缺少 LoRA 的 probe 或 log 文件，自动退化为两方对比。

## 3. 作为模块使用

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ckpt = "outputs/grpo_ckpt"   # 换成基座路径即可对比训练前行为
tokenizer = AutoTokenizer.from_pretrained(ckpt)
model = AutoModelForCausalLM.from_pretrained(ckpt, dtype=torch.bfloat16, device_map="cuda")

msgs = [
    {"role": "system", "content": "你是一个算术助手。用户会给你一道算术题，请计算出结果，并把最终答案放在 <answer> 标签中，例如 <answer>42</answer>。不要输出其他内容。"},
    {"role": "user", "content": "计算：47 + 38 = ?"},
]
text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
enc = tokenizer(text, return_tensors="pt").to("cuda")
out = model.generate(**enc, max_new_tokens=32, do_sample=False, pad_token_id=tokenizer.pad_token_id)
print(tokenizer.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True))
# 基座模型: '85'  （无格式）
# 训练后:   '<answer>85</answer>'
```

修改奖励函数做对照实验，例如把格式分从 0.2 提到 1.0：

```python
def reward_format(completions, **kwargs):
    return [1.0 if parse_output(comp[0]["content"], 0)[0] else 0.0 for comp in completions]
```

## 4. 常见问题

**Q1：`from trl import GRPOTrainer` 报 `No module named 'vllm'`？**
trl 0.21 与 transformers 5.x 的已知不兼容（`_is_package_available` 返回值类型变化）。
`src/train_grpo.py` 在导入 trl 之前先 `import trl_compat`，确保从仓库根目录运行，不要删除该文件。

**Q2：训练奖励恒为 0、补全全是乱码？**
确认 `gradient_checkpointing` 为关闭状态。transformers 5.x 下该选项会使 `generate`
输出损坏。显存不足请改用 `--lora`，不要重新打开 checkpointing。

**Q3：训练一步后模型输出全变成 `!!!!!`？**
权重被 fp16 训废了。确认 `train_grpo.py` 中存在
`model_init_kwargs={"torch_dtype": "bfloat16"}`。基座 `config.json` 标注 fp16，
不显式指定会按 fp16 加载，AdamW 的 `eps=1e-8` 在 fp16 下下溢为 0。

**Q4：CUDA OOM？**
依次尝试：`--lora`；把 `per_device_train_batch_size` 8 → 4，同时把
`gradient_accumulation_steps` 4 → 8，保持每步 prompt 数不变。

**Q5：想换任务或题型？**
改三处：`probe_baseline.py` 的 `make_problem`（题目生成）、`train_grpo.py` 的
`LEVEL_MIX`（难度配比）与两个 reward function（奖励判定）。
先跑 probe 确认 informative group rate 落在 0.3~0.8 再开始训练。

**Q6：`epoch` 显示 0.8，是否没跑完一轮？**
正常。训练集 1000 条，200 步 × 4 条/步 = 800 条，不足一个 epoch。
GRPO 是在线采样算法，同一题目每次都会重新采样 8 条，题目是否重复不影响训练。

**Q7：想检查是否发生灾难性遗忘？**
运行 `python src/test_general_dialog.py`，脚本会依次加载基座、全量、LoRA 三个模型，
在知识问答、常识推理、翻译、创意写作、指令跟随、逻辑推理、数学解释、拒绝能力
8 类场景输出对照。
