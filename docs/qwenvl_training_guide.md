# QwenVL 模型训练指南

本文档介绍如何使用 Qwen2.5-VL 和 Qwen3-VL 系列模型（2B、3B、4B、8B）在 InternNav 框架下训练 InternVLA-N1 System-2。

## 目录

- [QwenVL 模型训练指南](#qwenvl-模型训练指南)
  - [目录](#目录)
  - [模型规格概览](#模型规格概览)
  - [环境依赖](#环境依赖)
  - [训练模式说明](#训练模式说明)
  - [快速开始：本地多 GPU 训练](#快速开始本地多-gpu-训练)
    - [使用默认 GPU（GPU 6、7）](#使用默认-gpugpu-67)
    - [指定 GPU](#指定-gpu)
    - [修改模型大小](#修改模型大小)
  - [集群训练（SLURM）](#集群训练slurm)
    - [Qwen2.5-VL（System-2 训练）](#qwen25-vlsystem-2-训练)
    - [Qwen3-VL（System-2 训练）](#qwen3-vlsystem-2-训练)
  - [按模型大小配置](#按模型大小配置)
    - [Qwen3-VL-2B](#qwen3-vl-2b)
    - [Qwen2.5-VL-3B（约 3B 级别）](#qwen25-vl-3b约-3b-级别)
    - [Qwen3-VL-4B](#qwen3-vl-4b)
    - [Qwen3-VL-8B / Qwen2.5-VL-7B](#qwen3-vl-8b--qwen25-vl-7b)
      - [Qwen3-VL-8B（使用现有脚本，默认配置）](#qwen3-vl-8b使用现有脚本默认配置)
      - [Qwen2.5-VL-7B（使用现有脚本，默认配置）](#qwen25-vl-7b使用现有脚本默认配置)
  - [双系统联合训练](#双系统联合训练)
    - [前提条件](#前提条件)
    - [启动双系统训练](#启动双系统训练)
  - [训练参数详解](#训练参数详解)
    - [模型参数（ModelArguments）](#模型参数modelarguments)
    - [数据参数（DataArguments）](#数据参数dataarguments)
    - [训练参数（TrainingArguments）](#训练参数trainingarguments)
    - [数据集标识符](#数据集标识符)
  - [DeepSpeed 配置选择](#deepspeed-配置选择)
  - [显存需求参考](#显存需求参考)
  - [常见问题](#常见问题)
    - [Q: Qwen3-VL 训练时报错 `data_flatten` 相关错误？](#q-qwen3-vl-训练时报错-data_flatten-相关错误)
    - [Q: 如何降低显存占用？](#q-如何降低显存占用)
    - [Q: 训练后如何评估？](#q-训练后如何评估)
    - [Q: 如何使用自定义本地模型路径？](#q-如何使用自定义本地模型路径)
    - [Q: Wandb 报告失败怎么办？](#q-wandb-报告失败怎么办)
    - [Q: 训练中断如何续训？](#q-训练中断如何续训)
  - [相关文档](#相关文档)

---

## 模型规格概览

InternNav 支持以下 QwenVL 模型作为 System-2 语言推理主干：

| 系列 | 模型 | 参数量 | 推荐用途 |
|------|------|--------|---------|
| Qwen3-VL | `Qwen/Qwen3-VL-2B-Instruct` | ~2B | 资源极限受限、边缘设备、快速实验 |
| Qwen2.5-VL | `Qwen/Qwen2.5-VL-3B-Instruct` | ~3B | 资源受限环境、快速实验 |
| Qwen2.5-VL | `Qwen/Qwen2.5-VL-7B-Instruct` | ~7B | 标准训练（推荐） |
| Qwen3-VL | `Qwen/Qwen3-VL-4B-Instruct` | ~4B | 轻量级 Qwen3 架构 |
| Qwen3-VL | `Qwen/Qwen3-VL-8B-Instruct` | ~8B | 高性能 Qwen3 架构 |
| Qwen3-VL | `Qwen/Qwen3-VL-32B-Instruct` | ~32B | 超大规模实验 |

> **注意：** Qwen3-VL MoE 变体（如 30B-A3B、235B-A22B）暂不支持，需要额外的 `Qwen3VLMoeForConditionalGeneration` 类。

模型类自动检测逻辑（`internvla_n1_trainer.py`）：

```
model_name_or_path 包含 "internvla-n1-system2"  →  InternVLAN1ForCausalLM（Qwen2.5-VL 主干）
model_name_or_path 包含 "qwen3"                 →  Qwen3VLForConditionalGeneration（System-2 模式）
                                                    InternVLAN1ForCausalLMQwen3（双系统模式）
model_name_or_path 包含 "qwen2.5"               →  Qwen2_5_VLForConditionalGeneration
其他                                             →  Qwen2VLForConditionalGeneration
```

---

## 环境依赖

```bash
# 安装基础依赖
pip install -e ".[internvla_n1]"

# Qwen2.5-VL 要求
pip install transformers>=4.51.0

# Qwen3-VL 额外要求（版本更高）
pip install transformers>=4.57.0

# 推荐安装 flash-attention 以加速训练
pip install flash-attn --no-build-isolation

# DeepSpeed（分布式训练必须）
pip install deepspeed
```

初始化子模块：

```bash
git submodule update --init --recursive
```

---

## 训练模式说明

InternVLA-N1 有两种训练阶段：

| 阶段 | 脚本 | 说明 |
|------|------|------|
| **Stage 1：仅训练 System-2** | `train_system2.sh` / `train_system2_qwen3vl.sh` | 微调视觉-语言模型，`system1="none"` |
| **Stage 2：双系统联合训练** | `train_dual_system.sh` | 固定 System-2，训练 System-1 扩散策略头 |

推荐顺序：先完成 Stage 1，再用 Stage 1 的 checkpoint 进行 Stage 2。

---

## 快速开始：本地多 GPU 训练

适用于单机多卡（无 SLURM 调度器）的环境。

### 使用默认 GPU（GPU 6、7）

```bash
bash scripts/train/qwenvl_train/train_system2_qcyl.sh
```

### 指定 GPU

```bash
# 使用 GPU 0、1
CUDA_VISIBLE_DEVICES=0,1 bash scripts/train/qwenvl_train/train_system2_qcyl.sh

# 使用全部 8 张 GPU
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/train/qwenvl_train/train_system2_qcyl.sh
```

### 修改模型大小

打开 `scripts/train/qwenvl_train/train_system2_qcyl.sh`，修改 `llm` 变量：

```bash
# 改为 3B 模型（约 2B 量级）
llm=Qwen/Qwen2.5-VL-3B-Instruct

# 改为 7B 模型（默认）
llm=Qwen/Qwen2.5-VL-7B-Instruct
```

---

## 集群训练（SLURM）

适用于有 SLURM 作业调度系统的 HPC 集群。

### Qwen2.5-VL（System-2 训练）

```bash
sbatch scripts/train/qwenvl_train/train_system2.sh
```

默认配置：8 节点 × 8 GPU = 64 GPU，模型为 `Qwen2.5-VL-7B-Instruct`。

### Qwen3-VL（System-2 训练）

```bash
sbatch scripts/train/qwenvl_train/train_system2_qwen3vl.sh
```

默认模型为 `Qwen3-VL-8B-Instruct`，修改为其他大小见[下节](#按模型大小配置)。

---

## 按模型大小配置

### Qwen3-VL-2B

**推荐场景：** 显存极度受限（单卡 24GB）、边缘部署验证、消融实验

> **重要：** Qwen3-VL 必须设置 `data_flatten False`，不支持 flash attention 序列打平优化。

本地单卡或 2 卡启动：

```bash
export CUDA_VISIBLE_DEVICES=0         # 单卡
# export CUDA_VISIBLE_DEVICES=0,1    # 双卡
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

torchrun --standalone --nproc_per_node=${NUM_GPUS} \
    internnav/trainer/internvla_n1_trainer.py \
    --deepspeed scripts/train/qwenvl_train/zero2.json \
    --model_name_or_path Qwen/Qwen3-VL-2B-Instruct \
    --vln_dataset_use r2r_125cm_0_30,r2r_125cm_0_45,r2r_60cm_15_15,r2r_60cm_30_30,rxr_125cm_0_30,rxr_125cm_0_45,rxr_60cm_15_15,rxr_60cm_30_30 \
    --data_flatten False \
    --tune_mm_vision True \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --num_history 8 \
    --data_augmentation True \
    --resize_h 384 --resize_w 384 \
    --sample_step 4 \
    --num_future_steps 4 \
    --predict_step_num 32 \
    --pixel_goal_only False \
    --system1 "none" \
    --output_dir checkpoints/InternVLA-N1-System2-Qwen3VL-2B \
    --num_train_epochs 2.0 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --max_pixels 313600 --min_pixels 3136 \
    --learning_rate 2e-5 \
    --vision_tower_lr 5e-6 \
    --weight_decay 0 \
    --warmup_ratio 0.003 \
    --max_grad_norm 1 \
    --lr_scheduler_type cosine \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --save_strategy steps \
    --save_steps 5000 \
    --save_total_limit 5 \
    --logging_steps 1 \
    --run_name InternVLA-N1-System2-Qwen3VL-2B \
    --report_to wandb
```

SLURM 集群训练，在 `train_system2_qwen3vl.sh` 中修改：

```bash
llm=Qwen/Qwen3-VL-2B-Instruct
run_name=InternVLA-N1-System2-Qwen3VL-2B
output_dir=checkpoints/${run_name}
batch_size=8          # 2B 模型显存占用最小，可大幅增大 batch
```

---

### Qwen2.5-VL-3B（约 3B 级别）

**推荐场景：** 资源有限、快速原型验证、单机 2-4 卡

修改 `train_system2_qcyl.sh`：

```bash
llm=Qwen/Qwen2.5-VL-3B-Instruct
lr=2e-5
vision_tower_lr=5e-6
batch_size=4          # 3B 显存占用更小，可适当增大 batch
grad_accum_steps=1
```

启动命令：

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/train/qwenvl_train/train_system2_qcyl.sh
```

或直接运行：

```bash
export CUDA_VISIBLE_DEVICES=0,1
NUM_GPUS=2

torchrun --standalone --nproc_per_node=${NUM_GPUS} \
    internnav/trainer/internvla_n1_trainer.py \
    --deepspeed scripts/train/qwenvl_train/zero2.json \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --vln_dataset_use r2r_125cm_0_30,r2r_125cm_0_45,r2r_60cm_15_15,r2r_60cm_30_30 \
    --data_flatten False \
    --tune_mm_vision True \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --num_history 8 \
    --data_augmentation True \
    --resize_h 384 --resize_w 384 \
    --sample_step 4 \
    --num_future_steps 4 \
    --predict_step_num 32 \
    --pixel_goal_only False \
    --system1 "none" \
    --output_dir checkpoints/InternVLA-N1-System2-3B \
    --num_train_epochs 2.0 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --max_pixels 313600 --min_pixels 3136 \
    --learning_rate 2e-5 \
    --vision_tower_lr 5e-6 \
    --weight_decay 0 \
    --warmup_ratio 0.003 \
    --max_grad_norm 1 \
    --lr_scheduler_type cosine \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --save_strategy steps \
    --save_steps 5000 \
    --save_total_limit 5 \
    --logging_steps 1 \
    --run_name InternVLA-N1-System2-3B \
    --report_to wandb
```

---

### Qwen3-VL-4B

**推荐场景：** 中等资源、需要 Qwen3 架构能力、单机 4 卡

> **重要：** Qwen3-VL 必须设置 `data_flatten False`，不支持 flash attention 序列打平优化。

修改 `train_system2_qwen3vl.sh`：

```bash
llm=Qwen/Qwen3-VL-4B-Instruct
run_name=InternVLA-N1-System2-Qwen3VL-4B
output_dir=checkpoints/${run_name}
batch_size=4          # 4B 显存占用中等，可适当增大
```

SLURM 提交：

```bash
sbatch scripts/train/qwenvl_train/train_system2_qwen3vl.sh
```

本地 4 卡启动：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_GPUS=4

torchrun --standalone --nproc_per_node=${NUM_GPUS} \
    internnav/trainer/internvla_n1_trainer.py \
    --deepspeed scripts/train/qwenvl_train/zero2.json \
    --model_name_or_path Qwen/Qwen3-VL-4B-Instruct \
    --vln_dataset_use r2r_125cm_0_30,r2r_125cm_0_45,r2r_60cm_15_15,r2r_60cm_30_30,rxr_125cm_0_30,rxr_125cm_0_45,rxr_60cm_15_15,rxr_60cm_30_30 \
    --data_flatten False \
    --tune_mm_vision True \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --num_history 8 \
    --data_augmentation True \
    --resize_h 384 --resize_w 384 \
    --sample_step 4 \
    --num_future_steps 4 \
    --predict_step_num 32 \
    --pixel_goal_only False \
    --system1 "none" \
    --output_dir checkpoints/InternVLA-N1-System2-Qwen3VL-4B \
    --num_train_epochs 2.0 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --max_pixels 313600 --min_pixels 3136 \
    --learning_rate 2e-5 \
    --vision_tower_lr 5e-6 \
    --weight_decay 0 \
    --warmup_ratio 0.003 \
    --max_grad_norm 1 \
    --lr_scheduler_type cosine \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --save_strategy steps \
    --save_steps 5000 \
    --save_total_limit 5 \
    --logging_steps 1 \
    --run_name InternVLA-N1-System2-Qwen3VL-4B \
    --report_to wandb
```

---

### Qwen3-VL-8B / Qwen2.5-VL-7B

**推荐场景：** 主力训练配置，兼顾性能和资源消耗，8 卡或以上

#### Qwen3-VL-8B（使用现有脚本，默认配置）

```bash
# SLURM 集群
sbatch scripts/train/qwenvl_train/train_system2_qwen3vl.sh

# 本地 8 卡
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/train/qwenvl_train/train_system2_qcyl.sh
# 注意：需将 train_system2_qcyl.sh 中 llm 改为 Qwen/Qwen3-VL-8B-Instruct
```

#### Qwen2.5-VL-7B（使用现有脚本，默认配置）

```bash
# SLURM 集群
sbatch scripts/train/qwenvl_train/train_system2.sh

# 本地 8 卡
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/train/qwenvl_train/train_system2_qcyl.sh
```

两者的超参数配置相同：

| 参数 | 值 |
|------|-----|
| learning_rate | 2e-5 |
| vision_tower_lr | 5e-6 |
| batch_size (per GPU) | 2 |
| gradient_accumulation_steps | 1 |
| num_train_epochs | 2.0 |
| max_pixels | 313600 |
| min_pixels | 3136 |
| DeepSpeed | zero2 |

---

## 双系统联合训练

双系统训练在 System-2 checkpoint 基础上，联合训练 System-1 扩散策略头（NavDP 或 NextDiT）。**必须先完成 System-2 训练。**

### 前提条件

- 已有 System-2 checkpoint（如 `checkpoints/InternVLA-N1-System2`）
- System-1 类型选择：
  - `nextdit_async`：NextDiT 扩散轨迹模型（推荐）
  - `navdp_async`：NavDP 扩散策略（速度更快）
  - `nextdit`：NextDiT 同步版本

### 启动双系统训练

```bash
sbatch scripts/train/qwenvl_train/train_dual_system.sh
```

关键配置差异：

```bash
# 冻结 System-2 的所有参数
--tune_mm_vision False
--tune_mm_mlp False
--tune_mm_llm False

# 指定 System-1 类型
--system1 nextdit_async

# 只使用像素级目标
--pixel_goal_only True

# 更高学习率（只训练 System-1 头）
--learning_rate 1e-4

# 更多训练轮次
--num_train_epochs 3.0
```

---

## 训练参数详解

### 模型参数（ModelArguments）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_name_or_path` | `Qwen/Qwen2.5-VL-3B-Instruct` | 模型路径或 HuggingFace ID |
| `--tune_mm_vision` | False | 是否训练视觉编码器 |
| `--tune_mm_mlp` | False | 是否训练视觉-语言投影层（merger） |
| `--tune_mm_llm` | False | 是否训练 LLM 主干 |
| `--system1` | `nextdit` | System-1 类型：`none` / `nextdit` / `navdp` / `nextdit_async` / `navdp_async` |
| `--n_query` | 4 | 潜变量查询 token 数量 |

### 数据参数（DataArguments）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--vln_dataset_use` | `""` | 使用的 VLN 数据集，逗号分隔 |
| `--data_flatten` | False | 序列打平（**Qwen3-VL 必须为 False**） |
| `--data_augmentation` | False | 启用颜色抖动、随机海报化等数据增强 |
| `--num_history` | 8 | 历史观测帧数 |
| `--predict_step_num` | 32 | 预测轨迹步数 |
| `--resize_h` / `--resize_w` | 384 | 图像输入分辨率 |
| `--max_pixels` | 313600 | 最大图像像素数（动态分辨率上限） |
| `--min_pixels` | 3136 | 最小图像像素数（动态分辨率下限） |
| `--sample_step` | 4 | 轨迹采样间隔步长 |
| `--num_future_steps` | 4 | 预测未来步数 |
| `--pixel_goal_only` | False | 是否仅使用像素目标（双系统训练时为 True） |

### 训练参数（TrainingArguments）

| 参数 | System-2 默认 | 双系统默认 | 说明 |
|------|--------------|-----------|------|
| `--learning_rate` | 2e-5 | 1e-4 | 主干学习率 |
| `--vision_tower_lr` | 5e-6 | N/A | 视觉编码器学习率 |
| `--num_train_epochs` | 2.0 | 3.0 | 训练轮次 |
| `--per_device_train_batch_size` | 2 | 2 | 每卡批大小 |
| `--gradient_accumulation_steps` | 1 | 1 | 梯度累积步数 |
| `--warmup_ratio` | 0.003 | 0.003 | 预热比例 |
| `--lr_scheduler_type` | cosine | cosine_with_min_lr | 学习率调度器 |
| `--model_max_length` | 8192 | 8192 | 最大序列长度 |
| `--gradient_checkpointing` | True | True | 梯度检查点（节省显存） |
| `--save_steps` | 5000 | 5000 | 检查点保存间隔 |
| `--save_total_limit` | 5 | 5 | 保留检查点数量 |

### 数据集标识符

数据集使用逗号分隔的标识符，格式为 `{数据集}_{相机高度}_{俯仰角}_{偏转角}`：

```
r2r_125cm_0_30       # R2R 数据集，相机高度125cm，无俯仰，偏转30°
r2r_125cm_0_45       # R2R 数据集，偏转45°
r2r_60cm_15_15       # R2R 数据集，相机高度60cm，俯仰15°，偏转15°
r2r_60cm_30_30       # R2R 数据集，俯仰30°，偏转30°
rxr_125cm_0_30       # RxR 多语言数据集（同上格式）
rxr_125cm_0_45
rxr_60cm_15_15
rxr_60cm_30_30
scalevln_125cm_0_30  # ScaleVLN 大规模数据集
scalevln_60cm_30_30
```

双系统训练支持按比例采样（`%30` 表示采样30%）：

```bash
vln_datasets=r2r_125cm_0_30%30,scalevln_125cm_0_30%30
```

---

## DeepSpeed 配置选择

配置文件位于 `scripts/train/qwenvl_train/`：

| 文件 | 阶段 | 适用场景 |
|------|------|---------|
| `zero2.json` | ZeRO-2 | 默认推荐，平衡速度与显存 |
| `zero3.json` | ZeRO-3 | 显存紧张时（如 7B/8B 单节点训练） |
| `zero3_offload.json` | ZeRO-3 + CPU Offload | 极度显存受限（如 32B 模型） |

修改脚本中的 `deepspeed` 变量切换配置：

```bash
# 切换到 ZeRO-3
deepspeed=scripts/train/qwenvl_train/zero3.json

# 切换到 ZeRO-3 + CPU 卸载
deepspeed=scripts/train/qwenvl_train/zero3_offload.json
```

---

## 显存需求参考

以下为 A100 80GB 或 H100 80GB 单卡估算（bf16，batch_size=2，384×384 输入）：

| 模型 | ZeRO-2 所需 GPU 数 | ZeRO-3 所需 GPU 数 | 备注 |
|------|-------------------|-------------------|------|
| Qwen3-VL-2B | 1 张 80GB（或 2 张 24GB） | 1 张 24GB | 资源要求最低 |
| Qwen2.5-VL-3B | 2 张 80GB | 1 张 80GB | |
| Qwen3-VL-4B | 2 张 80GB | 1 张 80GB | |
| Qwen2.5-VL-7B | 4 张 80GB | 2 张 80GB | 官方推荐配置 |
| Qwen3-VL-8B | 4 张 80GB | 2 张 80GB | 官方推荐配置 |
| Qwen3-VL-32B | 8 张 80GB | 4 张 80GB（+CPU Offload） | 需要 zero3_offload.json |

> **提示：** 使用 `--gradient_checkpointing True` 可降低约 30-40% 的激活值显存占用，但会轻微降低训练速度。

---

## 常见问题

### Q: Qwen3-VL 训练时报错 `data_flatten` 相关错误？

Qwen3-VL 的 flash attention 补丁尚未适配，必须禁用序列打平：

```bash
--data_flatten False
```

### Q: 如何降低显存占用？

按优先级尝试：
1. 启用梯度检查点：`--gradient_checkpointing True`（默认已启用）
2. 减小 batch size：`--per_device_train_batch_size 1`，增大 `--gradient_accumulation_steps`
3. 切换到 ZeRO-3：`--deepspeed scripts/train/qwenvl_train/zero3.json`
4. 切换到 ZeRO-3 + CPU Offload：`--deepspeed scripts/train/qwenvl_train/zero3_offload.json`
5. 降低图像分辨率：`--resize_h 256 --resize_w 256`，同时相应降低 `--max_pixels`

### Q: 训练后如何评估？

System-2 评估使用 `scripts/eval/` 下的配置：

```bash
# Qwen2.5-VL System-2 评估
python scripts/eval/eval.py --config scripts/eval/configs/habitat_s2_qwenvl_cfg.py

# Qwen3-VL System-2 评估
python scripts/eval/eval.py --config scripts/eval/configs/habitat_s2_qwen3vl_cfg.py
```

评估指标：SR（成功率）、SPL（路径加权成功率）、NE（导航误差）、OS（轨迹覆盖率）。

### Q: 如何使用自定义本地模型路径？

将 `--model_name_or_path` 设置为本地路径：

```bash
--model_name_or_path /path/to/your/Qwen3-VL-8B-Instruct
```

**注意：** 路径中必须包含对应的模型系列关键词（`qwen3` / `qwen2.5`）以确保自动检测正确加载模型类。若使用自定义名称，请确保路径包含相应字符串，或手动修改 `internvla_n1_trainer.py` 中的检测逻辑。

### Q: Wandb 报告失败怎么办？

可以禁用 Wandb 改用 TensorBoard：

```bash
--report_to tensorboard
```

或完全禁用：

```bash
--report_to none
```

### Q: 训练中断如何续训？

指定最后一个检查点路径：

```bash
torchrun ... internnav/trainer/internvla_n1_trainer.py \
    --model_name_or_path checkpoints/InternVLA-N1-System2/checkpoint-10000 \
    ...
```

---

## 相关文档

- [Qwen3-VL 支持指南](qwen3vl_support_guide.md) — Qwen3-VL 架构差异与迁移说明
- [CLAUDE.md](../CLAUDE.md) — 项目总体架构说明
- 训练脚本目录：`scripts/train/qwenvl_train/`
- 评估配置目录：`scripts/eval/configs/`
