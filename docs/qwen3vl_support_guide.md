# Qwen3-VL 支持说明

> 本文档介绍如何在 InternNav 框架中使用 **Qwen3-VL** 作为 System-2 (S2) 的视觉语言模型后端，
> 以及与现有 Qwen2.5-VL 路径的差异与兼容性说明。

## 目录

- [1. 背景与范围](#1-背景与范围)
- [2. 环境依赖](#2-环境依赖)
- [3. 快速开始](#3-快速开始)
- [4. 训练](#4-训练)
- [5. 评估](#5-评估)
- [6. 改动文件一览](#6-改动文件一览)
- [7. 模型检测逻辑](#7-模型检测逻辑)
- [8. Qwen2.5-VL 与 Qwen3-VL 主要差异](#8-qwen25-vl-与-qwen3-vl-主要差异)
- [9. 当前限制](#9-当前限制)
- [10. FAQ](#10-faq)

---

## 1. 背景与范围

InternVLA-N1 的 System-2 (S2) 原本以 **Qwen2.5-VL** 作为视觉语言理解骨干。
Qwen3-VL 在以下方面有所提升：

- **Interleaved MRoPE**：时间、高度、宽度三维位置编码，视频理解更精准
- **DeepStack**：多层 ViT 特征融合，细粒度视觉识别增强
- **Text-Timestamp Alignment**：视频帧级时间戳对齐，支持事件精确定位

本次改动仅针对 **S2 单独运行模式**（`mode: system2`），双系统联合推理架构（`mode: dual_system`）中 `InternVLAN1ForCausalLM` 的骨干仍为 Qwen2.5-VL，不受影响。

---

## 2. 环境依赖

Qwen3-VL 需要 `transformers >= 4.57.0`：

```bash
pip install -U transformers
```

验证是否已支持：

```python
from transformers import Qwen3VLForConditionalGeneration  # 不报错即为已支持
```

其余依赖（`flash_attn`、`torch`、`qwen-vl-utils`）与 Qwen2.5-VL 相同，无需额外安装。

---

## 3. 快速开始

框架通过 `model_path` 自动判断模型类型，**无需更改任何代码**，只需将配置中的 `model_path` 指向 Qwen3-VL checkpoint 即可。

### 3.1 评估（单机）

```bash
python scripts/eval/eval.py \
    --config scripts/eval/configs/habitat_s2_qwen3vl_cfg.py
```

配置文件位于 `scripts/eval/configs/habitat_s2_qwen3vl_cfg.py`，核心字段：

```python
AgentCfg(
    model_name='internvla_n1',
    model_settings={
        "mode": "system2",
        "model_path": "checkpoints/InternVLA-N1-System2-Qwen3VL",  # 修改为实际路径
        "num_history": 8,
        "resize_w": 384,
        "resize_h": 384,
        "max_new_tokens": 1024,
    },
)
```

### 3.2 评估（分布式，8 GPU）

```bash
# 参考 scripts/eval/bash/eval_system2.sh，将 CONFIG 改为 Qwen3-VL 配置
CONFIG="scripts/eval/configs/habitat_s2_qwen3vl_cfg.py"

srun -p <YOUR_PARTITION_NAME> \
    --gres=gpu:8 --ntasks=8 --ntasks-per-node=8 --cpus-per-task=16 \
    python scripts/eval/eval.py --config $CONFIG \
    > logs/InternVLA-N1-Qwen3VL_log.txt 2>&1
```

### 3.3 用已有 Qwen3-VL checkpoint 直接推理

若已有 HuggingFace 上的原始 Qwen3-VL 权重，直接将 `model_path` 设为对应路径即可：

```python
model_settings={
    "mode": "system2",
    "model_path": "Qwen/Qwen3-VL-7B-Instruct",  # 或本地路径
    ...
}
```

---

## 4. 训练

使用 `scripts/train/qwenvl_train/train_system2_qwen3vl.sh`：

```bash
bash scripts/train/qwenvl_train/train_system2_qwen3vl.sh
```

默认配置：

| 参数 | 值 | 说明 |
|------|----|------|
| `llm` | `Qwen/Qwen3-VL-7B-Instruct` | 基础模型，可替换为其他尺寸 |
| `data_flatten` | `False` | Qwen3-VL 暂不支持 varlen flash-attn patch |
| `system1` | `none` | S2 单独训练，不含 S1 组件 |
| `run_name` | `InternVLA-N1-System2-Qwen3VL` | checkpoint 保存目录名 |

**切换模型尺寸**，仅需修改脚本中的 `llm` 变量：

```bash
# 可选模型（非 MoE）
llm=Qwen/Qwen3-VL-4B-Instruct
llm=Qwen/Qwen3-VL-8B-Instruct
llm=Qwen/Qwen3-VL-32B-Instruct
```

> **注意**：MoE 变体（`Qwen3-VL-30B-A3B`、`Qwen3-VL-235B-A22B`）需要 `Qwen3VLMoeForConditionalGeneration`，当前版本尚未支持。

---

## 5. 评估

### 5.1 指标

与 Qwen2.5-VL S2 评估完全一致，输出：

| 指标 | 说明 |
|------|------|
| SR (Success Rate) | 任务成功率 |
| SPL | 加权路径长度 |
| NE (Navigation Error) | 导航误差（米） |
| OS (Oracle Success) | 理论最优成功率 |

### 5.2 对比基线

若需与 Qwen2.5-VL S2 对比，使用：

```bash
# Qwen2.5-VL baseline
python scripts/eval/eval.py --config scripts/eval/configs/habitat_s2_cfg.py

# Qwen3-VL 新配置
python scripts/eval/eval.py --config scripts/eval/configs/habitat_s2_qwen3vl_cfg.py
```

---

## 6. 改动文件一览

本次新增 Qwen3-VL 支持涉及以下文件：

### 修改的文件

| 文件 | 改动位置 | 说明 |
|------|---------|------|
| `internnav/habitat_extensions/vln/habitat_vln_evaluator.py` | L31–36, L125–140 | 评估器 `system2` 分支增加 Qwen3-VL 加载逻辑 |
| `internnav/agent/dialog_agent.py` | L21–30, L88–103 | DialogAgent `system2` 分支增加 Qwen3-VL 加载逻辑 |
| `internnav/trainer/internvla_n1_trainer.py` | L40–43, L165–178 | 训练入口新增 `elif "qwen3"` 分支 |

### 新增的文件

| 文件 | 说明 |
|------|------|
| `scripts/train/qwenvl_train/train_system2_qwen3vl.sh` | Qwen3-VL S2 训练脚本 |
| `scripts/eval/configs/habitat_s2_qwen3vl_cfg.py` | Qwen3-VL S2 评估配置 |
| `docs/qwen3vl_support_guide.md` | 本文档 |

### 未改动的文件

以下文件**不受影响**，Qwen2.5-VL 的全部路径保持原样：

- `internnav/model/basemodel/internvla_n1/internvla_n1.py`（双系统架构）
- `internnav/trainer/qwenvl_base.py`（attention patch 工具）
- `scripts/eval/configs/habitat_s2_cfg.py`（原有 Qwen2.5-VL 评估配置）
- `scripts/train/qwenvl_train/train_system2.sh`（原有 Qwen2.5-VL 训练脚本）

---

## 7. 模型检测逻辑

框架通过 `model_path` 字符串自动判断要加载的模型类：

```
model_path.lower() 包含 "qwen3"  →  Qwen3VLForConditionalGeneration
model_path.lower() 包含 "qwen2.5" →  Qwen2_5_VLForConditionalGeneration
其他                              →  Qwen2VLForConditionalGeneration（兜底）
```

因此以下路径均会正确识别为 Qwen3-VL：

```
Qwen/Qwen3-VL-7B-Instruct
Qwen/Qwen3-VL-8B-Instruct
checkpoints/InternVLA-N1-System2-Qwen3VL
/mnt/data/models/qwen3-vl-finetune
```

若 `transformers` 版本过低导致 `Qwen3VLForConditionalGeneration` 不可用，程序会在模型加载阶段抛出明确错误：

```
ImportError: Qwen3VLForConditionalGeneration not found.
Please upgrade transformers: pip install -U transformers
```

---

## 8. Qwen2.5-VL 与 Qwen3-VL 主要差异

| 特性 | Qwen2.5-VL | Qwen3-VL |
|------|-----------|---------|
| 主模型类 | `Qwen2_5_VLForConditionalGeneration` | `Qwen3VLForConditionalGeneration` |
| Config 类 | `Qwen2_5_VLConfig` | `Qwen3VLConfig` |
| position_ids 维度 | 2D `(B, seq_len)` | 3D `(3, B, seq_len)`，MRoPE |
| 视频时间戳参数 | 无 | `second_per_grid_ts`（帧级时间戳） |
| MoE 变体 | 无 | `Qwen3VLMoeForConditionalGeneration` |
| 特殊 token ID | `image=151655, video=151656` | 相同 |
| Processor API | `AutoProcessor` | `AutoProcessor`（兼容） |
| 生成 API | `model.generate()` | `model.generate()`（兼容） |

对于 **S2 单独推理**，两者的 `model.generate()` 接口完全一致，框架层无需感知 position_ids 的变化（由模型内部处理）。

---

## 9. 当前限制

### 9.1 双系统联合推理（`mode: dual_system`）不支持

`InternVLAN1ForCausalLM` 通过多重继承直接基于 `Qwen2_5_VLForConditionalGeneration`，将 Qwen3-VL 替换为双系统骨干需要重新实现：

```python
# 当前架构（未变）
class InternVLAN1ForCausalLM(Qwen2_5_VLForConditionalGeneration, InternVLAN1MetaForCausalLM):
    ...
```

若需支持，需参考 `internvla_n1.py` 新建 `InternVLAN1ForCausalLMQwen3` 类并继承 `Qwen3VLForConditionalGeneration`。

### 9.2 `data_flatten` 训练模式不支持

`qwenvl_base.py` 中的 `replace_qwen2_vl_attention_class()` 仅 patch 了 Qwen2VL 和 Qwen2.5VL 的 flash-attn 路径。Qwen3-VL 训练时需保持 `data_flatten=False`（已在训练脚本中设置）。

### 9.3 MoE 变体不支持

`Qwen3-VL-30B-A3B` 和 `Qwen3-VL-235B-A22B` 使用 `Qwen3VLMoeForConditionalGeneration`，当前检测逻辑统一走 `Qwen3VLForConditionalGeneration`，加载会失败。如需支持 MoE 变体，需在检测逻辑中额外判断路径是否含 `"moe"` 或 `"-a"` 标识。

---

## 10. FAQ

**Q: 如何确认当前推理用的是 Qwen3-VL 还是 Qwen2.5-VL？**

在评估日志中搜索 `model_path`，或在代码中打印：

```python
print(type(self.model).__name__)
# Qwen3-VL → Qwen3VLForConditionalGeneration
# Qwen2.5-VL → Qwen2_5_VLForConditionalGeneration
```

---

**Q: 能否同时运行 Qwen2.5-VL 和 Qwen3-VL 的评估进程？**

可以。两个进程加载不同的 checkpoint，使用不同的 GPU（通过 `local_rank` 控制），互不干扰。

---

**Q: Qwen3-VL 的 checkpoint 是否与 Qwen2.5-VL 的权重格式兼容（能否直接 finetune 已有的 S2 权重）？**

不兼容。两者架构不同（MRoPE 维度、attention 结构），权重不可直接迁移，需从 Qwen3-VL 官方预训练权重开始 finetune。

---

**Q: `max_new_tokens` 设置多少合适？**

与 Qwen2.5-VL 相同，推荐 `1024`。若仅输出坐标或离散动作，`256` 即可，速度更快。
