# Qwen3-VL S2 部署与使用指南

> 本文档介绍如何在 InternNav 框架中**单独启动和调用 Qwen3-VL** 作为 System-2 (S2) 视觉语言模型。
> 提供两种部署方案：**GPU 服务器部署**（仿真评测 / 实机 LingNav 服务端）和
> **Wheeltec 小车上调用**（Jetson 通过 HTTP 远程调用 S2 服务）。

## 目录

- [1. 背景与范围](#1-背景与范围)
- [2. 环境依赖](#2-环境依赖)
- [3. 部署方案一：GPU 服务器](#3-部署方案一gpu-服务器)
- [4. 部署方案二：Wheeltec 小车调用](#4-部署方案二wheeltec-小车调用)
- [5. 训练](#5-训练)
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

本文档同时覆盖以下使用场景：
- **S2 单独运行**（`mode: system2`）：Habitat 仿真评测、LingNav 实机服务端
- **双系统联合推理**（`mode: dual_system`）：Qwen2.5-VL 路径完全不受影响

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

## 3. 部署方案一：GPU 服务器

GPU 服务器上有两种启动方式，分别对应**仿真评测**和**实机/LingNav**场景。

### 3.1 实机 / LingNav：wheeltec_s2_server.py

`wheeltec_s2_server.py` 是为实机导航（LingNav）设计的轻量 HTTP 推理服务，
接收机器人相机图像和导航指令，返回目标像素坐标与转向建议。

```bash
conda activate internnav
cd /path/to/InternNav

# 7B 模型（推荐，约需 16GB 显存）
python scripts/realworld2/wheeltec_s2_server.py \
    --model_path /data2/ltd/checkpoints/Qwen3-VL/Qwen3-VL-8B-Instruct \
    --port 8890 \
    --host 0.0.0.0 \
    --device auto

# 显存不足时用 3B 版本
python scripts/realworld2/wheeltec_s2_server.py \
    --model_path Qwen/Qwen3-VL-3B-Instruct \
    --port 8890 \
    --host 0.0.0.0 \
    --device cuda:0
```

启动成功输出：
```
[S2] Loading processor from /data2/ltd/checkpoints/Qwen3-VL/Qwen3-VL-8B-Instruct …
[S2] Loaded with attn_implementation=flash_attention_2
[S2] Model ready.
[S2] Listening on http://0.0.0.0:8890
```

**服务器参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_path` | `/data2/ltd/checkpoints/Qwen3-VL/Qwen3-VL-8B-Instruct` | 模型 ID 或本地路径 |
| `--port` | `8890` | 监听端口 |
| `--host` | `127.0.0.1` | 监听地址（`0.0.0.0` 接受外部连接） |
| `--device` | `auto` | `auto` 自动多卡 / `cuda:0` 指定单卡 |
| `--image_width` | `640` | 机器人相机宽（用于坐标转换） |
| `--image_height` | `480` | 机器人相机高 |
| `--resize_w` | `640` | 传入模型前的缩放宽（须为 32 的倍数） |
| `--resize_h` | `480` | 传入模型前的缩放高 |
| `--max_new_tokens` | `128` | 最大生成 token 数 |

**S2 服务器响应格式：**

```
{"target": "red chair", "point_2d": [412, 680]}
↑↑←
```
- `point_2d` 为归一化坐标 [0, 1000]，服务器自动转为像素坐标
- 目标不可见：`{"target": null, "point_2d": null}` + 旋转符号（如 `←←`）
- 到达目标：`{"target": "chair", ...}` + `stop`

**验证 S2 服务器（可选）：**

```bash
# 连通性测试（随机图片）
python scripts/realworld2/test_s2_client.py \
    --host 192.168.1.100 --port 8890 \
    --random \
    --instruction "Go to the red chair"

# 真实图片测试
python scripts/realworld2/test_s2_client.py \
    --host 192.168.1.100 --port 8890 \
    --image /path/to/test.jpg \
    --instruction "Navigate to the door"
```

### 3.2 仿真评测（Habitat）

框架通过 `model_path` 自动判断模型类型，只需修改配置中的 `model_path` 即可。

**单机评估：**

```bash
python scripts/eval/eval.py \
    --config scripts/eval/configs/habitat_s2_qwen3vl_cfg.py
```

配置文件核心字段：

```python
AgentCfg(
    model_name='internvla_n1',
    model_settings={
        "mode": "system2",
        "model_path": "/data2/ltd/checkpoints/Qwen3-VL/Qwen3-VL-8B-Instruct",  # 服务器本地路径
        "num_history": 8,
        "resize_w": 384,
        "resize_h": 384,
        "max_new_tokens": 1024,
    },
)
```

**分布式评估（8 GPU）：**

```bash
CONFIG="scripts/eval/configs/habitat_s2_qwen3vl_cfg.py"

srun -p <YOUR_PARTITION_NAME> \
    --gres=gpu:8 --ntasks=8 --ntasks-per-node=8 --cpus-per-task=16 \
    python scripts/eval/eval.py --config $CONFIG \
    > logs/InternVLA-N1-Qwen3VL_log.txt 2>&1
```

**使用 HuggingFace 上的原始 Qwen3-VL 权重：**

```python
model_settings={
    "mode": "system2",
    "model_path": "Qwen/Qwen3-VL-8B-Instruct",  # 或本地路径
    ...
}
```

**双系统联合推理（`mode: dual_system`）：**

```python
model_settings={
    "mode": "dual_system",
    "model_path": "/data2/ltd/checkpoints/InternVLA-N1/InternVLA-N1-w-NavDP",  # 默认双系统（Qwen2.5-VL）；Qwen3-VL 双系统路径须含 "qwen3"
    ...
}
```

框架会自动加载 `InternVLAN1ForCausalLMQwen3`；路径不含 `qwen3` 时仍加载
`InternVLAN1ForCausalLM`（Qwen2.5-VL 骨干）。

---

## 4. 部署方案二：Wheeltec 小车调用

Qwen3-VL 7B 需要约 16GB 显存，**不能在 Jetson Orin NX 16GB 上运行**（显存被 NavDP 和系统占用）。
因此 Jetson 侧只需通过 HTTP 远程调用 GPU 服务器上的 S2 服务。

### 4.1 Jetson 端依赖安装（首次）

```bash
pip3 install numpy requests Pillow opencv-python casadi scipy \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

sudo apt install ros-humble-cv-bridge ros-humble-message-filters
```

### 4.2 测试 S2 连通性

在 Jetson 上验证能否访问 GPU 服务器的 S2：

```bash
# 随机图片连通性测试
python3 scripts/realworld2/test_s2_client.py \
    --host 192.168.1.100 --port 8890 \
    --random \
    --instruction "Go to the red chair"

# 用本地图片测试识别效果
python3 scripts/realworld2/test_s2_client.py \
    --host 192.168.1.100 --port 8890 \
    --image /path/to/test.jpg \
    --instruction "Navigate to the door"
```

### 4.3 在 LingNav ROS2 中使用

S2 始终运行在 GPU 服务器上，Jetson 侧通过 `--s2_host` 指定服务器 IP，无需额外配置。

**LingNav 模式 A（S1 也在服务器，完整双服务器）：**

```bash
# Jetson Terminal 3
python3 scripts/realworld2/lingnav_ros_client.py \
    --instruction "Go to the red chair" \
    --s2_host 192.168.1.100 \
    --s2_port 8890 \
    --s1_host 192.168.1.100 \
    --s1_port 8901
```

**LingNav 模式 B（S1 在 Jetson 本地）：**

```bash
# Jetson Terminal 3
python3 scripts/realworld2/lingnav_ros_client.py \
    --instruction "Go to the red chair" \
    --s2_host 192.168.1.100 \
    --s2_port 8890 \
    --local_s1 \
    --s1_checkpoint /home/wheeltec/VLN/checkpoints/navdp-weights.ckpt \
    --s1_device cuda:0 \
    --s1_half
```

**ROS2 客户端参数（S2 相关）：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--s2_host` | `127.0.0.1` | S2 服务器 IP（GPU 服务器局域网 IP）|
| `--s2_port` | `8890` | S2 端口 |

### 4.4 在 Python 代码中直接调用 S2

```python
import requests
import numpy as np
import cv2

S2_URL = "http://192.168.1.100:8890"

# 健康检查
resp = requests.get(f"{S2_URL}/health")
print(resp.json())  # {'status': 'ok', 'model': '/data2/ltd/checkpoints/Qwen3-VL/Qwen3-VL-8B-Instruct'}

# 推理：发送图片 + 指令
rgb = cv2.imread("/path/to/image.jpg")
_, img_bytes = cv2.imencode(".jpg", rgb)

resp = requests.post(
    f"{S2_URL}/step",
    files={"image": img_bytes.tobytes()},
    data={"instruction": "Go to the red chair"},
)
result = resp.json()
print(result)
# {
#   "target": "red chair",
#   "point_2d": [412, 680],      # 像素坐标
#   "point_2d_norm": [644, 1000], # 归一化坐标 [0,1000]
#   "navigation": "↑↑←"
# }
```

---

## 5. 训练

使用 `scripts/train/qwenvl_train/train_system2_qwen3vl.sh`：

```bash
bash scripts/train/qwenvl_train/train_system2_qwen3vl.sh
```

默认配置：

| 参数 | 值 | 说明 |
|------|----|------|
| `llm` | `Qwen/Qwen3-VL-8B-Instruct` | 基础模型，可替换为其他尺寸 |
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

## 6. 改动文件一览

### 修改的文件

| 文件 | 改动位置 | 说明 |
|------|---------|------|
| `internnav/habitat_extensions/vln/habitat_vln_evaluator.py` | L31–36, L118–142 | 评估器 `dual_system` 和 `system2` 分支均增加 Qwen3-VL 加载逻辑 |
| `internnav/agent/dialog_agent.py` | L21–30, L88–103 | DialogAgent `system2` 分支增加 Qwen3-VL 加载逻辑 |
| `internnav/trainer/internvla_n1_trainer.py` | L40–43, L165–178 | 训练入口新增 `elif "qwen3"` 分支 |

### 新增的文件

| 文件 | 说明 |
|------|------|
| `internnav/model/basemodel/internvla_n1/internvla_n1_qwen3.py` | Qwen3-VL 双系统模型类（`InternVLAN1ForCausalLMQwen3`） |
| `scripts/train/qwenvl_train/train_system2_qwen3vl.sh` | Qwen3-VL S2 训练脚本 |
| `scripts/eval/configs/habitat_s2_qwen3vl_cfg.py` | Qwen3-VL S2 评估配置 |
| `scripts/realworld2/wheeltec_s2_server.py` | LingNav 实机 S2 推理服务 |

### 未改动的文件

以下文件**不受影响**，Qwen2.5-VL 的全部路径保持原样：

- `internnav/model/basemodel/internvla_n1/internvla_n1.py`
- `internnav/trainer/qwenvl_base.py`
- `scripts/eval/configs/habitat_s2_cfg.py`
- `scripts/train/qwenvl_train/train_system2.sh`

---

## 7. 模型检测逻辑

框架通过 `model_path` 字符串自动判断要加载的模型类：

```
mode: system2
  model_path 含 "qwen3"  →  Qwen3VLForConditionalGeneration
  其他                   →  Qwen2_5_VLForConditionalGeneration（或 Qwen2VL 兜底）

mode: dual_system
  model_path 含 "qwen3"  →  InternVLAN1ForCausalLMQwen3   （Qwen3VL 骨干）
  其他                   →  InternVLAN1ForCausalLM         （Qwen2.5VL 骨干）
```

以下路径均会正确识别为 Qwen3-VL：

```
Qwen/Qwen3-VL-8B-Instruct
checkpoints/InternVLA-N1-System2-Qwen3VL
checkpoints/InternVLA-N1-Qwen3VL
/mnt/data/models/qwen3-vl-finetune
```

若 `transformers` 版本过低，程序会抛出明确错误：

```
ImportError: Qwen3VLForConditionalGeneration not found.
Please upgrade transformers: pip install -U transformers
```

---

## 8. Qwen2.5-VL 与 Qwen3-VL 主要差异

| 特性 | Qwen2.5-VL | Qwen3-VL |
|------|-----------|---------|
| 主模型类 | `Qwen2_5_VLForConditionalGeneration` | `Qwen3VLForConditionalGeneration` |
| 双系统模型类 | `InternVLAN1ForCausalLM` | `InternVLAN1ForCausalLMQwen3` |
| Config 类 | `Qwen2_5_VLConfig` / `InternVLAN1ModelConfig` | `Qwen3VLConfig` / `InternVLAN1ModelConfigQwen3` |
| position_ids 维度 | 2D `(B, seq_len)` → 框架内扩为 3D | 3D `(3, B, seq_len)`，MRoPE |
| 视频时间戳参数 | 无 | `second_per_grid_ts`（帧级时间戳） |
| MoE 变体 | 无 | `Qwen3VLMoeForConditionalGeneration`（暂不支持） |
| Processor API | `AutoProcessor` | `AutoProcessor`（兼容） |
| 生成 API | `model.generate()` | `model.generate()`（兼容） |

---

## 9. 当前限制

### 9.1 `data_flatten` 训练模式不支持

`qwenvl_base.py` 中的 `replace_qwen2_vl_attention_class()` 仅 patch 了 Qwen2VL 和 Qwen2.5VL 的 flash-attn 路径。Qwen3-VL 训练时需保持 `data_flatten=False`（已在训练脚本中设置）。

### 9.2 MoE 变体不支持

`Qwen3-VL-30B-A3B` 和 `Qwen3-VL-235B-A22B` 使用 `Qwen3VLMoeForConditionalGeneration`，当前检测逻辑统一走 `Qwen3VLForConditionalGeneration`，加载 MoE checkpoint 会失败。如需支持，需在检测逻辑中额外判断路径是否含 `"moe"` 或 `"-a"` 标识。

### 9.3 Jetson 上不可运行 Qwen3-VL

Jetson Orin NX 16GB 显存不足以运行 Qwen3-VL（最小 3B 版也需约 6-8GB，加上系统和 NavDP 后超限）。S2 始终需要部署在 GPU 服务器上。

---

## 10. FAQ

**Q: 如何确认当前推理用的是 Qwen3-VL 还是 Qwen2.5-VL？**

```python
print(type(self.model).__name__)
# system2 Qwen3-VL   → Qwen3VLForConditionalGeneration
# system2 Qwen2.5-VL → Qwen2_5_VLForConditionalGeneration
# dual_system Qwen3  → InternVLAN1ForCausalLMQwen3
# dual_system Q2.5   → InternVLAN1ForCausalLM
```

---

**Q: 能否同时运行 Qwen2.5-VL 和 Qwen3-VL 的评估进程？**

可以。两个进程加载不同 checkpoint，使用不同 GPU（通过 `local_rank` 控制），互不干扰。

---

**Q: Qwen3-VL 的 checkpoint 是否与 Qwen2.5-VL 兼容（能否直接 finetune 已有的 S2 权重）？**

不兼容。两者架构不同（MRoPE 维度、attention 结构），需从 Qwen3-VL 官方预训练权重开始 finetune。

---

**Q: `max_new_tokens` 设置多少合适？**

推荐 `1024`。若仅输出坐标或离散动作（如 LingNav 场景），`128` 即可，速度更快。

---

**Q: 双系统模式下 `InternVLAN1ForCausalLMQwen3` 与原版有什么本质区别？**

仅继承链不同：`InternVLAN1ForCausalLMQwen3` 继承自 `Qwen3VLForConditionalGeneration`，`InternVLAN1ForCausalLM` 继承自 `Qwen2_5_VLForConditionalGeneration`。两者共享同一套 `InternVLAN1MetaForCausalLM` mixin（`generate_latents`、`generate_traj` 等），S1 扩散头逻辑完全一致。
