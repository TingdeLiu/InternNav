# NavDP S1 单独推理技术文档

> NavDP (Navigation Diffusion Policy) 是 InternVLA-N1 双系统中的 System-1 低层运动控制策略。
> 本文档介绍如何在 InternNav 框架中**单独启动和调用 NavDP S1**，无需 S2 (VLM) 参与。

## 目录

- [1. 架构概览](#1-架构概览)
- [2. 环境准备](#2-环境准备)
- [3. 快速开始](#3-快速开始)
- [4. API 接口详解](#4-api-接口详解)
- [5. 数据格式说明](#5-数据格式说明)
- [6. 推理流程详解](#6-推理流程详解)
- [7. 与双系统集成对比](#7-与双系统集成对比)
- [8. 自定义 S1 策略网络](#8-自定义-s1-策略网络)
- [9. FAQ](#9-faq)

---

## 1. 架构概览

### 1.1 Client-Server 解耦架构

NavDP S1 采用 **HTTP Client-Server 解耦架构**，将模型推理与环境仿真完全分离：

```
┌─────────────────┐    HTTP    ┌──────────────────┐
│  评测 / 机器人    │ ────────▶ │  navdp_server.py │
│  (IsaacSim /     │ ◀──────── │  (Flask)         │
│   ROS / 自定义)  │  JSON      │                  │
└─────────────────┘           │  ┌──────────────┐ │
                               │  │ NavDPAgent   │ │
                               │  │ ┌──────────┐ │ │
                               │  │ │NavDP_    │ │ │
                               │  │ │Policy    │ │ │
                               │  │ └──────────┘ │ │
                               │  └──────────────┘ │
                               └──────────────────┘
```

**优势：**
- 模型推理和评测环境可运行在不同机器 / 不同 conda 环境
- 天然支持异步推理（规划线程与控制线程分离）
- 兼容 NavDP 项目的 IsaacSim 评测脚本

### 1.2 核心组件

| 文件 | 功能 |
|------|------|
| `internnav/agent/navdp_agent.py` | Agent 封装：记忆队列、RGBD 预处理、轨迹可视化 |
| `internnav/model/basemodel/navdp/policy_network.py` | NavDP_Policy：5 种任务模式的策略网络 |
| `internnav/model/basemodel/navdp/policy_backbone.py` | RGBD / ImageGoal / PixelGoal 编码器 |
| `scripts/inference/NavDP/navdp_server.py` | Flask HTTP 推理服务，暴露 5 个 API 端点 |
| `scripts/inference/NavDP/navdp_client.py` | HTTP 客户端 Python API |

### 1.3 支持的导航任务

| 任务模式 | 输入 | 端点 | 说明 |
|----------|------|------|------|
| PointGoal | RGBD + 3D 目标点 | `/pointgoal_step` | 导航到指定相对坐标 |
| NoGoal | RGBD | `/nogoal_step` | 自主探索，无明确目标 |
| ImageGoal | RGBD + 目标图像 | `/imagegoal_step` | 导航到目标图像所示位置 |
| PixelGoal | RGBD + 目标像素 | `/pixelgoal_step` | 导航到图像中指定像素位置 |
| MixGoal | RGBD + 点目标 + 图像目标 | （扩展） | 混合目标导航 |

---

## 2. 环境准备

### 2.1 前提条件

1. **模型权重**：从 [NavDP 发布页](https://docs.google.com/forms/d/e/1FAIpQLSdl3RvajO5AohwWZL5C0yM-gkSqrNaLGp1OzN9oF24oNLfikw/viewform) 获取 checkpoint（.ckpt 文件）

> **注意**：NavDP 的策略网络（`NavDP_Policy`）和 Backbone 已内置于 `internnav/model/basemodel/navdp/` 中，
> 无需额外克隆 NavDP 项目。

### 2.2 依赖安装

```bash
# 在 InternNav 的 conda 环境中
pip install flask torch diffusers opencv-python imageio
```

---

## 3. 快速开始

### 3.1 启动 Server

```bash
cd InternNav/

# 基本启动
python scripts/inference/NavDP/navdp_server.py \
    --port 8901 \
    --checkpoint /path/to/navdp_checkpoint.ckpt

# 完整参数
python scripts/inference/NavDP/navdp_server.py \
    --port 8901 \
    --checkpoint /path/to/navdp_checkpoint.ckpt \
    --device cuda:7 \
    --image_size 224 \
    --memory_size 8 \
    --predict_size 24
```

Server 启动后会在指定端口监听 HTTP 请求。

### 3.2 使用 Python Client 调用

```python
import numpy as np
from scripts.inference.NavDP.navdp_client import NavDPClient

# 1. 创建客户端
client = NavDPClient(port=8888)

# 2. 重置（传入相机内参）
camera_intrinsic = np.array([
    [386.5, 0.0, 328.9, 0.0],
    [0.0, 386.5, 244.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
])
client.reset(camera_intrinsic, batch_size=1, stop_threshold=-3.0)

# 3. 点目标导航：目标在前方 3m、左偏 1m
goal = np.array([[3.0, 1.0]])  # (B, 2) x=前, y=左
rgb = np.random.randint(0, 255, (1, 480, 640, 3), dtype=np.uint8)  # BGR
depth = np.ones((1, 480, 640), dtype=np.float32) * 2.0  # 2m

trajectory, all_traj, values = client.pointgoal_step(goal, rgb, depth)
print(f"最优轨迹形状: {trajectory.shape}")  # (1, 24, 3)
print(f"候选轨迹数: {all_traj.shape[1]}, critic 最高分: {values.max():.2f}")

# 4. 无目标探索
trajectory, all_traj, values = client.nogoal_step(rgb, depth)
```

### 3.3 直接使用 Agent（无需 Server）

```python
import numpy as np
from internnav.agent.navdp_agent import NavDPAgent

# 构造相机内参
intrinsic = np.array([
    [386.5, 0.0, 328.9, 0.0],
    [0.0, 386.5, 244.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
])

# 创建 Agent
agent = NavDPAgent(
    camera_intrinsic=intrinsic,
    checkpoint="/path/to/navdp_checkpoint.ckpt",
    device="cuda:0",
)
agent.reset(batch_size=1)

# 逐帧推理
for step in range(100):
    rgb = get_rgb_from_sensor()      # (1, H, W, 3) BGR
    depth = get_depth_from_sensor()  # (1, H, W, 1) float32 米

    # 点目标
    goal = np.array([[3.0, 1.0, 0.0]])  # (1, 3)
    best_traj, all_traj, values, vis_img = agent.step_pointgoal(goal, rgb, depth)

    # best_traj: (1, 24, 3) 最优轨迹
    # vis_img: 轨迹投影到图像的可视化
    action = best_traj[0, 0]  # 取第一步动作 [dx, dy, dz]
```

### 3.4 配合 NavDP 项目 IsaacSim 评测

Server 兼容 NavDP 项目的 HTTP 协议，可直接使用 NavDP 的评测脚本：

```bash
# Terminal 1: 在 InternNav 环境启动 server
cd InternNav/
python scripts/inference/NavDP/navdp_server.py --port 8901 --checkpoint /path/to/ckpt

# Terminal 2: 在 NavDP/IsaacSim 环境运行评测
cd NavDP/
python eval_pointgoal_wheeled.py --port 8901 --scene_dir /path/to/scenes --scene_index 0
python eval_nogoal_wheeled.py --port 8901 --scene_dir /path/to/scenes --scene_index 0
python eval_imagegoal_wheeled.py --port 8901 --scene_dir /path/to/scenes --scene_index 0
```

---

## 4. API 接口详解

### 4.1 Server 端点一览

| 端点 | 方法 | 功能 |
|------|------|------|
| `/navigator_reset` | POST | 初始化/重置导航器 |
| `/navigator_reset_env` | POST | 重置指定环境 |
| `/pointgoal_step` | POST | 点目标导航推理 |
| `/nogoal_step` | POST | 无目标探索推理 |
| `/imagegoal_step` | POST | 图像目标导航推理 |
| `/pixelgoal_step` | POST | 像素目标导航推理 |

### 4.2 `/navigator_reset`

**请求 (JSON):**
```json
{
    "intrinsic": [[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    "stop_threshold": -3.0,
    "batch_size": 1
}
```

**响应:**
```json
{"algo": "navdp"}
```

### 4.3 `/pointgoal_step`

**请求 (multipart/form-data):**
- `image`: JPEG 编码的 RGB 图像（多 batch 垂直拼接）
- `depth`: PNG uint16 编码的深度图（深度值 × 10000）
- `goal_data`: JSON 字符串 `{"goal_x": [3.0], "goal_y": [1.0]}`

**响应 (JSON):**
```json
{
    "trajectory": [[[dx, dy, dz], ...]],
    "all_trajectory": [[[[dx, dy, dz], ...], ...]],
    "all_values": [[v1, v2, ...]]
}
```

### 4.4 NavDPAgent Python API

```python
class NavDPAgent:
    def reset(batch_size=1, stop_threshold=-3.0) -> None
    def reset_env(env_id: int) -> None

    def step_pointgoal(goals, images, depths) -> (traj, all_traj, values, vis)
    def step_nogoal(images, depths) -> (traj, all_traj, values, vis)
    def step_imagegoal(goal_images, images, depths) -> (traj, all_traj, values, vis)
    def step_pixelgoal(pixel_goals, images, depths) -> (traj, all_traj, values, vis)
    def step_mixgoal(point_goals, image_goals, images, depths) -> (traj, all_traj, values, vis)
```

### 4.5 NavDPClient Python API

```python
class NavDPClient:
    def reset(camera_intrinsic, batch_size=1, stop_threshold=-3.0) -> str
    def reset_env(env_id: int) -> str

    def pointgoal_step(goals, rgb, depth) -> (traj, all_traj, values)
    def nogoal_step(rgb, depth) -> (traj, all_traj, values)
    def imagegoal_step(goal_images, rgb, depth) -> (traj, all_traj, values)
    def pixelgoal_step(pixel_goals, rgb, depth) -> (traj, all_traj, values)
```

---

## 5. 数据格式说明

### 5.1 输入

| 名称 | 形状 | 类型 | 说明 |
|------|------|------|------|
| RGB 图像 | `(B, H, W, 3)` | uint8 BGR | 当前观测 |
| 深度图 | `(B, H, W, 1)` | float32 | 单位：米，有效范围 0.1~5.0m |
| 点目标 | `(B, 3)` | float32 | 相机坐标系：x=前，y=左，z=上 |
| 像素目标 | `(B, 2)` | int | 图像坐标 [x, y] |
| 图像目标 | `(B, H, W, 3)` | uint8 BGR | 目标位置的参考图像 |

### 5.2 输出

| 名称 | 形状 | 说明 |
|------|------|------|
| `trajectory` | `(B, predict_size, 3)` | 最优轨迹（累积位移序列） |
| `all_trajectory` | `(B, sample_num, predict_size, 3)` | 所有候选轨迹 |
| `all_values` | `(B, sample_num)` | Critic 评分（越高越好） |
| `vis_image` | `(H, B*W, 3)` | 轨迹投影可视化（仅 Agent 返回） |

### 5.3 轨迹坐标系

```
       x (前进方向)
       ↑
       │
  y ←──┼── → -y
       │
       ↓
      -x

轨迹点 = cumsum(action_steps / 4.0)
每个点 (dx, dy, dz) 是相对当前位置的累积位移（单位：米）
```

### 5.4 记忆队列机制

NavDPAgent 内部维护每个 batch 的 RGBD 历史帧队列（`memory_size=8`）：

```
step 0:  [frame_0, pad, pad, pad, pad, pad, pad, pad]
step 1:  [frame_0, frame_1, pad, pad, pad, pad, pad, pad]
  ...
step 7:  [frame_0, frame_1, ..., frame_7]
step 8:  [frame_1, frame_2, ..., frame_8]  ← 最旧帧被移除
```

每次调用 `step_*()` 方法时自动更新记忆队列，新 episode 需调用 `reset()` 清空。

---

## 6. 推理流程详解

### 6.1 单次推理完整流程

```
输入: RGB (B,H,W,3) + Depth (B,H,W,1) + Goal
                │
        ┌───────┴───────┐
        │  预处理 & 记忆  │
        │  process_image │  → 缩放到 224×224，归一化 [0,1]
        │  process_depth │  → 缩放，过滤异常值
        │  _update_memory│  → 维护 8 帧历史队列
        └───────┬───────┘
                │
        ┌───────┴───────┐
        │  特征编码       │
        │  rgbd_encoder  │  → (B, 128, 384) 场景记忆 token
        │  goal_encoder  │  → (B, 1, 384) 目标 token
        └───────┬───────┘
                │
        ┌───────┴───────┐
        │  扩散采样       │  DDPM 10 步
        │  sample_num=16 │  采样 16 条候选轨迹
        │  predict_noise │  条件 = [time, goal×3, rgbd_embed]
        │  scheduler.step│  逆过程去噪
        └───────┬───────┘
                │
        ┌───────┴───────┐
        │  Critic 评分    │
        │  predict_critic│  只看 rgbd_embed（无目标信息）
        │  排序选择 top-2  │  正样本 & 负样本
        └───────┬───────┘
                │
        输出: trajectory (B, 24, 3) + values (B, 16)
```

### 6.2 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `image_size` | 224 | 模型输入分辨率 |
| `memory_size` | 8 | 历史帧数量 |
| `predict_size` | 24 | 轨迹预测点数 |
| `sample_num` | 16 | 每次扩散采样的候选轨迹数 |
| `token_dim` | 384 | Transformer token 维度 |
| `temporal_depth` | 16 | Transformer 解码器层数 |
| `heads` | 8 | 注意力头数 |
| `num_train_timesteps` | 10 | DDPM 扩散步数 |

---

## 7. 与双系统集成对比

### 7.1 三种 NavDP 使用方式

```
方式 1: NavDP 独立推理（本文档）
  ┌───────┐    ┌──────────┐    ┌──────────────┐
  │ Sensor │───▶│ NavDPAgent│───▶│ NavDP_Policy │
  └───────┘    └──────────┘    └──────────────┘
  目标来源: 外部指定（坐标/图像/像素）

方式 2: InternVLA-N1 双系统
  ┌───────┐    ┌──────────────────────────┐
  │ Sensor │───▶│ InternVLAN1Agent          │
  │+ 指令  │    │  ┌─────┐  traj_latents  ┌────────────────────────┐
  └───────┘    │  │ S2  │──────────────▶│ NavDP_DPT_CriticSum_DAT│
               │  │(VLM)│               │ (接受 VLM latent 条件)  │
               │  └─────┘               └────────────────────────┘
               └──────────────────────────┘
  目标来源: VLM 从语言指令生成 traj_latents

方式 3: NavDP 项目原生
  ┌──────────┐  HTTP  ┌───────────────────────┐
  │ IsaacSim │──────▶│ NavDP/baselines/navdp/ │
  │ eval脚本  │◀──────│ navdp_server.py        │
  └──────────┘       └───────────────────────┘
```

### 7.2 模型差异

| 特性 | NavDP_Policy (本文档) | NavDP_DPT_CriticSum_DAT (双系统) |
|------|----------------------|----------------------------------|
| 条件输入 | pointgoal / imagegoal / pixelgoal / nogoal | VLM traj_latents (3584→384) |
| RGBD 编码 | DepthAnythingV2 backbone | DAT_RGBD_Patch_Backbone |
| 独立使用 | ✅ 完全独立 | ❌ 需要 S2 VLM 提供 latent |
| 训练方式 | 单独训练 | 双系统联合训练 |
| 扩散步数 | 10 步 | 20 步 |

---

## 8. 自定义 S1 策略网络

如果你想**替换 NavDP 为自己的 S1 策略网络**，只需要实现以下接口：

### 8.1 最小接口要求

```python
class MyS1Policy(nn.Module):
    """自定义 S1 策略网络"""

    def predict_pointgoal_action(self, goal_point, input_images, input_depths, sample_num=16):
        """点目标推理

        Args:
            goal_point: (B, 3) float32 目标坐标
            input_images: (B, memory_size, 224, 224, 3) float32 [0,1] 归一化 RGB
            input_depths: (B, 224, 224, 1) float32 当前深度图

        Returns:
            all_trajectory: (B, sample_num, predict_size, 3) numpy 所有候选轨迹
            all_values: (B, sample_num) numpy critic 评分
            positive_trajectory: (B, 2, predict_size, 3) numpy top-2 最优
            negative_trajectory: (B, 2, predict_size, 3) numpy top-2 最差
        """
        ...

    def predict_nogoal_action(self, input_images, input_depths, sample_num=16):
        """无目标推理（接口同上，去掉 goal_point）"""
        ...

    # 可选：实现 predict_imagegoal_action / predict_pixelgoal_action / predict_ip_action
```

### 8.2 集成步骤

1. **替换 Policy 类**：修改 `internnav/agent/navdp_agent.py` 中的 import：
   ```python
   # from policy_network import NavDP_Policy
   from my_custom_policy import MyS1Policy as NavDP_Policy
   ```

2. **适配初始化参数**（如果构造函数签名不同）

3. **保持输入输出格式不变**，Server 和 Client 无需修改

### 8.3 集成到双系统

如果想将自定义 S1 集成到 InternVLA-N1 双系统，参考 `docs/custom_s1_s2_development_guide.md`：
- 在 `internvla_n1_arch.py` 中注册新的 build 函数
- 在 `internvla_n1.py` 的 `initialize_vision_modules()` 中添加分支
- 确保接受 `traj_latents` 作为条件输入

---

## 9. FAQ

### Q: 报错找不到 `policy_network` 模块？

`NavDP_Policy` 已内置于 `internnav/model/basemodel/navdp/policy_network.py`，
确保 InternNav 包已正确安装（`pip install -e .`）或项目根目录在 `PYTHONPATH` 中。

### Q: 推理速度如何？

在单张 RTX 3090 上：
- 单次推理（16 条候选轨迹）约 50-100ms
- 扩散采样 10 步是主要耗时
- 可通过减少 `sample_num` 加速

### Q: 如何在 ROS 中使用？

1. 启动 Server：`python scripts/inference/NavDP/navdp_server.py --port 8888 --checkpoint /path/to/ckpt`
2. 在 ROS 节点中使用 `NavDPClient` 调用
3. 将返回的轨迹转换为 ROS 消息发布

### Q: InternNav 的 NavDPNet 和 NavDP 项目的 NavDP_Policy 有什么区别？

两者架构相同，但：
- **NavDPNet** (`internnav/model/basemodel/navdp/`)：仅支持 pointgoal/nogoal，使用 HuggingFace PreTrainedModel 接口
- **NavDP_Policy** (`NavDP/baselines/navdp/`)：支持全部 5 种任务模式，直接 `torch.load` 加载

本方案选择复用 NavDP_Policy 以获得完整功能。

### Q: 如何训练自己的 NavDP 模型？

使用 InternNav 的训练脚本：
```bash
cd InternNav/
bash scripts/train/base_train/start_train.sh navdp
```
训练配置见 `scripts/train/base_train/configs/navdp.py`。

### Q: 深度图必须是真实的吗？

建议使用真实深度（如 RGB-D 相机）。如果不可用：
- 可以使用 DepthAnything V2 从 RGB 估计深度（NavDP 的 backbone 本身就基于 DepthAnything）
- 使用常量深度（如 2.0m）会降低 critic 评分的准确性，但轨迹方向仍可参考
