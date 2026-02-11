# NavDP 技术文档

## 目录
1. [概述](#1-概述)
2. [模型架构](#2-模型架构)
3. [Diffusion Policy 详解](#3-diffusion-policy-详解)
4. [三种目标表示](#4-三种目标表示)
   - 4.1 [目标类型对比](#41-目标类型对比)
   - 4.2 [Pixel Goal 的处理](#42-pixel-goal-的处理)
   - 4.3 [训练时的 Goal 选用机制：双路径 + 随机混合](#43-训练时的-goal-选用机制双路径--随机混合)
   - 4.4 [推理时的 Goal 选用规则](#44-推理时的-goal-选用规则)
   - 4.5 [Critic 的特殊处理](#45-critic-的特殊处理)
   - 4.6 [多目标联合训练的思考](#46-多目标联合训练的思考)
5. [训练流程](#5-训练流程)
6. [数据集与数据加载](#6-数据集与数据加载)
7. [关键配置参数](#7-关键配置参数)
8. [训练优化建议](#8-训练优化建议)
9. [推理与部署](#9-推理与部署)
10. [常见问题](#10-常见问题)
11. [当前 Pixel Goal 的局限性](#11-当前-pixel-goal-的局限性)
12. [VLM + Diffusion 物体导航方案](#12-vlm--diffusion-物体导航方案)

---

## 1. 概述

NavDP (Navigation Diffusion Policy) 是一个基于扩散模型的视觉导航策略网络，采用**模仿学习 (Imitation Learning)** 的方式从专家轨迹中学习导航策略。

### 1.1 核心特点

| 特点 | 说明 |
|------|------|
| **Diffusion Policy** | 使用 DDPM 生成动作序列，而非直接回归 |
| **多目标融合** | 支持 Point Goal / Image Goal / Pixel Goal 三种目标表示 |
| **DepthAnything** | 使用预训练的深度估计模型作为视觉编码器 |
| **Critic 辅助** | 预测轨迹质量分数，用于推理时选择最优轨迹 |

### 1.2 输入输出

```
输入:
  - 历史 RGB 图像: (B, 8, 224, 224, 3)
  - 当前深度图: (B, 224, 224, 1)
  - Point Goal: (B, 3) [x, y, theta]
  - Image Goal: (B, 224, 224, 6)
  - Pixel Goal: (B, 224, 224, 4)

输出:
  - 动作轨迹: (B, 24, 3) [x, y, theta] × 24 步
  - Critic 分数: (B, 1) 轨迹质量评估
```

---

## 2. 模型架构

### 2.1 架构总览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           NavDP_Policy_DPT_CriticSum_DAT                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐     │
│  │ Point Goal  │   │ Image Goal  │   │ Pixel Goal  │   │  RGB-D      │     │
│  │   (3,)      │   │(224,224,6)  │   │(224,224,4)  │   │ Memory      │     │
│  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘   └──────┬──────┘     │
│         │                 │                 │                 │             │
│         ▼                 ▼                 ▼                 ▼             │
│  ┌──────────────┐  ┌──────────────────────────────┐   ┌──────────────┐     │
│  │point_encoder│  │      vlm_embed_mlp           │   │ rgbd_encoder │     │
│  │ Linear(3→384)│  │  3584 → 896 → 448 → 384     │   │(DAT Backbone)│     │
│  └──────┬──────┘  └──────────────┬───────────────┘   └──────┬──────┘     │
│         │                        │                          │             │
│         │                        ▼                          │             │
│         │                 ┌──────────────┐                  │             │
│         │                 │goal_compressor│                  │             │
│         │                 │(TokenCompressor)│                │             │
│         │                 └──────┬───────┘                  │             │
│         │                        │                          │             │
│         ▼                        ▼                          ▼             │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                     Condition Embedding                              │  │
│  │        [time_embed, goal_embed, rgbd_embed] + pos_embed             │  │
│  │                      (1, memory_size*16+2, 384)                      │  │
│  └─────────────────────────────────┬───────────────────────────────────┘  │
│                                    │                                       │
│                                    ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                    Diffusion Process (DDPM)                          │  │
│  │                                                                      │  │
│  │   Noisy Action → Transformer Decoder (16层) → Noise Prediction      │  │
│  │                                                                      │  │
│  └─────────────────────────────────┬───────────────────────────────────┘  │
│                                    │                                       │
│                                    ▼                                       │
│  ┌──────────────────┐       ┌──────────────────┐                          │
│  │   action_head    │       │   critic_head    │                          │
│  │  Linear(384→3)   │       │  Linear(384→1)   │                          │
│  └──────────────────┘       └──────────────────┘                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 RGB-D 编码器 (DAT_RGBD_Patch_Backbone)

使用 **DepthAnything V2 ViT-S** 作为视觉编码器：

```python
# 处理流程
RGB Images (B, 8, 224, 224, 3)
    ↓ DepthAnything V2 ViT-S
image_token: (B, 8*256, 384)  # 每帧 256 个 patch token

Depth (B, 224, 224, 1) → 复制为3通道
    ↓ DepthAnything V2 ViT-S
depth_token: (B, 256, 384)

# 融合
former_token = [image_token, depth_token] + positional_encoding
    ↓ TransformerDecoder (2 layers)
memory_token: (B, 128, 384)
```

### 2.3 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `token_dim` | 384 | 主要特征维度 |
| `vlm_token_dim` | 3584 | VLM 输入维度 |
| `temporal_depth` | 16 | Transformer Decoder 层数 |
| `heads` | 8 | 注意力头数 |
| `memory_size` | 8 | 历史帧数 |
| `predict_size` | 24 | 预测步数 |
| `num_train_timesteps` | 20 | 扩散步数 |

---

## 3. Diffusion Policy 详解

### 3.1 什么是 Diffusion Policy

**核心思想**：不直接预测动作，而是学习**从噪声中恢复动作**的过程。

```
传统方法:  观测 → 模型 → 动作
Diffusion: 观测 → 模型 → 逐步去噪 (20步) → 动作
```

### 3.2 训练过程 (Forward Diffusion)

```
┌─────────────────────────────────────────────────────────────────┐
│                        训练时                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   GT Action (B,24,3)  +  随机噪声 ε  +  随机时间步 t∈[0,19]     │
│                         │                                        │
│                         ▼                                        │
│              ┌─────────────────────┐                            │
│              │   加噪公式 (DDPM)    │                            │
│              │ x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε                     │
│              └──────────┬──────────┘                            │
│                         │                                        │
│                         ▼                                        │
│              Noisy Action x_t (B,24,3)                          │
│                         │                                        │
│                         ▼                                        │
│              Transformer Decoder                                 │
│              (输入: x_t + time_embed + conditions)              │
│                         │                                        │
│                         ▼                                        │
│              预测噪声 ε_pred (B,24,3)                           │
│                         │                                        │
│                         ▼                                        │
│              Loss = MSE(ε, ε_pred)                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 推理过程 (Reverse Diffusion)

```
┌─────────────────────────────────────────────────────────────────┐
│                        推理时                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Step 1: 初始化 sample_num 条随机噪声                          │
│   x_19 ~ N(0,1), shape: (sample_num, 24, 3)                     │
│                                                                  │
│   Step 2: 迭代去噪 (t = 19 → 0)                                 │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  for t in [19, 18, ..., 1, 0]:                          │   │
│   │      ε_pred = model(x_t, t, conditions)                 │   │
│   │      x_{t-1} = denoise_step(x_t, ε_pred, t)             │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Step 3: 输出 sample_num 条不同的轨迹                          │
│   x_0: (sample_num, 24, 3)                                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.4 去噪过程可视化

```
t=19: ░░░░░░░░░░░░░░░░░░░░░░░░  (纯噪声)
      ↓ predict_noise + denoise
t=15: ░░░░░░░░▓▓▓▓░░░░░░░░░░░░  (开始有结构)
      ↓
t=10: ░░░░▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░  (轨迹轮廓出现)
      ↓
t=5:  ░░▓▓▓▓████████▓▓▓▓░░░░░░  (细节逐渐清晰)
      ↓
t=0:  ▓▓████████████████▓▓░░░░  (最终轨迹)

█ = 动作信号    ░ = 噪声
```

### 3.5 sample_num 的作用

`sample_num` 是**并行采样的轨迹数量**，用于生成多条候选轨迹：

```python
# 推理时
sample_num = 32  # 同时生成 32 条不同轨迹

# 因为初始噪声不同，去噪后得到不同的结果：
noise_0 = [0.3, -0.5, 0.8, ...]  → 去噪 → 轨迹 A
noise_1 = [-0.2, 0.7, -0.1, ...] → 去噪 → 轨迹 B (不同！)
noise_2 = [0.9, 0.1, -0.6, ...]  → 去噪 → 轨迹 C (不同！)
```

### 3.6 为什么用 Diffusion？

| 优势 | 说明 |
|------|------|
| **多模态输出** | 可以采样多条轨迹，选择最优 |
| **平滑轨迹** | 去噪过程天然产生平滑结果 |
| **条件生成** | 容易融合多种条件（目标、观测） |
| **不确定性** | 多次采样可以估计预测的不确定性 |

---

## 4. 三种目标表示

### 4.1 目标类型对比

| 目标类型 | 输入形状 | 编码方式 | 特点 |
|----------|----------|----------|------|
| **Point Goal** | (3,) [x, y, theta] | Linear(3→384) | 简单直接，始终可用 |
| **Image Goal** | (224, 224, 6) | ViT + Compressor | 包含目标帧+起始帧 |
| **Pixel Goal** | (224, 224, 4) | ViT + Compressor | mask(1) + current_image(3) |

### 4.2 Pixel Goal 的处理

**process_pixel_goal()** 将 3D 目标点投影到 2D 图像：

```
3D 目标点 (XYT) → 坐标系转换 → 相机坐标系 → 透视投影 → 2D 像素坐标 → 画矩形标记
```

当目标不在视野内时：
- `pixel_mask` 保持全黑（全零）
- `visible_flag = False`
- 模型依赖 Point Goal / Image Goal 进行导航

### 4.3 训练时的 Goal 选用机制：双路径 + 随机混合

训练时三种 Goal **不是"三选一"的互斥关系**，而是采用了**双路径 + 随机混合**的训练策略。

代码位置：`internnav/model/basemodel/navdp/navdp_policy.py` → `forward()`

#### 4.3.1 NG 路径（No-Goal，无目标路径）

条件中 Goal 位置**全部填零**，只依靠 RGBD 视觉信息：

```python
# Goal 槽位全部为零向量
ng_cond = [time_embed, zero, zero, zero, rgbd_embed]
```

**作用**：学习"无目标时的探索行为"，让模型仅凭视觉即可做出安全导航。

#### 4.3.2 MG 路径（Multi-Goal，多目标路径）

条件中有 **3 个 Goal 槽位**，每个槽位**独立随机**从 {Point Goal, Image Goal, Pixel Goal} 中选取：

```python
cand_goal_embed = [pointgoal_embed, imagegoal_embed, pixelgoal_embed]

# 3 个槽位各自独立随机选择 (batch 级别)
selections_0 = torch.randint(0, 3, (batch_size,))  # 槽位 0
selections_1 = torch.randint(0, 3, (batch_size,))  # 槽位 1
selections_2 = torch.randint(0, 3, (batch_size,))  # 槽位 2

mg_cond = [time_embed, selected_goal_0, selected_goal_1, selected_goal_2, rgbd_embed]
```

**可能的组合 (3^3 = 27 种)**：

| 槽位0 | 槽位1 | 槽位2 | 含义 |
|--------|--------|--------|------|
| Point | Point | Point | 纯 Point Goal 导航 |
| Pixel | Pixel | Pixel | 纯 Pixel Goal 导航 |
| Image | Image | Image | 纯 Image Goal 导航 |
| Point | Image | Pixel | 三种混合 |
| Point | Pixel | Pixel | Point + Pixel 混合 |
| ... | ... | ... | 所有组合均匀采样 |

**设计原理**（代码注释）：
> "Use random sampling to ensure all goal encoders receive gradients"

- 保证三种 Goal Encoder 都能均匀获得训练梯度
- 模型学会处理任意 Goal 组合（单一或混合）
- 推理时可灵活选择任意 Goal 类型填入

#### 4.3.3 训练条件嵌入结构

```
Condition Embedding (cond_pos_embed 维度: memory_size*16 + 4):
┌──────────┬────────┬────────┬────────┬──────────────────┐
│time_embed│ Goal_0 │ Goal_1 │ Goal_2 │   rgbd_embed     │
│  (1,384) │(1,384) │(1,384) │(1,384) │(mem*16, 384)     │
└──────────┴────────┴────────┴────────┴──────────────────┘
     ↑         ↑        ↑        ↑          ↑
  扩散时间步  随机选    随机选    随机选   视觉编码
              的Goal   的Goal   的Goal
```

#### 4.3.4 辅助损失（Auxiliary Loss）

三种 Goal 各自有独立的辅助预测头，确保每个 Encoder 学到有意义的特征：

```python
# 每个 Goal Encoder 输出都有对应的辅助损失
pointgoal_aux_pred = self.point_aux_head(pointgoal_embed)   # 预测 Point Goal
imagegoal_aux_pred = self.image_aux_head(imagegoal_embed)   # 预测 Point Goal
pixelgoal_aux_pred = self.pixel_aux_head(pixelgoal_embed)   # 预测 Point Goal

# aux_loss = (1/3) * (pg_pred + ig_pred + pixel_pred)
```

### 4.4 推理时的 Goal 选用规则

推理时**没有随机混合**，而是根据可用的 Goal 类型，将对应的 embed 填入 3 个槽位。

#### 4.4.1 已有的推理方法

| 推理方法 | 3 个 Goal 槽位填充 | 适用场景 |
|----------|---------------------|----------|
| `predict_pointgoal_batch_action_vel()` | `[point, point, point]` | 有明确 3D 坐标 |
| `predict_nogoal_batch_action_vel()` | `[zero, zero, zero]` | 无目标探索 |
| `predict_noise()` 通用接口 | `[goal, goal, goal]` 重复 3 次 | 任意单一 Goal |

#### 4.4.2 选用策略

```
推理决策流程：

有 Pixel Goal（VLM 输出物体像素位置）？
  ├── YES → 3 个槽位都填 pixelgoal_embed
  └── NO
        ├── 有 Point Goal（3D 坐标已知）？
        │     ├── YES → 3 个槽位都填 pointgoal_embed
        │     └── NO
        │           ├── 有 Image Goal（目标图像已知）？
        │           │     ├── YES → 3 个槽位都填 imagegoal_embed
        │           │     └── NO → 3 个槽位都填 zero（无目标探索）
```

#### 4.4.3 混合使用的可能性

由于训练时模型见过各种 Goal 组合，推理时也可以混合填入（但目前代码未提供该接口）：

```python
# 理论上可行但目前未实现
# 例: 同时有 Point Goal 和 Pixel Goal 时
mg_cond = [time_embed, pointgoal_embed, pixelgoal_embed, pixelgoal_embed, rgbd_embed]
```

### 4.5 Critic 的特殊处理

Critic 评分时**刻意屏蔽所有 Goal 信息**，只看 RGBD：

```python
# Critic 条件: Goal 槽位全为零 + RGBD 视觉
cond_critic_mask[:, 0:4] = float('-inf')  # 前 4 个位置 (time + 3 goals) 被 mask

# 这意味着 Critic 只评估 "轨迹在当前视觉环境中是否安全"
# 与目标无关，纯粹评判碰撞风险
```

### 4.6 多目标联合训练的思考

**潜在问题**：
- 三种目标的特征空间差异大
- 不同目标可能给出冲突信号
- 单任务性能可能不如专门训练

**好处**：
- 互补学习：Pixel Goal 不可见时用 Point Goal
- 特征共享：底层视觉特征可复用
- 真实场景需要多种目标配合
- 随机混合训练让模型对任何 Goal 组合都鲁棒

**如果只用 Pixel Goal**，可以考虑：
1. 训练时屏蔽其他目标输入（3 个槽位都填 pixelgoal_embed）
2. 修改模型结构，删除不需要的编码器
3. 调整 loss 权重，只优化 Pixel Goal 相关损失

---

## 5. 训练流程

### 5.1 启动训练

```bash
./scripts/train/base_train/start_train.sh --name navdp_train --model navdp
```

### 5.2 训练数据流

```
启动脚本
    ↓
train.py (主入口)
    ↓
加载配置 (navdp_exp_cfg)
    ↓
创建数据集 (NavDP_Base_Datset)
    ├── 扫描目录结构
    ├── 收集所有 episode 路径
    └── 存储: trajectory_data_dir, trajectory_rgb_path, etc.
    ↓
创建 DataLoader
    ├── batch_size=64
    ├── num_workers=16
    └── collate_fn=navdp_collate_fn
    ↓
训练循环
    ├── 1. 数据加载 (__getitem__)
    ├── 2. 前向传播 (Diffusion 加噪 + 预测噪声)
    ├── 3. 计算损失 (action_loss + critic_loss + aux_loss)
    ├── 4. 反向传播
    └── 5. 优化器更新
```

### 5.3 损失函数

```python
# 动作损失
ng_action_loss = (pred_ng - noise[0]).square().mean()
mg_action_loss = (pred_mg - noise[1]).square().mean()
action_loss = 0.5 * mg_action_loss + 0.5 * ng_action_loss

# 辅助损失 (Point Goal 预测)
aux_loss = (1/3) * (pg_pred_1 - gt_pg).square().mean() + ...

# Critic 损失
critic_loss = (critic_pred - gt_critic).square().mean()

# 总损失
loss = 0.8 * action_loss + 0.2 * critic_loss + 0.5 * aux_loss
```

---

## 6. 数据集与数据加载

### 6.1 数据目录结构

```
InternData-N1/vln_n1/traj_data/
├── matterport3d_d435i/              ← scene_dataset_dir
│   ├── 1LXtFkjw3qL/                 ← scene_dir
│   │   ├── meta/
│   │   │   └── pointcloud.ply       ← 障碍物点云 (可选)
│   │   ├── data/chunk-000/
│   │   │   ├── episode_000000.parquet  ← 轨迹数据
│   │   │   └── ...
│   │   └── videos/chunk-000/
│   │       ├── observation.images.rgb/
│   │       │   ├── episode_000000_000.jpg
│   │       │   └── ...
│   │       └── observation.images.depth/
│   │           ├── episode_000000_000.png
│   │           └── ...
```

### 6.2 Parquet 文件内容

每个 `episode_*.parquet` 包含一条完整轨迹：

| 字段 | 形状 | 说明 |
|------|------|------|
| `observation.camera_intrinsic` | (9,) | 相机内参矩阵 (3×3) |
| `observation.camera_extrinsic` | (16,) | 基础外参矩阵 (4×4) |
| `action` | 每行 (16,) | 每帧的相机位姿 (4×4) |

### 6.3 轨迹数据说明

**4×4 变换矩阵**：
```
[[R11, R12, R13, tx],     R = 旋转矩阵 (3×3) → 相机朝向
 [R21, R22, R23, ty],     T = 平移向量 (3×1) → 相机位置
 [R31, R32, R33, tz],
 [0,   0,   0,   1]]
```

- **位置 T**：告诉你"相机在哪"
- **旋转 R**：告诉你"相机看哪个方向"

### 6.4 数据采样参数

| 参数 | 作用 | 示例 |
|------|------|------|
| `scene_scale` | 场景采样比例 | 0.5 = 使用一半场景 |
| `trajectory_data_scale` | 轨迹采样比例 | 0.1 = 每10条取1条 |
| `memory_digit` | 历史帧采样间隔 | 4 = 每隔4帧取1帧 |
| `pred_digit` | 动作采样间隔 | 4 = 每隔4步预测1个动作 |

---

## 7. 关键配置参数

### 7.1 IlCfg (Imitation Learning 配置)

```python
il=IlCfg(
    # 训练参数
    epochs=1000,           # 训练轮数
    batch_size=64,         # 批大小 (建议 A100 用 64-128)
    lr=1e-4,               # 学习率
    num_workers=16,        # 数据加载线程数
    weight_decay=1e-4,     # 权重衰减
    warmup_ratio=0.05,     # 学习率预热比例 (前5%步数逐渐增加lr)

    # 数据参数
    root_dir='data/vln_n1/traj_data',
    image_size=224,
    memory_size=8,         # 历史帧数
    predict_size=24,       # 预测步数
    pixel_channel=4,       # 像素目标通道数

    # 数据增强
    scene_scale=1.0,       # 场景采样比例
    preload=False,         # 是否预加载数据路径
    random_digit=False,    # 是否随机采样间隔
    prior_sample=False,    # 是否使用障碍物加权采样
)
```

### 7.2 warmup_ratio 说明

`warmup_ratio=0.05` 表示训练开始时，学习率从很小的值逐渐增加到目标学习率：

```
学习率
  ↑
  │         ┌────────────────────
  │        /
  │       /  ← warmup 阶段 (前5%步数)
  │      /
  │_____/
  └─────────────────────────────→ 训练步数
```

**为什么需要 Warmup**：
- 稳定初始训练（模型参数随机初始化，梯度方向不稳定）
- 让 Adam 优化器的动量估计有时间准确
- Transformer 模型尤其需要

---

## 8. 训练优化建议

### 8.1 GPU 利用率不足的原因

1. **数据加载瓶颈**：每个样本需要读取 Parquet、加载多张图像
2. **batch_size 偏小**：A100 (80GB) 显存未充分利用
3. **同步操作**：`torch.cuda.synchronize()` 会阻塞

### 8.2 优化方案

| 方案 | 操作 | 效果 |
|------|------|------|
| **增大 batch_size** | 32 → 64-128 | 提高 GPU 利用率 |
| **增加 num_workers** | 8 → 16-24 | 加速数据加载 |
| **启用 AMP** | `bf16=True` | 减少显存，加速计算 |
| **预加载数据路径** | `preload=True` | 减少初始化时间 |

### 8.3 AMP (混合精度训练)

**BF16 的优势**：
- 显存占用减少 ~50%
- A100 Tensor Core 对 BF16 优化，吞吐量是 FP32 的 2-3 倍
- BF16 指数位与 FP32 相同，几乎不需要 loss scaling

**配置方法**：
```python
# TrainingArguments
bf16=True,
tf32=True,  # A100 支持
```

### 8.4 监控命令

```bash
# 查看 GPU 利用率
watch -n 1 nvidia-smi

# 查看详细使用情况
nvidia-smi dmon -s u

# 查看 CPU 和数据加载
htop
```

**诊断**：
- GPU 利用率波动大 (0-100%) → 数据加载瓶颈
- GPU 利用率稳定但低 (如 30%) → batch_size 太小

---

## 9. 推理与部署

### 9.1 推理流程

```python
# 1. 编码条件
vlm_embed = model.goal_compressor(model.vlm_embed_mlp(vlm_tokens))
rgbd_embed = model.rgbd_encoder(images, depths)

# 2. 从纯噪声开始
noisy_action = torch.randn((sample_num, 24, 3))

# 3. 迭代去噪 (20步)
for t in scheduler.timesteps:  # t: 19 → 0
    noise_pred = model.predict_noise(noisy_action, t, vlm_embed, rgbd_embed)
    noisy_action = scheduler.step(noise_pred, t, noisy_action).prev_sample

# 4. 输出 sample_num 条轨迹
trajectories = noisy_action  # (sample_num, 24, 3)
```

### 9.2 Critic 与轨迹选择

**训练时**：Critic 使用 GT 障碍物信息计算轨迹质量分数
**推理时**：没有 GT 信息，依赖训练好的 critic_head 预测

**潜在问题**：
- 分布偏移：生成轨迹可能与训练分布不同
- Sim2Real Gap：仿真数据 vs 真实环境差异

### 9.3 实机部署建议

| 优先级 | 建议 |
|--------|------|
| **高** | 用深度图做实时碰撞检测，比 Critic 更可靠 |
| **高** | 减少 sample_num，降低计算延迟 |
| **中** | Critic 仅作为参考，不完全依赖 |
| **中** | 加入轨迹平滑度约束 |

**推荐的轨迹选择策略**：

```python
def select_trajectory_robust(trajectories, depth_image, critic_head):
    """综合多种信号选择轨迹"""

    # 1. Critic 分数 (可能不准，权重低)
    critic_scores = critic_head(...)  # 权重 0.3

    # 2. 深度碰撞检测 (更可靠，权重高)
    collision_scores = depth_collision_check(trajectories, depth_image)  # 权重 0.5

    # 3. 平滑度
    smoothness_scores = compute_smoothness(trajectories)  # 权重 0.2

    # 加权融合
    final_scores = 0.3 * critic_scores + 0.5 * collision_scores + 0.2 * smoothness_scores

    return trajectories[final_scores.argmax()]
```

---

## 10. 常见问题

### Q1: 为什么有两个轨迹（target 和 augment）？
**A**: 数据增强策略。通过随机旋转 + 三次样条插值生成增强轨迹，提高模型鲁棒性。

### Q2: pixel_channel 为什么是 4？
**A**: 像素目标包含 1 个通道的掩码 + 3 个通道的 RGB 图像 = 4 通道。

### Q3: 点云数据是必须的吗？
**A**: 不是。如果点云不存在，会使用默认的 critic 值 (2.0)，训练仍可正常进行。

### Q4: Pixel Goal 不在图像中怎么办？
**A**: `pixel_mask` 保持全黑，`visible_flag=False`。模型需要学会依赖 Point Goal / Image Goal 导航。

### Q5: 为什么要转换成 XYT 坐标？
**A**: XYT (x, y, theta) 更适合导航任务，theta 表示朝向，有助于模型学习转向。

### Q6: Critic 在推理时准确吗？
**A**: 可能有偏差，因为训练和推理的轨迹分布不同。建议结合深度图碰撞检测使用。

### Q7: 如何只使用 Pixel Goal 训练？
**A**: 三种方案：
1. 训练时将 batch_pg 和 batch_ig 置零
2. 修改模型，删除不需要的编码器
3. 调整 loss 权重，只优化 Pixel Goal 相关损失

---

## 附录

### A. 数据流总结图

```
DataLoader.__getitem__(index)
    ├── 1. process_data_parquet()      → 读取 Parquet (内参、外参、轨迹)
    ├── 2. process_path_points()       → 加载路径点云 (可选)
    ├── 3. process_obstacle_points()   → 加载障碍物点云 (可选)
    ├── 4. 采样起始帧和目标帧          → pixel_start, memory_start, target
    ├── 5. process_memory()            → 加载历史RGB+深度图
    ├── 6. process_actions()           → 生成目标轨迹和增强轨迹
    ├── 7. xyz_to_xyt()                → 转换坐标系
    ├── 8. 计算 critic 值              → 根据障碍物距离
    ├── 9. 处理目标图像                → image_goal
    ├── 10. process_pixel_goal()       → 像素目标投影
    └── 返回 10 个 tensor
         ↓
    navdp_collate_fn()
         ↓
    Model Forward (Diffusion)
         ↓
    Compute Loss
         ↓
    Backward & Update
```

### B. 批量数据格式

```python
batch = {
    "batch_pg": (B, 3),              # Point Goal
    "batch_ig": (B, 224, 224, 6),    # Image Goal
    "batch_tg": (B, 224, 224, 4),    # Pixel Goal
    "batch_rgb": (B, 8, 224, 224, 3),# 历史 RGB
    "batch_depth": (B, 224, 224, 1), # 当前深度
    "batch_labels": (B, 24, 3),      # GT 动作
    "batch_augments": (B, 24, 3),    # 增强动作
    "batch_label_critic": (B,),      # GT Critic
    "batch_augment_critic": (B,),    # 增强 Critic
}
```

### C. Goal 数据的动态生成

训练数据中**只存储轨迹**，三种 Goal 都是在 `__getitem__` 中**动态计算生成**的：

| 数据类型 | 存储方式 | 生成方式 |
|----------|----------|----------|
| **轨迹** | Parquet 文件 | 直接读取 |
| **Point Goal** | ❌ 不存储 | 轨迹终点相对于当前帧的相对坐标 |
| **Pixel Goal** | ❌ 不存储 | 3D 轨迹终点投影到 2D + 画 mask |
| **Image Goal** | ❌ 不存储 | 目标帧图像 + 当前帧图像拼接 |

```python
# Point Goal 生成
target_xyt_actions = xyz_to_xyt(target_local_points, init_vector)
point_goal = target_xyt_actions[-1]  # (3,) 轨迹终点的相对坐标

# Pixel Goal 生成
pixel_goal, pixel_flag = process_pixel_goal(
    image_path,           # 当前帧图像
    target_xyt[-1],       # 目标点坐标
    camera_intrinsic,     # 相机内参
    camera_extrinsic      # 相机外参
)
# 将 3D 目标点投影到 2D 像素坐标，画白色矩形

# Image Goal 生成
image_goal = concatenate(target_frame_image, current_frame_image)
```

---

## 11. 当前 Pixel Goal 的局限性

### 11.1 位置导航 vs 物体导航

**当前 Pixel Goal 标记的是轨迹终点的地面投影，而不是目标物体本身**：

```
实际场景:
                    ┌─────────┐
                    │  沙发    │  ← 真正的目标物体
                    └─────────┘
                         ↑
                         │
    ●───●───●───●───●───●  ← 轨迹终点在沙发前面的地面
    ↑
  当前位置


当前 Pixel Goal 投影结果:
    ┌─────────────────┐
    │                 │
    │    [沙发]       │
    │                 │
    │        ■        │  ← Pixel Goal 在地面位置，不是沙发！
    │                 │
    └─────────────────┘
```

### 11.2 适用场景

| 场景 | 当前设计是否适用 |
|------|------------------|
| 走到某个**位置** | ✅ 适用 |
| 走到某个**物体旁边** | ⚠️ 可以工作，但不直观 |
| 走向**看到的物体** | ❌ 无法直接表达 |

### 11.3 如果要做"走向目标物体"

需要**真正的目标物体检测 + 像素标注**：

| 方案 | 说明 |
|------|------|
| **检测模型** | 输入图像 → 检测物体 → 框住物体像素 → Pixel Goal |
| **人工标注** | 数据集中存储目标物体的 bounding box / mask |
| **VLM 指向** | "去沙发" → VLM 输出沙发的像素位置 → Pixel Goal |

---

## 12. VLM + Diffusion 物体导航方案

### 12.1 目标架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    部署时流程                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  用户指令: "去沙发"                                              │
│       ↓                                                          │
│  VLM 理解 + 输出沙发像素位置 (bounding box / point)             │
│       ↓                                                          │
│  生成物体级 Pixel Goal mask                                     │
│       ↓                                                          │
│  NavDP (Diffusion) 生成轨迹                                     │
│       ↓                                                          │
│  机器人执行                                                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 12.2 数据集构建方案

**推荐：基于开源数据集改造 + 少量真实数据微调**

| 方案 | 优点 | 缺点 |
|------|------|------|
| **真实世界采集** | 数据真实、无 Sim2Real Gap | 成本高、多样性有限 |
| **开源数据集改造** | 规模大、场景多样、成本低 | 可能有 domain gap |

### 12.3 推荐的开源数据集

| 数据集 | 特点 | 推荐度 |
|--------|------|--------|
| **HM3D** | 大规模室内、有深度、轨迹 | ⭐⭐⭐ |
| **MP3D (Matterport)** | 真实扫描、多样场景 | ⭐⭐⭐ |
| **Habitat ObjectNav** | 专门的物体导航数据 | ⭐⭐⭐ |
| **Gibson** | 真实扫描、导航任务 | ⭐⭐ |
| **RoboTHOR** | 交互场景、物体丰富 | ⭐⭐ |

### 12.4 数据集改造流程

```
原始数据 (轨迹 + RGB-D)
        ↓
┌───────────────────────────────────┐
│  1. 用目标检测/分割模型标注物体    │
│     (Grounding DINO / SAM / YOLO) │
└───────────────────────────────────┘
        ↓
┌───────────────────────────────────┐
│  2. 生成物体级 Pixel Goal         │
│     - 检测目标物体                 │
│     - 生成 bounding box / mask    │
│     - 替换原来的地面投影 mask      │
└───────────────────────────────────┘
        ↓
┌───────────────────────────────────┐
│  3. 关联轨迹与目标物体             │
│     - 轨迹终点附近有什么物体？     │
│     - 生成 (轨迹, 目标物体) 对     │
└───────────────────────────────────┘
        ↓
新数据集 (轨迹 + 物体 Pixel Goal)
```

### 12.5 实现伪代码

```python
def create_object_pixel_goal_dataset(original_dataset):
    """将位置导航数据改造为物体导航数据"""

    # 1. 加载检测模型
    detector = load_grounding_dino()  # 或 YOLO、SAM

    for episode in original_dataset:
        # 2. 获取轨迹终点附近的图像
        target_image = episode.images[target_frame]

        # 3. 检测图像中的物体
        objects = detector.detect(target_image)
        # objects = [{"class": "sofa", "bbox": [x1,y1,x2,y2]}, ...]

        # 4. 选择一个目标物体（可以随机或按规则）
        target_object = select_target_object(objects)

        # 5. 在当前帧中定位该物体
        for frame_idx in range(len(episode)):
            current_image = episode.images[frame_idx]

            # 用跟踪或重检测找到同一物体
            pixel_goal_mask = detect_object_in_frame(
                current_image,
                target_object
            )

            # 6. 保存新的 Pixel Goal
            save_pixel_goal(episode_id, frame_idx, pixel_goal_mask)
```

### 12.6 推荐的工具链

| 任务 | 推荐工具 |
|------|----------|
| **开放词汇检测** | Grounding DINO |
| **分割** | SAM (Segment Anything) |
| **跟踪** | SAM 2 / XMem |
| **仿真平台** | Habitat-sim / Isaac Sim |

### 12.7 数据配比建议

```
训练数据组成：
├── 80% 开源数据集改造
│   ├── HM3D + 物体标注
│   ├── MP3D + 物体标注
│   └── Habitat ObjectNav
│
└── 20% 真实数据（微调用）
    ├── 实际部署场景
    └── 覆盖 corner cases
```

### 12.8 分阶段建议

| 阶段 | 建议 |
|------|------|
| **初期** | 用开源数据 + 自动标注，快速验证想法 |
| **中期** | 在仿真中测试，迭代改进 |
| **后期** | 采集少量真实数据微调，解决 domain gap |

---

**文档版本**: 2.2
**最后更新**: 2026-02-05
**作者**: AI Assistant
