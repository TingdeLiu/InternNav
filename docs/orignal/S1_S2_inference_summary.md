# InternVLA-N1: S1/S2 双系统分离训练与联合推理总结

## 1. 概述

InternVLA-N1 采用**双系统（Dual-System）**架构进行视觉语言导航：

| 系统 | 角色 | 核心技术 | 输出 |
|------|------|----------|------|
| **S2 (System 2)** | 高层规划与决策 | Qwen2.5-VL 多模态大模型 | 像素目标 / 离散动作 / 轨迹条件向量 |
| **S1 (System 1)** | 低层运动控制 | 扩散策略（NextDiT 或 NavDP） | 连续轨迹 → 离散动作序列 |

两个系统**独立训练、独立运行**，在推理时通过**条件传递**协同工作。

---

## 2. S2 系统详解

### 2.1 功能

S2 负责**理解自然语言指令 + 视觉观测**，做出高层导航决策：
- 输入：当前 RGB 图像 + 历史 RGB 图像 + 文本指令
- 输出（三选一）：
  1. **离散动作**：STOP / ↑ / ← / → / ↓（直接可执行，不需要 S1）
  2. **像素目标** `[x, y]`：图像上的下一个航路点坐标
  3. **轨迹条件向量** `traj_latents`：传递给 S1 的上下文嵌入

### 2.2 架构

```
文本指令 ──→ Tokenizer ──→ embed_tokens ──→ ┐
                                              ├──→ Qwen2.5-VL Transformer ──→ generate()输出文本
当前+历史RGB ──→ ViT 视觉编码 ──→ 替换 IMAGE token ──→ ┘
                                                          │
                                                    解析输出文本
                                                    ├─ 含数字 → 像素目标 [x,y]
                                                    └─ 含方向符号 → 离散动作
                                                          │
                                              (若为像素目标) │
                                                          ▼
                                              generate_latents()
                                              追加 latent_queries 到序列尾部
                                              ──→ Transformer 前向 ──→ 截取末尾 n_query 隐藏状态
                                              ──→ 输出 traj_latents
```

关键组件（`internvla_n1.py`）：
- **latent_queries**：可学习参数 `(1, n_query, hidden_size)`，训练时学会从图像和文本中提取轨迹相关信息
- **generate_latents()**：推理时将 latent_queries 拼到序列末尾，经 Transformer 全场注意力后提取末尾隐藏状态

### 2.3 独立训练

训练入口：`internnav/trainer/internvla_n1_trainer.py`

```
训练数据：LeRobot 格式数据集（RGB + 3D轨迹标签）
      │
      ▼
前向传播：
  1. 文本+图像 → Qwen2.5-VL → hidden_states
  2. 从 hidden_states 中提取轨迹查询位置的 traj_hidden_states
  3. traj_hidden_states 作为条件，配合真实轨迹 traj_poses 计算扩散损失
      │
      ▼
损失函数（取决于 S1 类型）：
  - NextDiT 分支：Flow Matching MSE Loss
      noise_pred = traj_dit(noisy_trajectory, timestep, condition=traj_hidden_states)
      target = noise - relative_poses
      loss = MSE(noise_pred, target)  # 带 loss_mask 忽略 padding 帧

  - NavDP 分支：DDPM MSE Loss
      pred_pg, noise = navdp.forward_vlm_traj(traj_hidden_states, images, depths, labels)
      loss = MSE(pred_pg, noise)  # 带 loss_mask
```

可训练参数控制：
```python
# S2 端可训练模块
modules = ['latent_queries', 'action_encoder', 'action_decoder',
           'traj_dit', 'cond_projector']
# 异步模式额外训练
modules += ['memory_encoder', 'rgb_resampler', 'rgb_model']
```

---

## 3. S1 系统详解

### 3.1 功能

S1 负责**精细运动规划**，从 S2 的条件向量生成可执行的轨迹：
- 输入：S2 输出的 `traj_latents` + 当前 RGB(-D) 观测
- 输出：离散动作序列 `[a₁, a₂, a₃, a₄]`（取前 4 步）

### 3.2 两种 S1 实现

#### 方案 A：NextDiT（仅 RGB）

```
traj_latents ──→ cond_projector ──→ ┐
                                     ├──→ 拼接为 hidden_states
RGB图像对 ──→ DAv2编码 ──→ MemoryEncoder ──→ QFormer ──→ memory_tokens ──→ ┘
                                                                           │
                                                                           ▼
                                                            Diffusion 采样循环（10步）：
                                                            for t in timesteps:
                                                              noise_pred = traj_dit(latent, t, hidden_states)
                                                              latent = scheduler.step(noise_pred, t, latent)
                                                                           │
                                                                           ▼
                                                            输出：连续轨迹坐标 → traj_to_actions() → 离散动作
```

- 使用 `FlowMatchEulerDiscreteScheduler`
- 支持 Classifier-Free Guidance（无条件 + 有条件分支插值）
- 采样 `num_sample_trajs=32` 条轨迹，随机选一条

#### 方案 B：NavDP（RGB-D）

```
traj_latents ──→ ┐
                  ├──→ navdp.predict_pointgoal_action_async()
RGB + Depth ──→ DAT_RGBD编码 ──→ ┘
                                  │
                                  ▼
                    DDPM 去噪循环（20步）：
                    TransformerDecoder（16层）
                    ──→ action_head ──→ 预测轨迹
                    ──→ critic_head ──→ 轨迹质量评分
                                  │
                                  ▼
                    输出：多条轨迹 → chunk_token() → 离散动作
```

- 使用 `DDPMScheduler`（20 timesteps）
- 含 Critic Head 用于轨迹质量评估
- RGBD 编码器基于 DAT backbone

### 3.3 独立训练

训练入口：`internnav/trainer/navdp_trainer.py`

```
训练数据：LeRobot 格式数据集（RGB + Depth + 动作标签）
      │
      ▼
前向传播：
  NavDP 模型接收：
  - point_goal / image_goal / pixel_goal（多种目标表示）
  - RGB + Depth 图像对
  - 真实动作标签
      │
      ▼
损失函数：
  pred_ng, pred_mg, critic_pred, noise = navdp(goals, rgb, depth, labels)

  ng_action_loss = MSE(pred_ng, noise)    # 无目标动作损失
  mg_action_loss = MSE(pred_mg, noise)    # 有目标动作损失
  action_loss = 0.5 * mg_action_loss + 0.5 * ng_action_loss
  critic_loss = MSE(critic_pred, label_critic)
  aux_loss = 辅助预测损失

  total_loss = 0.8 * action_loss + 0.2 * critic_loss + 0.5 * aux_loss
```

---

## 4. 推理流程：S1 + S2 如何协同

### 4.1 整体流程

```
┌──────────────────────────────────────────────────────────────────┐
│                    每一个时间步 (Agent Step)                       │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ 需要调用 S2 吗？    │ ← 取决于上一轮 S2 的动作是否用完
                    └────────┬──────────┘
                             │
                    ┌────────┴────────┐
                    │ 是               │ 否（还有缓存动作）
                    ▼                 ▼
           ┌────────────────┐   ┌─────────────┐
           │ S2 推理         │   │ 执行缓存动作  │
           │ s2_step()      │   │ 从 output_   │
           │                │   │ action 队列   │
           │ 输入：RGB+指令  │   │ 弹出下一个    │
           │ 输出：S2Output │   └──────┬──────┘
           └───────┬────────┘          │
                   │                   │
          ┌────────┴────────┐          │
          │                 │          │
     离散动作？         像素目标？       │
          │                 │          │
          ▼                 ▼          │
   ┌────────────┐  ┌──────────────┐   │
   │ 直接执行     │  │ generate_    │   │
   │ 不需要 S1   │  │ latents()   │   │
   │             │  │ 生成条件向量  │   │
   └─────┬──────┘  └──────┬───────┘   │
         │                │           │
         │                ▼           │
         │       ┌──────────────┐     │
         │       │ S1 推理       │     │
         │       │ s1_step_     │     │
         │       │ latent()     │     │
         │       │              │     │
         │       │ generate_    │     │
         │       │ traj()       │     │
         │       │ → 扩散采样    │     │
         │       │ → 动作序列    │     │
         │       └──────┬───────┘     │
         │              │             │
         │      ┌───────┴───────┐     │
         │      │ 取第1个动作执行 │     │
         │      │ 其余缓存到     │     │
         │      │ output_action │     │
         │      └───────┬───────┘     │
         │              │             │
         └──────────────┼─────────────┘
                        │
                        ▼
              ┌──────────────────┐
              │ 返回动作给环境执行  │
              │ action → env.step │
              └──────────────────┘
```

### 4.2 关键代码路径

**`internvla_n1_policy.py`**

```
s2_step():                          # S2 推理入口
  ├─ 图像预处理 + 历史帧管理
  ├─ 构建对话模板（多轮对话支持 look_down）
  ├─ processor 编码 → model.generate()
  ├─ 解析输出文本：
  │   ├─ 含坐标数字 → pixel_goal → generate_latents() → output_latent
  │   └─ 含方向符号 → parse_actions() → output_action
  └─ 返回 S2Output

s1_step_latent():                   # S1 推理入口
  ├─ model.generate_traj(traj_latents, images, depths)
  │   ├─ NextDiT: 扩散采样 10 步 + CFG
  │   └─ NavDP:   DDPM 去噪 20 步
  ├─ 轨迹 → 离散动作转换
  └─ 返回 S1Output(idx=前4步动作)
```

### 4.3 运行模式

| 模式 | S2 频率 | S1 频率 | 说明 |
|------|---------|---------|------|
| **Sync（同步）** | 每帧 | 每帧 | S2→S1 串行执行 |
| **Partial Async（部分异步）** | 每 N 帧 | 每帧 | S2 在独立线程运行，S1 复用 S2 的缓存结果 |

异步模式下，S2 的 `output_action` 队列允许 S1 在 S2 规划下一步时继续执行动作，提升实时性。

---

## 5. 训练分离的关键设计

### 5.1 为什么可以分开训练

```
S2 训练目标：学会理解指令 + 视觉场景 → 输出有意义的条件向量
             │
             └─ latent_queries 学会"问对的问题"，从图文中提取轨迹规划所需信息

S1 训练目标：给定条件向量/像素目标 → 生成平滑的可执行轨迹
             │
             └─ 扩散模型学会将噪声去噪为合理的运动轨迹
```

### 5.2 连接桥梁

两个系统通过以下接口连接：

```
S2 → S1 的数据传递：
  traj_latents: torch.Tensor  # shape: (1, n_query, 3584)

  这是 S2 的 latent_queries 经过 Transformer 全场注意力后
  吸收了图像特征和文本指令信息的"浓缩条件向量"

  S1 通过 cond_projector 将其映射到自己的条件空间：
  3584 → LatentEmbSize(768) → 768
```

### 5.3 联合训练时的梯度流

虽然 S1 和 S2 可以独立训练，但在端到端训练时：
```
损失 = S1扩散损失(noise_pred vs target)
         │
梯度回传路径：
  traj_dit / navdp ← cond_projector ← traj_hidden_states ← Transformer ← latent_queries
                                                                          ← 视觉编码器
                                                                          ← 文本嵌入
```
即 S1 的损失可以驱动 S2 的 latent_queries 和条件投影器一起优化。

---

## 6. 一句话总结

> **S2（Qwen2.5-VL）看指令、看场景、做决策；S1（扩散策略）接收 S2 的条件向量，生成精细轨迹。两者独立训练各自能力，推理时串联协作：S2 输出像素目标 → 生成 latent 条件 → S1 扩散采样生成动作序列 → 逐步执行。**
