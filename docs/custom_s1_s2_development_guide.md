# InternNav 自定义 S1/S2 系统开发指南

> 本文档面向希望基于 InternNav 框架开发自定义 System 1（低层运动控制）或 System 2（高层规划决策）系统的开发者，提供完整的架构说明、接口规范与实操步骤。

---

## 目录

- [1. 双系统架构概览](#1-双系统架构概览)
- [2. 核心数据结构与接口](#2-核心数据结构与接口)
- [3. 添加自定义 S1 系统](#3-添加自定义-s1-系统)
- [4. 添加自定义 S2 系统](#4-添加自定义-s2-系统)
- [5. 训练流程集成](#5-训练流程集成)
- [6. 评估与部署](#6-评估与部署)
- [7. 端到端示例：从零接入一个新 S1](#7-端到端示例从零接入一个新-s1)
- [8. 常见问题与注意事项](#8-常见问题与注意事项)

---

## 1. 双系统架构概览

InternNav 的 InternVLA-N1 采用 **双系统（Dual-System）** 架构进行视觉语言导航：

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Agent (调度层)                               │
│                  internvla_n1_agent.py                               │
│                                                                     │
│  ┌──────────────────────┐        ┌──────────────────────────────┐   │
│  │    System 2 (S2)     │        │      System 1 (S1)           │   │
│  │  高层规划与决策        │──条件──→│  低层运动控制                 │   │
│  │                      │  传递  │                              │   │
│  │  • 理解语言指令        │        │  • 接收条件向量               │   │
│  │  • 分析视觉场景        │        │  • 生成精细轨迹               │   │
│  │  • 输出像素目标/动作    │        │  • 输出离散动作序列            │   │
│  └──────────────────────┘        └──────────────────────────────┘   │
│                                                                     │
│  两个系统独立训练，推理时通过条件向量 (traj_latents) 协同工作           │
└─────────────────────────────────────────────────────────────────────┘
```

### 各系统职责

| 系统 | 角色 | 输入 | 输出 | 现有实现 |
|------|------|------|------|----------|
| **S2** | 高层规划 | RGB图像 + 历史帧 + 文本指令 | 像素目标 / 离散动作 / 条件向量 `traj_latents` | Qwen2.5-VL 多模态大模型 |
| **S1** | 低层控制 | `traj_latents` + 当前 RGB(-D) | 离散动作序列 `[a₁, a₂, …]` | NextDiT 扩散策略 / NavDP |

### 运行模式

| 模式 | S2 频率 | S1 频率 | 适用场景 |
|------|---------|---------|----------|
| **sync** | 每帧 | 每帧 | 精度优先，S2→S1 串行 |
| **partial_async** | 每 N 帧 | 每帧 | 实时性优先，S2 在独立线程运行 |

---

## 2. 核心数据结构与接口

### 2.1 S1/S2 之间的数据协议

所有 S1/S2 通信通过以下数据类完成（`internnav/model/utils/vln_utils.py`）：

```python
@dataclass
class S2Input:
    idx: Optional[int] = -1           # 当前时间步
    instruction: Optional[str] = None  # 文本导航指令
    rgb: Optional[np.ndarray] = None   # 当前 RGB 图像
    depth: Optional[np.ndarray] = None # 当前深度图像
    pose: Optional[Tuple] = None       # 机器人位姿
    look_down: Optional[bool] = False  # 是否低头
    should_infer: Optional[bool] = False  # 是否触发推理

@dataclass
class S2Output:
    idx: Optional[int] = -1
    is_infering: Optional[bool] = False
    output_action: Optional[np.ndarray] = None    # 离散动作列表（直接执行，跳过S1）
    output_pixel: Optional[np.ndarray] = None     # 像素目标 [x, y]
    output_latent: Optional[torch.Tensor] = None  # 条件向量（传给S1）
    rgb_memory: Optional[np.ndarray] = None
    depth_memory: Optional[np.ndarray] = None

@dataclass
class S1Input:
    pixel_goal: Optional[np.ndarray] = None
    latent: Optional[np.ndarray] = None
    rgb: Optional[np.ndarray] = None
    depth: Optional[np.ndarray] = None

@dataclass
class S1Output:
    idx: Optional[list] = None              # 离散动作序列，如 [1, 1, 2, 3]
    trajectory: Optional[np.ndarray] = None # 连续轨迹
    linear_velocity: Optional[float] = None
    angular_velocity: Optional[float] = None
    vis_image: Optional[np.ndarray] = None  # 可视化图像
```

### 2.2 S2 → S1 条件传递的关键桥梁

```
traj_latents: torch.Tensor  # shape: (1, n_query, hidden_size)
                             # 默认 hidden_size=3584 (Qwen2.5-VL 7B)

S1 通过 cond_projector 将其映射到自身条件空间：
  3584 → LatentEmbSize (768) → 扩散模型条件维度
```

### 2.3 Policy 接口（`internvla_n1_policy.py`）

任何双系统 Policy 需要实现以下三个核心方法：

```python
class YourPolicy(PreTrainedModel):
    def reset(self):
        """重置历史状态"""
        ...

    def s2_step(self, rgb, depth, pose, instruction, intrinsic, look_down=False) -> S2Output:
        """S2 推理：理解指令+场景，输出高层决策"""
        ...

    def s1_step_latent(self, rgb, depth, latent) -> S1Output:
        """S1 推理：基于 S2 条件向量，生成低层动作"""
        ...
```

---

## 3. 添加自定义 S1 系统

### 3.1 适用场景

你希望替换底层运动控制模块（例如用 MPC、强化学习策略或其他扩散模型替代 NextDiT/NavDP），同时保留 InternVLA-N1 的 S2 高层规划能力。

### 3.2 开发步骤

#### Step 1：创建 S1 模型文件

在 `internnav/model/basemodel/internvla_n1/` 下创建你的 S1 模型：

```python
# internnav/model/basemodel/internvla_n1/my_s1_model.py

import torch
import torch.nn as nn

class MyS1Model(nn.Module):
    """自定义 S1 模型：从条件向量生成轨迹/动作"""

    def __init__(self, cond_dim=768, action_dim=3, horizon=32):
        super().__init__()
        self.cond_dim = cond_dim
        self.action_dim = action_dim
        self.horizon = horizon

        # 示例：一个简单的 MLP 策略网络
        self.policy_net = nn.Sequential(
            nn.Linear(cond_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, horizon * action_dim),
        )

    def forward(self, condition, images=None, depths=None):
        """
        训练时的前向传播
        Args:
            condition: traj_latents from S2, shape (B, n_query, hidden_size)
            images: 当前观测图像 (可选)
            depths: 当前深度图像 (可选)
        Returns:
            predicted_actions: shape (B, horizon, action_dim)
        """
        # 池化条件向量
        cond = condition.mean(dim=1)  # (B, hidden_size) → 需先经过 cond_projector 到 cond_dim
        actions = self.policy_net(cond)
        return actions.view(-1, self.horizon, self.action_dim)

    def predict(self, condition, images=None, depths=None):
        """
        推理时生成动作
        Returns:
            actions: shape (num_samples, horizon, action_dim)
        """
        with torch.no_grad():
            return self.forward(condition, images, depths)

    def load_model(self):
        """加载预训练权重（可选）"""
        pass
```

#### Step 2：在 `internvla_n1_arch.py` 中注册构建函数

在 `internnav/model/basemodel/internvla_n1/internvla_n1_arch.py` 中添加：

```python
def build_my_s1(config, memory_size=None):
    from .my_s1_model import MyS1Model
    model = MyS1Model(cond_dim=LatentEmbSize)  # LatentEmbSize=768
    return model
```

#### Step 3：在 `InternVLAN1MetaModel` 中集成

找到 `internvla_n1_arch.py` 中的 `InternVLAN1MetaModel.initialize_vision_modules()` 方法，添加你的 S1 初始化分支：

```python
def initialize_vision_modules(self, model_args):
    ...
    system1 = getattr(model_args, 'system1', 'nextdit')

    if 'my_s1' in system1:
        self.my_s1 = build_my_s1(model_args)
        self.cond_projector = nn.Linear(self.config.hidden_size, LatentEmbSize)
    elif 'nextdit' in system1:
        ...  # 现有逻辑
    elif 'navdp' in system1:
        ...  # 现有逻辑
```

#### Step 4：实现 `generate_traj()` 分支

在 `internvla_n1.py` 的 `InternVLAN1ForCausalLM.generate_traj()` 方法中添加你的推理逻辑：

```python
def generate_traj(self, traj_latents, images_dp=None, depths_dp=None):
    model = self.get_model()

    if hasattr(model, 'my_s1'):
        # 条件投影
        condition = model.cond_projector(traj_latents)  # (1, n_query, 768)
        # 生成动作
        actions = model.my_s1.predict(condition, images_dp, depths_dp)
        return actions

    elif hasattr(model, 'traj_dit'):
        ...  # 现有 NextDiT 逻辑
    elif hasattr(model, 'navdp'):
        ...  # 现有 NavDP 逻辑
```

#### Step 5：配置模型参数

在训练脚本中将 `--system1` 设置为你的 S1 标识：

```bash
# scripts/train/qwenvl_train/train_my_s1.sh
system1=my_s1

srun torchrun ... \
    internnav/trainer/internvla_n1_trainer.py \
    --system1 ${system1} \
    --model_name_or_path "checkpoints/InternVLA-N1-System2" \
    ...
```

#### Step 6：设置可训练参数

在 `internnav/trainer/internvla_n1_trainer.py` 的 `set_model()` 函数中添加：

```python
def set_model(model_args, model):
    ...
    if 'my_s1' in model_args.system1:
        # 冻结 S2，只训练 S1 相关模块
        for n, p in model.model.my_s1.named_parameters():
            p.requires_grad = True
        model.model.cond_projector.requires_grad = True
        model.model.latent_queries.requires_grad = True
    elif 'nextdit' in model_args.system1:
        ...  # 现有逻辑
```

### 3.3 S1 接口规范总结

| 项目 | 要求 |
|------|------|
| **输入** | `traj_latents: Tensor (1, n_query, hidden_size)` + 可选的 RGB/Depth |
| **输出** | 动作张量，需可通过 `traj_to_actions()` 或 `chunk_token()` 转为离散动作列表 |
| **注册位置** | `internvla_n1_arch.py` 的 `build_*` 函数 + `initialize_vision_modules()` |
| **推理入口** | `internvla_n1.py` 的 `generate_traj()` |
| **训练入口** | `internvla_n1.py` 的 `forward()` 中的损失计算分支 |

---

## 4. 添加自定义 S2 系统

### 4.1 适用场景

你希望替换高层规划模块（例如用不同的 VLM、基于 LLM 的规划器或传统SLAM规划器替代 Qwen2.5-VL）。

### 4.2 方案选择

根据你的需求，有两种集成方式：

| 方案 | 修改程度 | 说明 |
|------|---------|------|
| **方案 A: 替换 Policy 中的 S2** | 中等 | 修改 `internvla_n1_policy.py` 的 `s2_step()` |
| **方案 B: 创建全新 Agent + Policy** | 完整 | 新建 Agent 类、Policy 类和训练器 |

### 4.3 方案 A：替换 Policy 中的 S2 逻辑

#### Step 1：创建自定义 S2 模型

```python
# internnav/model/basemodel/my_s2/my_s2_model.py

import torch
import torch.nn as nn
import numpy as np

class MyS2Model(nn.Module):
    """自定义 S2：高层视觉-语言规划器"""

    def __init__(self, hidden_size=768, n_query=32):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_query = n_query

        # 示例：简单的视觉-语言融合模块
        self.vision_encoder = ...   # 你的视觉编码器
        self.language_encoder = ... # 你的语言编码器
        self.fusion = ...           # 多模态融合模块

        # 输出头
        self.action_head = nn.Linear(hidden_size, 6)  # STOP/↑/←/→/↓/look_down
        self.pixel_head = nn.Linear(hidden_size, 2)   # 像素坐标 (x, y)
        self.latent_queries = nn.Parameter(torch.randn(1, n_query, hidden_size))

    def forward(self, images, instruction, history_images=None):
        """
        Args:
            images: 当前 + 历史 RGB 图像
            instruction: 文本导航指令
        Returns:
            decision_type: 'action' 或 'pixel'
            action_logits: 离散动作概率 (如果是 action)
            pixel_coords: 像素坐标 (如果是 pixel)
            traj_latents: 条件向量给 S1 (如果是 pixel)
        """
        ...

    def generate_decision(self, images, instruction, history_images=None):
        """推理时的决策生成"""
        ...
        # 返回 S2Output 所需的字段
```

#### Step 2：创建自定义 Policy

```python
# internnav/model/basemodel/my_dual_system/my_policy.py

from transformers import PreTrainedModel
from internnav.model.utils.vln_utils import S1Output, S2Output

class MyDualSystemPolicy(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.s2_model = ...  # 你的 S2 模型
        self.s1_model = ...  # 可复用现有 S1 (NextDiT/NavDP) 或自定义

        self.rgb_list = []
        self.episode_idx = 0

    def reset(self):
        self.rgb_list = []
        self.episode_idx = 0

    def s2_step(self, rgb, depth, pose, instruction, intrinsic, look_down=False) -> S2Output:
        """实现 S2 推理逻辑"""
        output = S2Output()

        # 调用你的 S2 模型
        result = self.s2_model.generate_decision(rgb, instruction, self.rgb_list)

        if result['type'] == 'action':
            output.output_action = result['actions']   # 离散动作列表
        elif result['type'] == 'pixel':
            output.output_pixel = result['pixel_goal']  # [x, y]
            output.output_latent = result['traj_latents']  # Tensor

        self.rgb_list.append(rgb)
        self.episode_idx += 1
        return output

    def s1_step_latent(self, rgb, depth, latent) -> S1Output:
        """实现 S1 推理逻辑（可复用现有实现）"""
        actions = self.s1_model.predict(latent, rgb, depth)
        return S1Output(idx=actions[:4])

    def step_no_infer(self, rgb, depth, pose):
        """非推理帧：仅更新历史"""
        self.rgb_list.append(rgb)
        self.episode_idx += 1
```

#### Step 3：注册到框架

**注册 Policy：** 在 `internnav/model/__init__.py` 中添加：

```python
def get_policy(policy_name):
    ...
    elif policy_name == 'MyDualSystem_Policy':
        from .basemodel.my_dual_system.my_policy import MyDualSystemPolicy
        return MyDualSystemPolicy
    ...

def get_config(policy_name):
    ...
    elif policy_name == 'MyDualSystem_Policy':
        from .basemodel.my_dual_system.my_policy import MyDualSystemConfig
        return MyDualSystemConfig
    ...
```

**注册 Agent：** 如果需要自定义调度逻辑，在 `internnav/agent/` 下创建新 Agent：

```python
# internnav/agent/my_dual_agent.py

from internnav.agent.base import Agent
from internnav.configs.agent import AgentCfg

@Agent.register('my_dual_system')
class MyDualAgent(Agent):
    def __init__(self, config: AgentCfg):
        super().__init__(config)
        # 初始化你的 Policy
        ...

    def step(self, obs):
        # 实现调度逻辑：何时调用 S2，何时使用 S1 缓存
        ...

    def reset(self):
        self.policy.reset()
```

### 4.4 方案 B：完全独立的 S2

如果你的 S2 不是基于 VLM（例如是基于规则的路径规划器），可以更轻量地接入：

```python
# internnav/agent/rule_based_s2_agent.py

from internnav.agent.base import Agent
from internnav.model.utils.vln_utils import S2Output

@Agent.register('rule_s2_with_navdp')
class RuleS2WithNavDPAgent(Agent):
    """规则 S2 + 学习型 S1 的组合"""

    def __init__(self, config):
        super().__init__(config)
        # 初始化 S1 (例如独立的 NavDP)
        self.s1_policy = self._load_navdp(config)
        # S2 使用规则/传统方法
        self.planner = YourTraditionalPlanner()

    def step(self, obs):
        obs = obs[0]
        rgb, depth, instruction = obs['rgb'], obs['depth'], obs['instruction']

        # S2: 规则/传统方法生成子目标
        subgoal = self.planner.plan(rgb, depth, instruction)

        if subgoal['type'] == 'pixel':
            # 将子目标编码为 S1 可接受的条件
            latent = self._encode_subgoal(subgoal)
            # S1: 神经网络生成精细动作
            actions = self.s1_policy.predict(latent, rgb, depth)
            return [{'action': [actions[0]], 'ideal_flag': True}]
        else:
            return [{'action': subgoal['action'], 'ideal_flag': True}]

    def reset(self):
        self.s1_policy.reset()
        self.planner.reset()
```

### 4.5 S2 接口规范总结

| 项目 | 要求 |
|------|------|
| **输入** | RGB + 历史帧 + 文本指令 + 可选深度/位姿 |
| **输出** | `S2Output`，至少填充 `output_action` 或 (`output_pixel` + `output_latent`) |
| **注册位置** | `internnav/model/__init__.py` (Policy) + `internnav/agent/` (Agent) |
| **调度逻辑** | Agent 中的 `should_infer_s2()` 决定何时触发 S2 |

---

## 5. 训练流程集成

### 5.1 独立训练 S1（不涉及 S2）

适用于纯运动控制模型，例如 NavDP 的独立训练：

```bash
# 使用现有基础训练入口
torchrun --nproc_per_node=8 \
    scripts/train/base_train/train.py \
    --model_name=navdp \
    --name=my_navdp_experiment
```

需要的配置文件在 `scripts/train/base_train/configs/navdp.py`。

### 5.2 独立训练 S2（不涉及 S1）

训练纯 S2 模型（VLM 学会理解指令并输出像素目标/离散动作）：

```bash
# system1=none 表示不联合训练 S1
torchrun --nproc_per_node=8 \
    internnav/trainer/internvla_n1_trainer.py \
    --model_name_or_path "Qwen/Qwen2.5-VL-7B-Instruct" \
    --system1 "none" \
    --tune_mm_vision True \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --pixel_goal_only False \
    ...
```

### 5.3 联合训练 S1 + S2（Dual System）

以预训练好的 S2 为起点，联合微调 S1 和 S2 的连接层：

```bash
torchrun --nproc_per_node=8 \
    internnav/trainer/internvla_n1_trainer.py \
    --model_name_or_path "checkpoints/InternVLA-N1-System2" \
    --system1 "my_s1" \           # 你的自定义 S1 标识
    --tune_mm_vision False \       # 冻结视觉编码器
    --tune_mm_mlp False \          # 冻结 MLP
    --tune_mm_llm False \          # 冻结 LLM
    --pixel_goal_only True \       # 仅训练像素目标分支
    ...
```

联合训练时的梯度流：

```
S1 损失 (MSE)
    │
    ▼ 梯度回传
S1 模型 ← cond_projector ← traj_hidden_states ← Transformer ← latent_queries
                                                                ← 视觉编码器 (可选冻结)
                                                                ← 文本嵌入 (可选冻结)
```

### 5.4 自定义 S1 的训练集成 Checklist

- [ ] 在 `internvla_n1_arch.py` 添加 `build_my_s1()` 构建函数
- [ ] 在 `internvla_n1_arch.py` 的 `initialize_vision_modules()` 添加初始化分支
- [ ] 在 `internvla_n1.py` 的 `forward()` 中添加训练损失计算逻辑
- [ ] 在 `internvla_n1.py` 的 `generate_traj()` 中添加推理分支
- [ ] 在 `internvla_n1_trainer.py` 的 `set_model()` 中设置可训练参数
- [ ] 在 `internvla_n1_argument.py` 中确保 `system1` 参数包含你的选项

### 5.5 数据集要求

| 系统 | 数据格式 | 关键字段 |
|------|----------|----------|
| S1 单独训练 | LeRobot / 自定义 | RGB, Depth, 动作标签 (point_goal/image_goal/pixel_goal) |
| S2 单独训练 | LeRobot VLN | RGB, 文本指令, 3D轨迹标签, 像素目标标签 |
| 联合训练 | LeRobot VLN | RGB, 文本指令, 3D轨迹标签, 像素目标标签 |

---

## 6. 评估与部署

### 6.1 创建评估配置

在 `scripts/eval/configs/` 下创建配置文件：

```python
# scripts/eval/configs/habitat_my_dual_system_cfg.py

from internnav.configs.agent import AgentCfg
from internnav.configs.evaluator import EnvCfg, EvalCfg

eval_cfg = EvalCfg(
    agent=AgentCfg(
        model_name='internvla_n1',  # 或你自定义的 agent 注册名
        model_settings={
            "mode": "dual_system",
            "model_path": "checkpoints/MyDualSystem",
            "system1": "my_s1",       # 你的 S1 标识
            "num_history": 8,
            "resize_w": 384,
            "resize_h": 384,
            "max_new_tokens": 1024,
        },
    ),
    env=EnvCfg(
        env_type='habitat',
        env_settings={
            'config_path': 'scripts/eval/configs/vln_r2r.yaml',
        },
    ),
    eval_type='habitat_vln',
    eval_settings={
        "output_path": "./logs/habitat/my_dual_system",
        "save_video": False,
        "max_steps_per_episode": 500,
    },
)
```

### 6.2 运行评估

```bash
python scripts/eval/eval.py --config scripts/eval/configs/habitat_my_dual_system_cfg.py
```

### 6.3 InternUtopia 环境评估

```python
# scripts/eval/configs/h1_my_system_cfg.py
eval_cfg = EvalCfg(
    agent=AgentCfg(
        model_name='internvla_n1',
        model_settings={
            ...
            'infer_mode': 'partial_async',  # 支持异步推理
            'vis_debug': True,
            'vis_debug_path': './logs/my_system/vis_debug',
        },
    ),
    env=EnvCfg(env_type='internutopia', ...),
    eval_type='vln_distributed',
    ...
)
```

### 6.4 真实机器人部署

参考 `scripts/realworld/http_internvla_server.py`，修改 Agent 初始化为你的 Agent：

```python
agent_config = AgentCfg(
    model_name='my_dual_system',
    model_settings={...},
)
agent = Agent.init(agent_config)
```

---

## 7. 端到端示例：从零接入一个新 S1

以下是一个完整的「用 MLP 策略替代 NextDiT 作为 S1」的最小示例。

### 7.1 文件结构

```
internnav/model/basemodel/internvla_n1/
├── internvla_n1.py             # 修改：添加 generate_traj 分支
├── internvla_n1_arch.py        # 修改：添加 build 函数和初始化
├── internvla_n1_policy.py      # 修改：添加 s1_step_latent 分支
├── my_s1_model.py              # 新增：你的 S1 模型
└── ...
```

### 7.2 核心代码变更

**1) `my_s1_model.py`（新增）**

```python
import torch
import torch.nn as nn

class SimpleMLPS1(nn.Module):
    def __init__(self, cond_dim=768, horizon=32, action_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, 512), nn.GELU(),
            nn.Linear(512, 256), nn.GELU(),
            nn.Linear(256, horizon * action_dim),
        )
        self.horizon = horizon
        self.action_dim = action_dim

    def forward(self, condition):
        cond = condition.mean(dim=1)
        return self.net(cond).view(-1, self.horizon, self.action_dim)
```

**2) `internvla_n1_arch.py`（添加构建函数）**

```python
def build_simple_mlp_s1(config):
    from .my_s1_model import SimpleMLPS1
    return SimpleMLPS1(cond_dim=LatentEmbSize)
```

在 `initialize_vision_modules()` 中添加：

```python
if 'simple_mlp' in system1:
    self.simple_mlp_s1 = build_simple_mlp_s1(model_args)
    self.cond_projector = nn.Linear(self.config.hidden_size, LatentEmbSize)
```

**3) `internvla_n1.py`（添加 generate_traj 分支）**

```python
def generate_traj(self, traj_latents, images_dp=None, depths_dp=None):
    model = self.get_model()
    if hasattr(model, 'simple_mlp_s1'):
        cond = model.cond_projector(traj_latents)
        return model.simple_mlp_s1(cond)
    ...
```

**4) 训练脚本**

```bash
# train_simple_mlp_s1.sh
system1=simple_mlp
torchrun --nproc_per_node=8 \
    internnav/trainer/internvla_n1_trainer.py \
    --model_name_or_path "checkpoints/InternVLA-N1-System2" \
    --system1 ${system1} \
    --tune_mm_vision False --tune_mm_mlp False --tune_mm_llm False \
    ...
```

---

## 8. 常见问题与注意事项

### Q1: S1 和 S2 必须联合训练吗？

**不必须。** 框架设计为可分离训练：
- S2 单独训练：`--system1 "none"`，S2 学会输出像素目标和离散动作
- S1 单独训练：通过 `scripts/train/base_train/train.py --model_name=navdp`
- 联合微调：加载预训练 S2，冻结 LLM 主体，只训练 S1 + `latent_queries` + `cond_projector`

### Q2: 我的 S1 输出是连续轨迹而非离散动作，如何对接？

框架提供了两种转换工具（`internnav/model/utils/vln_utils.py`）：
- **`traj_to_actions()`**：将连续 (x, y, θ) 轨迹转为离散动作序列（STOP/前进/左转/右转）
- **`chunk_token()`**：将 (x, yaw) 逐步转为离散动作 token

你也可以在 `s1_step_latent()` 中实现自己的转换逻辑。

### Q3: 异步模式下 S2 线程安全如何保障？

Agent 使用三把锁管理线程安全（`internvla_n1_agent.py`）：
- `s2_input_lock`：保护 S2 输入数据
- `s2_output_lock`：保护 S2 输出数据
- `s2_agent_lock`：保护模型推理状态

自定义 S2 如需异步运行，应遵循相同的锁机制。

### Q4: 自定义 S2 的 `traj_latents` 维度必须和 Qwen2.5-VL 一样吗？

不需要。`traj_latents` 经过 `cond_projector` 映射到 `LatentEmbSize=768` 后再传给 S1。你只需确保：
- S2 输出的 latent 维度与 `cond_projector` 输入维度一致
- 或自定义 `cond_projector` 适配你的维度

### Q5: 如何只替换 S1 而完全复用官方 S2 权重？

```bash
# 1. 下载官方 S2 权重
#    InternVLA-N1-System2 from HuggingFace

# 2. 联合训练你的 S1（冻结 S2 所有参数）
torchrun ... \
    --model_name_or_path "checkpoints/InternVLA-N1-System2" \
    --system1 "my_s1" \
    --tune_mm_vision False \
    --tune_mm_mlp False \
    --tune_mm_llm False
```

### Q6: 关键文件速查表

| 需求 | 文件路径 |
|------|----------|
| Agent 调度逻辑 | `internnav/agent/internvla_n1_agent.py` |
| S1/S2 数据协议 | `internnav/model/utils/vln_utils.py` |
| Policy（推理入口） | `internnav/model/basemodel/internvla_n1/internvla_n1_policy.py` |
| 模型架构（S1/S2 组件） | `internnav/model/basemodel/internvla_n1/internvla_n1_arch.py` |
| 模型前向/生成 | `internnav/model/basemodel/internvla_n1/internvla_n1.py` |
| Policy 注册表 | `internnav/model/__init__.py` |
| Agent 注册表 | `internnav/agent/base.py` |
| S2 训练器 | `internnav/trainer/internvla_n1_trainer.py` |
| S1 (NavDP) 训练器 | `internnav/trainer/navdp_trainer.py` |
| S2 训练脚本 | `scripts/train/qwenvl_train/train_system2.sh` |
| 双系统联合训练脚本 | `scripts/train/qwenvl_train/train_dual_system.sh` |
| S1 基础训练脚本 | `scripts/train/base_train/train.py` |
| 评估配置 | `scripts/eval/configs/habitat_dual_system_cfg.py` |

---

*本文档基于 InternNav v0.3.0 编写，最后更新于 2026-02。*
