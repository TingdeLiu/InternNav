# InternNav 开发指南

本指南旨在帮助开发者快速理解 InternNav 的项目架构，并指导如何进行二次开发。

## 1. 项目架构概览

InternNav 是一个模块化的具身智能导航框架，采用 **Client-Server** 架构进行评估，并基于 **Transformers Trainer** 进行大规模预训练。

### 核心模块定义
- **`internnav/agent`**: 智能体逻辑层。负责处理环境观测（Observations）、调用模型推理（Inference）并输出动作（Actions）。
- **`internnav/model`**: 模型架构层。定义神经网络结构（如 InternVLA, NavDP 等）。
- **`internnav/dataset`**: 数据处理层。支持 LMDB、LeRobot 等多种格式，适配大规模轨迹数据。
- **`internnav/trainer`**: 训练逻辑层。继承自 `transformers.Trainer`，支持分布式训练和自定义损失函数。
- **`internnav/env`**: 仿真环境层。集成了 Habitat, InternUtopia 以及真实世界接口。

---

## 2. 核心组件开发

### 2.1 创建自定义 Agent
所有的 Agent 必须继承自 `internnav.agent.base.Agent` 并使用 `@Agent.register` 装饰器进行注册。

```python
from internnav.agent.base import Agent
from internnav.configs.agent import AgentCfg

@Agent.register('my_custom_agent')
class MyAgent(Agent):
    def __init__(self, config: AgentCfg):
        super().__init__(config)
        # 初始化模型或状态
        self.model = ... 

    def reset(self):
        """每个 Episode 开始前调用，重置隐藏状态或动作历史"""
        self.state = None

    def step(self, obs):
        """
        环境交互核心逻辑
        obs 包含: 'rgb', 'depth', 'instruction', 'globalgps', 'global_rotation'
        返回: 动作索引 (0: Stop, 1: Move Forward, 2: Left, 3: Right)
        """
        # 1. 预处理 obs
        # 2. 模型推理
        # 3. 返回 action
        return action
```
> **注意**：完成编写后，需在 `internnav/agent/__init__.py` 中导入该类，以确保注册生效。

### 2.2 数据集适配 (`Dataset`)
项目主要使用 `IterableDataset` 来处理长序列导航数据。基类为 `BaseDataset`。

- **数据 I/O 标准**：
    - 输入：`instruction` (str), `rgb` (HxWx3), `depth` (HxWx1)。
    - 辅助信息：`globalgps`, `global_rotation`, `prev_actions`。

### 2.3 模型训练 (`Trainer`)
训练器基于 HuggingFace 生态，支持高效的并行训练。可以通过修改 `internnav/trainer/base.py` 中的 `compute_loss` 来实现自定义任务。

---

## 3. 开发流程与常用脚本

### 3.1 训练模型
训练脚本通常位于 `scripts/train/`。
```bash
# 示例：启动训练
python scripts/train/base_train/train.py --config configs/model/navdp.py
```

### 3.2 评估模型 (Client-Server 模式)
评估过程通常分为两步：
1. **启动模型服务器**：
   ```bash
   python scripts/eval/start_server.py --port 12345 --agent_type internvla_n1
   ```
2. **运行评估客户端**：
   ```bash
   python scripts/eval/eval.py --server_address localhost:12345 --env habitat
   ```

---

## 4. 后续开发建议

1.  **参考官方教程**：
    详细的 Agent 自定义流程可参考 [InternNav 官方文档 - Agent 教程](https://internrobotics.github.io/user_guide/internnav/tutorials/agent.html)。
2.  **增加新任务**：
    如果需要增加如“物体寻回 (ObjectNav)”等新任务，建议在 `internnav/env` 中定义新的 Observation 空间，并在 `internnav/agent` 中实现相应的逻辑。
3.  **调试技巧**：
    - 使用 `internnav/utils/visual_tool.py` 可视化 Agent 的推理过程和注意力图。
    - 检查 `tests/` 目录下的单元测试以验证基础组件的正确性。

---
