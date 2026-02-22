# 修复 DDP 训练中 "Parameter indices which did not receive grad" 错误

## 问题描述

在使用 `torchrun` 进行分布式训练时，出现以下错误：

```
RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one.
This error indicates that your module has parameters that were not used in producing loss.
Parameter indices which did not receive grad for rank 0: 2 217 394
```

## 问题根因

1. **`point_encoder` 未参与损失计算**：在 `NavDPNet.forward()` 中，`point_encoder` 的输出 `pointgoal_embed` 被创建，但在 `ng_cond_embeddings` 中使用的是零向量 `nogoal_embed`，而在 `mg_cond_embeddings` 中通过随机选择机制，`pointgoal_embed` 可能不被选中，导致 `point_encoder` 的参数没有梯度。

2. **双重 DDP 包装**：`train.py` 中手动创建了 DDP 包装，而 Hugging Face Trainer 也会尝试创建 DDP，导致冲突。

## 修改文件

### 1. `internnav/model/basemodel/navdp/navdp_policy.py`

#### 修改 1.1：添加 `point_aux_head` 模块（`__init__` 方法）

```python
# 第 133-135 行
self.point_aux_head = nn.Linear(self.token_dim, 3)
self.pixel_aux_head = nn.Linear(self.token_dim, 3)
self.image_aux_head = nn.Linear(self.token_dim, 3)
```

#### 修改 1.2：在 `forward` 方法中计算 `pointgoal_aux_pred`

```python
# 第 206-208 行
pointgoal_aux_pred = self.point_aux_head(pointgoal_embed[:, 0])
imagegoal_aux_pred = self.image_aux_head(imagegoal_embed[:, 0])
pixelgoal_aux_pred = self.pixel_aux_head(pixelgoal_embed[:, 0])
```

#### 修改 1.3：更新 `forward` 方法的返回值

```python
# 第 264-271 行
return (
    noise_pred_ng,
    noise_pred_mg,
    cr_label_pred,
    cr_augment_pred,
    [ng_noise, mg_noise],
    [pointgoal_aux_pred, imagegoal_aux_pred, pixelgoal_aux_pred],  # 从 2 个变为 3 个
)
```

#### 修改 1.4：使用随机采样选择目标类型

```python
# 第 222-231 行
# Use random sampling to ensure all goal encoders receive gradients
# Each sample randomly selects goal types for the 3 goal slots
selections_0 = torch.randint(0, 3, (batch_size,), device=pointgoal_embed.device)
selections_1 = torch.randint(0, 3, (batch_size,), device=pointgoal_embed.device)
selections_2 = torch.randint(0, 3, (batch_size,), device=pointgoal_embed.device)
goal_embeds = torch.stack(cand_goal_embed, dim=0)  # [3, batch_size, 1, token_dim]
selected_goals_0 = goal_embeds[selections_0, torch.arange(batch_size, device=pointgoal_embed.device), :, :]
selected_goals_1 = goal_embeds[selections_1, torch.arange(batch_size, device=pointgoal_embed.device), :, :]
selected_goals_2 = goal_embeds[selections_2, torch.arange(batch_size, device=pointgoal_embed.device), :, :]
```

### 2. `internnav/trainer/navdp_trainer.py`

#### 修改 2.1：更新 `aux_loss` 计算以包含三个辅助预测

```python
# 第 92-97 行
# aux_pred now contains [pointgoal_aux_pred, imagegoal_aux_pred, pixelgoal_aux_pred]
aux_loss = (
    (1.0 / 3.0) * (inputs_on_device["batch_pg"] - aux_pred[0]).square().mean()
    + (1.0 / 3.0) * (inputs_on_device["batch_pg"] - aux_pred[1]).square().mean()
    + (1.0 / 3.0) * (inputs_on_device["batch_pg"] - aux_pred[2]).square().mean()
)
```

### 3. `scripts/train/base_train/train.py`

#### 修改 3.1：移除手动 DDP 包装

```python
# 第 136-145 行
# 原代码：
# if world_size > 1:
#     model = torch.nn.parallel.DistributedDataParallel(
#         model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True
#     )

# 修改后：
# Note: Do NOT manually wrap with DDP here - let HuggingFace Trainer handle it
# The Trainer will use ddp_find_unused_parameters from TrainingArguments
```

## 修复原理

1. **添加辅助预测头**：为 `point_encoder` 添加 `point_aux_head`，确保 `point_encoder` 的输出始终参与损失计算，不依赖于随机目标选择机制。

2. **统一 DDP 管理**：移除手动的 DDP 包装，让 Hugging Face Trainer 通过 `TrainingArguments` 中的 `ddp_find_unused_parameters=True` 自动管理分布式训练，避免双重包装导致的问题。

## 验证

运行训练命令验证修复：

```bash
./scripts/train/base_train/start_train.sh --name navdp_train --model navdp
```

如需更详细的调试信息，可设置环境变量：

```bash
TORCH_DISTRIBUTED_DEBUG=DETAIL ./scripts/train/base_train/start_train.sh --name navdp_train --model navdp
```
