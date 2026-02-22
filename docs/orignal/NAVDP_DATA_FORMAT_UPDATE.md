# NavDP 数据集格式更新说明

## 问题总结

当前 `navdp_lerobot_dataset.py` 使用的是旧版数据格式，与新的 InternData-N1 格式不兼容。

## 新旧格式对比

### 旧格式（原 navdp_lerobot_dataset.py 期望）
```
root_dir/<group_dir>/<scene_dir>/<traj_dir>/
  ├── videos/chunk-000/observation.images.rgb/<number>.jpg
  ├── videos/chunk-000/observation.images.depth/<number>.png
  ├── data/chunk-000/episode_000000.parquet  # 固定文件名
  └── data/chunk-000/path.ply  # 路径点云
```

### 新格式（InternData-N1）
```
vln_n1/traj_data/<scene_datasets>/<scene>/
  ├── meta/
  │   ├── episodes.jsonl  # episode 元数据
  │   ├── info.json
  │   ├── tasks.jsonl
  │   └── pointcloud.ply  # 场景点云（可选）
  ├── data/chunk-<chunk_id>/
  │   ├── episode_000000.parquet  # 多个 episode 文件
  │   ├── episode_000001.parquet
  │   └── ...
  └── videos/chunk-<chunk_id>/
      ├── observation.images.rgb/
      │   ├── episode_000000_000.jpg  # 新命名格式
      │   ├── episode_000000_001.jpg
      │   └── ...
      └── observation.images.depth/
          ├── episode_000000_000.png
          └── ...
```

## 关键差异

1. **目录层级**：从三层（group/scene/traj）变为两层（scene_datasets/scene）
2. **图像命名**：从 `<number>.jpg` 变为 `episode_<id>_<frame>.jpg`
3. **多 episode**：每个场景有多个 episode parquet 文件
4. **元数据**：新增 meta 目录存储 episodes.jsonl
5. **点云**：可选，可能不存在（旧代码强制要求）

## 推荐解决方案

### 方案 1：完全重写（推荐）
参考 `internvla_n1_lerobot_dataset.py` 的 `get_annotations_from_lerobot_data()` 函数：

```python
def get_annotations_from_lerobot_data(data_path, setting):
    # 从 meta/episodes.jsonl 读取元数据
    # 使用 pyarrow 读取 parquet 文件
    # 返回结构化的 annotations
```

**优点**：
- 与现有 InternVLA-N1 训练流程一致
- 支持新格式的所有特性
- 代码更清晰、可维护

**缺点**：
- 需要重构整个数据集类

### 方案 2：修改现有代码（已实施）
更新 `NavDP_Base_Datset.__init__()` 以适配新格式：

**已完成的修改**：
1. ✅ 更新目录遍历逻辑（scene_datasets/scene）
2. ✅ 更新图像文件名解析（episode_id_frame）
3. ✅ 支持多个 episode 文件
4. ✅ 点云文件改为可选

**仍需完善**：
- 读取 meta/episodes.jsonl 获取更多信息
- 从 parquet 中读取更丰富的字段（pose, goal等）
- 与 InternVLA-N1 训练器集成

## 当前状态

已修改 `navdp_lerobot_dataset.py`，能够：
- ✅ 正确识别新的目录结构
- ✅ 加载多个 episode
- ✅ 处理新的图像命名格式
- ✅ 优雅处理缺失的点云文件

**测试结果**：
```
数据集加载成功!
轨迹总数: 6
第一条轨迹信息:
  - 场景目录: data/datasets/InternData-N1/vln_n1/traj_data\matterport3d_d435i\1LXtFkjw3qL
  - 数据文件: data/datasets/InternData-N1/vln_n1/traj_data\matterport3d_d435i\1LXtFkjw3qL\data/chunk-000\episode_000000.parquet
  - RGB 图像数量: 122
  - Depth 图像数量: 122
```

## 下一步建议

### 短期（快速训练）
使用当前修改后的代码即可开始训练，因为：
- 基础数据加载已正常工作
- 图像和深度数据加载正确
- 点云缺失不会导致崩溃

### 长期（优化）
1. 参考 InternVLA-N1 的数据加载方式
2. 使用 pyarrow 高效读取 parquet
3. 从 meta/episodes.jsonl 读取完整元数据
4. 考虑统一到 `internvla_n1_lerobot_dataset.py` 的框架

## 配置文件更新

需要更新 `scripts/train/base_train/configs/navdp.py`：

```python
il=IlCfg(
    # ... 其他配置 ...
    root_dir='data/datasets/InternData-N1/vln_n1/traj_data',  # 指向新格式数据
    preload=False,  # 不使用预加载缓存
    scene_scale=1.0,  # 使用全部场景
    trajectory_data_scale=1.0,  # 使用全部轨迹
)
```

## 参考代码
- ✅ 新格式数据加载：`internnav/dataset/internvla_n1_lerobot_dataset.py`
- ✅ 训练脚本：`scripts/train/qwenvl_train/train_dual_system.sh`
- ✅ 训练器：`internnav/trainer/internvla_n1_trainer.py`
