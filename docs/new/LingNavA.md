# LingNav 模式 A — 双服务器模式

**目标平台：** Wheeltec Senior_4wd_bs (Jetson Orin NX 16GB + Gemini 336L/Astra S 深度相机)
**核心思路：** 用 Qwen3-VL 零样本做语义理解（S2），NavDP 做像素目标导航（S1），两者通过 HTTP 解耦，不改动原项目核心包。

> 另一种部署方案见 [`LingNavB.md`](LingNavB.md)（S1 NavDP 端侧部署，减少 S1 网络延迟）。

---

## 系统架构

S2 和 S1 **均在 GPU 服务器运行**，Jetson 通过 HTTP 调用两者。

```
┌──────────────────────────────────────────────────────────────────────┐
│                        GPU 服务器                                     │
│                                                                      │
│  wheeltec_s2_server.py          navdp_server.py                      │
│  ┌──────────────────────┐       ┌──────────────────────┐             │
│  │  S2: Qwen3-VL        │       │  S1: NavDP           │             │
│  │  port 8890           │       │  port 8901           │             │
│  └──────────────────────┘       └──────────────────────┘             │
│            ▲                              ▲                           │
│            │ HTTP                         │ HTTP                      │
└────────────┼──────────────────────────────┼───────────────────────────┘
             │                              │
┌────────────┼──────────────────────────────┼───────────────────────────┐
│            │    Jetson Orin NX            │                           │
│   lingnav_ros_client.py  (LingNavNode)                                │
│   规划线程: S2→HTTP→ pixel → S1→HTTP→ trajectory → MPC               │
└──────────────────────────────────────────────────────────────────────┘
```

| 指标 | 模式 A（本文） | 模式 B（端侧） |
|------|--------------|----------------|
| S1 推理延迟 | ~100-300ms（网络+服务器） | ~50-150ms（本地 fp16） |
| 网络依赖 | S2 + S1 均需网络 | 仅 S2 需网络 |
| Jetson 显存 | 极少（仅 ROS2） | ~200-400MB（NavDP fp16） |
| 部署复杂度 | 服务器需启动 2 个进程 | 服务器只需启动 S2 |

---

## S2 输出格式

Qwen3-VL 输出 JSON 任务数组：

```json
[
  {"task": "move",        "action": "←", "number": 4},
  {"task": "pixel_point", "target": "black chair", "point_2d": [710, 220]}
]
```

| task 类型 | 字段 | 说明 |
|-----------|------|------|
| `pixel_point` | `target`, `point_2d: [x, y]` | 归一化坐标 [0, 1000]，服务器自动转像素；目标不可见时 `point_2d: [null, null]` |
| `move` | `action: ←/→/↑/↓/stop`, `number` | 旋转每单位 15°，前进每单位 0.5m |

`parse_output` 将数组展开为 pipeline 所需字段：
- `navigation`：所有 `move` 任务合并为重复符号串，如 `"←←←←"`（4 次左转）
- `point_2d_pixel`：首个 `pixel_point` 坐标转换后的像素值
- 目标已命名但坐标为空（`[null, null]`）且无 `move` 任务时，pipeline 自动触发 15° 搜索旋转

---

## 文件说明

| 文件 | 运行位置 | 说明 |
|------|---------|------|
| `wheeltec_s2_server.py` | GPU 服务器 | S2 Qwen3-VL 推理服务（`scripts/realworld2/`） |
| `lingnav_pipeline.py` | 任意 | S2+S1 联合推理管线，`LingNavPipeline` 类（`scripts/realworld2/`） |
| `lingnav_ros_client.py` | Jetson | ROS2 完整导航节点（`scripts/realworld2/`） |
| `test_s2_client.py` | 任意 | S2 单独测试客户端（`scripts/realworld2/`） |
| `navdp_server.py` | GPU 服务器 | S1 NavDP 推理服务（`scripts/inference/NavDP/`） |
| `navdp_client.py` | 被 pipeline 导入 | S1 NavDP HTTP 客户端（`scripts/inference/NavDP/`） |
| `wheeltec_controllers.py` | 被 ros_client 导入 | MPC + PID 控制器（`scripts/realworld/`） |
| `wheeltec_thread_utils.py` | 被 ros_client 导入 | 读写锁（`scripts/realworld/`） |

---

## 启动方式

### 0. 首次依赖安装

**GPU 服务器：**

```bash
conda activate internnav
pip install flask transformers>=4.57.0
# flash-attn（可选，无则自动降级到 sdpa）
pip install flash-attn --no-build-isolation
```

**Jetson 端：**

```bash
pip3 install numpy requests Pillow opencv-python casadi scipy \
    -i https://pypi.tuna.tsinghua.edu.cn/simple
sudo apt install ros-humble-cv-bridge ros-humble-message-filters
```

---

### 1. GPU 服务器 — 启动 S2（Qwen3-VL）

```bash
conda activate internnav
cd /path/to/InternNav

python scripts/realworld2/wheeltec_s2_server.py \
    --model_path /data2/ltd/checkpoints/Qwen3-VL/Qwen3-VL-8B-Instruct \
    --port 8890 \
    --host 0.0.0.0 \
    --device auto
```

### 2. GPU 服务器 — 启动 S1（NavDP）

```bash
# 新终端
conda activate internnav
cd /path/to/InternNav

python scripts/inference/NavDP/navdp_server.py \
    --checkpoint /data2/ltd/checkpoints/NavDP/navdp-weights.ckpt \
    --port 8901 \
    --host 0.0.0.0 \
    --device cuda:0
```

启动成功输出：
```
NavDP S1 Server starting on 0.0.0.0:8901
  Checkpoint: /data2/ltd/checkpoints/NavDP/navdp-weights.ckpt
  Device: cuda:0
```

> NavDP checkpoint 下载：参见 `docs/new/s1_navdp_guide.md` 中的链接。

### 3. Jetson — 启动 ROS2 客户端

```bash
# Terminal 1 — 机器人底盘
source /opt/ros/humble/setup.bash
ros2 launch turn_on_wheeltec_robot turn_on_wheeltec_robot.launch.py

# Terminal 2 — 相机
ros2 launch turn_on_wheeltec_robot wheeltec_camera.launch.py

# Terminal 3 — LingNav 导航节点
source /opt/ros/humble/setup.bash
cd ~/InternNav

python3 scripts/realworld2/lingnav_ros_client.py \
    --instruction "Turn left 60 degree, then go to the black chair" \
    --s2_host 192.168.1.100 \
    --s2_port 8890 \
    --s1_host 192.168.1.100 \
    --s1_port 8901
```

---

### 可选：服务器联调测试（无需机器人）

```bash
# 本地测试 S2 连通性
python scripts/realworld2/test_s2_client.py \
    --host 服务器IP --port 8890 \
    --image /GitHub/InternNav/data/debug_raw_000.jpg \
    --instruction "Turn left 60 degree, then go to the black chair"

# 服务器上测试 S2 连通性
python scripts/realworld2/test_s2_client.py \
    --host 127.0.0.1 --port 8890 \
    --image /data3/ltd/Qwen3-VL/debug_raw_000.jpg \
    --instruction "Turn left 60 degree, then go to the black chair"

# 测试 S2+S1 联合推理（真实图片）
python scripts/realworld2/lingnav_pipeline.py \
    --s2_host 127.0.0.1 --s2_port 8890 \
    --s1_host 127.0.0.1 --s1_port 8901 \
    --image /path/to/test.jpg \
    --instruction "Turn left 60 degree, then go to the black chair"

# 仅测试 S2（NavDP 未启动时）
python scripts/realworld2/lingnav_pipeline.py \
    --s2_host 127.0.0.1 --s2_port 8890 \
    --s1_host 127.0.0.1 --s1_port 8901 \
    --random --instruction "Turn left 60 degree, then go to the black chair" --skip_s1
```

**`lingnav_pipeline.py` 参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--s2_host` | `127.0.0.1` | S2 服务器 IP |
| `--s2_port` | `8890` | S2 端口 |
| `--s1_host` | `127.0.0.1` | S1 服务器 IP |
| `--s1_port` | `8901` | S1 端口 |
| `--instruction` | `"Turn left 60 degree, then go to the black chair"` | 导航指令 |
| `--image` | — | 真实图片路径（与 `--random` 二选一）|
| `--random` | False | 使用随机噪声图（连通性测试）|
| `--skip_s1` | False | 跳过 S1 调用（NavDP 未启动时）|

**预期输出（S2+S1 均在线，真实图片）：**
```
[1] 服务器连通性检查 …
  [S2] OK  {'status': 'ok', 'model': '/data2/ltd/checkpoints/Qwen3-VL/Qwen3-VL-8B-Instruct'}
  [S1] reachable ...
[2] Reset pipeline …
[LingNav] Reset. instruction='Turn left 60 degree, then go to the black chair'
[3] 图像来源: /path/to/test.jpg
[4] 执行 step(), instruction='Turn left 60 degree, then go to the black chair' …

[Result]  (总耗时 2.70s)
  mode          : trajectory
  S2 target     : black chair
  S2 pixel_norm : [710, 220]
  S2 pixel_px   : [454, 105]
  S2 navigation : '←←←←'
  S2 raw        : '[{"task": "move", "action": "←", "number": 4}, {"task": "pixel_point", "target": "black chair", "point_2d": [710, 220]}]'
  S1 traj shape : (1, 24, 3)
  S1 traj[0,0]  : [0.142 -0.031  0.000]  (first waypoint, meters)
  S1 values max : 1.847
```

> 目标在当前帧可见时，pipeline 优先走 trajectory 模式（忽略前置转向）。
> 目标不可见时 `point_2d: [null, null]`，`pixel=None`，pipeline 自动触发 15° 搜索旋转。

---

## ROS2 话题

| 话题 | 方向 | 说明 |
|------|------|------|
| `/camera/color/image_raw` | 订阅 | Gemini 336L/Astra S RGB，`sensor_msgs/Image`，rgb8 |
| `/camera/depth/image` | 订阅 | Gemini 336L/Astra S 深度，`sensor_msgs/Image`，16UC1 (mm) |
| `/odom` | 订阅 | 里程计，`nav_msgs/Odometry` |
| `/cmd_vel` | 发布 | 速度指令，`geometry_msgs/Twist` |

**运行时日志示例：**

```
[LingNav] Reset. instruction='Turn left 60 degree, then go to the black chair'
[Plan] mode=trajectory | target=black chair | nav='←←←←'   ← 目标可见，直接导航（忽略前置转向）
[Plan] mode=trajectory | target=black chair | nav='←←←←'   ← 持续追踪
[Plan] mode=rotate     | target=black chair | nav=''        ← 目标不可见，自动搜索旋转 15°
[Plan] mode=stop       | target=black chair | nav='stop'
[LingNav] Stopped.
```

---

## LingNavPipeline API（代码调用）

```python
from scripts.realworld2.lingnav_pipeline import LingNavPipeline
import numpy as np

pipeline = LingNavPipeline(
    s2_host="192.168.1.100", s2_port=8890,
    s1_host="192.168.1.100", s1_port=8901,
)

# 每个导航任务开始时调用一次
pipeline.reset("Turn left 60 degree, then go to the black chair")

# 每个控制步（约 0.3s）调用
result = pipeline.step(rgb_bgr, depth_m)   # rgb: (H,W,3) BGR, depth: (H,W) float32 m

if result["mode"] == "trajectory":
    traj = result["trajectory"]      # (1, 24, 3)，单位米，x=前 y=左
    # → 送入 MPC 控制器跟踪
elif result["mode"] == "rotate":
    rad = result["rotation_rad"]     # 正=左转，负=右转
    # → 原地旋转 |rad| 弧度
elif result["mode"] == "stop":
    pass                             # → 停止机器人
elif result["mode"] == "error":
    print(result["message"])
```

---

## 开发进度

| Phase | 状态 | 描述 |
|-------|------|------|
| Phase 1 | ✅ 完成 | `wheeltec_s2_server.py`：Qwen3-VL 零样本 S2 服务，Prompt Engineering |
| Phase 2 | ✅ 完成 | `lingnav_pipeline.py`：S2+S1 联合推理管线，NavDP pixelgoal 对接 |
| Phase 3 | ✅ 完成 | `lingnav_ros_client.py`：Jetson ROS2 客户端，规划+控制双线程，MPC+PID，碰撞检测 |
| Phase 3.5 | ✅ 完成 | `navdp_local_client.py`：S1 NavDP 端侧部署（见 LingNavB.md） |
| Phase 4 | 🔲 按需 | S1 NavDP 在小车数据上 fine-tune（SR < 50% 时触发）|

---

## 端口约定

| 服务 | 端口 | 说明 |
|------|------|------|
| S2 Qwen3-VL | **8890** | LingNav 新增，不与原项目冲突 |
| S1 NavDP | **8901** | 复用 `scripts/inference/NavDP/navdp_server.py` |
| 原 InternVLA-N1 评估服务器 | 8087 | 原项目，不受影响 |
| 原实机服务器 | 8888 | 原项目，不受影响 |
