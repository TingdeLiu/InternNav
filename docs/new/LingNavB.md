# LingNav 模式 B — S1 端侧部署模式

**目标平台：** Wheeltec Senior_4wd_bs (Jetson Orin NX 16GB + Gemini 336L/Astra S 深度相机)
**核心思路：** 用 Qwen3-VL 零样本做语义理解（S2），NavDP 做像素目标导航（S1），两者通过 HTTP 解耦，不改动原项目核心包。

> 另一种部署方案见 [`LingNavA.md`](LingNavA.md)（S1 也在服务器运行，部署更简单）。

---

## 系统架构

NavDP (S1) **直接在 Jetson Orin NX 上运行**，减少 S1 网络延迟；
S2 (Qwen3-VL 8B) 仍在 GPU 服务器运行（Jetson 16GB 不足以同时跑 8B VLM）。

```
┌──────────────────────────────────────────────────────────────────────┐
│                        GPU 服务器                                     │
│                                                                      │
│  wheeltec_s2_server.py                                                │
│  ┌──────────────────────┐                                             │
│  │  S2: Qwen3-VL        │                                             │
│  │  port 8890           │                                             │
│  └──────────────────────┘                                             │
│            ▲                                                          │
│            │ HTTP                                                     │
└────────────┼─────────────────────────────────────────────────────────┘
             │
┌────────────┼─────────────────────────────────────────────────────────┐
│            │    Jetson Orin NX 16GB                                   │
│   lingnav_ros_client.py  (LingNavNode, --local_s1)                    │
│                                                                      │
│   规划线程:                                                            │
│     ① 队列空时: RGB+Depth → S2 (HTTP) → 任务队列 [move/pixel/move]   │
│     ② move 任务: 弹出 → 返回旋转角度（ROS 控制器限时执行）              │
│     ③ pixel_point 任务: 用初始像素坐标 → NavDPLocalClient → traj     │
│        (目标初始不可见时才调 S2 搜索；NavDP memory 处理坐标偏差)        │
│     ④ Critic < 阈值 → 弹出，执行下一任务                              │
│                                                                      │
│   NavDPLocalClient                                                    │
│   ┌──────────────────────┐                                             │
│   │  NavDP (fp16)        │ ← 约 200-400MB 显存                        │
│   │  navdp_local_client  │                                             │
│   └──────────────────────┘                                             │
│                                                                      │
│   控制线程 0.1s/次 → MPC → /cmd_vel                                   │
└──────────────────────────────────────────────────────────────────────┘
```

| 指标 | 模式 A（服务器） | 模式 B（本文） |
|------|----------------|--------------|
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

`parse_output` 将数组解析为两层字段：

**给 pipeline 顺序执行用（主路径）**
- `tasks`：原始任务对象列表，保留顺序；每个 `pixel_point` 任务附带 `point_2d_pixel: [u, v]`（服务器已完成 [0,1000] → 实际像素的转换），供 pipeline 直接使用

**兼容旧逻辑的折叠字段（fallback）**
- `navigation`：所有 `move` 任务合并为重复符号串，如 `"←←←←"`（4 次左转）
- `point_2d_pixel`：首个 `pixel_point` 坐标转换后的像素值

---

## 多步指令顺序执行

pipeline 维护一个**任务队列**，实现 move → pixel_point → move 任意组合的顺序执行。

### 队列生命周期

```
reset("Turn left 60°, go to black chair, rotate right 30°")
  → _task_queue = []，_tasks_loaded = False

第 1 步 step()：
  队列空 → 调 S2 → 解析 tasks 字段
  → _task_queue = [move(←×4), pixel_point(chair), move(→×2)]
  → _tasks_loaded = True
  → 取队首 move(←×4)，弹出，return rotate(-60°)

第 2~N 步 step()：
  队首 = pixel_point(chair)
  → 直接用初始像素坐标调 S1 NavDP（不再调 S2）
  → NavDP Critic < threshold → 弹出 pixel_point
  → return trajectory（本步仍正常行驶）

第 N+1 步 step()：
  队首 = move(→×2)，弹出，return rotate(+30°)

第 N+2 步 step()：
  _tasks_loaded=True + 队列空 → return stop
```

### pixel_point 任务完成信号

| 信号 | 触发条件 | 后续行为 |
|------|---------|---------|
| **NavDP Critic** | `values.max() < stop_threshold`（默认 -3.0） | 弹出任务，本步仍返回 trajectory，下步执行后续任务 |
| **S2 stop** | S2 返回 `action: "stop"` 的 move 任务 | 弹出任务，本步返回 stop 信号 |

### S2 调用频率

| 场景 | S2 调用 | 说明 |
|------|---------|------|
| 队列填充（每 episode 一次） | ✅ | 解析完整任务序列 |
| pixel_point 导航中（目标可见） | ❌ | 直接用初始坐标，NavDP memory queue 处理运动偏差 |
| pixel_point 导航中（目标不可见） | ❌ | 固定 15° 搜索旋转，NavDP 负责完成任务 |

### 目标不可见时的搜索行为

初始坐标为 None 时固定 15° 搜索旋转，不调 S2：

```
[LingNav] task=pixel_point (target='black chair', pixel=None) → search rotate
[LingNav] task=pixel_point (target='black chair', pixel=None) → search rotate
...（直到 reset() 被调用，或外部停止机器人）
```

> 实践中，先行的 move 任务（转向）会将目标带入视野。若初始即不可见，需在指令中增加前置转向任务。

---

## 文件说明

| 文件 | 运行位置 | 说明 |
|------|---------|------|
| `wheeltec_s2_server.py` | GPU 服务器 | S2 Qwen3-VL 推理服务（`scripts/realworld2/`） |
| `lingnav_pipeline.py` | 任意 | S2+S1 联合推理管线，`LingNavPipeline` 类（`scripts/realworld2/`） |
| `lingnav_ros_client.py` | Jetson | ROS2 完整导航节点，`--local_s1` 模式（`scripts/realworld2/`） |
| `test_s2_client.py` | 任意 | S2 单独测试客户端（`scripts/realworld2/`） |
| `navdp_local_client.py` | 被 ros_client 导入 | S1 NavDP 端侧本地推理客户端，无 HTTP，支持 fp16（`scripts/inference/NavDP/`） |
| `navdp_agent.py` | 被 local_client 导入 | NavDP Agent 封装，管理记忆队列（`scripts/inference/NavDP/`） |
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
# ROS2 依赖
pip3 install numpy requests Pillow opencv-python casadi scipy \
    -i https://pypi.tuna.tsinghua.edu.cn/simple
sudo apt install ros-humble-cv-bridge ros-humble-message-filters

# InternNav（含 NavDP 策略网络，端侧模式必须）
cd ~/InternNav
pip3 install -e .
git submodule update --init --recursive
export PYTHONPATH=$PYTHONPATH:~/InternNav/src/diffusion-policy
```

> torch/torchvision：JetPack 5.x 通常已自带，无需单独安装。

---

### 1. GPU 服务器 — 启动 S2（仅需一个进程）

```bash
conda activate internnav
cd /path/to/InternNav

python scripts/realworld2/wheeltec_s2_server.py \
    --model_path /data2/ltd/checkpoints/Qwen3-VL/Qwen3-VL-8B-Instruct \
    --port 8890 \
    --host 0.0.0.0 \
    --device cuda:7
```

### 2. Jetson — 启动 ROS2 客户端（端侧 S1 模式）

```bash
# Terminal 1 — 机器人底盘
source /opt/ros/humble/setup.bash
ros2 launch turn_on_wheeltec_robot turn_on_wheeltec_robot.launch.py

# Terminal 2 — 相机
ros2 launch turn_on_wheeltec_robot wheeltec_camera.launch.py

# Terminal 3 — LingNav 导航节点（端侧 S1）
source /opt/ros/humble/setup.bash
export PYTHONPATH=$PYTHONPATH:~/InternNav/src/diffusion-policy

python3 scripts/realworld2/lingnav_ros_client.py \
    --instruction "Turn left 60 degree, then go to the black chair" \
    --s2_host 192.168.1.100 \
    --s2_port 8890 \
    --local_s1 \
    --s1_checkpoint /home/wheeltec/VLN/checkpoints/navdp-weights.ckpt \
    --s1_device cuda:0 \
    --s1_half                    # 推荐：fp16 节省显存、加快推理
```

**端侧模式参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--local_s1` | False | 开启 S1 端侧本地推理（无需 navdp_server） |
| `--s1_checkpoint` | 必填 | NavDP 权重文件路径（.ckpt）|
| `--s1_device` | `cuda:0` | NavDP 推理设备 |
| `--s1_half` | False | 开启 fp16（Jetson 推荐，约 50% 显存节省）|

---

### 可选：服务器联调测试（无需机器人）

端侧模式下服务器只有 S2，可用以下命令验证 S2 连通性：

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

# 仅测试 S2（跳过 S1，端侧模式下无 navdp_server）
python scripts/realworld2/lingnav_pipeline.py \
    --s2_host 127.0.0.1 --s2_port 8890 \
    --s1_host 127.0.0.1 --s1_port 8901 \
    --random --instruction "Turn left 60 degree, then go to the black chair" --skip_s1
```

**预期输出（S2 在线，`--skip_s1`）：**
```
[1] 服务器连通性检查 …
  [S2] OK  {'status': 'ok', 'model': '/data2/ltd/checkpoints/Qwen3-VL/Qwen3-VL-8B-Instruct'}
  [S1] OFFLINE — http://127.0.0.1:8901/  ← 预期，端侧模式无 navdp_server
[2] Reset pipeline …（skip_s1 模式跳过）
[3] 图像来源: 随机噪声 640×480
[4] 执行 step(), instruction='Turn left 60 degree, then go to the black chair' …

[LingNav] task queue (2): ['move', 'pixel_point']
[LingNav] task=move | '←←←←' → -60.0° | remaining=1

[Result]  (总耗时 2.70s)
  mode          : rotate               ← 队首 move 任务先执行
  S2 target     : None
  S2 pixel_norm : None
  S2 pixel_px   : None
  S2 navigation : None
  S2 raw        : None
  rotation      : -1.0472 rad  (-60.0°)
```

> 注：`step()` 返回的 `rotate`/`trajectory` 结果中不含 `s2` 键，S2 原始字段需直接通过 S2 服务日志或 `parse_output()` 结果查看。

---

## ROS2 话题

| 话题 | 方向 | 说明 |
|------|------|------|
| `/camera/color/image_raw` | 订阅 | Gemini 336L/Astra S RGB，`sensor_msgs/Image`，rgb8 |
| `/camera/depth/image` | 订阅 | Gemini 336L/Astra S 深度，`sensor_msgs/Image`，16UC1 (mm) |
| `/odom` | 订阅 | 里程计，`nav_msgs/Odometry` |
| `/cmd_vel` | 发布 | 速度指令，`geometry_msgs/Twist` |

**运行时日志示例（指令：`Turn left 60°, go to black chair, rotate right 30°`）：**

```
[LingNav] Reset. instruction='Turn left 60 degree, then go to the black chair, rotate right 30 degree'
[LingNav] task queue (3): ['move', 'pixel_point', 'move']
[LingNav] task=move | '←←←←' → -60.0° | remaining=2
[Plan] mode=rotate | target=None | nav=None          ← 执行 60° 左转

[Plan] mode=trajectory | target=None | nav=None ← 目标可见，S1 导航中
[Plan] mode=trajectory | target=None | nav=None ← 持续追踪
[Plan] mode=rotate     | target=None | nav=None ← 目标不可见，搜索旋转 15°
[Plan] mode=trajectory | target=None | nav=None ← 目标重新入镜
[LingNav] task=pixel_point done (NavDP Critic) | target='black chair' | remaining=1

[LingNav] task=move | '→→' → 30.0° | remaining=0
[Plan] mode=rotate | target=None | nav=None          ← 执行 30° 右转

[Plan] mode=stop                                     ← 队列清空，任务完成
[LingNav] Stopped.
```

---

## LingNavPipeline API（代码调用）

```python
import sys
sys.path.insert(0, "/path/to/InternNav/scripts/inference/NavDP")

from scripts.realworld2.lingnav_pipeline import LingNavPipeline
import numpy as np

# 端侧模式：传入 NavDPLocalClient
from navdp_local_client import NavDPLocalClient
s1_client = NavDPLocalClient(
    checkpoint="/home/wheeltec/VLN/checkpoints/navdp-weights.ckpt",
    device="cuda:0",
    half=True,
)

pipeline = LingNavPipeline(
    s2_host="192.168.1.100", s2_port=8890,
    s1_client=s1_client,   # 传入本地客户端，s1_host/s1_port 被忽略
)

# 每个导航任务开始时调用一次（支持多步复合指令）
pipeline.reset("Turn left 60 degree, then go to the black chair, then rotate right 30 degree")
# pipeline 内部队列: [move(←×4), pixel_point(chair), move(→×2)]

# 控制循环（约 0.3s/步）
while True:
    result = pipeline.step(rgb_bgr, depth_m)   # rgb: (H,W,3) BGR, depth: (H,W) float32 m

    if result["mode"] == "trajectory":
        traj = result["trajectory"]      # (1, 24, 3)，单位米，x=前 y=左
        # → 送入 MPC 控制器跟踪
    elif result["mode"] == "rotate":
        rad = result["rotation_rad"]     # 正=左转（逆时针），负=右转（顺时针）
        # → 原地旋转 |rad| 弧度；pipeline 自动在旋转完成后切换到下一任务
    elif result["mode"] == "stop":
        break                            # → 所有任务完成，停止机器人
    elif result["mode"] == "error":
        print(result["message"])
        break
```

---

## 开发进度

| Phase | 状态 | 描述 |
|-------|------|------|
| Phase 1 | ✅ 完成 | `wheeltec_s2_server.py`：Qwen3-VL 零样本 S2 服务，Prompt Engineering |
| Phase 2 | ✅ 完成 | `lingnav_pipeline.py`：S2+S1 联合推理管线，NavDP pixelgoal 对接 |
| Phase 2.5 | ✅ 完成 | `lingnav_pipeline.py`：多步指令顺序执行（任务队列，move→pixel→move 任意组合） |
| Phase 3 | ✅ 完成 | `lingnav_ros_client.py`：Jetson ROS2 客户端，规划+控制双线程，MPC+PID，碰撞检测 |
| Phase 3.5 | ✅ 完成 | `navdp_local_client.py`：S1 NavDP 端侧部署（本文，fp16，无 HTTP 依赖） |
| Phase 4 | 🔲 按需 | S1 NavDP 在小车数据上 fine-tune（SR < 50% 时触发）|

---

## 端口约定

| 服务 | 端口 | 说明 |
|------|------|------|
| S2 Qwen3-VL | **8890** | LingNav 新增，不与原项目冲突 |
| S1 NavDP | **8901** | 模式 A 使用；模式 B 本地推理，不占端口 |
| 原 InternVLA-N1 评估服务器 | 8087 | 原项目，不受影响 |
| 原实机服务器 | 8888 | 原项目，不受影响 |
