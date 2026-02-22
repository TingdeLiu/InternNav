# LingNav — 基于 InternNav 的 Wheeltec 小车 VLN 优化项目

**目标平台：** Wheeltec Senior_4wd_bs (Jetson Orin NX 16GB + Astra S 深度相机)
**核心思路：** 用 Qwen3-VL 零样本做语义理解（S2），NavDP 做像素目标导航（S1），两者通过 HTTP 解耦，不改动原项目核心包。

---

## 系统架构

LingNav 提供两种 S1 (NavDP) 部署模式：

### 模式 A：双服务器模式（原方案）

S2 和 S1 均在 GPU 服务器运行，Jetson 通过 HTTP 调用。

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

### 模式 B：S1 端侧部署模式（新方案）

NavDP (S1) 直接在 Jetson Orin NX 上运行，减少 S1 网络延迟；
S2 (Qwen3-VL 7B) 仍在 GPU 服务器运行（Jetson 16GB 不足以同时跑 7B VLM）。

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
│     ① RGB+Depth → S2 (HTTP) → pixel / turn / stop                   │
│     ② pixel → NavDPLocalClient (本地推理) → trajectory               │
│     ③ 更新 MPC 控制器                                                 │
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

**模式对比：**

| 指标 | 模式 A（双服务器） | 模式 B（S1 端侧） |
|------|-----------------|----------------|
| S1 推理延迟 | ~100-300ms（网络+服务器） | ~50-150ms（本地 fp16） |
| 网络依赖 | S2 + S1 均需网络 | 仅 S2 需网络 |
| Jetson 显存 | 极少（仅 ROS2） | ~200-400MB（NavDP fp16） |
| 部署复杂度 | 服务器需启动 2 个进程 | 服务器只需启动 S2 |

**S2 输出格式**（Qwen3-VL 两行纯文本）：
```
{"target": "red chair", "point_2d": [412, 680]}
↑↑←
```
- `point_2d` 归一化坐标 [0, 1000]，服务器自动转为像素坐标
- 目标不可见时：`{"target": null, "point_2d": null}` + 旋转符号 `←←`
- 到达目标时：`{"target": "chair", ...}` + `stop`

---

## 文件说明

| 文件 | 运行位置 | 说明 |
|------|---------|------|
| `wheeltec_s2_server.py` | GPU 服务器 | S2 Qwen3-VL 推理服务（`scripts/realworld2/`） |
| `lingnav_pipeline.py` | 任意 | S2+S1 联合推理管线，`LingNavPipeline` 类（`scripts/realworld2/`） |
| `lingnav_ros_client.py` | Jetson | ROS2 完整导航节点，Phase 3 主文件（`scripts/realworld2/`） |
| `test_s2_client.py` | 任意 | S2 单独测试客户端（`scripts/realworld2/`） |
| `navdp_server.py` | GPU 服务器 | S1 NavDP 推理服务，复用原有（`scripts/inference/NavDP/`）|
| `navdp_client.py` | 被 pipeline 导入 | S1 NavDP HTTP 客户端，复用原有（`scripts/inference/NavDP/`）|
| `navdp_local_client.py` | 被 ros_client 导入 | **新增** S1 NavDP 端侧本地推理客户端，无 HTTP，支持 fp16（`scripts/inference/NavDP/`）|
| `navdp_agent.py` | 被 local_client 导入 | NavDP Agent 封装，管理记忆队列（`scripts/inference/NavDP/`）|
| `wheeltec_controllers.py` | 被 ros_client 导入 | MPC + PID 控制器，复用原有（`scripts/realworld/`）|
| `wheeltec_thread_utils.py` | 被 ros_client 导入 | 读写锁，复用原有（`scripts/realworld/`）|

---

## 启动方式

### 0. 依赖安装（首次）

```bash
conda activate internnav

# S2 服务器额外依赖
pip install flask transformers>=4.57.0

# flash-attn（可选，无则自动降级到 sdpa）
pip install flash-attn --no-build-isolation
```

### 1. 启动 S1 — NavDP 推理服务器

```bash
# 在 GPU 服务器上
conda activate internnav
cd /path/to/InternNav

python scripts/inference/NavDP/navdp_server.py \
    --checkpoint /path/to/navdp_checkpoint.ckpt \
    --port 8901 \
    --host 0.0.0.0 \
    --device cuda:0
```

> NavDP checkpoint 获取：参见 `docs/new/navdp_s1_standalone_guide.md` 中的下载链接。

启动成功输出：
```
NavDP S1 Server starting on 0.0.0.0:8901
  Checkpoint: /path/to/navdp_checkpoint.ckpt
  Device: cuda:0
```

### 2. 启动 S2 — Qwen3-VL 推理服务器

```bash
# 在 GPU 服务器上（新终端）
conda activate internnav
cd /path/to/InternNav

# 7B 模型（推荐，需约 16GB 显存）
python scripts/realworld2/wheeltec_s2_server.py \
    --model_path Qwen/Qwen3-VL-7B-Instruct \
    --port 8890 \
    --host 0.0.0.0 \
    --device auto

# 内存不足时用 3B 版本
python scripts/realworld2/wheeltec_s2_server.py \
    --model_path Qwen/Qwen3-VL-3B-Instruct \
    --port 8890 \
    --host 0.0.0.0 \
    --device cuda:0
```

启动成功输出：
```
[S2] Loading processor from Qwen/Qwen3-VL-7B-Instruct …
[S2] Loaded with attn_implementation=flash_attention_2
[S2] Model ready.
[S2] Listening on http://0.0.0.0:8890
```

**S2 服务器参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_path` | `Qwen/Qwen3-VL-7B-Instruct` | 模型 ID 或本地路径 |
| `--port` | `8890` | 监听端口 |
| `--device` | `auto` | `auto` 自动多卡 / `cuda:0` 指定 |
| `--image_width` | `640` | 机器人相机宽（用于坐标转换） |
| `--image_height` | `480` | 机器人相机高 |
| `--resize_w` | `640` | 传入模型前的缩放宽（必须为 32 的倍数） |
| `--resize_h` | `480` | 传入模型前的缩放高 |
| `--max_new_tokens` | `128` | 最大生成 token 数 |

### 3. 测试 S2 服务器（可选验证）

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

### 4. 测试 LingNav 管线（S2 + S1 联调，不需要机器人）

```bash
# S2 + S1 均在线，使用真实图片
python scripts/realworld2/lingnav_pipeline.py \
    --s2_host 192.168.1.100 --s2_port 8890 \
    --s1_host 192.168.1.100 --s1_port 8901 \
    --image /path/to/test.jpg \
    --instruction "Go to the red chair"

# 仅测试 S2 连通性（NavDP 未启动时），使用随机噪声图
python scripts/realworld2/lingnav_pipeline.py \
    --s2_host 192.168.1.100 --s2_port 8890 \
    --s1_host 192.168.1.100 --s1_port 8901 \
    --random \
    --instruction "Go to the door" \
    --skip_s1
```

**`lingnav_pipeline.py` 参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--s2_host` | `127.0.0.1` | S2 服务器 IP |
| `--s2_port` | `8890` | S2 端口 |
| `--s1_host` | `127.0.0.1` | S1 服务器 IP |
| `--s1_port` | `8901` | S1 端口 |
| `--instruction` | `"Go to the red chair"` | 导航指令 |
| `--image` | — | 真实图片路径（与 `--random` 二选一）|
| `--random` | False | 使用随机噪声图像（连通性测试，不验证识别结果）|
| `--skip_s1` | False | 跳过 S1 调用（NavDP 未启动时使用）|

**预期输出（S2+S1 均在线，真实图片）：**
```
[1] 服务器连通性检查 …
  [S2] OK  {'status': 'ok', 'model': 'Qwen/Qwen3-VL-7B-Instruct'}
  [S1] reachable ...
[2] Reset pipeline …
[LingNav] Reset. instruction='Go to the red chair'
[3] 图像来源: /path/to/test.jpg
[4] 执行 step(), instruction='Go to the red chair' …

[Result]  (总耗时 2.34s)
  mode          : trajectory
  S2 target     : red chair
  S2 pixel_norm : [412, 680]
  S2 pixel_px   : [264, 326]
  S2 navigation : '↑↑←'
  S1 traj shape : (1, 24, 3)
  S1 traj[0,0]  : [0.142 -0.031  0.000]  (first waypoint, meters)
  S1 values max : 1.847
```

### 5. 启动 Jetson ROS2 客户端 — 模式 A：双服务器（原方案）

**Jetson 端依赖安装（首次）：**

```bash
pip3 install numpy requests Pillow opencv-python casadi scipy \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

sudo apt install ros-humble-cv-bridge ros-humble-message-filters
```

**启动顺序（每次运行）：**

```bash
# ── Jetson 端：Terminal 1 — 启动机器人底盘
source /opt/ros/humble/setup.bash
ros2 launch turn_on_wheeltec_robot turn_on_wheeltec_robot.launch.py

# ── Jetson 端：Terminal 2 — 启动相机
ros2 launch turn_on_wheeltec_robot wheeltec_camera.launch.py

# ── Jetson 端：Terminal 3 — 启动 LingNav 导航客户端
source /opt/ros/humble/setup.bash
cd ~/InternNav

python3 scripts/realworld2/lingnav_ros_client.py \
    --instruction "Go to the red chair" \
    --s2_host 192.168.1.100 \
    --s1_host 192.168.1.100
```

**ROS2 客户端参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--instruction` | **必填** | 导航指令，如 `"Go to the red chair"` |
| `--s2_host` | `127.0.0.1` | S2 服务器 IP |
| `--s2_port` | `8890` | S2 端口 |
| `--s1_host` | `127.0.0.1` | S1 服务器 IP |
| `--s1_port` | `8901` | S1 端口 |
| `--max_linear` | `0.25` | 最大线速度 (m/s) |
| `--max_angular` | `0.30` | 最大角速度 (rad/s) |

### 6. 启动 Jetson ROS2 客户端 — 模式 B：S1 端侧部署（新方案）

NavDP 直接在 Jetson 上运行，无需启动 `navdp_server.py`。

**前提：** 在 Jetson 上安装 NavDP 运行环境

```bash
# Jetson 端（首次）
pip3 install torch torchvision   # JetPack 5.x 自带，或按官方 wheel 安装
pip3 install numpy requests Pillow opencv-python casadi scipy

# 克隆 NavDP 项目（与 InternNav 同级目录）
cd ~
git clone https://github.com/NavDP/NavDP  # 或本地拷贝
export NAVDP_ROOT=~/NavDP

# 确认 diffusion-policy 子模块在 PYTHONPATH
export PYTHONPATH=$PYTHONPATH:~/InternNav/src/diffusion-policy
```

**启动顺序（端侧模式）：**

```bash
# ── GPU 服务器：只启动 S2，不需要 navdp_server ──
python scripts/realworld2/wheeltec_s2_server.py \
    --model_path Qwen/Qwen3-VL-7B-Instruct \
    --port 8890 --host 0.0.0.0 --device auto

# ── Jetson 端：Terminal 1 — 启动机器人底盘
source /opt/ros/humble/setup.bash
ros2 launch turn_on_wheeltec_robot turn_on_wheeltec_robot.launch.py

# ── Jetson 端：Terminal 2 — 启动相机
ros2 launch turn_on_wheeltec_robot wheeltec_camera.launch.py

# ── Jetson 端：Terminal 3 — 启动 LingNav（端侧 S1 模式）
source /opt/ros/humble/setup.bash
export NAVDP_ROOT=~/NavDP
export PYTHONPATH=$PYTHONPATH:~/InternNav/src/diffusion-policy

python3 scripts/realworld2/lingnav_ros_client.py \
    --instruction "Go to the red chair" \
    --s2_host 192.168.1.100 \
    --local_s1 \
    --s1_checkpoint ~/NavDP/checkpoints/navdp.ckpt \
    --s1_device cuda:0 \
    --s1_half                  # 推荐：fp16 节省显存、加快推理
```

**端侧模式新增参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--local_s1` | False | 开启 S1 端侧本地推理（无需 navdp_server） |
| `--s1_checkpoint` | 必填 | NavDP 权重文件路径（.ckpt）|
| `--s1_device` | `cuda:0` | NavDP 推理设备 |
| `--s1_half` | False | 开启 fp16（Jetson 推荐，约 50% 显存节省）|

---

**ROS2 话题：**

| 话题 | 方向 | 说明 |
|------|------|------|
| `/camera/color/image_raw` | 订阅 | Astra S RGB，`sensor_msgs/Image`，rgb8 |
| `/camera/depth/image` | 订阅 | Astra S 深度，`sensor_msgs/Image`，16UC1 (mm) |
| `/odom` | 订阅 | 里程计，`nav_msgs/Odometry` |
| `/cmd_vel` | 发布 | 速度指令，`geometry_msgs/Twist` |

**运行时日志示例：**

```
[LingNav] Reset. instruction='Go to the red chair'
[Plan] mode=trajectory | target=red chair | nav='↑↑←'
[Plan] mode=trajectory | target=red chair | nav='↑↑'
[Plan] mode=rotate     | target=None      | nav='←←'
[Plan] mode=stop       | target=red chair | nav='stop'
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
pipeline.reset("Go to the red chair")

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
| Phase 3.5 | ✅ 完成 | `navdp_local_client.py`：S1 NavDP 端侧部署（Jetson 本地推理，fp16，无 HTTP 依赖） |
| Phase 4 | 🔲 按需 | S1 NavDP 在小车数据上 fine-tune（SR < 50% 时触发）|

---

## 端口约定

| 服务 | 端口 | 说明 |
|------|------|------|
| S2 Qwen3-VL | **8890** | LingNav 新增，不与原项目冲突 |
| S1 NavDP | **8901** | 复用 `scripts/inference/NavDP/navdp_server.py` |
| 原 InternVLA-N1 评估服务器 | 8087 | 原项目，不受影响 |
| 原实机服务器 | 8888 | 原项目，不受影响 |
