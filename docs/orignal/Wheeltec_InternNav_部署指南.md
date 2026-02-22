# InternNav 在 Wheeltec Senior_4wd_bs 机器人上的部署指南

**作者：** Claude Code
**日期：** 2026-01-13（更新）
**适用平台：** Wheeltec Senior_4wd_bs + Jetson Orin NX 16GB
**InternNav版本：** Latest (github.com/InternRobotics/InternNav)
**参考实现：** InternNav/scripts/realworld

---

## 📝 更新说明（v2.0）

本指南已基于 **InternNav 官方实机部署代码** (`scripts/realworld/`) 进行优化，主要改进：

### ✅ 架构更新
1. **实机服务器**：采用 `http_internvla_server.py`（Flask, 端口 5801）替代通用评估服务器
2. **多线程客户端**：基于官方 `http_internvla_client.py`，实现规划线程 + 控制线程分离
3. **双控制模式**：支持 MPC 轨迹跟踪 + PID 离散动作

### ✅ 关键修改
1. **话题适配**：从 Unitree Go2 话题映射到 Wheeltec 话题
2. **相机内参**：适配 Astra S 相机（640×480, fx=fy=570.3）
3. **速度安全**：限制为 Wheeltec 安全速度（0.25 m/s, 0.5 rad/s）
4. **依赖完善**：增加 CasADi、message_filters 等必要依赖

### ✅ 代码来源
- **服务器**：`InternNav/scripts/realworld/http_internvla_server.py`
- **客户端**：改编自 `InternNav/scripts/realworld/http_internvla_client.py`（Go2 → Wheeltec）
- **控制器**：`InternNav/scripts/realworld/controllers.py`（MPC + PID）

---

## 📋 目录

1. [系统概述](#1-系统概述)
2. [硬件配置](#2-硬件配置)
3. [部署架构](#3-部署架构)
4. [System2 服务器端部署](#4-system2-服务器端部署)
5. [System1 客户端部署（机器人端）](#5-system1-客户端部署机器人端)
6. [网络配置](#6-网络配置)
7. [启动与测试](#7-启动与测试)
8. [安全与优化](#8-安全与优化)
9. [故障排除](#9-故障排除)
10. [参考资料](#10-参考资料)

---

## 1. 系统概述

### 1.1 InternNav 简介

InternNav 是上海人工智能实验室推出的具身导航开源项目，核心优势：
- **模块化设计**：支持多种仿真平台和真实机器人
- **高性能模型**：InternVLA-N1 达到业界领先水平
- **完整工具链**：从训练、评测到实机部署的全流程支持

### 1.2 部署目标

本指南将帮助您在 Wheeltec Senior_4wd_bs 轮式机器人上部署 InternNav VLN（视觉语言导航）系统，实现：
- 通过自然语言指令控制机器人导航
- 基于 RGB-D 相机的视觉感知
- 低延迟的实时运动控制

### 1.3 系统要求

| 组件 | 最低要求 | 推荐配置 |
|------|---------|---------|
| **服务器GPU** | RTX 4090 24GB | A100 40GB/80GB (多张) |
| **服务器RAM** | 24GB | 80GB+ |
| **服务器OS** | Ubuntu 20.04/22.04 | Ubuntu 22.04 LTS |
| **机器人主控** | Jetson Orin NX 16GB | Jetson Orin NX 16GB |
| **机器人OS** | Ubuntu 22.04 + ROS2 Humble | 同左 |
| **网络** | 100Mbps LAN | 1Gbps LAN |

---

## 2. 硬件配置

### 2.1 Wheeltec Senior_4wd_bs 规格

| 项目 | 配置 |
|------|------|
| **车型** | Senior_4wd_bs（高级四驱版） |
| **主控** | Nvidia Jetson Orin NX 16GB |
| **操作系统** | Ubuntu 22.04 LTS |
| **JetPack** | 6.2 |
| **ROS版本** | ROS2 Humble |
| **CUDA** | 12.6 |
| **深度相机** | Astra S（奥比中光） |
| **激光雷达** | 镭神 M10P/M10P-PHY |
| **电池** | 24V 6000mAh 磷酸铁锂 |

### 2.2 Astra S 深度相机参数

| 参数 | 数值 |
|------|------|
| **有效深度范围** | 0.6m - 4.0m (推荐范围) |
| **分辨率** | 640×480 @ 30 FPS |
| **深度FOV** | H 58.4° × V 45.5° |
| **彩色FOV** | H 63.10° × V 49.4° |
| **延迟** | 30-45 ms |

**默认相机内参矩阵（640×480）：**
```python
camera_intrinsic = [
    [570.3,   0.0, 319.5],
    [  0.0, 570.3, 239.5],
    [  0.0,   0.0,   1.0]
]
```

⚠️ **建议**：使用标定工具获得准确内参以提升导航精度

### 2.3 导航速度限制

| 场景 | 推荐值 | 说明 |
|------|--------|------|
| **直线速度** | 0.15-0.25 m/s | 最大硬件速度 2.7m/s，导航必须限速 |
| **转弯速度** | < 0.2 m/s | 避免侧滑 |
| **转弯角速度** | < 30 deg/s | 保持稳定性 |
| **原地旋转** | < 10 deg/s | 必须极慢以避免抖动 |

---

## 3. 部署架构

### 3.1 Client-Server 架构

InternNav 采用分离式架构：

```
┌─────────────────────────────────────────────────────────────┐
│                     局域网 (192.168.x.x)                      │
│                                                               │
│  ┌─────────────────────┐         ┌─────────────────────┐   │
│  │   System2 服务器     │◄───────►│   System1 客户端     │   │
│  │                     │  HTTP   │  (Jetson Orin NX)   │   │
│  │  - A100 GPU(s)      │  8087   │                     │   │
│  │  - 模型推理         │         │  - 传感器数据采集    │   │
│  │  - 动作预测         │         │  - 运动控制         │   │
│  │                     │         │  - ROS2 通信        │   │
│  └─────────────────────┘         └──────────┬──────────┘   │
│                                              │              │
│                                              ▼              │
│                                    ┌──────────────────┐    │
│                                    │  Wheeltec 机器人  │    │
│                                    │  - 传感器        │    │
│                                    │  - 电机控制      │    │
│                                    └──────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 数据流程

1. **感知阶段**：
   - Astra S 发布 RGB + Depth 图像到 ROS2 话题
   - 轮式里程计发布位姿信息到 `/odom`

2. **通信阶段**：
   - System1 订阅传感器数据
   - 将观测数据（RGB, Depth, Pose, Instruction）打包
   - HTTP POST 发送到 System2 服务器

3. **推理阶段**：
   - System2 使用 InternVLA-N1 模型进行推理
   - 预测机器人下一步动作（线速度、角速度）

4. **执行阶段**：
   - System1 接收动作指令
   - 发布到 `/cmd_vel` 话题
   - 机器人底盘执行运动

### 3.3 System1 部署位置选择

#### ✅ 推荐方案：部署在 Jetson Orin NX 上

**优点：**
- ✅ 低延迟：直接访问 ROS2 话题，无额外网络跳转
- ✅ 自主性强：机器人可独立运行
- ✅ 简化网络：只需配置 Jetson 到服务器的连接
- ✅ 参考案例一致：Unitree Go2/G1 均采用此方案

**缺点：**
- ⚠️ Jetson 性能有限（但足够 System1 任务）

**计算需求分析：**
- System1 主要任务：数据采集、HTTP 通信、控制指令发布
- 计算量不大，Jetson Orin NX 16GB 完全胜任
- 重度计算（模型推理）由服务器端承担

#### ❌ 备选方案：部署在外部边缘设备

仅在以下场景考虑：
- 需要同时运行其他重度计算任务
- Jetson 资源不足（实际不太可能）

---

## 4. System2 服务器端部署

### 4.1 硬件要求

| 硬件 | 最低配置 | 推荐配置 | 备注 |
|------|---------|---------|------|
| **GPU** | RTX 4090 24GB | A100 40GB/80GB | A100 支持推理，不支持 Isaac Sim 仿真 |
| **CPU** | 8核 | 16核+ | Intel Xeon 或 AMD EPYC |
| **内存** | 24GB | 80GB+ | 大批量推理时需要 |
| **存储** | 100GB SSD | 500GB+ NVMe SSD | 存放模型权重 |

⚠️ **关于 A100 的说明：**
- ✅ **支持模型训练和推理**（官方文档明确支持）
- ❌ **不支持 Isaac Sim 仿真环境**（仅影响仿真，不影响实机）
- ✅ **多张 A100 可用于分布式推理或多机器人服务**

### 4.2 环境配置

#### Step 1: 系统要求

```bash
# 操作系统
Ubuntu 20.04 LTS 或 Ubuntu 22.04 LTS

# NVIDIA 驱动（A100）
nvidia-driver >= 535.216.01

# 验证 GPU
nvidia-smi
```

#### Step 2: 克隆代码仓库

```bash
# 克隆 InternNav 及子模块
git clone https://github.com/InternRobotics/InternNav.git --recursive
cd InternNav
```

#### Step 3: 创建 Conda 环境

```bash
# 创建独立的模型推理环境
conda create -n internnav python=3.10 libxcb=1.14
conda activate internnav
```

#### Step 4: 安装 PyTorch

```bash
# 安装 PyTorch (CUDA 11.8)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu118
```

**注意**：如果使用其他 CUDA 版本，请参考 [PyTorch 官网](https://pytorch.org/) 选择对应版本

#### Step 5: 安装 InternNav 及实机部署依赖

```bash
# 安装带模型依赖的 InternNav
pip install -e .[model] --no-build-isolation

# 安装实机部署额外依赖
pip install flask pillow
```

### 4.3 下载模型权重

#### InternVLA-N1 预训练模型

```bash
# 方法1: 使用 huggingface-cli（推荐）
pip install huggingface_hub
huggingface-cli download InternRobotics/InternVLA-N1 \
    --local-dir checkpoints/InternVLA-N1

# 方法2: 使用 git lfs
cd checkpoints/
git lfs install
git clone https://huggingface.co/InternRobotics/InternVLA-N1
```

#### DepthAnything V2 权重

```bash
# 下载 Depth Estimation 模型
wget https://huggingface.co/Ashoka74/Placement/resolve/main/depth_anything_v2_vits.pth \
    -O checkpoints/depth_anything_v2_vits.pth
```

#### 验证目录结构

```bash
InternNav/
├── checkpoints/
│   ├── InternVLA-N1/
│   │   ├── config.json
│   │   ├── model-00001-of-00004.safetensors
│   │   ├── model-00002-of-00004.safetensors
│   │   ├── model-00003-of-00004.safetensors
│   │   └── model-00004-of-00004.safetensors
│   └── depth_anything_v2_vits.pth
├── scripts/
└── ...
```

### 4.4 启动模型服务器

⚠️ **重要**：InternNav 提供两种服务器实现，根据使用场景选择：

#### 方案A: 实机部署服务器（推荐用于真实机器人）

使用 `scripts/realworld/http_internvla_server.py`，基于 Flask，专为实机优化：

**编辑服务器配置**

修改 `scripts/realworld/http_internvla_server.py` 中的参数（第 83-92 行）：

```python
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--model_path", type=str, default="checkpoints/InternVLA-N1")
parser.add_argument("--resize_w", type=int, default=384)
parser.add_argument("--resize_h", type=int, default=384)
parser.add_argument("--num_history", type=int, default=8)
args = parser.parse_args()

# ⚠️ 修改相机内参为 Astra S 参数（4x4 齐次矩阵格式）
args.camera_intrinsic = np.array([
    [570.3, 0.0, 319.5, 0.0],
    [0.0, 570.3, 239.5, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
])
```

**启动服务**

```bash
# 激活环境
conda activate internnav
cd ~/InternNav

# 启动实机服务器（端口 8888）
python scripts/realworld/http_internvla_server.py 

```
**预期输出**

```
read http data cost 0.002
init reset model!!!
 * Serving Flask app 'http_internvla_server'
 * Running on http://0.0.0.0:8888
Press CTRL+C to quit
```

#### 方案B: 通用评估服务器（用于仿真或多模式）

使用 `scripts/eval/start_server.py`，支持多种模式和配置：

```bash
# 启动通用服务器（端口 8087）
python scripts/eval/start_server.py --port 8087
```

**对比：**

| 特性 | 实机服务器 (A) | 通用服务器 (B) |
|------|--------------|--------------|
| **脚本路径** | `scripts/realworld/http_internvla_server.py` | `scripts/eval/start_server.py` |
| **端口** | 8888 | 8087 |
| **框架** | Flask | FastAPI/Uvicorn |
| **适用场景** | 真实机器人部署 | 仿真评估 |
| **配置方式** | 修改脚本 | 配置文件 |
| **推荐使用** | ✅ Wheeltec 实机 | 仿真环境 |

**本指南采用方案A（实机服务器）**

#### 测试服务器

从另一个终端测试：

```bash
# 测试服务器连通性（方案A）
curl http://115.190.160.32:8888/  #公网
curl http://192.168.1.224:8888/   #局域网
curl http://127.0.0.1:5801       #内网

# 或使用 Python 测试
python3 << 'EOF'
import requests
import numpy as np
from PIL import Image
import io
import json
import time

print("=" * 60)
print("Testing InternVLA-N1 HTTP Server")
print("=" * 60)

# 准备测试数据
print("\n[1/4] Preparing RGB image (480x640x3)...")
rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

print("[2/4] Preparing depth image (480x640)...")
# 深度值范围：1000-50000 (0.1米到5米，单位是毫米*10)
depth = np.random.randint(10000, 50000, (480, 640), dtype=np.uint32)

print("[3/4] Converting to image format...")
# RGB 图像
rgb_img = Image.fromarray(rgb, mode='RGB')
rgb_bytes = io.BytesIO()
rgb_img.save(rgb_bytes, format='JPEG')
rgb_bytes.seek(0)

# 深度图 (32-bit integer)
depth_img = Image.fromarray(depth, mode='I')
depth_bytes = io.BytesIO()
depth_img.save(depth_bytes, format='PNG')
depth_bytes.seek(0)

print("[4/4] Sending POST request to http://127.0.0.1:5801/eval_dual...")
print("\nRequest payload:")
print("  - RGB image: JPEG format")
print("  - Depth image: PNG format (32-bit)")
print("  - JSON data: {'reset': True, 'idx': 0}")
print()

# 发送请求
try:
    start = time.time()
    response = requests.post(
        'http://127.0.0.1:5801/eval_dual',
        files={
            'image': ('rgb.jpg', rgb_bytes, 'image/jpeg'),
            'depth': ('depth.png', depth_bytes, 'image/png')
        },
        data={
            'json': json.dumps({
                "reset": True,
                "idx": 0
            })
        },
        timeout=60
    )
    elapsed = time.time() - start
    
    print(f"Response received in {elapsed:.2f} seconds")
    print(f"Status Code: {response.status_code}")
    print()
    
    if response.status_code == 200:
        result = response.json()
        print("=" * 60)
        print("SUCCESS! Server Response:")
        print("=" * 60)
        print(json.dumps(result, indent=2))
        print()
        
        if 'discrete_action' in result:
            print(f"Discrete Action: {result['discrete_action']}")
        if 'trajectory' in result:
            print(f"Trajectory shape: {len(result['trajectory'])} waypoints")
        if 'pixel_goal' in result:
            print(f"Pixel Goal: {result['pixel_goal']}")
    else:
        print("ERROR Response:")
        print(response.text)
        
except requests.exceptions.Timeout:
    print("ERROR: Request timeout (60s)")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
EOF
```

### 4.5 多 GPU 配置（可选）

如果您有多张 A100，可以：

**方案1: 启动多个服务实例**

```bash
# Terminal 1: GPU 0
CUDA_VISIBLE_DEVICES=0 python scripts/eval/start_server.py --port 8087

# Terminal 2: GPU 1
CUDA_VISIBLE_DEVICES=1 python scripts/eval/start_server.py --port 8088

# Terminal 3: GPU 2
CUDA_VISIBLE_DEVICES=2 python scripts/eval/start_server.py --port 8089
```

**方案2: 使用分布式推理**

参考官方文档中的 Ray 分布式配置（适用于大规模评测）

---

## 5. System1 客户端部署（机器人端）

### 5.1 环境准备

#### Step 1: SSH 连接到 Jetson

```bash
# 方法1: 通过有线网络（推荐用于初始配置）
ssh -Y wheeltec@192.168.137.100

# 方法2: 通过 WiFi（配置后使用）
ssh -Y wheeltec@192.168.137.100
```

默认密码通常为 `dongguan`

#### Step 2: 验证 ROS2 环境

```bash
# 检查 ROS2 版本
printenv | grep ROS
# 应显示 ROS_DISTRO=humble

# 查看可用话题（需先启动机器人）
source /opt/ros/humble/setup.bash
ros2 topic list
```

预期看到的核心话题：
```
/camera/rgb/image_raw
/camera/depth/image
/camera/rgb/camera_info
/odom
/scan
/cmd_vel
/tf
```

### 5.2 安装依赖

#### 安装 Python 依赖

```bash
# 更新 pip
pip3 install --upgrade pip

# 安装 InternNav 客户端依赖
pip3 install numpy requests Pillow opencv-python \
-i https://pypi.tuna.tsinghua.edu.cn/simple

# ⚠️ 安装控制器依赖（MPC 需要）
pip3 install casadi scipy -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装 ROS2 Python 库（如未安装）
sudo apt install python3-rosdep python3-colcon-common-extensions 

# 安装 cv_bridge（用于图像转换）
sudo apt install ros-humble-cv-bridge

# 安装 message_filters（用于传感器同步）
sudo apt install ros-humble-message-filters
```

#### 验证相机驱动

```bash
# 启动相机节点
ros2 launch turn_on_wheeltec_robot wheeltec_camera.launch.py

# 在另一终端查看图像
ros2 run rqt_image_view rqt_image_view
# 选择 /camera/rgb/image_raw 查看 RGB 图像
# 选择 /camera/depth/image 查看深度图像
```

### 5.3 部署 InternNav 客户端代码

#### 克隆代码（在 Jetson 上）

```bash
# 克隆到机器人本地
cd ~
git clone https://github.com/InternRobotics/InternNav.git 

cd InternNav

# 复制实机脚本到工作目录
cp scripts/realworld/controllers.py scripts/realworld/wheeltec_controllers.py
cp scripts/realworld/thread_utils.py scripts/realworld/wheeltec_thread_utils.py
```

#### 创建 Wheeltec 客户端脚本

基于官方 `http_internvla_client.py` 改编，创建 `scripts/realworld/wheeltec_client.py`：

```python
#!/usr/bin/env python3
"""
Wheeltec InternNav Client - 改编自 InternNav scripts/realworld/http_internvla_client.py
适配 Wheeltec Senior_4wd_bs 机器人平台
主要修改：
1. 话题名称从 Unitree Go2 改为 Wheeltec
2. 相机内参修改为 Astra S
3. 服务器地址和端口配置
"""

import copy
import io
import json
import math
import threading
import time
from collections import deque
from enum import Enum

import numpy as np
import rclpy
import requests
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from PIL import Image as PIL_Image
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

# 导入控制器和线程工具
from wheeltec_controllers import Mpc_controller, PID_controller
from wheeltec_thread_utils import ReadWriteLock


class ControlMode(Enum):
    PID_Mode = 1
    MPC_Mode = 2


# ==================== 全局变量 ====================
policy_init = True
mpc = None
pid = PID_controller(Kp_trans=2.0, Kd_trans=0.0, Kp_yaw=1.5, Kd_yaw=0.0, max_v=0.25, max_w=0.5)
http_idx = -1
first_running_time = 0.0
last_pixel_goal = None
last_s2_step = -1
manager = None
current_control_mode = ControlMode.MPC_Mode
trajs_in_world = None

desired_v, desired_w = 0.0, 0.0
rgb_depth_rw_lock = ReadWriteLock()
odom_rw_lock = ReadWriteLock()
mpc_rw_lock = ReadWriteLock()

frame_data = {}

# ==================== 配置参数 ====================
SERVER_HOST = '192.168.1.100'  # ⚠️ 修改为你的服务器 IP
SERVER_PORT = 5801
SERVER_URL = f'http://{SERVER_HOST}:{SERVER_PORT}/eval_dual'

# Astra S 相机内参 (640x480)
CAMERA_INTRINSIC = np.array([
    [570.3, 0.0, 319.5, 0.0],
    [0.0, 570.3, 239.5, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
])

# ⚠️ 修改为你的导航指令
NAVIGATION_INSTRUCTION = "Go to the red chair"

# 速度限制（Wheeltec 安全参数）
MAX_LINEAR_VEL = 0.25   # m/s
MAX_ANGULAR_VEL = 0.5   # rad/s


# ==================== 服务器通信函数 ====================
def dual_sys_eval(image_bytes, depth_bytes, url=SERVER_URL):
    """向服务器发送图像并获取动作"""
    global policy_init, http_idx, first_running_time

    data = {"reset": policy_init, "idx": http_idx}
    json_data = json.dumps(data)

    policy_init = False
    files = {
        'image': ('rgb_image', image_bytes, 'image/jpeg'),
        'depth': ('depth_image', depth_bytes, 'image/png'),
    }

    start = time.time()
    try:
        response = requests.post(url, files=files, data={'json': json_data}, timeout=10)
        print(f"Server response: {response.text}")
        http_idx += 1
        if http_idx == 0:
            first_running_time = time.time()
        print(f"HTTP request {http_idx} took {time.time() - start:.3f}s")
        return json.loads(response.text)
    except requests.exceptions.RequestException as e:
        print(f"HTTP request failed: {e}")
        return {}


# ==================== 控制线程 ====================
def control_thread():
    """控制执行线程：根据控制模式执行 MPC 或 PID 控制"""
    global desired_v, desired_w

    while True:
        global current_control_mode

        if current_control_mode == ControlMode.MPC_Mode:
            # MPC 控制模式
            odom_rw_lock.acquire_read()
            odom = manager.odom.copy() if manager.odom else None
            odom_rw_lock.release_read()

            if mpc is not None and manager is not None and odom is not None:
                local_mpc = mpc
                opt_u_controls, opt_x_states = local_mpc.solve(np.array(odom))
                v, w = opt_u_controls[0, 0], opt_u_controls[0, 1]

                # 限速
                v = np.clip(v, 0, MAX_LINEAR_VEL)
                w = np.clip(w, -MAX_ANGULAR_VEL, MAX_ANGULAR_VEL)

                desired_v, desired_w = v, w
                manager.move(v, 0.0, w)

        elif current_control_mode == ControlMode.PID_Mode:
            # PID 控制模式
            odom_rw_lock.acquire_read()
            odom = manager.odom.copy() if manager.odom else None
            odom_rw_lock.release_read()

            homo_odom = manager.homo_odom.copy() if manager.homo_odom is not None else None
            vel = manager.vel.copy() if manager.vel is not None else None
            homo_goal = manager.homo_goal.copy() if manager.homo_goal is not None else None

            if homo_odom is not None and vel is not None and homo_goal is not None:
                v, w, e_p, e_r = pid.solve(homo_odom, homo_goal, vel)
                if v < 0.0:
                    v = 0.0

                # 限速
                v = np.clip(v, 0, MAX_LINEAR_VEL)
                w = np.clip(w, -MAX_ANGULAR_VEL, MAX_ANGULAR_VEL)

                desired_v, desired_w = v, w
                manager.move(v, 0.0, w)

        time.sleep(0.1)


# ==================== 规划线程 ====================
def planning_thread():
    """规划线程：定期向服务器请求并更新轨迹/动作"""
    global trajs_in_world

    while True:
        start_time = time.time()
        DESIRED_TIME = 0.3
        time.sleep(0.05)

        if not manager.new_image_arrived:
            time.sleep(0.01)
            continue

        manager.new_image_arrived = False

        # 读取传感器数据
        rgb_depth_rw_lock.acquire_read()
        rgb_bytes = copy.deepcopy(manager.rgb_bytes)
        depth_bytes = copy.deepcopy(manager.depth_bytes)
        infer_rgb = copy.deepcopy(manager.rgb_image)
        infer_depth = copy.deepcopy(manager.depth_image)
        rgb_time = manager.rgb_time
        rgb_depth_rw_lock.release_read()

        # 时间同步：找到最接近的 odom
        odom_rw_lock.acquire_read()
        min_diff = 1e10
        odom_infer = None
        for odom in manager.odom_queue:
            diff = abs(odom[0] - rgb_time)
            if diff < min_diff:
                min_diff = diff
                odom_infer = copy.deepcopy(odom[1])
        odom_rw_lock.release_read()

        if odom_infer is not None and rgb_bytes is not None and depth_bytes is not None:
            # 保存帧数据
            global frame_data
            frame_data[http_idx] = {
                'infer_rgb': copy.deepcopy(infer_rgb),
                'infer_depth': copy.deepcopy(infer_depth),
                'infer_odom': copy.deepcopy(odom_infer),
            }
            if len(frame_data) > 100:
                del frame_data[min(frame_data.keys())]

            # 请求服务器
            response = dual_sys_eval(rgb_bytes, depth_bytes)

            global current_control_mode

            # 处理轨迹输出 (MPC 模式)
            if 'trajectory' in response:
                trajectory = response['trajectory']
                trajs_in_world = []
                odom = odom_infer
                traj_len = np.linalg.norm(trajectory[-1][:2])
                print(f"Received trajectory, length: {traj_len:.2f}m")

                # 转换轨迹到世界坐标
                for i, traj in enumerate(trajectory):
                    if i < 3:
                        continue
                    x_, y_, yaw_ = odom[0], odom[1], odom[2]

                    w_T_b = np.array([
                        [np.cos(yaw_), -np.sin(yaw_), 0, x_],
                        [np.sin(yaw_), np.cos(yaw_), 0, y_],
                        [0.0, 0.0, 1.0, 0],
                        [0.0, 0.0, 0.0, 1.0],
                    ])
                    w_P = (w_T_b @ (np.array([traj[0], traj[1], 0.0, 1.0])).T)[:2]
                    trajs_in_world.append(w_P)

                trajs_in_world = np.array(trajs_in_world)
                manager.last_trajs_in_world = trajs_in_world

                # 更新 MPC 控制器
                mpc_rw_lock.acquire_write()
                global mpc
                if mpc is None:
                    mpc = Mpc_controller(np.array(trajs_in_world))
                else:
                    mpc.update_ref_traj(np.array(trajs_in_world))
                manager.request_cnt += 1
                mpc_rw_lock.release_write()

                current_control_mode = ControlMode.MPC_Mode

            # 处理离散动作输出 (PID 模式)
            elif 'discrete_action' in response:
                actions = response['discrete_action']
                if actions != [5] and actions != [9]:  # 5=look down, 9=stop
                    manager.incremental_change_goal(actions)
                    current_control_mode = ControlMode.PID_Mode
        else:
            print(f"Skipping planning: odom={odom_infer is not None}, "
                  f"rgb={rgb_bytes is not None}, depth={depth_bytes is not None}")
            time.sleep(0.1)

        time.sleep(max(0, DESIRED_TIME - (time.time() - start_time)))


# ==================== ROS2 管理节点 ====================
class WheeltecManager(Node):
    """Wheeltec 机器人管理节点 - 改编自 Go2Manager"""

    def __init__(self):
        super().__init__('wheeltec_manager')

        # ⚠️ Wheeltec 话题名称 (不同于 Go2)
        rgb_sub = Subscriber(self, Image, "/camera/color/image_raw")
        depth_sub = Subscriber(self, Image, "/camera/depth/image")

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # 时间同步器
        self.synchronizer = ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub], 1, 0.1
        )
        self.synchronizer.registerCallback(self.rgb_depth_callback)

        # ⚠️ Wheeltec 话题名称
        self.odom_sub = self.create_subscription(
            Odometry, "/odom", self.odom_callback, qos_profile
        )

        # 发布器
        self.control_pub = self.create_publisher(Twist, '/cmd_vel', 5)

        # 成员变量
        self.cv_bridge = CvBridge()
        self.rgb_image = None
        self.rgb_bytes = None
        self.depth_image = None
        self.depth_bytes = None
        self.new_image_arrived = False
        self.rgb_time = 0.0

        self.odom = None
        self.linear_vel = 0.0
        self.angular_vel = 0.0
        self.request_cnt = 0
        self.odom_cnt = 0
        self.odom_queue = deque(maxlen=50)
        self.odom_timestamp = 0.0

        self.last_trajs_in_world = None
        self.homo_odom = None
        self.homo_goal = None
        self.vel = None

        self.get_logger().info('Wheeltec InternNav Manager initialized')
        self.get_logger().info(f'Server: {SERVER_URL}')
        self.get_logger().info(f'Instruction: {NAVIGATION_INSTRUCTION}')

    def rgb_depth_callback(self, rgb_msg, depth_msg):
        """RGB-Depth 同步回调"""
        # 处理 RGB
        raw_image = self.cv_bridge.imgmsg_to_cv2(rgb_msg, 'rgb8')[:, :, :]
        self.rgb_image = raw_image
        image = PIL_Image.fromarray(self.rgb_image)
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='JPEG')
        image_bytes.seek(0)

        # 处理 Depth (Astra S 输出 16UC1, 单位毫米)
        raw_depth = self.cv_bridge.imgmsg_to_cv2(depth_msg, '16UC1')
        raw_depth[np.isnan(raw_depth)] = 0
        raw_depth[np.isinf(raw_depth)] = 0
        self.depth_image = raw_depth / 1000.0  # 转换为米
        self.depth_image[np.where(self.depth_image < 0)] = 0

        # 编码为 PNG (保存为 uint16, 单位 0.1mm)
        depth = (np.clip(self.depth_image * 10000.0, 0, 65535)).astype(np.uint16)
        depth = PIL_Image.fromarray(depth)
        depth_bytes = io.BytesIO()
        depth.save(depth_bytes, format='PNG')
        depth_bytes.seek(0)

        # 线程安全更新
        rgb_depth_rw_lock.acquire_write()
        self.rgb_bytes = image_bytes
        self.rgb_time = rgb_msg.header.stamp.sec + rgb_msg.header.stamp.nanosec / 1.0e9
        self.depth_bytes = depth_bytes
        rgb_depth_rw_lock.release_write()

        self.new_image_arrived = True

    def odom_callback(self, msg):
        """里程计回调"""
        self.odom_cnt += 1

        # 提取位姿
        odom_rw_lock.acquire_write()
        zz = msg.pose.pose.orientation.z
        ww = msg.pose.pose.orientation.w
        yaw = math.atan2(2 * zz * ww, 1 - 2 * zz * zz)
        self.odom = [msg.pose.pose.position.x, msg.pose.pose.position.y, yaw]
        self.odom_queue.append((time.time(), copy.deepcopy(self.odom)))
        self.odom_timestamp = time.time()
        self.linear_vel = msg.twist.twist.linear.x
        self.angular_vel = msg.twist.twist.angular.z
        odom_rw_lock.release_write()

        # 构建齐次变换矩阵
        R0 = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        self.homo_odom = np.eye(4)
        self.homo_odom[:2, :2] = R0
        self.homo_odom[:2, 3] = [msg.pose.pose.position.x, msg.pose.pose.position.y]
        self.vel = [msg.twist.twist.linear.x, msg.twist.twist.angular.z]

        # 初始化目标为当前位置
        if self.odom_cnt == 1:
            self.homo_goal = self.homo_odom.copy()

    def incremental_change_goal(self, actions):
        """根据离散动作增量更新目标"""
        if self.homo_goal is None:
            raise ValueError("Please initialize homo_goal before changing it!")

        homo_goal = self.homo_odom.copy()
        for each_action in actions:
            if each_action == 0:  # No action
                pass
            elif each_action == 1:  # Forward
                yaw = math.atan2(homo_goal[1, 0], homo_goal[0, 0])
                homo_goal[0, 3] += 0.25 * np.cos(yaw)
                homo_goal[1, 3] += 0.25 * np.sin(yaw)
            elif each_action == 2:  # Turn left
                angle = math.radians(15)
                rotation_matrix = np.array([
                    [math.cos(angle), -math.sin(angle), 0],
                    [math.sin(angle), math.cos(angle), 0],
                    [0, 0, 1]
                ])
                homo_goal[:3, :3] = np.dot(rotation_matrix, homo_goal[:3, :3])
            elif each_action == 3:  # Turn right
                angle = -math.radians(15.0)
                rotation_matrix = np.array([
                    [math.cos(angle), -math.sin(angle), 0],
                    [math.sin(angle), math.cos(angle), 0],
                    [0, 0, 1]
                ])
                homo_goal[:3, :3] = np.dot(rotation_matrix, homo_goal[:3, :3])

        self.homo_goal = homo_goal

    def move(self, vx, vy, vyaw):
        """发布运动指令"""
        request = Twist()
        request.linear.x = float(vx)
        request.linear.y = 0.0
        request.angular.z = float(vyaw)

        self.control_pub.publish(request)


# ==================== 主函数 ====================
if __name__ == '__main__':
    print("=" * 60)
    print("Wheeltec InternNav Client")
    print("=" * 60)
    print(f"Server: {SERVER_URL}")
    print(f"Instruction: {NAVIGATION_INSTRUCTION}")
    print(f"Max velocity: {MAX_LINEAR_VEL} m/s, {MAX_ANGULAR_VEL} rad/s")
    print("=" * 60)

    # 创建线程
    control_thread_instance = threading.Thread(target=control_thread)
    planning_thread_instance = threading.Thread(target=planning_thread)
    control_thread_instance.daemon = True
    planning_thread_instance.daemon = True

    # 初始化 ROS2
    rclpy.init()

    try:
        manager = WheeltecManager()

        # 启动线程
        control_thread_instance.start()
        planning_thread_instance.start()

        print("Threads started, spinning ROS2 node...")
        rclpy.spin(manager)

    except KeyboardInterrupt:
        print("\nShutdown requested...")
    finally:
        # 停止机器人
        if manager:
            manager.move(0.0, 0.0, 0.0)
            manager.destroy_node()
        rclpy.shutdown()
        print("Shutdown complete.")
```

#### 重点说明

**与官方实现的主要差异：**

1. **话题名称适配**：
   - RGB: `/camera/camera/color/image_raw` → `/camera/rgb/image_raw`
   - Depth: `/camera/camera/aligned_depth_to_color/image_raw` → `/camera/depth/image`
   - Odom: `/odom_bridge` → `/odom`
   - Cmd: `/cmd_vel_bridge` → `/cmd_vel`

2. **相机内参修改**：
   - 从 Unitree Go2 的内参改为 Astra S 内参

3. **服务器地址**：
   - 端口从 8087 改为 5801（实机服务器）

4. **速度限制**：
   - 降低为 Wheeltec 安全速度范围

**多线程架构：**

- **planning_thread**: 0.3s 周期请求服务器，更新轨迹/动作
- **control_thread**: 0.1s 周期执行运动控制
- **ReadWriteLock**: 保证多线程数据安全

**双控制模式：**

- **MPC 模式**: 轨迹跟踪，使用 CasADi 优化
- **PID 模式**: 离散动作执行

#### 设置执行权限

```bash
chmod +x scripts/realworld/wheeltec_client.py
```

---

## 6. 网络配置

### 6.1 网络拓扑

```
服务器 (A100)              Jetson (Orin NX)         Wheeltec Robot
192.168.1.100       <--->  192.168.1.50      <--->  (ROS2 通信)
   Port 5801                 WiFi/Ethernet           /cmd_vel, /odom
   (实机服务器)                                        /camera/*
```

### 6.2 服务器网络配置

#### 配置静态 IP（推荐）

编辑 `/etc/netplan/01-netcfg.yaml`：

```yaml
network:
  version: 2
  ethernets:
    eth0:  # 替换为实际网卡名
      dhcp4: no
      addresses: [192.168.1.100/24]
      gateway4: 192.168.1.1
      nameservers:
        addresses: [8.8.8.8, 8.8.4.4]
```

应用配置：

```bash
sudo netplan apply
```

#### 防火墙配置

```bash
# 允许 8087 端口
sudo ufw allow 8087/tcp

# 验证
sudo ufw status
```

### 6.3 Jetson 网络配置

#### 配置 WiFi 连接

```bash
# 方法1: 使用 nmtui（推荐）
sudo nmtui

# 方法2: 使用 nmcli
sudo nmcli device wifi connect <SSID> password <PASSWORD>
```

#### 配置静态 IP

编辑 WiFi 连接配置：

```bash
sudo nmcli connection modify <connection-name> \
    ipv4.addresses 192.168.1.50/24 \
    ipv4.gateway 192.168.1.1 \
    ipv4.dns "8.8.8.8 8.8.4.4" \
    ipv4.method manual

sudo nmcli connection up <connection-name>
```

#### 测试连通性

```bash
# Ping 服务器
ping 192.168.1.100

# 测试服务端口
curl http://192.168.1.100:8087/health

# 测量延迟
ping -c 10 192.168.1.100
```

### 6.4 ROS2 域配置

为避免与其他 ROS2 设备冲突，设置独立的域 ID：

在 Jetson 的 `~/.bashrc` 中添加：

```bash
export ROS_DOMAIN_ID=42  # 选择 0-232 之间的值
```

### 6.5 带宽优化

#### 压缩图像传输（可选）

如果网络带宽有限，可以在客户端压缩图像后再发送：

```python
import cv2

# 压缩 RGB
_, rgb_encoded = cv2.imencode('.jpg', rgb_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
rgb_compressed = rgb_encoded.tobytes()

# 压缩深度（PNG 无损）
_, depth_encoded = cv2.imencode('.png', (depth_image * 1000).astype(np.uint16))
depth_compressed = depth_encoded.tobytes()
```

服务器端对应解压缩。

---

## 7. 启动与测试

### 7.1 启动顺序

#### Step 1: 启动服务器（在工作站上）

```bash
# Terminal 1: 实机服务器
cd ~/InternNav
conda activate internnav

# ⚠️ 确保已修改 http_internvla_server.py 中的相机内参
python scripts/realworld/http_internvla_server.py 
```

等待看到：
```
read http data cost 0.002
init reset model!!!
 * Serving Flask app 'http_internvla_server'
 * Running on http://0.0.0.0:8888
Press CTRL+C to quit
```

#### Step 2: 启动机器人底盘（在 Jetson 上）

```bash
# SSH to Jetson
ssh wheeltec@192.168.137.100

# Terminal 1: 启动底盘
source /opt/ros/humble/setup.bash
ros2 launch turn_on_wheeltec_robot turn_on_wheeltec_robot.launch.py
```

验证话题：
```bash
# Terminal 2
ros2 topic list
ros2 topic echo /odom --once
```

# 应显示
```
/PowerVoltage
/RangerAvoidFlag
/chassis_security
/cmd_vel
/diagnostics
/imu/data_raw
/joint_states
/odom
/odom_combined
/parameter_events
/robot_charging_current
/robot_charging_flag
/robot_charging_mode
/robot_description
/robot_recharge_flag
/robot_red_flag
/rosout
/set_pose
/set_rgb_color
/tf
/tf_static
/ultrasonic_data_A
/ultrasonic_data_B
/ultrasonic_data_C
/ultrasonic_data_D
/ultrasonic_data_E
/ultrasonic_data_F
```

#### Step 3: 启动相机（在 Jetson 上）

```bash
# Terminal 3
ros2 launch turn_on_wheeltec_robot wheeltec_camera.launch.py
```

验证图像：
```bash
# Terminal 4
ros2 topic hz /camera/rgb/image_raw
ros2 topic hz /camera/color/image_raw
# 应显示约 30 Hz
```

#### Step 4: 启动导航客户端（在 Jetson 上）

```bash
# Terminal 5
cd ~/InternNav
python3 scripts/realworld/wheeltec_client.py
```

### 7.2 功能测试

#### 测试1: 手动发送单帧推理

在服务器端测试：

```bash
conda activate internnav
python3 << EOF
from internnav.configs.agent import AgentCfg
from internnav.utils import AgentClient
import numpy as np

# 初始化客户端
agent_cfg = AgentCfg(
    server_host='localhost',
    server_port=8888,
    model_name='internvla_n1',
    model_settings={
        'policy_name': "InternVLAN1_Policy",
        'model_path': "checkpoints/InternVLA-N1",
        'camera_intrinsic': [[570.3, 0.0, 319.5],
                             [0.0, 570.3, 239.5],
                             [0.0, 0.0, 1.0]],
        'width': 640,
        'height': 480,
        'hfov': 79,
        'resize_w': 384,
        'resize_h': 384,
        'device': 'cuda:0',
    }
)

agent = AgentClient(agent_cfg)

# 准备假数据
fake_obs = {
    'rgb': np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
    'depth': np.random.rand(480, 640).astype(np.float32),
    'instruction': 'Go forward'
}

# 推理
action = agent.step(fake_obs)
print(f"Action: {action}")
EOF
```

#### 测试2: 键盘控制验证

在启动导航客户端前，先用键盘控制验证机器人响应：

```bash
ros2 run wheeltec_robot_keyboard wheeltec_keyboard
```

测试前进、后退、转向是否正常。

#### 测试3: 简单导航任务

给定指令：
```python
self.instruction = "Move forward 2 meters"
```

观察机器人是否：
1. 平稳启动
2. 保持直线
3. 接近目标后减速
4. 最终停止

### 7.3 性能监控

#### 监控推理延迟

在客户端脚本中添加计时：

```python
import time

start_time = time.time()
action = self.request_action(observation)
latency = time.time() - start_time

self.get_logger().info(f'Inference latency: {latency*1000:.1f} ms')
```

**目标延迟：**
- 本地网络：< 100 ms
- WiFi：< 200 ms

#### 监控系统资源

**服务器端（GPU）：**
```bash
watch -n 1 nvidia-smi
```

关注：
- GPU 利用率
- 显存占用
- 温度

**Jetson 端（CPU/内存）：**
```bash
htop
```

关注：
- CPU 使用率
- 内存占用
- 网络流量

#### 监控 ROS2 话题频率

```bash
# 检查相机频率
ros2 topic hz /camera/rgb/image_raw

# 检查控制频率
ros2 topic hz /cmd_vel
```

---

## 8. 安全与优化

### 8.1 碰撞避障（必须）

⚠️ **重要**：InternVLA-N1 基础模型不包含基于规则的避障，必须显式添加深度碰撞检测！

#### 增强避障逻辑

在 `wheeltec_client.py` 中改进 `check_collision` 函数：

```python
def check_collision(self, depth):
    """增强的碰撞检测"""
    h, w = depth.shape

    # 定义多个检测区域
    regions = {
        'front_center': depth[h//3:2*h//3, 2*w//5:3*w//5],  # 前方中心
        'front_left': depth[h//3:2*h//3, w//5:2*w//5],      # 前方左侧
        'front_right': depth[h//3:2*h//3, 3*w//5:4*w//5],   # 前方右侧
    }

    # 检测各区域最小距离
    collision_risk = False
    for region_name, region in regions.items():
        valid_depths = region[(region > 0.1) & (region < 4.0)]

        if len(valid_depths) > 10:  # 至少10个有效点
            min_depth = np.min(valid_depths)
            median_depth = np.median(valid_depths)

            # 动态阈值：速度越快，安全距离越大
            safety_distance = 0.6 + abs(self.current_linear_vel) * 0.5

            if min_depth < safety_distance:
                self.get_logger().warn(
                    f'Collision risk in {region_name}: {min_depth:.2f}m < {safety_distance:.2f}m'
                )
                collision_risk = True

    return collision_risk
```

#### 紧急停止机制

添加物理急停按钮监听（如果硬件支持）：

```python
from std_msgs.msg import Bool

self.estop_sub = self.create_subscription(
    Bool, '/emergency_stop', self.estop_callback, 10)
self.is_estopped = False

def estop_callback(self, msg):
    self.is_estopped = msg.data
    if self.is_estopped:
        self.stop_robot()
        self.get_logger().error('EMERGENCY STOP ACTIVATED!')
```

### 8.2 速度限制与平滑

#### 速度斜坡（避免急加速）

```python
class VelocitySmoother:
    def __init__(self, max_accel=0.5, max_angular_accel=1.0):
        self.max_accel = max_accel  # m/s^2
        self.max_angular_accel = max_angular_accel  # rad/s^2
        self.prev_linear = 0.0
        self.prev_angular = 0.0

    def smooth(self, target_linear, target_angular, dt=0.1):
        """平滑速度变化"""
        # 限制线速度加速度
        delta_linear = target_linear - self.prev_linear
        if abs(delta_linear) > self.max_accel * dt:
            delta_linear = np.sign(delta_linear) * self.max_accel * dt

        # 限制角速度加速度
        delta_angular = target_angular - self.prev_angular
        if abs(delta_angular) > self.max_angular_accel * dt:
            delta_angular = np.sign(delta_angular) * self.max_angular_accel * dt

        # 更新
        self.prev_linear += delta_linear
        self.prev_angular += delta_angular

        return self.prev_linear, self.prev_angular

# 在客户端初始化
self.vel_smoother = VelocitySmoother()

# 在 execute_action 中使用
smooth_linear, smooth_angular = self.vel_smoother.smooth(cmd.linear.x, cmd.angular.z)
cmd.linear.x = smooth_linear
cmd.angular.z = smooth_angular
```

### 8.3 电池管理

#### 监控电池电压

```bash
# 在 Jetson 上查看电池信息（如果发布到 ROS2）
ros2 topic echo /battery_status
```

添加低电量保护：

```python
from sensor_msgs.msg import BatteryState

self.battery_sub = self.create_subscription(
    BatteryState, '/battery_status', self.battery_callback, 10)
self.battery_voltage = 24.0

def battery_callback(self, msg):
    self.battery_voltage = msg.voltage

    if self.battery_voltage < 22.0:
        self.get_logger().error('LOW BATTERY! Stopping navigation.')
        self.stop_robot()
        rclpy.shutdown()
```

### 8.4 日志与调试

#### 启用详细日志

```python
# 在客户端添加
import logging
logging.basicConfig(level=logging.DEBUG)

# 记录关键信息
self.get_logger().info(f'RGB shape: {self.rgb_image.shape}')
self.get_logger().info(f'Depth min/max: {self.depth_image.min():.2f}/{self.depth_image.max():.2f}')
self.get_logger().info(f'Odom: x={self.robot_pose.position.x:.2f}, y={self.robot_pose.position.y:.2f}')
```

#### 保存运行数据

```python
import pickle
from datetime import datetime

def save_trajectory_data(self):
    """保存轨迹数据用于分析"""
    data = {
        'timestamp': datetime.now().isoformat(),
        'rgb_frames': self.rgb_history,
        'depth_frames': self.depth_history,
        'poses': self.pose_history,
        'actions': self.action_history,
    }

    filename = f"trajectory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

    self.get_logger().info(f'Saved trajectory data to {filename}')
```

### 8.5 性能优化

#### Jetson 功率模式

```bash
# 查看当前模式
sudo nvpmodel -q

# 切换到最大性能模式（MaxN）
sudo nvpmodel -m 0

# 启用风扇全速
sudo jetson_clocks
```

#### 图像降采样（提升帧率）

如果推理延迟过高，可以降低图像分辨率：

```python
# 在发送前降采样
rgb_resized = cv2.resize(self.rgb_image, (320, 240))
depth_resized = cv2.resize(self.depth_image, (320, 240))
```

在服务器端配置对应的分辨率。

#### 禁用不必要的传感器

如果不需要激光雷达，可以不启动以节省资源：

```bash
# 仅启动相机和底盘
ros2 launch turn_on_wheeltec_robot turn_on_wheeltec_robot.launch.py
ros2 launch turn_on_wheeltec_robot wheeltec_camera.launch.py
```

---

## 9. 故障排除

### 9.1 服务器端问题

#### 问题1: CUDA Out of Memory

**症状：**
```
RuntimeError: CUDA out of memory
```

**解决方法：**
```bash
# 减少批处理大小或使用更大显存的 GPU
# 或启用梯度检查点（如果支持）

# 临时方案：清理 GPU 缓存
python3 -c "import torch; torch.cuda.empty_cache()"
```

#### 问题2: 模型加载失败

**症状：**
```
FileNotFoundError: checkpoints/InternVLA-N1/config.json not found
```

**解决方法：**
```bash
# 验证模型路径
ls -lh checkpoints/InternVLA-N1/

# 重新下载模型
huggingface-cli download InternRobotics/InternVLA-N1 \
    --local-dir checkpoints/InternVLA-N1
```

#### 问题3: 端口被占用

**症状：**
```
OSError: [Errno 98] Address already in use
```

**解决方法：**
```bash
# 查找占用端口的进程
lsof -i :8087

# 杀死进程
kill -9 <PID>

# 或使用其他端口
python scripts/eval/start_server.py --port 8088
```

### 9.2 网络连接问题

#### 问题1: 无法连接到服务器

**症状：**
```
requests.exceptions.ConnectionError: Connection refused
```

**排查步骤：**
```bash
# 1. 验证网络连通性
ping 192.168.1.100

# 2. 测试端口
telnet 192.168.1.100 8087
# 或
nc -zv 192.168.1.100 8087

# 3. 检查防火墙
sudo ufw status

# 4. 检查服务器是否运行
curl http://192.168.1.100:8087/health
```

#### 问题2: 推理延迟过高

**症状：**
```
Inference latency: 1500 ms
```

**解决方法：**
```bash
# 1. 检查网络延迟
ping -c 10 192.168.1.100
# 期望 < 10ms

# 2. 检查带宽
iperf3 -c 192.168.1.100

# 3. 使用有线连接替代 WiFi

# 4. 启用图像压缩（见 6.5 节）

# 5. 检查服务器 GPU 利用率
nvidia-smi
```

### 9.3 ROS2 问题

#### 问题1: 看不到相机话题

**症状：**
```bash
ros2 topic list
# /camera/* 话题不存在
```

**解决方法：**
```bash
# 1. 验证相机连接
lsusb | grep Orbbec

# 2. 重启相机节点
ros2 launch turn_on_wheeltec_robot wheeltec_camera.launch.py

# 3. 检查相机驱动日志
ros2 topic echo /rosout
```

#### 问题2: 机器人不响应 /cmd_vel

**症状：**
发布速度指令但机器人不动

**解决方法：**
```bash
# 1. 验证底盘连接
ros2 topic list | grep cmd_vel

# 2. 手动测试
ros2 topic pub /cmd_vel geometry_msgs/Twist \
    "{linear: {x: 0.1}, angular: {z: 0.0}}"

# 3. 检查急停状态（如果有）
ros2 topic echo /emergency_stop

# 4. 查看底盘日志
ros2 launch turn_on_wheeltec_robot turn_on_wheeltec_robot.launch.py
```

#### 问题3: TF 变换错误

**症状：**
```
tf2.LookupException: "camera_link" passed to lookupTransform argument
```

**解决方法：**
```bash
# 1. 查看 TF 树
ros2 run tf2_tools view_frames

# 2. 检查 TF 发布
ros2 topic echo /tf

# 3. 验证 URDF 配置
ros2 param get /robot_state_publisher robot_description
```

### 9.4 相机相关问题

#### 问题1: 深度图全是 NaN

**症状：**
深度图像全是无效值

**解决方法：**
```bash
# 1. 检查环境光照（需要充足均匀光照）

# 2. 避免对着玻璃、镜子、白墙

# 3. 确保物体在有效范围内（0.6-4.0m）

# 4. 重新标定相机
ros2 run camera_calibration cameracalibrator \
    --size 8x6 --square 0.03 \
    image:=/camera/rgb/image_raw
```

#### 问题2: 图像延迟或卡顿

**症状：**
```bash
ros2 topic hz /camera/rgb/image_raw
# 显示 < 15 Hz
```

**解决方法：**
```bash
# 1. 检查 USB 连接（确保使用 USB 3.0）
lsusb -t

# 2. 减少分辨率（在 launch 文件中配置）

# 3. 增加 USB 缓冲区
echo 1000 | sudo tee /sys/module/usbcore/parameters/usbfs_memory_mb

# 4. 关闭其他 USB 设备
```

### 9.5 Jetson 性能问题

#### 问题1: 内存不足

**症状：**
```
MemoryError: Unable to allocate array
```

**解决方法：**
```bash
# 1. 增加 swap 空间
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# 2. 减少图像缓存
# 在客户端代码中限制历史帧数量

# 3. 禁用 GUI
sudo systemctl set-default multi-user.target
```

#### 问题2: CPU 过热降频

**症状：**
```bash
sensors
# 显示 CPU > 80°C
```

**解决方法：**
```bash
# 1. 检查散热器安装

# 2. 启用风扇全速
sudo jetson_clocks

# 3. 降低功耗模式
sudo nvpmodel -m 2  # 10W 模式
```

---

## 10. 参考资料

### 10.1 官方文档

- **InternNav 项目主页**：https://github.com/InternRobotics/InternNav
- **InternNav 官方文档**：https://internrobotics.github.io/
- **InternVLA-N1 模型**：https://huggingface.co/InternRobotics/InternVLA-N1
- **技术报告**：https://internrobotics.github.io/internvla-n1.github.io/

### 10.2 相关项目

- **Unitree SDK2**：https://github.com/unitreerobotics/unitree_sdk2
- **Unitree ROS2**：https://github.com/unitreerobotics/unitree_ros2
- **Orbbec SDK ROS2**：https://github.com/orbbec/OrbbecSDK_ROS2
- **Habitat-Lab**：https://github.com/facebookresearch/habitat-lab
- **Isaac Sim**：https://developer.nvidia.com/isaac-sim

### 10.3 参考部署案例

1. **Unitree Go2 部署指南**（本仓库）：
   - `example_reference/Unitree_Go2_Go2W_B2_Edge_Deployment_Guide.md`

2. **Unitree G1 部署案例**（本仓库）：
   - `example_reference/unitree_go1_deployment.md`

3. **冠军队伍经验分享**（本仓库）：
   - `example_reference/InternNav模型部署全流程.md`

### 10.4 Wheeltec 资源

- **ROS2 常用指令**（本仓库）：
   - `wheeltec_ros2/ROS2-V5.0(humble)常用指令.txt`

- **Wheeltec 小车配置**（本仓库）：
   - `robotic_cof.md`

### 10.5 学习资源

- **ROS2 官方教程**：https://docs.ros.org/en/humble/Tutorials.html
- **Jetson 开发者指南**：https://developer.nvidia.com/embedded/learn/get-started-jetson-orin-nx-devkit
- **PyTorch 官方文档**：https://pytorch.org/docs/stable/index.html

### 10.6 社区支持

- **GitHub Issues**：https://github.com/InternRobotics/InternNav/issues
- **上海 AI Lab 官网**：https://www.shlab.org.cn/
- **InternRobotics 组织**：https://github.com/InternRobotics

---

## 附录 A: 快速启动脚本

### A.1 服务器端启动脚本

创建 `start_server.sh`：

```bash
#!/bin/bash
set -e

echo "=== Starting InternNav Server ==="

# 激活环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate internnav

# 进入工作目录
cd ~/InternNav

# 检查模型
if [ ! -d "checkpoints/InternVLA-N1" ]; then
    echo "Error: Model checkpoint not found!"
    exit 1
fi

# 启动服务器
echo "Starting server on 0.0.0.0:8087..."
python scripts/eval/start_server.py \
    --host 0.0.0.0 \
    --port 8087 \
    2>&1 | tee logs/server_$(date +%Y%m%d_%H%M%S).log
```

### A.2 Jetson 端启动脚本

创建 `start_wheeltec_nav.sh`：

```bash
#!/bin/bash
set -e

echo "=== Starting Wheeltec Navigation ==="

# ROS2 环境
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=42

# 检查服务器连通性
SERVER_IP="192.168.1.100"
echo "Testing connection to server $SERVER_IP..."
if ! ping -c 1 $SERVER_IP > /dev/null 2>&1; then
    echo "Error: Cannot reach server $SERVER_IP"
    exit 1
fi

# 启动底盘（后台）
echo "Starting robot base..."
ros2 launch turn_on_wheeltec_robot turn_on_wheeltec_robot.launch.py &
ROBOT_PID=$!
sleep 3

# 启动相机（后台）
echo "Starting camera..."
ros2 launch turn_on_wheeltec_robot wheeltec_camera.launch.py &
CAMERA_PID=$!
sleep 3

# 启动导航客户端
echo "Starting navigation client..."
cd ~/InternNav
python3 scripts/realworld/wheeltec_client.py

# 清理
kill $ROBOT_PID $CAMERA_PID
```

### A.3 一键重启脚本

创建 `restart_all.sh`：

```bash
#!/bin/bash

echo "=== Restarting All Nodes ==="

# 停止所有 ROS2 节点
pkill -f "ros2 launch"
pkill -f "wheeltec_client.py"

# 等待清理
sleep 2

# 重新启动
./start_wheeltec_nav.sh
```

---

## 附录 B: 配置文件模板

### B.1 相机校准文件

创建 `config/astra_s_calibration.yaml`：

```yaml
camera_name: astra_s
image_width: 640
image_height: 480

# 畸变模型
distortion_model: plumb_bob

# 内参矩阵
camera_matrix:
  rows: 3
  cols: 3
  data: [570.3, 0.0, 319.5,
         0.0, 570.3, 239.5,
         0.0, 0.0, 1.0]

# 畸变系数
distortion_coefficients:
  rows: 1
  cols: 5
  data: [0.0, 0.0, 0.0, 0.0, 0.0]

# 投影矩阵
projection_matrix:
  rows: 3
  cols: 4
  data: [570.3, 0.0, 319.5, 0.0,
         0.0, 570.3, 239.5, 0.0,
         0.0, 0.0, 1.0, 0.0]

# 修正矩阵
rectification_matrix:
  rows: 3
  cols: 3
  data: [1.0, 0.0, 0.0,
         0.0, 1.0, 0.0,
         0.0, 0.0, 1.0]
```

### B.2 导航参数配置

创建 `config/nav_params.yaml`：

```yaml
# 速度限制
velocity_limits:
  max_linear: 0.2      # m/s
  max_angular: 0.5     # rad/s
  max_linear_accel: 0.5    # m/s^2
  max_angular_accel: 1.0   # rad/s^2

# 避障参数
collision_avoidance:
  safety_distance: 0.6     # m
  slow_down_distance: 1.0  # m
  check_frequency: 10.0    # Hz

# 电池管理
battery:
  low_voltage_threshold: 22.0   # V
  critical_voltage_threshold: 21.0  # V

# 网络配置
network:
  server_host: "192.168.1.100"
  server_port: 8087
  timeout: 5.0  # s
  retry_attempts: 3

# 日志
logging:
  level: "INFO"  # DEBUG, INFO, WARN, ERROR
  save_trajectory: true
  output_dir: "/home/wheeltec/nav_logs"
```

---

## 附录 C: 常见导航指令示例

```python
# 简单移动
"Go forward"
"Turn left"
"Turn right"
"Go back"

# 目标导航
"Go to the red chair"
"Move to the table"
"Navigate to the door"

# 复合指令
"Go forward until you reach the chair, then turn right"
"Move to the table and stop in front of it"
"Turn left, go through the door, and stop"

# 空间关系
"Go to the chair next to the table"
"Move to the left side of the sofa"
"Stop between the two chairs"

# 注意事项：
# 1. 指令应清晰、具体
# 2. 避免过于复杂的多步指令（可能超出模型能力）
# 3. 确保指令中的物体在视野内可见
# 4. 首次部署建议从简单指令开始测试
```

---

## 结语

本指南提供了 InternNav 在 Wheeltec Senior_4wd_bs 轮式机器人上的完整部署流程。通过遵循本指南，您应该能够：

1. ✅ 在服务器端成功部署 InternVLA-N1 模型
2. ✅ 在 Jetson Orin NX 上配置客户端环境
3. ✅ 建立稳定的网络通信
4. ✅ 实现基于自然语言的机器人导航

### 重要提醒：

1. **安全第一**：在实际部署前务必添加避障逻辑
2. **充分测试**：从简单场景和指令开始逐步复杂化
3. **持续监控**：关注延迟、电池、温度等关键指标
4. **记录日志**：保存运行数据便于后续分析优化

### 获取帮助：

- 遇到问题请查阅[故障排除](#9-故障排除)章节
- 参考[官方示例](https://github.com/InternRobotics/InternNav/tree/main/scripts/realworld)
- 在 GitHub 提交 [Issue](https://github.com/InternRobotics/InternNav/issues)
- 查看 [CLAUDE.md](./CLAUDE.md) 获取更多技术细节

祝您部署成功！🚀

---

**文档版本：** v1.0
**最后更新：** 2026-01-12
**维护者：** Claude Code
**许可证：** 遵循 InternNav 原项目许可证
