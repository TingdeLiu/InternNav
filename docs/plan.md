# 开发规划：基于 InternNav 的 Wheeltec 小车 VLN 优化

**目标平台：** Wheeltec Senior_4wd_bs (Jetson Orin NX 16GB + Astra S 深度相机)
**核心目标：** 在自有小车上实现自然语言指令跟随导航，在原项目基础上优化，不破坏原有功能

---

## 一、现状分析

### 1.1 原项目在小车上效果差的根本原因

原 InternVLA-N1 双系统的 S2（Qwen2.5-VL）是**经过导航数据联合训练**的，其输出（`traj_latents`）是 S1（NavDP/NextDiT）的条件向量，两者耦合。在 Wheeltec 上表现差，主要原因推测：

| 原因 | 分析 |
|------|------|
| **相机差异** | 原训练数据来自 Unitree Go2/H1（不同视角、FOV），Astra S 内参完全不同 |
| **平台高度差异** | 小车摄像头高度 ~0.5m，原模型训练平台更高，感知视角不同 |
| **速度/动力学差异** | 小车最大安全速度 0.25m/s，原动作空间不匹配 |
| **训练分布偏移** | 原数据以室内大场景为主（Habitat MP3D/GRScenes），小车的实际场景可能不同 |

### 1.2 已完成的工作

- `docs/orignal/Wheeltec_InternNav_部署指南.md`：Wheeltec 适配的 Client-Server 部署方案（话题适配、内参修改、速度限制）
- `docs/new/navdp_s1_standalone_guide.md`：NavDP S1 解耦独立运行，支持 pixelgoal / pointgoal / imagegoal / nogoal 5种模式
- `docs/new/qwen3vl_support_guide.md`：Qwen3-VL 作为 S2 后端的训练+评估支持
- `docs/new/InternNav 自定义 S1-S2 系统开发指南.md`：S1/S2 解耦开发接口规范
- `scripts/realworld2/`：LingNav Phase 1-3 全部脚本（S2 服务、管线、ROS2 客户端）
- `scripts/inference/NavDP/navdp_local_client.py`：NavDP 端侧本地推理客户端（Phase 3.5）
- `lingnav_pipeline.py --random`：显式随机噪声图参数，与 `test_s2_client.py` 接口一致（Phase 3.5）

---

## 二、目标架构

### 2.1 新方案：零样本 S2 + 独立 NavDP S1

```
┌─────────────────────────────────────────────────────────────────────┐
│                    服务器端（GPU 机器）                               │
│                                                                     │
│  ┌────────────────────────┐    pixel (u,v)   ┌──────────────────┐  │
│  │    S2: Qwen3-VL        │ ─────────────── ▶│  S1: NavDP       │  │
│  │    (零样本推理)          │                  │  (pixelgoal 模式)│  │
│  │                        │                  │                  │  │
│  │  输入: RGB图像 + 指令    │                  │  输入: RGBD +    │  │
│  │  输出: 目标像素坐标(u,v) │                  │        pixel(u,v)│  │
│  │        or 停止/转身指令  │                  │  输出: 轨迹点序列  │  │
│  └────────────────────────┘                  └──────────────────┘  │
│                                                                     │
│  服务形式: Flask HTTP API（S2: /s2_step，S1: /pixelgoal_step）       │
└──────────────────────────────┬──────────────────────────────────────┘
                               │  局域网 HTTP
┌──────────────────────────────▼──────────────────────────────────────┐
│                    Jetson Orin NX（机器人端）                         │
│                                                                     │
│  RGB+Depth (Astra S)  →  规划线程 → 请求 S2 → 请求 S1 → 获取轨迹    │
│  Odom (/odom)         →  控制线程 → MPC 跟踪轨迹 → /cmd_vel         │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 关键设计决策

**为什么不训练 S2，而用 Prompt Engineering：**
- Qwen3-VL 是通用视觉大模型，零样本的目标定位能力已足够强
- 避免收集标注训练数据
- 可快速迭代 Prompt，调试周期短
- 后续验证效果不好再考虑 fine-tune

**S1 NavDP 选 pixelgoal 模式：**
- S2 输出像素坐标 (u,v) → S1 直接接收，接口简单
- pixelgoal 模式无需 3D 坐标，不依赖里程计
- NavDP 的 Critic 机制能自动过滤不合理轨迹

**保持原项目不变：**
- 新功能以**独立脚本/服务**形式开发，不修改 `internnav/` 核心包
- 仅在 `scripts/realworld/` 下增加 Wheeltec 特定脚本

---

## 三、开发阶段规划

### Phase 1：S2 Prompt Engineering（当前重点）

**目标：** 让 Qwen3-VL 零样本输入一张图 + 导航指令，稳定输出目标像素坐标 (u,v)

#### 1.1 S2 输出协议设计

S2 需要针对不同场景给出不同输出，定义以下输出格式（JSON）：

```json
{
  "action": "pixel_goal",
  "pixel": [320, 240],
  "confidence": 0.85,
  "reason": "目标椅子在画面中央偏左"
}
```

或：

```json
{
  "action": "turn_left",
  "angle": 30,
  "reason": "目标在画面外，需要向左转以找到目标"
}
```

或：

```json
{
  "action": "stop",
  "reason": "已到达目标位置前方 0.5m"
}
```

#### 1.2 Prompt 设计要点

```
System Prompt：
  你是一个在室内导航机器人上运行的视觉导航助手。
  相机分辨率 640×480，安装在高度约 0.5m 的小车上。
  你需要分析当前图像和导航指令，输出结构化 JSON 决策。

  输出规则：
  1. 如果目标在视野内：输出 pixel_goal，给出目标在图像中的 (u, v) 像素坐标
     - u: 水平方向，0=最左，640=最右
     - v: 垂直方向，0=最上，480=最下
     - 目标像素选取目标物体的底部中心点（接触地面处）
  2. 如果目标在视野外（需要转头找）：输出 turn_left 或 turn_right，给出估计角度
  3. 如果已到达目标附近（目标占画面超过 30%）：输出 stop

User Prompt：
  当前导航指令：{instruction}
  当前步骤：{step} / 历史轨迹：{history_summary}
  请分析图像，给出下一步导航决策。
  只输出 JSON，不要其他解释。
```

#### 1.3 实现文件

新建 `scripts/realworld/wheeltec_s2_server.py`：

```
功能：
  - 加载 Qwen3-VL (Qwen/Qwen3-VL-7B-Instruct 或 Qwen/Qwen3-VL-3B-Instruct)
  - Flask API: POST /s2_step { image, instruction, step, history }
              → { action, pixel/angle, confidence, reason }
  - 维护多轮对话历史（可选，用于长指令任务）

注意：
  - 使用 qwen_vl_utils 处理图像输入
  - transformers >= 4.57.0
  - 显存要求: 3B ~8GB, 7B ~16GB，选择适合服务器的尺寸
```

#### 1.4 验证方法

- 准备 50 张小车实拍场景图，标注各类导航指令
- 统计 pixel_goal 输出的 (u,v) 与人工标注的偏差（目标：均值 < 50 像素）
- 统计"需要转头"场景的 turn_left/right 输出准确率
- 统计误判停止（不该停却停了）的比例

---

### Phase 2：S1 NavDP PixelGoal 集成

**目标：** NavDP 独立运行，接收 S2 输出的像素坐标，输出轨迹

#### 2.1 接口对接

S2 输出 `pixel: [u, v]` → 直接传入 NavDP `/pixelgoal_step` 端点：

```json
POST /pixelgoal_step
{
  "pixel_x": [u],
  "pixel_y": [v]
}
+ image (JPEG) + depth (PNG)
→ trajectory: shape (1, 24, 3)
```

NavDP `pixelgoal_step` 内部将像素坐标反投影到 3D 目标方向，再扩散生成轨迹。

#### 2.2 相机内参适配

NavDP 使用相机内参做像素→3D 反投影，需传入 Astra S 内参：

```python
# reset 时传入
camera_intrinsic = np.array([
    [570.3, 0.0, 319.5, 0.0],
    [0.0, 570.3, 239.5, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
])
```

#### 2.3 S2→S1 连接逻辑

当 S2 返回 `action: pixel_goal` 时，将 pixel 传给 S1；
当 S2 返回 `action: turn_left/right` 时，跳过 S1，直接发布原地旋转指令；
当 S2 返回 `action: stop` 时，停止。

```python
def planning_step(rgb, depth, odom, instruction, step):
    # Step 1: 请求 S2
    s2_result = call_s2(rgb, instruction, step)

    if s2_result["action"] == "pixel_goal":
        u, v = s2_result["pixel"]
        # Step 2: 请求 S1
        traj = call_s1_pixelgoal(rgb, depth, u, v)
        return {"mode": "trajectory", "trajectory": traj}

    elif s2_result["action"] in ("turn_left", "turn_right"):
        return {"mode": "rotate", "action": s2_result}

    elif s2_result["action"] == "stop":
        return {"mode": "stop"}
```

#### 2.4 NavDP checkpoint

需要下载 NavDP 官方权重（支持 pixelgoal 模式的版本），参考 `navdp_s1_standalone_guide.md` 中的链接。

---

### Phase 3：机器人端部署

**目标：** Jetson 上运行 ROS2 客户端，完整跑通 指令 → 感知 → 规划 → 控制 闭环

#### 3.1 部署文件结构

```
scripts/realworld/wheeltec/
├── wheeltec_nav_client.py     # 主节点（ROS2 + 规划/控制双线程）
├── wheeltec_s2_server.py      # S2 服务器（GPU 机器上运行）
├── navdp_server.py            # S1 NavDP 服务器（复用现有）
├── config.yaml                # 统一配置（服务器 IP、内参、速度限制）
└── start_all.sh               # 一键启动脚本
```

#### 3.2 核心客户端架构

基于现有 `wheeltec_client.py`（`docs/orignal/Wheeltec_InternNav_部署指南.md`），增加：
- 将原来单个 HTTP 服务器调用拆分为 S2→S1 两跳
- `planning_thread` 中增加 `turn_left/right` 原地旋转分支
- 增加"目标未找到超时"重新探索逻辑（连续 N 步 S2 返回 turn 则加大旋转角度）

#### 3.3 控制参数（Wheeltec 安全范围）

| 参数 | 值 | 说明 |
|------|-----|------|
| 直线速度 | ≤ 0.25 m/s | 导航跟踪限速 |
| 角速度 | ≤ 0.3 rad/s | 原地旋转极慢，避免里程计漂移 |
| 规划频率 | 0.3 s/次 | S2+S1 总推理时间 < 1s 时 |
| 控制频率 | 0.1 s/次 | MPC 轨迹跟踪 |
| 碰撞安全距离 | 0.6m | 深度图中心区域最小值判断 |

#### 3.4 评估方法（无仿真环境）

在固定室内场景设计 20 条测试路线，每条路线：
- 起点、终点固定
- 自然语言指令（目标物体描述）
- 手动记录到达率（SR）和路径是否合理

---

### Phase 3.5：S1 NavDP 端侧部署

**目标：** NavDP (S1) 直接在 Jetson Orin NX 上运行，消除 S1 HTTP 网络依赖，降低推理延迟。

#### 3.5.1 方案背景

Phase 3 中 S1 通过 HTTP 调用 GPU 服务器，存在以下问题：
- S1 每次调用增加 100-300ms 网络往返
- 需要服务器同时维护 S2 + S1 两个进程
- 网络不稳定时 S1 调用容易超时

NavDP 模型本身较小（约 200-400MB fp16），Jetson Orin NX 16GB 完全可以直接承载。

#### 3.5.2 实现文件

| 文件 | 说明 |
|------|------|
| `scripts/inference/NavDP/navdp_local_client.py` | **新增** NavDP 端侧本地推理客户端，封装 `NavDPAgent`，与 `NavDPClient`（HTTP）接口完全一致 |

**`NavDPLocalClient` 关键接口：**

```python
from navdp_local_client import NavDPLocalClient

client = NavDPLocalClient(
    checkpoint="/path/to/navdp.ckpt",
    device="cuda:0",
    half=True,        # fp16，Jetson 推荐
)
client.reset(camera_intrinsic, batch_size=1)
traj, all_traj, values = client.pixelgoal_step(pixel_goals, rgb_images, depth_images)
```

#### 3.5.3 对上层的改动（向后兼容）

**`lingnav_pipeline.py`** — `LingNavPipeline` 新增 `s1_client` 参数：

```python
# 服务器模式（原有，不变）
pipeline = LingNavPipeline(s2_host="...", s1_host="...", s1_port=8901)

# 端侧模式（新增）
from navdp_local_client import NavDPLocalClient
pipeline = LingNavPipeline(
    s2_host="...",
    s1_client=NavDPLocalClient(checkpoint="...", half=True),
)
```

**`lingnav_ros_client.py`** — 新增 4 个 CLI 参数：

| 参数 | 说明 |
|------|------|
| `--local_s1` | 开启端侧 S1 模式 |
| `--s1_checkpoint` | NavDP 权重路径（必填）|
| `--s1_device` | 推理设备，默认 `cuda:0` |
| `--s1_half` | 开启 fp16 |

#### 3.5.4 Jetson 启动方式

```bash
# GPU 服务器：只需启动 S2
python scripts/realworld2/wheeltec_s2_server.py \
    --model_path Qwen/Qwen3-VL-7B-Instruct --port 8890

# Jetson 端：端侧 S1 模式
export NAVDP_ROOT=~/NavDP
export PYTHONPATH=$PYTHONPATH:~/InternNav/src/diffusion-policy

python3 scripts/realworld2/lingnav_ros_client.py \
    --instruction "Go to the red chair" \
    --s2_host 192.168.1.100 \
    --local_s1 \
    --s1_checkpoint ~/NavDP/checkpoints/navdp.ckpt \
    --s1_half
```

#### 3.5.5 两种模式对比

| 指标 | 模式 A（双服务器） | 模式 B（S1 端侧） |
|------|-----------------|----------------|
| S1 推理延迟 | ~100-300ms（网络+服务器） | ~50-150ms（本地 fp16） |
| 网络依赖 | S2 + S1 均需网络 | 仅 S2 需网络 |
| Jetson 显存 | 极少（仅 ROS2） | ~200-400MB（NavDP fp16） |
| 服务器进程数 | 2（S2 + S1） | 1（仅 S2） |

#### 3.5.6 Jetson 依赖安装（首次）

```bash
# PyTorch：使用 JetPack 配套 wheel（不要用 pip 默认版本）
# 参考：https://forums.developer.nvidia.com/t/pytorch-for-jetson

# 其他依赖
pip3 install numpy requests Pillow opencv-python casadi scipy matplotlib

# 克隆 NavDP 项目（与 InternNav 同级）
git clone https://github.com/NavDP/NavDP ~/NavDP
```

---

### Phase 4（可选）：S1 NavDP 训练优化

**触发条件：** Phase 3 测试后，SR < 50% 且主要失败原因是轨迹质量差（而非 S2 目标识别错误）

#### 4.1 数据采集

使用小车在真实环境中采集数据：

```
格式参考: internnav/dataset/navdp_dataset.py
数据内容:
  - RGB (640×480)
  - Depth (640×480, float32, 米)
  - 里程计位移 (Δx, Δy, Δyaw)
  - 相机内参

采集工具: 遥控小车，记录传感器+控制数据
目标量: 2-5 小时真实数据（约 20-50 万帧）
```

#### 4.2 训练配置

```bash
# 基于现有训练脚本
python scripts/train/train.py \
    --model_name=navdp \
    --name=wheeltec_navdp \
    --config scripts/train/base_train/configs/navdp.py
```

需要修改 `scripts/train/base_train/configs/navdp.py` 中：
- 相机内参 → Astra S
- 图像分辨率 → 640×480（或裁剪到模型输入 224×224）
- 速度归一化范围 → 匹配小车动力学

---

## 四、技术风险与应对

| 风险 | 概率 | 应对方案 |
|------|------|---------|
| Qwen3-VL 零样本像素定位不准（偏移 > 100 像素） | 中 | 调整 Prompt；加入图像预处理（标注网格/坐标轴）；换用 3B→7B 更大模型 |
| S2+S1 总推理延迟 > 1s，控制卡顿 | 中 | S2/S1 异步化（S2 在独立线程，S1 频率更高）；降低 S2 图像分辨率到 336×336 |
| NavDP pixelgoal 模式对 Astra S 内参不适应 | 低 | 验证内参传入是否正确；用 DepthAnything 替换深度估计 |
| 小车里程计漂移导致 MPC 失效 | 中 | 切换到 PID 模式；或用纯图像反馈（不依赖里程计） |
| Qwen3-VL 输出 JSON 格式不稳定 | 中 | 加强 Prompt 中的格式约束；增加 JSON 解析容错；设置 `temperature=0.1` |

---

## 五、不影响原项目的开发原则

1. **所有新文件放在 `scripts/realworld/wheeltec/` 下**，不修改 `internnav/` 核心包
2. **新服务器端口不与原有冲突**：S2 server 用 8890，NavDP S1 用 8901，避开原有 8087/8088
3. **Qwen3-VL S2 服务器与原有 InternVLA-N1 服务器完全独立**，可同时运行
4. **配置文件化**：所有 IP、端口、内参、速度限制放入 `config.yaml`，不硬编码在脚本里
5. **评估配置**：小车专用评估配置放 `scripts/eval/configs/wheeltec_*.py`，不修改现有配置

---

## 六、里程碑

| 里程碑 | 验收标准 |
|--------|---------|
| **M1: S2 Prompt 验证** | 50 张测试图的像素定位平均误差 < 80px，stop/turn 分类准确率 > 80% |
| **M2: S1 NavDP 通跑** | 服务器上模拟输入 pixel(320,240)，NavDP 输出合理轨迹，不报错 |
| **M3: 联调在仿真** | Habitat 中以 Qwen3-VL 零样本 + NavDP pixelgoal 跑通一条完整路线 |
| **M4: 机器人单场景** | 固定房间内，"去红色椅子"等 5 条指令 SR ≥ 60% |
| **M4.5: S1 端侧部署验证** | Jetson 本地加载 NavDP（fp16），`--local_s1` 模式下 pixelgoal_step 输出合理轨迹，端到端延迟 < 500ms |
| **M5: 多场景泛化** | 20 条测试路线 SR ≥ 50%（若不达标则进入 Phase 4 S1 训练） |
