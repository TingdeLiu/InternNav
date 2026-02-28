# LingNav — 基于 InternNav 的 Wheeltec 小车 VLN 优化项目

**目标平台：** Wheeltec Senior_4wd_bs (Jetson Orin NX 16GB + Gemini 336L/Astra S 深度相机)
**核心思路：** 用 Qwen3-VL 零样本做语义理解（S2），NavDP 做像素目标导航（S1），两者通过 HTTP 解耦，不改动原项目核心包。

---

## 两种部署模式

| | [模式 A — 双服务器](LingNavA.md) | [模式 B — S1 端侧](LingNavB.md) |
|---|---|---|
| S1 运行位置 | GPU 服务器（HTTP） | Jetson 本地（fp16） |
| S1 推理延迟 | ~100-300ms | ~50-150ms |
| 网络依赖 | S2 + S1 均需网络 | 仅 S2 需网络 |
| Jetson 显存 | 极少（仅 ROS2） | ~200-400MB |
| 部署复杂度 | 服务器启动 2 个进程 | 服务器只需启动 S2 |

**推荐选择：**
- 首次调试 / 服务器资源充足 → 模式 A（步骤简单，Jetson 无需额外配置）
- 追求低延迟 / 服务器网络不稳定 → 模式 B（S1 本地推理，更稳定）

---

## 共用组件

无论哪种模式，以下文件均会被使用：

| 文件 | 说明 |
|------|------|
| `scripts/realworld2/wheeltec_s2_server.py` | S2 Qwen3-VL 推理服务 |
| `scripts/realworld2/lingnav_pipeline.py` | S2+S1 联合推理管线 |
| `scripts/realworld2/lingnav_ros_client.py` | Jetson ROS2 导航节点 |
| `scripts/realworld2/test_s2_client.py` | S2 连通性测试客户端 |

---

## 端口约定

| 服务 | 端口 | 说明 |
|------|------|------|
| S2 Qwen3-VL | **8890** | LingNav 新增，不与原项目冲突 |
| S1 NavDP | **8901** | 模式 A 使用；模式 B 本地推理，不占端口 |
| 原 InternVLA-N1 评估服务器 | 8087 | 原项目，不受影响 |
| 原实机服务器 | 8888 | 原项目，不受影响 |
