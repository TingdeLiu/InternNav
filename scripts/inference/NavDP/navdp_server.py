"""
NavDP S1 HTTP 推理服务

启动 Flask HTTP Server，暴露 NavDP S1 的所有导航任务 API。
兼容 NavDP 项目的 eval_*_wheeled.py 评测客户端 HTTP 协议。

启动方式:
    python scripts/inference/NavDP/navdp_server.py \
        --port 8888 \
        --checkpoint /path/to/navdp_checkpoint.ckpt

需要设置环境变量 NAVDP_ROOT 指向 NavDP 项目根目录（默认与 InternNav 同级）:
    export NAVDP_ROOT=/path/to/NavDP

详见 docs/navdp_s1_standalone_guide.md
"""

import argparse
import datetime
import json
import os
import sys
import time

from pathlib import Path

import cv2
import imageio
import numpy as np
from flask import Flask, jsonify, request
from PIL import Image

# 将 InternNav 根目录加入 sys.path 以导入 internnav 包
_INTERNNAV_ROOT = str(Path(__file__).resolve().parents[3])
if _INTERNNAV_ROOT not in sys.path:
    sys.path.insert(0, _INTERNNAV_ROOT)

from internnav.agent.navdp_agent import NavDPAgent  # noqa: E402

app = Flask(__name__)

# 全局状态
navigator: NavDPAgent = None
fps_writer = None
args = None


def get_args():
    parser = argparse.ArgumentParser(description="NavDP S1 HTTP Inference Server")
    parser.add_argument("--port", type=int, default=8901, help="Server port")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to NavDP model checkpoint (.ckpt)",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Compute device")
    parser.add_argument("--image_size", type=int, default=224, help="Model input image size")
    parser.add_argument("--memory_size", type=int, default=8, help="History frame memory size")
    parser.add_argument("--predict_size", type=int, default=24, help="Trajectory prediction length")
    parser.add_argument("--temporal_depth", type=int, default=16, help="Transformer decoder layers")
    parser.add_argument("--heads", type=int, default=8, help="Attention heads")
    parser.add_argument("--token_dim", type=int, default=384, help="Token dimension")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host")
    return parser.parse_known_args()[0]


# ============================================================
# API 端点
# ============================================================


@app.route("/navigator_reset", methods=["POST"])
def navigator_reset():
    """重置导航器（新 episode 开始时调用）

    请求 JSON:
        intrinsic: 相机内参矩阵 (4×4 list)
        stop_threshold: 停止阈值 (float)
        batch_size: 批次大小 (int)
    """
    global navigator, fps_writer

    intrinsic = np.array(request.get_json().get("intrinsic"))
    threshold = float(request.get_json().get("stop_threshold", -3.0))
    batch_size = int(request.get_json().get("batch_size", 1))

    if navigator is None:
        navigator = NavDPAgent(
            camera_intrinsic=intrinsic,
            checkpoint=args.checkpoint,
            image_size=args.image_size,
            memory_size=args.memory_size,
            predict_size=args.predict_size,
            temporal_depth=args.temporal_depth,
            heads=args.heads,
            token_dim=args.token_dim,
            device=args.device,
        )
    navigator.camera_intrinsic = intrinsic
    navigator.reset(batch_size, threshold)

    # 重置视频记录器
    if fps_writer is not None:
        fps_writer.close()
    fmt_time = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H%M%S")
    fps_writer = imageio.get_writer(f"{fmt_time}_fps.mp4", fps=7)

    return jsonify({"algo": "navdp"})


@app.route("/navigator_reset_env", methods=["POST"])
def navigator_reset_env():
    """重置指定环境的记忆队列"""
    global navigator
    env_id = int(request.get_json().get("env_id"))
    navigator.reset_env(env_id)
    return jsonify({"algo": "navdp"})


def _decode_request_images(batch_size):
    """从 HTTP 请求中解码 RGB 和 Depth 图像"""
    # RGB
    image_file = request.files["image"]
    image = Image.open(image_file.stream).convert("RGB")
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = image.reshape((batch_size, -1, image.shape[1], 3))

    # Depth
    depth_file = request.files["depth"]
    depth = Image.open(depth_file.stream).convert("I")
    depth = np.asarray(depth)[:, :, np.newaxis].astype(np.float32) / 10000.0
    depth = depth.reshape((batch_size, -1, depth.shape[1], 1))

    return image, depth


@app.route("/pointgoal_step", methods=["POST"])
def pointgoal_step():
    """点目标导航推理

    请求:
        files: image (JPEG), depth (PNG uint16, 深度×10000)
        form: goal_data (JSON: {goal_x: [...], goal_y: [...]})

    返回:
        trajectory: 最优轨迹 (B, T, 3)
        all_trajectory: 所有候选轨迹 (B, N, T, 3)
        all_values: critic 评分 (B, N)
    """
    global navigator, fps_writer
    batch_size = navigator.batch_size

    goal_data = json.loads(request.form.get("goal_data"))
    goal_x = np.array(goal_data["goal_x"])
    goal_y = np.array(goal_data["goal_y"])
    goal = np.stack((goal_x, goal_y, np.zeros_like(goal_x)), axis=1)

    image, depth = _decode_request_images(batch_size)
    traj, all_traj, all_vals, vis = navigator.step_pointgoal(goal, image, depth)

    if fps_writer is not None:
        fps_writer.append_data(vis)

    return jsonify({
        "trajectory": traj.tolist(),
        "all_trajectory": all_traj.tolist(),
        "all_values": all_vals.tolist(),
    })


@app.route("/nogoal_step", methods=["POST"])
def nogoal_step():
    """无目标探索推理

    请求:
        files: image (JPEG), depth (PNG uint16)

    返回:
        trajectory, all_trajectory, all_values
    """
    global navigator, fps_writer
    batch_size = navigator.batch_size

    image, depth = _decode_request_images(batch_size)
    traj, all_traj, all_vals, vis = navigator.step_nogoal(image, depth)

    if fps_writer is not None:
        fps_writer.append_data(vis)

    return jsonify({
        "trajectory": traj.tolist(),
        "all_trajectory": all_traj.tolist(),
        "all_values": all_vals.tolist(),
    })


@app.route("/imagegoal_step", methods=["POST"])
def imagegoal_step():
    """图像目标导航推理

    请求:
        files: image (JPEG), depth (PNG uint16), goal (JPEG 目标图像)

    返回:
        trajectory, all_trajectory, all_values
    """
    global navigator, fps_writer
    batch_size = navigator.batch_size

    # 目标图像
    goal_file = request.files["goal"]
    goal = Image.open(goal_file.stream).convert("RGB")
    goal = np.asarray(goal)
    goal = cv2.cvtColor(goal, cv2.COLOR_RGB2BGR)
    goal = goal.reshape((batch_size, -1, goal.shape[1], 3))

    image, depth = _decode_request_images(batch_size)
    traj, all_traj, all_vals, vis = navigator.step_imagegoal(goal, image, depth)

    if fps_writer is not None:
        fps_writer.append_data(vis)

    return jsonify({
        "trajectory": traj.tolist(),
        "all_trajectory": all_traj.tolist(),
        "all_values": all_vals.tolist(),
    })


@app.route("/pixelgoal_step", methods=["POST"])
def pixelgoal_step():
    """像素目标导航推理

    请求:
        files: image (JPEG), depth (PNG uint16)
        form: goal_data (JSON: {goal_x: [...], goal_y: [...]})

    返回:
        trajectory, all_trajectory, all_values
    """
    global navigator, fps_writer
    batch_size = navigator.batch_size

    goal_data = json.loads(request.form.get("goal_data"))
    goal_x = np.array(goal_data["goal_x"])
    goal_y = np.array(goal_data["goal_y"])
    goal = np.stack((goal_x, goal_y), axis=1)

    image, depth = _decode_request_images(batch_size)
    traj, all_traj, all_vals, vis = navigator.step_pixelgoal(goal, image, depth)

    if fps_writer is not None:
        fps_writer.append_data(vis)

    return jsonify({
        "trajectory": traj.tolist(),
        "all_trajectory": all_traj.tolist(),
        "all_values": all_vals.tolist(),
    })


if __name__ == "__main__":
    args = get_args()
    print(f"NavDP S1 Server starting on {args.host}:{args.port}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Device: {args.device}")
    print(f"  Image size: {args.image_size}, Memory: {args.memory_size}, Predict: {args.predict_size}")
    app.run(host=args.host, port=args.port)
