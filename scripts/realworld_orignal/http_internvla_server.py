import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
from flask import Flask, jsonify, request
from PIL import Image

from internnav.agent.internvla_n1_agent_realworld import InternVLAN1AsyncAgent

app = Flask(__name__)
idx = 0
start_time = time.time()
output_dir = ''

"""
该脚本启动一个基于 Flask 的 HTTP 服务，用于在真实场景中调用 InternVLA-N1 异步推理 Agent。

接口：
- POST /eval_dual
  请求类型：multipart/form-data
  字段说明：
    - image: RGB 图像文件（例如 .png/.jpg），单张图片
    - depth: 深度图文件（16-bit 或兼容格式），与 image 同分辨率
    - json: 字符串形式的 JSON，包含控制参数，如 {"reset": true/false}
  行为：
    - 如果传入 {"reset": true}，则会重置 Agent（内部历史缓存清空、计数器归零等）
    - 对传入的单帧图像+深度进行推理，返回离散动作或连续轨迹

响应：
- JSON：
  - 当存在离散动作时：
      {"discrete_action": [int, ...]}
  - 当不存在离散动作时：
      {
        "trajectory": [[x, y], ...],   # 连续轨迹（像素/坐标系依实现而定）
        "pixel_goal": [y, x]           # 可选，目标像素点
      }

注意：
- 该服务面向单步推理（每次请求一帧），但 Agent 可能在内部维护历史帧（num_history）。
- 深度图会按 1/10000 进行缩放，得到米级单位的浮点深度。
"""

@app.route("/eval_dual", methods=['POST'])
def eval_dual():
    """
    处理单次推理请求：
    1) 解析 multipart/form-data（image、depth、json）
    2) 可选执行 reset（清理历史状态）
    3) 调用 InternVLAN1AsyncAgent.step 进行推理
    4) 返回离散动作或连续轨迹的 JSON 响应
    """
    global idx, output_dir, start_time
    start_time = time.time()

    # 从 HTTP 请求中提取文件与表单 JSON
    image_file = request.files['image']
    depth_file = request.files['depth']
    json_data = request.form['json']

    print(f"Received request with json: {json_data}")
    data = json.loads(json_data)

    # 读取并转换 RGB 图像为 numpy 数组，形状约为 (H, W, 3)，通道顺序为 RGB
    image = Image.open(image_file.stream)
    image = image.convert('RGB')
    image = np.asarray(image)

    # 读取深度图并转换为 float32，按 1/10000 缩放到米单位
    depth = Image.open(depth_file.stream)
    depth = depth.convert('I')  # 32-bit 整型格式
    depth = np.asarray(depth)
    depth = depth.astype(np.float32) / 10000.0
    print(f"read http data cost {time.time() - start_time}")

    # 相机位姿，这里使用单位矩阵作为占位（实际应用中应传入真实位姿）
    camera_pose = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    # 指令文本（示例），可根据业务需求改为从请求或配置中读取
    instruction = data.get(
        "instruction","stop moving")

    # 根据传入的 json 控制是否重置 Agent（清理内部状态）
    policy_init = data.get('reset', False)
    if policy_init:
        start_time = time.time()
        idx = 0
        output_dir = 'output/runs' + datetime.now().strftime('%m-%d-%H%M')
        os.makedirs(output_dir, exist_ok=True)
        print("init reset model!!!")
        agent.reset()

    idx += 1

    # look_down 用于控制是否“低头”视角（具体动作触发逻辑见下方）
    look_down = False
    t0 = time.time()
    dual_sys_output = {}

    # 进行一次推理
    dual_sys_output = agent.step(
        image, depth, camera_pose, instruction, intrinsic=args.camera_intrinsic, look_down=look_down
    )

    # 若输出动作为 [5]，表示需要“低头”再推理一次（具体语义取决于策略定义）
    if dual_sys_output.output_action is not None and dual_sys_output.output_action == [5]:
        look_down = True
        dual_sys_output = agent.step(
            image, depth, camera_pose, instruction, intrinsic=args.camera_intrinsic, look_down=look_down
        )

    # 组织返回的 JSON 数据：优先返回离散动作，否则返回轨迹与像素目标
    json_output = {}
    if dual_sys_output.output_action is not None:
        json_output['discrete_action'] = dual_sys_output.output_action
    else:
        json_output['trajectory'] = dual_sys_output.output_trajectory.tolist()
        if dual_sys_output.output_pixel is not None:
            json_output['pixel_goal'] = dual_sys_output.output_pixel

    t1 = time.time()
    generate_time = t1 - t0
    print(f"dual sys step {generate_time}")
    print(f"json_output {json_output}")
    return jsonify(json_output)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # 推理设备（CUDA 显卡或 CPU），例如 "cuda:7"
    parser.add_argument("--device", type=str, default="cuda:7")
    # 模型权重目录路径
    parser.add_argument("--model_path", type=str, default="checkpoints/InternVLA-N1-wo-dagger")
    # 输入图像缩放尺寸（模型前处理）
    parser.add_argument("--resize_w", type=int, default=384)
    parser.add_argument("--resize_h", type=int, default=384)
    # 历史帧长度（影响时序建模与缓存消耗）
    parser.add_argument("--num_history", type=int, default=8)
    # 规划步长间隔
    parser.add_argument('--plan_step_gap', type=int, default=8, help='Plan step gap')
    args = parser.parse_args()

    # 相机内参（4x4，占位示例）；实际使用中请根据设备标定结果设置
    args.camera_intrinsic = np.array([
        [570.3, 0.0, 319.5, 0.0],
        [0.0, 570.3, 239.5, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

    # 初始化异步 Agent，并进行一次“热身”调用后再 reset，减少首帧延迟
    agent = InternVLAN1AsyncAgent(args)
    agent.step(
        np.zeros((480, 640, 3), dtype=np.uint8),
        np.zeros((480, 640)),
        np.eye(4),
        "hello",
        args.camera_intrinsic,
    )
    agent.reset()

    # 启动服务：监听 0.0.0.0:8888
    app.run(host='0.0.0.0', port=8888)
