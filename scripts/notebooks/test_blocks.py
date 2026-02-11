"""
InternVLA-N1 模型测试脚本（分块版本）
使用方法：
1. 先在终端启动服务器：python scripts/eval/start_server.py --port 8087
2. 复制下面的代码块到 Jupyter Notebook 中
3. 依次运行每个代码块
每个代码块以 # ========== 分隔，可以独立运行
"""

# ========== 块 1: 导入依赖库 ==========
import sys
sys.path.append('.')
sys.path.append('..')

import os
import time
import glob
import numpy as np
import cv2
from PIL import Image

from internnav.configs.agent import AgentCfg
from internnav.utils import AgentClient

print("✓ 依赖库导入成功")


# ========== 块 2: 配置参数 ==========
# 模型检查点路径（请根据实际情况修改）
checkpoint_path = '/data3/ltd/InternNav/checkpoints/InternVLA-N1-wo-dagger'

# 使用 realworld_sample_data 数据集
data_dir = '/data3/ltd/InternNav/assets/realworld_sample_data1'
instruction_file = os.path.join(data_dir, 'instruction.txt')

# 服务器配置
server_host = 'localhost'
server_port = 8088

# 设备配置
device = 'cuda:7'

# 相机参数
camera_intrinsic = [[585.0, 0.0, 320.0], [0.0, 585.0, 240.0], [0.0, 0.0, 1.0]]
width = 640
height = 480
hfov = 79

# 可视化输出目录
vis_output_dir = './logs/test_blocks_vis'
os.makedirs(vis_output_dir, exist_ok=True)

print("配置参数：")
print(f"  检查点路径: {checkpoint_path}")
print(f"  数据目录: {data_dir}")
print(f"  指令文件: {instruction_file}")
print(f"  服务器: {server_host}:{server_port}")
print(f"  设备: {device}")
print(f"  可视化输出: {vis_output_dir}")


# ========== 块 3: 验证路径 ==========
# 检查检查点路径
if os.path.exists(checkpoint_path):
    print(f"✓ 检查点路径存在: {checkpoint_path}")
    files = os.listdir(checkpoint_path)
    print(f"  包含 {len(files)} 个文件/目录")
else:
    print(f"✗ 检查点路径不存在: {checkpoint_path}")
    print("  请修改 checkpoint_path 为正确的路径")

# 检查数据目录路径
if os.path.exists(data_dir):
    print(f"✓ 数据目录存在: {data_dir}")
    # 获取所有图像文件
    rgb_paths = sorted(glob.glob(os.path.join(data_dir, 'debug_raw_*.jpg')))
    print(f"  找到 {len(rgb_paths)} 张图像")
else:
    print(f"✗ 数据目录不存在: {data_dir}")
    print("  请修改 data_dir 为正确的路径")

# 检查指令文件
if os.path.exists(instruction_file):
    print(f"✓ 指令文件存在: {instruction_file}")
    with open(instruction_file, 'r') as f:
        instruction = f.read().strip()
    print(f"  指令内容: {instruction}")
else:
    print(f"✗ 指令文件不存在: {instruction_file}")
    print("  将使用默认指令")
    instruction = 'go to the white box'


# ========== 块 4: 配置 Agent ==========
agent_cfg = AgentCfg(
    server_host=server_host,
    server_port=server_port,
    model_name='internvla_n1',
    ckpt_path='',
    model_settings={
        'policy_name': "InternVLAN1_Policy",
        'state_encoder': None,
        'env_num': 1,
        'sim_num': 1,
        'model_path': checkpoint_path,
        'camera_intrinsic': camera_intrinsic,
        'width': width,
        'height': height,
        'hfov': hfov,
        'resize_w': 384,
        'resize_h': 384,
        'max_new_tokens': 1024,
        'num_frames': 32,
        'num_history': 8,
        'num_future_steps': 4,
        'device': device,
        'predict_step_nums': 32,
        'continuous_traj': True,
        'infer_mode': 'partial_async',
        # Debug settings
        'vis_debug': False,
        'vis_debug_path': './logs/test_n1/vis_debug',
    }
)

print("✓ Agent 配置创建成功")
print(f"  模型名称: {agent_cfg.model_name}")
print(f"  模型路径: {agent_cfg.model_settings['model_path']}")


# ========== 块 5: 初始化 Agent Client ==========
# ⚠️ 注意：运行此块前，请确保服务器已启动！
# 在终端运行：python scripts/eval/start_server.py --port 8088

print(f"连接到服务器 {server_host}:{server_port}...")

try:
    agent = AgentClient(agent_cfg)
    print("✓ Agent 客户端初始化成功！")
    print("  模型已加载到服务器")
except Exception as e:
    print(f"✗ 初始化失败: {e}")
    print("\n请检查：")
    print("  1. 服务器是否正在运行")
    print("  2. 端口号是否正确")
    print("  3. 检查点路径是否正确")
    raise


# ========== 块 6: 定义可视化函数 ==========
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image, ImageDraw, ImageFont

def annotate_image(idx, image, llm_output, trajectory, pixel_goal, output_dir):
    """
    在图像上添加文本注释和轨迹可视化

    参数:
        idx: 图像索引
        image: RGB图像 (numpy array)
        llm_output: 动作输出文本
        trajectory: 轨迹点列表
        pixel_goal: 像素目标点 [y, x]
        output_dir: 输出目录
    """
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    font_size = 20
    try:
        font = ImageFont.truetype("DejaVuSansMono.ttf", font_size)
    except:
        font = ImageFont.load_default()

    # 准备文本内容
    text_content = []
    text_content.append(f"Frame    Id  : {idx}")
    text_content.append(f"Actions      : {llm_output}")

    # 计算文本框大小
    max_width = 0
    total_height = 0
    for line in text_content:
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = 26
        max_width = max(max_width, text_width)
        total_height += text_height

    # 绘制文本框
    padding = 10
    box_x, box_y = 10, 10
    box_width = max_width + 2 * padding
    box_height = total_height + 2 * padding
    draw.rectangle([box_x, box_y, box_x + box_width, box_y + box_height], fill='black')

    # 绘制文本
    text_color = 'white'
    y_position = box_y + padding
    for line in text_content:
        draw.text((box_x + padding, y_position), line, fill=text_color, font=font)
        y_position += 26

    image = np.array(image)

    # 绘制轨迹可视化（右上角）
    if trajectory is not None and len(trajectory) > 0:
        img_height, img_width = image.shape[:2]

        # 窗口参数
        window_size = 200
        window_margin = 0
        window_x = img_width - window_size - window_margin
        window_y = window_margin

        # 提取轨迹点
        traj_points = []
        for point in trajectory:
            if isinstance(point, (list, tuple, np.ndarray)) and len(point) >= 2:
                traj_points.append([float(point[0]), float(point[1])])

        if len(traj_points) > 0:
            traj_array = np.array(traj_points)
            x_coords = traj_array[:, 0]
            y_coords = traj_array[:, 1]

            # 创建matplotlib图形
            fig, ax = plt.subplots(figsize=(2, 2), dpi=100)
            fig.patch.set_alpha(0.6)
            fig.patch.set_facecolor('gray')
            ax.set_facecolor('lightgray')

            # 绘制轨迹
            ax.plot(y_coords, x_coords, 'b-', linewidth=2, label='Trajectory')
            ax.plot(y_coords[0], x_coords[0], 'go', markersize=6, label='Start')
            ax.plot(y_coords[-1], x_coords[-1], 'ro', markersize=6, label='End')
            ax.plot(0, 0, 'w+', markersize=10, markeredgewidth=2, label='Origin')

            # 设置坐标轴
            ax.set_xlabel('Y (left +)', fontsize=8)
            ax.set_ylabel('X (up +)', fontsize=8)
            ax.invert_xaxis()
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.3, linewidth=0.5)
            ax.set_aspect('equal', adjustable='box')
            ax.legend(fontsize=6, loc='upper right')
            plt.tight_layout(pad=0.3)

            # 转换为numpy数组
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            plot_img = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
            plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)

            # 调整大小并叠加
            plot_img = cv2.resize(plot_img, (window_size, window_size))
            image[window_y:window_y+window_size, window_x:window_x+window_size] = plot_img

    # 绘制像素目标点
    if pixel_goal is not None:
        cv2.circle(image, (pixel_goal[1], pixel_goal[0]), 5, (255, 0, 0), -1)

    # 保存图像
    image = Image.fromarray(image).convert('RGB')
    image.save(f'{output_dir}/rgb_{idx}_annotated.png')
    return np.array(image)

print("✓ 可视化函数定义成功")


# ========== 块 7: 批量加载和处理图像 ==========
# 加载所有图像并进行推理
print(f"开始批量处理 {len(rgb_paths)} 张图像...")
print(f"指令: {instruction}")
print("=" * 80)

# 存储结果
results = []
visualized_images = []

# 相机内参（用于观察数据）
intrinsics = {
    'fx': camera_intrinsic[0][0],
    'fy': camera_intrinsic[1][1],
    'cx': camera_intrinsic[0][2],
    'cy': camera_intrinsic[1][2],
    'width': width,
    'height': height
}

for i, rgb_path in enumerate(rgb_paths):
    # 读取RGB图像（使用OpenCV读取为BGR格式）
    rgb_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)  # BGR格式

    # 如果读取失败，尝试使用PIL
    if rgb_bgr is None:
        rgb_rgb = np.asarray(Image.open(rgb_path).convert('RGB'))
        rgb_bgr = cv2.cvtColor(rgb_rgb, cv2.COLOR_RGB2BGR)

    # 创建虚拟深度图（如果没有实际深度数据）
    # 注意：在实际使用中应该使用真实的深度数据
    depth = np.full((rgb_bgr.shape[0], rgb_bgr.shape[1], 1), 10.0, dtype=np.float32)

    # 创建观察数据（必须包含所有必需字段）
    obs = {
        'rgb': rgb_bgr,  # BGR格式（OpenCV标准）
        'depth': depth,
        'instruction': instruction,
        'timestamp_s': time.time(),
        'intrinsics': intrinsics  # 添加相机内参
    }

    # 执行推理
    start_time = time.time()
    result = agent.step([obs])
    inference_time = time.time() - start_time

    # 提取动作
    action = result[0]['action'][0]

    # 动作映射
    action_map = {
        0: "MOVE_FORWARD",
        1: "TURN_RIGHT",
        2: "TURN_LEFT",
        3: "STOP"
    }

    action_names = {
        0: "前进",
        1: "右转",
        2: "左转",
        3: "停止"
    }

    # 输出结果
    print(f"[{i+1}/{len(rgb_paths)}] {os.path.basename(rgb_path)}")
    print(f"  动作: {action} ({action_map.get(action, 'Unknown')} / {action_names.get(action, '未知')})")
    print(f"  推理时间: {inference_time:.2f}s")

    # 保存结果
    result_dict = {
        'image_path': rgb_path,
        'action': action,
        'action_name': action_map.get(action, 'Unknown'),
        'inference_time': inference_time
    }
    results.append(result_dict)

    # 可视化（如果结果中有轨迹信息）
    trajectory = result[0].get('trajectory', None)
    pixel_goal = result[0].get('pixel_goal', None)

    # 创建可视化图像（转换为RGB用于可视化）
    rgb_rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    image_id = os.path.basename(rgb_path).replace('debug_raw_', '').replace('.jpg', '')
    vis_img = annotate_image(
        idx=image_id,
        image=rgb_rgb,  # 传递RGB格式用于可视化
        llm_output=f"{action_map.get(action, 'Unknown')}",
        trajectory=trajectory,
        pixel_goal=pixel_goal,
        output_dir=vis_output_dir
    )
    visualized_images.append(vis_img)
    print()

print("=" * 80)
print(f"✓ 批量处理完成！共处理 {len(results)} 张图像")
print(f"✓ 可视化结果已保存到: {vis_output_dir}")


# ========== 块 8: 显示可视化结果 ==========
# 在Jupyter Notebook中显示前几张可视化图像
print("\n显示前5张可视化结果：")
print("=" * 80)

num_display = min(5, len(visualized_images))
fig, axes = plt.subplots(1, num_display, figsize=(5*num_display, 5))

if num_display == 1:
    axes = [axes]

for i in range(num_display):
    axes[i].imshow(visualized_images[i])
    axes[i].set_title(f'Image {i+1}')
    axes[i].axis('off')

plt.tight_layout()
plt.show()

print(f"✓ 已显示 {num_display} 张可视化结果")
print(f"完整结果保存在: {vis_output_dir}")


# ========== 块 9: 查看所有可视化结果（可选）==========
# 显示所有保存的可视化图像
print("\n显示所有可视化结果：")
print("=" * 80)

vis_image_paths = sorted(glob.glob(os.path.join(vis_output_dir, '*_annotated.png')))
print(f"找到 {len(vis_image_paths)} 张可视化图像")

for img_path in vis_image_paths:
    img = Image.open(img_path)
    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    plt.title(os.path.basename(img_path))
    plt.axis('off')
    plt.show()

print("✓ 所有可视化结果显示完成！")


# ========== 块 10: 统计分析 ==========
print("\n动作统计分析：")
print("=" * 80)

# 统计每个动作的出现次数
action_counts = {}
action_map = {
    0: "MOVE_FORWARD",
    1: "TURN_RIGHT",
    2: "TURN_LEFT",
    3: "STOP"
}

for result in results:
    action = result['action']
    action_name = action_map.get(action, 'Unknown')
    action_counts[action_name] = action_counts.get(action_name, 0) + 1

print("动作分布：")
for action_name, count in sorted(action_counts.items()):
    percentage = (count / len(results)) * 100
    print(f"  {action_name}: {count} 次 ({percentage:.1f}%)")

# 平均推理时间
avg_inference_time = np.mean([r['inference_time'] for r in results])
print(f"\n平均推理时间: {avg_inference_time:.2f} 秒")
print("=" * 80)


# ========== 块 11: 保存测试结果 ==========
import json
from datetime import datetime

# 准备保存的结果
test_result = {
    'timestamp': datetime.now().isoformat(),
    'checkpoint': checkpoint_path,
    'data_dir': data_dir,
    'instruction': instruction,
    'device': device,
    'total_images': len(results),
    'avg_inference_time': float(avg_inference_time),
    'action_statistics': action_counts,
    'detailed_results': [
        {
            'image_path': r['image_path'],
            'action': int(r['action']),
            'action_name': r['action_name'],
            'inference_time': float(r['inference_time'])
        }
        for r in results
    ]
}

# 保存到文件
output_file = os.path.join(vis_output_dir, 'test_results.json')
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(test_result, f, indent=2, ensure_ascii=False)

print(f"\n✓ 测试结果已保存到: {output_file}")
print("\n摘要：")
print(f"  处理图像数: {test_result['total_images']}")
print(f"  平均推理时间: {test_result['avg_inference_time']:.2f}s")
print(f"  动作分布: {test_result['action_statistics']}")

print("\n" + "=" * 80)
print("✓ 所有测试完成！")
print("=" * 80)