import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from matplotlib.backends.backend_agg import FigureCanvasAgg #为了了在内存中渲染图像

def annotate_image(idx, image, llm_output, trajectory, pixel_goal, output_dir):
    """在图像上标注 LLM 输出、俯视轨迹和像素目标点。

    pixel_goal 可能在某些帧为空（模型未返回像素级目标），此时只标注文本和轨迹。
    """
    image_pil = Image.fromarray(image).convert('RGB')
    draw = ImageDraw.Draw(image_pil)
    try:
        font = ImageFont.truetype("DejaVuSansMono.ttf", 20) # 使用等宽字体以对齐文本
    except:
        font = ImageFont.load_default()

    # 文本信息：帧 ID、截断后的 LLM 输出、像素目标状态
    text_main = f"ID: {idx} | LLM: {llm_output[:50]}..."
    text_goal = (
        f"Pixel goal: ({int(pixel_goal[1])}, {int(pixel_goal[0])})" if pixel_goal is not None else "Pixel goal: 未提供"
    )
    draw.text((10, 10), text_main, fill='white', font=font, stroke_width=1, stroke_fill='black')
    draw.text((10, 35), text_goal, fill='white', font=font, stroke_width=1, stroke_fill='black')

    canvas_img = np.array(image_pil, copy=True)  # 确保可写

    # 轨迹缩略图：右上角叠加俯视轨迹，轨迹缺失时跳过
    if trajectory is not None and len(trajectory) > 0:
        img_h, img_w = canvas_img.shape[:2]
        window_size = 200
        margin = 0
        wx = img_w - window_size - margin
        wy = margin

        traj_xy = np.array(trajectory, dtype=np.float32)[:, :2] # 仅取 x,y 坐标
        # 按时间戳命名保存轨迹，避免依赖 self 上下文
        #ts_name = f"traj_{idx:04d}.txt"
        #traj_xy_txt_path = os.path.join(output_dir, ts_name)
        #np.savetxt(traj_xy_txt_path, traj_xy, fmt="%.6f", header="x y")

        fig, ax = plt.subplots(figsize=(2, 2), dpi=100)
        fig.patch.set_alpha(0.6)
        fig.patch.set_facecolor('gray')
        ax.set_facecolor('lightgray')

        # 轨迹绘制：坐标系与原先一致，x 前进、y 左右
        ax.plot(traj_xy[:, 1], traj_xy[:, 0], 'b-', linewidth=2, label='Trajectory')
        # 起点取轨迹第一点，绿色；终点取最后一点，红色
        ax.plot(traj_xy[0, 1], traj_xy[0, 0], 'go', markersize=6, label='Start')
        ax.plot(traj_xy[-1, 1], traj_xy[-1, 0], 'ro', markersize=6, label='End')

        ax.set_xlabel('Y (left +)', fontsize=8)
        ax.set_ylabel('X (up +)', fontsize=8)
        #ax.invert_xaxis()  # X轴反转

        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.set_aspect('equal', adjustable='box')
        ax.legend(fontsize=6, loc='upper right')
        plt.tight_layout(pad=0.3)

        fig_canvas = FigureCanvasAgg(fig)
        fig_canvas.draw()
        plot_rgba = np.asarray(fig_canvas.buffer_rgba())
        plot_rgb = plot_rgba[..., :3]
        plot_rgb = cv2.resize(plot_rgb, (window_size, window_size))
        canvas_img[wy:wy+window_size, wx:wx+window_size] = plot_rgb
        plt.close(fig)

    # 像素目标：仅在模型给出目标点时绘制
    if pixel_goal is not None:
        u, v = int(pixel_goal[1]), int(pixel_goal[0])
        cv2.circle(canvas_img, (u, v), 8, (255, 0, 0), -1)
        cv2.circle(canvas_img, (u, v), 9, (0, 0, 0), 2)

    # 始终保存当前帧的可视化，无论是否有目标点或轨迹
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'vis_{idx:04d}.png')
    Image.fromarray(canvas_img).save(save_path)
    print(f"save to {save_path}")
    return canvas_img