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

###=================== 系统架构图 ====================
'''┌─────────────────────────────────────────────────────────────┐
│                     Wheeltec Robot                          │
│                                                             │
│  ┌──────────────────┐         ┌─────────────────────────┐  │
│  │  Planning Thread │         │   Control Thread        │  │
│  │   (0.3s 周期)     │         │    (0.1s 周期)          │  │
│  │                  │         │                         │  │
│  │  1. 采集传感器   │         │  1. 读取控制模式        │  │
│  │  2. 发送到服务器 │────────>│  2. 执行 MPC/PID       │  │
│  │  3. 接收轨迹/动作│         │  3. 发布速度指令        │  │
│  │  4. 更新控制器   │         │                         │  │
│  └──────────────────┘         └─────────────────────────┘  │
│         │                              │                    │
│         │ (更新参考轨迹)                │ (发布 /cmd_vel)   │
│         ▼                              ▼                    │
│    [MPC/PID 控制器]            [机器人底盘]                 │
└─────────────────────────────────────────────────────────────┘'''

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
SERVER_HOST = '115.190.160.32'  # ⚠️ 修改为你的服务器 IP
SERVER_PORT = 8888
SERVER_URL = f'http://{SERVER_HOST}:{SERVER_PORT}/eval_dual'

# # Astra S 相机内参 (640x480)
# CAMERA_INTRINSIC = np.array([
#     [570.3, 0.0, 319.5, 0.0],
#     [0.0, 570.3, 239.5, 0.0],
#     [0.0, 0.0, 1.0, 0.0],
#     [0.0, 0.0, 0.0, 1.0]
# ])

#实测 Astra S 相机内参 (640x480)
CAMERA_INTRINSIC = np.array([
    [521.76904, 0.0,       325.47444, 0.0],
    [0.0,       525.44232, 240.21452, 0.0],
    [0.0,       0.0,       1.0,       0.0],
    [0.0,       0.0,       0.0,       1.0]
])

#NAVIGATION_INSTRUCTION = "Turn around and walk out of this office. Turn towards your slight right at the chair. Move forward to the walkway \
#and go near the red bin. You can see an open door on your right side, go inside the open door. Stop at the computer monitor"

# ⚠️ 修改为你的导航指令
#NAVIGATION_INSTRUCTION = "Turn right and you will find a green plant. Move forward to the green plant and stop near the green Plant."
#NAVIGATION_INSTRUCTION = "Do a full circle"
NAVIGATION_INSTRUCTION = "Turn right you will find a black bin. Move forward to the black bin. Then you will find a room near the bin. Step into this room and stop."
#NAVIGATION_INSTRUCTION = "Turn right"
#NAVIGATION_INSTRUCTION = "Move forward"
#NAVIGATION_INSTRUCTION = "Find a green plant and stop near it."
#NAVIGATION_INSTRUCTION = "Turn around and find a green plant. Move forward to the green plant and stop near the green Plant."


#NAVIGATION_INSTRUCTION = "Move forward and pass between two green pot plant. Approach the red pot plant, circle around it, and then stop"
#NAVIGATION_INSTRUCTION = "Move forward and pass between two green plant. Then turn towards your slight left at the red plant. Stop at the brown door"
#NAVIGATION_INSTRUCTION = "Turn left and you will find a green plant. Turn towards your slight right at the green plant. Move forward to the second green plant and stop near the green Plant."
#NAVIGATION_INSTRUCTION = "Turn right and you will find a green plant. Move forward to the green plant and stop near the green Plant."

#NAVIGATION_INSTRUCTION = "Turn around and walk out of this office. Then turn left, and you will see a black bin. Move forward to the door and stop near the door."
#NAVIGATION_INSTRUCTION = "Go to the black bin"
#NAVIGATION_INSTRUCTION = "Go to the black chair"
#NAVIGATION_INSTRUCTION = "Move forward and you will find a black bin. Turn left and find a water cup. Stop near the water cup."
#NAVIGATION_INSTRUCTION = "Navigate to the yellow object"
#NAVIGATION_INSTRUCTION = "Turn right"
#NAVIGATION_INSTRUCTION = "Turn left"
#NAVIGATION_INSTRUCTION = "Move forward and you will find a black chair. Then turn left. Stop near the wall."
#NAVIGATION_INSTRUCTION = "Move backward"


# 速度限制（Wheeltec 安全参数）
MAX_LINEAR_VEL = 0.25 #0.25   # m/s
MAX_ANGULAR_VEL = 0.5 #0.5   # rad/s


# ==================== 服务器通信函数 ====================
def dual_sys_eval(image_bytes, depth_bytes, url=SERVER_URL):
    """向服务器发送图像并获取动作"""
    global policy_init, http_idx, first_running_time

    data = {"reset": policy_init, "idx": http_idx, "instruction": NAVIGATION_INSTRUCTION}
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
    global desired_v, desired_w, current_control_mode

    while True:
        # 等待 manager 初始化
        if manager is None:
            time.sleep(0.1)
            continue

        if current_control_mode == ControlMode.MPC_Mode:
            # MPC 控制模式
            odom_rw_lock.acquire_read()
            odom = manager.odom.copy() if manager.odom else None
            odom_rw_lock.release_read()

            if mpc is not None and odom is not None:
                mpc_rw_lock.acquire_read()
                local_mpc = mpc
                mpc_rw_lock.release_read()

                try:
                    opt_u_controls, opt_x_states = local_mpc.solve(np.array(odom))
                    v, w = opt_u_controls[0, 0], opt_u_controls[0, 1]

                    # 限速
                    v = np.clip(v, 0, MAX_LINEAR_VEL)
                    w = np.clip(w, -MAX_ANGULAR_VEL, MAX_ANGULAR_VEL)

                    desired_v, desired_w = v, w
                    manager.move(v, 0.0, w)
                except Exception as e:
                    print(f"MPC solve error: {e}")

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
    global trajs_in_world, current_control_mode, mpc

    # 等待 manager 初始化
    while manager is None:
        time.sleep(0.1)

    while True: #等待新图像到达
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

        # 时间同步：找到最接近的里程计数据 odom
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

            # 请求服务器 HTTP POST 到服务器（InternVLA-N1 推理）
            response = dual_sys_eval(rgb_bytes, depth_bytes)
            # 返回格式：
            # - {"trajectory": [[x1,y1], [x2,y2], ...]}  # MPC 模式
            # - {"discrete_action": [1, 2, ...]}         # PID 模式

            # 处理轨迹输出 (MPC 模式)
            if 'trajectory' in response:
                trajectory = response['trajectory']
                trajs_in_world = []
                odom = odom_infer

                traj_array = np.array(trajectory)
                if len(traj_array) > 0:
                    traj_len = np.linalg.norm(traj_array[-1][:2])
                    print(f"Received trajectory, length: {traj_len:.2f}m, points: {len(trajectory)}")

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

                if len(trajs_in_world) > 0:
                    trajs_in_world = np.array(trajs_in_world)
                    manager.last_trajs_in_world = trajs_in_world

                    # 更新 MPC 控制器
                    mpc_rw_lock.acquire_write()
                    if mpc is None:
                        mpc = Mpc_controller(np.array(trajs_in_world))
                    else:
                        mpc.update_ref_traj(np.array(trajs_in_world))
                    manager.request_cnt += 1
                    mpc_rw_lock.release_write()

                    current_control_mode = ControlMode.MPC_Mode
                else:
                    print("Warning: No valid trajectory points after filtering")

            # 处理离散动作输出 (PID 模式)
            elif 'discrete_action' in response:
                actions = response['discrete_action']
                if actions != [5]:  # 5=look down
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
        depth_sub = Subscriber(self, Image, "/camera/depth/image_raw")

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
        """RGB-Depth 同步回调 - 修复版"""
        self.get_logger().info('Received synchronized RGB and Depth frames')
        
        # ========== 处理 RGB ==========
        try:
            raw_image = self.cv_bridge.imgmsg_to_cv2(rgb_msg, 'rgb8')
            self.rgb_image = raw_image
            
            # 编码为 JPEG
            image = PIL_Image.fromarray(self.rgb_image)
            image_bytes = io.BytesIO()
            image.save(image_bytes, format='JPEG', quality=95)
            image_bytes.seek(0)
        except Exception as e:
            self.get_logger().error(f'RGB conversion failed: {e}')
            return
        
        # ========== 处理 Depth ==========
        try:
            # Astra S 输出 16UC1, 单位毫米
            raw_depth = self.cv_bridge.imgmsg_to_cv2(depth_msg, '16UC1')
            raw_depth = raw_depth.astype(np.float32)
            
            # 清理无效值
            raw_depth[np.isnan(raw_depth)] = 0
            raw_depth[np.isinf(raw_depth)] = 0
            raw_depth[raw_depth < 0] = 0
            
            # 转换为米
            self.depth_image = raw_depth / 1000.0
            
            # 限制有效范围 (Astra S: 0.4m - 4.0m)
            self.depth_image[self.depth_image > 4.0] = 0
            
            # 检查有效深度比例
            valid_mask = (self.depth_image >= 0.4) & (self.depth_image <= 4.0)
            valid_ratio = np.sum(valid_mask) / self.depth_image.size
            
            if valid_ratio < 0.03:
                self.get_logger().warn(
                    f'Low valid depth: {valid_ratio:.1%} '
                    f'(min={self.depth_image[valid_mask].min():.2f}, '
                    f'max={self.depth_image[valid_mask].max():.2f})'
                )
            
            # ✅ 修复：编码为 32-bit PNG (服务器期望格式)
            # 单位 0.1mm, uint32 范围 0 - 4294967295 (约 429 米)
            depth_scaled = np.clip(self.depth_image * 10000.0, 0, 4294967295).astype(np.uint32)
            depth_pil = PIL_Image.fromarray(depth_scaled, mode='I')  # 'I' = 32-bit signed integer
            depth_bytes = io.BytesIO()
            depth_pil.save(depth_bytes, format='PNG')
            depth_bytes.seek(0)
            
        except Exception as e:
            self.get_logger().error(f'Depth conversion failed: {e}')
            return
        
        # ========== 线程安全更新 ==========
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