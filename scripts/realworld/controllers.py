#!/usr/bin/env python
"""
控制器模块（MPC 与 PID）

用途：
- `Mpc_controller`：基于模型预测控制（MPC）的轨迹跟踪控制器，适合跟踪服务端返回的连续参考轨迹。
- `PID_controller`：基于 PD（或 PID 中的 P、D 项）的姿态/速度控制器，适合执行离散动作并增量调整目标位姿。

注意：
- 本文件仅包含控制算法，不涉及 ROS 通信与感知前处理；
    实际使用中由客户端节点（例如 http_internvla_client.py）提供里程计、目标轨迹与调用节奏。
"""

import math
import os
import sys

import casadi as ca
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from scipy.interpolate import interp1d


class Mpc_controller:
    def __init__(self, global_planed_traj, N=20, desired_v=0.3, v_max=0.4, w_max=0.4, ref_gap=4):
        """Initialize the MPC controller.

        中文说明：
        - 初始化 MPC 控制器，构建基于非线性优化的轨迹跟踪问题。
        - 预测域长度 `N`、参考点间隔 `ref_gap`、时间步长 `T=0.1s` 共同决定控制频率与参考采样密度。
        - 通过 `v_max`、`w_max` 约束线速度与角速度，`desired_v` 影响参考点的弧长采样节奏。

        Args:
            global_planed_traj (np.ndarray): The global planned trajectory, shape (n, 2).
            N (int): Prediction horizon.
            desired_v (float): Desired linear velocity.
            v_max (float): Maximum linear velocity.
            w_max (float): Maximum angular velocity.
            ref_gap (int): Gap between reference points in the prediction horizon.
        """
        self.N, self.desired_v, self.ref_gap, self.T = N, desired_v, ref_gap, 0.1
        self.ref_traj = self.make_ref_denser(global_planed_traj)
        self.ref_traj_len = N // ref_gap + 1

        # setup mpc problem
        opti = ca.Opti()
        opt_controls = opti.variable(N, 2)
        v, w = opt_controls[:, 0], opt_controls[:, 1]

        opt_states = opti.variable(N + 1, 3)
        # x, y, theta = opt_states[:, 0], opt_states[:, 1], opt_states[:, 2]

        # parameters
        opt_x0 = opti.parameter(3)
        opt_xs = opti.parameter(3 * self.ref_traj_len)  # the intermidia state may also be the parameter

        # system dynamics for mobile manipulator
        f = lambda x_, u_: ca.vertcat(*[u_[0] * ca.cos(x_[2]), u_[0] * ca.sin(x_[2]), u_[1]])  # noqa

        # init_condition
        opti.subject_to(opt_states[0, :] == opt_x0.T)
        for i in range(N):
            x_next = opt_states[i, :] + f(opt_states[i, :], opt_controls[i, :]).T * self.T
            opti.subject_to(opt_states[i + 1, :] == x_next)

        # define the cost function
        Q = np.diag([10.0, 10.0, 0.0])
        R = np.diag([0.05, 0.2])
        obj = 0
        for i in range(N):
            obj = obj + ca.mtimes([opt_controls[i, :], R, opt_controls[i, :].T])
            if i % ref_gap == 0:
                nn = i // ref_gap
                obj = obj + ca.mtimes(
                    [
                        (opt_states[i, :] - opt_xs[nn * 3 : nn * 3 + 3].T),
                        Q,
                        (opt_states[i, :] - opt_xs[nn * 3 : nn * 3 + 3].T).T,
                    ]
                )

        opti.minimize(obj)

        # boundary and control conditions
        opti.subject_to(opti.bounded(0, v, v_max))
        opti.subject_to(opti.bounded(-w_max, w, w_max))

        opts_setting = {
            'ipopt.max_iter': 100,
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.acceptable_tol': 1e-8,
            'ipopt.acceptable_obj_change_tol': 1e-6,
        }
        opti.solver('ipopt', opts_setting)
        # opts_setting = { 'qpsol':'osqp','hessian_approximation':'limited-memory','max_iter':200,'convexify_strategy':'regularize','beta':0.5,'c1':1e-4,'tol_du':1e-3,'tol_pr':1e-6}
        # opti.solver('sqpmethod',opts_setting)

        self.opti = opti
        self.opt_xs = opt_xs
        self.opt_x0 = opt_x0
        self.opt_controls = opt_controls
        self.opt_states = opt_states
        self.last_opt_x_states = None
        self.last_opt_u_controls = None

    def make_ref_denser(self, ref_traj, ratio=50):
        """将稀疏参考轨迹插值加密

        参数：
        - `ref_traj`：形状 (n, 2) 的 xy 参考点序列。
        - `ratio`：每段线性插值的加密倍率（越大越密）。

        返回：
        - 加密后的参考轨迹（(n*ratio, 2)）。
        """
        x_orig = np.arange(len(ref_traj))
        new_x = np.linspace(0, len(ref_traj) - 1, num=len(ref_traj) * ratio)

        interp_func_x = interp1d(x_orig, ref_traj[:, 0], kind='linear')
        interp_func_y = interp1d(x_orig, ref_traj[:, 1], kind='linear')

        uniform_x = interp_func_x(new_x)
        uniform_y = interp_func_y(new_x)
        ref_traj = np.stack((uniform_x, uniform_y), axis=1)

        return ref_traj

    def update_ref_traj(self, global_planed_traj):
        """更新全局参考轨迹并重计算参考长度计数

        注意：`ref_traj_len = N // ref_gap + 1` 与 MPC 代价中采样点数量一致。
        """
        self.ref_traj = self.make_ref_denser(global_planed_traj)
        self.ref_traj_len = self.N // self.ref_gap + 1

    def solve(self, x0):
        """求解一次 MPC 优化，得到控制序列与状态序列

        参数：
        - `x0`：当前状态 `[x, y, yaw]`

        流程：
        - 从加密后的全局轨迹中，基于当前位置选取一定弧长间隔的参考点（`find_reference_traj`）。
        - 将参考点拼接为 `[x, y, 0]`（此处 yaw 作为 0 的占位，可扩展为有向参考）。
        - 设置初值与参数，调用 IPOPT 求解器求解。
        - 返回最优控制与状态，用于控制线程取第一步指令。
        """
        ref_traj = self.find_reference_traj(x0, self.ref_traj)
        # fake a yaw angle
        ref_traj = np.concatenate((ref_traj, np.zeros((ref_traj.shape[0], 1))), axis=1).reshape(-1, 1)

        self.opti.set_value(self.opt_xs, ref_traj.reshape(-1, 1))
        u0 = np.zeros((self.N, 2)) if self.last_opt_u_controls is None else self.last_opt_u_controls
        x00 = np.zeros((self.N + 1, 3)) if self.last_opt_x_states is None else self.last_opt_x_states

        self.opti.set_value(self.opt_x0, x0)
        self.opti.set_initial(self.opt_controls, u0)
        self.opti.set_initial(self.opt_states, x00)

        sol = self.opti.solve()

        self.last_opt_u_controls = sol.value(self.opt_controls)
        self.last_opt_x_states = sol.value(self.opt_states)

        return self.last_opt_u_controls, self.last_opt_x_states

    def reset(self):
        """重置 MPC 的初值缓存（下次求解将不使用上次解的 warm start）"""
        self.last_opt_x_states = None
        self.last_opt_u_controls = None

    def find_reference_traj(self, x0, global_planed_traj):
        """基于当前位置 `x0` 在全局轨迹中选取参考点序列

        逻辑：
        - 找到与当前位置最近的全局轨迹点 `nearest_idx`；
        - 计算累计弧长 `cum_dist`；
        - 以 `desired_v * ref_gap * T` 作为步长弧长，依次选取参考点，直至达到 `ref_traj_len`；
        - 若不足则重复最后一个点。
        """
        ref_traj_pts = []
        # find the nearest point in global_planed_traj
        nearest_idx = np.argmin(np.linalg.norm(global_planed_traj - x0[:2].reshape((1, 2)), axis=1))
        desire_arc_length = self.desired_v * self.ref_gap * self.T
        cum_dist = np.cumsum(np.linalg.norm(np.diff(global_planed_traj, axis=0), axis=1))

        # select the reference points from the nearest point to the end of global_planed_traj
        for i in range(nearest_idx, len(global_planed_traj) - 1):
            if cum_dist[i] - cum_dist[nearest_idx] >= desire_arc_length * len(ref_traj_pts):
                ref_traj_pts.append(global_planed_traj[i, :])
                if len(ref_traj_pts) == self.ref_traj_len:
                    break
        # if the target is reached before the reference trajectory is complete, add the last point of global_planed_traj
        while len(ref_traj_pts) < self.ref_traj_len:
            ref_traj_pts.append(global_planed_traj[-1, :])
        return np.array(ref_traj_pts)


class PID_controller:
    def __init__(self, Kp_trans=1.0, Kd_trans=0.1, Kp_yaw=1.0, Kd_yaw=1.0, max_v=1.0, max_w=1.2):
        """Initialize the PID controller.

        中文说明：
        - 简化的 PD 控制（未包含 I 项）。
        - `Kp_trans/Kd_trans` 针对平移误差，`Kp_yaw/Kd_yaw` 针对朝向误差。
        - `max_v/max_w` 用于限幅，保证输出速度在安全范围内。

        Args:
            Kp_trans (float): Proportional gain for translational error.
            Kd_trans (float): Derivative gain for translational error.
            Kp_yaw (float): Proportional gain for yaw error.
            Kd_yaw (float): Derivative gain for yaw error.
            max_v (float): Maximum linear velocity.
            max_w (float): Maximum angular velocity.
        """
        self.Kp_trans = Kp_trans
        self.Kd_trans = Kd_trans
        self.Kp_yaw = Kp_yaw
        self.Kd_yaw = Kd_yaw
        self.max_v = max_v
        self.max_w = max_w

    def solve(self, odom, target, vel=np.zeros(2)):
        """计算一次控制输出

        参数：
        - `odom`：当前位置的 4x4 齐次变换（仅用到前 2x2 旋转与位移）
        - `target`：目标位姿的 4x4 齐次变换
        - `vel`：当前线/角速度 `[v, w]`，用于 D 项

        返回：
        - `v, w, translation_error, yaw_error`
        """
        translation_error, yaw_error = self.calculate_errors(odom, target)
        v, w = self.pd_step(translation_error, yaw_error, vel[0], vel[1])
        return v, w, translation_error, yaw_error

    def pd_step(self, translation_error, yaw_error, linear_vel, angular_vel):
        """PD 步进计算：按误差与当前速度给出限幅后的控制量"""
        translation_error = max(-1.0, min(1.0, translation_error))
        yaw_error = max(-1.0, min(1.0, yaw_error))

        linear_velocity = self.Kp_trans * translation_error - self.Kd_trans * linear_vel
        angular_velocity = self.Kp_yaw * yaw_error - self.Kd_yaw * angular_vel

        linear_velocity = max(-self.max_v, min(self.max_v, linear_velocity))
        angular_velocity = max(-self.max_w, min(self.max_w, angular_velocity))

        return linear_velocity, angular_velocity

    def calculate_errors(self, odom, target):
        """计算平移与航向误差（在机器人本体坐标系）

        - 平移误差取目标相对位移在本体系前向的投影（前向为正）。
        - 航向误差取 `target_yaw - odom_yaw`，并归一化到 [-pi, pi]。
        """

        dx = target[0, 3] - odom[0, 3]
        dy = target[1, 3] - odom[1, 3]

        odom_yaw = math.atan2(odom[1, 0], odom[0, 0])
        target_yaw = math.atan2(target[1, 0], target[0, 0])

        translation_error = dx * np.cos(odom_yaw) + dy * np.sin(odom_yaw)

        yaw_error = target_yaw - odom_yaw
        yaw_error = (yaw_error + math.pi) % (2 * math.pi) - math.pi

        return translation_error, yaw_error
