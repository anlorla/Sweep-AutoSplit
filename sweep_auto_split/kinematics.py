"""
前向运动学模块

使用 Piper SDK 的 FK 实现计算末端执行器位置
支持工具偏移（刷尖）和桌面距离计算

改进点：
1. FK 返回完整变换矩阵（position + rotation）
2. 支持四元数转旋转矩阵
3. 工具偏移配置，计算刷尖位置
4. 桌面平面配置，计算刷尖到桌面距离
5. 速度统一为 m/s，额外输出 v_z
"""

import math
import numpy as np
from typing import Tuple, Optional, NamedTuple
from dataclasses import dataclass, field


# ============================================================
# 配置类
# ============================================================

@dataclass
class ToolConfig:
    """
    工具配置（刷子）

    定义从法兰坐标系原点到工具末端（刷尖）的偏移
    """
    # 工具偏移向量（法兰坐标系下），单位：米
    # 默认：刷子沿 Z 轴向下 30cm
    tip_offset: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.30]))

    # 工具名称（用于日志）
    name: str = "brush"


@dataclass
class TablePlane:
    """
    桌面平面配置

    定义桌面平面方程：n^T(x - x0) = 0
    用于计算刷尖到桌面的有符号距离
    """
    # 桌面法向量（指向上方）
    normal: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))

    # 桌面上一点（相对于机器人基座坐标系），单位：米
    # 默认：基座在桌面上，基座原点距桌面 5cm
    # 所以桌面 z = -0.05m（在基座原点下方）
    point: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -0.05]))

    def distance_to_plane(self, point: np.ndarray) -> float:
        """
        计算点到平面的有符号距离

        正值：点在平面上方
        负值：点在平面下方（穿透桌面）

        Args:
            point: 3D 点坐标 [x, y, z]

        Returns:
            有符号距离（米）
        """
        return np.dot(self.normal, point - self.point)

    def distance_to_plane_batch(self, points: np.ndarray) -> np.ndarray:
        """
        批量计算点到平面的距离

        Args:
            points: 点坐标数组 [N, 3]

        Returns:
            距离数组 [N]
        """
        return np.dot(points - self.point, self.normal)


@dataclass
class KinematicsConfig:
    """运动学计算的完整配置"""
    tool: ToolConfig = field(default_factory=ToolConfig)
    table: TablePlane = field(default_factory=TablePlane)
    fps: float = 10.0  # 帧率，用于速度计算


class KinematicsResult(NamedTuple):
    """运动学计算结果"""
    # 法兰位置轨迹 [N, 3]（米）
    flange_positions: np.ndarray

    # 法兰姿态轨迹 [N, 3, 3]（旋转矩阵）
    flange_rotations: np.ndarray

    # 刷尖位置轨迹 [N, 3]（米）
    tip_positions: np.ndarray

    # 刷尖到桌面距离 [N]（米），正值表示在桌面上方
    tip_to_table_distance: np.ndarray

    # 水平速度 [N]（米/秒）
    v_xy: np.ndarray

    # 垂向速度 [N]（米/秒），正值表示向上
    v_z: np.ndarray

    # 总速度 [N]（米/秒）
    v_total: np.ndarray


# ============================================================
# 辅助函数
# ============================================================

def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """
    四元数转旋转矩阵

    Args:
        q: 四元数 [qx, qy, qz, qw]（注意顺序！）

    Returns:
        3x3 旋转矩阵
    """
    qx, qy, qz, qw = q

    # 归一化
    norm = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    if norm < 1e-10:
        return np.eye(3)
    qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm

    # 旋转矩阵
    R = np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)]
    ])

    return R


def quaternion_to_rotation_matrix_batch(q_batch: np.ndarray) -> np.ndarray:
    """
    批量四元数转旋转矩阵

    Args:
        q_batch: 四元数数组 [N, 4]，格式 [qx, qy, qz, qw]

    Returns:
        旋转矩阵数组 [N, 3, 3]
    """
    N = len(q_batch)
    R_batch = np.zeros((N, 3, 3))

    for i in range(N):
        R_batch[i] = quaternion_to_rotation_matrix(q_batch[i])

    return R_batch


def compute_tip_position(flange_pos: np.ndarray, flange_rot: np.ndarray,
                         tip_offset: np.ndarray) -> np.ndarray:
    """
    计算工具末端（刷尖）位置

    p_tip = p_flange + R_flange @ o_tip

    Args:
        flange_pos: 法兰位置 [3]
        flange_rot: 法兰姿态（旋转矩阵）[3, 3]
        tip_offset: 工具偏移（法兰坐标系下）[3]

    Returns:
        刷尖位置 [3]
    """
    return flange_pos + flange_rot @ tip_offset


def compute_tip_positions_batch(flange_positions: np.ndarray,
                                 flange_rotations: np.ndarray,
                                 tip_offset: np.ndarray) -> np.ndarray:
    """
    批量计算刷尖位置

    Args:
        flange_positions: 法兰位置 [N, 3]
        flange_rotations: 法兰姿态 [N, 3, 3]
        tip_offset: 工具偏移 [3]

    Returns:
        刷尖位置 [N, 3]
    """
    N = len(flange_positions)
    tip_positions = np.zeros((N, 3))

    for i in range(N):
        tip_positions[i] = compute_tip_position(
            flange_positions[i], flange_rotations[i], tip_offset
        )

    return tip_positions


def compute_velocities(positions: np.ndarray, fps: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算速度（m/s）

    Args:
        positions: 位置轨迹 [N, 3]
        fps: 帧率

    Returns:
        (v_xy, v_z, v_total)：水平速度、垂向速度、总速度 [N]
    """
    N = len(positions)
    dt = 1.0 / fps

    if N < 2:
        return np.zeros(N), np.zeros(N), np.zeros(N)

    # 计算位移差
    delta = np.diff(positions, axis=0)  # [N-1, 3]

    # 水平速度 v_xy = sqrt(dx^2 + dy^2) / dt
    v_xy_diff = np.linalg.norm(delta[:, :2], axis=1) / dt

    # 垂向速度 v_z = dz / dt（正值表示向上）
    v_z_diff = delta[:, 2] / dt

    # 总速度
    v_total_diff = np.linalg.norm(delta, axis=1) / dt

    # 首帧处理：用第二帧的值代替 0（避免影响分位数阈值）
    v_xy = np.concatenate([[v_xy_diff[0]], v_xy_diff])
    v_z = np.concatenate([[v_z_diff[0]], v_z_diff])
    v_total = np.concatenate([[v_total_diff[0]], v_total_diff])

    return v_xy, v_z, v_total


# ============================================================
# Piper 前向运动学
# ============================================================

class PiperForwardKinematics:
    """
    Piper 机器人前向运动学

    基于 Denavit-Hartenberg 参数计算末端执行器位置和姿态
    参考: /home/zeno-yifan/NPM-Project/NPM-Ros/piper_sdk/piper_sdk/kinematics/piper_fk.py
    """

    def __init__(self, dh_is_offset: int = 0x01):
        """
        初始化 DH 参数

        Args:
            dh_is_offset: DH参数偏移标志 (0x00 或 0x01)
        """
        self.PI = math.pi
        self.RADIAN = 180 / self.PI

        # Denavit-Hartenberg 参数（单位：mm）
        if dh_is_offset == 0x01:
            self._a = [0, 0, 285.03, -21.98, 0, 0]
            self._alpha = [0, -self.PI / 2, 0, self.PI / 2, -self.PI / 2, self.PI / 2]
            self._theta = [0, -self.PI * 172.22 / 180, -102.78 / 180 * self.PI, 0, 0, 0]
            self._d = [123, 0, 0, 250.75, 0, 91]
        else:
            self._a = [0, 0, 285.03, -21.98, 0, 0]
            self._alpha = [0, -self.PI / 2, 0, self.PI / 2, -self.PI / 2, self.PI / 2]
            self._theta = [0, -self.PI * 174.22 / 180, -100.78 / 180 * self.PI, 0, 0, 0]
            self._d = [123, 0, 0, 250.75, 0, 91]

    def _link_transformation(self, alpha: float, a: float, theta: float, d: float) -> np.ndarray:
        """
        计算单个连杆的变换矩阵

        Args:
            alpha: 连杆扭转角 (rad)
            a: 连杆长度 (mm)
            theta: 关节角度 (rad)
            d: 连杆偏移 (mm)

        Returns:
            4x4 变换矩阵
        """
        calpha = math.cos(alpha)
        salpha = math.sin(alpha)
        ctheta = math.cos(theta)
        stheta = math.sin(theta)

        T = np.array([
            [ctheta, -stheta, 0, a],
            [stheta * calpha, ctheta * calpha, -salpha, -salpha * d],
            [stheta * salpha, ctheta * salpha, calpha, calpha * d],
            [0, 0, 0, 1]
        ])

        return T

    def forward_kinematics(self, joint_angles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算前向运动学，返回末端执行器位置和姿态

        Args:
            joint_angles: 关节角度 [6] 或 [7]（rad），第7个是 gripper

        Returns:
            (position, rotation_matrix)
            - position: 末端位置 [x, y, z]（米）
            - rotation_matrix: 末端姿态 [3, 3]
        """
        # 只取前 6 个关节（不包括 gripper）
        q = joint_angles[:6] if len(joint_angles) > 6 else joint_angles

        # 计算各连杆变换矩阵
        T = np.eye(4)
        for i in range(6):
            theta_i = q[i] + self._theta[i]
            T_i = self._link_transformation(self._alpha[i], self._a[i], theta_i, self._d[i])
            T = T @ T_i

        # 提取位置 (mm -> m)
        position = T[:3, 3] / 1000.0

        # 提取旋转矩阵
        rotation = T[:3, :3]

        return position, rotation

    def forward_kinematics_full(self, joint_angles: np.ndarray) -> np.ndarray:
        """
        计算前向运动学，返回完整 4x4 变换矩阵

        Args:
            joint_angles: 关节角度 [6] 或 [7]（rad）

        Returns:
            4x4 变换矩阵（位置单位：米）
        """
        q = joint_angles[:6] if len(joint_angles) > 6 else joint_angles

        T = np.eye(4)
        for i in range(6):
            theta_i = q[i] + self._theta[i]
            T_i = self._link_transformation(self._alpha[i], self._a[i], theta_i, self._d[i])
            T = T @ T_i

        # mm -> m
        T[:3, 3] /= 1000.0

        return T

    def compute_trajectory(self, joint_trajectory: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算整个轨迹的末端位置和姿态

        Args:
            joint_trajectory: 关节轨迹 [N, 6+]（rad）

        Returns:
            (positions, rotations)
            - positions: 末端位置轨迹 [N, 3]（米）
            - rotations: 末端姿态轨迹 [N, 3, 3]
        """
        N = len(joint_trajectory)
        positions = np.zeros((N, 3))
        rotations = np.zeros((N, 3, 3))

        for i, q in enumerate(joint_trajectory):
            positions[i], rotations[i] = self.forward_kinematics(q)

        return positions, rotations


# ============================================================
# 双臂运动学
# ============================================================

class DualArmKinematics:
    """
    双臂运动学

    处理双臂机器人的前向运动学计算，支持工具偏移和桌面距离
    """

    def __init__(self, config: Optional[KinematicsConfig] = None):
        """
        初始化双臂 FK

        Args:
            config: 运动学配置（包含工具偏移、桌面平面、fps）
        """
        self.fk_left = PiperForwardKinematics()
        self.fk_right = PiperForwardKinematics()
        self.config = config or KinematicsConfig()

    def compute_full_kinematics(
        self,
        state_trajectory: np.ndarray,
        arm: str = "left"
    ) -> KinematicsResult:
        """
        计算完整的运动学信息（推荐使用）

        包括：法兰位置/姿态、刷尖位置、桌面距离、各方向速度

        Args:
            state_trajectory: 状态轨迹 [N, 14]
                - [:, 0:7]: 左臂关节状态
                - [:, 7:14]: 右臂关节状态
            arm: 使用哪只手臂 ("left" | "right")

        Returns:
            KinematicsResult 包含所有运动学信息
        """
        if arm not in ["left", "right"]:
            raise ValueError(f"arm 必须是 'left' 或 'right'，不支持自动选择。当前值: {arm}")

        # 选择手臂
        if arm == "left":
            joint_traj = state_trajectory[:, :7]
            fk = self.fk_left
        else:
            joint_traj = state_trajectory[:, 7:14]
            fk = self.fk_right

        # 计算法兰位置和姿态
        flange_positions, flange_rotations = fk.compute_trajectory(joint_traj)

        # 计算刷尖位置
        tip_positions = compute_tip_positions_batch(
            flange_positions, flange_rotations, self.config.tool.tip_offset
        )

        # 计算刷尖到桌面距离
        tip_to_table = self.config.table.distance_to_plane_batch(tip_positions)

        # 计算速度（基于刷尖位置）
        v_xy, v_z, v_total = compute_velocities(tip_positions, self.config.fps)

        return KinematicsResult(
            flange_positions=flange_positions,
            flange_rotations=flange_rotations,
            tip_positions=tip_positions,
            tip_to_table_distance=tip_to_table,
            v_xy=v_xy,
            v_z=v_z,
            v_total=v_total
        )

    def compute_end_effector_trajectory(
        self,
        state_trajectory: np.ndarray,
        arm: str = "left"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算末端执行器轨迹和速度（兼容旧接口）

        注意：此函数返回的是刷尖位置，不是法兰位置

        Args:
            state_trajectory: 状态轨迹 [N, 14]
            arm: 使用哪只手臂 ("left" | "right")

        Returns:
            positions: 刷尖位置 [N, 3]（米）
            v_xy: 水平速度 [N]（米/秒）
        """
        result = self.compute_full_kinematics(state_trajectory, arm)
        return result.tip_positions, result.v_xy

    def get_table_distance(
        self,
        state_trajectory: np.ndarray,
        arm: str = "left"
    ) -> np.ndarray:
        """
        获取刷尖到桌面的距离（用于低位检测）

        Args:
            state_trajectory: 状态轨迹 [N, 14]
            arm: 使用哪只手臂

        Returns:
            距离数组 [N]（米），正值表示在桌面上方
        """
        result = self.compute_full_kinematics(state_trajectory, arm)
        return result.tip_to_table_distance


# ============================================================
# 从 ee_pose 计算运动学
# ============================================================

def compute_kinematics_from_ee_pose(
    ee_pose_trajectory: np.ndarray,
    arm: str = "left",
    config: Optional[KinematicsConfig] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从 ee_pose 数据计算刷尖位置和速度（兼容旧接口）

    Args:
        ee_pose_trajectory: 末端位姿轨迹 [N, 14]
            - [:, 0:3]: 左臂位置 (x, y, z)
            - [:, 3:7]: 左臂四元数 (qx, qy, qz, qw)
            - [:, 7:10]: 右臂位置
            - [:, 10:14]: 右臂四元数
        arm: 使用哪只手臂 ("left" | "right")
        config: 运动学配置

    Returns:
        (tip_positions, v_xy)
        - tip_positions: 刷尖位置 [N, 3]（米）
        - v_xy: 水平速度 [N]（米/秒）
    """
    result = compute_full_kinematics_from_ee_pose(ee_pose_trajectory, arm, config)
    return result.tip_positions, result.v_xy


def compute_full_kinematics_from_ee_pose(
    ee_pose_trajectory: np.ndarray,
    arm: str = "left",
    config: Optional[KinematicsConfig] = None
) -> KinematicsResult:
    """
    从 ee_pose 数据计算完整运动学信息

    Args:
        ee_pose_trajectory: 末端位姿轨迹 [N, 14]
            - [:, 0:3]: 左臂位置 (x, y, z)
            - [:, 3:7]: 左臂四元数 (qx, qy, qz, qw)
            - [:, 7:10]: 右臂位置
            - [:, 10:14]: 右臂四元数
        arm: 使用哪只手臂 ("left" | "right")
        config: 运动学配置

    Returns:
        KinematicsResult 包含所有运动学信息
    """
    if config is None:
        config = KinematicsConfig()

    if arm not in ["left", "right"]:
        raise ValueError(f"arm 必须是 'left' 或 'right'，不支持自动选择。当前值: {arm}")

    # 提取位置和四元数
    if arm == "left":
        flange_positions = ee_pose_trajectory[:, 0:3]
        quaternions = ee_pose_trajectory[:, 3:7]
    else:
        flange_positions = ee_pose_trajectory[:, 7:10]
        quaternions = ee_pose_trajectory[:, 10:14]

    # 四元数转旋转矩阵
    flange_rotations = quaternion_to_rotation_matrix_batch(quaternions)

    # 计算刷尖位置
    tip_positions = compute_tip_positions_batch(
        flange_positions, flange_rotations, config.tool.tip_offset
    )

    # 计算刷尖到桌面距离
    tip_to_table = config.table.distance_to_plane_batch(tip_positions)

    # 计算速度
    v_xy, v_z, v_total = compute_velocities(tip_positions, config.fps)

    return KinematicsResult(
        flange_positions=flange_positions,
        flange_rotations=flange_rotations,
        tip_positions=tip_positions,
        tip_to_table_distance=tip_to_table,
        v_xy=v_xy,
        v_z=v_z,
        v_total=v_total
    )


# ============================================================
# 便捷函数
# ============================================================

def create_default_kinematics_config(
    tip_offset_z: float = 0.30,
    table_z: float = -0.05,
    fps: float = 10.0
) -> KinematicsConfig:
    """
    创建默认运动学配置

    Args:
        tip_offset_z: 刷尖相对法兰的 Z 偏移（米），默认 0.30m
        table_z: 桌面 Z 坐标（相对基座），默认 -0.05m
        fps: 帧率，默认 10

    Returns:
        KinematicsConfig
    """
    return KinematicsConfig(
        tool=ToolConfig(
            tip_offset=np.array([0.0, 0.0, tip_offset_z]),
            name="brush"
        ),
        table=TablePlane(
            normal=np.array([0.0, 0.0, 1.0]),
            point=np.array([0.0, 0.0, table_z])
        ),
        fps=fps
    )