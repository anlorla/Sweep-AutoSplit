"""
Sweep 检测模块

检测 sweep 动作的关键点 (P_t0, P_t1)

改进的实现（基于刷尖到桌面距离 + v_xy 峰值检测）：
1. 计算刷尖位置和到桌面距离 d(t)
2. 计算 v_xy（水平速度，单位 m/s）
3. 检测低位区（d(t) 小）+ 高速段（v_xy 大）
4. 使用 v_z 过滤误检（sweep 主要是水平运动）
5. 质量过滤: L_23 = P_t1 - P_t0 + 1 落在合理范围
"""

import numpy as np
from typing import List, Optional

from .config import SweepSegmentConfig, SweepKeypoint
from .signal_processing import smooth_signal, Region
from .kinematics import (
    DualArmKinematics,
    KinematicsConfig,
    KinematicsResult,
    compute_full_kinematics_from_ee_pose,
    create_default_kinematics_config,
)


def detect_vxy_high_regions(
    v_xy: np.ndarray,
    threshold: float,
    min_duration: int = 3,
    merge_gap: int = 2
) -> List[Region]:
    """
    检测 v_xy 高速区间（水平运动剧烈的区间）

    Args:
        v_xy: 水平速度信号 (m/s)
        threshold: 速度阈值 (m/s)
        min_duration: 最小持续帧数
        merge_gap: 合并相近区间的间隔

    Returns:
        高速区间列表
    """
    regions = []
    in_high_region = False
    region_start = 0

    for i, v in enumerate(v_xy):
        if not in_high_region:
            if v > threshold:
                in_high_region = True
                region_start = i
        else:
            if v <= threshold:
                in_high_region = False
                if i - region_start >= min_duration:
                    regions.append(Region(start=region_start, end=i - 1))

    # 处理末尾
    if in_high_region:
        if len(v_xy) - region_start >= min_duration:
            regions.append(Region(start=region_start, end=len(v_xy) - 1))

    # 合并相近的区间
    if len(regions) > 1:
        merged = [regions[0]]
        for region in regions[1:]:
            last = merged[-1]
            if region.start - last.end <= merge_gap:
                merged[-1] = Region(start=last.start, end=region.end)
            else:
                merged.append(region)
        regions = merged

    return regions


def detect_low_distance_regions(
    distance: np.ndarray,
    threshold_on: float,
    threshold_off: float,
    min_duration: int = 3
) -> List[Region]:
    """
    检测低距离区间（刷尖接近桌面的区间）

    使用滞回阈值避免抖动

    Args:
        distance: 刷尖到桌面距离 (m)，正值表示在桌面上方
        threshold_on: 进入低位区阈值（低于此值进入）
        threshold_off: 退出低位区阈值（高于此值退出）
        min_duration: 最小持续帧数

    Returns:
        低距离区间列表
    """
    regions = []
    in_low_region = False
    region_start = 0

    for i, d in enumerate(distance):
        if not in_low_region:
            if d < threshold_on:
                in_low_region = True
                region_start = i
        else:
            if d > threshold_off:
                in_low_region = False
                if i - region_start >= min_duration:
                    regions.append(Region(start=region_start, end=i - 1))

    # 处理末尾
    if in_low_region:
        if len(distance) - region_start >= min_duration:
            regions.append(Region(start=region_start, end=len(distance) - 1))

    return regions


class SweepDetector:
    """
    Sweep 关键点检测器

    使用刷尖到桌面距离 + v_xy 检测方法检测 sweep 动作的关键点

    改进点：
    1. 使用刷尖位置而非法兰位置
    2. 使用刷尖到桌面距离 d(t) 而非 z 坐标
    3. 速度单位统一为 m/s
    4. 基于低位区间检测 sweep（刷尖接近桌面时的水平运动）
    """

    def __init__(self, config: SweepSegmentConfig,
                 kinematics_config: Optional[KinematicsConfig] = None):
        """
        初始化检测器

        Args:
            config: 切分配置
            kinematics_config: 运动学配置（工具偏移、桌面平面等）
        """
        self.config = config
        self.kin_config = kinematics_config or create_default_kinematics_config()
        self.kinematics = DualArmKinematics(self.kin_config)

    def detect_keypoints(
        self,
        state_trajectory: np.ndarray,
        ee_pose_trajectory: Optional[np.ndarray] = None
    ) -> List[SweepKeypoint]:
        """
        检测 sweep 关键点

        改进的方法（基于低位区间）：
        1. 计算刷尖到桌面距离 d(t)
        2. 检测低位区间（d(t) < 阈值，刷尖接近桌面）
        3. 在每个低位区间内验证有水平运动 v_xy
        4. 每个有效的低位区间对应一个 sweep

        Args:
            state_trajectory: 关节状态轨迹 [N, 14]
            ee_pose_trajectory: 末端位姿轨迹 [N, 14]（可选，如果提供则跳过 FK）

        Returns:
            关键点列表 [SweepKeypoint, ...]
        """
        config = self.config

        # 如果是 both 模式，分别检测两只手臂并合并结果
        if config.active_arm == "both":
            return self._detect_keypoints_both_arms(state_trajectory, ee_pose_trajectory)

        # 单臂检测
        return self._detect_keypoints_single_arm(
            state_trajectory, ee_pose_trajectory, config.active_arm
        )

    def _detect_keypoints_both_arms(
        self,
        state_trajectory: np.ndarray,
        ee_pose_trajectory: Optional[np.ndarray] = None
    ) -> List[SweepKeypoint]:
        """
        双臂检测模式：分别检测左右臂，合并结果

        合并策略：按时间顺序合并所有 sweep，避免重叠
        """
        config = self.config

        # 分别检测左右臂
        left_keypoints = self._detect_keypoints_single_arm(
            state_trajectory, ee_pose_trajectory, "left"
        )
        right_keypoints = self._detect_keypoints_single_arm(
            state_trajectory, ee_pose_trajectory, "right"
        )

        if config.verbose:
            print(f"[SweepDetector] 左臂检测到 {len(left_keypoints)} 个 sweep")
            print(f"[SweepDetector] 右臂检测到 {len(right_keypoints)} 个 sweep")

        # 合并并按开始时间排序
        all_keypoints = []
        for kp in left_keypoints:
            all_keypoints.append((kp.P_t0, kp.P_t1, kp.is_valid, "left"))
        for kp in right_keypoints:
            all_keypoints.append((kp.P_t0, kp.P_t1, kp.is_valid, "right"))

        # 按开始时间排序
        all_keypoints.sort(key=lambda x: x[0])

        # 合并重叠区间（取并集）
        merged = []
        for P_t0, P_t1, is_valid, arm in all_keypoints:
            if not merged:
                merged.append([P_t0, P_t1, is_valid])
            else:
                last = merged[-1]
                # 如果有重叠，合并
                if P_t0 <= last[1] + 1:
                    last[1] = max(last[1], P_t1)
                    last[2] = last[2] or is_valid  # 任一有效则有效
                else:
                    merged.append([P_t0, P_t1, is_valid])

        # 转换为 SweepKeypoint
        keypoints = []
        for idx, (P_t0, P_t1, is_valid) in enumerate(merged):
            keypoints.append(SweepKeypoint(
                sweep_idx=idx,
                P_t0=P_t0,
                P_t1=P_t1,
                L23=P_t1 - P_t0 + 1,
                is_valid=is_valid
            ))

        if config.verbose:
            print(f"[SweepDetector] 合并后共 {len(keypoints)} 个 sweep")

        return keypoints

    def _detect_keypoints_single_arm(
        self,
        state_trajectory: np.ndarray,
        ee_pose_trajectory: Optional[np.ndarray],
        arm: str
    ) -> List[SweepKeypoint]:
        """
        单臂检测逻辑
        """
        config = self.config

        # Step 1: 计算完整运动学信息
        if ee_pose_trajectory is not None:
            kin_result = compute_full_kinematics_from_ee_pose(
                ee_pose_trajectory, arm=arm, config=self.kin_config
            )
        else:
            kin_result = self.kinematics.compute_full_kinematics(
                state_trajectory, arm=arm
            )

        # 提取关键信号
        tip_to_table = kin_result.tip_to_table_distance  # 刷尖到桌面距离
        v_xy = kin_result.v_xy  # 水平速度 (m/s)
        v_z = kin_result.v_z    # 垂向速度 (m/s)

        # Step 2: 平滑信号
        d_smooth = smooth_signal(tip_to_table, config.smoothing_window)
        v_xy_smooth = smooth_signal(v_xy, config.smoothing_window)
        v_z_smooth = smooth_signal(np.abs(v_z), config.smoothing_window)  # 取绝对值

        if config.verbose:
            print(f"[SweepDetector] 刷尖到桌面距离 d(t) 范围: [{d_smooth.min():.4f}, {d_smooth.max():.4f}] m")
            print(f"[SweepDetector] 水平速度 v_xy 范围: [{v_xy_smooth.min():.4f}, {v_xy_smooth.max():.4f}] m/s")
            print(f"[SweepDetector] 垂向速度 |v_z| 范围: [{v_z_smooth.min():.4f}, {v_z_smooth.max():.4f}] m/s")

        # Step 3: 计算低位阈值
        # 使用配置的 z_on 和 z_off（单位：米）
        d_threshold_on = config.z_on   # 进入低位区阈值（默认 0.01m = 1cm）
        d_threshold_off = config.z_off  # 退出低位区阈值（默认 0.06m = 6cm）

        if config.verbose:
            print(f"[SweepDetector] 低位阈值: d_on={d_threshold_on:.4f}m, d_off={d_threshold_off:.4f}m")

        # Step 4: 检测低位区间（刷尖接近桌面的区间）
        low_regions = detect_low_distance_regions(
            d_smooth,
            threshold_on=d_threshold_on,
            threshold_off=d_threshold_off,
            min_duration=config.low_region_min_frames
        )

        if config.verbose:
            print(f"[SweepDetector] 检测到 {len(low_regions)} 个低位区间")

        # Step 5: 计算 v_xy 阈值用于验证
        v_xy_percentile = 60  # 默认使用 60 百分位数
        v_xy_threshold = np.percentile(v_xy_smooth, v_xy_percentile)
        v_xy_mean = np.mean(v_xy_smooth)
        v_xy_std = np.std(v_xy_smooth)
        min_threshold = v_xy_mean + 0.3 * v_xy_std
        v_xy_threshold = max(v_xy_threshold, min_threshold, 1e-6)

        if config.verbose:
            print(f"[SweepDetector] v_xy 阈值 (自适应): {v_xy_threshold:.6f} m/s")

        # Step 6: 将低位区间转换为 keypoints，验证有水平运动
        keypoints = []
        for sweep_idx, region in enumerate(low_regions):
            P_t0 = region.start
            P_t1 = region.end
            L23 = region.length

            # 区间内的信号
            v_xy_in_region = v_xy_smooth[P_t0:P_t1+1]
            v_z_in_region = v_z_smooth[P_t0:P_t1+1]
            d_in_region = d_smooth[P_t0:P_t1+1]


            # 验证1：长度在合理范围（放宽限制）
            min_length = 3  # 最少 3 帧
            max_length = 200  # 最多 200 帧（放宽）
            length_valid = min_length <= L23 <= max_length

            # 验证2：区间内有明显的水平运动
            v_xy_max = np.max(v_xy_in_region)
            v_xy_mean_region = np.mean(v_xy_in_region)
            has_horizontal_motion = v_xy_max > v_xy_threshold or v_xy_mean_region > v_xy_threshold * 0.5

            # 验证3：确实在低位（d(t) 的最小值足够低）
            d_min = np.min(d_in_region)
            truly_low = d_min < d_threshold_off  # 至少曾经接近桌面

            # 综合有效性（放宽了 v_z 比例要求）
            is_valid = length_valid and has_horizontal_motion and truly_low

            if config.verbose:
                status = "✓" if is_valid else "✗"
                reasons = []
                if not length_valid:
                    reasons.append(f"长度={L23}不在[{min_length},{max_length}]")
                if not has_horizontal_motion:
                    reasons.append(f"v_xy_max={v_xy_max:.4f}<{v_xy_threshold:.4f}")
                if not truly_low:
                    reasons.append(f"d_min={d_min:.4f}m>{d_threshold_off:.4f}m")
                reason_str = f" ({', '.join(reasons)})" if reasons else ""
                print(f"  [{status}] Sweep {sweep_idx}: P_t0={P_t0}, P_t1={P_t1}, L23={L23}, d_min={d_min:.4f}m{reason_str}")

            keypoint = SweepKeypoint(
                sweep_idx=sweep_idx,
                P_t0=P_t0,
                P_t1=P_t1,
                L23=L23,
                is_valid=is_valid
            )
            keypoints.append(keypoint)

        return keypoints

    def detect_keypoints_from_kinematics(
        self,
        kin_result: KinematicsResult
    ) -> List[SweepKeypoint]:
        """
        直接从运动学结果检测关键点

        用于已经计算好运动学的情况

        Args:
            kin_result: 运动学计算结果

        Returns:
            关键点列表
        """
        config = self.config

        # 提取信号
        d = kin_result.tip_to_table_distance
        v_xy = kin_result.v_xy
        v_z = kin_result.v_z

        # 平滑信号
        d_smooth = smooth_signal(d, config.smoothing_window)
        v_xy_smooth = smooth_signal(v_xy, config.smoothing_window)
        v_z_smooth = smooth_signal(np.abs(v_z), config.smoothing_window)

        # 使用配置的低位阈值
        d_threshold_on = config.z_on
        d_threshold_off = config.z_off

        # 检测低位区间
        low_regions = detect_low_distance_regions(
            d_smooth,
            threshold_on=d_threshold_on,
            threshold_off=d_threshold_off,
            min_duration=config.low_region_min_frames
        )

        if config.verbose:
            print(f"[SweepDetector] 检测到 {len(low_regions)} 个低位区间")

        # 计算 v_xy 阈值
        v_xy_percentile = 60  # 固定使用 60 百分位数
        v_xy_threshold = np.percentile(v_xy_smooth, v_xy_percentile)
        v_xy_mean = np.mean(v_xy_smooth)
        v_xy_std = np.std(v_xy_smooth)
        min_threshold = v_xy_mean + 0.3 * v_xy_std
        v_xy_threshold = max(v_xy_threshold, min_threshold, 1e-6)

        # 转换为 keypoints
        keypoints = []

        for sweep_idx, region in enumerate(low_regions):
            P_t0 = region.start
            P_t1 = region.end
            L23 = region.length

            # 区间内的信号
            v_xy_in_region = v_xy_smooth[P_t0:P_t1+1]
            d_in_region = d_smooth[P_t0:P_t1+1]

            # 验证条件（与 detect_keypoints 一致）
            min_length = 3
            max_length = 200
            length_valid = min_length <= L23 <= max_length

            v_xy_max = np.max(v_xy_in_region)
            v_xy_mean_region = np.mean(v_xy_in_region)
            has_horizontal_motion = v_xy_max > v_xy_threshold or v_xy_mean_region > v_xy_threshold * 0.5

            d_min = np.min(d_in_region)
            truly_low = d_min < d_threshold_off

            is_valid = length_valid and has_horizontal_motion and truly_low

            keypoint = SweepKeypoint(
                sweep_idx=sweep_idx,
                P_t0=P_t0,
                P_t1=P_t1,
                L23=L23,
                is_valid=is_valid
            )
            keypoints.append(keypoint)

            if config.verbose:
                status = "✓" if is_valid else "✗"
                print(f"  [{status}] Sweep {sweep_idx}: P_t0={P_t0}, P_t1={P_t1}, L23={L23}, d_min={d_min:.4f}m")

        return keypoints

    def get_diagnostic_data(
        self,
        state_trajectory: np.ndarray,
        ee_pose_trajectory: Optional[np.ndarray] = None
    ) -> dict:
        """
        获取诊断数据（用于可视化和调试）

        Returns:
            包含完整运动学信息和检测结果的字典
        """
        config = self.config

        # 对于 "both" 模式，诊断数据默认使用左臂
        arm = config.active_arm if config.active_arm != "both" else "left"

        # 计算运动学
        if ee_pose_trajectory is not None:
            kin_result = compute_full_kinematics_from_ee_pose(
                ee_pose_trajectory, arm=arm, config=self.kin_config
            )
        else:
            kin_result = self.kinematics.compute_full_kinematics(
                state_trajectory, arm=arm
            )

        # 平滑信号
        d_smooth = smooth_signal(kin_result.tip_to_table_distance, config.smoothing_window)
        v_xy_smooth = smooth_signal(kin_result.v_xy, config.smoothing_window)
        v_z_smooth = smooth_signal(np.abs(kin_result.v_z), config.smoothing_window)

        # 使用配置的低位阈值
        d_threshold_on = config.z_on
        d_threshold_off = config.z_off

        # 检测低位区间
        low_regions = detect_low_distance_regions(
            d_smooth,
            threshold_on=d_threshold_on,
            threshold_off=d_threshold_off,
            min_duration=config.low_region_min_frames
        )

        # 计算 v_xy 阈值
        v_xy_percentile = 60  # 固定使用 60 百分位数
        v_xy_threshold = np.percentile(v_xy_smooth, v_xy_percentile)
        v_xy_mean = np.mean(v_xy_smooth)
        v_xy_std = np.std(v_xy_smooth)
        min_threshold = v_xy_mean + 0.3 * v_xy_std
        v_xy_threshold = max(v_xy_threshold, min_threshold, 1e-6)

        keypoints = self.detect_keypoints_from_kinematics(kin_result)

        return {
            # 原始运动学数据
            "flange_positions": kin_result.flange_positions,
            "flange_rotations": kin_result.flange_rotations,
            "tip_positions": kin_result.tip_positions,
            "tip_to_table_distance": kin_result.tip_to_table_distance,
            "v_xy": kin_result.v_xy,
            "v_z": kin_result.v_z,
            "v_total": kin_result.v_total,

            # 平滑后的信号
            "d_smooth": d_smooth,
            "v_xy_smooth": v_xy_smooth,
            "v_z_smooth": v_z_smooth,

            # 检测结果
            "v_xy_threshold": v_xy_threshold,
            "d_threshold_on": d_threshold_on,
            "d_threshold_off": d_threshold_off,
            "low_regions": low_regions,
            "keypoints": keypoints,

            # 配置信息
            "config": {
                "v_xy_percentile": v_xy_percentile,
                "v_xy_threshold": v_xy_threshold,
                "d_threshold_on": d_threshold_on,
                "d_threshold_off": d_threshold_off,
                "tool_tip_offset": self.kin_config.tool.tip_offset.tolist(),
                "table_z": self.kin_config.table.point[2],
                "fps": self.kin_config.fps,
            }
        }
