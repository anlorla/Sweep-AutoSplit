"""
基于 Action 的 Sweep 检测模块

不依赖 FK，直接使用 action 数据检测运动模式：
1. 计算 action 变化量（运动能量）
2. 用能量峰值检测 sweep 区间
3. 更简单、更鲁棒
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

from .config import SweepSegmentConfig, SweepKeypoint
from .signal_processing import smooth_signal, Region


@dataclass
class ActionBasedThresholds:
    """基于 action 的自适应阈值"""
    energy_threshold: float     # 能量阈值
    energy_percentiles: dict    # 能量百分位数


def compute_action_energy(action_trajectory: np.ndarray) -> np.ndarray:
    """
    计算 action 的运动能量

    energy(t) = ||action(t) - action(t-1)||_2

    Args:
        action_trajectory: action 轨迹 [N, action_dim]

    Returns:
        能量序列 [N]
    """
    N = len(action_trajectory)
    if N < 2:
        return np.zeros(N)

    # 计算相邻帧的差异
    diff = np.diff(action_trajectory, axis=0)
    energy = np.linalg.norm(diff, axis=1)

    # 补齐第一帧
    energy = np.concatenate([[0], energy])

    return energy


def compute_action_adaptive_thresholds(
    energy: np.ndarray,
    high_percentile: int = 60,
    low_percentile: int = 30
) -> ActionBasedThresholds:
    """
    计算自适应能量阈值

    Args:
        energy: 能量序列
        high_percentile: 高能量百分位数 (可以是任意值 1-99)
        low_percentile: 低能量百分位数

    Returns:
        ActionBasedThresholds
    """
    # 标准百分位数用于诊断输出
    percentiles = {
        10: np.percentile(energy, 10),
        30: np.percentile(energy, 30),
        50: np.percentile(energy, 50),
        60: np.percentile(energy, 60),
        70: np.percentile(energy, 70),
        90: np.percentile(energy, 90),
    }

    # 使用指定的百分位数作为阈值
    energy_threshold = np.percentile(energy, high_percentile)

    # 确保阈值不会太低
    energy_mean = np.mean(energy)
    energy_std = np.std(energy)
    min_threshold = energy_mean + 0.3 * energy_std  # 降低最小阈值，更敏感

    energy_threshold = max(energy_threshold, min_threshold, 1e-6)

    return ActionBasedThresholds(
        energy_threshold=energy_threshold,
        energy_percentiles=percentiles
    )


def detect_high_energy_regions(
    energy: np.ndarray,
    threshold: float,
    min_duration: int = 3,
    merge_gap: int = 5
) -> List[Region]:
    """
    检测高能量区间

    Args:
        energy: 能量序列
        threshold: 能量阈值
        min_duration: 最小持续帧数
        merge_gap: 合并相近区间的间隔

    Returns:
        高能量区间列表
    """
    regions = []
    in_high_region = False
    region_start = 0

    for i, e in enumerate(energy):
        if not in_high_region:
            if e > threshold:
                in_high_region = True
                region_start = i
        else:
            if e <= threshold:
                in_high_region = False
                if i - region_start >= min_duration:
                    regions.append(Region(start=region_start, end=i - 1))

    # 处理末尾
    if in_high_region:
        if len(energy) - region_start >= min_duration:
            regions.append(Region(start=region_start, end=len(energy) - 1))

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


class ActionBasedSweepDetector:
    """
    基于 Action 的 Sweep 检测器

    直接使用 action 数据检测运动模式，不依赖 FK
    """

    def __init__(self, config: SweepSegmentConfig):
        self.config = config

    def detect_keypoints(
        self,
        action_trajectory: np.ndarray,
        state_trajectory: Optional[np.ndarray] = None
    ) -> Tuple[List[SweepKeypoint], dict]:
        """
        检测 sweep 关键点

        Args:
            action_trajectory: action 轨迹 [N, action_dim]
            state_trajectory: state 轨迹（可选，用于补充信息）

        Returns:
            (关键点列表, 诊断信息)
        """
        config = self.config

        # Step 1: 计算运动能量
        energy = compute_action_energy(action_trajectory)

        # Step 2: 平滑能量信号
        energy_smooth = smooth_signal(energy, config.smoothing_window)

        # Step 3: 计算自适应阈值
        thresholds = compute_action_adaptive_thresholds(
            energy_smooth,
            high_percentile=config.energy_percentile
        )

        if config.verbose:
            print(f"[ActionDetector] Energy range: [{energy_smooth.min():.6f}, {energy_smooth.max():.6f}]")
            print(f"[ActionDetector] Energy threshold: {thresholds.energy_threshold:.6f}")
            print(f"[ActionDetector] Energy percentile: {config.energy_percentile}%")
            print(f"[ActionDetector] Merge gap: {config.merge_gap} frames")
            print(f"[ActionDetector] Energy percentiles: {thresholds.energy_percentiles}")

        # Step 4: 检测高能量区间
        high_energy_regions = detect_high_energy_regions(
            energy_smooth,
            threshold=thresholds.energy_threshold,
            min_duration=config.low_region_min_frames,
            merge_gap=config.merge_gap
        )

        if config.verbose:
            print(f"[ActionDetector] Found {len(high_energy_regions)} high-energy regions")

        # Step 5: 将高能量区间转换为 keypoints
        keypoints = []
        for sweep_idx, region in enumerate(high_energy_regions):
            P_t0 = region.start
            P_t1 = region.end
            L23 = region.length

            # 质量过滤：使用更宽松的范围
            # 原来的 L23_min/max 是针对 FK 检测设计的，这里需要调整
            min_length = max(3, config.L23_min // 3)  # 更宽松
            max_length = config.L23_max * 2  # 更宽松
            is_valid = min_length <= L23 <= max_length

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
                print(f"  [{status}] Sweep {sweep_idx}: P_t0={P_t0}, P_t1={P_t1}, L23={L23}")

        # 诊断信息
        diagnostics = {
            "energy": energy,
            "energy_smooth": energy_smooth,
            "thresholds": thresholds,
            "high_energy_regions": high_energy_regions,
        }

        return keypoints, diagnostics


def analyze_action_data(action_trajectory: np.ndarray, verbose: bool = True) -> dict:
    """
    分析 action 数据，帮助调参

    Args:
        action_trajectory: action 轨迹
        verbose: 是否打印

    Returns:
        分析结果
    """
    energy = compute_action_energy(action_trajectory)
    energy_smooth = smooth_signal(energy, window_size=7)
    thresholds = compute_action_adaptive_thresholds(energy_smooth)

    analysis = {
        "action_shape": action_trajectory.shape,
        "action_range": {
            "min": action_trajectory.min(axis=0).tolist(),
            "max": action_trajectory.max(axis=0).tolist(),
        },
        "energy": {
            "min": float(energy_smooth.min()),
            "max": float(energy_smooth.max()),
            "mean": float(energy_smooth.mean()),
            "std": float(energy_smooth.std()),
        },
        "suggested_threshold": thresholds.energy_threshold,
        "percentiles": thresholds.energy_percentiles,
    }

    if verbose:
        print("\n" + "=" * 50)
        print("ACTION DATA ANALYSIS")
        print("=" * 50)
        print(f"Shape: {analysis['action_shape']}")
        print(f"Energy range: [{analysis['energy']['min']:.6f}, {analysis['energy']['max']:.6f}]")
        print(f"Energy mean ± std: {analysis['energy']['mean']:.6f} ± {analysis['energy']['std']:.6f}")
        print(f"Suggested threshold: {analysis['suggested_threshold']:.6f}")
        print(f"Percentiles: {analysis['percentiles']}")
        print("=" * 50)

    return analysis
