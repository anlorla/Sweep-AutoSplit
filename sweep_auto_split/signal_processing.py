"""
信号处理模块

用于平滑信号、检测低位区间和高速段

包含：
- 信号平滑（滑动均值）
- 滞回阈值检测
- 自适应阈值计算（百分位数方法）
- 高速段检测
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class Region:
    """区间数据结构"""
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start + 1


def smooth_signal(signal: np.ndarray, window_size: int = 7) -> np.ndarray:
    """
    滑动均值平滑

    Args:
        signal: 输入信号 [N]
        window_size: 平滑窗口大小（建议 5-9）

    Returns:
        平滑后的信号 [N]
    """
    if window_size < 1:
        return signal.copy()

    # 使用卷积实现滑动均值
    kernel = np.ones(window_size) / window_size
    # 边界用 'same' 模式，保持长度不变
    smoothed = np.convolve(signal, kernel, mode='same')

    # 处理边界效应：边界处使用有效数据的均值
    half_win = window_size // 2
    for i in range(half_win):
        # 左边界
        smoothed[i] = np.mean(signal[:i + half_win + 1])
        # 右边界
        smoothed[-(i + 1)] = np.mean(signal[-(i + half_win + 1):])

    return smoothed


def detect_low_regions_with_hysteresis(
    z: np.ndarray,
    z_on: float,
    z_off: float,
    min_duration: int = 3
) -> List[Region]:
    """
    使用滞回阈值检测低位区间

    低位区定义：
    - 当 z < z_on 时进入低位区
    - 当 z > z_off 时退出低位区
    - 要求 z_on < z_off（滞回）

    Args:
        z: z 坐标序列 [N]
        z_on: 进入低位区的阈值
        z_off: 退出低位区的阈值
        min_duration: 最小持续帧数

    Returns:
        低位区间列表 [(start, end), ...]
    """
    assert z_on < z_off, "滞回阈值要求 z_on < z_off"

    regions = []
    in_low_region = False
    region_start = 0

    for i, z_val in enumerate(z):
        if not in_low_region:
            # 检查是否进入低位区
            if z_val < z_on:
                in_low_region = True
                region_start = i
        else:
            # 检查是否退出低位区
            if z_val > z_off:
                in_low_region = False
                # 检查持续时间是否足够
                if i - region_start >= min_duration:
                    regions.append(Region(start=region_start, end=i - 1))

    # 处理末尾仍在低位区的情况
    if in_low_region:
        if len(z) - region_start >= min_duration:
            regions.append(Region(start=region_start, end=len(z) - 1))

    return regions


def detect_threshold_crossing(
    signal: np.ndarray,
    threshold: float,
    min_duration: int = 1
) -> List[Region]:
    """
    检测信号超过阈值的区间

    Args:
        signal: 输入信号 [N]
        threshold: 阈值
        min_duration: 最小持续帧数

    Returns:
        超阈值区间列表
    """
    regions = []
    above_threshold = False
    region_start = 0

    for i, val in enumerate(signal):
        if not above_threshold:
            if val > threshold:
                above_threshold = True
                region_start = i
        else:
            if val <= threshold:
                above_threshold = False
                if i - region_start >= min_duration:
                    regions.append(Region(start=region_start, end=i - 1))

    # 处理末尾仍超阈值的情况
    if above_threshold:
        if len(signal) - region_start >= min_duration:
            regions.append(Region(start=region_start, end=len(signal) - 1))

    return regions


def find_longest_high_speed_segment(
    v_xy: np.ndarray,
    threshold: float,
    region_start: int = 0,
    region_end: Optional[int] = None
) -> Tuple[int, int]:
    """
    在指定区间内找到最长的高速连续段

    这是文档中 Step 2 的核心：
    "在低位区间内，用 v_xy(t) 超阈值的最长连续段作为 stroke 主体"

    Args:
        v_xy: 水平速度信号 [N]
        threshold: 速度阈值
        region_start: 搜索区间起点
        region_end: 搜索区间终点（默认到末尾）

    Returns:
        (segment_start, segment_end): 最长高速段的起止帧（相对于 v_xy 的绝对索引）
    """
    if region_end is None:
        region_end = len(v_xy) - 1

    # 在指定区间内检测所有高速段
    sub_signal = v_xy[region_start:region_end + 1]
    high_speed_regions = detect_threshold_crossing(sub_signal, threshold, min_duration=1)

    if not high_speed_regions:
        # 没有高速段，返回区间中点
        mid = (region_start + region_end) // 2
        return mid, mid

    # 找最长的段
    longest_region = max(high_speed_regions, key=lambda r: r.length)

    # 转换为绝对索引
    abs_start = region_start + longest_region.start
    abs_end = region_start + longest_region.end

    return abs_start, abs_end


def merge_close_regions(regions: List[Region], gap_threshold: int = 3) -> List[Region]:
    """
    合并相近的区间

    如果两个区间之间的间隔小于 gap_threshold，则合并它们

    Args:
        regions: 区间列表
        gap_threshold: 最大间隔

    Returns:
        合并后的区间列表
    """
    if not regions:
        return []

    # 按起点排序
    sorted_regions = sorted(regions, key=lambda r: r.start)

    merged = [sorted_regions[0]]
    for region in sorted_regions[1:]:
        last = merged[-1]
        # 检查是否需要合并
        if region.start - last.end <= gap_threshold:
            # 合并：扩展上一个区间的终点
            merged[-1] = Region(start=last.start, end=max(last.end, region.end))
        else:
            merged.append(region)

    return merged


def compute_signal_energy(signal: np.ndarray, window_size: int = 10) -> np.ndarray:
    """
    计算信号的局部能量（用于运动检测）

    Args:
        signal: 输入信号 [N] 或 [N, D]
        window_size: 能量计算窗口

    Returns:
        能量序列 [N]
    """
    # 如果是多维信号，先计算每帧的模
    if signal.ndim == 2:
        signal = np.linalg.norm(signal, axis=1)

    # 计算滑动窗口内的能量（平方和）
    energy = np.zeros_like(signal)
    half_win = window_size // 2

    for i in range(len(signal)):
        start = max(0, i - half_win)
        end = min(len(signal), i + half_win + 1)
        energy[i] = np.sum(signal[start:end] ** 2)

    return energy


# ============================================================
# 自适应阈值计算（按 guidance.md 3.1 节）
# ============================================================

@dataclass
class AdaptiveThresholds:
    """自适应阈值结果"""
    z_on: float           # 低位区进入阈值
    z_off: float          # 低位区退出阈值
    v_xy_threshold: float # 水平速度阈值

    # 诊断信息
    z_percentiles: Dict[int, float] = None
    v_xy_percentiles: Dict[int, float] = None

    def __repr__(self) -> str:
        return (
            f"AdaptiveThresholds(\n"
            f"  z_on={self.z_on:.4f}, z_off={self.z_off:.4f},\n"
            f"  v_xy_threshold={self.v_xy_threshold:.4f}\n"
            f")"
        )


def compute_adaptive_thresholds(
    z: np.ndarray,
    v_xy: np.ndarray,
    z_percentile_low: int = 20,
    z_percentile_high: int = 30,
    v_xy_percentile: int = 70,
    min_z_delta: float = 0.005,
    min_v_xy_threshold: float = 0.005
) -> AdaptiveThresholds:
    """
    计算自适应阈值（百分位数方法）

    按 guidance.md 3.1 节：
    - z_on = percentile(z, 20%) + delta
    - z_off = z_on + delta
    - v_xy_threshold = percentile(v_xy, 70%)

    其中 delta = percentile(z, 30%) - percentile(z, 20%)

    Args:
        z: z 坐标序列 [N]
        v_xy: 水平速度序列 [N]
        z_percentile_low: z 低百分位数（默认 20）
        z_percentile_high: z 高百分位数（默认 30）
        v_xy_percentile: v_xy 百分位数（默认 70）
        min_z_delta: 最小 z 变化量（避免阈值过近）
        min_v_xy_threshold: 最小速度阈值

    Returns:
        AdaptiveThresholds 对象
    """
    # 计算 Z 阈值
    z_p_low = np.percentile(z, z_percentile_low)
    z_p_high = np.percentile(z, z_percentile_high)
    delta = max(z_p_high - z_p_low, min_z_delta)

    z_on = z_p_low + delta
    z_off = z_on + delta

    # 计算 V_xy 阈值
    v_xy_threshold = max(
        np.percentile(v_xy, v_xy_percentile),
        min_v_xy_threshold
    )

    # 收集诊断信息
    z_percentiles = {
        10: np.percentile(z, 10),
        20: np.percentile(z, 20),
        30: np.percentile(z, 30),
        50: np.percentile(z, 50),
        70: np.percentile(z, 70),
        90: np.percentile(z, 90),
    }

    v_xy_percentiles = {
        10: np.percentile(v_xy, 10),
        30: np.percentile(v_xy, 30),
        50: np.percentile(v_xy, 50),
        70: np.percentile(v_xy, 70),
        90: np.percentile(v_xy, 90),
    }

    return AdaptiveThresholds(
        z_on=z_on,
        z_off=z_off,
        v_xy_threshold=v_xy_threshold,
        z_percentiles=z_percentiles,
        v_xy_percentiles=v_xy_percentiles,
    )


def compute_adaptive_thresholds_robust(
    z: np.ndarray,
    v_xy: np.ndarray,
    expected_sweeps: int = 5,
    sweep_z_drop: float = 0.03,
    sweep_v_xy_ratio: float = 2.0
) -> AdaptiveThresholds:
    """
    鲁棒的自适应阈值计算

    使用更复杂的启发式方法，适用于信号变化范围不确定的情况

    Args:
        z: z 坐标序列 [N]
        v_xy: 水平速度序列 [N]
        expected_sweeps: 预期的 sweep 数量（用于校准）
        sweep_z_drop: 预期的 sweep 时 z 下降量
        sweep_v_xy_ratio: 预期的 sweep 时速度相对于基线的比率

    Returns:
        AdaptiveThresholds 对象
    """
    # 方法 1: 基于信号分布的阈值
    z_median = np.median(z)
    z_std = np.std(z)
    z_on_v1 = z_median - 0.5 * z_std
    z_off_v1 = z_median - 0.3 * z_std

    # 方法 2: 基于百分位数的阈值
    thresholds_v2 = compute_adaptive_thresholds(z, v_xy)

    # 方法 3: 基于局部极值的阈值
    z_local_min = find_local_minima(z, window=20)
    z_local_max = find_local_maxima(z, window=20)

    if len(z_local_min) > 0 and len(z_local_max) > 0:
        z_low_values = z[z_local_min]
        z_high_values = z[z_local_max]
        z_on_v3 = np.percentile(z_low_values, 80)  # 低位的高端
        z_off_v3 = np.percentile(z_high_values, 20)  # 高位的低端
    else:
        z_on_v3 = z_on_v1
        z_off_v3 = z_off_v1

    # 融合多种方法（加权平均）
    z_on = 0.3 * z_on_v1 + 0.4 * thresholds_v2.z_on + 0.3 * z_on_v3
    z_off = 0.3 * z_off_v1 + 0.4 * thresholds_v2.z_off + 0.3 * z_off_v3

    # 确保滞回性质
    if z_on >= z_off:
        delta = (z_off - z_on) / 2 + 0.005
        z_on = z_on - delta
        z_off = z_off + delta

    # V_xy 阈值：使用分布方法
    v_xy_mean = np.mean(v_xy)
    v_xy_std = np.std(v_xy)
    v_xy_threshold = max(
        v_xy_mean + 0.5 * v_xy_std,
        thresholds_v2.v_xy_threshold,
        0.005
    )

    return AdaptiveThresholds(
        z_on=z_on,
        z_off=z_off,
        v_xy_threshold=v_xy_threshold,
        z_percentiles=thresholds_v2.z_percentiles,
        v_xy_percentiles=thresholds_v2.v_xy_percentiles,
    )


def find_local_minima(signal: np.ndarray, window: int = 10) -> np.ndarray:
    """
    查找局部最小值点

    Args:
        signal: 输入信号 [N]
        window: 搜索窗口大小

    Returns:
        局部最小值的索引数组
    """
    minima = []
    half_win = window // 2

    for i in range(half_win, len(signal) - half_win):
        local_region = signal[i - half_win:i + half_win + 1]
        if signal[i] == np.min(local_region):
            minima.append(i)

    return np.array(minima)


def find_local_maxima(signal: np.ndarray, window: int = 10) -> np.ndarray:
    """
    查找局部最大值点

    Args:
        signal: 输入信号 [N]
        window: 搜索窗口大小

    Returns:
        局部最大值的索引数组
    """
    maxima = []
    half_win = window // 2

    for i in range(half_win, len(signal) - half_win):
        local_region = signal[i - half_win:i + half_win + 1]
        if signal[i] == np.max(local_region):
            maxima.append(i)

    return np.array(maxima)


def analyze_signal_distribution(
    z: np.ndarray,
    v_xy: np.ndarray,
    verbose: bool = True
) -> Dict:
    """
    分析信号分布，用于调试和参数调优

    Args:
        z: z 坐标序列
        v_xy: 水平速度序列
        verbose: 是否打印详细信息

    Returns:
        分析结果字典
    """
    analysis = {
        "z": {
            "min": float(np.min(z)),
            "max": float(np.max(z)),
            "mean": float(np.mean(z)),
            "std": float(np.std(z)),
            "range": float(np.max(z) - np.min(z)),
            "percentiles": {
                p: float(np.percentile(z, p))
                for p in [10, 20, 30, 50, 70, 80, 90]
            }
        },
        "v_xy": {
            "min": float(np.min(v_xy)),
            "max": float(np.max(v_xy)),
            "mean": float(np.mean(v_xy)),
            "std": float(np.std(v_xy)),
            "percentiles": {
                p: float(np.percentile(v_xy, p))
                for p in [10, 30, 50, 70, 90]
            }
        }
    }

    # 计算建议阈值
    thresholds = compute_adaptive_thresholds(z, v_xy)
    analysis["suggested_thresholds"] = {
        "z_on": thresholds.z_on,
        "z_off": thresholds.z_off,
        "v_xy_threshold": thresholds.v_xy_threshold,
    }

    if verbose:
        print("\n" + "=" * 50)
        print("SIGNAL DISTRIBUTION ANALYSIS")
        print("=" * 50)
        print(f"\nZ coordinate:")
        print(f"  Range: [{analysis['z']['min']:.4f}, {analysis['z']['max']:.4f}] m")
        print(f"  Mean ± Std: {analysis['z']['mean']:.4f} ± {analysis['z']['std']:.4f} m")
        print(f"  Percentiles: {analysis['z']['percentiles']}")

        print(f"\nV_xy (horizontal velocity):")
        print(f"  Range: [{analysis['v_xy']['min']:.4f}, {analysis['v_xy']['max']:.4f}] m/frame")
        print(f"  Mean ± Std: {analysis['v_xy']['mean']:.4f} ± {analysis['v_xy']['std']:.4f}")
        print(f"  Percentiles: {analysis['v_xy']['percentiles']}")

        print(f"\nSuggested thresholds:")
        print(f"  z_on = {thresholds.z_on:.4f} m")
        print(f"  z_off = {thresholds.z_off:.4f} m")
        print(f"  v_xy_threshold = {thresholds.v_xy_threshold:.4f} m/frame")
        print("=" * 50)

    return analysis
