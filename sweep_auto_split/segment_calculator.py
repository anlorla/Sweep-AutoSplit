"""
Segment 边界计算模块

实现文档 Section 3-4 的核心公式：

约束：
1. s <= P_t0 - A_min          (窗口左侧有足够 Approach)
2. s >= P_t1 - H + 1 + R_min  (窗口右侧有足够 Retreat)
3. s >= P_{t-1,1} + 1         (不混入上一个 sweep)
4. s <= P_{t+1,0} - H         (不混入下一个 sweep)

合格区间：
    s_min = max(P_{t-1,1}+1, P_t1-H+1+R_min)
    s_max = min(P_t0-A_min, P_{t+1,0}-H)

Segment 边界：
    T_t0 = s_min
    T_t1 = s_min + H - 1  (使用 s_min 作为起点)

    注意：多样性 diversity = s_max - s_min + 1 表示训练时可用的起点数量
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

from .config import SweepSegmentConfig, SweepKeypoint, SegmentBoundary


class SegmentCalculator:
    """
    Segment 边界计算器

    根据 sweep 关键点计算每个 segment 的边界，
    以最大化训练样本多样性
    """

    def __init__(self, config: SweepSegmentConfig):
        """
        初始化计算器

        Args:
            config: 切分配置
        """
        self.config = config

    def calculate_boundaries(
        self,
        keypoints: List[SweepKeypoint],
        episode_length: int
    ) -> List[SegmentBoundary]:
        """
        计算 segment 边界

        实现文档 Section 4 的公式

        Args:
            keypoints: sweep 关键点列表
            episode_length: episode 总帧数

        Returns:
            segment 边界列表
        """
        config = self.config
        H = config.H
        A_min = config.A_min
        R_min = config.R_min

        # 只处理有效的关键点
        valid_keypoints = [kp for kp in keypoints if kp.is_valid]

        if not valid_keypoints:
            if config.verbose:
                print("[SegmentCalculator] No valid keypoints found")
            return []

        boundaries = []

        for t, kp in enumerate(valid_keypoints):
            # 获取相邻 sweep 的关键点
            # P_{t-1,1}: 上一个 sweep 的 Stroke 结束帧
            P_prev_1 = valid_keypoints[t - 1].P_t1 if t > 0 else -1

            # P_{t+1,0}: 下一个 sweep 的 Engage 开始帧
            P_next_0 = valid_keypoints[t + 1].P_t0 if t < len(valid_keypoints) - 1 else episode_length + H

            # 当前 sweep 的关键点
            P_t0 = kp.P_t0  # Engage 开始
            P_t1 = kp.P_t1  # Stroke 结束

            # ============================================================
            # 计算 s_min（下界）
            # ============================================================
            # 约束 3: s >= P_{t-1,1} + 1（不混入上一个 sweep）
            constraint_3 = P_prev_1 + 1

            # 约束 2: s >= P_t1 - H + 1 + R_min（窗口右侧有足够 Retreat）
            constraint_2 = P_t1 - H + 1 + R_min

            s_min = max(constraint_3, constraint_2)

            # ============================================================
            # 计算 s_max（上界）
            # ============================================================
            # 约束 1: s <= P_t0 - A_min（窗口左侧有足够 Approach）
            constraint_1 = P_t0 - A_min

            # 约束 4: s <= P_{t+1,0} - H（不混入下一个 sweep）
            constraint_4 = P_next_0 - H

            s_max = min(constraint_1, constraint_4)

            # ============================================================
            # 检查有效性
            # ============================================================
            is_valid = s_min <= s_max

            # ============================================================
            # 计算 segment 边界
            # ============================================================
            # 使用 s_min 作为起点
            # T_t0 = s_min
            # T_t1 = s_min + H - 1 (固定长度 H)
            T_t0 = s_min
            T_t1 = s_min + H - 1

            # 多样性
            diversity = s_max - s_min + 1 if is_valid else 0

            boundary = SegmentBoundary(
                sweep_idx=kp.sweep_idx,
                s_min=s_min,
                s_max=s_max,
                T_t0=T_t0,
                T_t1=T_t1,
                diversity=diversity,
                is_valid=is_valid
            )
            boundaries.append(boundary)

            if config.verbose:
                status = "✓" if is_valid else "✗"
                print(f"  [{status}] Sweep {kp.sweep_idx}:")
                print(f"      P_t0={P_t0}, P_t1={P_t1}")
                print(f"      s_min={s_min} (constraint_3={constraint_3}, constraint_2={constraint_2})")
                print(f"      s_max={s_max} (constraint_1={constraint_1}, constraint_4={constraint_4})")
                print(f"      T=[{T_t0}, {T_t1}], diversity={diversity}")

        return boundaries

    def get_valid_boundaries(
        self,
        keypoints: List[SweepKeypoint],
        episode_length: int
    ) -> List[SegmentBoundary]:
        """
        获取有效的 segment 边界

        Args:
            keypoints: sweep 关键点列表
            episode_length: episode 总帧数

        Returns:
            有效的 segment 边界列表
        """
        boundaries = self.calculate_boundaries(keypoints, episode_length)
        return [b for b in boundaries if b.is_valid]

    def calculate_total_diversity(self, boundaries: List[SegmentBoundary]) -> int:
        """
        计算总多样性（所有有效 segment 的多样性之和）

        Args:
            boundaries: segment 边界列表

        Returns:
            总多样性
        """
        return sum(b.diversity for b in boundaries if b.is_valid)

    def validate_no_overlap(self, boundaries: List[SegmentBoundary]) -> bool:
        """
        验证 segment 之间没有重叠

        Args:
            boundaries: segment 边界列表

        Returns:
            是否无重叠
        """
        valid_boundaries = [b for b in boundaries if b.is_valid]
        if len(valid_boundaries) < 2:
            return True

        # 按起点排序
        sorted_boundaries = sorted(valid_boundaries, key=lambda b: b.T_t0)

        for i in range(len(sorted_boundaries) - 1):
            current = sorted_boundaries[i]
            next_seg = sorted_boundaries[i + 1]
            if current.T_t1 >= next_seg.T_t0:
                return False

        return True

    def get_statistics(
        self,
        keypoints: List[SweepKeypoint],
        boundaries: List[SegmentBoundary]
    ) -> dict:
        """
        获取统计信息

        Args:
            keypoints: 关键点列表
            boundaries: 边界列表

        Returns:
            统计信息字典
        """
        valid_keypoints = [kp for kp in keypoints if kp.is_valid]
        valid_boundaries = [b for b in boundaries if b.is_valid]

        stats = {
            "total_keypoints": len(keypoints),
            "valid_keypoints": len(valid_keypoints),
            "invalid_keypoints": len(keypoints) - len(valid_keypoints),
            "total_boundaries": len(boundaries),
            "valid_boundaries": len(valid_boundaries),
            "invalid_boundaries": len(boundaries) - len(valid_boundaries),
            "total_diversity": self.calculate_total_diversity(boundaries),
            "avg_diversity": (
                sum(b.diversity for b in valid_boundaries) / len(valid_boundaries)
                if valid_boundaries else 0
            ),
            "avg_segment_length": (
                sum(b.segment_length for b in valid_boundaries) / len(valid_boundaries)
                if valid_boundaries else 0
            ),
            "no_overlap": self.validate_no_overlap(boundaries),
        }

        return stats


def print_segment_summary(
    keypoints: List[SweepKeypoint],
    boundaries: List[SegmentBoundary],
    config: SweepSegmentConfig
):
    """
    打印切分摘要

    Args:
        keypoints: 关键点列表
        boundaries: 边界列表
        config: 配置
    """
    calculator = SegmentCalculator(config)
    stats = calculator.get_statistics(keypoints, boundaries)

    print("\n" + "=" * 60)
    print("SWEEP AUTO SPLIT SUMMARY")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  H (action horizon) = {config.H}")
    print(f"  A_min = {config.A_min}, R_min = {config.R_min}")
    print(f"  L23 range = [{config.L23_min}, {config.L23_max}]")
    print()
    print(f"Results:")
    print(f"  Keypoints: {stats['valid_keypoints']}/{stats['total_keypoints']} valid")
    print(f"  Segments:  {stats['valid_boundaries']}/{stats['total_boundaries']} valid")
    print(f"  Total diversity (|S|): {stats['total_diversity']}")
    print(f"  Avg diversity per segment: {stats['avg_diversity']:.1f}")
    print(f"  Avg segment length: {stats['avg_segment_length']:.1f} frames")
    print(f"  No overlap: {stats['no_overlap']}")
    print("=" * 60)

    # 详细列表
    valid_boundaries = [b for b in boundaries if b.is_valid]
    if valid_boundaries:
        print("\nValid Segments:")
        print("-" * 60)
        for b in valid_boundaries:
            kp = keypoints[b.sweep_idx]
            print(f"  Sweep {b.sweep_idx}: T=[{b.T_t0:4d}, {b.T_t1:4d}] "
                  f"(len={b.segment_length:3d}), "
                  f"diversity={b.diversity:3d}, "
                  f"P=[{kp.P_t0:4d}, {kp.P_t1:4d}]")
