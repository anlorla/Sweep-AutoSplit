"""
可视化模块

用于调试和验证 sweep 检测结果
"""

import numpy as np
from typing import List, Optional
from pathlib import Path

from .config import SweepSegmentConfig, SweepKeypoint, SegmentBoundary
from .signal_processing import Region


def plot_sweep_detection(
    z: np.ndarray,
    v_xy: np.ndarray,
    keypoints: List[SweepKeypoint],
    boundaries: List[SegmentBoundary],
    config: SweepSegmentConfig,
    low_regions: Optional[List[Region]] = None,
    z_smooth: Optional[np.ndarray] = None,
    v_xy_smooth: Optional[np.ndarray] = None,
    save_path: Optional[Path] = None,
    title: str = "Sweep Detection"
):
    """
    绘制 sweep 检测结果

    Args:
        z: z 坐标序列
        v_xy: 水平速度序列
        keypoints: 检测到的关键点
        boundaries: 计算的边界
        config: 配置
        low_regions: 低位区间（可选）
        z_smooth: 平滑后的 z（可选）
        v_xy_smooth: 平滑后的 v_xy（可选）
        save_path: 保存路径（可选）
        title: 图表标题
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("Warning: matplotlib not available, skipping visualization")
        return

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

    frames = np.arange(len(z))

    # ============================================================
    # 子图 1: Z 坐标
    # ============================================================
    ax1 = axes[0]
    ax1.plot(frames, z, 'b-', alpha=0.3, label='z (raw)')
    if z_smooth is not None:
        ax1.plot(frames, z_smooth, 'b-', linewidth=2, label='z (smooth)')

    # 绘制阈值线
    ax1.axhline(y=config.z_on, color='g', linestyle='--', label=f'z_on={config.z_on}')
    ax1.axhline(y=config.z_off, color='r', linestyle='--', label=f'z_off={config.z_off}')

    # 标记低位区间
    if low_regions:
        for region in low_regions:
            ax1.axvspan(region.start, region.end, alpha=0.2, color='yellow', label='_low_region')

    ax1.set_ylabel('Z (m)')
    ax1.legend(loc='upper right')
    ax1.set_title(f'{title} - Z Coordinate')
    ax1.grid(True, alpha=0.3)

    # ============================================================
    # 子图 2: 水平速度
    # ============================================================
    ax2 = axes[1]
    ax2.plot(frames, v_xy, 'b-', alpha=0.3, label='v_xy (raw)')
    if v_xy_smooth is not None:
        ax2.plot(frames, v_xy_smooth, 'b-', linewidth=2, label='v_xy (smooth)')

    # 绘制阈值线
    ax2.axhline(y=config.v_xy_threshold, color='r', linestyle='--',
                label=f'threshold={config.v_xy_threshold}')

    # 标记关键点
    for kp in keypoints:
        color = 'green' if kp.is_valid else 'red'
        ax2.axvline(x=kp.P_t0, color=color, linestyle='-', alpha=0.7)
        ax2.axvline(x=kp.P_t1, color=color, linestyle='-', alpha=0.7)
        ax2.axvspan(kp.P_t0, kp.P_t1, alpha=0.3, color=color)

        # 标注
        mid = (kp.P_t0 + kp.P_t1) / 2
        ax2.annotate(f'S{kp.sweep_idx}\nL={kp.L23}',
                    xy=(mid, config.v_xy_threshold * 1.5),
                    ha='center', fontsize=8)

    ax2.set_ylabel('v_xy (m/frame)')
    ax2.legend(loc='upper right')
    ax2.set_title('Horizontal Velocity with Sweep Keypoints')
    ax2.grid(True, alpha=0.3)

    # ============================================================
    # 子图 3: Segment 边界
    # ============================================================
    ax3 = axes[2]

    # 绘制时间轴
    ax3.axhline(y=0, color='black', linewidth=2)

    # 绘制每个 segment
    colors = plt.cm.tab10(np.linspace(0, 1, len(boundaries)))
    for i, boundary in enumerate(boundaries):
        if not boundary.is_valid:
            continue

        # 绘制 segment 区间
        ax3.axvspan(boundary.T_t0, boundary.T_t1, alpha=0.3, color=colors[i])

        # 绘制合格窗口起点区间 [s_min, s_max]
        ax3.plot([boundary.s_min, boundary.s_max], [0.5, 0.5],
                'o-', color=colors[i], linewidth=3, markersize=8,
                label=f'S{boundary.sweep_idx}: div={boundary.diversity}')

        # 绘制 P_t0, P_t1
        kp = keypoints[boundary.sweep_idx]
        ax3.plot([kp.P_t0, kp.P_t1], [-0.5, -0.5],
                's-', color=colors[i], linewidth=2, markersize=6)

        # 标注
        ax3.annotate(f'T=[{boundary.T_t0},{boundary.T_t1}]',
                    xy=((boundary.T_t0 + boundary.T_t1) / 2, 0.8),
                    ha='center', fontsize=7)

    ax3.set_ylabel('Segment')
    ax3.set_ylim(-1, 1.5)
    ax3.legend(loc='upper right', fontsize=8)
    ax3.set_title(f'Segment Boundaries (H={config.H}, A_min={config.A_min}, R_min={config.R_min})')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlabel('Frame')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_multi_episode_summary(
    results: dict,  # {ep_idx: (keypoints, boundaries)}
    config: SweepSegmentConfig,
    save_path: Optional[Path] = None
):
    """
    绘制多 episode 的汇总统计

    Args:
        results: 处理结果字典
        config: 配置
        save_path: 保存路径（可选）
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available, skipping visualization")
        return

    # 收集统计数据
    episode_indices = []
    diversities = []
    segment_lengths = []
    L23_values = []

    for ep_idx, (keypoints, boundaries) in results.items():
        for boundary in boundaries:
            if boundary.is_valid:
                episode_indices.append(ep_idx)
                diversities.append(boundary.diversity)
                segment_lengths.append(boundary.segment_length)

        for kp in keypoints:
            if kp.is_valid:
                L23_values.append(kp.L23)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 多样性分布
    ax1 = axes[0, 0]
    ax1.hist(diversities, bins=20, edgecolor='black', alpha=0.7)
    ax1.axvline(x=np.mean(diversities), color='r', linestyle='--',
                label=f'Mean={np.mean(diversities):.1f}')
    ax1.set_xlabel('Diversity |S_t|')
    ax1.set_ylabel('Count')
    ax1.set_title('Diversity Distribution')
    ax1.legend()

    # Segment 长度分布
    ax2 = axes[0, 1]
    ax2.hist(segment_lengths, bins=20, edgecolor='black', alpha=0.7)
    ax2.axvline(x=np.mean(segment_lengths), color='r', linestyle='--',
                label=f'Mean={np.mean(segment_lengths):.1f}')
    ax2.set_xlabel('Segment Length (frames)')
    ax2.set_ylabel('Count')
    ax2.set_title('Segment Length Distribution')
    ax2.legend()

    # L23 分布
    ax3 = axes[1, 0]
    ax3.hist(L23_values, bins=20, edgecolor='black', alpha=0.7)
    ax3.axvline(x=config.L23_min, color='g', linestyle='--', label=f'L23_min={config.L23_min}')
    ax3.axvline(x=config.L23_max, color='r', linestyle='--', label=f'L23_max={config.L23_max}')
    ax3.set_xlabel('L23 (Engage+Stroke frames)')
    ax3.set_ylabel('Count')
    ax3.set_title('L23 Distribution')
    ax3.legend()

    # 每个 episode 的统计
    ax4 = axes[1, 1]
    ep_stats = {}
    for ep_idx, (keypoints, boundaries) in results.items():
        valid_count = sum(1 for b in boundaries if b.is_valid)
        total_div = sum(b.diversity for b in boundaries if b.is_valid)
        ep_stats[ep_idx] = (valid_count, total_div)

    ep_indices = list(ep_stats.keys())
    valid_counts = [ep_stats[ep][0] for ep in ep_indices]
    total_divs = [ep_stats[ep][1] for ep in ep_indices]

    ax4.bar(ep_indices, valid_counts, alpha=0.7, label='Valid Segments')
    ax4_twin = ax4.twinx()
    ax4_twin.plot(ep_indices, total_divs, 'r-o', label='Total Diversity')
    ax4.set_xlabel('Episode Index')
    ax4.set_ylabel('Valid Segments', color='blue')
    ax4_twin.set_ylabel('Total Diversity', color='red')
    ax4.set_title('Per-Episode Statistics')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved summary to {save_path}")
    else:
        plt.show()

    plt.close()
