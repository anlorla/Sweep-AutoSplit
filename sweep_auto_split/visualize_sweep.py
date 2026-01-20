"""
Sweep 检测可视化模块

功能：
1. 可视化单个 episode 的刷子 tip 到桌面的距离曲线
2. 根据低位阈值（5cm）检测 sweep 区间
3. 在图像中标记 sweep（engage+stroke）区间
4. 支持生成静态图和动画视频
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, FFMpegWriter
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path

from .config import SweepSegmentConfig, SweepKeypoint
from .kinematics import (
    DualArmKinematics,
    KinematicsConfig,
    create_default_kinematics_config,
)
from .signal_processing import smooth_signal
from .sweep_detector import detect_low_distance_regions, SweepDetector
from .data_loader import LeRobotDataLoader, EpisodeData


# ============================================================
# 核心可视化函数
# ============================================================

def plot_sweep_detection(
    episode_data: EpisodeData,
    config: SweepSegmentConfig,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 6),
    show_dual_axis: bool = True,
    sweep_threshold: float = 0.05,  # 5cm
) -> Tuple[plt.Figure, Dict[str, Any]]:
    """
    绘制 sweep 检测可视化图

    根据低位阈值检测 sweep 区间，绘制刷子 tip 到桌面的距离曲线

    Args:
        episode_data: Episode 数据
        config: 切分配置
        output_path: 输出图片路径（可选）
        figsize: 图片尺寸
        show_dual_axis: 是否显示双坐标轴（帧数 + 秒数）
        sweep_threshold: sweep 判定阈值（米），默认 0.05m = 5cm

    Returns:
        (fig, diagnostic_data): matplotlib 图对象和诊断数据
    """
    # 初始化运动学
    kin_config = create_default_kinematics_config(fps=episode_data.fps)
    kinematics = DualArmKinematics(kin_config)

    # 计算运动学
    state_trajectory = episode_data.state_trajectory
    ee_pose_trajectory = episode_data.ee_pose_trajectory

    # 计算左右两臂的运动学数据
    if ee_pose_trajectory is not None:
        from .kinematics import compute_full_kinematics_from_ee_pose
        kin_result_right = compute_full_kinematics_from_ee_pose(
            ee_pose_trajectory, arm="right", config=kin_config
        )
        kin_result_left = compute_full_kinematics_from_ee_pose(
            ee_pose_trajectory, arm="left", config=kin_config
        )
    else:
        kinematics = DualArmKinematics(kin_config)
        kin_result_right = kinematics.compute_full_kinematics(
            state_trajectory, arm="right"
        )
        kin_result_left = kinematics.compute_full_kinematics(
            state_trajectory, arm="left"
        )

    # 提取数据
    tip_to_table_right = kin_result_right.tip_to_table_distance
    tip_to_table_left = kin_result_left.tip_to_table_distance

    N = len(tip_to_table_right)

    # 平滑信号
    d_smooth_right = smooth_signal(tip_to_table_right, config.smoothing_window)
    d_smooth_left = smooth_signal(tip_to_table_left, config.smoothing_window)

    # 检测低位区间（sweep 区间）
    # 使用单一阈值检测（进入和退出使用同一阈值）
    sweep_regions = detect_low_distance_regions(
        d_smooth,
        threshold_on=sweep_threshold,
        threshold_off=sweep_threshold + 0.02,  # 滞回 2cm
        min_duration=config.low_region_min_frames
    )

    # 创建图表 - 2个子图（上：右臂，下：左臂）
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # 标记 sweep 区间的颜色
    colors = plt.cm.Set1(np.linspace(0, 1, max(len(keypoints), 1)))

    # === 子图1：右臂刷尖到桌面距离 ===
    ax1 = axes[0]
    ax1.plot(time_axis, tip_to_table_right * 100, 'b-', alpha=0.3, label='Raw d(t)')
    ax1.plot(time_axis, d_smooth_right * 100, 'b-', linewidth=2, label='Smoothed d(t)')
    ax1.axhline(y=0, color='brown', linestyle='--', linewidth=2, label='Table surface')

    # 绘制距离曲线
    ax1.plot(frames, d_smooth * 100, 'b-', linewidth=1.5, label='Tip-to-Table Distance')
    ax1.axhline(y=sweep_threshold * 100, color='r', linestyle='--', linewidth=1.5,
                label=f'Sweep Threshold ({sweep_threshold*100:.0f}cm)')

    # 标记 sweep 区间
    for i, kp in enumerate(keypoints):
        start_t = kp.P_t0 / fps if show_seconds else kp.P_t0
        end_t = kp.P_t1 / fps if show_seconds else kp.P_t1
        color = colors[i % len(colors)]
        alpha = 0.4 if kp.is_valid else 0.15

        ax1.axvspan(start_t, end_t, alpha=alpha, color=color,
                   label=f'Sweep {i} {"✓" if kp.is_valid else "✗"}')

    ax1.set_ylabel('Tip-to-Table Distance (cm)')
    ax1.set_title(f'Episode {episode_data.episode_id}: Sweep Detection - Right Arm')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # === 子图2：左臂刷尖到桌面距离 ===
    ax2 = axes[1]
    ax2.plot(time_axis, tip_to_table_left * 100, 'g-', alpha=0.3, label='Raw d(t)')
    ax2.plot(time_axis, d_smooth_left * 100, 'g-', linewidth=2, label='Smoothed d(t)')
    ax2.axhline(y=0, color='brown', linestyle='--', linewidth=2, label='Table surface')

    # 绘制低位阈值线
    ax2.axhline(y=config.z_on * 100, color='orange', linestyle=':', linewidth=1.5,
                label=f'z_on threshold ({config.z_on*100:.1f}cm)')
    ax2.axhline(y=config.z_off * 100, color='red', linestyle=':', linewidth=1.5,
                label=f'z_off threshold ({config.z_off*100:.1f}cm)')

    # 标记 sweep 区间
    for i, kp in enumerate(keypoints):
        start_t = kp.P_t0 / fps if show_seconds else kp.P_t0
        end_t = kp.P_t1 / fps if show_seconds else kp.P_t1
        color = colors[i % len(colors)]
        alpha = 0.4 if kp.is_valid else 0.15
        ax2.axvspan(start_t, end_t, alpha=alpha, color=color,
                   label=f'Sweep {i} {"✓" if kp.is_valid else "✗"}')

    ax2.set_ylabel('Tip-to-Table Distance (cm)')
    ax2.set_title(f'Episode {episode_data.episode_id}: Sweep Detection - Left Arm')
    ax2.set_xlabel(xlabel)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=-2)

    plt.tight_layout()

    # 保存
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        if config.verbose:
            print(f"[visualize_sweep] Saved detailed plot to: {output_path}")

    return fig, diag


def generate_sweep_detection_video(
    episode_data: EpisodeData,
    config: SweepSegmentConfig,
    output_path: str,
    figsize: Tuple[int, int] = (12, 6),
    sweep_threshold: float = 0.05,
) -> bool:
    """
    生成 sweep 检测动画视频

    显示当前帧位置随时间移动的动画

    Args:
        episode_data: Episode 数据
        config: 切分配置
        output_path: 输出视频路径
        figsize: 图片尺寸
        sweep_threshold: sweep 判定阈值（米）

    Returns:
        是否成功
    """
    # 初始化运动学
    kin_config = create_default_kinematics_config(fps=episode_data.fps)
    kinematics = DualArmKinematics(kin_config)

    arm = config.active_arm if config.active_arm != "both" else "left"
    kin_result = kinematics.compute_full_kinematics(
        episode_data.state_trajectory,
        arm=arm
    )

    # 计算运动学
    state_trajectory = episode_data.state_trajectory
    ee_pose_trajectory = episode_data.ee_pose_trajectory

    # 计算左右两臂的运动学数据
    if ee_pose_trajectory is not None:
        kin_result_right = compute_full_kinematics_from_ee_pose(
            ee_pose_trajectory, arm="right", config=kin_config
        )
        kin_result_left = compute_full_kinematics_from_ee_pose(
            ee_pose_trajectory, arm="left", config=kin_config
        )
    else:
        kinematics = DualArmKinematics(kin_config)
        kin_result_right = kinematics.compute_full_kinematics(
            state_trajectory, arm="right"
        )
        kin_result_left = kinematics.compute_full_kinematics(
            state_trajectory, arm="left"
        )

    # 提取数据
    tip_to_table_right = kin_result_right.tip_to_table_distance
    tip_to_table_left = kin_result_left.tip_to_table_distance

    N = len(tip_to_table_right)

    # 平滑信号
    d_smooth_right = smooth_signal(tip_to_table_right, config.smoothing_window)
    d_smooth_left = smooth_signal(tip_to_table_left, config.smoothing_window)

    # 检测 sweep（用于最终标注）- 这里用原始的 active_arm，可能是 "both"
    detector = SweepDetector(config, kinematics_config=kin_config)
    keypoints = detector.detect_keypoints(state_trajectory, ee_pose_trajectory)

    # 时间轴（秒）
    time_axis = np.arange(N) / fps

    # 创建图表 - 2个子图（上：右臂，下：左臂）
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # 设置坐标轴范围
    ax1, ax2 = axes

    # 子图1：右臂刷尖到桌面距离
    ax1.set_xlim(0, time_axis[-1])
    ax1.set_ylim(-2, max(d_smooth_right.max() * 100, 50) * 1.1)
    ax1.set_ylabel('Tip-to-Table Distance (cm)')
    ax1.set_title(f'Episode {episode_data.episode_id}: Sweep Detection - Right Arm (Dynamic)')
    ax1.grid(True, alpha=0.3)

    # 绘制静态元素
    ax1.axhline(y=0, color='brown', linestyle='--', linewidth=2, label='Table surface')
    ax1.axhline(y=config.z_on * 100, color='orange', linestyle=':', linewidth=1.5,
                label=f'z_on ({config.z_on*100:.1f}cm)')
    ax1.axhline(y=config.z_off * 100, color='red', linestyle=':', linewidth=1.5,
                label=f'z_off ({config.z_off*100:.1f}cm)')

    # 子图2：左臂刷尖到桌面距离
    ax2.set_xlim(0, time_axis[-1])
    ax2.set_ylim(-2, max(d_smooth_left.max() * 100, 50) * 1.1)
    ax2.set_ylabel('Tip-to-Table Distance (cm)')
    ax2.set_title(f'Episode {episode_data.episode_id}: Sweep Detection - Left Arm (Dynamic)')
    ax2.set_xlabel('Time (seconds)')
    ax2.grid(True, alpha=0.3)

    # 绘制静态元素
    ax2.axhline(y=0, color='brown', linestyle='--', linewidth=2, label='Table surface')
    ax2.axhline(y=config.z_on * 100, color='orange', linestyle=':', linewidth=1.5,
                label=f'z_on ({config.z_on*100:.1f}cm)')
    ax2.axhline(y=config.z_off * 100, color='red', linestyle=':', linewidth=1.5,
                label=f'z_off ({config.z_off*100:.1f}cm)')

    # 添加图例
    ax1.legend(loc='upper right', fontsize=8)
    ax2.legend(loc='upper right', fontsize=8)

    # 创建动态线条
    line1_raw, = ax1.plot([], [], 'b-', alpha=0.3, label='Raw d(t)')
    line1_smooth, = ax1.plot([], [], 'b-', linewidth=2, label='Smoothed d(t)')
    line2_raw, = ax2.plot([], [], 'g-', alpha=0.3, label='Raw d(t)')
    line2_smooth, = ax2.plot([], [], 'g-', linewidth=2, label='Smoothed d(t)')

    # 当前帧指示线
    vline1 = ax1.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
    vline2 = ax2.axvline(x=0, color='gray', linestyle='-', alpha=0.5)

    # 当前位置点
    dot1, = ax1.plot([], [], 'ro', markersize=8)
    dot2, = ax2.plot([], [], 'ro', markersize=8)

    # sweep 区间遮罩（将在检测到时添加）
    sweep_patches = []

    def init():
        line1_raw.set_data([], [])
        line1_smooth.set_data([], [])
        line2_raw.set_data([], [])
        line2_smooth.set_data([], [])
        dot1.set_data([], [])
        dot2.set_data([], [])
        return line1_raw, line1_smooth, line2_raw, line2_smooth, dot1, dot2

    def animate(frame):
        # 当前时间点
        t = time_axis[:frame+1]

        # 更新线条
        line1_raw.set_data(t, tip_to_table_right[:frame+1] * 100)
        line1_smooth.set_data(t, d_smooth_right[:frame+1] * 100)
        line2_raw.set_data(t, tip_to_table_left[:frame+1] * 100)
        line2_smooth.set_data(t, d_smooth_left[:frame+1] * 100)

        # 更新当前帧指示线
        current_time = time_axis[frame]
        vline1.set_xdata([current_time, current_time])
        vline2.set_xdata([current_time, current_time])

        # 更新当前位置点
        dot1.set_data([current_time], [d_smooth_right[frame] * 100])
        dot2.set_data([current_time], [d_smooth_left[frame] * 100])

        # 检查是否在 sweep 区间内，添加遮罩
        colors = plt.cm.Set1(np.linspace(0, 1, max(len(keypoints), 1)))
        for i, kp in enumerate(keypoints):
            # 当动画进入 sweep 区间时添加遮罩
            if frame >= kp.P_t0 and frame <= kp.P_t1:
                # 检查是否已经添加了这个 sweep 的遮罩
                patch_id = f"sweep_{i}"
                if patch_id not in [p[0] for p in sweep_patches]:
                    start_t = kp.P_t0 / fps
                    end_t = kp.P_t1 / fps
                    color = colors[i % len(colors)]
                    alpha = 0.4 if kp.is_valid else 0.15

                    p1 = ax1.axvspan(start_t, end_t, alpha=alpha, color=color)
                    p2 = ax2.axvspan(start_t, end_t, alpha=alpha, color=color)
                    sweep_patches.append((patch_id, p1, p2))

        return line1_raw, line1_smooth, line2_raw, line2_smooth, dot1, dot2, vline1, vline2

    # 创建动画
    anim = FuncAnimation(fig, update, frames=range(num_frames),
                        init_func=init, blit=True, interval=100)

    # 保存视频
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        writer = FFMpegWriter(fps=episode_data.fps, bitrate=2000)
        anim.save(output_path, writer=writer)
        plt.close(fig)
        if config.verbose:
            print(f"[visualize_sweep] Saved video to: {output_path}")
        return True
    except Exception as e:
        plt.close(fig)
        if config.verbose:
            print(f"[visualize_sweep] Failed to save video: {e}")
        return False


# ============================================================
# 批量处理函数
# ============================================================

def visualize_dataset_sweeps(
    dataset_path: str,
    output_dir: str,
    config: Optional[SweepSegmentConfig] = None,
    episode_ids: Optional[List[int]] = None,
    max_episodes: int = 10,
    sweep_threshold: float = 0.05,
    generate_videos: bool = False,
):
    """
    批量可视化数据集中的 sweep 检测结果

    Args:
        dataset_path: LeRobot 数据集路径
        output_dir: 输出目录
        config: 切分配置
        episode_ids: 指定要可视化的 episode ID 列表（可选）
        max_episodes: 最大可视化 episode 数量
        sweep_threshold: sweep 判定阈值（米）
        generate_videos: 是否生成动画视频
    """
    if config is None:
        config = SweepSegmentConfig()

    # 加载数据集
    data_loader = LeRobotDataLoader(dataset_path)

    if config.verbose:
        print(f"[visualize_sweep] Dataset: {dataset_path}")
        print(f"[visualize_sweep] Total episodes: {data_loader.total_episodes}")

    # 确定要可视化的 episode
    if episode_ids is None:
        episode_ids = data_loader.get_episode_list()[:max_episodes]
    else:
        episode_ids = episode_ids[:max_episodes]

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 统计信息
    stats = {
        "total_episodes": len(episode_ids),
        "total_sweeps": 0,
        "episodes_with_sweeps": 0,
    }

    # 处理每个 episode
    for ep_id in episode_ids:
        if config.verbose:
            print(f"\n[visualize_sweep] Processing episode {ep_id}...")

        episode_data = data_loader.load_episode(ep_id)

        # 生成静态图
        fig, diag = plot_sweep_detection_detailed(
            episode_data,
            config,
            output_path=str(output_path / f"episode_{ep_id:06d}_sweep.png"),
            sweep_threshold=sweep_threshold,
        )
        plt.close(fig)

        # 统计
        num_sweeps = len(diag.get("low_regions", []))
        stats["total_sweeps"] += num_sweeps
        if num_sweeps > 0:
            stats["episodes_with_sweeps"] += 1

        if config.verbose:
            print(f"  - Found {num_sweeps} sweeps")

        # 生成视频（可选）
        if generate_videos:
            video_path = str(output_path / f"episode_{ep_id:06d}_sweep.mp4")
            generate_sweep_detection_video(
                episode_data, config, video_path,
                sweep_threshold=sweep_threshold
            )

    # 打印统计
    if config.verbose:
        print(f"\n{'='*50}")
        print(f"SWEEP VISUALIZATION SUMMARY")
        print(f"{'='*50}")
        print(f"Total episodes processed: {stats['total_episodes']}")
        print(f"Episodes with sweeps: {stats['episodes_with_sweeps']}")
        print(f"Total sweeps detected: {stats['total_sweeps']}")
        print(f"Output directory: {output_path}")
        print(f"{'='*50}")

    return stats


# ============================================================
# 命令行入口
# ============================================================

def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize sweep detection for LeRobot dataset"
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to LeRobot dataset"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="./sweep_viz",
        help="Output directory (default: ./sweep_viz)"
    )
    parser.add_argument(
        "-n", "--max-episodes",
        type=int,
        default=10,
        help="Maximum number of episodes to visualize (default: 10)"
    )
    parser.add_argument(
        "-e", "--episodes",
        type=int,
        nargs="+",
        help="Specific episode IDs to visualize"
    )
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=0.05,
        help="Sweep threshold in meters (default: 0.05 = 5cm)"
    )
    parser.add_argument(
        "-v", "--video",
        action="store_true",
        help="Generate animation videos"
    )
    parser.add_argument(
        "--arm",
        type=str,
        choices=["left", "right", "both"],
        default="left",
        help="Which arm to analyze (default: left)"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )

    args = parser.parse_args()

    # 创建配置
    config = SweepSegmentConfig(
        active_arm=args.arm,
        verbose=not args.quiet,
    )

    # 运行可视化
    visualize_dataset_sweeps(
        dataset_path=args.dataset_path,
        output_dir=args.output,
        config=config,
        episode_ids=args.episodes,
        max_episodes=args.max_episodes,
        sweep_threshold=args.threshold,
        generate_videos=args.video,
    )


if __name__ == "__main__":
    main()
