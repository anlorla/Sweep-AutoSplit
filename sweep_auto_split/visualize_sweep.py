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
    arm = config.active_arm if config.active_arm != "both" else "left"
    kin_result = kinematics.compute_full_kinematics(
        episode_data.state_trajectory,
        arm=arm
    )

    # 提取信号
    tip_to_table = kin_result.tip_to_table_distance  # 刷尖到桌面距离 (m)
    v_xy = kin_result.v_xy  # 水平速度 (m/s)

    # 平滑信号
    d_smooth = smooth_signal(tip_to_table, config.smoothing_window)
    v_xy_smooth = smooth_signal(v_xy, config.smoothing_window)

    # 检测低位区间（sweep 区间）
    # 使用单一阈值检测（进入和退出使用同一阈值）
    sweep_regions = detect_low_distance_regions(
        d_smooth,
        threshold_on=sweep_threshold,
        threshold_off=sweep_threshold + 0.02,  # 滞回 2cm
        min_duration=config.low_region_min_frames
    )

    # 帧数和时间
    num_frames = len(d_smooth)
    frames = np.arange(num_frames)
    times = frames / episode_data.fps

    # 创建图像
    fig, ax1 = plt.subplots(figsize=figsize)

    # 绘制距离曲线
    ax1.plot(frames, d_smooth * 100, 'b-', linewidth=1.5, label='Tip-to-Table Distance')
    ax1.axhline(y=sweep_threshold * 100, color='r', linestyle='--', linewidth=1.5,
                label=f'Sweep Threshold ({sweep_threshold*100:.0f}cm)')

    # 标记 sweep 区间
    for i, region in enumerate(sweep_regions):
        start_frame = region.start
        end_frame = region.end
        ax1.axvspan(start_frame, end_frame, alpha=0.3, color='green',
                   label='Sweep Region' if i == 0 else None)

        # 在区间中间标注 "Sweep"
        mid_frame = (start_frame + end_frame) / 2
        d_min = np.min(d_smooth[start_frame:end_frame+1]) * 100
        ax1.text(mid_frame, d_min - 1, f'S{i+1}', ha='center', va='top',
                fontsize=10, fontweight='bold', color='darkgreen')

    # 设置轴标签和标题
    ax1.set_xlabel('Frame', fontsize=12)
    ax1.set_ylabel('Distance to Table (cm)', fontsize=12, color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_ylim(bottom=-2)  # 允许负值显示（穿透桌面的情况）

    # 添加第二个 x 轴（时间）
    if show_dual_axis:
        ax2 = ax1.twiny()
        ax2.set_xlim(ax1.get_xlim()[0] / episode_data.fps,
                     ax1.get_xlim()[1] / episode_data.fps)
        ax2.set_xlabel('Time (s)', fontsize=12)

    # 标题
    title = f'Episode {episode_data.episode_id}: Sweep Detection'
    if episode_data.task:
        # 截断过长的任务描述
        task_short = episode_data.task[:50] + '...' if len(episode_data.task) > 50 else episode_data.task
        title += f'\nTask: {task_short}'
    ax1.set_title(title, fontsize=14)

    # 图例
    ax1.legend(loc='upper right', fontsize=10)

    # 添加网格
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图片
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        if config.verbose:
            print(f"[visualize_sweep] Saved plot to: {output_path}")

    # 诊断数据
    diagnostic_data = {
        "num_frames": num_frames,
        "fps": episode_data.fps,
        "duration_s": num_frames / episode_data.fps,
        "d_smooth": d_smooth,
        "v_xy_smooth": v_xy_smooth,
        "sweep_regions": sweep_regions,
        "num_sweeps": len(sweep_regions),
        "sweep_threshold": sweep_threshold,
        "d_min": float(np.min(d_smooth)),
        "d_max": float(np.max(d_smooth)),
    }

    return fig, diagnostic_data


def plot_sweep_detection_detailed(
    episode_data: EpisodeData,
    config: SweepSegmentConfig,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10),
    sweep_threshold: float = 0.05,
) -> Tuple[plt.Figure, Dict[str, Any]]:
    """
    绘制详细的 sweep 检测可视化图（包含速度信息）

    包含两个子图：
    1. 刷子 tip 到桌面的距离
    2. 水平速度 v_xy

    Args:
        episode_data: Episode 数据
        config: 切分配置
        output_path: 输出图片路径（可选）
        figsize: 图片尺寸
        sweep_threshold: sweep 判定阈值（米）

    Returns:
        (fig, diagnostic_data)
    """
    # 初始化检测器获取完整诊断数据
    detector = SweepDetector(config)
    diag = detector.get_diagnostic_data(
        episode_data.state_trajectory,
        episode_data.ee_pose_trajectory
    )

    # 帧数和时间
    num_frames = len(diag["d_smooth"])
    frames = np.arange(num_frames)
    times = frames / episode_data.fps

    # 创建图像
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # ========== 子图1: 距离曲线 ==========
    ax1.plot(frames, diag["d_smooth"] * 100, 'b-', linewidth=1.5, label='Tip-to-Table Distance')

    # 阈值线
    ax1.axhline(y=diag["d_threshold_on"] * 100, color='r', linestyle='--',
                linewidth=1.5, label=f'Enter Threshold (z_on={diag["d_threshold_on"]*100:.0f}cm)')
    ax1.axhline(y=diag["d_threshold_off"] * 100, color='orange', linestyle=':',
                linewidth=1.5, label=f'Exit Threshold (z_off={diag["d_threshold_off"]*100:.0f}cm)')
    ax1.axhline(y=sweep_threshold * 100, color='green', linestyle='-.',
                linewidth=1.5, label=f'Sweep Threshold ({sweep_threshold*100:.0f}cm)')

    # 标记 sweep 区间
    for i, region in enumerate(diag["low_regions"]):
        ax1.axvspan(region.start, region.end, alpha=0.25, color='green',
                   label='Sweep Region' if i == 0 else None)
        # 标注
        mid_frame = (region.start + region.end) / 2
        ax1.text(mid_frame, -1, f'S{i+1}', ha='center', va='top',
                fontsize=10, fontweight='bold', color='darkgreen')

    ax1.set_ylabel('Distance to Table (cm)', fontsize=12)
    ax1.set_ylim(bottom=-3)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f'Episode {episode_data.episode_id}: Tip-to-Table Distance', fontsize=12)

    # ========== 子图2: 速度曲线 ==========
    ax2.plot(frames, diag["v_xy_smooth"] * 100, 'purple', linewidth=1.5, label='Horizontal Velocity v_xy')
    ax2.axhline(y=diag["v_xy_threshold"] * 100, color='red', linestyle='--',
                linewidth=1.5, label=f'v_xy Threshold ({diag["v_xy_threshold"]*100:.2f}cm/frame)')

    # 标记 sweep 区间
    for region in diag["low_regions"]:
        ax2.axvspan(region.start, region.end, alpha=0.25, color='green')

    ax2.set_xlabel('Frame', fontsize=12)
    ax2.set_ylabel('Velocity (cm/frame)', fontsize=12)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Horizontal Velocity v_xy', fontsize=12)

    # 添加时间轴（顶部）
    ax_time = ax1.twiny()
    ax_time.set_xlim(0, num_frames / episode_data.fps)
    ax_time.set_xlabel('Time (s)', fontsize=12)

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

    # 信号
    d_smooth = smooth_signal(kin_result.tip_to_table_distance, config.smoothing_window)
    num_frames = len(d_smooth)
    frames = np.arange(num_frames)

    # 检测 sweep 区间
    sweep_regions = detect_low_distance_regions(
        d_smooth,
        threshold_on=sweep_threshold,
        threshold_off=sweep_threshold + 0.02,
        min_duration=config.low_region_min_frames
    )

    # 创建图像
    fig, ax = plt.subplots(figsize=figsize)

    # 静态元素
    ax.plot(frames, d_smooth * 100, 'b-', linewidth=1.5, alpha=0.7)
    ax.axhline(y=sweep_threshold * 100, color='r', linestyle='--', linewidth=1.5)

    for i, region in enumerate(sweep_regions):
        ax.axvspan(region.start, region.end, alpha=0.25, color='green')

    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylabel('Distance to Table (cm)', fontsize=12)
    ax.set_title(f'Episode {episode_data.episode_id}: Sweep Detection', fontsize=14)
    ax.set_ylim(bottom=-2, top=max(d_smooth * 100) + 5)
    ax.grid(True, alpha=0.3)

    # 动态元素
    current_line = ax.axvline(x=0, color='red', linewidth=2, label='Current Frame')
    current_point, = ax.plot(0, d_smooth[0] * 100, 'ro', markersize=10)
    status_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                         fontsize=12, verticalalignment='top',
                         fontweight='bold')

    def init():
        current_line.set_xdata([0])
        current_point.set_data([0], [d_smooth[0] * 100])
        status_text.set_text('')
        return current_line, current_point, status_text

    def update(frame):
        current_line.set_xdata([frame])
        current_point.set_data([frame], [d_smooth[frame] * 100])

        # 检查是否在 sweep 区间
        in_sweep = False
        sweep_idx = -1
        for i, region in enumerate(sweep_regions):
            if region.start <= frame <= region.end:
                in_sweep = True
                sweep_idx = i + 1
                break

        d_current = d_smooth[frame] * 100
        if in_sweep:
            status_text.set_text(f'Frame {frame} | d={d_current:.1f}cm | SWEEP #{sweep_idx}')
            status_text.set_color('green')
        else:
            status_text.set_text(f'Frame {frame} | d={d_current:.1f}cm')
            status_text.set_color('black')

        return current_line, current_point, status_text

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
