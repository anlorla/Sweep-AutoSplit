#!/usr/bin/env python3
"""
Sweep 检测可视化脚本

生成刷尖高度、速度和检测区间的可视化图表
支持静态图片和动态视频两种输出格式
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional, Tuple

from .config import SweepSegmentConfig
from .data_loader import LeRobotDataLoader
from .sweep_detector import SweepDetector
from .kinematics import (
    create_default_kinematics_config,
    compute_full_kinematics_from_ee_pose,
    DualArmKinematics,
)
from .signal_processing import smooth_signal

# 尝试导入视频生成所需的库
try:
    import matplotlib.animation as animation
    ANIMATION_AVAILABLE = True
except ImportError:
    ANIMATION_AVAILABLE = False


def plot_sweep_detection_for_episode(
    episode_data,
    config: SweepSegmentConfig,
    output_path: Optional[str] = None,
    fps: float = 10.0,
    show_seconds: bool = True,
    figsize: Tuple[int, int] = (16, 10)
):
    """
    为单个 episode 绘制 sweep 检测可视化图

    Args:
        episode_data: EpisodeData 对象
        config: 配置
        output_path: 输出路径（如果为 None 则显示）
        fps: 帧率
        show_seconds: 横轴是否显示秒数（否则显示帧数）
        figsize: 图像尺寸
    """
    # 创建运动学配置
    kin_config = create_default_kinematics_config(
        tip_offset_z=0.30,
        table_z=-0.05,
        fps=fps
    )

    # 计算运动学
    state_trajectory = episode_data.state_trajectory
    ee_pose_trajectory = episode_data.ee_pose_trajectory

    # 对于 "both" 模式，可视化默认显示左臂数据
    viz_arm = config.active_arm if config.active_arm != "both" else "left"

    if ee_pose_trajectory is not None:
        from .kinematics import compute_full_kinematics_from_ee_pose
        kin_result = compute_full_kinematics_from_ee_pose(
            ee_pose_trajectory, arm=viz_arm, config=kin_config
        )
    else:
        kinematics = DualArmKinematics(kin_config)
        kin_result = kinematics.compute_full_kinematics(
            state_trajectory, arm=viz_arm
        )

    # 提取数据
    tip_to_table = kin_result.tip_to_table_distance  # 刷尖到桌面距离
    v_xy = kin_result.v_xy
    v_z = kin_result.v_z

    N = len(tip_to_table)

    # 平滑信号
    d_smooth = smooth_signal(tip_to_table, config.smoothing_window)
    v_xy_smooth = smooth_signal(v_xy, config.smoothing_window)

    # 检测 sweep
    detector = SweepDetector(config, kinematics_config=kin_config)
    keypoints = detector.detect_keypoints(state_trajectory, ee_pose_trajectory)

    # 时间轴
    if show_seconds:
        time_axis = np.arange(N) / fps
        xlabel = "Time (seconds)"
    else:
        time_axis = np.arange(N)
        xlabel = "Frame"

    # 创建图表
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    # === 子图1：刷尖到桌面距离 ===
    ax1 = axes[0]
    ax1.plot(time_axis, tip_to_table * 100, 'b-', alpha=0.3, label='Raw d(t)')
    ax1.plot(time_axis, d_smooth * 100, 'b-', linewidth=2, label='Smoothed d(t)')
    ax1.axhline(y=0, color='brown', linestyle='--', linewidth=2, label='Table surface')

    # 绘制低位阈值线
    ax1.axhline(y=config.z_on * 100, color='orange', linestyle=':', linewidth=1.5,
                label=f'z_on threshold ({config.z_on*100:.1f}cm)')
    ax1.axhline(y=config.z_off * 100, color='red', linestyle=':', linewidth=1.5,
                label=f'z_off threshold ({config.z_off*100:.1f}cm)')

    # 标记 sweep 区间
    colors = plt.cm.Set1(np.linspace(0, 1, max(len(keypoints), 1)))
    for i, kp in enumerate(keypoints):
        start_t = kp.P_t0 / fps if show_seconds else kp.P_t0
        end_t = kp.P_t1 / fps if show_seconds else kp.P_t1
        color = colors[i % len(colors)]
        alpha = 0.4 if kp.is_valid else 0.15

        ax1.axvspan(start_t, end_t, alpha=alpha, color=color,
                   label=f'Sweep {i} {"✓" if kp.is_valid else "✗"}')

    ax1.set_ylabel('Tip-to-Table Distance (cm)')
    ax1.set_title(f'Episode {episode_data.episode_id}: Sweep Detection Visualization (Low Region Based)')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=-2)

    # === 子图2：水平速度 v_xy ===
    ax2 = axes[1]
    ax2.plot(time_axis, v_xy * 100, 'g-', alpha=0.3, label='Raw v_xy')
    ax2.plot(time_axis, v_xy_smooth * 100, 'g-', linewidth=2, label='Smoothed v_xy')

    # 计算阈值
    v_xy_percentile = config.energy_percentile
    v_xy_threshold = np.percentile(v_xy_smooth, v_xy_percentile)
    v_xy_mean = np.mean(v_xy_smooth)
    v_xy_std = np.std(v_xy_smooth)
    min_threshold = v_xy_mean + 0.3 * v_xy_std
    v_xy_threshold = max(v_xy_threshold, min_threshold, 1e-6)

    ax2.axhline(y=v_xy_threshold * 100, color='r', linestyle='--',
               label=f'v_xy threshold ({v_xy_percentile}th pct)')

    # 标记 sweep 区间
    for i, kp in enumerate(keypoints):
        start_t = kp.P_t0 / fps if show_seconds else kp.P_t0
        end_t = kp.P_t1 / fps if show_seconds else kp.P_t1
        color = colors[i % len(colors)]
        alpha = 0.4 if kp.is_valid else 0.15
        ax2.axvspan(start_t, end_t, alpha=alpha, color=color)

    ax2.set_ylabel('Horizontal Velocity v_xy (cm/s)')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # === 子图3：垂向速度 v_z ===
    ax3 = axes[2]
    ax3.plot(time_axis, v_z * 100, 'm-', alpha=0.5, linewidth=1, label='v_z')
    ax3.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    # 标记 sweep 区间
    for i, kp in enumerate(keypoints):
        start_t = kp.P_t0 / fps if show_seconds else kp.P_t0
        end_t = kp.P_t1 / fps if show_seconds else kp.P_t1
        color = colors[i % len(colors)]
        alpha = 0.4 if kp.is_valid else 0.15
        ax3.axvspan(start_t, end_t, alpha=alpha, color=color)

    ax3.set_ylabel('Vertical Velocity v_z (cm/s)')
    ax3.set_xlabel(xlabel)
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存或显示
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    else:
        plt.show()

    return keypoints


def generate_sweep_detection_video(
    episode_data,
    config: SweepSegmentConfig,
    output_path: str,
    fps: float = 10.0,
    video_fps: int = 10,
    figsize: Tuple[int, int] = (16, 10)
):
    """
    为单个 episode 生成动态 sweep 检测视频

    折线图随时间动态绘制，展示检测过程

    Args:
        episode_data: EpisodeData 对象
        config: 配置
        output_path: 输出视频路径 (.mp4)
        fps: 数据帧率
        video_fps: 视频帧率
        figsize: 图像尺寸
    """
    if not ANIMATION_AVAILABLE:
        print("Warning: matplotlib.animation not available, cannot generate video")
        return None

    # 创建运动学配置
    kin_config = create_default_kinematics_config(
        tip_offset_z=0.30,
        table_z=-0.05,
        fps=fps
    )

    # 计算运动学
    state_trajectory = episode_data.state_trajectory
    ee_pose_trajectory = episode_data.ee_pose_trajectory

    # 对于 "both" 模式，可视化默认显示左臂数据
    viz_arm = config.active_arm if config.active_arm != "both" else "left"

    if ee_pose_trajectory is not None:
        kin_result = compute_full_kinematics_from_ee_pose(
            ee_pose_trajectory, arm=viz_arm, config=kin_config
        )
    else:
        kinematics = DualArmKinematics(kin_config)
        kin_result = kinematics.compute_full_kinematics(
            state_trajectory, arm=viz_arm
        )

    # 提取数据
    tip_to_table = kin_result.tip_to_table_distance
    v_xy = kin_result.v_xy
    v_z = kin_result.v_z

    N = len(tip_to_table)

    # 平滑信号
    d_smooth = smooth_signal(tip_to_table, config.smoothing_window)
    v_xy_smooth = smooth_signal(v_xy, config.smoothing_window)

    # 检测 sweep（用于最终标注）- 这里用原始的 active_arm，可能是 "both"
    detector = SweepDetector(config, kinematics_config=kin_config)
    keypoints = detector.detect_keypoints(state_trajectory, ee_pose_trajectory)

    # 时间轴（秒）
    time_axis = np.arange(N) / fps

    # 计算 v_xy 阈值
    v_xy_percentile = config.energy_percentile
    v_xy_threshold = np.percentile(v_xy_smooth, v_xy_percentile)
    v_xy_mean = np.mean(v_xy_smooth)
    v_xy_std = np.std(v_xy_smooth)
    min_threshold = v_xy_mean + 0.3 * v_xy_std
    v_xy_threshold = max(v_xy_threshold, min_threshold, 1e-6)

    # 创建图表
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    # 设置坐标轴范围
    ax1, ax2, ax3 = axes

    # 子图1：刷尖到桌面距离
    ax1.set_xlim(0, time_axis[-1])
    ax1.set_ylim(-2, max(d_smooth.max() * 100, 50) * 1.1)
    ax1.set_ylabel('Tip-to-Table Distance (cm)')
    ax1.set_title(f'Episode {episode_data.episode_id}: Sweep Detection (Dynamic)')
    ax1.grid(True, alpha=0.3)

    # 绘制静态元素
    ax1.axhline(y=0, color='brown', linestyle='--', linewidth=2, label='Table surface')
    ax1.axhline(y=config.z_on * 100, color='orange', linestyle=':', linewidth=1.5,
                label=f'z_on ({config.z_on*100:.1f}cm)')
    ax1.axhline(y=config.z_off * 100, color='red', linestyle=':', linewidth=1.5,
                label=f'z_off ({config.z_off*100:.1f}cm)')

    # 子图2：水平速度
    ax2.set_xlim(0, time_axis[-1])
    ax2.set_ylim(0, max(v_xy_smooth.max() * 100, 30) * 1.1)
    ax2.set_ylabel('Horizontal Velocity v_xy (cm/s)')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=v_xy_threshold * 100, color='r', linestyle='--',
                label=f'v_xy threshold ({v_xy_percentile}th pct)')

    # 子图3：垂向速度
    ax3.set_xlim(0, time_axis[-1])
    v_z_max = max(abs(v_z.min()), abs(v_z.max())) * 100
    ax3.set_ylim(-v_z_max * 1.2, v_z_max * 1.2)
    ax3.set_ylabel('Vertical Velocity v_z (cm/s)')
    ax3.set_xlabel('Time (seconds)')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    # 添加图例
    ax1.legend(loc='upper right', fontsize=8)
    ax2.legend(loc='upper right', fontsize=8)

    # 创建动态线条
    line1_raw, = ax1.plot([], [], 'b-', alpha=0.3, label='Raw d(t)')
    line1_smooth, = ax1.plot([], [], 'b-', linewidth=2, label='Smoothed d(t)')
    line2_raw, = ax2.plot([], [], 'g-', alpha=0.3)
    line2_smooth, = ax2.plot([], [], 'g-', linewidth=2)
    line3, = ax3.plot([], [], 'm-', alpha=0.5, linewidth=1)

    # 当前帧指示线
    vline1 = ax1.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
    vline2 = ax2.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
    vline3 = ax3.axvline(x=0, color='gray', linestyle='-', alpha=0.5)

    # 当前位置点
    dot1, = ax1.plot([], [], 'ro', markersize=8)
    dot2, = ax2.plot([], [], 'ro', markersize=8)
    dot3, = ax3.plot([], [], 'ro', markersize=8)

    # sweep 区间遮罩（将在检测到时添加）
    sweep_patches = []

    def init():
        line1_raw.set_data([], [])
        line1_smooth.set_data([], [])
        line2_raw.set_data([], [])
        line2_smooth.set_data([], [])
        line3.set_data([], [])
        dot1.set_data([], [])
        dot2.set_data([], [])
        dot3.set_data([], [])
        return line1_raw, line1_smooth, line2_raw, line2_smooth, line3, dot1, dot2, dot3

    def animate(frame):
        # 当前时间点
        t = time_axis[:frame+1]

        # 更新线条
        line1_raw.set_data(t, tip_to_table[:frame+1] * 100)
        line1_smooth.set_data(t, d_smooth[:frame+1] * 100)
        line2_raw.set_data(t, v_xy[:frame+1] * 100)
        line2_smooth.set_data(t, v_xy_smooth[:frame+1] * 100)
        line3.set_data(t, v_z[:frame+1] * 100)

        # 更新当前帧指示线
        current_time = time_axis[frame]
        vline1.set_xdata([current_time, current_time])
        vline2.set_xdata([current_time, current_time])
        vline3.set_xdata([current_time, current_time])

        # 更新当前位置点
        dot1.set_data([current_time], [d_smooth[frame] * 100])
        dot2.set_data([current_time], [v_xy_smooth[frame] * 100])
        dot3.set_data([current_time], [v_z[frame] * 100])

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
                    p3 = ax3.axvspan(start_t, end_t, alpha=alpha, color=color)
                    sweep_patches.append((patch_id, p1, p2, p3))

        return line1_raw, line1_smooth, line2_raw, line2_smooth, line3, dot1, dot2, dot3, vline1, vline2, vline3

    # 创建动画
    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=N, interval=1000/video_fps, blit=False
    )

    # 保存视频
    print(f"Generating video: {output_path} ({N} frames)")
    try:
        writer = animation.FFMpegWriter(fps=video_fps, bitrate=2000)
        anim.save(output_path, writer=writer)
        print(f"Saved: {output_path}")
    except Exception as e:
        print(f"FFMpeg not available, trying pillow: {e}")
        try:
            # 如果没有 ffmpeg，尝试保存为 gif
            gif_path = output_path.replace('.mp4', '.gif')
            anim.save(gif_path, writer='pillow', fps=video_fps)
            print(f"Saved as GIF: {gif_path}")
        except Exception as e2:
            print(f"Failed to save video: {e2}")

    plt.close()
    return keypoints


def visualize_dataset(
    dataset_path: str,
    output_dir: str,
    max_episodes: int = 5,
    config: Optional[SweepSegmentConfig] = None
):
    """
    可视化数据集中的多个 episode

    Args:
        dataset_path: 数据集路径
        output_dir: 输出目录
        max_episodes: 最大处理 episode 数
        config: 配置
    """
    if config is None:
        config = SweepSegmentConfig(
            verbose=False,
            energy_percentile=50,
            merge_gap=3,
            active_arm="left"  # 必须指定手臂
        )

    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据集
    print(f"Loading dataset: {dataset_path}")
    data_loader = LeRobotDataLoader(dataset_path)
    print(f"  Total episodes: {data_loader.total_episodes}")
    print(f"  FPS: {data_loader.fps}")

    # 处理每个 episode
    episode_ids = data_loader.get_episode_list()[:max_episodes]

    for i, ep_id in enumerate(episode_ids):
        print(f"\nProcessing Episode {i+1}/{len(episode_ids)} (ID: {ep_id})")

        episode_data = data_loader.load_episode(ep_id)
        output_path = output_dir / f"sweep_detection_ep{ep_id:04d}.png"

        keypoints = plot_sweep_detection_for_episode(
            episode_data=episode_data,
            config=config,
            output_path=str(output_path),
            fps=data_loader.fps,
            show_seconds=True
        )

        print(f"  Detected {len(keypoints)} sweeps, "
              f"{sum(1 for kp in keypoints if kp.is_valid)} valid")

    print(f"\n✓ Saved {len(episode_ids)} visualizations to {output_dir}")


def visualize_dataset_video(
    dataset_path: str,
    output_dir: str,
    max_episodes: int = 5,
    config: Optional[SweepSegmentConfig] = None,
    video_fps: int = 10
):
    """
    为数据集生成动态检测视频

    Args:
        dataset_path: 数据集路径
        output_dir: 输出目录
        max_episodes: 最大处理 episode 数
        config: 配置
        video_fps: 视频帧率
    """
    if config is None:
        config = SweepSegmentConfig(
            verbose=False,
            energy_percentile=50,
            merge_gap=3,
            active_arm="left"
        )

    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据集
    print(f"Loading dataset: {dataset_path}")
    data_loader = LeRobotDataLoader(dataset_path)
    print(f"  Total episodes: {data_loader.total_episodes}")
    print(f"  FPS: {data_loader.fps}")

    # 处理每个 episode
    episode_ids = data_loader.get_episode_list()[:max_episodes]

    for i, ep_id in enumerate(episode_ids):
        print(f"\nProcessing Episode {i+1}/{len(episode_ids)} (ID: {ep_id})")

        episode_data = data_loader.load_episode(ep_id)
        output_path = output_dir / f"sweep_detection_ep{ep_id:04d}.mp4"

        keypoints = generate_sweep_detection_video(
            episode_data=episode_data,
            config=config,
            output_path=str(output_path),
            fps=data_loader.fps,
            video_fps=video_fps
        )

        if keypoints:
            print(f"  Detected {len(keypoints)} sweeps, "
                  f"{sum(1 for kp in keypoints if kp.is_valid)} valid")

    print(f"\n✓ Saved {len(episode_ids)} videos to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize sweep detection")
    parser.add_argument("--input", "-i", type=str, required=True,
                       help="Input dataset path")
    parser.add_argument("--output", "-o", type=str, default="./sweep_viz",
                       help="Output directory")
    parser.add_argument("--max-episodes", type=int, default=5,
                       help="Max episodes to visualize")
    parser.add_argument("--energy-percentile", type=int, default=50)
    parser.add_argument("--merge-gap", type=int, default=3)
    parser.add_argument("--arm", type=str, default="left",
                       choices=["left", "right", "both"],
                       help="Which arm to analyze")
    parser.add_argument("--video", action="store_true",
                       help="Generate animated video instead of static images")
    parser.add_argument("--video-fps", type=int, default=10,
                       help="Video frame rate (default: 10)")

    args = parser.parse_args()

    config = SweepSegmentConfig(
        verbose=False,
        energy_percentile=args.energy_percentile,
        merge_gap=args.merge_gap,
        active_arm=args.arm
    )

    if args.video:
        visualize_dataset_video(
            dataset_path=args.input,
            output_dir=args.output,
            max_episodes=args.max_episodes,
            config=config,
            video_fps=args.video_fps
        )
    else:
        visualize_dataset(
            dataset_path=args.input,
            output_dir=args.output,
            max_episodes=args.max_episodes,
            config=config
        )
