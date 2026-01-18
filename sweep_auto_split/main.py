#!/usr/bin/env python3
"""
Sweep Auto Split - 主程序

基于 Pi/LeRobot 的 Sweep 动作自动切分工具

使用方法:
    # 分析模式（不导出）
    python -m sweep_auto_split.main --input /path/to/dataset --analyze

    # 完整处理（分析 + 导出）
    python -m sweep_auto_split.main --input /path/to/dataset --output /path/to/output

    # 使用自适应阈值
    python -m sweep_auto_split.main --input /path/to/dataset --adaptive-thresholds

    # 启用视觉质检
    python -m sweep_auto_split.main --input /path/to/dataset --visual-check
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

from .config import SweepSegmentConfig, SweepKeypoint, SegmentBoundary
from .data_loader import LeRobotDataLoader, EpisodeData
from .sweep_detector import SweepDetector
from .action_detector import ActionBasedSweepDetector, analyze_action_data
from .segment_calculator import SegmentCalculator, print_segment_summary
from .signal_processing import (
    compute_adaptive_thresholds,
    analyze_signal_distribution,
)
from .kinematics import (
    DualArmKinematics,
    KinematicsConfig,
    create_default_kinematics_config,
    compute_full_kinematics_from_ee_pose,
)
from .visual_checker import VisualQualityChecker, create_default_roi_for_sweep
from .lerobot_exporter import export_segmented_dataset, ExportConfig, LeRobotSegmentExporter
from .diagnostics import DiagnosticsCollector, generate_paper_statistics


def process_episode_action_based(
    episode_data: EpisodeData,
    config: SweepSegmentConfig,
    action_detector: ActionBasedSweepDetector,
    calculator: SegmentCalculator,
    visual_checker: Optional[VisualQualityChecker] = None,
) -> tuple:
    """
    使用 action-based 检测处理单个 episode

    Args:
        episode_data: episode 数据
        config: 配置
        action_detector: action-based sweep 检测器
        calculator: 边界计算器
        visual_checker: 视觉质检器（可选）

    Returns:
        (keypoints, boundaries, visual_results, diagnostics)
    """
    action_trajectory = episode_data.action_trajectory
    state_trajectory = episode_data.state_trajectory

    # Step 1: 使用 action-based 检测
    keypoints, diagnostics = action_detector.detect_keypoints(
        action_trajectory, state_trajectory
    )

    # Step 2: 计算 segment 边界
    episode_length = len(action_trajectory)
    boundaries = calculator.calculate_boundaries(keypoints, episode_length)

    # Step 3: 视觉质检（可选）
    visual_results = None
    if visual_checker and episode_data.video_paths:
        # 尝试使用 main 相机，如果没有则尝试其他相机
        video_path = None
        for cam_name in ["main", "cam_main", "observation.images.main"]:
            video_path = episode_data.video_paths.get(cam_name)
            if video_path:
                break

        # 如果都没有，使用第一个可用的相机
        if not video_path and episode_data.video_paths:
            video_path = list(episode_data.video_paths.values())[0]

        if video_path:
            visual_results = visual_checker.validate_all_sweeps(
                video_path, keypoints, episode_length
            )

            # 根据视觉质检结果更新 keypoints 的有效性
            for kp, vr in zip(keypoints, visual_results):
                if not vr.is_valid:
                    kp.is_valid = False

            # 重新计算边界
            boundaries = calculator.calculate_boundaries(keypoints, episode_length)

    return keypoints, boundaries, visual_results, diagnostics


def process_episode_fk(
    episode_data: EpisodeData,
    config: SweepSegmentConfig,
    detector: SweepDetector,
    calculator: SegmentCalculator,
    visual_checker: Optional[VisualQualityChecker] = None,
) -> tuple:
    """
    使用 FK-based 检测处理单个 episode

    改进的流程:
    1. 计算刷尖到桌面距离 d(t) 和速度 v_xy(t), v_z(t)
    2. 自适应阈值检测高速区间
    3. 验证低位条件 + v_z 过滤
    4. 四约束计算 → (T_t0, T_t1)

    Args:
        episode_data: episode 数据
        config: 配置
        detector: sweep 检测器（已包含运动学配置）
        calculator: 边界计算器
        visual_checker: 视觉质检器（可选）

    Returns:
        (keypoints, boundaries, visual_results, diagnostic_data)
    """
    state_trajectory = episode_data.state_trajectory
    ee_pose_trajectory = episode_data.ee_pose_trajectory

    # Step 1: 使用 SweepDetector 检测关键点（内部已处理运动学计算）
    keypoints = detector.detect_keypoints(state_trajectory, ee_pose_trajectory)

    # Step 2: 计算 segment 边界 (T_t0, T_t1) - 四约束
    episode_length = len(state_trajectory)
    boundaries = calculator.calculate_boundaries(keypoints, episode_length)

    # Step 3: 视觉质检（可选）
    visual_results = None
    if visual_checker and episode_data.video_paths:
        video_path = None
        for cam_name in ["main", "cam_main", "observation.images.main"]:
            video_path = episode_data.video_paths.get(cam_name)
            if video_path:
                break
        if not video_path and episode_data.video_paths:
            video_path = list(episode_data.video_paths.values())[0]

        if video_path:
            visual_results = visual_checker.validate_all_sweeps(
                video_path, keypoints, episode_length
            )
            for kp, vr in zip(keypoints, visual_results):
                if not vr.is_valid:
                    kp.is_valid = False
            boundaries = calculator.calculate_boundaries(keypoints, episode_length)

    # 获取诊断数据（包含运动学信息和阈值）
    diagnostic_data = detector.get_diagnostic_data(state_trajectory, ee_pose_trajectory)

    return keypoints, boundaries, visual_results, diagnostic_data


def process_dataset(
    dataset_path: str,
    config: SweepSegmentConfig,
    max_episodes: Optional[int] = None,
    enable_visual_check: bool = False,
    detection_method: str = "fk",  # "fk" (recommended) or "action"
) -> tuple:
    """
    处理整个数据集

    Args:
        dataset_path: 数据集路径
        config: 配置
        max_episodes: 最大处理 episode 数
        enable_visual_check: 是否启用视觉质检
        detection_method: 检测方法，"fk" (基于低位阈值+v_xy) 或 "action" (基于能量)

    Returns:
        (results, data_loader, diagnostics_collector)
    """
    # 加载数据集
    if config.verbose:
        print(f"Loading dataset: {dataset_path}")

    data_loader = LeRobotDataLoader(dataset_path)

    if config.verbose:
        print(f"  Total episodes: {data_loader.total_episodes}")
        print(f"  Total frames: {data_loader.total_frames}")
        print(f"  FPS: {data_loader.fps}")
        print(f"  Detection method: {detection_method}")

    # 初始化组件
    calculator = SegmentCalculator(config)
    diagnostics = DiagnosticsCollector(config, dataset_path)

    # 根据检测方法初始化检测器
    if detection_method == "action":
        action_detector = ActionBasedSweepDetector(config)
        fk_detector = None
    else:  # fk (default, recommended)
        fk_detector = SweepDetector(config)
        action_detector = None

    # 初始化视觉质检器
    visual_checker = None
    if enable_visual_check:
        visual_checker = VisualQualityChecker(
            k_mad=3.0,  # 阈值系数：med_bg + 3 * MAD_bg
            verbose=config.verbose
        )

    # 处理每个 episode
    results = {}
    num_episodes = data_loader.total_episodes
    if max_episodes is not None:
        num_episodes = min(num_episodes, max_episodes)

    episode_ids = data_loader.get_episode_list()[:num_episodes]

    for i, ep_id in enumerate(episode_ids):
        if config.verbose:
            print(f"\n{'=' * 40}")
            print(f"Processing Episode {i + 1}/{num_episodes} (ID: {ep_id})")
            print(f"{'=' * 40}")

        # 加载 episode 数据
        episode_data = data_loader.load_episode(ep_id)

        # 处理 episode
        if detection_method == "action":
            keypoints, boundaries, visual_results, action_diagnostics = process_episode_action_based(
                episode_data=episode_data,
                config=config,
                action_detector=action_detector,
                calculator=calculator,
                visual_checker=visual_checker,
            )
            thresholds = None
            if action_diagnostics and "thresholds" in action_diagnostics:
                thresholds = {
                    "energy_threshold": action_diagnostics["thresholds"].energy_threshold,
                }
        else:  # fk
            keypoints, boundaries, visual_results, diagnostic_data = process_episode_fk(
                episode_data=episode_data,
                config=config,
                detector=fk_detector,
                calculator=calculator,
                visual_checker=visual_checker,
            )
            thresholds = None
            if diagnostic_data and "config" in diagnostic_data:
                thresholds = {
                    "v_xy_threshold": diagnostic_data["config"]["v_xy_threshold"],
                    "tool_tip_offset": diagnostic_data["config"]["tool_tip_offset"],
                    "table_z": diagnostic_data["config"]["table_z"],
                    "fps": diagnostic_data["config"]["fps"],
                }

        results[ep_id] = (keypoints, boundaries)

        diagnostics.add_episode_result(
            episode_id=ep_id,
            episode_length=episode_data.length,
            keypoints=keypoints,
            boundaries=boundaries,
            visual_check_results=visual_results,
            thresholds=thresholds,
        )

    # 清理资源
    if visual_checker:
        visual_checker.release()

    return results, data_loader, diagnostics


def main():
    parser = argparse.ArgumentParser(
        description="Sweep Auto Split - 基于 Pi/LeRobot 的 Sweep 动作自动切分工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 分析数据集 (默认使用 action-based 检测)
  python -m sweep_auto_split.main --input /path/to/dataset --analyze

  # 使用 FK-based 检测
  python -m sweep_auto_split.main --input /path/to/dataset --detection-method fk --analyze

  # 使用自适应阈值 (仅 FK-based 检测支持)
  python -m sweep_auto_split.main --input /path/to/dataset --detection-method fk --adaptive-thresholds

  # 完整处理并导出
  python -m sweep_auto_split.main --input /path/to/dataset --output /path/to/output

  # 启用视觉质检
  python -m sweep_auto_split.main --input /path/to/dataset --visual-check
        """
    )

    # 数据集参数
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="输入 LeRobot 数据集路径"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="输出数据集路径（不指定则只分析不导出）"
    )

    # 模式参数
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="只分析不导出"
    )
    parser.add_argument(
        "--adaptive-thresholds",
        action="store_true",
        help="使用自适应阈值（百分位数方法）"
    )
    parser.add_argument(
        "--visual-check",
        action="store_true",
        help="启用视觉质检（ROI 帧差检测）"
    )

    # 核心参数
    parser.add_argument("--H", type=int, default=30, help="action horizon (default: 30)")
    parser.add_argument("--A-min", type=int, default=2, help="Approach 最少帧数 (default: 2)")
    parser.add_argument("--R-min", type=int, default=2, help="Retreat 最少帧数 (default: 2)")

    # 检测参数
    parser.add_argument("--z-on", type=float, default=0.05, help="低位区进入阈值")
    parser.add_argument("--z-off", type=float, default=0.06, help="低位区退出阈值")
    parser.add_argument("--v-xy-threshold", type=float, default=0.02, help="水平速度阈值")
    parser.add_argument("--smoothing-window", type=int, default=7, help="平滑窗口大小")

    # 质量过滤
    parser.add_argument("--L23-min", type=int, default=15, help="Engage+Stroke 最小长度")
    parser.add_argument("--L23-max", type=int, default=28, help="Engage+Stroke 最大长度")

    # Action-based 检测参数（用于调节准确性）
    parser.add_argument(
        "--energy-percentile",
        type=int,
        default=60,
        help="能量阈值百分位数 (default: 60)。较低的值会捕获更多运动，防止sweep过早截断"
    )
    parser.add_argument(
        "--merge-gap",
        type=int,
        default=2,
        help="合并间隔帧数 (default: 2)。较小的值可以防止两个独立的sweep被合并成一个"
    )

    # 其他参数
    parser.add_argument(
        "--arm",
        type=str,
        default="left",
        choices=["left", "right", "both"],
        help="使用哪只手臂检测 (default: left)"
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="最大处理 episode 数（用于测试）"
    )
    parser.add_argument(
        "--task-prefix",
        type=str,
        default="sweep",
        help="任务前缀 (default: sweep)"
    )
    parser.add_argument(
        "--diagnostics-output",
        type=str,
        default=None,
        help="诊断报告输出路径"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="详细输出"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="生成可视化"
    )
    parser.add_argument(
        "--detection-method",
        type=str,
        default="action",
        choices=["action", "fk"],
        help="检测方法: action (基于action能量) 或 fk (基于正运动学) (default: action)"
    )
    parser.add_argument(
        "--export-mask",
        action="store_true",
        default=True,
        help="导出 sweep mask 作为第四观测 (default: True)"
    )
    parser.add_argument(
        "--no-export-mask",
        dest="export_mask",
        action="store_false",
        help="不导出 sweep mask"
    )
    parser.add_argument(
        "--mask-method",
        type=str,
        default="hsv",
        choices=["hsv", "sam3"],
        help="mask 分割方法: hsv (颜色分割) 或 sam3 (SAM3模型) (default: hsv)"
    )

    args = parser.parse_args()

    # 创建配置
    config = SweepSegmentConfig(
        H=args.H,
        A_min=args.A_min,
        R_min=args.R_min,
        z_on=args.z_on,
        z_off=args.z_off,
        v_xy_threshold=args.v_xy_threshold,
        smoothing_window=args.smoothing_window,
        L23_min=args.L23_min,
        L23_max=args.L23_max,
        active_arm=args.arm,
        verbose=args.verbose,
        visualize=args.visualize,
        # Action-based 检测参数
        energy_percentile=args.energy_percentile,
        merge_gap=args.merge_gap,
    )

    # 处理数据集
    results, data_loader, diagnostics = process_dataset(
        dataset_path=args.input,
        config=config,
        max_episodes=args.max_episodes,
        enable_visual_check=args.visual_check,
        detection_method=args.detection_method,
    )

    # 打印摘要
    diagnostics.print_summary()

    # 保存诊断报告
    if args.diagnostics_output:
        output_path = Path(args.diagnostics_output)
        if output_path.suffix == '.json':
            diagnostics.save_report(output_path, format='json')
        else:
            diagnostics.save_report(output_path, format='txt')
        print(f"\nDiagnostics saved to: {output_path}")

    # 生成可视化
    if args.visualize:
        try:
            from .visualization import plot_multi_episode_summary
            viz_path = Path(args.input).parent / "sweep_split_summary.png"
            plot_multi_episode_summary(results, config, save_path=viz_path)
            print(f"\nVisualization saved to: {viz_path}")
        except ImportError:
            print("Warning: matplotlib not available, skipping visualization")

    # 导出（如果指定了输出路径且不是纯分析模式）
    if args.output and not args.analyze:
        print(f"\nExporting to: {args.output}")

        # 收集所有边界
        all_boundaries = {ep_idx: boundaries for ep_idx, (_, boundaries) in results.items()}

        # 导出
        export_stats = export_segmented_dataset(
            source_dataset_path=args.input,
            output_path=args.output,
            all_boundaries=all_boundaries,
            config=config,
            task_prefix=args.task_prefix,
            export_mask=args.export_mask,
            mask_method=args.mask_method,
        )

        print(f"\nExport complete:")
        print(f"  Episodes: {export_stats['total_episodes']}")
        print(f"  Frames: {export_stats['total_frames']}")
        print(f"  Tasks: {export_stats['total_tasks']}")

    # 生成论文统计（如果需要）
    if args.verbose:
        paper_stats = generate_paper_statistics(diagnostics.finalize())
        print("\n" + "=" * 60)
        print("PAPER STATISTICS")
        print("=" * 60)
        import json
        print(json.dumps(paper_stats, indent=2))


if __name__ == "__main__":
    main()
