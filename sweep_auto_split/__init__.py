"""
Sweep Auto Split - 基于 Pi/LeRobot 的 Sweep 动作自动切分工具

主要模块：
- config: 配置参数定义
- data_loader: LeRobot 2.1 数据加载
- kinematics: 前向运动学计算
- signal_processing: 信号处理与自适应阈值
- sweep_detector: Sweep 关键点检测
- segment_calculator: Segment 边界计算
- visual_checker: 视觉质检
- lerobot_exporter: LeRobot 2.1 格式导出
- diagnostics: 诊断统计
- visualization: 可视化工具
"""

from .config import (
    SweepSegmentConfig,
    SweepKeypoint,
    SegmentBoundary,
)

from .data_loader import (
    LeRobotDataLoader,
    EpisodeData,
)

from .kinematics import (
    PiperForwardKinematics,
    DualArmKinematics,
    compute_kinematics_from_ee_pose,
)

from .signal_processing import (
    smooth_signal,
    detect_low_regions_with_hysteresis,
    find_longest_high_speed_segment,
    compute_adaptive_thresholds,
    compute_adaptive_thresholds_robust,
    analyze_signal_distribution,
    AdaptiveThresholds,
    Region,
)

from .sweep_detector import SweepDetector

from .segment_calculator import (
    SegmentCalculator,
    print_segment_summary,
)

from .visual_checker import (
    VisualQualityChecker,
    ROI,
    VisualCheckResult,
    create_default_roi_for_sweep,
    print_visual_check_summary,
)

from .lerobot_exporter import (
    LeRobotSegmentExporter,
    ExportConfig,
    export_segmented_dataset,
)

from .diagnostics import (
    DiagnosticsCollector,
    DatasetDiagnostics,
    EpisodeDiagnostics,
    generate_paper_statistics,
)

__version__ = "0.2.0"
__all__ = [
    # Config
    "SweepSegmentConfig",
    "SweepKeypoint",
    "SegmentBoundary",
    # Data loading
    "LeRobotDataLoader",
    "EpisodeData",
    # Kinematics
    "PiperForwardKinematics",
    "DualArmKinematics",
    "compute_kinematics_from_ee_pose",
    # Signal processing
    "smooth_signal",
    "detect_low_regions_with_hysteresis",
    "find_longest_high_speed_segment",
    "compute_adaptive_thresholds",
    "compute_adaptive_thresholds_robust",
    "analyze_signal_distribution",
    "AdaptiveThresholds",
    "Region",
    # Detection
    "SweepDetector",
    # Calculation
    "SegmentCalculator",
    "print_segment_summary",
    # Visual check
    "VisualQualityChecker",
    "ROI",
    "VisualCheckResult",
    "create_default_roi_for_sweep",
    "print_visual_check_summary",
    # Export
    "LeRobotSegmentExporter",
    "ExportConfig",
    "export_segmented_dataset",
    # Diagnostics
    "DiagnosticsCollector",
    "DatasetDiagnostics",
    "EpisodeDiagnostics",
    "generate_paper_statistics",
]
