"""
配置参数定义

根据 pi_sweep_segment_recipe.md 文档中的推荐参数
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List


@dataclass
class SweepSegmentConfig:
    """Sweep 切分配置参数"""

    # ============================================================
    # 核心参数（来自文档 Section 7）
    # ============================================================
    H: int = 30              # action horizon（action chunk 长度）
    A_min: int = 2           # 窗口左侧 Approach 最少帧数
    R_min: int = 2           # 窗口右侧 Retreat 最少帧数

    # ============================================================
    # 检测参数（来自文档 Section 6, Step 2）
    # ============================================================
    # 低位区检测（滞回阈值）- 刷尖到桌面距离
    # 进入低位区：d(t) < z_on
    # 退出低位区：d(t) > z_off
    z_on: float = 0.05       # 低位区进入阈值（单位：米，默认 5cm）
    z_off: float = 0.10      # 低位区退出阈值（滞回，默认 10cm）

    # 高速段检测
    v_xy_threshold: float = 0.02  # 水平速度阈值（单位：米/帧）

    # 平滑参数
    smoothing_window: int = 7     # 平滑窗口大小（5-9 帧，文档建议）

    # 低位区持续帧数
    low_region_min_frames: int = 5  # 低位区最小持续帧数（文档建议 2-3）

    # ============================================================
    # Action-based 检测参数
    # ============================================================
    # 能量百分位数阈值（用于自适应阈值计算）
    # 较低的值会捕获更多的运动，较高的值会过滤掉弱运动
    energy_percentile: int = 60  # 默认 60%，原来是 70%

    # 合并间隔：两个高能量区域之间小于此帧数会被合并
    # 较小的值可以防止两个独立的 sweep 被合并
    merge_gap: int = 2  # 默认 2 帧，原来是 5 帧

    # ============================================================
    # 质量过滤参数（来自文档 Section 6, Step 2）
    # ============================================================
    L23_min: int = 15        # Engage+Stroke 最小长度
    L23_max: int = 28        # Engage+Stroke 最大长度

    # ============================================================
    # 数据路径配置
    # ============================================================
    # LeRobot 数据集路径（输入）
    input_dataset_path: Optional[Path] = None

    # 输出数据集路径
    output_dataset_path: Optional[Path] = None

    # URDF 路径（用于 FK）
    urdf_path: Path = field(default_factory=lambda: Path(
        "/home/zeno-yifan/NPM-Project/NPM-Ros/piper_ros/src/piper_description/urdf/piper_description.urdf"
    ))

    # ============================================================
    # 双臂配置
    # ============================================================
    # 使用哪只手臂进行 sweep 检测
    # "left" | "right" | "both"（两只手臂分别检测，合并结果）
    active_arm: str = "left"

    # 关节维度
    joints_per_arm: int = 7  # 每只手臂的关节数（6 DOF + 1 gripper）

    # ============================================================
    # 数据集参数
    # ============================================================
    fps: int = 10            # 数据集帧率

    # ============================================================
    # 调试参数
    # ============================================================
    visualize: bool = False  # 是否生成可视化
    verbose: bool = True     # 是否打印详细日志

    def __post_init__(self):
        """验证参数有效性"""
        assert self.H > 0, "action horizon H 必须大于 0"
        assert self.A_min >= 0, "A_min 必须非负"
        assert self.R_min >= 0, "R_min 必须非负"
        assert self.z_on < self.z_off, "滞回阈值要求 z_on < z_off"
        assert self.L23_min < self.L23_max, "L23_min 必须小于 L23_max"
        assert self.active_arm in ["left", "right", "both"], "active_arm 必须是 left/right/both"


@dataclass
class SweepKeypoint:
    """单个 Sweep 的关键点信息"""
    sweep_idx: int           # sweep 索引 (t)
    P_t0: int                # Engage 开始帧
    P_t1: int                # Stroke 结束帧
    L23: int                 # Engage+Stroke 长度
    is_valid: bool = True    # 是否通过质量过滤

    @property
    def duration(self) -> int:
        """Engage+Stroke 持续帧数"""
        return self.P_t1 - self.P_t0 + 1


@dataclass
class SegmentBoundary:
    """Segment 边界信息"""
    sweep_idx: int           # 对应的 sweep 索引
    s_min: int               # 合格窗口起点下界
    s_max: int               # 合格窗口起点上界
    T_t0: int                # segment 起始帧
    T_t1: int                # segment 结束帧
    diversity: int           # 多样性 |S_t| = s_max - s_min + 1
    is_valid: bool = True    # 是否有效（s_min <= s_max）

    @property
    def segment_length(self) -> int:
        """segment 长度"""
        return self.T_t1 - self.T_t0 + 1
