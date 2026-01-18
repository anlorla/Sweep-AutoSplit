"""
视觉质检模块

按规范实现 ROI 帧差检测，用于过滤"空挥/扫偏"的假阳性

规范 7.1：ROI 帧差能量
m(t) = (1/|Ω|) * Σ_{u∈Ω} ||I_t(u) - I_{t-1}(u)||_1

规范 7.2：背景集合
B = {t | t ∉ ∪_k [c_k^start, c_k^end]}  (low regions 的补集)
m_bg = {m(t) | t ∈ B}

规范 7.3：视觉活动度与阈值
M_vis = mean(m[a:b])
med_bg = median(m_bg)
MAD_bg = median(|m_bg - med_bg|)
过滤条件：M_vis >= med_bg + 3 * MAD_bg

不满足则丢弃该候选段
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: cv2 not available, visual checking will be disabled")


@dataclass
class ROI:
    """感兴趣区域"""
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height

    def to_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    @classmethod
    def from_center(cls, cx: int, cy: int, w: int, h: int) -> "ROI":
        """从中心点和尺寸创建 ROI"""
        return cls(
            x1=cx - w // 2,
            y1=cy - h // 2,
            x2=cx + w // 2,
            y2=cy + h // 2
        )

    @classmethod
    def from_relative(cls, frame_w: int, frame_h: int,
                     rel_x: float, rel_y: float,
                     rel_w: float, rel_h: float) -> "ROI":
        """从相对坐标创建 ROI（用于不同分辨率）"""
        return cls(
            x1=int(rel_x * frame_w),
            y1=int(rel_y * frame_h),
            x2=int((rel_x + rel_w) * frame_w),
            y2=int((rel_y + rel_h) * frame_h)
        )


@dataclass
class VisualCheckResult:
    """视觉质检结果"""
    sweep_idx: int
    is_valid: bool
    M_vis: float                # 候选段视觉活动度 mean(m[a:b])
    med_bg: float               # 背景中位数 median(m_bg)
    MAD_bg: float               # 背景 MAD median(|m_bg - med_bg|)
    threshold: float            # 过滤阈值 med_bg + 3 * MAD_bg
    message: str = ""

    @property
    def margin(self) -> float:
        """活动度超过阈值的量"""
        return self.M_vis - self.threshold

    def __repr__(self) -> str:
        status = "✓" if self.is_valid else "✗"
        return (
            f"[{status}] Sweep {self.sweep_idx}: "
            f"M_vis={self.M_vis:.2f}, threshold={self.threshold:.2f}, "
            f"margin={self.margin:.2f}"
        )


class VisualQualityChecker:
    """
    视觉质检器（按规范 7 实现）

    使用 ROI 帧差能量和鲁棒统计量来验证 sweep 是否真实发生

    过滤条件：M_vis >= med_bg + 3 * MAD_bg
    """

    def __init__(
        self,
        roi: Optional[ROI] = None,
        k_mad: float = 3.0,
        grayscale: bool = True,
        verbose: bool = False
    ):
        """
        初始化质检器

        Args:
            roi: 感兴趣区域（默认使用全图中心区域）
            k_mad: MAD 系数，默认 3.0（即 med_bg + 3 * MAD_bg）
            grayscale: 是否转换为灰度图计算
            verbose: 是否输出详细信息
        """
        self.roi = roi
        self.k_mad = k_mad
        self.grayscale = grayscale
        self.verbose = verbose

        # 缓存
        self._video_cap = None
        self._current_video_path = None
        self._frame_energies_cache = {}  # 缓存帧差能量

    def _get_video_capture(self, video_path: str):
        """获取视频捕获对象（带缓存）"""
        if not CV2_AVAILABLE:
            return None

        if self._current_video_path != video_path:
            if self._video_cap is not None:
                self._video_cap.release()
            self._video_cap = cv2.VideoCapture(video_path)
            self._current_video_path = video_path
            self._frame_energies_cache = {}  # 清空缓存

        return self._video_cap

    def _get_frame(self, cap, frame_idx: int) -> Optional[np.ndarray]:
        """获取指定帧"""
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            return None

        if self.grayscale and len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        return frame

    def _get_roi_region(self, frame: np.ndarray, roi: ROI) -> np.ndarray:
        """提取 ROI 区域"""
        if len(frame.shape) == 2:
            return frame[roi.y1:roi.y2, roi.x1:roi.x2]
        else:
            return frame[roi.y1:roi.y2, roi.x1:roi.x2, :]

    def _auto_detect_roi(self, cap) -> ROI:
        """自动检测 ROI（使用图像中心 50% 区域）"""
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        if ret:
            h, w = frame.shape[:2]
            return ROI.from_relative(w, h, 0.25, 0.25, 0.5, 0.5)
        else:
            # 默认值
            return ROI(x1=160, y1=120, x2=480, y2=360)

    def compute_all_frame_energies(
        self,
        video_path: str,
        total_frames: int,
        roi: Optional[ROI] = None
    ) -> np.ndarray:
        """
        计算整个视频的帧差能量序列 m(t)

        规范 7.1：m(t) = (1/|Ω|) * Σ_{u∈Ω} ||I_t(u) - I_{t-1}(u)||_1

        Args:
            video_path: 视频路径
            total_frames: 总帧数
            roi: 感兴趣区域

        Returns:
            能量序列 [total_frames-1]（第一帧无法计算差分）
        """
        # 检查缓存
        cache_key = (video_path, total_frames)
        if cache_key in self._frame_energies_cache:
            return self._frame_energies_cache[cache_key]

        if not CV2_AVAILABLE:
            return np.array([])

        cap = self._get_video_capture(video_path)
        if cap is None:
            return np.array([])

        roi = roi or self.roi
        if roi is None:
            roi = self._auto_detect_roi(cap)
            self.roi = roi  # 保存以供后续使用

        if self.verbose:
            print(f"[VisualChecker] ROI: ({roi.x1}, {roi.y1}) - ({roi.x2}, {roi.y2})")
            print(f"[VisualChecker] Computing frame energies for {total_frames} frames...")

        energies = []
        prev_roi_frame = None

        for frame_idx in range(total_frames):
            frame = self._get_frame(cap, frame_idx)
            if frame is None:
                # 如果帧读取失败，用 0 填充
                if prev_roi_frame is not None:
                    energies.append(0.0)
                continue

            roi_frame = self._get_roi_region(frame, roi).astype(np.float32)

            if prev_roi_frame is not None:
                # 规范 7.1：计算 L1 帧差能量
                diff = np.abs(roi_frame - prev_roi_frame)
                energy = np.mean(diff)  # (1/|Ω|) * Σ||...||_1
                energies.append(energy)

            prev_roi_frame = roi_frame

        result = np.array(energies)

        # 缓存结果
        self._frame_energies_cache[cache_key] = result

        if self.verbose and len(result) > 0:
            print(f"[VisualChecker] Energy range: [{result.min():.2f}, {result.max():.2f}]")
        elif self.verbose:
            print(f"[VisualChecker] Warning: No frames could be read from video")

        return result

    def compute_background_statistics(
        self,
        energies: np.ndarray,
        low_regions: List[Tuple[int, int]]
    ) -> Tuple[float, float]:
        """
        计算背景统计量

        规范 7.2：B = {t | t ∉ ∪_k [c_k^start, c_k^end]}
        规范 7.3：med_bg = median(m_bg), MAD_bg = median(|m_bg - med_bg|)

        Args:
            energies: 帧差能量序列 [N-1]
            low_regions: low regions 列表 [(start, end), ...]

        Returns:
            (med_bg, MAD_bg)
        """
        N = len(energies) + 1  # 总帧数

        # 构建 low region 掩码
        is_low_region = np.zeros(N, dtype=bool)
        for start, end in low_regions:
            # 确保边界有效
            start = max(0, start)
            end = min(N - 1, end)
            is_low_region[start:end + 1] = True

        # 背景帧：low regions 的补集
        # 注意：energies[t] 对应帧 t+1 和 t 的差分
        # 所以背景帧的能量索引需要调整
        background_indices = []
        for t in range(len(energies)):
            # t 对应 energies[t] = diff(frame[t+1], frame[t])
            # 只有当 t 和 t+1 都不在 low region 时才算背景
            if not is_low_region[t] and not is_low_region[t + 1]:
                background_indices.append(t)

        if len(background_indices) == 0:
            # 如果没有背景帧，使用所有能量的统计
            if self.verbose:
                print("[VisualChecker] Warning: No background frames found, using all energies")
            m_bg = energies
        else:
            m_bg = energies[background_indices]

        # 规范 7.3：鲁棒统计量
        med_bg = np.median(m_bg)
        MAD_bg = np.median(np.abs(m_bg - med_bg))

        if self.verbose:
            print(f"[VisualChecker] Background frames: {len(background_indices)}/{len(energies)}")
            print(f"[VisualChecker] med_bg={med_bg:.4f}, MAD_bg={MAD_bg:.4f}")
            print(f"[VisualChecker] Threshold: {med_bg + self.k_mad * MAD_bg:.4f}")

        return med_bg, MAD_bg

    def validate_sweep(
        self,
        energies: np.ndarray,
        P_t0: int,
        P_t1: int,
        sweep_idx: int,
        med_bg: float,
        MAD_bg: float
    ) -> VisualCheckResult:
        """
        验证单个 sweep 是否有效

        规范 7.3：M_vis >= med_bg + 3 * MAD_bg

        Args:
            energies: 帧差能量序列
            P_t0: Engage 开始帧
            P_t1: Stroke 结束帧
            sweep_idx: sweep 索引
            med_bg: 背景中位数
            MAD_bg: 背景 MAD

        Returns:
            VisualCheckResult
        """
        # 调整索引：energies[t] 对应 diff(frame[t+1], frame[t])
        # 所以 sweep 区间 [P_t0, P_t1] 对应 energies[P_t0-1:P_t1]
        # 但为简化，我们使用 [P_t0:P_t1]
        start_idx = max(0, P_t0)
        end_idx = min(len(energies), P_t1)

        if start_idx >= end_idx:
            return VisualCheckResult(
                sweep_idx=sweep_idx,
                is_valid=False,
                M_vis=0.0,
                med_bg=med_bg,
                MAD_bg=MAD_bg,
                threshold=med_bg + self.k_mad * MAD_bg,
                message="Invalid frame range"
            )

        # 规范 7.3：M_vis = mean(m[a:b])
        M_vis = np.mean(energies[start_idx:end_idx])

        # 阈值：med_bg + k * MAD_bg
        threshold = med_bg + self.k_mad * MAD_bg

        # 过滤条件
        is_valid = M_vis >= threshold

        message = ""
        if not is_valid:
            message = f"M_vis={M_vis:.2f} < threshold={threshold:.2f}"

        return VisualCheckResult(
            sweep_idx=sweep_idx,
            is_valid=is_valid,
            M_vis=M_vis,
            med_bg=med_bg,
            MAD_bg=MAD_bg,
            threshold=threshold,
            message=message
        )

    def validate_all_sweeps(
        self,
        video_path: str,
        keypoints: List,  # List[SweepKeypoint]
        total_frames: int,
        low_regions: Optional[List[Tuple[int, int]]] = None
    ) -> List[VisualCheckResult]:
        """
        验证所有 sweep

        Args:
            video_path: 视频路径
            keypoints: sweep 关键点列表
            total_frames: 总帧数
            low_regions: low regions 列表（如果不提供，使用 sweep 区间作为近似）

        Returns:
            验证结果列表
        """
        if not CV2_AVAILABLE:
            return [
                VisualCheckResult(
                    sweep_idx=kp.sweep_idx,
                    is_valid=True,
                    M_vis=0.0,
                    med_bg=0.0,
                    MAD_bg=0.0,
                    threshold=0.0,
                    message="cv2 not available"
                )
                for kp in keypoints
            ]

        # Step 1: 计算整个视频的帧差能量
        energies = self.compute_all_frame_energies(video_path, total_frames)

        if len(energies) == 0:
            # 视频无法读取时，跳过视觉检测（默认有效）
            if self.verbose:
                print(f"[VisualChecker] Warning: Cannot read video, skipping visual check")
            return [
                VisualCheckResult(
                    sweep_idx=kp.sweep_idx,
                    is_valid=True,  # 无法检测时默认有效
                    M_vis=0.0,
                    med_bg=0.0,
                    MAD_bg=0.0,
                    threshold=0.0,
                    message="Video unreadable, skipped"
                )
                for kp in keypoints
            ]

        # Step 2: 确定 low regions（背景的补集）
        # 如果没有提供 low_regions，使用 sweep 区间作为近似
        if low_regions is None:
            low_regions = [(kp.P_t0, kp.P_t1) for kp in keypoints]

        # Step 3: 计算背景统计量
        med_bg, MAD_bg = self.compute_background_statistics(energies, low_regions)

        # Step 4: 验证每个 sweep
        results = []
        for kp in keypoints:
            result = self.validate_sweep(
                energies=energies,
                P_t0=kp.P_t0,
                P_t1=kp.P_t1,
                sweep_idx=kp.sweep_idx,
                med_bg=med_bg,
                MAD_bg=MAD_bg
            )
            results.append(result)

            if self.verbose:
                print(f"  {result}")

        return results

    def release(self):
        """释放资源"""
        if self._video_cap is not None:
            self._video_cap.release()
            self._video_cap = None
            self._current_video_path = None
        self._frame_energies_cache = {}

    def __del__(self):
        # 安全检查，避免初始化失败时的 AttributeError
        if hasattr(self, '_video_cap'):
            self.release()


# ============================================================
# 便捷函数
# ============================================================

def create_default_roi_for_sweep(
    frame_width: int,
    frame_height: int,
    workspace_position: str = "center"
) -> ROI:
    """
    为 sweep 任务创建默认的 ROI

    Args:
        frame_width: 帧宽度
        frame_height: 帧高度
        workspace_position: 工作区位置 ("center", "left", "right", "bottom")

    Returns:
        ROI 对象
    """
    if workspace_position == "center":
        return ROI.from_relative(frame_width, frame_height, 0.25, 0.25, 0.5, 0.5)
    elif workspace_position == "left":
        return ROI.from_relative(frame_width, frame_height, 0.1, 0.25, 0.4, 0.5)
    elif workspace_position == "right":
        return ROI.from_relative(frame_width, frame_height, 0.5, 0.25, 0.4, 0.5)
    elif workspace_position == "bottom":
        return ROI.from_relative(frame_width, frame_height, 0.25, 0.4, 0.5, 0.5)
    else:
        return ROI.from_relative(frame_width, frame_height, 0.25, 0.25, 0.5, 0.5)


def print_visual_check_summary(results: List[VisualCheckResult]):
    """打印视觉质检摘要"""
    print("\n" + "=" * 60)
    print("视觉质检摘要 (Visual Quality Check Summary)")
    print("=" * 60)

    valid_count = sum(1 for r in results if r.is_valid)
    total_count = len(results)

    print(f"有效 sweep: {valid_count}/{total_count}")

    if results:
        avg_M_vis = np.mean([r.M_vis for r in results])
        avg_threshold = np.mean([r.threshold for r in results])
        print(f"平均活动度 M_vis: {avg_M_vis:.2f}")
        print(f"平均阈值: {avg_threshold:.2f}")

        if results[0].med_bg > 0:
            print(f"背景统计: med_bg={results[0].med_bg:.2f}, MAD_bg={results[0].MAD_bg:.2f}")

    print("\n详细结果:")
    for r in results:
        print(f"  {r}")

    # 列出被过滤的 sweep
    filtered = [r for r in results if not r.is_valid]
    if filtered:
        print(f"\n被过滤的 sweep ({len(filtered)}):")
        for r in filtered:
            print(f"  Sweep {r.sweep_idx}: {r.message}")

    print("=" * 60)
