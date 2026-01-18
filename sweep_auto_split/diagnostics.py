"""
诊断统计模块

用于记录和输出处理过程中的诊断信息，便于调参和写论文

按 guidance.md 6.3 节建议统计：
- 每个 sweep 的 L_23
- |S_t| 的分布（多样性）
- 被视觉质检过滤的比例
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime

from .config import SweepSegmentConfig, SweepKeypoint, SegmentBoundary


@dataclass
class EpisodeDiagnostics:
    """单个 episode 的诊断信息"""
    episode_id: int
    episode_length: int

    # Keypoints
    total_keypoints: int = 0
    valid_keypoints: int = 0
    invalid_keypoints: int = 0
    L23_values: List[int] = field(default_factory=list)

    # Boundaries
    total_boundaries: int = 0
    valid_boundaries: int = 0
    invalid_boundaries: int = 0
    diversities: List[int] = field(default_factory=list)
    segment_lengths: List[int] = field(default_factory=list)

    # Visual check
    visual_check_passed: int = 0
    visual_check_failed: int = 0

    # Thresholds used
    z_on: Optional[float] = None
    z_off: Optional[float] = None
    v_xy_threshold: Optional[float] = None

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class DatasetDiagnostics:
    """整个数据集的诊断信息"""
    # 基本信息
    dataset_path: str = ""
    processed_at: str = ""
    config: Dict = field(default_factory=dict)

    # Episode 统计
    total_episodes: int = 0
    processed_episodes: int = 0

    # Keypoints 统计
    total_keypoints: int = 0
    valid_keypoints: int = 0
    L23_distribution: Dict = field(default_factory=dict)

    # Boundaries 统计
    total_boundaries: int = 0
    valid_boundaries: int = 0
    invalid_boundaries: int = 0
    diversity_distribution: Dict = field(default_factory=dict)
    segment_length_distribution: Dict = field(default_factory=dict)

    # Visual check 统计
    visual_check_total: int = 0
    visual_check_passed: int = 0
    visual_check_failed: int = 0

    # 输出统计
    total_output_frames: int = 0
    total_diversity: int = 0
    avg_diversity: float = 0.0

    # Per-episode 诊断
    episode_diagnostics: List[EpisodeDiagnostics] = field(default_factory=list)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d['episode_diagnostics'] = [ep.to_dict() for ep in self.episode_diagnostics]
        return d


class DiagnosticsCollector:
    """
    诊断信息收集器

    在处理过程中收集统计信息，最后生成报告
    """

    def __init__(self, config: SweepSegmentConfig, dataset_path: str = ""):
        """
        初始化收集器

        Args:
            config: 切分配置
            dataset_path: 数据集路径
        """
        self.config = config
        self.diagnostics = DatasetDiagnostics(
            dataset_path=dataset_path,
            processed_at=datetime.now().isoformat(),
            config={
                "H": config.H,
                "A_min": config.A_min,
                "R_min": config.R_min,
                "z_on": config.z_on,
                "z_off": config.z_off,
                "v_xy_threshold": config.v_xy_threshold,
                "L23_min": config.L23_min,
                "L23_max": config.L23_max,
            }
        )

        # 收集所有值用于分布计算
        self._all_L23 = []
        self._all_diversities = []
        self._all_segment_lengths = []

    def add_episode_result(
        self,
        episode_id: int,
        episode_length: int,
        keypoints: List[SweepKeypoint],
        boundaries: List[SegmentBoundary],
        visual_check_results: Optional[List] = None,
        thresholds: Optional[Dict] = None
    ):
        """
        添加单个 episode 的处理结果

        Args:
            episode_id: Episode ID
            episode_length: Episode 长度
            keypoints: 检测到的关键点
            boundaries: 计算的边界
            visual_check_results: 视觉质检结果（可选）
            thresholds: 使用的阈值（可选）
        """
        ep_diag = EpisodeDiagnostics(
            episode_id=episode_id,
            episode_length=episode_length
        )

        # Keypoints 统计
        ep_diag.total_keypoints = len(keypoints)
        ep_diag.valid_keypoints = sum(1 for kp in keypoints if kp.is_valid)
        ep_diag.invalid_keypoints = ep_diag.total_keypoints - ep_diag.valid_keypoints
        ep_diag.L23_values = [kp.L23 for kp in keypoints]

        # Boundaries 统计
        ep_diag.total_boundaries = len(boundaries)
        ep_diag.valid_boundaries = sum(1 for b in boundaries if b.is_valid)
        ep_diag.invalid_boundaries = ep_diag.total_boundaries - ep_diag.valid_boundaries
        ep_diag.diversities = [b.diversity for b in boundaries if b.is_valid]
        ep_diag.segment_lengths = [b.segment_length for b in boundaries if b.is_valid]

        # Visual check 统计
        if visual_check_results:
            ep_diag.visual_check_passed = sum(1 for r in visual_check_results if r.is_valid)
            ep_diag.visual_check_failed = len(visual_check_results) - ep_diag.visual_check_passed

        # 使用的阈值
        if thresholds:
            ep_diag.z_on = thresholds.get("z_on")
            ep_diag.z_off = thresholds.get("z_off")
            ep_diag.v_xy_threshold = thresholds.get("v_xy_threshold")

        # 添加到全局统计
        self.diagnostics.episode_diagnostics.append(ep_diag)
        self._all_L23.extend(ep_diag.L23_values)
        self._all_diversities.extend(ep_diag.diversities)
        self._all_segment_lengths.extend(ep_diag.segment_lengths)

    def finalize(self) -> DatasetDiagnostics:
        """
        完成统计，计算分布和汇总信息

        Returns:
            完整的诊断信息
        """
        diag = self.diagnostics

        # 汇总 episode 统计
        diag.total_episodes = len(diag.episode_diagnostics)
        diag.processed_episodes = diag.total_episodes

        # 汇总 keypoints
        diag.total_keypoints = sum(ep.total_keypoints for ep in diag.episode_diagnostics)
        diag.valid_keypoints = sum(ep.valid_keypoints for ep in diag.episode_diagnostics)

        # 汇总 boundaries
        diag.total_boundaries = sum(ep.total_boundaries for ep in diag.episode_diagnostics)
        diag.valid_boundaries = sum(ep.valid_boundaries for ep in diag.episode_diagnostics)
        diag.invalid_boundaries = diag.total_boundaries - diag.valid_boundaries

        # 汇总 visual check
        diag.visual_check_total = sum(
            ep.visual_check_passed + ep.visual_check_failed
            for ep in diag.episode_diagnostics
        )
        diag.visual_check_passed = sum(ep.visual_check_passed for ep in diag.episode_diagnostics)
        diag.visual_check_failed = sum(ep.visual_check_failed for ep in diag.episode_diagnostics)

        # 计算分布
        diag.L23_distribution = self._compute_distribution(self._all_L23, bins=range(0, 50, 2))
        diag.diversity_distribution = self._compute_distribution(self._all_diversities, bins=range(0, 50, 2))
        diag.segment_length_distribution = self._compute_distribution(
            self._all_segment_lengths, bins=range(0, 100, 5)
        )

        # 输出统计
        diag.total_output_frames = sum(self._all_segment_lengths)
        diag.total_diversity = sum(self._all_diversities)
        diag.avg_diversity = (
            np.mean(self._all_diversities) if self._all_diversities else 0.0
        )

        return diag

    def _compute_distribution(self, values: List, bins) -> Dict:
        """计算分布"""
        if not values:
            return {"bins": list(bins), "counts": [0] * (len(list(bins)) - 1)}

        counts, bin_edges = np.histogram(values, bins=bins)
        return {
            "bins": bin_edges.tolist(),
            "counts": counts.tolist(),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "median": float(np.median(values)),
        }

    def save_report(self, output_path: Path, format: str = "json"):
        """
        保存诊断报告

        Args:
            output_path: 输出路径
            format: 输出格式 ("json" | "txt")
        """
        diag = self.finalize()

        if format == "json":
            with open(output_path, 'w') as f:
                json.dump(diag.to_dict(), f, indent=2)
        elif format == "txt":
            self._save_text_report(output_path, diag)
        else:
            raise ValueError(f"Unknown format: {format}")

    def _save_text_report(self, output_path: Path, diag: DatasetDiagnostics):
        """保存文本格式报告"""
        lines = []
        lines.append("=" * 70)
        lines.append("SWEEP AUTO SPLIT - DIAGNOSTICS REPORT")
        lines.append("=" * 70)
        lines.append(f"Generated at: {diag.processed_at}")
        lines.append(f"Dataset: {diag.dataset_path}")
        lines.append("")

        lines.append("CONFIGURATION")
        lines.append("-" * 40)
        for key, value in diag.config.items():
            lines.append(f"  {key}: {value}")
        lines.append("")

        lines.append("SUMMARY")
        lines.append("-" * 40)
        lines.append(f"  Episodes processed: {diag.processed_episodes}")
        lines.append(f"  Total keypoints: {diag.valid_keypoints}/{diag.total_keypoints} valid")
        lines.append(f"  Total segments: {diag.valid_boundaries}/{diag.total_boundaries} valid")
        lines.append(f"  Total output frames: {diag.total_output_frames}")
        lines.append(f"  Total diversity: {diag.total_diversity}")
        lines.append(f"  Avg diversity per segment: {diag.avg_diversity:.2f}")
        lines.append("")

        if diag.visual_check_total > 0:
            lines.append("VISUAL QUALITY CHECK")
            lines.append("-" * 40)
            pass_rate = diag.visual_check_passed / diag.visual_check_total * 100
            lines.append(f"  Passed: {diag.visual_check_passed}/{diag.visual_check_total} ({pass_rate:.1f}%)")
            lines.append(f"  Failed: {diag.visual_check_failed}/{diag.visual_check_total}")
            lines.append("")

        lines.append("L23 DISTRIBUTION (Engage+Stroke length)")
        lines.append("-" * 40)
        if diag.L23_distribution:
            lines.append(f"  Min: {diag.L23_distribution.get('min', 'N/A')}")
            lines.append(f"  Max: {diag.L23_distribution.get('max', 'N/A')}")
            lines.append(f"  Mean: {diag.L23_distribution.get('mean', 'N/A'):.2f}")
            lines.append(f"  Median: {diag.L23_distribution.get('median', 'N/A'):.2f}")
        lines.append("")

        lines.append("DIVERSITY DISTRIBUTION")
        lines.append("-" * 40)
        if diag.diversity_distribution:
            lines.append(f"  Min: {diag.diversity_distribution.get('min', 'N/A')}")
            lines.append(f"  Max: {diag.diversity_distribution.get('max', 'N/A')}")
            lines.append(f"  Mean: {diag.diversity_distribution.get('mean', 'N/A'):.2f}")
            lines.append(f"  Median: {diag.diversity_distribution.get('median', 'N/A'):.2f}")
        lines.append("")

        lines.append("PER-EPISODE DETAILS")
        lines.append("-" * 40)
        for ep in diag.episode_diagnostics:
            lines.append(f"  Episode {ep.episode_id}:")
            lines.append(f"    Length: {ep.episode_length} frames")
            lines.append(f"    Keypoints: {ep.valid_keypoints}/{ep.total_keypoints} valid")
            lines.append(f"    Segments: {ep.valid_boundaries}/{ep.total_boundaries} valid")
            if ep.diversities:
                lines.append(f"    Diversities: {ep.diversities}")
            if ep.visual_check_passed + ep.visual_check_failed > 0:
                lines.append(f"    Visual check: {ep.visual_check_passed} passed, {ep.visual_check_failed} failed")
        lines.append("")

        lines.append("=" * 70)

        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))

    def print_summary(self):
        """打印摘要到控制台"""
        diag = self.finalize()

        print("\n" + "=" * 60)
        print("SWEEP AUTO SPLIT - SUMMARY")
        print("=" * 60)

        print(f"\nDataset: {diag.dataset_path}")
        print(f"Processed: {diag.processed_at}")

        print(f"\nConfiguration:")
        print(f"  H={diag.config['H']}, A_min={diag.config['A_min']}, R_min={diag.config['R_min']}")
        print(f"  L23 range: [{diag.config['L23_min']}, {diag.config['L23_max']}]")

        print(f"\nResults:")
        print(f"  Episodes: {diag.processed_episodes}")
        print(f"  Keypoints: {diag.valid_keypoints}/{diag.total_keypoints} valid "
              f"({diag.valid_keypoints/max(1,diag.total_keypoints)*100:.1f}%)")
        print(f"  Segments: {diag.valid_boundaries}/{diag.total_boundaries} valid "
              f"({diag.valid_boundaries/max(1,diag.total_boundaries)*100:.1f}%)")

        print(f"\nOutput:")
        print(f"  Total frames: {diag.total_output_frames}")
        print(f"  Total diversity: {diag.total_diversity}")
        print(f"  Avg diversity: {diag.avg_diversity:.2f}")

        if diag.visual_check_total > 0:
            pass_rate = diag.visual_check_passed / diag.visual_check_total * 100
            print(f"\nVisual Quality Check:")
            print(f"  Pass rate: {pass_rate:.1f}% ({diag.visual_check_passed}/{diag.visual_check_total})")

        if diag.L23_distribution and 'mean' in diag.L23_distribution:
            print(f"\nL23 (Engage+Stroke):")
            print(f"  Mean: {diag.L23_distribution['mean']:.1f}")
            print(f"  Range: [{diag.L23_distribution['min']}, {diag.L23_distribution['max']}]")

        if diag.diversity_distribution and 'mean' in diag.diversity_distribution:
            print(f"\nDiversity:")
            print(f"  Mean: {diag.diversity_distribution['mean']:.1f}")
            print(f"  Range: [{diag.diversity_distribution['min']}, {diag.diversity_distribution['max']}]")

        print("=" * 60)


def generate_paper_statistics(diagnostics: DatasetDiagnostics) -> Dict:
    """
    生成论文中可用的统计数据

    Args:
        diagnostics: 诊断信息

    Returns:
        论文统计字典
    """
    stats = {
        "dataset": {
            "total_episodes": diagnostics.total_episodes,
            "total_sweeps_detected": diagnostics.total_keypoints,
            "valid_sweeps": diagnostics.valid_keypoints,
            "valid_segments": diagnostics.valid_boundaries,
        },
        "quality": {
            "sweep_detection_rate": diagnostics.valid_keypoints / max(1, diagnostics.total_keypoints),
            "segment_validity_rate": diagnostics.valid_boundaries / max(1, diagnostics.total_boundaries),
        },
        "output": {
            "total_frames": diagnostics.total_output_frames,
            "total_diversity": diagnostics.total_diversity,
            "avg_diversity_per_segment": diagnostics.avg_diversity,
        }
    }

    if diagnostics.visual_check_total > 0:
        stats["visual_check"] = {
            "total_checked": diagnostics.visual_check_total,
            "passed": diagnostics.visual_check_passed,
            "pass_rate": diagnostics.visual_check_passed / diagnostics.visual_check_total,
        }

    if diagnostics.L23_distribution and 'mean' in diagnostics.L23_distribution:
        stats["L23"] = {
            "mean": diagnostics.L23_distribution['mean'],
            "std": diagnostics.L23_distribution['std'],
            "min": diagnostics.L23_distribution['min'],
            "max": diagnostics.L23_distribution['max'],
        }

    if diagnostics.diversity_distribution and 'mean' in diagnostics.diversity_distribution:
        stats["diversity"] = {
            "mean": diagnostics.diversity_distribution['mean'],
            "std": diagnostics.diversity_distribution['std'],
            "min": diagnostics.diversity_distribution['min'],
            "max": diagnostics.diversity_distribution['max'],
        }

    return stats
