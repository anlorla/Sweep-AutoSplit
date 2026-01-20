"""
LeRobot 2.1 数据集导出模块

将切分后的 segments 导出为新的 LeRobot 2.1 格式数据集

功能：
- 创建标准目录结构
- 导出 parquet 数据（支持批量）
- 导出视频片段
- 生成元数据文件
- 支持并行处理
"""

import json
import os
import shutil
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from tqdm import tqdm

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from .config import SegmentBoundary, SweepSegmentConfig
from .mask_generator import (
    SweepMaskGenerator,
    create_mask_generator,
    MaskConfig,
    ROIConfig,
    load_roi_config,
    get_roi_config,
    convert_video_to_h264,
    check_video_codec,
    _CONVERTED_VIDEO_CACHE,
)


def check_ffmpeg_available() -> bool:
    """检查 FFmpeg 是否可用"""
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


FFMPEG_AVAILABLE = check_ffmpeg_available()


def _get_readable_video_path(video_path: str, verbose: bool = False) -> str:
    """
    获取可读的视频路径，如果需要则转换为 H.264

    如果源视频是 AV1 等 OpenCV 无法解码的格式，自动转换为 H.264

    Args:
        video_path: 源视频路径
        verbose: 是否打印详细信息

    Returns:
        可读的视频路径（可能是原路径或转换后的 H.264 路径）
    """
    # 检查是否已有缓存的转换视频
    if video_path in _CONVERTED_VIDEO_CACHE:
        cached_path = _CONVERTED_VIDEO_CACHE[video_path]
        if os.path.exists(cached_path):
            if verbose:
                print(f"[_get_readable_video_path] Using cached H.264: {cached_path}")
            return cached_path

    # 检查视频编解码器
    codec = check_video_codec(video_path)
    if codec and codec.lower() not in ['h264', 'avc', 'avc1']:
        if verbose:
            print(f"[_get_readable_video_path] Source codec: {codec}, converting to H.264...")
        h264_path = convert_video_to_h264(video_path)
        if h264_path:
            if verbose:
                print(f"[_get_readable_video_path] Converted to: {h264_path}")
            return h264_path

    return video_path


@dataclass
class ExportConfig:
    """导出配置"""
    output_path: Path
    fps: int = 10
    chunks_size: int = 1000
    num_workers: int = 8
    export_workers: int = 1
    overwrite: bool = True
    export_videos: bool = True
    video_codec: str = "mp4v"
    verbose: bool = True
    # Mask export options
    export_mask: bool = True  # 是否导出 sweep mask
    # ROI config path (for mask filtering)
    roi_config_path: Optional[str] = None


@dataclass
class SegmentExportInfo:
    """单个 segment 导出信息"""
    source_episode_id: int
    new_episode_id: int
    boundary: SegmentBoundary
    task_string: str
    task_index: int


class LeRobotSegmentExporter:
    """
    LeRobot 2.1 格式数据集导出器

    将原始 episode 按 segment 边界切分，导出为新的数据集
    支持导出 sweep mask 作为额外观测
    """

    def __init__(self, config: ExportConfig, source_metadata: Dict):
        """
        初始化导出器

        Args:
            config: 导出配置
            source_metadata: 源数据集元数据
        """
        self.config = config
        self.source_metadata = source_metadata
        self.output_path = config.output_path

        # 加载 ROI 配置
        self.roi_config = load_roi_config(config.roi_config_path)

        # 初始化 mask 生成器
        if config.export_mask:
            self.mask_generator = create_mask_generator(roi_config=self.roi_config)
        else:
            self.mask_generator = None

    def create_directory_structure(self, data_loader=None):
        """创建 LeRobot 目录结构

        Args:
            data_loader: 数据加载器（用于检查实际存在的视频目录）
        """
        if self.output_path.exists():
            if self.config.overwrite:
                shutil.rmtree(self.output_path)
            else:
                raise FileExistsError(f"Output path already exists: {self.output_path}")

        # 创建基本目录
        self.output_path.mkdir(parents=True, exist_ok=True)
        (self.output_path / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
        (self.output_path / "meta").mkdir(parents=True, exist_ok=True)

        # 检测实际存在的视频目录
        if self.config.export_videos and data_loader is not None:
            # 获取第一个 episode 的视频路径来检测实际存在的相机
            episode_ids = data_loader.get_episode_list()
            if episode_ids:
                first_episode = data_loader.load_episode(episode_ids[0])
                actual_cameras = list(first_episode.video_paths.keys()) if first_episode.video_paths else []

                # 只为实际存在的相机创建目录
                for camera_name in actual_cameras:
                    video_key = f"observation.images.{camera_name}"
                    video_dir = self.output_path / "videos" / "chunk-000" / video_key
                    video_dir.mkdir(parents=True, exist_ok=True)

                if self.config.verbose:
                    print(f"Creating video directories for cameras: {actual_cameras}")
        elif self.config.export_videos:
            # 回退：基于 metadata 创建（可能包含不存在的目录）
            for feature_name in self.source_metadata.get("features", {}):
                if feature_name.startswith("observation.images."):
                    video_dir = self.output_path / "videos" / "chunk-000" / feature_name
                    video_dir.mkdir(parents=True, exist_ok=True)

        # 创建 mask 视频目录
        if self.config.export_mask:
            mask_dir = self.output_path / "videos" / "chunk-000" / "observation.images.sweep_mask"
            mask_dir.mkdir(parents=True, exist_ok=True)

    def _export_parquet_segment(
        self,
        source_df: pd.DataFrame,
        segment_info: SegmentExportInfo,
        source_start_frame: int
    ) -> Tuple[pd.DataFrame, int]:
        """
        导出单个 segment 的 parquet 数据

        Args:
            source_df: 源 episode 的 DataFrame
            segment_info: segment 导出信息
            source_start_frame: 源 episode 在全局索引中的起始帧

        Returns:
            (导出的 DataFrame, 帧数)
        """
        boundary = segment_info.boundary

        # 计算在 DataFrame 中的索引范围
        local_start = boundary.T_t0
        local_end = boundary.T_t1

        # 提取 segment 数据
        segment_df = source_df.iloc[local_start:local_end + 1].copy()

        # 重置帧索引和时间戳
        segment_df['frame_index'] = range(len(segment_df))
        segment_df['episode_index'] = segment_info.new_episode_id

        # 重置时间戳从 0 开始
        if 'timestamp' in segment_df.columns and len(segment_df) > 0:
            first_timestamp = segment_df['timestamp'].iloc[0]
            segment_df['timestamp'] = segment_df['timestamp'] - first_timestamp

        # 更新 index
        segment_df['index'] = range(len(segment_df))

        # 设置 task_index
        segment_df['task_index'] = segment_info.task_index

        # 保存 parquet
        new_episode_id = segment_info.new_episode_id
        chunk_idx = new_episode_id // self.config.chunks_size
        parquet_path = (
            self.output_path / "data" / f"chunk-{chunk_idx:03d}" /
            f"episode_{new_episode_id:06d}.parquet"
        )
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        segment_df.to_parquet(parquet_path, index=False)

        return segment_df, len(segment_df)

    def _export_video_segment(
        self,
        source_video_path: str,
        output_video_path: str,
        start_frame: int,
        end_frame: int,
        fps: float = None
    ) -> bool:
        """
        导出视频片段

        Args:
            source_video_path: 源视频路径
            output_video_path: 输出视频路径
            start_frame: 起始帧
            end_frame: 结束帧
            fps: 帧率 (如果为 None，从源视频获取)

        Returns:
            是否成功
        """
        # 优先使用 FFmpeg，因为它可以更好地处理各种编码格式（包括 AV1）
        if FFMPEG_AVAILABLE:
            return self._export_video_segment_ffmpeg(
                source_video_path, output_video_path, start_frame, end_frame, fps
            )
        elif CV2_AVAILABLE:
            return self._export_video_segment_cv2(
                source_video_path, output_video_path, start_frame, end_frame
            )
        else:
            print(f"Warning: Neither FFmpeg nor OpenCV available, skipping video export")
            return False

    def _export_video_segment_ffmpeg(
        self,
        source_video_path: str,
        output_video_path: str,
        start_frame: int,
        end_frame: int,
        fps: float = None
    ) -> bool:
        """
        使用 FFmpeg 导出视频片段（软件解码）

        关键：使用 setpts=PTS-STARTPTS 确保切片后时间戳归零

        Args:
            source_video_path: 源视频路径
            output_video_path: 输出视频路径
            start_frame: 起始帧
            end_frame: 结束帧
            fps: 帧率

        Returns:
            是否成功
        """
        try:
            # 如果 fps 未指定，从源数据集元数据获取
            if fps is None:
                fps = self.source_metadata.get('fps', 10)

            # 计算时间参数
            start_time = start_frame / fps
            duration = (end_frame - start_frame + 1) / fps
            num_frames = end_frame - start_frame + 1

            # 构建 video filter：选择帧范围 + 重置时间戳 + 固定帧率
            # setpts=PTS-STARTPTS 确保时间戳从 0 开始
            # fps={fps} 确保输出帧率正确
            vf_filter = f'select=gte(n\\,{start_frame})*lte(n\\,{end_frame}),setpts=PTS-STARTPTS,fps={fps}'

            cmd = [
                'ffmpeg',
                '-hide_banner',
                '-loglevel', 'error',
                '-nostdin',
                '-hwaccel', 'none',  # 禁用硬件加速
                '-i', source_video_path,
                '-vf', vf_filter,
                '-c:v', 'libx264',  # 使用 H.264 编码
                '-preset', 'fast',
                '-crf', '18',
                '-pix_fmt', 'yuv420p',
                '-an',  # 无音频
                '-y',
                output_video_path
            ]

            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=120
            )

            if result.returncode == 0:
                return True

            # 如果帧号选择失败，回退到时间范围方式
            if self.config.verbose:
                print(f"Frame-based selection failed, trying time-based seek for {source_video_path}")

            # 时间范围方式：同样使用 setpts 重置时间戳
            vf_filter_time = f'setpts=PTS-STARTPTS,fps={fps}'

            cmd_time_based = [
                'ffmpeg',
                '-hide_banner',
                '-loglevel', 'error',
                '-nostdin',
                '-hwaccel', 'none',  # 禁用硬件加速
                '-ss', str(start_time),
                '-i', source_video_path,
                '-t', str(duration),
                '-vf', vf_filter_time,
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '18',
                '-pix_fmt', 'yuv420p',
                '-an',
                '-y',
                output_video_path
            ]

            result = subprocess.run(
                cmd_time_based,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=120
            )

            if result.returncode != 0:
                # 记录详细错误
                stderr_lines = result.stderr.decode('utf-8', errors='ignore').split('\n')
                stderr_preview = '\n'.join(stderr_lines[-4:]) if len(stderr_lines) > 4 else result.stderr.decode('utf-8', errors='ignore')
                cmd_str = ' '.join(cmd_time_based)
                if self.config.verbose:
                    print(f"Error: FFmpeg video segment export failed")
                    print(f"  Source: {source_video_path}")
                    print(f"  Output: {output_video_path}")
                    print(f"  Frames: {start_frame}-{end_frame}")
                    print(f"  Return code: {result.returncode}")
                    print(f"  Command: {cmd_str}")
                    print(f"  Stderr (last 4 lines):\n{stderr_preview}")
                return False

            return True

        except subprocess.TimeoutExpired:
            if self.config.verbose:
                print(f"Error: FFmpeg timeout (120s) exporting video segment")
                print(f"  Source: {source_video_path}")
                print(f"  Frames: {start_frame}-{end_frame}")
            return False
        except Exception as e:
            if self.config.verbose:
                print(f"Error: Unexpected exception in _export_video_segment_ffmpeg")
                print(f"  Source: {source_video_path}")
                print(f"  Exception: {e}")
            return False

    def _export_video_segment_cv2(
        self,
        source_video_path: str,
        output_video_path: str,
        start_frame: int,
        end_frame: int
    ) -> bool:
        """
        使用 OpenCV 导出视频片段 (备选方案)

        注意: 如果源视频是 AV1 等格式，会先转换为 H.264
        """
        if not CV2_AVAILABLE:
            return False

        # 如果源视频是 AV1 等格式，转换为 H.264 以便 OpenCV 读取
        readable_video_path = _get_readable_video_path(source_video_path, verbose=self.config.verbose)

        cap = cv2.VideoCapture(readable_video_path)
        if not cap.isOpened():
            return False

        # 获取视频属性
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 创建输出视频
        fourcc = cv2.VideoWriter_fourcc(*self.config.video_codec)
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        # 定位到起始帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # 复制帧
        for _ in range(end_frame - start_frame + 1):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)

        cap.release()
        out.release()

        return True

    def _create_mask_video(
        self,
        mask: np.ndarray,
        output_video_path: str,
        num_frames: int,
        fps: float = None,
        source_video_path: str = None,
        start_frame: int = 0
    ) -> bool:
        """
        创建 mask 视频（叠加在原始图像上）

        Args:
            mask: 二值 mask (H, W), dtype=uint8
            output_video_path: 输出视频路径
            num_frames: 视频帧数
            fps: 帧率
            source_video_path: 源视频路径（如果提供，则叠加显示）
            start_frame: 源视频起始帧

        Returns:
            是否成功
        """
        if fps is None:
            fps = self.source_metadata.get('fps', 10)

        if not CV2_AVAILABLE:
            print("Warning: OpenCV not available, skipping mask video export")
            return False

        # 如果提供了源视频，创建叠加效果
        if source_video_path and os.path.exists(source_video_path):
            return self._create_mask_overlay_video(
                mask, output_video_path, source_video_path,
                start_frame, num_frames, fps
            )

        # 否则只输出 mask（灰度视频）
        # mask 转换为 3 通道（灰度视频）
        if mask.ndim == 2:
            mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        else:
            mask_3ch = mask

        height, width = mask_3ch.shape[:2]

        # 方法1: 使用 FFmpeg (更好的兼容性)
        tmp_img_path = None
        try:
            import tempfile

            # 先保存为临时图片
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                tmp_img_path = tmp_file.name

            cv2.imwrite(tmp_img_path, mask_3ch)

            # 使用 FFmpeg 生成视频 (H.264 编码，更好的兼容性)
            duration = num_frames / fps
            cmd = [
                'ffmpeg',
                '-hide_banner',
                '-loglevel', 'error',
                '-nostdin',
                '-hwaccel', 'none',  # 禁用硬件加速
                '-loop', '1',
                '-i', tmp_img_path,
                '-c:v', 'libx264',
                '-t', str(duration),
                '-pix_fmt', 'yuv420p',
                '-r', str(fps),
                '-preset', 'fast',
                '-y',
                output_video_path
            ]

            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=60
            )

            if result.returncode == 0:
                return True
            else:
                if self.config.verbose:
                    stderr_preview = result.stderr.decode('utf-8', errors='ignore')[:200]
                    print(f"FFmpeg mask video creation failed: {stderr_preview}")
        except Exception as e:
            if self.config.verbose:
                print(f"FFmpeg mask video creation failed: {e}")
        finally:
            # 清理临时文件
            if tmp_img_path is not None and os.path.exists(tmp_img_path):
                try:
                    os.unlink(tmp_img_path)
                except Exception:
                    pass

        # 方法2: 使用 OpenCV (备选)
        for codec in ['avc1', 'mp4v', 'XVID']:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

            if out.isOpened():
                break
        else:
            print(f"Warning: Cannot create video writer for {output_video_path}")
            return False

        # 重复写入相同的 mask 帧
        for _ in range(num_frames):
            out.write(mask_3ch)

        out.release()
        return True

    def _create_mask_overlay_video(
        self,
        mask: np.ndarray,
        output_video_path: str,
        source_video_path: str,
        start_frame: int,
        num_frames: int,
        fps: float
    ) -> bool:
        """
        创建 mask 叠加视频

        将 sweep mask 以半透明方式叠加在源视频上
        优先使用 FFmpeg 进行视频编码（更可靠）

        Args:
            mask: 二值 mask (H, W), dtype=uint8
            output_video_path: 输出视频路径
            source_video_path: 源视频路径
            start_frame: 起始帧
            num_frames: 帧数
            fps: 帧率

        Returns:
            是否成功
        """
        if not CV2_AVAILABLE:
            return False

        # 如果源视频是 AV1 等格式，转换为 H.264 以便 OpenCV 读取
        readable_video_path = _get_readable_video_path(source_video_path, verbose=self.config.verbose)

        # 打开源视频
        cap = cv2.VideoCapture(readable_video_path)
        if not cap.isOpened():
            print(f"Warning: Cannot open source video for overlay: {readable_video_path}")
            return False

        # 获取视频属性
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 调整 mask 尺寸以匹配视频
        if mask.shape[0] != height or mask.shape[1] != width:
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

        # 获取 mask 颜色和透明度（从 ROI 配置）
        if self.roi_config is not None:
            mask_color_bgr = list(self.roi_config.mask_color)
            alpha = self.roi_config.mask_alpha
        else:
            mask_color_bgr = [255, 0, 255]  # 默认紫色 (BGR)
            alpha = 0.4  # 默认透明度

        # 创建彩色 mask
        mask_color = np.zeros((height, width, 3), dtype=np.uint8)
        mask_color[mask > 0] = mask_color_bgr

        # 定位到起始帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # 方法1: 使用 FFmpeg 管道写入（更可靠）
        if FFMPEG_AVAILABLE:
            try:
                # 启动 FFmpeg 进程，通过管道接收原始帧数据
                cmd = [
                    'ffmpeg',
                    '-hide_banner',
                    '-loglevel', 'error',
                    '-nostdin',
                    '-y',
                    '-f', 'rawvideo',
                    '-vcodec', 'rawvideo',
                    '-s', f'{width}x{height}',
                    '-pix_fmt', 'bgr24',
                    '-r', str(fps),
                    '-i', '-',  # 从 stdin 读取
                    '-c:v', 'libx264',
                    '-preset', 'fast',
                    '-crf', '18',
                    '-pix_fmt', 'yuv420p',
                    output_video_path
                ]

                process = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )

                # 处理每一帧并写入管道
                for _ in range(num_frames):
                    ret, frame = cap.read()
                    if not ret:
                        frame = np.zeros((height, width, 3), dtype=np.uint8)

                    # 创建叠加图像
                    overlay = frame.copy()
                    overlay[mask > 0] = mask_color[mask > 0]

                    # 混合原始帧和叠加层
                    blended = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)

                    # 写入管道
                    process.stdin.write(blended.tobytes())

                # 关闭管道并等待完成
                process.stdin.close()
                process.wait(timeout=120)

                cap.release()

                if process.returncode == 0:
                    return True
                else:
                    stderr = process.stderr.read().decode('utf-8', errors='ignore')
                    if self.config.verbose:
                        print(f"FFmpeg pipe encoding failed: {stderr[:200]}")
                    # 继续尝试 OpenCV 方法
            except Exception as e:
                if self.config.verbose:
                    print(f"FFmpeg pipe encoding failed: {e}")
                cap.release()
                # 重新打开视频尝试 OpenCV 方法
                cap = cv2.VideoCapture(readable_video_path)
                if not cap.isOpened():
                    return False
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # 方法2: 使用 OpenCV VideoWriter（备选）
        out = None
        for codec in ['avc1', 'mp4v', 'XVID']:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            if out.isOpened():
                break
            out = None

        if out is None:
            cap.release()
            print(f"Warning: Cannot create video writer for overlay: {output_video_path}")
            return False

        # 处理每一帧
        for _ in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((height, width, 3), dtype=np.uint8)

            # 创建叠加图像
            overlay = frame.copy()
            overlay[mask > 0] = mask_color[mask > 0]

            # 混合原始帧和叠加层
            blended = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)

            out.write(blended)

        cap.release()
        out.release()

        return True

    def export_segment(
        self,
        source_df: pd.DataFrame,
        segment_info: SegmentExportInfo,
        source_video_paths: Dict[str, str],
        source_start_frame: int = 0
    ) -> Dict[str, Any]:
        """
        导出单个 segment（parquet + videos + mask）

        Args:
            source_df: 源 episode DataFrame
            segment_info: segment 导出信息
            source_video_paths: 源视频路径 {camera_name: path}
            source_start_frame: 源 episode 起始帧

        Returns:
            导出结果信息
        """
        boundary = segment_info.boundary
        new_episode_id = segment_info.new_episode_id
        chunk_idx = new_episode_id // self.config.chunks_size

        # 导出 parquet
        exported_df, frame_count = self._export_parquet_segment(
            source_df, segment_info, source_start_frame
        )

        self.export_segment_videos(segment_info, source_video_paths, frame_count=frame_count)

        # 返回 episode 元数据
        return {
            "episode_index": new_episode_id,
            "tasks": [segment_info.task_string],
            "length": frame_count
        }

    def export_segment_videos(
        self,
        segment_info: SegmentExportInfo,
        source_video_paths: Dict[str, str],
        frame_count: Optional[int] = None,
    ) -> None:
        """导出单个 segment 的视频与 mask"""
        if not (self.config.export_videos or self.config.export_mask):
            return

        boundary = segment_info.boundary
        new_episode_id = segment_info.new_episode_id
        chunk_idx = new_episode_id // self.config.chunks_size
        if frame_count is None:
            frame_count = boundary.T_t1 - boundary.T_t0 + 1

        # 导出视频
        if self.config.export_videos:
            for camera_name, source_path in source_video_paths.items():
                video_key = f"observation.images.{camera_name}"
                output_path = (
                    self.output_path / "videos" / f"chunk-{chunk_idx:03d}" /
                    video_key / f"episode_{new_episode_id:06d}.mp4"
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)

                self._export_video_segment(
                    source_path,
                    str(output_path),
                    boundary.T_t0,
                    boundary.T_t1
                )

        # 导出 mask 视频
        if self.config.export_mask and self.mask_generator is not None:
            try:
                # 从主相机视频生成 mask
                main_video_path = None
                for cam_name in ["main", "cam_main", "observation.images.main"]:
                    main_video_path = source_video_paths.get(cam_name)
                    if main_video_path:
                        break

                # 如果没有 main 相机，使用第一个可用的相机
                if not main_video_path and source_video_paths:
                    main_video_path = list(source_video_paths.values())[0]

                if main_video_path:
                    # 生成 sweep mask (T_t0 到 T_t1 的变化)
                    fps = self.source_metadata.get('fps', 10)
                    sweep_mask = self.mask_generator.generate_sweep_mask(
                        main_video_path,
                        boundary.T_t0,
                        boundary.T_t1,
                        fps=fps
                    )

                    # 导出 mask 视频（叠加在主相机图像上）
                    mask_output_path = (
                        self.output_path / "videos" / f"chunk-{chunk_idx:03d}" /
                        "observation.images.sweep_mask" / f"episode_{new_episode_id:06d}.mp4"
                    )
                    mask_output_path.parent.mkdir(parents=True, exist_ok=True)

                    success = self._create_mask_video(
                        sweep_mask,
                        str(mask_output_path),
                        frame_count,
                        fps=fps,
                        source_video_path=main_video_path,
                        start_frame=boundary.T_t0
                    )

                    if not success and self.config.verbose:
                        print(f"Warning: Failed to create mask video for episode {new_episode_id}")
            except Exception as e:
                if self.config.verbose:
                    print(f"Warning: Mask generation failed for episode {new_episode_id}: {e}")
                # 继续执行，不要中断整个导出流程

    def export_all_segments(
        self,
        data_loader,  # LeRobotDataLoader
        all_boundaries: Dict[int, List[SegmentBoundary]],
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        导出所有 segments

        Args:
            data_loader: 数据加载器
            all_boundaries: {episode_idx: [boundaries]} 的字典
            progress_callback: 进度回调函数

        Returns:
            导出统计信息
        """
        self.create_directory_structure(data_loader)

        # 固定的 task 名称
        fixed_task_name = "<skill>sweep<skill> Sweep away the red beads that are inside the green masked region and do not disturb beads outside the masked area."
        task_to_index = {fixed_task_name: 0}

        if self.config.verbose:
            print(f"Using fixed task name: {fixed_task_name}")

        # Phase 2: 准备导出信息
        export_infos = []
        new_episode_id = 0

        for ep_idx, boundaries in all_boundaries.items():
            for boundary in boundaries:
                if not boundary.is_valid:
                    continue

                export_info = SegmentExportInfo(
                    source_episode_id=ep_idx,
                    new_episode_id=new_episode_id,
                    boundary=boundary,
                    task_string=fixed_task_name,
                    task_index=0
                )
                export_infos.append(export_info)
                new_episode_id += 1

        if self.config.verbose:
            print(f"Prepared {len(export_infos)} segments for export")

        # Phase 3: 导出 segments
        episode_metadata_list = []
        episode_data_list = []
        total_frames = 0
        video_tasks = []

        # 使用 tqdm 显示导出进度
        with tqdm(total=len(export_infos), desc="Exporting segments", unit="segment") as pbar:
            for i, export_info in enumerate(export_infos):
                # 加载源数据
                source_data = data_loader.load_episode(export_info.source_episode_id)
                source_df = data_loader.load_episode_raw(export_info.source_episode_id)

                # 导出 parquet
                exported_df, frame_count = self._export_parquet_segment(
                    source_df, export_info, 0
                )
                episode_meta = {
                    "episode_index": export_info.new_episode_id,
                    "tasks": [export_info.task_string],
                    "length": frame_count
                }
                episode_metadata_list.append(episode_meta)
                total_frames += episode_meta["length"]

                if self.config.export_videos or self.config.export_mask:
                    video_tasks.append((export_info, source_data.video_paths or {}))

                # 加载导出的数据用于统计
                chunk_idx = export_info.new_episode_id // self.config.chunks_size
                parquet_path = (
                    self.output_path / "data" / f"chunk-{chunk_idx:03d}" /
                    f"episode_{export_info.new_episode_id:06d}.parquet"
                )
                episode_data_list.append(pd.read_parquet(parquet_path))

                # 更新进度条，显示已完成和剩余的数量
                pbar.set_postfix({
                    'completed': i + 1,
                    'remaining': len(export_infos) - i - 1
                })
                pbar.update(1)

                if progress_callback:
                    progress_callback(i + 1, len(export_infos))

        if self.config.export_videos or self.config.export_mask:
            if self.config.export_workers <= 1:
                with tqdm(total=len(video_tasks), desc="Exporting videos", unit="video") as pbar:
                    for export_info, source_video_paths in video_tasks:
                        self.export_segment_videos(export_info, source_video_paths)
                        pbar.update(1)
            else:
                if self.config.verbose:
                    print(f"Exporting videos with {self.config.export_workers} workers")
                with ProcessPoolExecutor(
                    max_workers=self.config.export_workers,
                    initializer=_init_export_worker,
                    initargs=(self.config, self.source_metadata),
                ) as executor:
                    future_to_episode = {
                        executor.submit(
                            _export_segment_videos_task,
                            export_info,
                            source_video_paths,
                        ): export_info.new_episode_id
                        for export_info, source_video_paths in video_tasks
                    }
                    with tqdm(total=len(future_to_episode), desc="Exporting videos", unit="video") as pbar:
                        for future in as_completed(future_to_episode):
                            episode_id = future_to_episode[future]
                            try:
                                future.result()
                            except Exception as e:
                                if self.config.verbose:
                                    print(f"Warning: Video export failed for episode {episode_id}: {e}")
                            pbar.update(1)

        # Phase 4: 计算统计信息
        if self.config.verbose:
            print("Computing dataset statistics...")

        self._compute_and_save_stats(episode_data_list)

        # Phase 5: 生成元数据
        self._create_metadata(episode_metadata_list, task_to_index, total_frames)

        stats = {
            "total_episodes": len(episode_metadata_list),
            "total_frames": total_frames,
            "total_tasks": len(task_to_index),
        }

        if self.config.verbose:
            print(f"\nExport complete:")
            print(f"  Episodes: {stats['total_episodes']}")
            print(f"  Frames: {stats['total_frames']}")
            print(f"  Tasks: {stats['total_tasks']}")

        return stats

    def _compute_and_save_stats(self, episode_data_list: List[pd.DataFrame]):
        """计算并保存 episode 统计信息"""
        stats_path = self.output_path / "meta" / "episodes_stats.jsonl"

        with open(stats_path, 'w') as f:
            for episode_index, episode_data in enumerate(episode_data_list):
                episode_stats = {
                    "episode_index": episode_index,
                    "stats": {}
                }

                # action 统计
                if 'action' in episode_data.columns:
                    actions = np.stack(episode_data['action'].values)
                    episode_stats['stats']['action'] = {
                        'min': actions.min(axis=0).tolist(),
                        'max': actions.max(axis=0).tolist(),
                        'mean': actions.mean(axis=0).tolist(),
                        'std': actions.std(axis=0).tolist(),
                        'count': [len(actions)]
                    }

                # observation.state 统计
                if 'observation.state' in episode_data.columns:
                    states = np.stack(episode_data['observation.state'].values)
                    episode_stats['stats']['observation.state'] = {
                        'min': states.min(axis=0).tolist(),
                        'max': states.max(axis=0).tolist(),
                        'mean': states.mean(axis=0).tolist(),
                        'std': states.std(axis=0).tolist(),
                        'count': [len(states)]
                    }

                # timestamp 统计
                if 'timestamp' in episode_data.columns:
                    timestamps = episode_data['timestamp'].values
                    episode_stats['stats']['timestamp'] = {
                        'min': [float(timestamps.min())],
                        'max': [float(timestamps.max())],
                        'mean': [float(timestamps.mean())],
                        'std': [float(timestamps.std()) if len(timestamps) > 1 else 0.0],
                        'count': [len(timestamps)]
                    }

                # frame_index 统计
                if 'frame_index' in episode_data.columns:
                    frame_indices = episode_data['frame_index'].values
                    episode_stats['stats']['frame_index'] = {
                        'min': [int(frame_indices.min())],
                        'max': [int(frame_indices.max())],
                        'mean': [float(frame_indices.mean())],
                        'std': [float(frame_indices.std()) if len(frame_indices) > 1 else 0.0],
                        'count': [len(frame_indices)]
                    }

                # episode_index 统计
                episode_stats['stats']['episode_index'] = {
                    'min': [episode_index],
                    'max': [episode_index],
                    'mean': [float(episode_index)],
                    'std': [0.0],
                    'count': [len(episode_data)]
                }

                # task_index 统计
                if 'task_index' in episode_data.columns:
                    task_indices = episode_data['task_index'].values
                    episode_stats['stats']['task_index'] = {
                        'min': [int(task_indices.min())],
                        'max': [int(task_indices.max())],
                        'mean': [float(task_indices.mean())],
                        'std': [0.0],
                        'count': [len(task_indices)]
                    }

                f.write(json.dumps(episode_stats) + '\n')

    def _create_metadata(
        self,
        episode_metadata_list: List[Dict],
        task_to_index: Dict[str, int],
        total_frames: int
    ):
        """创建元数据文件"""
        # info.json
        info = self.source_metadata.copy()
        info["total_episodes"] = len(episode_metadata_list)
        info["total_frames"] = total_frames
        info["total_tasks"] = len(task_to_index)
        info["splits"] = {"train": f"0:{len(episode_metadata_list)}"}

        # 过滤 features：只保留实际存在的图像特征
        features = info.get("features", {}).copy()
        filtered_features = {}

        for feature_name, feature_info in features.items():
            if feature_name.startswith("observation.images."):
                # 检查对应的视频目录是否存在且有内容
                video_dir = self.output_path / "videos" / "chunk-000" / feature_name
                if video_dir.exists() and any(video_dir.iterdir()):
                    filtered_features[feature_name] = feature_info
            else:
                # 非图像特征直接保留
                filtered_features[feature_name] = feature_info

        # 添加 mask 特征到 features（如果启用）
        if self.config.export_mask:
            # 获取现有图像特征的形状（假设 mask 尺寸相同）
            mask_shape = None
            for feature_name, feature_info in filtered_features.items():
                if feature_name.startswith("observation.images."):
                    # 从现有图像特征复制形状
                    mask_shape = feature_info.get("shape", [480, 640, 3])
                    break

            if mask_shape is None:
                mask_shape = [480, 640, 3]  # 默认形状

            # 添加 mask 特征
            filtered_features["observation.images.sweep_mask"] = {
                "dtype": "video",
                "shape": mask_shape,
                "names": ["height", "width", "channel"],
                "video_info": {
                    "video.fps": info.get("fps", 10),
                    "video.codec": "h264",  # 使用 h264 编码
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False
                }
            }

        info["features"] = filtered_features

        # 计算视频数量（包括 mask）
        camera_count = sum(
            1 for key in info.get("features", {})
            if key.startswith("observation.images.")
        )
        info["total_videos"] = len(episode_metadata_list) * camera_count

        with open(self.output_path / "meta" / "info.json", 'w') as f:
            json.dump(info, f, indent=2)

        # episodes.jsonl
        with open(self.output_path / "meta" / "episodes.jsonl", 'w') as f:
            for ep_meta in episode_metadata_list:
                f.write(json.dumps(ep_meta) + '\n')

        # tasks.jsonl
        tasks = [
            {"task_index": idx, "task": task}
            for task, idx in sorted(task_to_index.items(), key=lambda x: x[1])
        ]
        with open(self.output_path / "meta" / "tasks.jsonl", 'w') as f:
            for task in tasks:
                f.write(json.dumps(task) + '\n')


def export_segmented_dataset(
    source_dataset_path: str,
    output_path: str,
    all_boundaries: Dict[int, List[SegmentBoundary]],
    config: Optional[SweepSegmentConfig] = None,
    export_mask: bool = True,
    roi_config_path: Optional[str] = None,
    export_workers: int = 1,
) -> Dict[str, Any]:
    """
    便捷函数：导出切分后的数据集

    Args:
        source_dataset_path: 源数据集路径
        output_path: 输出路径
        all_boundaries: 所有 episode 的边界
        config: 切分配置（可选）
        export_mask: 是否导出 sweep mask
        roi_config_path: ROI 配置文件路径（用于 mask 过滤）
        export_workers: 视频导出并行进程数

    Returns:
        导出统计信息
    """
    from .data_loader import LeRobotDataLoader

    # 加载源数据集
    data_loader = LeRobotDataLoader(source_dataset_path)

    # 创建导出配置
    export_config = ExportConfig(
        output_path=Path(output_path),
        fps=data_loader.fps,
        verbose=config.verbose if config else True,
        export_mask=export_mask,
        roi_config_path=roi_config_path,
        export_workers=export_workers,
    )

    # 创建导出器
    exporter = LeRobotSegmentExporter(export_config, data_loader.metadata)

    # 导出
    stats = exporter.export_all_segments(
        data_loader=data_loader,
        all_boundaries=all_boundaries,
    )

    return stats


_EXPORTER_WORKER = None


def _init_export_worker(export_config: ExportConfig, source_metadata: Dict[str, Any]) -> None:
    global _EXPORTER_WORKER
    _EXPORTER_WORKER = LeRobotSegmentExporter(export_config, source_metadata)


def _export_segment_videos_task(
    segment_info: SegmentExportInfo,
    source_video_paths: Dict[str, str],
) -> int:
    if _EXPORTER_WORKER is None:
        raise RuntimeError("Exporter worker not initialized")
    _EXPORTER_WORKER.export_segment_videos(segment_info, source_video_paths)
    return segment_info.new_episode_id
