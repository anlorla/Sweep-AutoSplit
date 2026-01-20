"""
Mask 生成模块

从视频帧中生成 sweep 的动态 mask：
M_t = mask(frame_{T_t0}) AND NOT mask(frame_{T_t1})

该 mask 表示"这次 sweep 扫走了哪些红色区域"（首帧有但尾帧没有的区域），
作为训练的 visual prompt。

使用 HSV 颜色分割方法（快速，适合红色乐高积木）
"""

import os
import sys
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Union
from dataclasses import dataclass, field

import yaml

# D. 在 import cv2 前禁用 OpenCV 的硬件加速
# 这会影响 cv2.VideoCapture 的 FFmpeg backend 行为
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "hwaccel;none")

import numpy as np
import cv2


# C. 打印 FFmpeg 路径和版本信息（模块初始化时执行一次）
def _check_ffmpeg_info():
    """检查并打印 FFmpeg 配置信息"""
    try:
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5
            )
            version_line = result.stdout.decode().split('\n')[0] if result.returncode == 0 else "Unknown"

            print(f"[mask_generator] FFmpeg info:")
            print(f"  Path: {ffmpeg_path}")
            print(f"  Version: {version_line}")

            return ffmpeg_path
        else:
            print("[mask_generator] Warning: ffmpeg not found in PATH")
            return None
    except Exception as e:
        print(f"[mask_generator] Warning: Failed to check ffmpeg info: {e}")
        return None


# 模块初始化时检查一次
_FFMPEG_PATH = _check_ffmpeg_info()

# 转换后的视频缓存目录
_CONVERTED_VIDEO_CACHE: Dict[str, str] = {}


@dataclass
class ROIConfig:
    """四边形 ROI 区域配置"""
    # 四个角点坐标 (顺时针或逆时针)
    corners: List[Tuple[float, float]] = field(default_factory=lambda: [
        (82.0, 45.0),    # 左上
        (150.0, 44.0),   # 右上
        (164.0, 158.0),  # 右下
        (73.0, 161.0),   # 左下
    ])

    # Mask 渲染颜色 (BGR 格式)
    mask_color: Tuple[int, int, int] = (255, 0, 255)  # 紫色

    # Mask 透明度 (0.0-1.0)
    mask_alpha: float = 0.4

    # 图像尺寸
    image_width: int = 224
    image_height: int = 224


def load_roi_config(config_path: Optional[str] = None) -> ROIConfig:
    """
    从 YAML 配置文件加载 ROI 配置

    Args:
        config_path: 配置文件路径，如果为 None 则使用默认路径

    Returns:
        ROIConfig 实例
    """
    if config_path is None:
        # 默认查找项目根目录的 config.yaml
        default_paths = [
            Path(__file__).parent.parent / "config.yaml",
            Path.cwd() / "config.yaml",
        ]
        for path in default_paths:
            if path.exists():
                config_path = str(path)
                break

    if config_path is None or not Path(config_path).exists():
        print("[load_roi_config] No config file found, using defaults")
        return ROIConfig()

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)

        if config_data is None:
            return ROIConfig()

        # 解析角点
        corners = config_data.get('corners', None)
        if corners is not None:
            corners = [tuple(pt) for pt in corners]
        else:
            corners = ROIConfig().corners

        # 解析 mask 渲染设置
        mask_rendering = config_data.get('mask_rendering', {})
        mask_color = tuple(mask_rendering.get('color', [255, 0, 255]))
        mask_alpha = mask_rendering.get('alpha', 0.4)

        # 解析图像尺寸
        image_size = config_data.get('image_size', {})
        image_width = image_size.get('width', 224)
        image_height = image_size.get('height', 224)

        roi_config = ROIConfig(
            corners=corners,
            mask_color=mask_color,
            mask_alpha=mask_alpha,
            image_width=image_width,
            image_height=image_height,
        )

        print(f"[load_roi_config] Loaded config from: {config_path}")
        print(f"  Corners: {roi_config.corners}")
        print(f"  Mask color (BGR): {roi_config.mask_color}")
        print(f"  Mask alpha: {roi_config.mask_alpha}")

        return roi_config

    except Exception as e:
        print(f"[load_roi_config] Error loading config: {e}")
        return ROIConfig()


def create_quadrilateral_mask(
    corners: List[Tuple[float, float]],
    image_width: int,
    image_height: int
) -> np.ndarray:
    """
    创建四边形 ROI 的二值 mask

    Args:
        corners: 四个角点坐标列表 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        image_width: 图像宽度
        image_height: 图像高度

    Returns:
        二值 mask (H, W), dtype=uint8, 四边形内部为 255，外部为 0
    """
    # 创建空白 mask
    mask = np.zeros((image_height, image_width), dtype=np.uint8)

    # 将角点转换为 numpy 数组并取整
    pts = np.array(corners, dtype=np.int32)

    # 使用 fillPoly 填充多边形
    cv2.fillPoly(mask, [pts], 255)

    return mask


def apply_roi_mask(
    sweep_mask: np.ndarray,
    roi_config: ROIConfig
) -> np.ndarray:
    """
    将 sweep mask 限制在四边形 ROI 区域内

    Args:
        sweep_mask: 原始 sweep mask (H, W), dtype=uint8
        roi_config: ROI 配置

    Returns:
        过滤后的 mask，只保留四边形内部的部分
    """
    h, w = sweep_mask.shape[:2]

    # 创建四边形 mask
    # 如果图像尺寸与配置不同，需要缩放角点坐标
    if w != roi_config.image_width or h != roi_config.image_height:
        # 计算缩放比例
        scale_x = w / roi_config.image_width
        scale_y = h / roi_config.image_height

        # 缩放角点坐标
        scaled_corners = [
            (pt[0] * scale_x, pt[1] * scale_y)
            for pt in roi_config.corners
        ]
    else:
        scaled_corners = roi_config.corners

    # 创建四边形 mask
    roi_mask = create_quadrilateral_mask(scaled_corners, w, h)

    # 应用 mask：只保留四边形内部的部分
    result = cv2.bitwise_and(sweep_mask, roi_mask)

    return result


# 全局 ROI 配置（延迟加载）
_ROI_CONFIG: Optional[ROIConfig] = None


def get_roi_config() -> ROIConfig:
    """获取全局 ROI 配置（单例模式）"""
    global _ROI_CONFIG
    if _ROI_CONFIG is None:
        _ROI_CONFIG = load_roi_config()
    return _ROI_CONFIG


def set_roi_config(config: ROIConfig):
    """设置全局 ROI 配置"""
    global _ROI_CONFIG
    _ROI_CONFIG = config


def _get_cache_dir() -> Path:
    """获取转换视频的缓存目录"""
    cache_dir = Path(tempfile.gettempdir()) / "sweep_auto_split_h264_cache"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir


def convert_video_to_h264(video_path: str, force: bool = False) -> Optional[str]:
    """
    将视频转换为 H.264 格式（用于解决 AV1 等编码无法解码的问题）

    Args:
        video_path: 原始视频路径
        force: 是否强制重新转换（即使已有缓存）

    Returns:
        转换后的视频路径，如果转换失败则返回 None
    """
    video_path = os.path.abspath(video_path)

    # 检查缓存
    if not force and video_path in _CONVERTED_VIDEO_CACHE:
        cached_path = _CONVERTED_VIDEO_CACHE[video_path]
        if os.path.exists(cached_path):
            print(f"[convert_video_to_h264] Using cached H.264 video: {cached_path}")
            return cached_path

    # 生成缓存文件名（基于原始文件路径的哈希）
    import hashlib
    path_hash = hashlib.md5(video_path.encode()).hexdigest()[:12]
    original_name = Path(video_path).stem
    cache_path = _get_cache_dir() / f"{original_name}_{path_hash}_h264.mp4"

    # 如果缓存文件已存在且不强制转换，直接使用
    if not force and cache_path.exists():
        print(f"[convert_video_to_h264] Found existing H.264 cache: {cache_path}")
        _CONVERTED_VIDEO_CACHE[video_path] = str(cache_path)
        return str(cache_path)

    print(f"[convert_video_to_h264] Converting video to H.264...")
    print(f"  Input: {video_path}")
    print(f"  Output: {cache_path}")

    try:
        # 使用 FFmpeg 转换为 H.264
        cmd = [
            'ffmpeg',
            '-hide_banner',
            '-loglevel', 'warning',
            '-nostdin',
            '-i', video_path,
            '-c:v', 'libx264',    # H.264 编码器
            '-crf', '18',          # 质量参数（越小质量越高，18-23 是常用范围）
            '-preset', 'fast',     # 编码速度（fast 平衡速度和压缩率）
            '-pix_fmt', 'yuv420p', # 兼容性好的像素格式
            '-c:a', 'copy',        # 音频直接复制（如果有的话）
            '-y',                  # 覆盖输出文件
            str(cache_path)
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=600  # 10分钟超时，视频转换可能较慢
        )

        if result.returncode == 0 and cache_path.exists():
            file_size = cache_path.stat().st_size / (1024 * 1024)  # MB
            print(f"[convert_video_to_h264] Conversion successful!")
            print(f"  Output size: {file_size:.2f} MB")

            # 加入缓存
            _CONVERTED_VIDEO_CACHE[video_path] = str(cache_path)
            return str(cache_path)
        else:
            stderr = result.stderr.decode('utf-8', errors='ignore')
            print(f"[convert_video_to_h264] Conversion failed!")
            print(f"  Return code: {result.returncode}")
            print(f"  Stderr: {stderr[:500]}")
            return None

    except subprocess.TimeoutExpired:
        print(f"[convert_video_to_h264] Conversion timeout (600s)")
        return None
    except Exception as e:
        print(f"[convert_video_to_h264] Conversion error: {e}")
        return None


def check_video_codec(video_path: str) -> Optional[str]:
    """
    检查视频的编解码器

    Args:
        video_path: 视频路径

    Returns:
        编解码器名称（如 'h264', 'av1', 'hevc'），失败返回 None
    """
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=codec_name',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10
        )

        if result.returncode == 0:
            codec = result.stdout.decode().strip()
            return codec
        return None
    except Exception as e:
        print(f"[check_video_codec] Error: {e}")
        return None


@dataclass
class MaskConfig:
    """Mask生成配置"""
    # HSV 红色范围（乐高积木）
    # 红色在HSV中跨越0度，需要两个范围
    hsv_lower1: Tuple[int, int, int] = (0, 100, 100)      # H:0-10
    hsv_upper1: Tuple[int, int, int] = (10, 255, 255)
    hsv_lower2: Tuple[int, int, int] = (160, 100, 100)    # H:160-180
    hsv_upper2: Tuple[int, int, int] = (180, 255, 255)

    # 形态学操作参数
    morph_kernel_size: int = 5
    morph_iterations: int = 2

    # 高斯模糊参数
    gaussian_ksize: int = 5

    # 最小连通域面积（过滤噪声）
    min_area: int = 100

    # 输出mask尺寸（如果为None则使用原始尺寸）
    output_size: Optional[Tuple[int, int]] = None

    # ROI 配置（用于限制 mask 在四边形区域内）
    roi_config: Optional[ROIConfig] = None

    # 是否启用 ROI 过滤
    enable_roi_filter: bool = True


class HSVLegoSegmenter:
    """
    基于HSV颜色的乐高积木分割器

    快速高效，适合红色乐高积木的颜色分割
    """

    def __init__(self, config: Optional[MaskConfig] = None):
        """
        初始化分割器

        Args:
            config: 分割配置
        """
        self.config = config or MaskConfig()

    def segment(self, image: np.ndarray) -> np.ndarray:
        """
        分割红色乐高积木

        Args:
            image: BGR图像 (H, W, 3)

        Returns:
            二值mask (H, W), dtype=uint8, 值为0或255
        """
        config = self.config

        # 高斯模糊去噪
        if config.gaussian_ksize > 0:
            image = cv2.GaussianBlur(
                image,
                (config.gaussian_ksize, config.gaussian_ksize),
                0
            )

        # BGR to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 红色分割（两个范围）
        mask1 = cv2.inRange(hsv, config.hsv_lower1, config.hsv_upper1)
        mask2 = cv2.inRange(hsv, config.hsv_lower2, config.hsv_upper2)
        mask = cv2.bitwise_or(mask1, mask2)

        # 形态学操作
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (config.morph_kernel_size, config.morph_kernel_size)
        )

        # 闭运算填充小孔
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, kernel,
            iterations=config.morph_iterations
        )

        # 开运算去除噪点
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, kernel,
            iterations=config.morph_iterations
        )

        # 过滤小连通域
        if config.min_area > 0:
            mask = self._filter_small_regions(mask, config.min_area)

        return mask

    def _filter_small_regions(
        self,
        mask: np.ndarray,
        min_area: int
    ) -> np.ndarray:
        """过滤小于min_area的连通域"""
        # 找连通域
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        # 创建输出mask
        output = np.zeros_like(mask)

        # 保留大于min_area的连通域（跳过背景label=0）
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                output[labels == i] = 255

        return output


class VideoFrameExtractor:
    """从视频中提取指定帧"""

    @staticmethod
    def _make_select_filter(frame_idx: int) -> str:
        """
        E. 构建 filter 字符串（避免转义错误）

        Args:
            frame_idx: 帧索引

        Returns:
            filter 字符串
        """
        return f"select=eq(n\\,{frame_idx})"

    @staticmethod
    def extract_frame_ffmpeg(video_path: str, frame_idx: int, fps: float = 10.0) -> Optional[np.ndarray]:
        """
        使用 FFmpeg 从视频中提取指定帧（按帧号选择，软件解码）

        Args:
            video_path: 视频路径
            frame_idx: 帧索引（从0开始）
            fps: 视频帧率（此参数保留兼容性，实际不使用时间seek）

        Returns:
            BGR图像 (H, W, 3) 或 None（如果失败）
        """
        tmp_path = None
        try:
            # 创建临时文件
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                tmp_path = tmp_file.name

            # 不再强制指定 AV1 解码器，让 FFmpeg 自动选择
            # 这样可以更好地兼容各种视频格式（包括 H.264、H.265、AV1 等）

            # 使用 FFmpeg 提取帧
            # P0-1: 强制软件解码，避免硬件加速导致的像素格式协商失败
            # P0-2: 使用帧号选择而非时间seek，避免关键帧问题
            cmd = [
                'ffmpeg',
                '-hide_banner',
                '-loglevel', 'error',
                '-nostdin',
                '-hwaccel', 'none',  # 禁用硬件加速
                '-i', video_path,
                '-vf', VideoFrameExtractor._make_select_filter(frame_idx),  # E. 使用辅助函数
                '-vsync', '0',  # 禁用帧率同步
                '-frames:v', '1',  # 只输出一帧
                '-pix_fmt', 'bgr24',  # 明确指定像素格式
                '-y',
                tmp_path
            ]

            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30
            )

            if result.returncode == 0 and os.path.exists(tmp_path):
                # 读取图像
                frame = cv2.imread(tmp_path)
                if frame is not None:
                    return frame
                else:
                    # imread 失败，记录错误
                    print(f"Error: cv2.imread failed for {tmp_path} (video: {video_path}, frame: {frame_idx})")
                    return None
            else:
                # E. FFmpeg 执行失败，记录详细错误信息（stderr 最后 20 行）
                stderr_full = result.stderr.decode('utf-8', errors='ignore')
                stderr_lines = stderr_full.split('\n')
                stderr_preview = '\n'.join(stderr_lines[-20:]) if len(stderr_lines) > 20 else stderr_full
                cmd_str = ' '.join(cmd)

                print(f"Error: FFmpeg frame extraction failed")
                print(f"  Video: {video_path}")
                print(f"  Frame index: {frame_idx}")
                print(f"  Return code: {result.returncode}")
                print(f"  Command: {cmd_str}")
                print(f"  Stderr (last 20 lines):")
                print(f"{stderr_preview}")
                print(f"  Hint: If the video is AV1 encoded and decoding fails, consider converting it to H.264 first:")
                print(f"        ffmpeg -i <input.mp4> -c:v libx264 -crf 18 -preset fast <output.mp4>")
                return None

        except subprocess.TimeoutExpired:
            print(f"Error: FFmpeg timeout (30s) extracting frame {frame_idx} from {video_path}")
            return None
        except Exception as e:
            print(f"Error: Unexpected exception in extract_frame_ffmpeg")
            print(f"  Video: {video_path}")
            print(f"  Frame index: {frame_idx}")
            print(f"  Exception: {e}")
            return None
        finally:
            # P1-5: 确保临时文件总是被清理
            if tmp_path is not None and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception as e:
                    print(f"Warning: Failed to delete temporary file {tmp_path}: {e}")

    @staticmethod
    def extract_frame(
        video_path: str,
        frame_idx: int,
        fps: float = 10.0,
        use_opencv_fallback: bool = True,
        auto_convert_h264: bool = True
    ) -> Optional[np.ndarray]:
        """
        从视频中提取指定帧（优先使用FFmpeg，失败时自动转换为H.264再重试）

        Args:
            video_path: 视频路径
            frame_idx: 帧索引
            fps: 视频帧率（保留兼容性）
            use_opencv_fallback: 是否在FFmpeg失败时使用OpenCV fallback
            auto_convert_h264: 是否在解码失败时自动转换为H.264格式

        Returns:
            BGR图像 (H, W, 3) 或 None（如果失败）
        """
        _ = fps  # 未使用，保留兼容性

        # 首先尝试直接用 FFmpeg 提取
        frame = VideoFrameExtractor.extract_frame_ffmpeg(video_path, frame_idx)
        if frame is not None:
            return frame

        # FFmpeg 失败，尝试自动转换为 H.264
        if auto_convert_h264:
            # 检查视频编解码器
            codec = check_video_codec(video_path)
            print(f"[VideoFrameExtractor] FFmpeg failed for {video_path}")
            print(f"  Detected codec: {codec}")

            # 如果不是 H.264，尝试转换
            if codec != 'h264':
                print(f"[VideoFrameExtractor] Attempting to convert to H.264...")
                h264_path = convert_video_to_h264(video_path)

                if h264_path is not None:
                    # 使用转换后的视频重试
                    frame = VideoFrameExtractor.extract_frame_ffmpeg(h264_path, frame_idx)
                    if frame is not None:
                        print(f"[VideoFrameExtractor] Successfully extracted frame from H.264 converted video")
                        return frame
                    else:
                        print(f"[VideoFrameExtractor] Still failed after H.264 conversion")
                else:
                    print(f"[VideoFrameExtractor] H.264 conversion failed")

        # OpenCV fallback
        if not use_opencv_fallback:
            return None

        print(f"[VideoFrameExtractor] Trying OpenCV fallback for frame {frame_idx}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: OpenCV cannot open video: {video_path}")
            return None

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            print(f"Warning: OpenCV cannot read frame {frame_idx} from {video_path}")
            return None

        print(f"[VideoFrameExtractor] OpenCV fallback succeeded for frame {frame_idx}")
        return frame

    @staticmethod
    def extract_frames(
        video_path: str,
        frame_indices: List[int],
        fps: float = 10.0
    ) -> Dict[int, np.ndarray]:
        """
        从视频中提取多个帧

        Args:
            video_path: 视频路径
            frame_indices: 帧索引列表
            fps: 视频帧率

        Returns:
            {frame_idx: frame} 字典
        """
        frames = {}
        for idx in frame_indices:
            frame = VideoFrameExtractor.extract_frame(video_path, idx, fps)
            if frame is not None:
                frames[idx] = frame

        return frames


class SweepMaskGenerator:
    """
    Sweep动态mask生成器

    核心功能：生成 M_t = mask(T_t0) AND NOT mask(T_t1)

    该mask表示一次sweep扫走的红色区域（首帧有但尾帧没有），作为训练时的visual prompt。
    支持四边形 ROI 过滤，只保留指定区域内的 mask。
    """

    def __init__(
        self,
        segmenter: Optional[HSVLegoSegmenter] = None,
        config: Optional[MaskConfig] = None,
        roi_config: Optional[ROIConfig] = None
    ):
        """
        初始化mask生成器

        Args:
            segmenter: 分割器实例（如果为None则创建默认的）
            config: mask配置
            roi_config: ROI 配置（如果为None则从全局配置加载）
        """
        self.config = config or MaskConfig()

        # 加载或使用 ROI 配置
        if roi_config is not None:
            self.roi_config = roi_config
        elif self.config.roi_config is not None:
            self.roi_config = self.config.roi_config
        else:
            self.roi_config = get_roi_config()

        self.segmenter = segmenter or HSVLegoSegmenter(self.config)
        self.frame_extractor = VideoFrameExtractor()

    def generate_sweep_mask(
        self,
        video_path: str,
        T_t0: int,
        T_t1: int,
        fps: float = 10.0,
        return_intermediate: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        """
        生成单次sweep的动态mask

        M_t = mask(frame_{T_t0}) AND NOT mask(frame_{T_t1})

        只保留首帧有但尾帧没有的红色区域（被sweep扫走的LEGO）

        Args:
            video_path: 视频路径
            T_t0: segment起始帧
            T_t1: segment结束帧
            fps: 视频帧率
            return_intermediate: 是否返回中间结果（用于调试）

        Returns:
            如果return_intermediate=False: mask (H, W), uint8
            如果return_intermediate=True: (mask, intermediate_dict)
        """
        # 提取帧
        frame_t0 = self.frame_extractor.extract_frame(video_path, T_t0, fps)
        frame_t1 = self.frame_extractor.extract_frame(video_path, T_t1, fps)

        if frame_t0 is None or frame_t1 is None:
            # 返回空mask
            if frame_t0 is not None:
                h, w = frame_t0.shape[:2]
            elif frame_t1 is not None:
                h, w = frame_t1.shape[:2]
            else:
                h, w = 480, 640  # 默认尺寸

            empty_mask = np.zeros((h, w), dtype=np.uint8)
            if return_intermediate:
                return empty_mask, {"error": "Failed to extract frames"}
            return empty_mask

        # 分割两帧
        mask_t0 = self.segmenter.segment(frame_t0)
        mask_t1 = self.segmenter.segment(frame_t1)

        # 计算差异 (只保留 t0 有但 t1 没有的区域)
        # 这些是被 sweep 走的红色区域（消失的 LEGO）
        # sweep_mask = mask_t0 AND (NOT mask_t1)
        sweep_mask = cv2.bitwise_and(mask_t0, cv2.bitwise_not(mask_t1))

        # 应用四边形 ROI 过滤
        if self.config.enable_roi_filter and self.roi_config is not None:
            sweep_mask = apply_roi_mask(sweep_mask, self.roi_config)

        # 可选：调整输出尺寸
        if self.config.output_size is not None:
            sweep_mask = cv2.resize(
                sweep_mask,
                self.config.output_size,
                interpolation=cv2.INTER_NEAREST
            )

        if return_intermediate:
            intermediate = {
                "frame_t0": frame_t0,
                "frame_t1": frame_t1,
                "mask_t0": mask_t0,
                "mask_t1": mask_t1,
                "T_t0": T_t0,
                "T_t1": T_t1,
            }
            return sweep_mask, intermediate

        return sweep_mask

    def generate_masks_for_episode(
        self,
        video_path: str,
        boundaries: List,  # List[SegmentBoundary]
        verbose: bool = False
    ) -> Dict[int, np.ndarray]:
        """
        为整个episode的所有有效segment生成mask

        Args:
            video_path: 视频路径
            boundaries: segment边界列表
            verbose: 是否打印详细信息

        Returns:
            {sweep_idx: mask} 字典
        """
        masks = {}

        for boundary in boundaries:
            if not boundary.is_valid:
                continue

            sweep_idx = boundary.sweep_idx
            T_t0 = boundary.T_t0
            T_t1 = boundary.T_t1

            if verbose:
                print(f"  Generating mask for sweep {sweep_idx}: T=[{T_t0}, {T_t1}]")

            mask = self.generate_sweep_mask(video_path, T_t0, T_t1)
            masks[sweep_idx] = mask

        return masks

    def visualize_mask(
        self,
        video_path: str,
        T_t0: int,
        T_t1: int,
        output_path: Optional[str] = None
    ) -> np.ndarray:
        """
        可视化sweep mask（用于调试）

        创建2x2网格：
        - 左上: T_t0帧
        - 右上: T_t1帧
        - 左下: mask_t0叠加在T_t0上
        - 右下: sweep_mask叠加在T_t1上

        Args:
            video_path: 视频路径
            T_t0: segment起始帧
            T_t1: segment结束帧
            output_path: 保存路径（如果为None则不保存）

        Returns:
            可视化图像 (2H, 2W, 3)
        """
        sweep_mask, intermediate = self.generate_sweep_mask(
            video_path, T_t0, T_t1, return_intermediate=True
        )

        if "error" in intermediate:
            print(f"Error: {intermediate['error']}")
            return np.zeros((480, 640, 3), dtype=np.uint8)

        frame_t0 = intermediate["frame_t0"]
        frame_t1 = intermediate["frame_t1"]
        mask_t0 = intermediate["mask_t0"]
        # mask_t1 is available but not used in visualization
        # mask_t1 = intermediate["mask_t1"]

        h, w = frame_t0.shape[:2]

        # 获取 mask 渲染配置
        mask_color = list(self.roi_config.mask_color) if self.roi_config else [255, 0, 255]
        alpha = self.roi_config.mask_alpha if self.roi_config else 0.5

        # 创建2x2网格
        grid = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)

        # 左上: T_t0帧
        grid[:h, :w] = frame_t0
        self._add_label(grid, f"Frame T_t0={T_t0}", (10, 30))

        # 右上: T_t1帧
        grid[:h, w:] = frame_t1
        self._add_label(grid, f"Frame T_t1={T_t1}", (w + 10, 30))

        # 左下: mask_t0叠加
        overlay_t0 = frame_t0.copy()
        overlay_t0[mask_t0 > 0] = [0, 255, 0]  # 绿色
        blended_t0 = cv2.addWeighted(frame_t0, 1 - alpha, overlay_t0, alpha, 0)
        grid[h:, :w] = blended_t0
        self._add_label(grid, f"Mask T_t0 (area: {np.sum(mask_t0 > 0)}px)", (10, h + 30))

        # 右下: sweep_mask叠加（使用配置的颜色）
        overlay_sweep = frame_t1.copy()
        overlay_sweep[sweep_mask > 0] = mask_color
        blended_sweep = cv2.addWeighted(frame_t1, 1 - alpha, overlay_sweep, alpha, 0)
        grid[h:, w:] = blended_sweep
        self._add_label(grid, f"Sweep Mask (t0 & ~t1, area: {np.sum(sweep_mask > 0)}px)", (w + 10, h + 30))

        # 绘制分隔线
        grid[h-1:h+1, :] = [128, 128, 128]
        grid[:, w-1:w+1] = [128, 128, 128]

        if output_path:
            cv2.imwrite(output_path, grid)
            print(f"Visualization saved to: {output_path}")

        return grid

    def _add_label(
        self,
        image: np.ndarray,
        text: str,
        position: Tuple[int, int]
    ):
        """在图像上添加带背景的文字标签"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2

        (text_w, text_h), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )

        x, y = position
        cv2.rectangle(
            image,
            (x - 5, y - text_h - 5),
            (x + text_w + 5, y + baseline + 5),
            (0, 0, 0), -1
        )
        cv2.putText(
            image, text, position, font,
            font_scale, (255, 255, 255), thickness
        )


def create_mask_generator(
    config: Optional[MaskConfig] = None,
    roi_config: Optional[ROIConfig] = None,
    config_path: Optional[str] = None
) -> SweepMaskGenerator:
    """
    工厂函数：创建mask生成器（使用HSV颜色分割）

    Args:
        config: mask配置
        roi_config: ROI 配置（如果为 None 则从配置文件加载）
        config_path: 配置文件路径

    Returns:
        SweepMaskGenerator实例
    """
    config = config or MaskConfig()

    # 加载 ROI 配置
    if roi_config is None:
        roi_config = load_roi_config(config_path)

    segmenter = HSVLegoSegmenter(config)
    return SweepMaskGenerator(segmenter, config, roi_config)


def render_mask_overlay_video(
    video_path: str,
    mask: np.ndarray,
    output_path: str,
    roi_config: Optional[ROIConfig] = None,
    show_roi_boundary: bool = True,
    fps: Optional[float] = None
) -> bool:
    """
    将 mask 叠加到视频上并渲染导出

    Args:
        video_path: 源视频路径
        mask: 二值 mask (H, W), dtype=uint8
        output_path: 输出视频路径
        roi_config: ROI 配置（如果为 None 则使用全局配置）
        show_roi_boundary: 是否显示 ROI 边界线
        fps: 输出帧率（如果为 None 则从源视频获取）

    Returns:
        是否成功
    """
    if roi_config is None:
        roi_config = get_roi_config()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video: {video_path}")
        return False

    # 获取视频属性
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps is None:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 10.0

    # 调整 mask 尺寸以匹配视频
    if mask.shape[0] != height or mask.shape[1] != width:
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

    # 获取渲染参数
    mask_color = list(roi_config.mask_color)
    alpha = roi_config.mask_alpha

    # 创建彩色 mask
    mask_colored = np.zeros((height, width, 3), dtype=np.uint8)
    mask_colored[mask > 0] = mask_color

    # 计算缩放后的 ROI 边界点
    roi_pts = None
    if show_roi_boundary:
        scale_x = width / roi_config.image_width
        scale_y = height / roi_config.image_height
        roi_pts = np.array([
            (int(pt[0] * scale_x), int(pt[1] * scale_y))
            for pt in roi_config.corners
        ], dtype=np.int32)

    # 创建输出视频
    # 尝试多种编码器
    for codec in ['avc1', 'mp4v', 'XVID']:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if out.isOpened():
            break
    else:
        cap.release()
        print(f"Error: Cannot create video writer: {output_path}")
        return False

    # 处理每一帧
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 创建叠加图像
        overlay = frame.copy()
        overlay[mask > 0] = mask_colored[mask > 0]

        # 混合原始帧和叠加层
        blended = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)

        # 绘制 ROI 边界（可选）
        if roi_pts is not None:
            cv2.polylines(blended, [roi_pts], isClosed=True, color=(0, 255, 0), thickness=2)

        # 添加标签
        # cv2.putText(blended, "Sweep Mask (ROI filtered)", (10, 30),
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        # cv2.putText(blended, f"Frame: {frame_count}/{total_frames}", (10, 60),
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        out.write(blended)
        frame_count += 1

    cap.release()
    out.release()

    print(f"Rendered mask overlay video: {output_path} ({frame_count} frames)")
    return True


def render_segment_with_mask(
    video_path: str,
    T_t0: int,
    T_t1: int,
    output_path: str,
    mask_generator: Optional[SweepMaskGenerator] = None,
    roi_config: Optional[ROIConfig] = None,
    show_roi_boundary: bool = True
) -> bool:
    """
    对视频的指定片段生成 mask 并渲染导出

    这是一个便捷函数，结合了 mask 生成和视频渲染

    Args:
        video_path: 源视频路径
        T_t0: segment 起始帧
        T_t1: segment 结束帧
        output_path: 输出视频路径
        mask_generator: mask 生成器（如果为 None 则创建默认的）
        roi_config: ROI 配置
        show_roi_boundary: 是否显示 ROI 边界线

    Returns:
        是否成功
    """
    if roi_config is None:
        roi_config = get_roi_config()

    if mask_generator is None:
        mask_generator = create_mask_generator(roi_config=roi_config)

    # 生成 sweep mask
    sweep_mask = mask_generator.generate_sweep_mask(video_path, T_t0, T_t1)

    if np.sum(sweep_mask > 0) == 0:
        print(f"Warning: Generated mask is empty for frames [{T_t0}, {T_t1}]")

    # 打开源视频提取片段
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video: {video_path}")
        return False

    # 获取视频属性
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 10.0

    # 调整 mask 尺寸以匹配视频
    if sweep_mask.shape[0] != height or sweep_mask.shape[1] != width:
        sweep_mask = cv2.resize(sweep_mask, (width, height), interpolation=cv2.INTER_NEAREST)

    # 获取渲染参数
    mask_color = list(roi_config.mask_color)
    alpha = roi_config.mask_alpha

    # 创建彩色 mask
    mask_colored = np.zeros((height, width, 3), dtype=np.uint8)
    mask_colored[sweep_mask > 0] = mask_color

    # 计算缩放后的 ROI 边界点
    roi_pts = None
    if show_roi_boundary:
        scale_x = width / roi_config.image_width
        scale_y = height / roi_config.image_height
        roi_pts = np.array([
            (int(pt[0] * scale_x), int(pt[1] * scale_y))
            for pt in roi_config.corners
        ], dtype=np.int32)

    # 创建输出视频
    for codec in ['avc1', 'mp4v', 'XVID']:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if out.isOpened():
            break
    else:
        cap.release()
        print(f"Error: Cannot create video writer: {output_path}")
        return False

    # 定位到起始帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, T_t0)

    # 处理指定范围的帧
    num_frames = T_t1 - T_t0 + 1
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            # 如果读取失败，使用黑帧
            frame = np.zeros((height, width, 3), dtype=np.uint8)

        # 创建叠加图像
        overlay = frame.copy()
        overlay[sweep_mask > 0] = mask_colored[sweep_mask > 0]

        # 混合原始帧和叠加层
        blended = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)

        # 绘制 ROI 边界（可选）
        if roi_pts is not None:
            cv2.polylines(blended, [roi_pts], isClosed=True, color=(0, 255, 0), thickness=2)

        # 添加标签
        cv2.putText(blended, f"Sweep [{T_t0}-{T_t1}]", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(blended, f"Frame: {T_t0 + i}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(blended, f"Mask area: {np.sum(sweep_mask > 0)}px", (10, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        out.write(blended)

    cap.release()
    out.release()

    print(f"Rendered segment with mask: {output_path} ({num_frames} frames)")
    return True


# ============================================================
# 测试代码
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test mask generation with ROI filtering")
    parser.add_argument("--video", type=str, required=True, help="Video path")
    parser.add_argument("--t0", type=int, required=True, help="Start frame")
    parser.add_argument("--t1", type=int, required=True, help="End frame")
    parser.add_argument("--output", type=str, default="mask_test.png", help="Output path for visualization image")
    parser.add_argument("--output-video", type=str, default=None, help="Output path for rendered video")
    parser.add_argument("--config", type=str, default=None, help="Config file path")
    parser.add_argument("--no-roi", action="store_true", help="Disable ROI filtering")

    args = parser.parse_args()

    # 加载 ROI 配置
    roi_config = load_roi_config(args.config)

    # 创建 MaskConfig
    mask_config = MaskConfig(
        enable_roi_filter=not args.no_roi
    )

    # 创建生成器
    generator = create_mask_generator(config=mask_config, roi_config=roi_config)

    # 生成可视化图像
    viz = generator.visualize_mask(args.video, args.t0, args.t1, args.output)
    print(f"Visualization saved to: {args.output}")

    # 如果指定了视频输出路径，则渲染视频
    if args.output_video:
        success = render_segment_with_mask(
            args.video,
            args.t0,
            args.t1,
            args.output_video,
            mask_generator=generator,
            roi_config=roi_config,
            show_roi_boundary=True
        )
        if success:
            print(f"Video rendered to: {args.output_video}")
        else:
            print("Failed to render video")
