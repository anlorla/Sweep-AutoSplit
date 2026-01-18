#!/usr/bin/env python3
"""
AV1 帧提取验证测试脚本

验证修复后的帧提取功能是否满足以下要求：
1. 连续提取 frame_idx in {0, 1, 5, 10, 20, 30} 均成功返回非空 BGR 图像
2. 运行过程中不得出现 AV1 相关错误关键词
3. 失败时返回 None 并打印清晰的错误日志
"""

import sys
import os
from pathlib import Path

# 添加项目路径到 sys.path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from sweep_auto_split.mask_generator import VideoFrameExtractor
import cv2
import subprocess


def check_video_codec(video_path: str) -> str:
    """检查视频编码格式"""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream=codec_name', '-of', 'default=noprint_wrappers=1:nokey=1',
             video_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5
        )
        codec = result.stdout.decode().strip()
        return codec
    except Exception as e:
        return f"Unknown (error: {e})"


def check_video_info(video_path: str):
    """检查视频基本信息"""
    try:
        # 获取帧数
        result_frames = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-count_packets', '-show_entries', 'stream=nb_read_packets',
             '-of', 'csv=p=0', video_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5
        )
        total_frames = result_frames.stdout.decode().strip()

        # 获取分辨率
        result_size = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream=width,height',
             '-of', 'csv=s=x:p=0', video_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5
        )
        resolution = result_size.stdout.decode().strip()

        # 获取帧率
        result_fps = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream=r_frame_rate',
             '-of', 'default=noprint_wrappers=1:nokey=1', video_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5
        )
        fps_str = result_fps.stdout.decode().strip()

        return {
            'total_frames': total_frames,
            'resolution': resolution,
            'fps': fps_str
        }
    except Exception as e:
        return {'error': str(e)}


def test_frame_extraction(video_path: str, test_indices: list = None, fps: float = 10.0):
    """
    测试帧提取功能

    Args:
        video_path: 视频路径
        test_indices: 要测试的帧索引列表
        fps: 视频帧率
    """
    if test_indices is None:
        test_indices = [0, 1, 5, 10, 20, 30]

    print("=" * 80)
    print("AV1 帧提取验证测试")
    print("=" * 80)

    # 检查视频文件
    if not os.path.exists(video_path):
        print(f"❌ 错误: 视频文件不存在: {video_path}")
        return False

    print(f"\n📹 视频路径: {video_path}")

    # 检查视频编码
    codec = check_video_codec(video_path)
    print(f"🎬 视频编码: {codec}")

    if codec.lower() != 'av1':
        print(f"⚠️  警告: 该视频不是 AV1 编码，而是 {codec}")

    # 检查视频信息
    info = check_video_info(video_path)
    if 'error' not in info:
        print(f"📊 视频信息:")
        print(f"   - 总帧数: {info['total_frames']}")
        print(f"   - 分辨率: {info['resolution']}")
        print(f"   - 帧率: {info['fps']}")

    print(f"\n🔍 测试帧索引: {test_indices}")
    print(f"⏱️  帧率参数: {fps} fps")

    # ==================== 第一轮测试：禁用 OpenCV fallback ====================
    print("\n" + "=" * 80)
    print("第一轮测试：仅使用 FFmpeg（禁用 OpenCV fallback）")
    print("目的：确认报错来源是 FFmpeg 还是 OpenCV")
    print("=" * 80)

    results_ffmpeg_only = {}
    for idx in test_indices:
        print(f"\n测试帧 #{idx}:")
        try:
            # 直接调用 extract_frame_ffmpeg，或使用 use_opencv_fallback=False
            frame = VideoFrameExtractor.extract_frame_ffmpeg(video_path, idx, fps)

            if frame is not None:
                h, w, c = frame.shape
                print(f"  ✅ 成功提取")
                print(f"     形状: {h}x{w}x{c}")
                results_ffmpeg_only[idx] = True
            else:
                print(f"  ❌ 失败: 返回 None")
                results_ffmpeg_only[idx] = False

        except Exception as e:
            print(f"  ❌ 异常: {e}")
            results_ffmpeg_only[idx] = False

    # ==================== 第二轮测试：启用 OpenCV fallback ====================
    print("\n" + "=" * 80)
    print("第二轮测试：FFmpeg + OpenCV fallback")
    print("目的：对比是否有差异，确认 fallback 行为")
    print("=" * 80)

    results_with_fallback = {}
    for idx in test_indices:
        print(f"\n测试帧 #{idx}:")
        try:
            frame = VideoFrameExtractor.extract_frame(video_path, idx, fps, use_opencv_fallback=True)

            if frame is not None:
                h, w, c = frame.shape
                print(f"  ✅ 成功提取")
                print(f"     形状: {h}x{w}x{c}")
                results_with_fallback[idx] = True
            else:
                print(f"  ❌ 失败: 返回 None")
                results_with_fallback[idx] = False

        except Exception as e:
            print(f"  ❌ 异常: {e}")
            results_with_fallback[idx] = False

    # 打印测试总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)

    success_ffmpeg = sum(1 for v in results_ffmpeg_only.values() if v)
    success_fallback = sum(1 for v in results_with_fallback.values() if v)
    total_count = len(test_indices)

    print(f"\n第一轮（仅 FFmpeg）: {success_ffmpeg}/{total_count} ({success_ffmpeg/total_count*100:.1f}%)")
    print(f"第二轮（FFmpeg + fallback）: {success_fallback}/{total_count} ({success_fallback/total_count*100:.1f}%)")

    # 分析差异
    if success_ffmpeg == total_count:
        print("\n✅ FFmpeg 完全成功！")
        print("   - 所有帧均通过 FFmpeg 提取")
        print("   - 没有触发 OpenCV fallback")
        if success_fallback == total_count:
            print("   - 第二轮测试也全部通过（符合预期）")
    elif success_fallback > success_ffmpeg:
        print(f"\n⚠️  OpenCV fallback 起作用了")
        print(f"   - FFmpeg 失败的帧: {[idx for idx, ok in results_ffmpeg_only.items() if not ok]}")
        print(f"   - OpenCV 成功救回: {success_fallback - success_ffmpeg} 帧")
        print(f"   - 建议检查上面的错误日志，确认是 filter/decoder 问题还是其他原因")
    else:
        print(f"\n❌ 两轮测试成功率一致，但未达到 100%")
        failed = [idx for idx, ok in results_ffmpeg_only.items() if not ok]
        print(f"   - 失败的帧索引: {failed}")

    # 检查是否看到关键错误
    print("\n" + "=" * 80)
    print("关键检查项")
    print("=" * 80)
    print("请检查上面的日志中是否出现以下关键词：")
    print("  ❌ 'Missing Sequence Header'")
    print("  ❌ 'Failed to get pixel format'")
    print("  ❌ 'Your platform doesn't support hardware accelerated AV1'")
    print("\n如果没有出现这些关键词，说明修复生效！")

    all_success = (success_ffmpeg == total_count)
    print("\n" + "=" * 80)

    return all_success


def main():
    """主函数"""
    # 默认测试视频路径（根据你之前提供的路径）
    default_video = "/home/zeno-yifan/NPM-Project/lerobot21_dataset/videos/chunk-000/observation.images.main/episode_000000.mp4"

    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = default_video

    # 运行测试
    success = test_frame_extraction(video_path)

    # 返回退出码
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
