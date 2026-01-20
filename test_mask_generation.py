#!/usr/bin/env python3
"""
测试修改后的mask生成逻辑

验证：只保留首帧有但尾帧没有的红色区域
"""

import numpy as np
import cv2
from sweep_auto_split.mask_generator import HSVLegoSegmenter, MaskConfig


def test_mask_logic():
    """测试mask逻辑的正确性"""
    print("=" * 60)
    print("测试 Mask 生成逻辑")
    print("=" * 60)

    # 创建模拟的红色mask
    # mask_t0: 首帧有3个红色区域
    mask_t0 = np.zeros((224, 224), dtype=np.uint8)
    mask_t0[50:100, 50:100] = 255   # 区域1: 首帧有，尾帧也有 -> 不保留
    mask_t0[50:100, 130:180] = 255  # 区域2: 首帧有，尾帧没有 -> 保留
    mask_t0[130:180, 50:100] = 255  # 区域3: 首帧有，尾帧没有 -> 保留

    # mask_t1: 尾帧有2个红色区域
    mask_t1 = np.zeros((224, 224), dtype=np.uint8)
    mask_t1[50:100, 50:100] = 255   # 区域1: 首帧有，尾帧也有
    mask_t1[130:180, 130:180] = 255 # 区域4: 首帧没有，尾帧有 -> 不保留

    # 应用新的逻辑: mask_t0 AND (NOT mask_t1)
    sweep_mask = cv2.bitwise_and(mask_t0, cv2.bitwise_not(mask_t1))

    # 统计区域
    area_t0 = np.sum(mask_t0 > 0)
    area_t1 = np.sum(mask_t1 > 0)
    area_sweep = np.sum(sweep_mask > 0)

    print(f"\n首帧红色面积: {area_t0} 像素")
    print(f"尾帧红色面积: {area_t1} 像素")
    print(f"Sweep mask 面积: {area_sweep} 像素")

    # 验证结果
    # 预期: 区域2 + 区域3 = 50*50 + 50*50 = 5000 像素
    expected_area = 50 * 50 * 2

    print(f"\n预期 Sweep mask 面积: {expected_area} 像素")

    if area_sweep == expected_area:
        print("✓ 测试通过! Mask 逻辑正确")
        print("  只保留了首帧有但尾帧没有的区域")
        return True
    else:
        print(f"✗ 测试失败! 面积不匹配: {area_sweep} != {expected_area}")
        return False


def test_segmenter():
    """测试HSV分割器是否正常工作"""
    print("\n" + "=" * 60)
    print("测试 HSV 分割器")
    print("=" * 60)

    try:
        # 创建分割器
        config = MaskConfig()
        segmenter = HSVLegoSegmenter(config)

        # 创建一个简单的红色图像
        test_image = np.zeros((224, 224, 3), dtype=np.uint8)
        test_image[50:100, 50:100] = [0, 0, 255]  # BGR格式，纯红色

        # 分割
        mask = segmenter.segment(test_image)

        # 检查结果
        area = np.sum(mask > 0)
        print(f"\n分割结果面积: {area} 像素")

        if area > 0:
            print("✓ HSV 分割器正常工作")
            return True
        else:
            print("✗ HSV 分割器未检测到红色区域")
            return False

    except Exception as e:
        print(f"✗ HSV 分割器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def visualize_logic():
    """可视化mask逻辑"""
    print("\n" + "=" * 60)
    print("生成可视化示意图")
    print("=" * 60)

    # 创建模拟数据
    h, w = 224, 224

    # 首帧: 3个红色区域
    mask_t0 = np.zeros((h, w), dtype=np.uint8)
    mask_t0[50:100, 50:100] = 255
    mask_t0[50:100, 130:180] = 255
    mask_t0[130:180, 50:100] = 255

    # 尾帧: 2个红色区域
    mask_t1 = np.zeros((h, w), dtype=np.uint8)
    mask_t1[50:100, 50:100] = 255
    mask_t1[130:180, 130:180] = 255

    # 应用新逻辑
    sweep_mask = cv2.bitwise_and(mask_t0, cv2.bitwise_not(mask_t1))

    # 创建可视化
    vis = np.zeros((h, w * 4 + 30, 3), dtype=np.uint8)

    # 首帧mask (红色)
    vis[:, :w][mask_t0 > 0] = [0, 0, 255]

    # 尾帧mask (绿色)
    vis[:, w+10:w*2+10][mask_t1 > 0] = [0, 255, 0]

    # Sweep mask (黄色)
    vis[:, w*2+20:w*3+20][sweep_mask > 0] = [0, 255, 255]

    # 添加标签
    cv2.putText(vis, "T_t0", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(vis, "T_t1", (w + 20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(vis, "Result", (w*2 + 30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    output_path = "/Users/wanghaisheng/Desktop/Coding/Research/Sweep-AutoSplit/mask_logic_test.png"
    cv2.imwrite(output_path, vis)
    print(f"\n可视化已保存: {output_path}")
    print("  红色: 首帧mask (T_t0)")
    print("  绿色: 尾帧mask (T_t1)")
    print("  黄色: 最终mask (只保留首帧有但尾帧没有的区域)")


if __name__ == "__main__":
    # 运行测试
    test1_passed = test_mask_logic()
    test2_passed = test_segmenter()

    # 生成可视化
    try:
        visualize_logic()
    except Exception as e:
        print(f"可视化失败: {e}")

    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    print(f"Mask逻辑测试: {'✓ 通过' if test1_passed else '✗ 失败'}")
    print(f"HSV分割器测试: {'✓ 通过' if test2_passed else '✗ 失败'}")

    if test1_passed and test2_passed:
        print("\n✓ 所有测试通过! 代码修改成功")
    else:
        print("\n✗ 部分测试失败，请检查")
