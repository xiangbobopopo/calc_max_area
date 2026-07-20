"""
设计图边缘检测、图像分解与面积计算工具
=======================================
适用于：边缘清晰的设计图（线条分明的工程图、建筑图、平面图等）

处理流程：
  1. 读取图像 → 灰度化 → 去噪
  2. 边缘检测（Canny / 自适应阈值 / Harris 角点）
  3. 形态学操作闭合边缘，形成封闭区域
  4. 检测每个封闭区域（轮廓/连通域）
  5. 计算每个区域的面积（像素面积 + 物理面积）
  6. 可视化结果并输出面积报告
"""

import os
import sys
import argparse
import cv2
import numpy as np


# ============================================================================
#  辅助函数
# ============================================================================

def cvshow(img, title="image", wait=True):
    """显示图像，按任意键关闭"""
    cv2.imshow(title, img)
    if wait:
        cv2.waitKey(0)
    cv2.destroyAllWindows()


def calc_centroid(contour):
    """计算轮廓重心"""
    M = cv2.moments(contour)
    if M['m00'] != 0:
        return int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
    return None


def format_area(area_px, scale=1.0, unit="px²"):
    """格式化面积输出"""
    physical = area_px * scale
    if physical >= 100:
        return area_px, physical, f"{physical:.1f} {unit}"
    elif physical >= 1:
        return area_px, physical, f"{physical:.3f} {unit}"
    else:
        return area_px, physical, f"{physical:.6f} {unit}"


# ============================================================================
#  核心处理方法
# ============================================================================

def method_canny_edge_decomposition(img, debug=False, scale=1.0, unit="px²",
                                     canny_low=50, canny_high=150,
                                     min_area=50, close_kernel=5):
    """
    方法一：Canny 边缘检测 + 形态学闭合 → 提取封闭区域
    适用于：边缘清晰连续的设计图

    步骤：
    1. 灰度化 + 高斯模糊去噪
    2. Canny 检测边缘
    3. 形态学 CLOSE 操作闭合边缘间隙
    4. 反转二值图（边缘=黑，背景=白）
    5. 从四边 flood fill 标记背景
    6. 剩余白色区域 = 封闭的独立区域
    7. 连通域分析 / 轮廓检测提取每个区域
    8. 计算并标注每个区域的面积
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 1. 高斯模糊去噪
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    if debug:
        cvshow(blurred, "1. 高斯模糊")

    # 2. Canny 边缘检测
    edges = cv2.Canny(blurred, canny_low, canny_high)
    if debug:
        cvshow(edges, "2. Canny 边缘检测")

    # 3. 形态学 CLOSE：闭合边缘间隙
    kernel = np.ones((close_kernel, close_kernel), np.uint8)
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    # 再膨胀一下确保完全闭合
    edges_closed = cv2.dilate(edges_closed, kernel, iterations=1)
    if debug:
        cvshow(edges_closed, "3. 形态学闭合边缘")

    # 4. 反转：边缘→黑色(0)，背景→白色(255)
    edge_mask = cv2.bitwise_not(edges_closed)
    if debug:
        cvshow(edge_mask, "4. 反转二值图")

    # 5. Flood fill 从四边标记背景
    #    创建一个稍大的图像用于 flood fill（加1像素边框避免边缘泄漏）
    h2, w2 = h + 2, w + 2
    mask = np.zeros((h2, w2), np.uint8)

    # 从四个角开始 flood fill（种子点必须在原图范围内）
    fill_pts = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]
    for pt in fill_pts:
        if edge_mask[pt[1], pt[0]] == 255:  # 只从白色区域开始
            cv2.floodFill(edge_mask, mask, pt, 0, loDiff=0, upDiff=0)

    # 现在 edge_mask 中：
    #   - 黑色(0) = 外部背景 或 边缘线
    #   - 白色(255) = 封闭区域内部
    if debug:
        cvshow(edge_mask, "5. Flood fill 后（白色=封闭区域）")

    # 6. 提取每个封闭区域
    #    使用 findContours 检测白色区域
    regions = edge_mask.copy()
    contours, hierarchy = cv2.findContours(regions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 过滤并排序（按面积从大到小）
    results = []
    for i, cnt in enumerate(contours):
        area_px = cv2.contourArea(cnt)
        if area_px >= min_area:
            centroid = calc_centroid(cnt)
            _, _, area_str = format_area(area_px, scale, unit)
            results.append({
                "id": i + 1,
                "area_px": area_px,
                "area_str": area_str,
                "contour": cnt,
                "centroid": centroid,
            })

    # 按面积从大到小排序
    results.sort(key=lambda r: r["area_px"], reverse=True)
    for idx, r in enumerate(results):
        r["id"] = idx + 1

    return results, edges_closed, regions


def method_adaptive_threshold(img, debug=False, scale=1.0, unit="px²",
                               block_size=11, c_value=2,
                               min_area=50, close_kernel=3):
    """
    方法二：自适应阈值 + 形态学操作
    适用于：光照不均的设计图
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 自适应阈值
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block_size, c_value
    )
    if debug:
        cvshow(binary, "1. 自适应阈值（边缘=白）")

    # 闭运算连接断开的边缘
    kernel = np.ones((close_kernel, close_kernel), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    if debug:
        cvshow(closed, "2. 形态学闭合")

    # 膨胀加粗边缘
    closed = cv2.dilate(closed, kernel, iterations=1)

    # Invert and flood fill
    edge_mask = cv2.bitwise_not(closed)
    h, w = gray.shape
    h2, w2 = h + 2, w + 2
    mask = np.zeros((h2, w2), np.uint8)

    fill_pts = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]
    for pt in fill_pts:
        if edge_mask[pt[1], pt[0]] == 255:
            cv2.floodFill(edge_mask, mask, pt, 0, loDiff=0, upDiff=0)

    if debug:
        cvshow(edge_mask, "3. Flood fill 后（白色=封闭区域）")

    contours, _ = cv2.findContours(edge_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = []
    for i, cnt in enumerate(contours):
        area_px = cv2.contourArea(cnt)
        if area_px >= min_area:
            centroid = calc_centroid(cnt)
            _, _, area_str = format_area(area_px, scale, unit)
            results.append({
                "id": i + 1,
                "area_px": area_px,
                "area_str": area_str,
                "contour": cnt,
                "centroid": centroid,
            })

    results.sort(key=lambda r: r["area_px"], reverse=True)
    for idx, r in enumerate(results):
        r["id"] = idx + 1

    return results, closed, edge_mask


def method_harris_corner_decomposition(img, debug=False, scale=1.0, unit="px²",
                                        min_area=50):
    """
    方法三：Harris 角点检测 → 膨胀连接 → 提取区域
    这是原 get_max_area.py 的增强版
    适用于：用角点/圆点标记边界的特殊设计图
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Harris 角点检测
    gray_float = np.float32(blurred)
    dst = cv2.cornerHarris(gray_float, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)

    # 阈值化
    threshold = 0.01 * dst.max()
    corners = np.zeros_like(gray)
    corners[dst > threshold] = 255
    if debug:
        cvshow(corners, "1. Harris 角点")

    # 开运算去噪
    kernel = np.ones((4, 4), np.uint8)
    corners = cv2.morphologyEx(corners, cv2.MORPH_OPEN, kernel)
    if debug:
        cvshow(corners, "2. 开运算去噪")

    # 多次膨胀连接角点成线
    for radius in [3, 5, 7]:
        k = np.ones((radius * 2 + 1, radius * 2 + 1), np.uint8)
        corners = cv2.dilate(corners, k)

    if debug:
        cvshow(corners, "3. 膨胀连接角点")

    # 反转 + flood fill
    edge_mask = cv2.bitwise_not(corners)
    h, w = gray.shape
    h2, w2 = h + 2, w + 2
    mask = np.zeros((h2, w2), np.uint8)

    fill_pts = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]
    for pt in fill_pts:
        if edge_mask[pt[1], pt[0]] == 255:
            cv2.floodFill(edge_mask, mask, pt, 0, loDiff=0, upDiff=0)

    if debug:
        cvshow(edge_mask, "4. Flood fill 后（白色=封闭区域）")

    contours, _ = cv2.findContours(edge_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = []
    for i, cnt in enumerate(contours):
        area_px = cv2.contourArea(cnt)
        if area_px >= min_area:
            centroid = calc_centroid(cnt)
            _, _, area_str = format_area(area_px, scale, unit)
            results.append({
                "id": i + 1,
                "area_px": area_px,
                "area_str": area_str,
                "contour": cnt,
                "centroid": centroid,
            })

    results.sort(key=lambda r: r["area_px"], reverse=True)
    for idx, r in enumerate(results):
        r["id"] = idx + 1

    return results, corners, edge_mask


# ============================================================================
#  可视化与报告
# ============================================================================

def draw_results(img, results, show_id=True, show_area=True):
    """在原图上绘制所有检测到的区域和面积标注"""
    canvas = img.copy()
    output = img.copy()

    # 为每个区域分配随机颜色
    colors = []
    for i in range(len(results)):
        hue = int(i * 30) % 180
        color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(int(c) for c in color))

    for idx, r in enumerate(results):
        color = colors[idx]

        # 绘制轮廓
        cv2.drawContours(canvas, [r["contour"]], -1, color, 2)
        cv2.drawContours(output, [r["contour"]], -1, color, 2)

        # 绘制重心
        if r["centroid"]:
            cx, cy = r["centroid"]
            cv2.circle(canvas, (cx, cy), 4, color, -1)
            cv2.circle(output, (cx, cy), 4, color, -1)

            # 标注文本
            label = f"#{r['id']}"
            if show_area:
                label += f" {r['area_str']}"
            cv2.putText(canvas, label, (cx + 8, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(canvas, label, (cx + 8, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return canvas, output


def print_report(results, image_name="", scale=1.0, unit="px²"):
    """打印面积分析报告"""
    print("=" * 60)
    if image_name:
        print(f"  图像分析报告: {image_name}")
    print("=" * 60)
    print(f"{'#':>4} | {'面积(像素)':>12} | {'面积(物理)':>14} | {'重心坐标':>12}")
    print("-" * 60)
    total_px = 0
    for r in results:
        _, physical, _ = format_area(r["area_px"], scale, unit)
        centroid_str = f"({r['centroid'][0]},{r['centroid'][1]})" if r["centroid"] else "N/A"
        print(f"{r['id']:>4} | {r['area_px']:>12.1f} | {physical:>13.3f} | {centroid_str}")
        total_px += r["area_px"]
    print("-" * 60)
    total_physical = total_px * scale
    print(f"{'总计':>4} | {total_px:>12.1f} | {total_physical:>13.3f} |")
    print(f"区域数量: {len(results)}")
    print(f"比例尺: 1 px = {scale} {unit}" if scale != 1.0 else "")
    print("=" * 60)


# ============================================================================
#  主函数
# ============================================================================

def process_image(file_path, method="canny", debug=False, scale=1.0, unit="px²",
                  min_area=50, save_output=None,
                  canny_low=50, canny_high=150, close_kernel=5):
    """
    处理单张图像

    参数:
        file_path: 图像路径
        method: 检测方法 (canny / adaptive / harris)
        debug: 是否显示中间步骤
        scale: 每个像素对应的物理单位数
        unit: 物理单位名称
        min_area: 最小面积阈值，小于此值的区域被过滤
        save_output: 保存输出图像的路径（不保存则为 None）
        canny_low: Canny 低阈值
        canny_high: Canny 高阈值
        close_kernel: 形态学闭合核大小
    """
    img = cv2.imread(file_path)
    if img is None:
        print(f"错误: 无法读取图像 {file_path}")
        return None

    print(f"\n处理图像: {file_path}")
    print(f"图像尺寸: {img.shape[1]} x {img.shape[0]}")

    # 选择方法
    if method == "canny":
        results, edge_img, region_img = method_canny_edge_decomposition(
            img, debug, scale, unit,
            canny_low=canny_low, canny_high=canny_high,
            min_area=min_area, close_kernel=close_kernel
        )
    elif method == "adaptive":
        results, edge_img, region_img = method_adaptive_threshold(
            img, debug, scale, unit, min_area=min_area,
            close_kernel=close_kernel
        )
    elif method == "harris":
        results, edge_img, region_img = method_harris_corner_decomposition(
            img, debug, scale, unit, min_area=min_area
        )
    else:
        print(f"未知方法: {method}，可选: canny, adaptive, harris")
        return None

    # 报告
    print_report(results, os.path.basename(file_path), scale, unit)

    # 可视化
    canvas, output = draw_results(img, results)

    # 显示或保存结果
    if save_output:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        out_path = save_output if os.path.isdir(save_output) else \
            os.path.join(os.path.dirname(save_output),
                         f"{base_name}_result.png") if save_output else None
        if os.path.isdir(save_output):
            out_path = os.path.join(save_output, f"{base_name}_result.png")
        if out_path:
            cv2.imwrite(out_path, output)
            print(f"结果已保存: {out_path}")

    if debug:
        cvshow(output, "最终结果")
        if len(results) > 0:
            # 显示最大区域
            max_r = results[0]
            mask = np.zeros(img.shape[:2], np.uint8)
            cv2.drawContours(mask, [max_r["contour"]], -1, 255, -1)
            highlighted = img.copy()
            highlighted[mask == 255] = cv2.addWeighted(
                img[mask == 255], 0.5,
                np.full_like(img[mask == 255], (0, 255, 0)), 0.5, 0
            )
            cvshow(highlighted, f"最大区域 #{max_r['id']}: {max_r['area_str']}")

    return results, canvas, output


def batch_process(input_dir, method="canny", debug=False, scale=1.0, unit="px²",
                  min_area=50, save_output=None):
    """批量处理目录中的所有图像"""
    supported = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    files = sorted([f for f in os.listdir(input_dir)
                    if f.lower().endswith(supported)])

    if not files:
        print(f"在 {input_dir} 中未找到图像文件")
        return

    print(f"找到 {len(files)} 张图像，开始批量处理...")
    for fname in files:
        fpath = os.path.join(input_dir, fname)
        process_image(fpath, method, debug, scale, unit, min_area, save_output)


# ============================================================================
#  命令行入口
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="设计图边缘检测、分解与面积计算工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用 Canny 边缘检测处理单张图像
  python decompose_and_measure.py images/demo.png

  # 使用自适应阈值法并显示中间步骤
  python decompose_and_measure.py images/demo.png -m adaptive --debug

  # 使用比例尺计算物理面积（1像素=0.5mm）
  python decompose_and_measure.py images/demo.png -s 0.5 -u mm²

  # 批量处理整个目录
  python decompose_and_measure.py images2/ --batch

  # 使用 Harris 角点法处理
  python decompose_and_measure.py images/demo.png -m harris

  # 调整 Canny 阈值和最小面积过滤
  python decompose_and_measure.py images/demo.png --canny-low 30 --canny-high 100 --min-area 100
        """
    )

    parser.add_argument("input", help="输入图像文件或目录（配合 --batch）")
    parser.add_argument("-m", "--method", choices=["canny", "adaptive", "harris"],
                        default="canny", help="检测方法 (默认: canny)")
    parser.add_argument("--debug", action="store_true", help="显示中间处理步骤")
    parser.add_argument("-s", "--scale", type=float, default=1.0,
                        help="比例尺（每个像素对应的物理单位数，默认: 1.0）")
    parser.add_argument("-u", "--unit", default="px²",
                        help="物理单位名称 (默认: px²)")
    parser.add_argument("--min-area", type=float, default=50,
                        help="最小面积阈值，过滤小区域 (默认: 50)")
    parser.add_argument("--batch", action="store_true",
                        help="批量模式：输入路径为目录，处理所有图像")
    parser.add_argument("--save", help="保存结果图像的路径或目录")
    parser.add_argument("--canny-low", type=int, default=50,
                        help="Canny 低阈值 (默认: 50)")
    parser.add_argument("--canny-high", type=int, default=150,
                        help="Canny 高阈值 (默认: 150)")
    parser.add_argument("--close-kernel", type=int, default=5,
                        help="形态学闭合核大小 (默认: 5)")

    args = parser.parse_args()

    # 保存输出路径
    save_path = args.save

    if args.batch:
        batch_process(args.input, args.method, args.debug,
                      args.scale, args.unit, args.min_area, save_path)
    else:
        # 单张图像处理
        process_image(
            args.input, args.method, args.debug,
            args.scale, args.unit, args.min_area, save_path,
            canny_low=args.canny_low, canny_high=args.canny_high,
            close_kernel=args.close_kernel
        )
