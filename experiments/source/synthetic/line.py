"""
线数据源: 1像素宽的线 mask 生成。

支持位置、旋转、长度的变化。
"""
import numpy as np
from experiments.source.synthetic.config import IMG_SIZE


def make_line_mask(start_x, start_y, angle, length, img_size=IMG_SIZE):
    """
    生成1像素宽的线 mask。

    Args:
        start_x, start_y: 起点坐标
        angle: 线的角度（度，0=水平向右，90=垂直向下）
        length: 线的长度（像素）
        img_size: 图像大小

    Returns:
        mask: (img_size, img_size) 二值 mask
    """
    m = np.zeros((img_size, img_size), dtype=np.float32)

    # 计算终点
    rad = np.deg2rad(angle)
    end_x = start_x + length * np.cos(rad)
    end_y = start_y + length * np.sin(rad)

    # Bresenham 画线算法
    x0, y0 = int(round(start_x)), int(round(start_y))
    x1, y1 = int(round(end_x)), int(round(end_y))

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        if 0 <= x0 < img_size and 0 <= y0 < img_size:
            m[y0, x0] = 1.0

        if x0 == x1 and y0 == y1:
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return m


def generate_line_dataset(positions=None, angles=None, lengths=None, seed=42):
    """
    生成线 mask 数据集。

    Args:
        positions: 线的起点位置列表 [(x, y), ...]
        angles: 线的角度列表 [0, 45, 90, 135, ...]
        lengths: 线的长度列表 [10, 20, 30, ...]
        seed: 随机种子

    Returns:
        masks: (N, IMG_SIZE, IMG_SIZE) 二值 mask
        labels: {field: {value: [indices]}}
    """
    rng = np.random.RandomState(seed)
    s = IMG_SIZE

    # 默认值
    if positions is None:
        # 5x5 网格位置
        positions = [(x, y) for y in range(8, s, 12) for x in range(8, s, 12)]
    if angles is None:
        angles = [0, 45, 90, 135, 180, 225, 270, 315]
    if lengths is None:
        lengths = [15, 25, 35]

    masks = []
    labels = {}

    # 遍历所有组合
    for pos_x, pos_y in positions:
        for angle in angles:
            for length in lengths:
                # 确保线在图像内
                rad = np.deg2rad(angle)
                end_x = pos_x + length * np.cos(rad)
                end_y = pos_y + length * np.sin(rad)

                if 0 <= end_x < s and 0 <= end_y < s:
                    m = make_line_mask(pos_x, pos_y, angle, length, s)
                    masks.append(m)
                    idx = len(masks) - 1

                    # 添加标签
                    pos_key = f"({pos_x},{pos_y})"
                    angle_key = f"{angle}"
                    length_key = f"{length}"

                    labels.setdefault("position", {}).setdefault(pos_key, []).append(idx)
                    labels.setdefault("rotation", {}).setdefault(angle_key, []).append(idx)
                    labels.setdefault("length", {}).setdefault(length_key, []).append(idx)

                    # 添加 x/y 坐标标签
                    x_key = f"x={pos_x}"
                    y_key = f"y={pos_y}"
                    labels.setdefault("x_pos", {}).setdefault(x_key, []).append(idx)
                    labels.setdefault("y_pos", {}).setdefault(y_key, []).append(idx)

    if not masks:
        raise ValueError("没有生成任何线 mask，请检查参数")

    return np.stack(masks, axis=0).astype(np.float32), labels
