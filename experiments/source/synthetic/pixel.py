"""
像素数据源: 单像素 / 多像素 mask 生成。

人类能一眼看到黑屏上的白点并定位。
Gabor 全局积分后的 embedding 能做到吗？
"""
import numpy as np
from experiments.source.synthetic.config import IMG_SIZE


def make_single_pixel_mask(x, y, img_size=IMG_SIZE):
    """只有 (x,y) 处为 1，其余为 0"""
    m = np.zeros((img_size, img_size), dtype=np.float32)
    m[y, x] = 1.0
    return m


def make_multi_pixel_mask(positions, img_size=IMG_SIZE):
    """多个指定位置为 1"""
    m = np.zeros((img_size, img_size), dtype=np.float32)
    for x, y in positions:
        m[y, x] = 1.0
    return m


def generate_pixel_dataset(grid_step=4, n_multi=None, seed=42):
    """
    生成像素 mask 数据集。

    Args:
        grid_step: 单点网格步长 (px)。4 则生成 16×16=256 个单点 mask
        n_multi:  每个多点 K 值生成的样本数。None=自动

    Returns:
        masks:    (N, IMG_SIZE, IMG_SIZE) 二值 mask
        labels:   {"pixel": {"single_0_0": [0], ..., "multi_2pt": [...]}}
        metadata: [{"type": "single"|"multi_k", "pos": (x,y) or [...], ...}]
    """
    rng = np.random.RandomState(seed)
    s = IMG_SIZE
    masks, labels, metas = [], {}, []

    # ── 单点: 所有 grid_step 间隔的位置 ──
    labels["single_pos"] = {}
    labels["single"] = []
    for y in range(0, s, grid_step):
        for x in range(0, s, grid_step):
            m = make_single_pixel_mask(x, y)
            masks.append(m)
            idx = len(masks) - 1
            key = f"({x},{y})"
            labels["single_pos"][key] = [idx]
            labels["single"].append(idx)
            metas.append({"type": "single", "pos": (x, y), "x": x, "y": y})

    n_single = len(labels["single"])
    # 按 x/y 分组方便实验
    labels["x_pos"] = {}
    labels["y_pos"] = {}
    for i, idx in enumerate(labels["single"]):
        meta = metas[idx]
        xk = f"x={meta['x']}"
        yk = f"y={meta['y']}"
        labels["x_pos"].setdefault(xk, []).append(idx)
        labels["y_pos"].setdefault(yk, []).append(idx)

    # ── 多点: 2/3/5/10 个随机位置 ──
    if n_multi is None:
        n_multi = n_single // 4
    labels["multi_k"] = {}
    for k in [2, 3, 5, 10]:
        labels["multi_k"][str(k)] = []
        for _ in range(n_multi):
            pts = [(rng.randint(0, s), rng.randint(0, s)) for _ in range(k)]
            m = make_multi_pixel_mask(pts)
            masks.append(m)
            idx = len(masks) - 1
            labels["multi_k"][str(k)].append(idx)
            metas.append({"type": f"multi_{k}", "points": pts, "k": k})

    # ── 数量标签 (用于 1点 vs N点 对比) ──
    labels["count"] = {"1pt": list(range(n_single))}
    for k in [2, 3, 5, 10]:
        labels["count"][f"{k}pt"] = labels["multi_k"][str(k)]

    return np.stack(masks, axis=0).astype(np.float32), labels, metas
