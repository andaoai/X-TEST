"""
形状数据源: 640×640 二值几何形状 mask。

6 种形状(圆/正方形/三角形/椭圆/线/十字) × 5×5 位置网格 × 3 档大小 × 4 档旋转。
用于位置编码(PE)实验: 验证"像素坐标 → 向量"这类算法能否表达位置/形状/大小/旋转。

设计:
  - 形状用 PIL 多边形栅格化(填充),全部绕质心旋转
  - 圆用 24 边形近似(旋转不变);椭圆用 24 边形 + 旋转顶点
  - mask 存 uint8(0/1)节省内存(1800×640×640 ≈ 460MB,float32 要 1.8GB)
  - 首次生成后缓存到 results/_cache/,后续直接读盘;同进程内再走内存缓存

标签字段(统一 {field: {value: [indices]}} 格式):
  shape    : circle/square/triangle/ellipse/line/cross
  position : "(x,y)" 质心坐标
  size     : small/medium/large
  rotation : 0/45/90/135(度)
"""
import pickle

import numpy as np
from PIL import Image, ImageDraw

from experiments.source.synthetic.config import OUTPUT_ROOT, SEED

SHAPE_IMG_SIZE = 640

SHAPES = ["circle", "square", "triangle", "ellipse", "line", "cross"]
# 旋转下外观会变化的形状(圆/正方形旋转 45° 变菱形也算变化,但圆严格不变;
# 旋转实验只取非平凡形状,见实验文件)
ROTATION_SENSITIVE = ["triangle", "ellipse", "line", "cross"]

SIZES = {"small": 40, "medium": 58, "large": 78}   # 特征半径 r(像素)
ROTATIONS = [0, 45, 90, 135]
GRID_MARGIN = 100
GRID_STEPS = 5

_CACHE_DIR = OUTPUT_ROOT / "_cache"
_CACHE_NPZ = _CACHE_DIR / "shape_dataset.npz"
_CACHE_LABELS = _CACHE_DIR / "shape_labels.pkl"


# ─────────────────────────────────────────────────────────────────────────
# 形状顶点(规范形,质心在原点;y 轴向下为正,与图像坐标一致)
# ─────────────────────────────────────────────────────────────────────────
def _polygon_vertices(kind: str, r: float, angle_deg: float):
    """返回 (polys, width_lines):polys = 多边形顶点列表(每个是 [(x,y),...]),
    width_lines = 粗线 [(x1,y1,x2,y2,width),...](用于十字的横臂,也可用多边形)。
    全部统一用多边形表示。
    """
    th = np.deg2rad(angle_deg)
    cos_t, sin_t = np.cos(th), np.sin(th)

    def rot(pts):
        pts = np.asarray(pts, dtype=np.float64)
        return [(x * cos_t - y * sin_t, x * sin_t + y * cos_t) for x, y in pts]

    def ngon(n, rx, ry=None, phase=np.pi / 2):
        ry = rx if ry is None else ry
        a = np.linspace(0, 2 * np.pi, n, endpoint=False) + phase
        return list(zip(rx * np.cos(a), ry * np.sin(a)))

    polys = []
    if kind == "circle":
        polys.append(ngon(24, r))                       # 旋转不变,直接 24 边形
    elif kind == "square":
        h = r * 0.85
        base = [(-h, -h), (h, -h), (h, h), (-h, h)]
        polys.append(rot(base))
    elif kind == "triangle":
        polys.append(rot(ngon(3, r)))
    elif kind == "ellipse":
        polys.append(rot(ngon(24, r, r * 0.55)))
    elif kind == "line":
        half_l, half_w = r, r * 0.14
        base = [(-half_l, -half_w), (half_l, -half_w),
                (half_l, half_w), (-half_l, half_w)]
        polys.append(rot(base))
    elif kind == "cross":
        arm, half_w = r * 0.9, r * 0.22
        vert = [(-half_w, -arm), (half_w, -arm), (half_w, arm), (-half_w, arm)]
        horz = [(-arm, -half_w), (arm, -half_w), (arm, half_w), (-arm, half_w)]
        polys.append(rot(vert))
        polys.append(rot(horz))
    else:
        raise ValueError(f"未知形状: {kind}")
    return polys


def _rasterize(polys, cx, cy, size=SHAPE_IMG_SIZE):
    """把质心在 (cx, cy) 的多边形栅格化成 (size, size) uint8 mask。"""
    img = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(img)
    for pts in polys:
        abs_pts = [(cx + x, cy + y) for x, y in pts]
        d.polygon(abs_pts, fill=1)
    return np.asarray(img, dtype=np.uint8)


def _generate(seed=SEED, verbose=False):
    rng = np.random.RandomState(seed)
    s = SHAPE_IMG_SIZE
    grid = np.linspace(GRID_MARGIN, s - GRID_MARGIN, GRID_STEPS).astype(int)

    masks = []
    labels: dict = {}

    def _tag(field, value, idx):
        labels.setdefault(field, {}).setdefault(str(value), []).append(idx)

    for shape in SHAPES:
        for size_name, r in SIZES.items():
            for rot in ROTATIONS:
                polys0 = _polygon_vertices(shape, r, rot)
                for cy in grid:
                    for cx in grid:
                        m = _rasterize(polys0, int(cx), int(cy), s)
                        idx = len(masks)
                        masks.append(m)
                        _tag("shape", shape, idx)
                        _tag("size", size_name, idx)
                        _tag("rotation", rot, idx)
                        _tag("position", (int(cx), int(cy)), idx)

    masks = np.stack(masks, axis=0)  # (N, 640, 640) uint8
    if verbose:
        print(f"  shape 数据集: {masks.shape}, 标签字段 {list(labels)}")
    return masks, labels


_MEM_CACHE: dict = {}


def generate_shape_dataset(seed=SEED, verbose=False, use_cache=True):
    """生成(或读缓存)形状数据集。返回 (masks uint8, labels)。

    三级缓存: 进程内内存 → results/_cache/ 磁盘 → 实时生成。
    """
    if seed in _MEM_CACHE:
        return _MEM_CACHE[seed]

    if use_cache and _CACHE_NPZ.exists() and _CACHE_LABELS.exists():
        masks = np.load(_CACHE_NPZ)["masks"]
        with open(_CACHE_LABELS, "rb") as f:
            labels = pickle.load(f)
        if verbose:
            print(f"  shape 数据集(磁盘缓存): {masks.shape}")
    else:
        masks, labels = _generate(seed=seed, verbose=verbose)
        if use_cache:
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(_CACHE_NPZ, masks=masks)
            with open(_CACHE_LABELS, "wb") as f:
                pickle.dump(labels, f)

    _MEM_CACHE[seed] = (masks, labels)
    return masks, labels


if __name__ == "__main__":
    import cv2
    masks, labels = generate_shape_dataset(verbose=True)
    print("每类标签数量:", {k: len(v) for k, v in labels.items()})
    # 拼一张 6×4 的缩略图总览(形状×旋转,固定 medium/中心位)
    cells = []
    for shape in SHAPES:
        row = []
        idxs = labels["shape"][shape]
        # 取 medium size、中心位置的 4 个旋转
        for rot in ROTATIONS:
            cand = sorted(set(idxs) & set(labels["size"]["medium"])
                          & set(labels["rotation"][str(rot)]))
            m = masks[cand[len(cand) // 2]]
            small = cv2.resize(m * 255, (128, 128), interpolation=cv2.INTER_NEAREST)
            row.append(small)
        cells.append(np.hstack(row))
    grid_img = np.vstack(cells)
    out = OUTPUT_ROOT / "_cache" / "shape_overview.png"
    cv2.imwrite(str(out), grid_img)
    print(f"总览图 → {out}")
