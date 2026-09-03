"""
位置编码(Positional Encoding)算法族。

思路: 不看像素值内容,只看"前景点在画面里的坐标"。
  mask (N,H,W) → 每个前景点 (x,y) 算出一个 128 维坐标特征向量
              → 对 mask 内所有点的向量取平均(mean 池化)→ L2 归一化 → (128,)

坐标先归一化到 [0,1](与分辨率无关,640 和 64 行为一致),
频率上限取奈奎斯特频率(每画面宽 W/2 个周期)。

4 种方法:
  pe_sincos  : Transformer 标准 sin/cos,几何级数频率 1→W/2,x/y 各占一半维度
  pe_fourier : 随机傅里叶特征(Rahimi & Recht),频率 log 均匀采样(seed 42 确定性)
  pe_rbf     : 128 个锚点上的高斯 RBF(exp(-距离²/2σ²)),x/y 联合编码
  pe_coord   : 归一化坐标 (ux,uy) + 确定性随机线性投影补足 128 维(基线)

注意: 这类编码显式表达位置,不具备平移不变性——
      "位置可分性"应通过、"平移独立性"应失败,二者互斥自洽。
"""
import numpy as np
from algorithms.base import BaseAlgorithm, EMBEDDING_DIM


# ─────────────────────────────────────────────────────────────────────────
# 单像素坐标特征场: (H*W, EMBEDDING_DIM)
# ─────────────────────────────────────────────────────────────────────────
def build_pe_field(method: str, H: int, W: int, dim: int = EMBEDDING_DIM,
                   seed: int = 42) -> np.ndarray:
    """返回 (H*W, dim) float32:每个像素位置的坐标特征向量。"""
    ys, xs = np.mgrid[0:H, 0:W]
    ux = (xs / max(W - 1, 1)).astype(np.float32).reshape(-1)   # (H*W,) ∈ [0,1]
    uy = (ys / max(H - 1, 1)).astype(np.float32).reshape(-1)
    f_max = max(W, H) / 2.0                                     # 奈奎斯特: 每画面宽最多 W/2 周期
    rng = np.random.RandomState(seed)
    field = np.zeros((H * W, dim), dtype=np.float32)

    if method == "sincos":
        half = dim // 2                                         # 每轴 64 维
        freqs = f_max ** (np.arange(0, half, 2) / half)         # 1 → f_max 几何级数
        ang = 2 * np.pi * freqs                                 # (half/2,)
        # x 占前 half 维: even=sin, odd=cos
        field[:, 0:half:2] = np.sin(ux[:, None] * ang)
        field[:, 1:half:2] = np.cos(ux[:, None] * ang)
        # y 占后 half 维
        field[:, half::2] = np.sin(uy[:, None] * ang)
        field[:, half + 1::2] = np.cos(uy[:, None] * ang)

    elif method == "fourier":
        n_freq = dim // 4                                       # 每频率 4 维: sin/cos × x/y
        log_f = rng.uniform(0, np.log(f_max), size=n_freq)      # log 均匀采样频率
        freqs = np.exp(log_f)
        ang = 2 * np.pi * freqs
        block = np.empty((H * W, n_freq, 4), dtype=np.float32)
        block[:, :, 0] = np.sin(ux[:, None] * ang)
        block[:, :, 1] = np.cos(ux[:, None] * ang)
        block[:, :, 2] = np.sin(uy[:, None] * ang)
        block[:, :, 3] = np.cos(uy[:, None] * ang)
        field = block.reshape(H * W, dim)

    elif method == "rbf":
        n_anchor = dim
        n_side = int(np.ceil(np.sqrt(n_anchor)))
        gx, gy = np.mgrid[0:n_side, 0:n_side]
        anchors = np.stack([gx.ravel()[:n_anchor], gy.ravel()[:n_anchor]], 1)
        anchors = (anchors + 0.5) / n_side                     # 网格中心 ∈ [0,1]
        anchors = anchors + rng.uniform(-0.02, 0.02, anchors.shape)  # 轻微抖动
        sigma = 1.0 / n_side                                   # ≈ 锚点间距
        pts = np.stack([ux, uy], 1)                            # (H*W, 2)
        d2 = ((pts[:, None, :] - anchors[None, :, :]) ** 2).sum(-1)
        field = np.exp(-d2 / (2 * sigma ** 2)).astype(np.float32)

    elif method == "coord":
        field[:, 0] = ux
        field[:, 1] = uy
        proj = rng.randn(2, dim - 2).astype(np.float32) * 0.5
        field[:, 2:] = np.stack([ux, uy], 1) @ proj

    else:
        raise ValueError(f"未知 PE 方法: {method!r}")

    return field


# ─────────────────────────────────────────────────────────────────────────
# 算法基类: mean 池化 + L2 归一化
# ─────────────────────────────────────────────────────────────────────────
class _PositionalEncodingBase(BaseAlgorithm):
    """坐标 → 位置特征场 → mask 内 mean 池化。"""
    uses_rgb = False
    method = "sincos"

    def encode(self, inputs: np.ndarray, verbose: bool = True) -> np.ndarray:
        inputs = np.asarray(inputs)
        N, H, W = inputs.shape[:3]
        field = build_pe_field(self.method, H, W, EMBEDDING_DIM)  # (H*W, 128)
        flat = inputs.reshape(N, -1) > 0                          # (N, H*W) bool

        emb = np.zeros((N, EMBEDDING_DIM), dtype=np.float32)
        for i in range(N):
            pts = flat[i]
            if pts.any():
                emb[i] = field[pts].mean(axis=0)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        return emb / np.maximum(norms, 1e-8)


class SinCosPEAlgo(_PositionalEncodingBase):
    name = "pe_sincos"
    method = "sincos"


class FourierPEAlgo(_PositionalEncodingBase):
    name = "pe_fourier"
    method = "fourier"


class RBFPEAlgo(_PositionalEncodingBase):
    name = "pe_rbf"
    method = "rbf"


class CoordPEAlgo(_PositionalEncodingBase):
    name = "pe_coord"
    method = "coord"
