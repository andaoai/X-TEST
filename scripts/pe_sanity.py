"""
位置编码(PE)数学性质体检 —— 不经过实验框架,直接检验坐标特征场本身。

3 项检查(4 种方法各跑一遍):
  1. 正交性: 网格点 PE 向量两两应近似正交(Gram 矩阵非对角元应接近 0)
  2. 可加性: PE(a+b) 是否 ≈ PE(a) + PE(b)?
     —— 注意: sin/cos 不满足可加性(sin(a+b) ≠ sin a + sin b),
        这里如实测量,不预设结论。
  3. 距离-相似度: 两点 PE 的余弦相似度随空间距离怎么衰减
     (位置编码的"分辨尺度")。

输出:
  results/pe_sanity/gram.png          4 种方法的 Gram 矩阵
  results/pe_sanity/distance_curve.png  相似度 vs 距离曲线
  results/pe_sanity/metrics.json       数值结果

用法:
  uv run python scripts/pe_sanity.py
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.positional import build_pe_field
from algorithms.base import EMBEDDING_DIM
from experiments.source.synthetic.config import OUTPUT_ROOT

METHODS = ["sincos", "fourier", "rbf", "coord"]
GRID = 21          # 正交性采样网格 21×21
N_PAIRS = 4000     # 距离曲线采样点对数
SEED = 42

OUT = OUTPUT_ROOT / "pe_sanity"
OUT.mkdir(parents=True, exist_ok=True)


def field_at(field2d, u, v):
    """从 (H,W,D) 特征场按归一化坐标最近邻取值 → (...,D)。"""
    H, W, D = field2d.shape
    xi = np.clip((u * (W - 1)).round().astype(int), 0, W - 1)
    yi = np.clip((v * (H - 1)).round().astype(int), 0, H - 1)
    return field2d[yi, xi]


def cosine(a, b):
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
    return (a * b).sum(-1)


def main():
    rng = np.random.RandomState(SEED)
    H = W = 640
    metrics = {}

    # ── 1. 正交性 ──
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    sample_u = np.linspace(0.05, 0.95, GRID)
    uu, vv = np.meshgrid(sample_u, sample_u)
    uu, vv = uu.ravel(), vv.ravel()

    for ax, method in zip(axes, METHODS):
        field = build_pe_field(method, H, W, EMBEDDING_DIM).reshape(H, W, EMBEDDING_DIM)
        vecs = field_at(field, uu, vv)                       # (G, D)
        vecs_n = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8)
        gram = vecs_n @ vecs_n.T
        off = ~np.eye(len(uu), dtype=bool)
        mean_off = float(np.abs(gram[off]).mean())
        max_off = float(np.abs(gram[off]).max())
        metrics.setdefault(method, {})["orth_mean_abs"] = mean_off
        metrics[method]["orth_max_abs"] = max_off

        im = ax.imshow(gram, cmap="RdBu", vmin=-1, vmax=1)
        ax.set_title(f"{method}\nmean|off|={mean_off:.3f} max|off|={max_off:.3f}",
                     fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes, fraction=0.02)
    fig.suptitle("Gram matrices of PE vectors at 21x21 grid points (off-diagonal should be ~0)")
    fig.tight_layout()
    fig.savefig(OUT / "gram.png", dpi=110)
    plt.close(fig)

    # ── 2. 可加性 ──
    # PE(a+b) vs PE(a)+PE(b),a/b 取小位移保证 a+b 不出界
    for method in METHODS:
        field = build_pe_field(method, H, W, EMBEDDING_DIM).reshape(H, W, EMBEDDING_DIM)
        base_u = rng.uniform(0.2, 0.7, N_PAIRS)
        base_v = rng.uniform(0.2, 0.7, N_PAIRS)
        du = rng.uniform(0.02, 0.15, N_PAIRS)                # b 点在原点附近
        dv = rng.uniform(0.02, 0.15, N_PAIRS)
        pe_a = field_at(field, base_u, base_v)
        pe_b = field_at(field, du, dv)                       # f(b): 位移(近原点)
        # 可加性: f(a+b) ≈ f(a) + f(b)
        pe_ab = field_at(field, base_u + du, base_v + dv)
        sim = cosine(pe_ab, pe_a + pe_b)
        metrics[method]["additivity_cos"] = float(np.mean(sim))

    # ── 3. 距离-相似度曲线 ──
    fig, ax = plt.subplots(figsize=(7, 5))
    dist_bins = np.linspace(0.02, 0.9, 18)
    for method in METHODS:
        field = build_pe_field(method, H, W, EMBEDDING_DIM).reshape(H, W, EMBEDDING_DIM)
        u1 = rng.uniform(0, 1, N_PAIRS); v1 = rng.uniform(0, 1, N_PAIRS)
        ang = rng.uniform(0, 2 * np.pi, N_PAIRS)
        d = rng.uniform(0.02, 0.9, N_PAIRS)
        u2 = np.clip(u1 + d * np.cos(ang), 0, 1)
        v2 = np.clip(v1 + d * np.sin(ang), 0, 1)
        p1 = field_at(field, u1, v1)
        p2 = field_at(field, u2, v2)
        sim = cosine(p1, p2)
        actual_d = np.sqrt((u1 - u2) ** 2 + (v1 - v2) ** 2)
        bin_idx = np.digitize(actual_d, dist_bins)
        curve = [sim[bin_idx == k].mean() if (bin_idx == k).any() else np.nan
                 for k in range(1, len(dist_bins))]
        ax.plot(dist_bins[1:], curve, marker="o", label=method)
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("spatial distance (normalized)")
    ax.set_ylabel("mean cosine similarity")
    ax.set_title("PE similarity vs spatial distance (faster decay = finer position scale)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "distance_curve.png", dpi=110)
    plt.close(fig)

    # ── 汇总 ──
    print(f"{'method':<10}{'正交 mean|off|':>14}{'正交 max|off|':>14}{'可加性 cos':>12}")
    for m in METHODS:
        d = metrics[m]
        print(f"{m:<10}{d['orth_mean_abs']:>14.4f}{d['orth_max_abs']:>14.4f}"
              f"{d['additivity_cos']:>12.4f}")
    with open(OUT / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n图 → {OUT}/gram.png , {OUT}/distance_curve.png")


if __name__ == "__main__":
    main()
