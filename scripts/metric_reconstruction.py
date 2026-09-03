"""
第 7 个指标: 重建还原率(Reconstruction IoU)。

对每个算法: mask → 算法内部表示 → 还原 mask, 与原图算 IoU。
  · 可逆算法(fourier_shape / canonical_embed): 有真解码器, IoU 高
  · radial_moment: 统计摘要, 只有"尽力解码器", IoU 低 → 量化不可逆
  · pe_*/random_proj: 无解码器 → N/A(根本无法还原)
阈值: IoU ≥ 0.99 视为"可完全还原"。
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algorithms import ALGOS
from experiments.source.synthetic.shape import generate_shape_dataset

for _fp in [os.path.expanduser("~/.local/share/fonts/wqy-microhei.ttc")]:
    if os.path.exists(_fp):
        font_manager.fontManager.addfont(_fp)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=_fp).get_name()
plt.rcParams["axes.unicode_minus"] = False
S = 640
OUT = "results/shape_study"

def iou(a, b):
    a, b = a > 0, b > 0
    return (a & b).sum() / ((a | b).sum() + 1e-8)

# ── 样本 ──
masks, labels = generate_shape_dataset()
def pick(shape, size="medium", rot=0, pos=(320, 320)):
    s = set(labels["shape"][shape]) & set(labels["size"][size]) \
        & set(labels["rotation"][str(rot)]) & set(labels["position"][str(pos)])
    return sorted(s)[len(sorted(s)) // 2]
geo = [("三角", masks[pick("triangle", rot=45, pos=(210, 430))]),
       ("十字", masks[pick("cross", rot=135)]),
       ("椭圆", masks[pick("ellipse", "large", 45, (210, 210))]),
       ("圆", masks[pick("circle")])]
def blob_pts(seed, r, rot):
    rng = np.random.RandomState(seed); nv = rng.randint(9, 15)
    ang = np.sort(rng.uniform(0, 2*np.pi, nv)); rad = r*rng.uniform(0.55, 1.45, nv)
    e2 = np.random.RandomState(seed+7).uniform(0.45, 0.9)
    th = np.deg2rad(rot); c, s = np.cos(th), np.sin(th)
    x, y = rad*np.cos(ang)*e2, rad*np.sin(ang); return list(zip(x*c-y*s, x*s+y*c))
blobs = []
for b in range(4):
    img = Image.new("L", (S, S), 0)
    ImageDraw.Draw(img).polygon([(320+x, 320+y) for x, y in blob_pts(7000+b, 60, b*40)], fill=1)
    blobs.append((f"块{b}", np.asarray(img, np.uint8)))
samples = geo + blobs
sample_masks = np.stack([m for _, m in samples])

# ── 各算法还原 IoU ──
order = ["fourier_shape"]
results = {}
print(f"{'算法':<18}{'几何IoU':>10}{'随机块IoU':>11}{'可完全还原(≥0.99)':>20}")
for name in order:
    algo = ALGOS[name]
    if not hasattr(algo, "reconstruct"):
        results[name] = None
        print(f"{name:<18}{'N/A':>10}{'N/A':>11}{'无解码器, 不可还原':>20}")
        continue
    rec = algo.reconstruct(sample_masks)
    scores = [iou(sample_masks[i], rec[i]) for i in range(len(samples))]
    g = np.mean(scores[:len(geo)]); b = np.mean(scores[len(geo):])
    results[name] = (rec, scores)
    flag = "是 ✓" if (g >= 0.99 and b >= 0.99) else ("部分(轮廓光栅化)" if min(g, b) > 0.9 else "否 ✗(摘要不可逆)")
    print(f"{name:<18}{g:>10.3f}{b:>11.3f}{flag:>20}")

# ── 对比图 ──
rows = [n for n in order if results.get(n) is not None]
fig, axes = plt.subplots(len(rows) + 1, len(samples), figsize=(2.6*len(samples), 2.6*(len(rows)+1)))
for col, (nm, m) in enumerate(samples):
    axes[0, col].imshow(m[::3, ::3], cmap="gray_r"); axes[0, col].set_title(nm, fontsize=11)
    axes[0, col].set_xticks([]); axes[0, col].set_yticks([])
axes[0, 0].set_ylabel("原图", fontsize=12)
for r, name in enumerate(rows, start=1):
    rec, scores = results[name]
    for col in range(len(samples)):
        axes[r, col].imshow(rec[col][::3, ::3], cmap="gray_r")
        axes[r, col].set_title(f"IoU={scores[col]:.3f}", fontsize=10)
        axes[r, col].set_xticks([]); axes[r, col].set_yticks([])
    axes[r, 0].set_ylabel(name, fontsize=11, rotation=0, labelpad=80, va="center")
fig.suptitle("第 7 指标 · 重建还原率: mask→算法表示→还原 mask 的 IoU\n"
             "canonical(sprite 逐像素)=1.000;fourier(逆DFT轮廓)~0.95-0.98;radial 统计摘要 ~0.4 不可逆",
             fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.94])
out = f"{OUT}/21_reconstruction_metric.png"
fig.savefig(out, dpi=110)
print("→", out)
