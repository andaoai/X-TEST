"""
通用可视化工具 —— 每个实验的 viz() 方法会调用这些函数。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
from sklearn.manifold import TSNE
from experiments.source.synthetic.config import SEED

# 中文字体
for _p in ['C:/Windows/Fonts/simhei.ttf','C:/Windows/Fonts/msyh.ttc',
           'C:/Windows/Fonts/msyhbd.ttc']:
    if Path(_p).exists():
        plt.rcParams['font.family'] = fm.FontProperties(fname=_p).get_name()
        break
plt.rcParams['axes.unicode_minus'] = False


def tsne_plot(emb: np.ndarray, labels: dict, palette: dict,
              title: str, out_path: Path, subtitle: str = ""):
    """通用 t-SNE 散点图"""
    n = len(emb)
    xy = TSNE(n_components=2, random_state=SEED,
              perplexity=min(30, n - 1)).fit_transform(emb)
    fig, ax = plt.subplots(figsize=(8, 7))
    for key, color in palette.items():
        if key not in labels:
            continue
        idxs = labels[key]
        ax.scatter(xy[idxs, 0], xy[idxs, 1], color=color, label=key,
                   alpha=0.7, s=25, edgecolors="black", linewidth=0.2)
    full = f"{title}\n{subtitle}" if subtitle else title
    ax.set_title(full, fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, ncol=2, loc="best")
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def summary_bar(results: dict, algo_name: str, out_path: Path):
    """所有实验的摘要条形图"""
    labels_text = []
    vals = []
    oks = []
    for exp_key in sorted(results.keys()):
        r = results[exp_key]
        labels_text.append(r.get("name", exp_key)[:8])
        vals.append(r.get("separation", 0))
        oks.append(r.get("is_correct", False))

    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ["#2ecc71" if ok else "#e74c3c" for ok in oks]
    bars = ax.bar(labels_text, vals, color=colors, edgecolor="black", linewidth=1)
    for b, v, ok in zip(bars, vals, oks):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.002,
                f"{v:.4f} {'OK' if ok else 'NO'}",
                ha="center", fontsize=11, fontweight="bold")
    ax.set_title(f"Mask Lab [{algo_name}]", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
