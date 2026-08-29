"""
Token-level + Image-level 双层演化可视化(流式:每 epoch 立即画 PNG)。

每训练完一轮立即输出:
  - 1 张 PNG 帧(results/train_<tag>/frame_eXX.png)
  - 日志追加 1 行
  - 跑完自动合成 GIF

帧布局:
  - 上半部分:TOKEN 空间(底层 32-d)+ 前/背景数量饼图
  - 下半部分:IMAGE 空间(聚合后 128-d)· 4 个属性子图

用法:
  cd /home/andaoai/wf/work-dir/github.com/andaoai/X-TEST
  uv run python scripts/train_token_evolution.py [--epochs 30]
"""
import argparse
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).parent.parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from PIL import Image

from algorithms import ALGOS
from algorithms.base import EMBEDDING_DIM
from algorithms.color_tokens import _hsv_decompose
from experiments.source.synthetic.data import Dataset
from experiments.source.synthetic.config import POS_HEX, ROTATION_HEX

# ── 中文字体 ──
for _p in [
    os.path.expanduser("~/.local/share/fonts/wqy-microhei.ttc"),
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
]:
    if Path(_p).exists():
        from matplotlib import font_manager as fm
        plt.rcParams["font.family"] = fm.FontProperties(fname=_p).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False


def get_subset(labels_full, attr, max_per_value):
    out = []
    for v, idxs in labels_full[attr].items():
        if len(idxs) > max_per_value:
            idxs = list(np.random.RandomState(0).choice(idxs, max_per_value, replace=False))
        out.extend(idxs)
    return sorted(out)


def remap(labels_full, sub_idx):
    remap = {old: new for new, old in enumerate(sub_idx)}
    return {
        f: {v: [remap[i] for i in idxs if i in remap] for v, idxs in classes.items()}
        for f, classes in labels_full.items()
    }


def encode_image_batch(algo, rgb_batch):
    algo.eval_all()
    out = np.zeros((len(rgb_batch), EMBEDDING_DIM), dtype=np.float32)
    with torch.no_grad():
        for i in range(len(rgb_batch)):
            out[i] = algo._encode_one_train(rgb_batch[i]).cpu().numpy()
    return out


def encode_tokens(algo, rgb_batch):
    """直接调 algo.encode_tokens,新架构已实现"""
    return algo.encode_tokens(rgb_batch)


def tsne_2d(emb, seed=42):
    n = len(emb)
    perp = min(30, max(5, n // 20))
    return TSNE(n_components=2, random_state=seed,
                perplexity=perp, init="pca",
                learning_rate="auto").fit_transform(emb)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-per-class", type=int, default=50)
    p.add_argument("--fps", type=float, default=4.0)
    p.add_argument("--tag", default="token_evo")
    args = p.parse_args()

    out_dir = _HERE / "results" / f"train_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"

    print("=" * 70)
    print(f" Token+Image 双层演化(epochs={args.epochs}, max_per_class={args.max_per_class})")
    print("=" * 70)

    # ── 数据 ──
    print("\n[1/4] 加载数据 + 子集...")
    t0 = time.time()
    ds = Dataset().generate(verbose=False)
    masks_full = ds.masks()
    labels_full = ds.labels

    sub_idx = get_subset(labels_full, "label", args.max_per_class)
    masks_sub = masks_full[sub_idx]
    rgb_sub = np.repeat(masks_sub[..., None], 3, axis=-1).astype(np.float32)
    labels_sub = remap(labels_full, sub_idx)

    sample_label = np.empty(len(sub_idx), dtype=object)
    for cls, idxs in labels_sub["label"].items():
        for i in idxs:
            sample_label[i] = cls
    print(f"  全集 {masks_full.shape[0]} → 子集 {len(sub_idx)} 张,{time.time()-t0:.1f}s")

    # ── 模型 + 多任务头 ──
    print("\n[2/4] 初始化模型 + 多任务分类头...")
    t0 = time.time()
    algo = ALGOS["color_tokens"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    algo._build(device)

    tasks = ("label", "position", "rotation", "size")
    algo.clfs = nn.ModuleDict()
    ys = {}
    for task in tasks:
        n_classes = len(labels_sub[task])
        torch.manual_seed(42)
        # 新架构:分类头接在 cls_trunk 输出(64 维)
        algo.clfs[task] = nn.Linear(64, n_classes).to(device)
        cls_to_int = {c: i for i, c in enumerate(sorted(labels_sub[task].keys()))}
        y = np.full(len(rgb_sub), -1, dtype=np.int64)
        for cls, idxs in labels_sub[task].items():
            for i in idxs:
                y[i] = cls_to_int[cls]
        ys[task] = y

    params = (list(algo.encoder.parameters()) +
              list(algo.geom_mlp.parameters()) +
              list(algo.color_mlp.parameters()) +
              list(algo.fusion.parameters()) +
              list(algo.self_attn.parameters()) +
              list(algo.emb_head.parameters()) +
              list(algo.cls_trunk.parameters()))
    for t in tasks:
        params += list(algo.clfs[t].parameters())
    opt = torch.optim.Adam(params, lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    print(f"  device={device}, {time.time()-t0:.1f}s")

    # 颜色映射
    unique_letters = sorted(set(sample_label))
    cmap_letter = plt.get_cmap("tab20")
    letter_color = {l: cmap_letter(i % 20 / 20) for i, l in enumerate(unique_letters)}

    attr_list = [
        ("label",    "letter (20 类)",       None),
        ("position", "position (5 类)",      POS_HEX),
        ("rotation", "rotation (4 类)",      ROTATION_HEX),
        ("size",     "size (3 类)",
         {"small": "#3b82f6", "medium": "#10b981", "large": "#f59e0b"}),
    ]

    def render_frame(ep, img_emb, token_emb, token_sid, token_fg, accs, out_path):
        fig = plt.figure(figsize=(20, 11))
        gs = fig.add_gridspec(2, 4, height_ratios=[1, 1], hspace=0.25, wspace=0.3)

        ax_token = fig.add_subplot(gs[0, :2])
        ax_token_fg = fig.add_subplot(gs[0, 2:])

        xy_t = tsne_2d(token_emb, seed=42)
        token_label = np.array([sample_label[s] for s in token_sid])

        for l in unique_letters:
            m_fg = (token_label == l) & (token_fg == 1)
            if m_fg.any():
                ax_token.scatter(xy_t[m_fg, 0], xy_t[m_fg, 1],
                                 color=[letter_color[l]], marker="o", s=18,
                                 alpha=0.7, edgecolors="none")
            m_bg = (token_label == l) & (token_fg == 0)
            if m_bg.any():
                ax_token.scatter(xy_t[m_bg, 0], xy_t[m_bg, 1],
                                 color=[letter_color[l]], marker="^", s=22,
                                 alpha=0.5, edgecolors="black", linewidths=0.3)
        ax_token.set_title(f"TOKEN 空间(底层 32-d) · {len(token_emb)} 个 token\n"
                           f"●前景 ▲背景 · 颜色=字符",
                           fontsize=11, fontweight="bold")
        ax_token.set_xticks([]); ax_token.set_yticks([])

        n_fg = int((token_fg == 1).sum())
        n_bg = int((token_fg == 0).sum())
        ax_token_fg.pie([n_fg, n_bg], labels=[f"前景\n{n_fg}", f"背景\n{n_bg}"],
                        colors=["#3b82f6", "#94a3b8"], autopct="", startangle=90)
        ax_token_fg.set_title("前/背景 token 数量", fontsize=11, fontweight="bold")

        xy_i = tsne_2d(img_emb, seed=42)
        for col, (field, title, palette) in enumerate(attr_list):
            ax = fig.add_subplot(gs[1, col])
            for v, idxs in labels_sub[field].items():
                if palette is not None and v in palette:
                    color = palette[v]
                elif field == "label":
                    color = letter_color.get(v, "#666666")
                else:
                    color = "#666666"
                idxs_arr = np.array(idxs)
                ax.scatter(xy_i[idxs_arr, 0], xy_i[idxs_arr, 1],
                           color=color, alpha=0.7, s=10,
                           edgecolors="none", label=v)
            if col == 3:
                ax.legend(fontsize=7, loc="upper right",
                          framealpha=0.7, markerscale=2)
            ax.set_title(title, fontsize=11)
            ax.set_xticks([]); ax.set_yticks([])

        accs_str = "  ".join(f"{t}={accs[t]:.0%}" for t in tasks)
        fig.suptitle(f"color_tokens 双层演化 · epoch {ep}/{args.epochs}\n"
                     f"acc: {accs_str}",
                     fontsize=14, fontweight="bold", y=0.995)
        plt.tight_layout()
        plt.savefig(out_path, dpi=80, bbox_inches="tight")
        plt.close()

    # ── 训练循环 + 每 epoch 立即画 PNG ──
    print(f"\n[3/4] 训练 {args.epochs} epoch · 每 epoch 立即画 PNG 帧...")
    N = len(rgb_sub)
    frame_paths = []
    train_t0 = time.time()

    for epoch in range(args.epochs + 1):
        if epoch > 0:
            algo.train_all()

            perm = np.random.permutation(N)
            total_loss = 0.0
            correct = {t: 0 for t in tasks}
            for i in perm:
                pool = algo._pool_one(rgb_sub[i])
                trunk = torch.nn.functional.relu(algo.cls_trunk(pool))
                opt.zero_grad()
                loss = torch.tensor(0.0, device=device)
                for task in tasks:
                    logits = algo.clfs[task](trunk)
                    target = torch.tensor([ys[task][i]], device=device)
                    loss = loss + loss_fn(logits.unsqueeze(0), target)
                    if logits.argmax().item() == ys[task][i]:
                        correct[task] += 1
                loss.backward()
                opt.step()
                total_loss += loss.item()
            accs = {t: correct[t] / N for t in tasks}
            avg_loss = total_loss / N
        else:
            accs = {t: 0.0 for t in tasks}
            avg_loss = 0.0

        t_frame_start = time.time()
        img_emb = encode_image_batch(algo, rgb_sub)
        token_emb, token_sid, token_fg = encode_tokens(algo, rgb_sub)
        fp = out_dir / f"frame_e{epoch:02d}.png"
        render_frame(epoch, img_emb, token_emb, token_sid, token_fg, accs, fp)
        frame_paths.append(fp)
        frame_dur = time.time() - t_frame_start
        elapsed = time.time() - train_t0

        msg = (f"  ep{epoch:>2}/{args.epochs}  loss={avg_loss:.3f}  "
               f"{'  '.join(f'{t}={accs[t]:.0%}' for t in tasks)}  "
               f"frame={frame_dur:.1f}s  total={elapsed:.0f}s  →  {fp.name}")
        with open(log_path, "a") as f:
            f.write(msg + "\n")
        print(msg, flush=True)

    # ── 合成 GIF ──
    print(f"\n[4/4] 合成 GIF...")
    t0 = time.time()
    images = [Image.open(fp).convert("RGB") for fp in frame_paths]
    gif_path = out_dir / "token_image_evolution.gif"
    images[0].save(
        gif_path, save_all=True, append_images=images[1:],
        duration=int(1000 / args.fps), loop=0, optimize=False,
    )
    print(f"  GIF 耗时 {time.time()-t0:.1f}s")
    print(f"\n  ✓ PNG 帧: {out_dir}/frame_e*.png ({len(frame_paths)} 张)")
    print(f"  ✓ GIF: {gif_path}({len(images)} 帧,{len(images)/args.fps:.1f}s)")
    print(f"  ✓ log: {log_path}")
    print(f"\n  实时查看最新一帧:")
    print(f"    open {frame_paths[-1]}")


if __name__ == "__main__":
    main()
