"""诊断可视化:把诊断数据画成图,方便肉眼确认。

跑: uv run python scripts/visualize_diagnose.py
"""
import os, sys, random, string
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.source.synthetic.word_data import WordDataset
from algorithms.image_decomposition import ColorTokenEncoder
from scripts.train_token2text import (
    build_targets, encode_tokens, LETTER2ID, ID2LETTER, BG_ID,
    DEVICE, OUT_DIM, N_CLASSES,
)


def quick_train(epochs=10, samples_per_word=20, fuse_mode="gate"):
    out_dir = Path("results/diag_vis")
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = WordDataset(samples_per_word=samples_per_word, invalid_ratio=0.0)
    ds.generate(verbose=False)
    valid = [s for s in ds.samples
             if 2 <= len(s["target_word"]) <= 5 and s["word_match"]]
    random.Random(42).shuffle(valid)
    n_val = int(len(valid) * 0.15)
    val_samples, train_samples = valid[:n_val], valid[n_val:]
    encoder = ColorTokenEncoder(fuse_mode=fuse_mode, out_dim=OUT_DIM).to(DEVICE).train()
    head = nn.Linear(OUT_DIM, N_CLASSES).to(DEVICE).train()
    opt = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=1e-3)
    for ep in range(epochs):
        random.Random(42 + ep).shuffle(train_samples)
        sum_loss, correct, total = 0, 0, 0
        for s in train_samples:
            try:
                vecs, tokens = encode_tokens(s["rgb"], encoder, char_hints=list(s["rendered_word"]))
            except Exception:
                continue
            targets = build_targets(len(tokens), s["rendered_word"]).to(DEVICE)
            logits = head(vecs)
            loss = F.cross_entropy(logits, targets)
            opt.zero_grad(); loss.backward(); opt.step()
            sum_loss += loss.item()
            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                correct += (pred == targets).sum().item()
            total += len(targets)
        print(f"  ep {ep}: loss={sum_loss/len(train_samples):.3f} tok_acc={correct/max(1,total):.3f}")
    return encoder, head, val_samples, out_dir


def collect(encoder, head, val_samples):
    encoder.eval(); head.eval()
    records = []
    embs, char_ids, pos_ids = [], [], []
    word_correct = 0
    with torch.no_grad():
        for s in val_samples:
            word = s["target_word"]
            rendered = s["rendered_word"]
            try:
                vecs, tokens = encode_tokens(s["rgb"], encoder, char_hints=list(rendered))
            except Exception:
                continue
            targets = build_targets(len(tokens), rendered).to(DEVICE)
            logits = head(vecs)
            probs = F.softmax(logits, dim=-1)
            pred = logits.argmax(dim=-1).cpu().tolist()
            for j in range(len(targets)):
                gt = int(targets[j].item())
                pr = pred[j]
                gt_p = float(probs[j][gt].item())
                records.append((gt, pr, gt_p, j > 0))
                if j > 0 and gt != BG_ID:
                    embs.append(vecs[j].cpu().numpy())
                    char_ids.append(gt)
                    pos_ids.append(j - 1)
            fg_pred = pred[1:]
            decoded = "".join(ID2LETTER.get(p, "?") if p < 26 else "?" for p in fg_pred)
            if decoded == word:
                word_correct += 1
    print(f"\n>> 收集 {len(records)} token,val 整词 acc: {word_correct}/{len(val_samples)} = {word_correct/len(val_samples):.2%}")
    return records, np.stack(embs), np.array(char_ids), np.array(pos_ids)


def plot_confusion(records, out_path):
    fg_records = [r for r in records if r[3] and r[0] != BG_ID]
    print(f">> 前景 token 总数: {len(fg_records)}")
    cm = np.zeros((26, 26), dtype=np.int32)
    for gt, pr, _, _ in fg_records:
        if gt < 26 and pr < 26:
            cm[gt, pr] += 1
    row_sum = cm.sum(axis=1, keepdims=True) + 1e-8
    cm_norm = cm / row_sum
    fig, ax = plt.subplots(1, 1, figsize=(11, 10))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=0.5)
    labels = list(string.ascii_lowercase)
    ax.set_xticks(range(26))
    ax.set_yticks(range(26))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("预测")
    ax.set_ylabel("真值")
    ax.set_title(f"混淆矩阵(前景 token,val {len(fg_records)} 个)\n对角线=正确预测;非对角=误判")
    for i in range(26):
        for j in range(26):
            if cm[i, j] >= 3:
                color = "white" if cm_norm[i, j] > 0.3 else "black"
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=7)
    plt.colorbar(im, ax=ax, label="row-normalized")
    fig.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f">> saved {out_path}")


def plot_pos_confidence(records, out_path):
    char_probs = defaultdict(list)
    for gt, pr, gt_p, is_fg in records:
        if is_fg and gt < 26:
            char_probs[chr(ord("a") + gt)].append(gt_p)
    chars_sorted = sorted(char_probs.keys())
    fig, ax = plt.subplots(1, 1, figsize=(14, 5))
    data = [char_probs[c] for c in chars_sorted]
    bp = ax.boxplot(data, tick_labels=chars_sorted, showmeans=True)
    ax.set_ylabel("GT 字符的预测概率")
    ax.set_title("每个字符的预测置信度(箱线图,值=GT 类的 softmax 概率)")
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.5, label="0.5 阈值")
    ax.axhline(1/27, color="gray", linestyle=":", alpha=0.5, label=f"基线 {1/27:.3f}")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f">> saved {out_path}")


def plot_emb_pca(embs, char_ids, pos_ids, out_path):
    if len(embs) < 5:
        return
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    pca = PCA(n_components=2)
    xy = pca.fit_transform(embs)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    ax = axes[0]
    cmap = plt.get_cmap("tab20")
    unique_chars = sorted(set(char_ids.tolist()))
    for c_id in unique_chars:
        mask = char_ids == c_id
        label = chr(ord("a") + c_id) if c_id < 26 else "BG"
        ax.scatter(xy[mask, 0], xy[mask, 1], c=[cmap(c_id % 20)], label=label,
                   s=30, alpha=0.7, edgecolors="black", linewidth=0.3)
    ax.set_title(f"按字符({len(unique_chars)} 类)\n同色 = 同一字符,应该聚一起")
    ax.legend(loc="upper right", fontsize=7, ncol=3, markerscale=1.2)
    ax = axes[1]
    pos_colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]
    unique_pos = sorted(set(pos_ids.tolist()))
    for p in unique_pos:
        mask = pos_ids == p
        ax.scatter(xy[mask, 0], xy[mask, 1], c=[pos_colors[p % 5]], label=f"pos {p}",
                   s=30, alpha=0.7, edgecolors="black", linewidth=0.3)
    ax.set_title(f"按位置({len(unique_pos)} 个)\n同色 = 同一位置")
    ax.legend(loc="upper right", fontsize=9)
    sil_c = silhouette_score(embs, char_ids)
    sil_p = silhouette_score(embs, pos_ids)
    fig.suptitle(f"embedding 2D PCA  |  silhouette(字符)={sil_c:.3f}  silhouette(位置)={sil_p:.3f}",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f">> saved {out_path}")


def main():
    print(">> 训练 10 epoch (20/词)...")
    encoder, head, val_samples, out_dir = quick_train(epochs=10, samples_per_word=20)
    print(">> 收集数据...")
    records, embs, char_ids, pos_ids = collect(encoder, head, val_samples)
    print(">> 画混淆矩阵...")
    plot_confusion(records, out_dir / "diag_confusion.png")
    print(">> 画置信度分布...")
    plot_pos_confidence(records, out_dir / "diag_confidence.png")
    print(">> 画 embedding PCA...")
    plot_emb_pca(embs, char_ids, pos_ids, out_dir / "diag_emb_pca.png")
    print(f"\n>> 全部图在 {out_dir}/")
    print("  - diag_confusion.png  (字符混淆矩阵)")
    print("  - diag_confidence.png (字符预测置信度)")
    print("  - diag_emb_pca.png    (embedding PCA,左按字符,右按位置)")


if __name__ == "__main__":
    main()
