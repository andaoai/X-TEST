"""
ColorTokenEncoder 端到端"图像→文字"训练。

数据: WordDataset 2-5 字母(50 词表, valid 模式)
目标: 每 token 输出 26 字母 + <BG>
评估: 4 个指标
  1. per-token 字符 acc  (含 BG,BG 也计对)
  2. word-level 整词 acc (前景 token 按 x 拼成词,完全匹配)
  3. top-1 / top-5 检索 acc (50 词表 prototype 检索)
  4. token embedding silhouette (前景 token 按字符 id 聚类质量)
可视化: 每个 epoch 在 val 上画
  - tokens_grid.png : 8 个 val 样本的 token 关联(每个 token 标 GT 字符)
  - token_pca.png   : 前景 token embedding 2D PCA,按字符着色

关键设计:
  - 背景 token 也走 ColorTokenEncoder + head,ground truth = <BG> (id=26)
  - 训练时 encoder + head 一起端到端优化(单样本 SGD, 跟项目现有 train 脚本一致)
"""
import argparse
import os
import random
import string
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 字体加载:用 wqy(支持中文)—— 与 train_word_attention.py 一致
_CN_FONT_PATHS = [
    os.path.expanduser("~/.local/share/fonts/wqy-microhei.ttc"),
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]
for _fp in _CN_FONT_PATHS:
    if os.path.exists(_fp):
        try:
            font_manager.fontManager.addfont(_fp)
        except Exception:
            pass
plt.rcParams["font.family"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.source.synthetic.word_data import WordDataset
from algorithms.image_decomposition import (
    ColorTokenEncoder,
    decompose_image_to_tokens,
    N_H_BINS, N_S_BINS, N_V_BINS, COLOR_ONEHOT_DIM,
)

# ── 常量 ──
LETTERS = list(string.ascii_lowercase)
LETTER2ID = {c: i for i, c in enumerate(LETTERS)}    # 26
ID2LETTER = {i: c for c, i in LETTER2ID.items()}
BG_ID = 26
N_CLASSES = 27
OUT_DIM = 32
VAL_RATIO = 0.15
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── 工具 ──
def build_targets(n_tokens: int, rendered_word: str) -> torch.Tensor:
    """
    把 ground truth 字符序列分配到 token 位置:
      token 0 = BG (id=26)
      token 1..n_tokens-1 = rendered_word[0..n-1] (按 x 顺序,跟 decompose 的 char_hints 一致)

    若 n_tokens != 1 + len(rendered_word):
      - 多了  → 截断多余的尾部 token 当 BG
      - 少了  → 用 BG 填充
    """
    chars = [c for c in rendered_word if c in LETTER2ID]
    n_word = len(chars)
    n_fg_tokens = n_tokens - 1
    if n_fg_tokens == n_word:
        return torch.tensor([BG_ID] + [LETTER2ID[c] for c in chars],
                            dtype=torch.long)
    if n_fg_tokens < n_word:
        # 多了字符,截断 rendered_word 前面 n_fg_tokens 个
        chars = chars[:n_fg_tokens]
    else:
        # 少了 token,用 BG 填充
        chars = chars + ["<bg>"] * (n_fg_tokens - n_word)
    targets = [BG_ID] + [LETTER2ID.get(c, BG_ID) for c in chars]
    return torch.tensor(targets[:n_tokens], dtype=torch.long)


def encode_tokens(rgb: np.ndarray, encoder: ColorTokenEncoder,
                  char_hints=None) -> tuple:
    """单样本:图像 → (K, out_dim) vecs + 原始 tokens list"""
    tokens, _ = decompose_image_to_tokens(rgb, mode="spatial",
                                          char_hints=char_hints)
    hsv_ids = torch.tensor([t[0] for t in tokens], dtype=torch.long).to(DEVICE)
    masks = np.stack([t[1] for t in tokens])[:, None, :, :]
    masks_t = torch.from_numpy(masks).to(DEVICE).float()
    vecs = encoder(hsv_ids, masks_t)   # (K, out_dim)
    return vecs, tokens


# ── 评估:4 个指标 ──
def build_word_prototypes(samples, encoder, target_words):
    """用 train 集算每个词的 prototype(平均前景 token 嵌入)。"""
    encoder.eval()
    word_proto = {w: [] for w in target_words}
    with torch.no_grad():
        for s in samples:
            word = s["target_word"]
            rendered = s["rendered_word"]
            word_chars = list(rendered)
            try:
                vecs, tokens = encode_tokens(s["rgb"], encoder,
                                             char_hints=word_chars)
            except Exception:
                continue
            if len(tokens) > 1:
                fg = vecs[1:].mean(dim=0).cpu().numpy()
                word_proto[word].append(fg)
    proto_dict = {}
    for w, vl in word_proto.items():
        if vl:
            proto = np.mean(np.stack(vl), axis=0)
            proto /= np.linalg.norm(proto) + 1e-8
            proto_dict[w] = proto
    return proto_dict


def evaluate(val_samples, encoder, head, target_words, proto_dict=None):
    """val 上算 4 个指标。proto_dict 必须从 train 集预计算后传入。"""
    encoder.eval()
    head.eval()
    correct_tok, total_tok = 0, 0
    word_correct, word_total = 0, 0
    all_token_embs, all_token_labels = [], []
    word_query = {w: [] for w in target_words}
    with torch.no_grad():
        for s in val_samples:
            rgb = s["rgb"]
            word = s["target_word"]
            rendered = s["rendered_word"]
            word_chars = list(rendered)
            try:
                vecs, tokens = encode_tokens(rgb, encoder,
                                             char_hints=word_chars)
            except Exception:
                continue
            n_tok = len(tokens)
            targets = build_targets(n_tok, rendered).to(DEVICE)
            logits = head(vecs)
            pred = logits.argmax(dim=-1)

            # 1) per-token acc
            correct_tok += (pred == targets).sum().item()
            total_tok += len(targets)
            # 2) word acc:前景 token 拼成词
            if n_tok > 1:
                fg_pred = pred[1:].cpu().tolist()
                decoded = "".join(ID2LETTER.get(p, "?") for p in fg_pred)
                if decoded == word:
                    word_correct += 1
            word_total += 1
            # 收集 embedding(用于 silhouette)
            for j, t in enumerate(targets):
                if t.item() != BG_ID:
                    all_token_embs.append(vecs[j].cpu().numpy())
                    all_token_labels.append(t.item())
            # 收集 query(用于 top-K 检索)
            if n_tok > 1:
                fg_vecs = vecs[1:].mean(dim=0).cpu().numpy()
                word_query[word].append(fg_vecs)

    # 3) top-1 / top-5 检索:用外部传入的 proto_dict(train 算的)
    if proto_dict is None:
        proto_dict = build_word_prototypes(val_samples, encoder,
                                            target_words)
    proto_words = list(proto_dict.keys())
    proto_mat = np.stack([proto_dict[w] for w in proto_words])   # (V, 32)
    top1, top5 = 0, 0
    n_total = 0
    for w, vl in word_query.items():
        for v in vl:
            vn = v / (np.linalg.norm(v) + 1e-8)
            sims = proto_mat @ vn
            top_idx = np.argsort(-sims)
            if proto_words[top_idx[0]] == w:
                top1 += 1
            if w in [proto_words[i] for i in top_idx[:5]]:
                top5 += 1
            n_total += 1

    # 4) silhouette
    if len(all_token_embs) > 10 and len(set(all_token_labels)) > 1:
        from sklearn.metrics import silhouette_score
        sil = silhouette_score(np.stack(all_token_embs),
                               np.array(all_token_labels))
    else:
        sil = 0.0

    encoder.train()
    head.train()
    return {
        "tok_acc": correct_tok / max(1, total_tok),
        "word_acc": word_correct / max(1, word_total),
        "top1": top1 / max(1, n_total),
        "top5": top5 / max(1, n_total),
        "silhouette": sil,
    }


# ── 可视化 ──
def save_epoch_viz(epoch, val_samples, encoder, head, out_dir):
    """每个 epoch 出 2 张图:token 关联 + embedding 空间"""
    ep_dir = out_dir / f"epoch_{epoch:03d}"
    ep_dir.mkdir(exist_ok=True)
    encoder.eval()
    head.eval()

    # 选 8 个不同长度的 val 样本
    by_len = {}
    for s in val_samples:
        n = len(s["target_word"])
        by_len.setdefault(n, []).append(s)
    samples = []
    for L in sorted(by_len.keys()):
        rng = random.Random(SEED + epoch + L)
        if by_len[L]:
            samples.append(rng.choice(by_len[L]))
        if len(samples) >= 8:
            break
    while len(samples) < 8 and samples:
        samples.append(samples[len(samples) % len(samples)])

    # 1) tokens_grid.png: 8 个样本的 token 关联 + 预测字符
    n = len(samples)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    axes = axes.flat
    for i, s in enumerate(samples[:8]):
        ax = axes[i]
        rgb = s["rgb"]
        word = s["target_word"]
        rendered = s["rendered_word"]
        word_chars = list(rendered)
        try:
            with torch.no_grad():
                vecs, tokens = encode_tokens(rgb, encoder, char_hints=word_chars)
                n_tok = len(tokens)
                targets = build_targets(n_tok, rendered)
                logits = head(vecs)
                pred = logits.argmax(dim=-1).cpu().tolist()
                probs = F.softmax(logits, dim=-1).cpu().numpy()
        except Exception:
            ax.set_title(f"sample {i}: decompose error")
            ax.axis("off")
            continue
        # K x K 关联矩阵:token i 跟 token j 的 cosine 相似度
        norms = vecs.norm(dim=-1, keepdim=True) + 1e-8
        sim = (vecs @ vecs.T) / (norms * norms.T)
        sim = sim.detach().cpu().numpy()
        # 画图
        im = ax.imshow(sim, cmap="viridis", vmin=0, vmax=1)
        labels = []
        for j in range(n_tok):
            if j == 0:
                labels.append("BG")
            else:
                gt = ID2LETTER.get(int(targets[j].item()), "?")
                pr = ID2LETTER.get(int(pred[j]), "?")
                p_gt = probs[j][int(targets[j].item())]
                labels.append(f"gt={gt}\npr={pr}\n{p_gt:.2f}")
        ax.set_xticks(range(n_tok))
        ax.set_xticklabels(labels, fontsize=6)
        ax.set_yticks(range(n_tok))
        ax.set_yticklabels(labels, fontsize=6)
        word_str = "→".join(ID2LETTER.get(int(t), "?")
                            for t in pred[1:]) if n_tok > 1 else ""
        ax.set_title(f"[{rendered}] pred: {word_str}", fontsize=9)
        # 写每格数值
        for r in range(n_tok):
            for c in range(n_tok):
                ax.text(c, r, f"{sim[r, c]:.2f}", ha="center", va="center",
                        color="white" if sim[r, c] < sim.max() * 0.6 else "black",
                        fontsize=5)
    for i in range(len(samples), 8):
        axes[i].axis("off")
    fig.suptitle(f"epoch {epoch}: token association (cosine sim)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(ep_dir / "tokens_grid.png", dpi=100, bbox_inches="tight")
    plt.close(fig)

    # 2) token_pca.png: 前景 token embedding 2D PCA,按字符着色
    all_embs, all_chars = [], []
    with torch.no_grad():
        for s in val_samples:
            word = s["target_word"]
            rendered = s["rendered_word"]
            word_chars = list(rendered)
            try:
                vecs, tokens = encode_tokens(rgb=s["rgb"], encoder=encoder,
                                              char_hints=word_chars)
            except Exception:
                continue
            targets = build_targets(len(tokens), rendered)
            for j in range(1, len(tokens)):
                t = int(targets[j].item())
                if t != BG_ID:
                    all_embs.append(vecs[j].cpu().numpy())
                    all_chars.append(t)
    if len(all_embs) > 5:
        embs = np.stack(all_embs)
        chars = np.array(all_chars)
        # 2D PCA(若 dim < 2 则退化)
        if embs.shape[0] >= 2 and embs.shape[1] >= 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            xy = pca.fit_transform(embs)
            fig, ax = plt.subplots(1, 1, figsize=(8, 7))
            cmap = plt.get_cmap("tab20")
            for c_id in sorted(set(chars.tolist())):
                mask = chars == c_id
                ax.scatter(xy[mask, 0], xy[mask, 1],
                           c=[cmap(c_id % 20)], label=ID2LETTER.get(c_id, "?"),
                           s=20, alpha=0.7, edgecolors="black", linewidth=0.3)
            ax.set_title(f"epoch {epoch}: token embedding (PCA 2D, "
                         f"{len(set(chars.tolist()))} chars)")
            ax.legend(loc="upper right", fontsize=7, ncol=3,
                      markerscale=1.5)
            fig.tight_layout()
            fig.savefig(ep_dir / "token_pca.png", dpi=100, bbox_inches="tight")
            plt.close(fig)

    encoder.train()
    head.train()


# ── 主流程 ──
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--samples-per-word", type=int, default=30)
    p.add_argument("--fuse-mode", type=str, default="gate",
                   choices=ColorTokenEncoder.FUSE_MODES)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--tag", type=str, default="v1")
    p.add_argument("--val-ratio", type=float, default=VAL_RATIO)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # 1) 数据
    ds = WordDataset(samples_per_word=args.samples_per_word,
                     invalid_ratio=0.0)
    ds.generate(verbose=True)
    # 切片 2-5 字母 valid 样本
    valid_idx = [i for i, s in enumerate(ds.samples)
                 if 2 <= len(s["target_word"]) <= 5
                 and s["word_match"]]
    samples = [ds.samples[i] for i in valid_idx]
    random.Random(SEED).shuffle(samples)
    n_val = max(1, int(len(samples) * args.val_ratio))
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]
    print(f"\n>> train={len(train_samples)} val={len(val_samples)}")
    print(f">> 词表: {len(ds.target_words)} 词,sample 长度: 2-5 字母")

    # 2) 模型
    encoder = ColorTokenEncoder(fuse_mode=args.fuse_mode,
                                out_dim=OUT_DIM).to(DEVICE).train()
    head = nn.Linear(OUT_DIM, N_CLASSES).to(DEVICE).train()
    opt = torch.optim.Adam(
        list(encoder.parameters()) + list(head.parameters()),
        lr=args.lr,
    )

    # 3) 训练循环
    out_dir = Path(f"results/token2text_{args.tag}")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"
    with open(log_path, "w") as f:
        f.write("epoch\ttrain_loss\ttok_acc\tword_acc\ttop1\ttop5\t"
                "silhouette\n")

    for epoch in range(args.epochs):
        random.Random(SEED + epoch).shuffle(train_samples)
        sum_loss, correct_tok, total_tok = 0.0, 0, 0
        n_skipped = 0
        for s in train_samples:
            rgb = s["rgb"]
            rendered = s["rendered_word"]
            word_chars = list(rendered)
            try:
                vecs, tokens = encode_tokens(rgb, encoder, char_hints=word_chars)
            except Exception:
                n_skipped += 1
                continue
            targets = build_targets(len(tokens), rendered).to(DEVICE)
            logits = head(vecs)
            loss = F.cross_entropy(logits, targets)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sum_loss += loss.item()
            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                correct_tok += (pred == targets).sum().item()
            total_tok += len(targets)
        train_loss = sum_loss / max(1, len(train_samples) - n_skipped)
        train_tok_acc = correct_tok / max(1, total_tok)

        # val 评估(prototype 用 train 集算,避免 val 自己当自己的假信号)
        proto_dict = build_word_prototypes(train_samples, encoder,
                                            ds.target_words)
        metrics = evaluate(val_samples, encoder, head, ds.target_words,
                           proto_dict=proto_dict)

        # 每 epoch 出图
        save_epoch_viz(epoch, val_samples, encoder, head, out_dir)

        # 写 log
        with open(log_path, "a") as f:
            f.write(f"{epoch}\t{train_loss:.4f}\t{train_tok_acc:.4f}\t"
                    f"{metrics['word_acc']:.4f}\t{metrics['top1']:.4f}\t"
                    f"{metrics['top5']:.4f}\t{metrics['silhouette']:.4f}\n")
        if (epoch + 1) % 3 == 0 or epoch == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch:3d}: loss={train_loss:.3f} "
                  f"tok={train_tok_acc:.3f} word={metrics['word_acc']:.3f} "
                  f"top1={metrics['top1']:.3f} top5={metrics['top5']:.3f} "
                  f"sil={metrics['silhouette']:.3f} "
                  f"[skipped={n_skipped}]", flush=True)


if __name__ == "__main__":
    main()
