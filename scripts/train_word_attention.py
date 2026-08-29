"""
单词数据集 v2 训练 + 注意力 / 各层 embedding 可视化。

数据:50 词 2-6 字母,4 种生成模式(valid / wrong1 / wrong2 / scrambled),1500 张
任务(5 头,所有 acc 全打):
  - word_match  : 整体二分类 (pool 97-d)
  - word_id     : 50 类整词    (pool 97-d)
  - length      : 5 类词长     (pool 97-d)
  - first_letter: 首字母 26 类   (attention 后第 1 个 fg token 32-d,仅 valid 算)
  - last_letter : 末字母 26 类   (attention 后最后 fg token 32-d,仅 valid 算)

3 张核心结构图(用户最关注):
  1. word_attention_2to6.png  - 按词长(2/3/4/5/6)× valid/invalid 画 attention
  2. embedding_layers_pca.png - 各层 embedding 空间(token / 背景 / 整图)
  3. attention_vs_layout.png  - 同一目标词 valid vs scrambled 注意力对比

附加:
  - word_id_confusion.png    - 50 词混淆矩阵
  - train.log                - 5 头 acc 完整 log

用法:
  uv run python scripts/train_word_attention.py --epochs 30 [--tag word_v2]
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
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from algorithms import ALGOS
from algorithms.color_tokens import (
    _hsv_decompose, _hsv_id_to_onehot,
    POOL_IN_DIM, TOKEN_GEOM_DIM,
)
from experiments.source.synthetic.word_data import WordDataset, DEFAULT_WORDS_50


for _p in [
    os.path.expanduser("~/.local/share/fonts/wqy-microhei.ttc"),
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
]:
    if Path(_p).exists():
        from matplotlib import font_manager as fm
        plt.rcParams["font.family"] = fm.FontProperties(fname=_p).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False


def _safe_pca(X: np.ndarray, n_dim: int = 2) -> np.ndarray:
    n = X.shape[0]
    d = X.shape[1]
    n_dim = min(n_dim, max(1, n - 1), d)
    if n_dim < 1:
        return np.zeros((n, 2), dtype=np.float32)
    pca = PCA(n_components=n_dim, random_state=42)
    out = pca.fit_transform(X)
    if out.shape[1] < 2:
        out = np.hstack([out, np.zeros((n, 2 - out.shape[1]), dtype=np.float32)])
    return out[:, :2]


def _pool_from_tokens(tok_attn: torch.Tensor) -> torch.Tensor:
    mean = tok_attn.mean(dim=0)
    mx = tok_attn.max(dim=0).values
    mn = tok_attn.min(dim=0).values
    k_feat = torch.tensor([float(tok_attn.size(0))], device=tok_attn.device)
    return torch.cat([mean, mx, mn, k_feat])


def _encode_with_full(algo, rgb_i, device):
    tokens, _ = _hsv_decompose(rgb_i, mode="spatial")
    masks = np.stack([m for _, m in tokens])
    onehots = np.stack([_hsv_id_to_onehot(*hsv) for hsv, _ in tokens])
    masks_t = torch.from_numpy(masks).unsqueeze(1).to(device)
    onehots_t = torch.from_numpy(onehots).to(device)
    geom = algo.encoder(masks_t)
    g = algo.geom_mlp(geom)
    c = algo.color_mlp(onehots_t)
    tok = algo.fusion(g, c)
    tok_attn, attn = algo.self_attn(tok.unsqueeze(0), return_weights=True)
    tok_attn = tok_attn.squeeze(0)
    pool = _pool_from_tokens(tok_attn)
    return tok_attn, pool, attn.squeeze(0).detach().cpu().numpy(), tokens


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--samples-per-word", type=int, default=30)
    p.add_argument("--target-words", nargs="+", default=DEFAULT_WORDS_50)
    p.add_argument("--tag", default="word_v2")
    args = p.parse_args()

    out_dir = _HERE / "results" / f"train_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"
    print("=" * 70)
    print(f" 单词数据集 v2 训练 · 5 头 · 3 张结构图")
    print(f" epochs={args.epochs} 词数={len(args.target_words)} 样本/词={args.samples_per_word}")
    print("=" * 70)

    print("\n[1/4] 加载单词数据集 v2...")
    ds = WordDataset(target_words=args.target_words,
                     samples_per_word=args.samples_per_word,
                     invalid_ratio=0.5)
    ds.generate(verbose=True)
    rgbs = ds.rgbs()
    N = len(ds.samples)

    word_to_int = {w: i for i, w in enumerate(sorted(set(s["target_word"] for s in ds.samples)))}
    n_words = len(word_to_int)
    lengths = sorted(set(len(s["target_word"]) for s in ds.samples))
    length_to_int = {l: i for i, l in enumerate(lengths)}
    n_lengths = len(lengths)

    y_word = np.array([word_to_int[s["target_word"]] for s in ds.samples], dtype=np.int64)
    y_length = np.array([length_to_int[len(s["target_word"])] for s in ds.samples], dtype=np.int64)
    y_wm = np.array([int(s["word_match"]) for s in ds.samples], dtype=np.int64)
    y_is_valid = np.array([int(s["word_match"]) for s in ds.samples], dtype=bool)

    all_letters = sorted(set(t["char"] for s in ds.samples for t in s["tokens"]))
    letter_to_int = {c: i for i, c in enumerate(all_letters)}
    n_letters = len(all_letters)
    y_first = np.array([letter_to_int[s["tokens"][0]["char"]] for s in ds.samples], dtype=np.int64)
    y_last = np.array([letter_to_int[s["tokens"][-1]["char"]] for s in ds.samples], dtype=np.int64)
    print(f"  字典: words={n_words} lengths={n_lengths} letters={n_letters}")

    print("\n[2/4] 构建模型 + 5 头 (共享 emb + 各自投影 MLP)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    algo = ALGOS["color_tokens"]
    algo._build(device)

    # ── 共享中间 embedding ──
    # 整图类任务:pool 97-d → shared_emb 64-d (1 Linear + ReLU)
    #  token 类任务:first/last 32-d token → shared_tok 32-d (1 Linear + ReLU)
    SHARED_DIM_IMG = 64
    SHARED_DIM_TOK = 32

    torch.manual_seed(42)
    shared_img = nn.Sequential(
        nn.Linear(POOL_IN_DIM, SHARED_DIM_IMG),
        nn.ReLU(),
    ).to(device)
    torch.manual_seed(42)
    shared_tok = nn.Sequential(
        nn.Linear(TOKEN_GEOM_DIM, SHARED_DIM_TOK),
        nn.ReLU(),
    ).to(device)

    # ── 5 个投影头:shared → 各自 32-d 判别空间 → 分类 ──
    PROJ_DIM = 32

    def make_head(in_dim, n_classes):
        torch.manual_seed(42)
        return nn.Sequential(
            nn.Linear(in_dim, PROJ_DIM),
            nn.ReLU(),
            nn.Linear(PROJ_DIM, n_classes),
        ).to(device)

    head_word_match = make_head(SHARED_DIM_IMG, 2)
    head_word_id = make_head(SHARED_DIM_IMG, n_words)
    head_length = make_head(SHARED_DIM_IMG, n_lengths)
    head_first = make_head(SHARED_DIM_TOK, n_letters)
    head_last = make_head(SHARED_DIM_TOK, n_letters)

    print(f"  shared_img: {POOL_IN_DIM}→{SHARED_DIM_IMG}  shared_tok: {TOKEN_GEOM_DIM}→{SHARED_DIM_TOK}")
    print(f"  5 heads: shared → 32-d 投影 → 分类 (word_match=2 word_id={n_words} length={n_lengths} first/last={n_letters})")

    params = (list(algo.encoder.parameters()) +
              list(algo.geom_mlp.parameters()) +
              list(algo.color_mlp.parameters()) +
              list(algo.fusion.parameters()) +
              list(algo.self_attn.parameters()) +
              list(algo.emb_head.parameters()) +
              list(algo.cls_trunk.parameters()) +
              list(shared_img.parameters()) +
              list(shared_tok.parameters()) +
              list(head_word_match.parameters()) +
              list(head_word_id.parameters()) +
              list(head_length.parameters()) +
              list(head_first.parameters()) +
              list(head_last.parameters()))
    opt = torch.optim.Adam(params, lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    print(f"\n[3/4] 训练 {args.epochs} epoch ...")
    algo.train_all()
    shared_img.train(); shared_tok.train()
    for h in (head_word_match, head_word_id, head_length, head_first, head_last):
        h.train()
    train_t0 = time.time()

    for epoch in range(args.epochs):
        perm = np.random.permutation(N)
        sum_loss = 0.0
        correct = {k: 0 for k in ["word_match", "word_id", "length", "first", "last"]}
        counts = {k: 0 for k in ["word_match", "word_id", "length", "first", "last"]}

        for i in perm:
            tok_attn, pool, _, _ = _encode_with_full(algo, rgbs[i], device)
            n_fg = tok_attn.size(0) - 1

            first_tok = tok_attn[1] if n_fg > 0 else None
            last_tok = tok_attn[-1] if n_fg > 0 else None

            # ── 共享 embedding ──
            shared_img_emb = shared_img(pool)               # (64,)
            shared_tok_emb = None
            if first_tok is not None:
                shared_tok_emb = shared_tok(first_tok)      # (32,) for first
                shared_tok_emb_last = shared_tok(last_tok)  # (32,) for last
            else:
                shared_tok_emb_last = None

            opt.zero_grad()
            loss = torch.tensor(0.0, device=device)

            # 3 个整图任务
            img_tasks = [
                (head_word_match, "word_match", y_wm[i]),
                (head_word_id, "word_id", y_word[i]),
                (head_length, "length", y_length[i]),
            ]
            for head, key, yi in img_tasks:
                logits = head(shared_img_emb).unsqueeze(0)
                target = torch.tensor([yi], device=device)
                l = loss_fn(logits, target)
                loss = loss + l
                if logits.argmax().item() == yi:
                    correct[key] += 1
                counts[key] += 1

            # 2 个 token 任务(仅 valid)
            if y_is_valid[i] and shared_tok_emb is not None and n_fg > 0:
                lf = loss_fn(head_first(shared_tok_emb).unsqueeze(0),
                             torch.tensor([y_first[i]], device=device))
                ll = loss_fn(head_last(shared_tok_emb_last).unsqueeze(0),
                             torch.tensor([y_last[i]], device=device))
                loss = loss + 0.5 * lf + 0.5 * ll
                if head_first(shared_tok_emb).argmax().item() == y_first[i]:
                    correct["first"] += 1
                counts["first"] += 1
                if head_last(shared_tok_emb_last).argmax().item() == y_last[i]:
                    correct["last"] += 1
                counts["last"] += 1

            loss.backward()
            opt.step()
            sum_loss += loss.item()

        acc = {k: correct[k] / max(1, counts[k]) for k in correct}
        avg_loss = sum_loss / N
        elapsed = time.time() - train_t0
        msg = (f"  ep{epoch+1:>2}/{args.epochs}  loss={avg_loss:.3f}  "
               f"wm={acc['word_match']:.1%}  wid={acc['word_id']:.1%}  "
               f"len={acc['length']:.1%}  first={acc['first']:.1%}  "
               f"last={acc['last']:.1%}  ({elapsed:.0f}s)")
        with open(log_path, "a") as f:
            f.write(msg + "\n")
        if (epoch + 1) % 3 == 0 or epoch == 0:
            print(msg, flush=True)

    print("\n[4/4] 收集 embedding + 5 个投影空间 + 画 4 张图 ...")
    algo.eval_all()
    shared_img.eval(); shared_tok.eval()
    for h in (head_word_match, head_word_id, head_length, head_first, head_last):
        h.eval()

    all_attn, all_tok, all_bg_tok, all_pool, all_shared_img = [], [], [], [], []
    proj_wm, proj_wid, proj_len = [], [], []
    proj_first, proj_last = [], []
    all_modes, all_lens, all_wms = [], [], []
    fg_vecs_list, fg_chars_list, fg_pos_list = [], [], []
    valid_indices = []  # 仅 valid 样本,first/last 投影只对这些算
    word_id_pred = np.zeros(N, dtype=np.int64)
    with torch.no_grad():
        for i in range(N):
            tok_attn, pool, attn, tokens = _encode_with_full(algo, rgbs[i], device)
            all_attn.append(attn)
            t_np = tok_attn.cpu().numpy()
            all_tok.append(t_np)
            pool_np = pool.cpu().numpy()
            all_pool.append(pool_np)
            all_modes.append(ds.samples[i]["mode"])
            all_lens.append(len(ds.samples[i]["target_word"]))
            all_wms.append(int(ds.samples[i]["word_match"]))
            chars = [t["char"] for t in ds.samples[i]["tokens"]]
            n_letters = len(chars)
            K = t_np.shape[0]
            n_fg = K - 1 if K > n_letters else K
            fg_chars = (chars + ["?"] * n_fg)[:n_fg]
            if K > n_letters:
                fg_t = t_np[1:1 + n_fg]
            else:
                fg_t = t_np[:n_fg]
            for j in range(n_fg):
                fg_vecs_list.append(fg_t[j])
                fg_chars_list.append(fg_chars[j])
                fg_pos_list.append(j)
            if K > n_letters:
                all_bg_tok.append(t_np[0])

            # 共享 embedding + 各任务 32-d 投影空间
            shared_img_emb = shared_img(pool)             # (64,)
            all_shared_img.append(shared_img_emb.cpu().numpy())
            proj_wm.append(head_word_match[0](shared_img_emb).cpu().numpy())  # 第 0 层 Linear
            proj_wid.append(head_word_id[0](shared_img_emb).cpu().numpy())
            proj_len.append(head_length[0](shared_img_emb).cpu().numpy())

            if ds.samples[i]["word_match"] and n_fg > 0:
                first_t = tok_attn[1]              # spatial 模式第 0 个前景 token
                last_t = tok_attn[1 + n_fg - 1] if (1 + n_fg - 1) < K else tok_attn[K - 1]
                shared_t_first = shared_tok(first_t)
                shared_t_last = shared_tok(last_t)
                proj_first.append(head_first[0](shared_t_first).cpu().numpy())
                proj_last.append(head_last[0](shared_t_last).cpu().numpy())
                valid_indices.append(i)

            word_id_pred[i] = int(head_word_id(shared_img_emb).argmax().item())
    all_pool = np.stack(all_pool)
    all_shared_img = np.stack(all_shared_img)
    proj_wm = np.stack(proj_wm)
    proj_wid = np.stack(proj_wid)
    proj_len = np.stack(proj_len)
    proj_first = np.stack(proj_first) if proj_first else np.zeros((0, PROJ_DIM), np.float32)
    proj_last = np.stack(proj_last) if proj_last else np.zeros((0, PROJ_DIM), np.float32)
    fg_vecs = np.stack(fg_vecs_list) if fg_vecs_list else np.zeros((0, TOKEN_GEOM_DIM), np.float32)
    fg_chars_arr = np.array(fg_chars_list)
    fg_pos_arr = np.array(fg_pos_list)

    # ── 图 1: word_attention_2to6.png ──
    print("  - 画 word_attention_2to6.png ...")
    fig, axes = plt.subplots(2, 5, figsize=(18, 6.5))
    cmap = plt.get_cmap("viridis")
    for col, L in enumerate([2, 3, 4, 5, 6]):
        valid_idx = next((i for i, s in enumerate(ds.samples)
                          if len(s["target_word"]) == L and s["mode"] == "valid"), None)
        scr_idx = next((i for i, s in enumerate(ds.samples)
                        if len(s["target_word"]) == L and s["mode"] == "scrambled"), None)
        for row, idx in enumerate([valid_idx, scr_idx]):
            ax = axes[row, col]
            if idx is None:
                ax.set_title(f"L={L} (无样本)", fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])
                continue
            attn = all_attn[idx]
            K = attn.shape[0]
            # K 不一定等于 n_letters+1(可能有碎裂),统一按 K 切
            chars = [t["char"] for t in ds.samples[idx]["tokens"]]
            # 取 K-1 个前景字符,不够补 '?'
            fg_chars = (chars + ["?"] * (K - 1))[:K - 1]
            labels = ["BG"] + fg_chars
            labels = labels[:K]
            im = ax.imshow(attn, cmap=cmap, vmin=0, vmax=max(attn.max(), 1e-6))
            ax.set_xticks(range(K))
            ax.set_xticklabels(labels, fontsize=8)
            ax.set_yticks(range(K))
            ax.set_yticklabels(labels, fontsize=8)
            for r in range(K):
                for c in range(K):
                    ax.text(c, r, f"{attn[r, c]:.2f}",
                            ha="center", va="center",
                            color="white" if attn[r, c] < attn.max() * 0.6 else "black",
                            fontsize=7)
            s = ds.samples[idx]
            title_mode = "valid" if row == 0 else "scrambled"
            ax.set_title(f"L={L} {title_mode} · {s['target_word']}", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{title_mode}\nquery", fontsize=9)
            if row == 1:
                ax.set_xlabel("key", fontsize=9)
    fig.suptitle("Attention Map by 词长(行: valid / scrambled) · K=字母+背景",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(out_dir / "word_attention_2to6.png", dpi=100, bbox_inches="tight")
    plt.close()

    # ── 图 2: embedding_layers_pca.png ──
    print("  - 画 embedding_layers_pca.png ...")
    bg_vecs = np.stack(all_bg_tok) if all_bg_tok else np.zeros((1, TOKEN_GEOM_DIM), np.float32)
    fg_chars = fg_chars_arr
    fg_pos = fg_pos_arr
    all_lens_arr = np.array(all_lens)
    all_wms_arr = np.array(all_wms)

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    # a) 前景 token 按字符着色
    if len(fg_vecs) > 1:
        xy = _safe_pca(fg_vecs, 2)
        ax = axes[0, 0]
        cmap20 = plt.get_cmap("tab20")
        uniq = sorted(set(fg_chars))
        color_map = {c: cmap20(i % 20 / 20) for i, c in enumerate(uniq)}
        for c in uniq:
            m = fg_chars == c
            ax.scatter(xy[m, 0], xy[m, 1], color=color_map[c], s=8, alpha=0.6,
                       edgecolors="none")
        ax.set_title(f"前景 token · 按字符 ({len(fg_chars)} tokens, {len(uniq)} 类)", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        if len(uniq) <= 30:
            ax.legend(fontsize=6, loc="best", ncol=2, framealpha=0.7, markerscale=1.5)
    else:
        axes[0, 0].text(0.5, 0.5, "无前景 token", ha="center", va="center")
        axes[0, 0].set_xticks([]); axes[0, 0].set_yticks([])

    # b) 按词内位置
    ax = axes[0, 1]
    if len(fg_vecs) > 1:
        for p in sorted(set(fg_pos)):
            m = fg_pos == p
            ax.scatter(xy[m, 0], xy[m, 1], color=plt.get_cmap("tab10")(p % 10 / 10),
                       s=8, alpha=0.6, edgecolors="none", label=f"pos {p}")
    ax.set_title("前景 token · 按词内位置着色", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=7, loc="best", framealpha=0.7)

    # c) 背景 token 按词长
    ax = axes[0, 2]
    if len(bg_vecs) > 1:
        xy_bg = _safe_pca(bg_vecs, 2)
        bg_lens_list = []
        for i, s in enumerate(ds.samples):
            t = all_tok[i]
            n_fg = len(s["tokens"])
            if t.shape[0] > n_fg:
                bg_lens_list.append(len(s["target_word"]))
        bg_lens = np.array(bg_lens_list[:len(bg_vecs)])
        for L in sorted(set(bg_lens)):
            m = bg_lens == L
            ax.scatter(xy_bg[m, 0], xy_bg[m, 1],
                       color=plt.get_cmap("tab10")(L % 10 / 10), s=10,
                       alpha=0.6, edgecolors="none", label=f"L={L}")
    ax.set_title(f"背景 token · 按词长 ({len(bg_vecs)} bg tokens)", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=7, loc="best", framealpha=0.7)

    # d) 前景 vs 背景
    ax = axes[0, 3]
    if len(fg_vecs) > 0 and len(bg_vecs) > 0:
        all_v = np.concatenate([fg_vecs, bg_vecs])
        is_bg = np.concatenate([np.zeros(len(fg_vecs), dtype=bool),
                                np.ones(len(bg_vecs), dtype=bool)])
        xy_all = _safe_pca(all_v, 2)
        ax.scatter(xy_all[~is_bg, 0], xy_all[~is_bg, 1], color="#3b82f6", s=5,
                   alpha=0.4, edgecolors="none", label="前景")
        ax.scatter(xy_all[is_bg, 0], xy_all[is_bg, 1], color="#ef4444", s=12,
                   alpha=0.8, edgecolors="black", linewidths=0.3, label="背景")
    ax.set_title("前景 vs 背景 token", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=8, loc="best", framealpha=0.8)

    # row 1: 整图 128-d
    xy_img = _safe_pca(all_pool, 2)
    # e) 按词长
    ax = axes[1, 0]
    for L in sorted(set(all_lens_arr)):
        m = all_lens_arr == L
        ax.scatter(xy_img[m, 0], xy_img[m, 1],
                   color=plt.get_cmap("tab10")(L % 10 / 10), s=10,
                   alpha=0.6, edgecolors="none", label=f"L={L}")
    ax.set_title(f"整图 128-d · 按词长 ({N} samples)", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=7, loc="best", framealpha=0.7)

    # f) 按 word_match
    ax = axes[1, 1]
    for v, col_v, lab in [(0, "#ef4444", "invalid"), (1, "#10b981", "valid")]:
        m = all_wms_arr == v
        ax.scatter(xy_img[m, 0], xy_img[m, 1], color=col_v, s=10, alpha=0.6,
                   edgecolors="none", label=lab)
    ax.set_title("整图 128-d · 按 word_match", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=8, loc="best", framealpha=0.8)

    # g) 按 mode
    ax = axes[1, 2]
    mode_colors = {"valid": "#10b981", "wrong1": "#f59e0b",
                   "wrong2": "#ef4444", "scrambled": "#8b5cf6"}
    for mode, col_v in mode_colors.items():
        m = np.array([m == mode for m in all_modes])
        if m.any():
            ax.scatter(xy_img[m, 0], xy_img[m, 1], color=col_v, s=10, alpha=0.6,
                       edgecolors="none", label=f"{mode} ({m.sum()})")
    ax.set_title("整图 128-d · 按 mode", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=7, loc="best", framealpha=0.8)

    # h) valid only, 按 word_id
    ax = axes[1, 3]
    valid_idx = np.where(all_wms_arr == 1)[0]
    if len(valid_idx) > 0:
        cmap_w = plt.get_cmap("hsv")
        word_ints = np.array([word_to_int[ds.samples[i]["target_word"]]
                              for i in valid_idx])
        for w_i in sorted(set(word_ints)):
            m = word_ints == w_i
            ax.scatter(xy_img[valid_idx[m], 0], xy_img[valid_idx[m], 1],
                       color=cmap_w(w_i / n_words), s=12, alpha=0.7,
                       edgecolors="none")
    ax.set_title(f"整图 128-d · valid only, 按 word_id ({n_words} 词)", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("各层 Embedding 空间(PCA 2D)· 前景 token / 背景 / 整图",
                 fontsize=13, fontweight="bold", y=1.005)
    plt.tight_layout()
    plt.savefig(out_dir / "embedding_layers_pca.png", dpi=100, bbox_inches="tight")
    plt.close()

    # ── 图 3: attention_vs_layout.png ──
    print("  - 画 attention_vs_layout.png ...")
    pairs = []
    for L in [2, 3, 4, 5, 6]:
        seen = set()
        for i, s in enumerate(ds.samples):
            if len(s["target_word"]) != L or s["target_word"] in seen:
                continue
            j = next((k for k, t in enumerate(ds.samples)
                      if t["target_word"] == s["target_word"] and t["mode"] == "scrambled"),
                     None)
            if j is not None:
                pairs.append((i, j, s["target_word"]))
                seen.add(s["target_word"])
                if sum(1 for p in pairs if len(p[2]) == L) >= 2:
                    break
    pairs = pairs[:8]
    n_pairs = len(pairs)
    if n_pairs > 0:
        fig, axes = plt.subplots(2, n_pairs, figsize=(2.4 * n_pairs, 5.2))
        if n_pairs == 1:
            axes = axes.reshape(2, 1)
        for col, (vi, si, word) in enumerate(pairs):
            for row, (idx, lbl) in enumerate([(vi, "valid"), (si, "scrambled")]):
                ax = axes[row, col]
                attn = all_attn[idx]
                K = attn.shape[0]
                chars = [t["char"] for t in ds.samples[idx]["tokens"]]
                fg_chars = (chars + ["?"] * (K - 1))[:K - 1]
                labels = ["BG"] + fg_chars
                labels = labels[:K]
                im = ax.imshow(attn, cmap=cmap, vmin=0, vmax=max(attn.max(), 1e-6))
                ax.set_xticks(range(K))
                ax.set_xticklabels(labels, fontsize=8)
                ax.set_yticks(range(K))
                ax.set_yticklabels(labels, fontsize=8)
                for r in range(K):
                    for c in range(K):
                        ax.text(c, r, f"{attn[r, c]:.2f}",
                                ha="center", va="center",
                                color="white" if attn[r, c] < attn.max() * 0.6 else "black",
                                fontsize=7)
                ax.set_title(f"{word} {lbl}", fontsize=9)
        fig.suptitle(f"Attention 对比 · 同一目标词 · valid(上) vs scrambled(下)",
                     fontsize=12, fontweight="bold", y=1.01)
        plt.tight_layout()
        plt.savefig(out_dir / "attention_vs_layout.png", dpi=100, bbox_inches="tight")
        plt.close()

    # ── 混淆矩阵 ──
    print("  - 画 word_id_confusion.png ...")
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_word, word_id_pred, labels=range(n_words))
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    ax.set_xticks(range(n_words))
    ax.set_xticklabels(sorted(word_to_int.keys()), rotation=90, fontsize=7)
    ax.set_yticks(range(n_words))
    ax.set_yticklabels(sorted(word_to_int.keys()), fontsize=7)
    ax.set_xlabel("预测")
    ax.set_ylabel("真实")
    ax.set_title(f"word_id 混淆矩阵 ({n_words} 词, N={N})", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    for i in range(n_words):
        for j in range(n_words):
            if cm[i, j] > 0:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() * 0.6 else "black",
                        fontsize=6)
    plt.tight_layout()
    plt.savefig(out_dir / "word_id_confusion.png", dpi=100, bbox_inches="tight")
    plt.close()

    # ── 图 5: 5 个分类头的 32-d 投影空间(共享中间 embedding 之后) ──
    print("  - 画 projection_spaces.png ...")
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))

    # 1) word_match 投影 (32-d) - 整图 shared
    ax = axes[0, 0]
    xy = _safe_pca(proj_wm, 2)
    for v, col_v, lab in [(0, "#ef4444", "invalid"), (1, "#10b981", "valid")]:
        m = all_wms_arr == v
        ax.scatter(xy[m, 0], xy[m, 1], color=col_v, s=10, alpha=0.6,
                   edgecolors="none", label=lab)
    ax.set_title(f"word_match 投影(32-d)· N={len(proj_wm)}", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=8, framealpha=0.8)

    # 2) word_id 投影 (32-d) - valid only, 按 word
    ax = axes[0, 1]
    valid_idx = np.where(all_wms_arr == 1)[0]
    xy = _safe_pca(proj_wid, 2)
    if len(valid_idx) > 0:
        cmap_w = plt.get_cmap("hsv")
        for w_i in range(n_words):
            m = np.array([word_to_int[ds.samples[i]["target_word"]] == w_i
                          for i in valid_idx])
            if m.any():
                ax.scatter(xy[valid_idx[m], 0], xy[valid_idx[m], 1],
                           color=cmap_w(w_i / n_words), s=14, alpha=0.7,
                           edgecolors="none")
    ax.set_title(f"word_id 投影(32-d)· valid only · {n_words} 词", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

    # 3) length 投影 (32-d) - 按词长
    ax = axes[0, 2]
    xy = _safe_pca(proj_len, 2)
    for L in sorted(set(all_lens_arr)):
        m = all_lens_arr == L
        ax.scatter(xy[m, 0], xy[m, 1],
                   color=plt.get_cmap("tab10")(L % 10 / 10), s=12, alpha=0.6,
                   edgecolors="none", label=f"L={L}")
    ax.set_title("length 投影(32-d)", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=8, framealpha=0.8)

    # 4) first_letter 投影 (32-d) - valid only, 按首字母
    ax = axes[1, 0]
    if len(proj_first) > 0:
        xy = _safe_pca(proj_first, 2)
        first_chars = np.array([ds.samples[i]["target_word"][0] for i in valid_indices])
        uniq = sorted(set(first_chars))
        cmap20 = plt.get_cmap("tab20")
        for c in uniq:
            m = first_chars == c
            ax.scatter(xy[m, 0], xy[m, 1], color=cmap20(hash(c) % 20 / 20),
                       s=18, alpha=0.7, edgecolors="none", label=c)
    ax.set_title(f"first_letter 投影(32-d)· {len(proj_first)} valid", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    if len(uniq) <= 30:
        ax.legend(fontsize=6, loc="best", ncol=2, framealpha=0.7, markerscale=1.2)

    # 5) last_letter 投影 (32-d) - valid only, 按末字母
    ax = axes[1, 1]
    if len(proj_last) > 0:
        xy = _safe_pca(proj_last, 2)
        last_chars = np.array([ds.samples[i]["target_word"][-1] for i in valid_indices])
        uniq = sorted(set(last_chars))
        cmap20 = plt.get_cmap("tab20")
        for c in uniq:
            m = last_chars == c
            ax.scatter(xy[m, 0], xy[m, 1], color=cmap20(hash(c) % 20 / 20),
                       s=18, alpha=0.7, edgecolors="none", label=c)
    ax.set_title(f"last_letter 投影(32-d)· {len(proj_last)} valid", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    if len(uniq) <= 30:
        ax.legend(fontsize=6, loc="best", ncol=2, framealpha=0.7, markerscale=1.2)

    # 6) 共享中间 embedding 64-d - 整体 sanity 看
    ax = axes[1, 2]
    xy = _safe_pca(all_shared_img, 2)
    for L in sorted(set(all_lens_arr)):
        m = all_lens_arr == L
        ax.scatter(xy[m, 0], xy[m, 1],
                   color=plt.get_cmap("tab10")(L % 10 / 10), s=12, alpha=0.6,
                   edgecolors="none", label=f"L={L}")
    ax.set_title("共享 img embedding(64-d)· 参考", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=8, framealpha=0.8)

    fig.suptitle("5 个分类头的 32-d 投影空间 + 共享 64-d 参考\n"
                 "shared → head[0](Linear) → 32-d 投影(分类前最后一层)",
                 fontsize=12, fontweight="bold", y=1.005)
    plt.tight_layout()
    plt.savefig(out_dir / "projection_spaces.png", dpi=100, bbox_inches="tight")
    plt.close()

    print(f"\n  最终精度: wm={acc['word_match']:.1%}  wid={acc['word_id']:.1%}  "
          f"len={acc['length']:.1%}  first={acc['first']:.1%}  last={acc['last']:.1%}")
    print(f"\n  ✓ 全部输出: {out_dir}/")
    for f in sorted(out_dir.iterdir()):
        print(f"    - {f.name}")


if __name__ == "__main__":
    main()
