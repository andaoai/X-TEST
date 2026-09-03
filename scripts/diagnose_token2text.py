"""诊断脚本:用 v1 训练结果(已存图)+ 新训一个 5-epoch 小模型,定位问题。

不需要 checkpoint(训练脚本没存)。直接:
  - 跑一个 5 epoch 的小训练(10 分钟内)
  - 期间每 epoch 存 encoder.pt + head.pt
  - 评估:按字符/位置错分布 + embedding 聚类
"""
import argparse, os, sys, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.source.synthetic.word_data import WordDataset
from algorithms.image_decomposition import ColorTokenEncoder
from scripts.train_token2text import (
    build_targets, encode_tokens, LETTER2ID, ID2LETTER, BG_ID,
    DEVICE, OUT_DIM, N_CLASSES,
)


def quick_train(epochs=8, samples_per_word=30, fuse_mode="gate", tag="diag"):
    """快速训一个模型,每 epoch 存 checkpoint。"""
    out_dir = Path(f"results/diag_{tag}")
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
        sum_loss = 0
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
        # 存
        torch.save(encoder.state_dict(), out_dir / f"enc_ep{ep}.pt")
        torch.save(head.state_dict(), out_dir / f"head_ep{ep}.pt")
        print(f"  ep {ep}: loss={sum_loss/len(train_samples):.3f}")

    return encoder, head, val_samples, ds.target_words, out_dir


def diagnose(encoder, head, val_samples, out_dir, ep):
    encoder.eval(); head.eval()
    char_wrong = Counter()
    pos_wrong = Counter()
    len_acc = defaultdict(lambda: [0, 0])
    samples_per_word = defaultdict(lambda: [0, 0])
    word_results = []
    K_mismatch = 0
    with torch.no_grad():
        for s in val_samples:
            word = s["target_word"]
            rendered = s["rendered_word"]
            try:
                vecs, tokens = encode_tokens(s["rgb"], encoder, char_hints=list(rendered))
            except Exception:
                continue
            if len(tokens) != len(rendered) + 1:
                K_mismatch += 1
            targets = build_targets(len(tokens), rendered).to(DEVICE)
            pred = head(vecs).argmax(dim=-1).cpu().tolist()
            target_list = targets.cpu().tolist()
            fg_pred = pred[1:]
            fg_target = target_list[1:]
            decoded = "".join(ID2LETTER.get(p, "?") for p in fg_pred)
            word_ok = decoded == word
            samples_per_word[word][1] += 1
            if word_ok:
                samples_per_word[word][0] += 1
            else:
                diffs = []
                for j, (p_, t) in enumerate(zip(fg_pred, fg_target)):
                    if p_ != t:
                        char_wrong[("BG" if t == 26 else chr(ord("a")+t))] += 1
                        pos_wrong[j] += 1
                        diffs.append(f"pos{j}({('BG' if t==26 else chr(ord('a')+t))}->{('BG' if p_==26 else chr(ord('a')+p_))})")
                word_results.append((word, decoded, diffs))
            len_acc[len(word)][1] += 1
            if word_ok:
                len_acc[len(word)][0] += 1

    print(f"\n{'='*60}")
    print(f" 诊断报告 (ep={ep})")
    print(f"{'='*60}")
    print(f"val 样本: {len(val_samples)}, K 不匹配: {K_mismatch} ({K_mismatch/max(1,len(val_samples)):.0%})")

    print(f"\n词长 vs word_acc:")
    for L in sorted(len_acc):
        ok, tot = len_acc[L]
        print(f"  {L} 字母: {ok}/{tot} = {ok/max(1,tot):.2%}")

    print(f"\n错最多被预测错的字符(Top 10):")
    for c, n in char_wrong.most_common(10):
        print(f"  '{c}': {n} 次")

    print(f"\n错最多在哪个 token 位置(0=第1个前景):")
    for pos in sorted(pos_wrong):
        print(f"  pos {pos}: {pos_wrong[pos]} 次")

    print(f"\nword_acc 低的词(Top 8):")
    bad = [(w, ok, tot) for w, (ok, tot) in samples_per_word.items()
           if tot > 0 and ok/max(1, tot) < 0.8]
    bad.sort(key=lambda x: ok/max(1, x[2]))
    for w, ok, tot in bad[:8]:
        print(f"  {w}: {ok}/{tot} = {ok/max(1,tot):.2%}")

    print(f"\n5 个错例样本:")
    for w, dec, diffs in word_results[:5]:
        print(f"  GT={w}  pred={dec}  diffs={diffs}")

    # 嵌入分析
    embs, char_ids, pos_ids = [], [], []
    with torch.no_grad():
        for s in val_samples[:100]:
            word = s["target_word"]
            rendered = s["rendered_word"]
            try:
                vecs, tokens = encode_tokens(s["rgb"], encoder, char_hints=list(rendered))
            except Exception:
                continue
            targets = build_targets(len(tokens), rendered).to(DEVICE)
            for j in range(1, len(tokens)):
                t = int(targets[j].item())
                if t != BG_ID:
                    embs.append(vecs[j].cpu().numpy())
                    char_ids.append(t)
                    pos_ids.append(j - 1)
    embs = np.stack(embs)
    from sklearn.metrics import silhouette_score
    sil_char = silhouette_score(embs, np.array(char_ids))
    sil_pos = silhouette_score(embs, np.array(pos_ids))
    print(f"\nEmbedding 聚类:")
    print(f"  silhouette (按字符): {sil_char:.4f}")
    print(f"  silhouette (按位置): {sil_pos:.4f}")
    if sil_pos > sil_char:
        print("  -> 位置主导:同位置 token 聚一起,字符身份没学到")
    elif sil_char > 0.1:
        print("  -> 字符聚类:模型已学到字符身份")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--fuse-mode", type=str, default="gate")
    p.add_argument("--samples-per-word", type=int, default=20)
    args = p.parse_args()

    print(f">> 快速训练 {args.epochs} epoch ({args.samples_per_word}/词)...")
    encoder, head, val_samples, target_words, out_dir = quick_train(
        epochs=args.epochs,
        samples_per_word=args.samples_per_word,
        fuse_mode=args.fuse_mode,
    )
    print(f">> 训完,开始诊断")
    diagnose(encoder, head, val_samples, out_dir, ep=args.epochs - 1)


if __name__ == "__main__":
    main()
