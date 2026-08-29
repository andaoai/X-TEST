"""
颜色分解 + 几何/颜色 MLP 融合 + Self-Attention(NLP 类比 word2vec + Transformer)

架构(类比 NLP):
  inputs (N, H, W) 或 (N, H, W, 3)
    ↓ 提升到 RGB + HSV 桶量化(全部像素 → 空间连通域 = K 个 token,无 bg/fg 区分)
  K tokens per sample, each = (hsv_id_tuple, mask)        ← 类比"词"
    ↓ mask 过小 CNN → geom(32-d)  → geom_mlp(32-d)
    ↓ hsv_id → [onehot_H ⊕ onehot_S ⊕ onehot_V] = 46-d → color_mlp(46→64→32)
    ↓ fusion(geom⊕color → 32-d)                          ← "词向量"由几何+颜色共同决定
  K 个 token → Self-Attention(4 heads) → K 个 (32-d)    ← "上下文编码"
    ↓ mean+max+min+K = 97-d
    ├─→ emb_head(97→128) + L2 norm                       ← cosine 相似度
    └─→ cls_trunk(97→64) + ReLU → 4 个分类头              ← 注意力驱动的分类

注意:
  - K 天然可变(简单合成 K=2,真实图 K 可 20+)
  - 颜色用 HSV 三通道各自 15 间隔分桶:H(12 类) ⊕ S(17 类) ⊕ V(17 类) = 46 维
  - mask 模式 (N,H,W) 走 _to_rgb 灰度复制 → HSV 分解(灰度 S=0,V=255 → H_id=0,S_id=0,V_id=16)
  - 模型延迟构造,首次 encode() 时初始化,torch.manual_seed(42) 保证确定性
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

from algorithms.base import BaseAlgorithm, EMBEDDING_DIM
from algorithms.image_decomposition import (
    decompose_image_to_tokens as _hsv_decompose,
    hsv_id_to_onehot as _hsv_id_to_onehot,
    batch_decompose,
    visualize_tokens as visualize_decomposition,
    token_stats,
    N_H_BINS, N_S_BINS, N_V_BINS, COLOR_ONEHOT_DIM,
    H_BIN_SIZE, SV_BIN_SIZE,
    BG_V_THRESHOLD as _BG_V_THRESHOLD,
    MIN_MASK_PIXELS as _MIN_MASK_PIXELS,
)

# ── Token 维度 ──
TOKEN_GEOM_DIM = 32
TOKEN_DIM = 32            # fusion 后单个 token 维度
POOL_IN_DIM = 3 * TOKEN_DIM + 1   # mean+max+min+K = 97
EMB_HEAD_HIDDEN = 64
CLS_TRUNK_HIDDEN = 64
N_ATTN_HEADS = 4



class _MaskEncoder(nn.Module):
    """单 mask → 32-d 几何向量(类比 word embedding 的 w_i 部分)。"""

    def __init__(self, geom_dim: int = TOKEN_GEOM_DIM):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.fc = nn.Linear(32, geom_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, H, W) ∈ [0, 1] → (B, geom_dim)"""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.mean(dim=(2, 3))
        x = self.fc(x)
        return F.normalize(x, dim=-1)


class _ColorMLP(nn.Module):
    """46 维 one-hot HSV id → 32 维颜色向量。"""

    def __init__(self, in_dim: int = COLOR_ONEHOT_DIM, out_dim: int = TOKEN_GEOM_DIM):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 64)
        self.fc2 = nn.Linear(64, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 46) → (B, 32)"""
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class _TokenFusion(nn.Module):
    """geom(32) ⊕ color(32) → token(32)。"""

    def __init__(self, in_dim: int = 2 * TOKEN_GEOM_DIM, out_dim: int = TOKEN_GEOM_DIM):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, geom: torch.Tensor, color: torch.Tensor) -> torch.Tensor:
        """geom: (B, 32)  color: (B, 32)  →  (B, 32)"""
        return self.fc(torch.cat([geom, color], dim=-1))


class _TokenSelfAttention(nn.Module):
    """K 个 token 内部 self-attention(K 可变)。"""

    def __init__(self, dim: int = TOKEN_GEOM_DIM, heads: int = N_ATTN_HEADS):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor, return_weights: bool = False,
                key_padding_mask: torch.Tensor = None):
        """
        tokens: (B, K, 32) → (B, K, 32)
        K=1 时退化为 self-loop,K=2 时开始有信息交互。
        return_weights=True 时返回平均的 (B, K, K) attention weights。
        key_padding_mask: (B, K) bool,True=该位置是 padding(忽略)。
                          内部把 K=0 padding 位置在 softmax 前置 -inf。
        """
        if tokens.size(1) == 0:
            return (tokens, None) if return_weights else tokens
        out, weights = self.attn(tokens, tokens, tokens,
                                 key_padding_mask=key_padding_mask,
                                 need_weights=return_weights)
        out = self.norm(tokens + out)
        if return_weights:
            return out, weights
        return out


class ColorTokensAlgo(BaseAlgorithm):
    name = "color_tokens"
    uses_rgb = True

    def __init__(self):
        self.encoder: _MaskEncoder | None = None
        self.geom_mlp: nn.Linear | None = None
        self.color_mlp: _ColorMLP | None = None
        self.fusion: _TokenFusion | None = None
        # ── 4 个独立 attention 模块:每个任务用自己的关联 ──
        # attn_token:instance grouping(token_belongs)
        # attn_wm:整体 valid/invalid(word_match)
        # attn_n:词数(n_words)
        # attn_wid:整词身份(word_id)
        self.attn_token: _TokenSelfAttention | None = None
        self.attn_wm: _TokenSelfAttention | None = None
        self.attn_n: _TokenSelfAttention | None = None
        self.attn_wid: _TokenSelfAttention | None = None
        # 兼容旧名(单 attention 时代的引用):指向 attn_wm
        self.self_attn: _TokenSelfAttention | None = None
        self.emb_head: nn.Linear | None = None     # pool(97) → 128
        self.cls_trunk: nn.Linear | None = None    # pool(97) → 64
        self.clfs: nn.ModuleDict = nn.ModuleDict()  # 4 个分类头(延迟构造)
        self._device: torch.device | None = None

    def _build(self, device: torch.device):
        torch.manual_seed(42)
        self.encoder = _MaskEncoder(TOKEN_GEOM_DIM).to(device)
        torch.manual_seed(42)
        self.geom_mlp = nn.Linear(TOKEN_GEOM_DIM, TOKEN_GEOM_DIM).to(device)
        torch.manual_seed(42)
        self.color_mlp = _ColorMLP(COLOR_ONEHOT_DIM, TOKEN_GEOM_DIM).to(device)
        torch.manual_seed(42)
        self.fusion = _TokenFusion(2 * TOKEN_GEOM_DIM, TOKEN_GEOM_DIM).to(device)
        # ── 4 个独立 attention(分别 seed,保证各自起点不同)──
        torch.manual_seed(42)
        self.attn_token = _TokenSelfAttention(TOKEN_GEOM_DIM, N_ATTN_HEADS).to(device)
        torch.manual_seed(43)
        self.attn_wm = _TokenSelfAttention(TOKEN_GEOM_DIM, N_ATTN_HEADS).to(device)
        torch.manual_seed(44)
        self.attn_n = _TokenSelfAttention(TOKEN_GEOM_DIM, N_ATTN_HEADS).to(device)
        torch.manual_seed(45)
        self.attn_wid = _TokenSelfAttention(TOKEN_GEOM_DIM, N_ATTN_HEADS).to(device)
        # 兼容引用
        self.self_attn = self.attn_wm
        torch.manual_seed(42)
        # emb_head: pool 97 → 128(用 std=0.1 让初始 embedding 数值合理)
        self.emb_head = nn.Linear(POOL_IN_DIM, EMBEDDING_DIM).to(device)
        with torch.no_grad():
            self.emb_head.weight.normal_(mean=0.0, std=0.1)
            self.emb_head.bias.zero_()
        torch.manual_seed(42)
        self.cls_trunk = nn.Linear(POOL_IN_DIM, CLS_TRUNK_HIDDEN).to(device)
        self._device = device

    @staticmethod
    def _to_rgb(inputs: np.ndarray) -> np.ndarray:
        if inputs.ndim == 3:
            return np.repeat(inputs[..., None], 3, axis=-1)
        return inputs

    def _encode_one_train(self, rgb_i: np.ndarray, mode: str = "bucket") -> torch.Tensor:
        """单样本前向(训练模式,带梯度)→ 返回 L2 归一化后 (EMBEDDING_DIM,) 向量。"""
        tokens, _ = _hsv_decompose(rgb_i, mode=mode)
        masks = np.stack([m for _, m in tokens])                          # (K, H, W)
        onehots = np.stack([_hsv_id_to_onehot(*hsv) for hsv, _ in tokens])  # (K, 46)

        masks_t = torch.from_numpy(masks).unsqueeze(1).to(self._device)   # (K, 1, H, W)
        onehots_t = torch.from_numpy(onehots).to(self._device)             # (K, 46)

        geom = self.encoder(masks_t)                                        # (K, 32)
        g = self.geom_mlp(geom)                                             # (K, 32)
        c = self.color_mlp(onehots_t)                                       # (K, 32)
        tok = self.fusion(g, c)                                             # (K, 32)
        tok = self.self_attn(tok.unsqueeze(0)).squeeze(0)                   # (K, 32) 含上下文

        # pool3 + K 标量
        mean = tok.mean(dim=0)
        mx   = tok.max(dim=0).values
        mn   = tok.min(dim=0).values
        k_feat = torch.tensor([float(tokens.__len__())], device=self._device)
        pool = torch.cat([mean, mx, mn, k_feat])                            # (97,)

        emb = self.emb_head(pool)                                           # (128,)
        return F.normalize(emb, dim=-1)

    def _pool_one(self, rgb_i: np.ndarray, mode: str = "bucket") -> torch.Tensor:
        """返回 pool 后的 97 维向量(供分类头用,不经过 emb_head)。"""
        tokens, _ = _hsv_decompose(rgb_i, mode=mode)
        masks = np.stack([m for _, m in tokens])
        onehots = np.stack([_hsv_id_to_onehot(*hsv) for hsv, _ in tokens])

        masks_t = torch.from_numpy(masks).unsqueeze(1).to(self._device)
        onehots_t = torch.from_numpy(onehots).to(self._device)

        geom = self.encoder(masks_t)
        g = self.geom_mlp(geom)
        c = self.color_mlp(onehots_t)
        tok = self.fusion(g, c)
        tok = self.self_attn(tok.unsqueeze(0)).squeeze(0)

        mean = tok.mean(dim=0)
        mx   = tok.max(dim=0).values
        mn   = tok.min(dim=0).values
        k_feat = torch.tensor([float(tokens.__len__())], device=self._device)
        return torch.cat([mean, mx, mn, k_feat])

    def _tokens_one(self, rgb_i: np.ndarray, return_attn: bool = False, mode: str = "bucket"):
        """返回 attention 之后的 (K, 32) token(供 token-level 可视化)。"""
        tokens, _ = _hsv_decompose(rgb_i, mode=mode)
        masks = np.stack([m for _, m in tokens])
        onehots = np.stack([_hsv_id_to_onehot(*hsv) for hsv, _ in tokens])
        masks_t = torch.from_numpy(masks).unsqueeze(1).to(self._device)
        onehots_t = torch.from_numpy(onehots).to(self._device)
        geom = self.encoder(masks_t)
        g = self.geom_mlp(geom)
        c = self.color_mlp(onehots_t)
        tok = self.fusion(g, c)
        out = self.self_attn(tok.unsqueeze(0), return_weights=return_attn)
        if return_attn:
            tok, attn = out
            return tok.squeeze(0), attn.squeeze(0)  # (K, 32), (K, K)
        return tok.squeeze(0)

    @torch.no_grad()
    def encode(self, inputs: np.ndarray, verbose: bool = True, mode: str = "bucket") -> np.ndarray:
        rgb_batch = self._to_rgb(inputs)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.encoder is None:
            self._build(device)
        self.eval_all()

        N = rgb_batch.shape[0]
        out = np.zeros((N, EMBEDDING_DIM), dtype=np.float32)
        for i in range(N):
            out[i] = self._encode_one_train(rgb_batch[i], mode=mode).cpu().numpy()
        return out.astype(np.float32)

    @torch.no_grad()
    def encode_tokens(self, inputs: np.ndarray, mode: str = "bucket") -> tuple:
        """返回 (N*K, 32) attention 后 token + sample_id + is_fg。"""
        rgb_batch = self._to_rgb(inputs)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.encoder is None:
            self._build(device)
        self.eval_all()

        all_vecs, all_sid, all_fg = [], [], []
        for sample_i in range(len(rgb_batch)):
            tokens, _ = _hsv_decompose(rgb_batch[sample_i], mode=mode)
            tok = self._tokens_one(rgb_batch[sample_i], mode=mode)               # (K, 32)
            all_vecs.append(tok.cpu().numpy())
            n = len(tokens)
            is_fg = np.zeros(n, dtype=np.int32)
            if n > 1:
                is_fg[1:] = 1
            elif n == 1 and tokens[0][1].mean() > 0.1:
                is_fg[0] = 1
            all_sid.append(np.full(n, sample_i, dtype=np.int32))
            all_fg.append(is_fg)
        return (np.concatenate(all_vecs).astype(np.float32),
                np.concatenate(all_sid),
                np.concatenate(all_fg))

    @torch.no_grad()
    def encode_with_attention(self, inputs: np.ndarray, mode: str = "spatial") -> list:
        """返回每张图的 (K, K) attention 矩阵。默认 mode="spatial",适合单词数据集。"""
        rgb_batch = self._to_rgb(inputs)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.encoder is None:
            self._build(device)
        self.eval_all()

        all_attn = []
        for sample_i in range(len(rgb_batch)):
            tokens, _ = _hsv_decompose(rgb_batch[sample_i], mode=mode)
            masks = np.stack([m for _, m in tokens])
            onehots = np.stack([_hsv_id_to_onehot(*hsv) for hsv, _ in tokens])
            masks_t = torch.from_numpy(masks).unsqueeze(1).to(device)
            onehots_t = torch.from_numpy(onehots).to(device)
            geom = self.encoder(masks_t)
            g = self.geom_mlp(geom)
            c = self.color_mlp(onehots_t)
            tok = self.fusion(g, c)
            _, attn = self.self_attn(tok.unsqueeze(0), return_weights=True)
            # attn: (1, K, K) → (K, K)
            all_attn.append(attn.squeeze(0).cpu().numpy())
        return all_attn

    def eval_all(self):
        for m in [self.encoder, self.geom_mlp, self.color_mlp,
                  self.fusion, self.self_attn, self.emb_head, self.cls_trunk]:
            m.eval()

    def train_all(self):
        for m in [self.encoder, self.geom_mlp, self.color_mlp,
                  self.fusion, self.self_attn, self.emb_head, self.cls_trunk]:
            m.train()
        for t in self.clfs:
            self.clfs[t].train()

    def fit(self, inputs: np.ndarray, labels: dict,
            n_epochs: int = 20, lr: float = 1e-3,
            tasks: tuple = ("label",),
            task_weights: dict | None = None,
            verbose: bool = True, mode: str = "bucket") -> dict:
        rgb_batch = self._to_rgb(inputs)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.encoder is None:
            self._build(device)

        task_weights = task_weights or {}
        ys = {}
        for task in tasks:
            if task not in labels:
                raise ValueError(f"labels 缺字段 '{task}'")
            classes = sorted(labels[task].keys())
            cls_to_int = {c: i for i, c in enumerate(classes)}
            y = np.full(len(inputs), -1, dtype=np.int64)
            for cls, idxs in labels[task].items():
                for i in idxs:
                    y[i] = cls_to_int[cls]
            ys[task] = y
            torch.manual_seed(42)
            self.clfs[task] = nn.Linear(CLS_TRUNK_HIDDEN, len(classes)).to(device)

        if verbose:
            print(f"  fit: 多任务 {tasks}, {len(inputs)} 张, n_epochs={n_epochs}")
            for task in tasks:
                print(f"    [{task}] {len(labels[task])} 类")

        params = (list(self.encoder.parameters()) +
                  list(self.geom_mlp.parameters()) +
                  list(self.color_mlp.parameters()) +
                  list(self.fusion.parameters()) +
                  list(self.self_attn.parameters()) +
                  list(self.emb_head.parameters()) +
                  list(self.cls_trunk.parameters()) +
                  list(self.clfs.parameters()))
        opt = torch.optim.Adam(params, lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        self.train_all()
        N = len(inputs)
        loss_curve = []
        for epoch in range(n_epochs):
            perm = np.random.permutation(N)
            total_loss = 0.0
            correct = {t: 0 for t in tasks}
            for i in perm:
                # 共享一次 encode 路径(只过 attn + pool)
                pool = self._pool_one(rgb_batch[i], mode=mode)                       # (97,)

                # 分类通过 cls_trunk
                trunk = F.relu(self.cls_trunk(pool))                      # (64,)

                opt.zero_grad()
                loss = torch.tensor(0.0, device=device)
                for task in tasks:
                    logits = self.clfs[task](trunk)
                    target = torch.tensor([ys[task][i]], device=device)
                    task_loss = loss_fn(logits.unsqueeze(0), target)
                    w = task_weights.get(task, 1.0)
                    loss = loss + w * task_loss
                    if logits.argmax().item() == ys[task][i]:
                        correct[task] += 1

                loss.backward()
                opt.step()
                total_loss += loss.item()

            avg_loss = total_loss / N
            acc = {t: correct[t] / N for t in tasks}
            loss_curve.append(avg_loss)
            if verbose and (epoch + 1) % max(1, n_epochs // 5) == 0:
                accs_str = "  ".join(f"{t}={acc[t]:.1%}" for t in tasks)
                print(f"    epoch {epoch+1:>2}/{n_epochs}  loss={avg_loss:.4f}  acc: {accs_str}")

        self.eval_all()
        return {"final_loss": loss_curve[-1], "final_acc": acc,
                "loss_curve": loss_curve, "tasks": tasks}


# ════════════════════════════════════════════════════════════════════════════
# Batch 版流水线(A + B 改造)
# ──────────────────────────────────────────────────────────────────────────
# 核心思想(与单图路径完全等价,只是用张量并行做):
#   1) _hsv_decompose 仍在 CPU 上 per-sample 跑(连通域是 2D 算法,GPU 不划算;
#      单图通常 < 1ms,瓶颈不在这里)
#   2) 把 N 个样本的 (K_i, H, W) mask、(K_i, 46) onehot 堆到固定 K_max:
#        masks_padded   : (N, K_max, H, W)   padding=0
#        onehots_padded : (N, K_max, 46)     padding=0
#        key_padding_mask: (N, K_max)        True=无效位置
#   3) 一次性送 GPU,CNN → fusion → attn(用 key_padding_mask)→ pool(用 mask)
#   4) 4 个分类头 + token_belongs 头全部向量化
#
# 不变(思想保护):
#   - HSV 桶量化 + 空间连通域
#   - 背景算 token、BG 在最前
#   - token 顺序不强制对齐 char
#   - 无 group_id 信号
# ════════════════════════════════════════════════════════════════════════════

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class _PerSampleTokens:
    """每样本 token 化结果(保留与 _hsv_decompose 同样的字段)。"""
    masks: np.ndarray           # (K, H, W) float32
    hsv_ids: np.ndarray         # (K, 3) int32
    K: int


def precompute_tokens(rgbs: np.ndarray, mode: str = "spatial",
                     char_hints_list: list = None) -> List[_PerSampleTokens]:
    """
    批量跑 _hsv_decompose(仍在 CPU,因连通域是 2D 操作)。
    每样本独立 → 不损失信息。
    char_hints_list: 可选,长度 = N,每个元素是该样本渲染端的字符序列
                    (用于把前景 token 按 x 排序 = word 字符顺序)。
                    None 时前景保持扫描顺序。
    """
    out = []
    for i in range(rgbs.shape[0]):
        hints = char_hints_list[i] if char_hints_list is not None else None
        tokens, _ = _hsv_decompose(rgbs[i], mode=mode, char_hints=hints)
        masks = np.stack([m for _, m in tokens]).astype(np.float32)
        hsv_ids = np.array([hsv for hsv, _ in tokens], dtype=np.int32)
        out.append(_PerSampleTokens(masks=masks, hsv_ids=hsv_ids, K=len(tokens)))
    return out


def pad_tokens_to_batch(per_sample: List[_PerSampleTokens], K_max: int
                        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    把变长 K 的 token 序列 padding 到 K_max。
    Returns:
        masks_padded:   (N, K_max, H, W) float32 on CPU(送 GPU 之前)
        onehots_padded: (N, K_max, 46)   float32
        kp_mask:         (N, K_max)        bool  True=padding(忽略)
        K_true:          (N,)              int32 真实 K
    """
    N = len(per_sample)
    H, W = per_sample[0].masks.shape[1], per_sample[0].masks.shape[2]
    masks_padded = np.zeros((N, K_max, H, W), dtype=np.float32)
    onehots_padded = np.zeros((N, K_max, COLOR_ONEHOT_DIM), dtype=np.float32)
    kp_mask = np.ones((N, K_max), dtype=bool)   # 默认全 True=padding
    K_true = np.zeros(N, dtype=np.int32)

    for i, ps in enumerate(per_sample):
        k = min(ps.K, K_max)
        masks_padded[i, :k] = ps.masks[:k]
        # onehot
        for j in range(k):
            h_id, s_id, v_id = ps.hsv_ids[j]
            onehots_padded[i, j] = _hsv_id_to_onehot(int(h_id), int(s_id), int(v_id))
        kp_mask[i, :k] = False                     # 有效位置
        K_true[i] = k
    return (torch.from_numpy(masks_padded),
            torch.from_numpy(onehots_padded),
            torch.from_numpy(kp_mask),
            torch.from_numpy(K_true))


@torch.no_grad()
def encode_packed_batch(algo: "ColorTokensAlgo",
                        masks_padded: torch.Tensor,
                        onehots_padded: torch.Tensor,
                        kp_mask: torch.Tensor,
                        K_true: torch.Tensor,
                        return_attn: bool = False
                        ) -> dict:
    """
    一次性前向 N 个 padding 后样本,返回:
      tok:        (N, K_max, 32)  attention 后 token
      pool:       (N, 97)        mean+max+min+K(只对有效 token 池化)
      K_true:     (N,)            int32
      attn:       (N, K_max, K_max) float32  可选
    注:padding 位置的 tok 全 0(attn mask 保证),pool 通过 K_true 屏蔽。
    """
    device = algo._device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        algo._build(device)
    masks = masks_padded.to(device, non_blocking=True)               # (N, K, H, W)
    onehots = onehots_padded.to(device, non_blocking=True)           # (N, K, 46)
    kp = kp_mask.to(device)                                          # (N, K) True=padding
    N, K_max = masks.shape[0], masks.shape[1]

    # ── 几何 token:CNN 看每张 mask ──
    # reshape (N, K, H, W) → (N*K, 1, H, W) → 过 CNN → (N*K, 32) → reshape (N, K, 32)
    masks_flat = masks.reshape(N * K_max, 1, masks.shape[-2], masks.shape[-1])
    geom_flat = algo.encoder(masks_flat)                             # (N*K, 32)
    geom = geom_flat.view(N, K_max, TOKEN_GEOM_DIM)
    g = algo.geom_mlp(geom)                                          # (N, K, 32)
    c = algo.color_mlp(onehots)                                       # (N, K, 32)
    tok = algo.fusion(g, c)                                           # (N, K, 32)

    # ── 4 个独立 Self-attention:每个任务用自己的关联 ──
    # attn_token:instance grouping(token_belongs)
    # attn_wm:整体 valid/invalid(word_match)
    # attn_n:词数(n_words)
    # attn_wid:整词身份(word_id)
    tok_token, attn_w_token = algo.attn_token(tok, return_weights=True,
                                              key_padding_mask=kp)
    tok_wm, attn_w_wm = algo.attn_wm(tok, return_weights=True,
                                     key_padding_mask=kp)
    tok_n, attn_w_n = algo.attn_n(tok, return_weights=True,
                                  key_padding_mask=kp)
    tok_wid, attn_w_wid = algo.attn_wid(tok, return_weights=True,
                                        key_padding_mask=kp)

    # ── Pool:每个 attention 输出独立 pool(mean+max+min+K) ──
    def _pool(tok_attn):
        valid_mask = (~kp)                                                  # (N, K) bool
        safe_count = valid_mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
        tok_masked = tok_attn * valid_mask.float().unsqueeze(-1)            # (N, K, 32)
        mean = tok_masked.sum(dim=1) / safe_count                           # (N, 32)
        NEG_INF = torch.finfo(tok_attn.dtype).min
        POS_INF = torch.finfo(tok_attn.dtype).max
        tok_for_max = torch.masked_fill(tok_attn, kp.unsqueeze(-1), NEG_INF)
        tok_for_min = torch.masked_fill(tok_attn, kp.unsqueeze(-1), POS_INF)
        mx = tok_for_max.max(dim=1).values
        mn = tok_for_min.min(dim=1).values
        k_feat = K_true.float().to(device).unsqueeze(-1) / float(K_max)
        return torch.cat([mean, mx, mn, k_feat], dim=-1)                     # (N, 97)

    pool_token = _pool(tok_token)
    pool_wm    = _pool(tok_wm)
    pool_n     = _pool(tok_n)
    pool_wid   = _pool(tok_wid)
    # 保留旧 pool(用 tok_wm,即 attn_wm 输出,跟旧 attn 一致)
    pool = pool_wm

    out = {
        # 4 套 attention 输出 + 4 套 pool
        "tok_token": tok_token, "pool_token": pool_token,
        "tok_wm":    tok_wm,    "pool_wm":    pool_wm,
        "tok_n":     tok_n,     "pool_n":     pool_n,
        "tok_wid":   tok_wid,   "pool_wid":   pool_wid,
        # 兼容旧引用
        "tok":       tok_wm,    "tok_pre_attn": tok, "pool": pool,
        "K_true":    K_true,
    }
    if return_attn:
        out["attn_token"] = attn_w_token
        out["attn_wm"]    = attn_w_wm
        out["attn_n"]     = attn_w_n
        out["attn_wid"]   = attn_w_wid
        out["attn"]       = attn_w_wm   # 兼容旧
    return out
