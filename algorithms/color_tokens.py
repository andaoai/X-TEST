"""
颜色分解 + 几何/颜色 MLP 融合 + Self-Attention(NLP 类比 word2vec + Transformer)

架构(类比 NLP):
  inputs (N, H, W) 或 (N, H, W, 3)
    ↓ 提升到 RGB + HSV 桶量化(背景也算 token)
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

# ── HSV 桶配置(15 间隔)──
# OpenCV H ∈ [0, 179],S/V ∈ [0, 255]
H_BIN_SIZE = 15
SV_BIN_SIZE = 15
N_H_BINS = 180 // H_BIN_SIZE             # 12
N_S_BINS = (256 + SV_BIN_SIZE - 1) // SV_BIN_SIZE   # 17
N_V_BINS = (256 + SV_BIN_SIZE - 1) // SV_BIN_SIZE   # 17
COLOR_ONEHOT_DIM = N_H_BINS + N_S_BINS + N_V_BINS   # 12 + 17 + 17 = 46

_BG_V_THRESHOLD = 10
_MIN_MASK_PIXELS = 4

# ── Token 维度 ──
TOKEN_GEOM_DIM = 32
TOKEN_DIM = 32            # fusion 后单个 token 维度
POOL_IN_DIM = 3 * TOKEN_DIM + 1   # mean+max+min+K = 97
EMB_HEAD_HIDDEN = 64
CLS_TRUNK_HIDDEN = 64
N_ATTN_HEADS = 4


def _hsv_decompose(rgb: np.ndarray, mode: str = "bucket"):
    """
    rgb: (H, W, 3) float32 ∈ [0, 1]

    mode="bucket"  : 按 (h_id, s_id, v_id) 三元组细粒度分桶,K 可能 1-20+
                     (单字母合成数据,边缘抗锯齿会被细分)
    mode="spatial" : 前景像素先按 HSV 桶粗归一得 mask,再对**前景 mask 做连通域**,
                     K = 连通域数 + 背景。
                     适合"连续色块"图(单词数据集渲染的就是连续色块字母,
                     三个字母水平摆放,空间不连通)

    returns:
        tokens: list of (hsv_id_tuple, mask)
            hsv_id_tuple: (h_id, s_id, v_id) 各 int
            mask:         (H, W) float32 二值
        bg_mask: (H, W) bool
    """
    h, w, _ = rgb.shape
    rgb_u8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)             # (H, W, 3) uint8

    H_ch = hsv[:, :, 0]
    S_ch = hsv[:, :, 1]
    V_ch = hsv[:, :, 2]

    bg_mask = V_ch < _BG_V_THRESHOLD
    fg_pixels = ~bg_mask

    h_id = (H_ch // H_BIN_SIZE).astype(np.int32)
    s_id = np.minimum(S_ch // SV_BIN_SIZE, N_S_BINS - 1).astype(np.int32)
    v_id = np.minimum(V_ch // SV_BIN_SIZE, N_V_BINS - 1).astype(np.int32)

    tokens = []

    if bg_mask.sum() > _MIN_MASK_PIXELS:
        bg_h = int(h_id[bg_mask].mean())
        bg_s = int(s_id[bg_mask].mean())
        bg_v = int(v_id[bg_mask].mean())
        tokens.append(((bg_h, bg_s, bg_v), bg_mask.astype(np.float32)))

    if fg_pixels.sum() > _MIN_MASK_PIXELS:
        if mode == "spatial":
            # 连通域:对前景像素做 connected components
            from scipy import ndimage
            fg_u8 = fg_pixels.astype(np.uint8)
            labeled, n_comp = ndimage.label(fg_u8, structure=np.ones((3, 3)))
            for comp_id in range(1, n_comp + 1):
                m = (labeled == comp_id)
                if m.sum() > _MIN_MASK_PIXELS:
                    hi = int(h_id[m].mean())
                    si = int(s_id[m].mean())
                    vi = int(v_id[m].mean())
                    tokens.append(((hi, si, vi), m.astype(np.float32)))
        else:
            # bucket 模式(原版):按 HSV 桶细粒度分,抗锯齿可能 K=10+
            bucket = h_id * (N_S_BINS * N_V_BINS) + s_id * N_V_BINS + v_id
            fg_buckets = np.unique(bucket[fg_pixels])
            for b in fg_buckets:
                m = (bucket == b) & fg_pixels
                if m.sum() > _MIN_MASK_PIXELS:
                    hi = int(h_id[m].mean())
                    si = int(s_id[m].mean())
                    vi = int(v_id[m].mean())
                    tokens.append(((hi, si, vi), m.astype(np.float32)))

    if not tokens:
        tokens.append(((0, 0, 0), np.zeros((h, w), np.float32)))

    return tokens, bg_mask


def _hsv_id_to_onehot(h_id: int, s_id: int, v_id: int) -> np.ndarray:
    """(h_id, s_id, v_id) → (46,) 拼接 one-hot。"""
    v = np.zeros(COLOR_ONEHOT_DIM, dtype=np.float32)
    if 0 <= h_id < N_H_BINS:
        v[h_id] = 1.0
    s_off = N_H_BINS
    if 0 <= s_id < N_S_BINS:
        v[s_off + s_id] = 1.0
    sv_off = N_H_BINS + N_S_BINS
    if 0 <= v_id < N_V_BINS:
        v[sv_off + v_id] = 1.0
    return v


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

    def forward(self, tokens: torch.Tensor, return_weights: bool = False):
        """
        tokens: (B, K, 32) → (B, K, 32)
        K=1 时退化为 self-loop,K=2 时开始有信息交互。
        return_weights=True 时返回平均的 (B, K, K) attention weights。
        """
        if tokens.size(1) == 0:
            return (tokens, None) if return_weights else tokens
        out, weights = self.attn(tokens, tokens, tokens, need_weights=return_weights)
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
        torch.manual_seed(42)
        self.self_attn = _TokenSelfAttention(TOKEN_GEOM_DIM, N_ATTN_HEADS).to(device)
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
