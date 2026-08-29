"""
图像分解为 token 列表 —— 端到端实例分割的"感知前端"。

两步确定性分解(无学习参数):
  1. HSV 颜色分桶
     - H(0-179)  按 15 间隔分 12 桶
     - S/V(0-255) 按 15 间隔各分 17 桶
     - 三者拼成 46 维颜色 ID
  2. 同色像素按 8-连通域拆分
     - 同 HSV 桶 + 空间连通 = 一个 token
     - 黑像素(V<10)单成 BG token,固定放位置 0

设计原则:
  - 背景也是 token(本质也是 token,关联度是 0)
  - token 顺序可乱(由 char_hints 决定时,前景按 x 排 = word 字符序)
  - 端到端:输入 → 此模块 → token → 后续模型 → loss
  - 永远不减 token:每个连通域保留,过滤 < 4 像素小斑点

用法:
  import numpy as np
  from algorithms.image_decomposition import (
      decompose_image_to_tokens,
      batch_decompose,
      visualize_tokens,
      token_stats,
  )

  rgb = ...  # (H, W, 3) float32 [0, 1]
  tokens, bg_mask = decompose_image_to_tokens(rgb, mode="spatial")
  # tokens: [(hsv_id, mask), ...], 位置 0 = BG, 1..K-1 = 前景
"""
from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np
import cv2
from scipy import ndimage

# ── HSV 桶配置(15 间隔)──
# OpenCV H ∈ [0, 179],S/V ∈ [0, 255]
H_BIN_SIZE = 15
SV_BIN_SIZE = 15
N_H_BINS = 180 // H_BIN_SIZE                    # 12
N_S_BINS = (256 + SV_BIN_SIZE - 1) // SV_BIN_SIZE   # 17
N_V_BINS = (256 + SV_BIN_SIZE - 1) // SV_BIN_SIZE   # 17
COLOR_ONEHOT_DIM = N_H_BINS + N_S_BINS + N_V_BINS   # 12 + 17 + 17 = 46

# 背景判定阈值 & mask 最小像素数
BG_V_THRESHOLD = 10
MIN_MASK_PIXELS = 4


# ─────────────────────────────────────────────────────────────────────────
# 核心:单图分解
# ─────────────────────────────────────────────────────────────────────────
def decompose_image_to_tokens(
    rgb: np.ndarray,
    mode: str = "spatial",
    char_hints: Optional[List[str]] = None,
    min_pixels: int = MIN_MASK_PIXELS,
    bg_v_thresh: int = BG_V_THRESHOLD,
) -> Tuple[List[Tuple[Tuple[int, int, int], np.ndarray]], np.ndarray]:
    """
    把 RGB 图像分解成 token 列表(每 token = (hsv_id, mask))。

    参数:
        rgb:  (H, W, 3) float32 ∈ [0, 1]
        mode: "spatial" 前景做 8-连通域拆分(每个连通域 = 一个 token)
              "bucket"  前景按 HSV 桶整桶分(同色但不相邻 → 一个 token)
        char_hints: 可选。单词渲染时的字符序列 ["g","o"] 等,
                    用于把前景 token 按质心 x 排序,让 token 顺序 = 字符顺序。
                    None 时前景按扫描顺序。
        min_pixels:  小于此像素数的 mask 丢弃(过滤噪点)
        bg_v_thresh: V < 此值视为背景

    返回:
        tokens:  list of (hsv_id_tuple, mask)
                 hsv_id_tuple: (h_id, s_id, v_id) 各 int
                 mask:         (H, W) float32 二值
                 **位置 0 = BG token**(若存在);位置 1..K-1 = 前景
        bg_mask: (H, W) bool,对外参考用
    """
    assert rgb.ndim == 3 and rgb.shape[-1] == 3, f"期望 (H, W, 3),得到 {rgb.shape}"
    h, w, _ = rgb.shape
    rgb_u8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)             # (H, W, 3) uint8

    H_ch = hsv[:, :, 0]
    S_ch = hsv[:, :, 1]
    V_ch = hsv[:, :, 2]

    # 判定"非彩色像素":V 太低(黑)或 S 太低且 V 高(白/灰)
    is_colored = (V_ch >= bg_v_thresh) | (S_ch >= bg_v_thresh)
    h_id = (H_ch // H_BIN_SIZE).astype(np.int32)
    s_id = np.minimum(S_ch // SV_BIN_SIZE, N_S_BINS - 1).astype(np.int32)
    v_id = np.minimum(V_ch // SV_BIN_SIZE, N_V_BINS - 1).astype(np.int32)

    bg_mask = V_ch < bg_v_thresh
    tokens: List[Tuple[Tuple[int, int, int], np.ndarray]] = []

    # ── 1. BG token(固定位置 0)──
    if bg_mask.sum() > min_pixels:
        bh = int(h_id[bg_mask].mean())
        bs = int(s_id[bg_mask].mean())
        bv = int(v_id[bg_mask].mean())
        tokens.append(((bh, bs, bv), bg_mask.astype(np.float32)))

    # ── 2. 前景 token:H 桶 + 连通域 ──
    if mode == "spatial":
        # 8-连通域拆分:同色 + 连通 = 一个 token
        fg_u8 = is_colored.astype(np.uint8)
        labeled, n_comp = ndimage.label(fg_u8, structure=np.ones((3, 3)))
        fg_tokens = []
        for comp_id in range(1, n_comp + 1):
            m = (labeled == comp_id)
            if m.sum() > min_pixels:
                hi = int(h_id[m].mean())
                si = int(s_id[m].mean())
                vi = int(v_id[m].mean())
                fg_tokens.append(((hi, si, vi), m.astype(np.float32)))
    elif mode == "bucket":
        # 仅按 HSV 桶整桶:同色不相邻也算同一 token
        bucket = h_id * (N_S_BINS * N_V_BINS) + s_id * N_V_BINS + v_id
        fg_buckets = np.unique(bucket[is_colored])
        fg_tokens = []
        for b in fg_buckets:
            m = (bucket == b) & is_colored
            if m.sum() > min_pixels:
                hi = int(h_id[m].mean())
                si = int(s_id[m].mean())
                vi = int(v_id[m].mean())
                fg_tokens.append(((hi, si, vi), m.astype(np.float32)))
    else:
        raise ValueError(f"mode 必须是 'spatial' 或 'bucket',得到 {mode!r}")

    # ── 3. 排序:char_hints 决定 x 序(左→右 = 单词字符顺序)──
    if char_hints and len(char_hints) == len(fg_tokens):
        centroids = []
        for _, m in fg_tokens:
            ys, xs = np.where(m > 0.5)
            centroids.append(xs.mean() if len(xs) else 0.0)
        order = np.argsort(centroids)
        fg_tokens = [fg_tokens[i] for i in order]

    tokens.extend(fg_tokens)

    # ── 4. 兜底:全空时给一个空 mask ──
    if not tokens:
        tokens.append(((0, 0, 0), np.zeros((h, w), np.float32)))

    return tokens, bg_mask


# ─────────────────────────────────────────────────────────────────────────
# 批量:N 张图同时分解(每图独立,CPU 跑;连通域是 2D 算法,GPU 不划算)
# ─────────────────────────────────────────────────────────────────────────
def batch_decompose(
    rgbs: np.ndarray,
    mode: str = "spatial",
    char_hints_list: Optional[List[List[str]]] = None,
    **kwargs,
) -> List[Tuple[List[Tuple[Tuple[int, int, int], np.ndarray]], np.ndarray]]:
    """
    rgbs: (N, H, W, 3) float32
    char_hints_list: 可选,长度 N,每样本的 char hints

    返回:长度 N 的 list,每项 = (tokens, bg_mask)
    """
    out = []
    for i in range(rgbs.shape[0]):
        hints = char_hints_list[i] if char_hints_list is not None else None
        out.append(decompose_image_to_tokens(rgbs[i], mode=mode,
                                             char_hints=hints, **kwargs))
    return out


# ─────────────────────────────────────────────────────────────────────────
# 工具:统计
# ─────────────────────────────────────────────────────────────────────────
def token_stats(tokens: List[Tuple[Tuple[int, int, int], np.ndarray]]) -> dict:
    """返回 token 集合的统计信息。"""
    K = len(tokens)
    sizes = [int(m.sum()) for _, m in tokens]
    is_bg = [int(m.sum() > 0.5 * m.size) for _, m in tokens]  # 像素占比 > 50% 当 BG
    return {
        "K": K,
        "K_fg": sum(1 for x in is_bg if x == 0),
        "K_bg": sum(1 for x in is_bg if x == 1),
        "min_size": min(sizes) if sizes else 0,
        "max_size": max(sizes) if sizes else 0,
        "mean_size": float(np.mean(sizes)) if sizes else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────
# 工具:可视化(把 token 染成不同颜色叠回原图)
# ─────────────────────────────────────────────────────────────────────────
# 12 个区分色(给前景 token 上色用,纯视觉)
_TOK_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255),
    (0, 128, 255), (128, 255, 0), (255, 0, 128), (128, 128, 128),
]


def visualize_tokens(
    rgb: np.ndarray,
    tokens: List[Tuple[Tuple[int, int, int], np.ndarray]],
    show_index: bool = True,
) -> np.ndarray:
    """
    把 token 列表叠回原图:每个前景 token 用一种区分色覆盖,带索引号。

    rgb:    (H, W, 3) float32 [0, 1]
    tokens: decompose_image_to_tokens 的输出
    show_index: 是否在每个 token 中心写索引号

    返回: (H, W, 3) uint8,可直接 imwrite / plt.imshow
    """
    import cv2
    h, w, _ = rgb.shape
    canvas = (np.clip(rgb, 0, 1) * 255).astype(np.uint8).copy()
    overlay = canvas.copy()

    # 位置 0 = BG,半透明灰覆盖
    for i, (hsv_id, m) in enumerate(tokens):
        color = _TOK_COLORS[i % len(_TOK_COLORS)]
        mask_u8 = (m > 0.5).astype(np.uint8)
        if i == 0 and mask_u8.sum() > 0.3 * h * w:
            # BG 特殊处理:灰色
            color = (80, 80, 80)
        overlay[mask_u8 > 0] = color

    # 50% 混合
    cv2.addWeighted(overlay, 0.5, canvas, 0.5, 0, canvas)

    if show_index:
        for i, (_, m) in enumerate(tokens):
            ys, xs = np.where(m > 0.5)
            if len(xs) == 0:
                continue
            cx, cy = int(xs.mean()), int(ys.mean())
            cv2.putText(canvas, str(i), (cx - 6, cy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    return canvas


# ─────────────────────────────────────────────────────────────────────────
# 工具:HSV id → one-hot 编码
# ─────────────────────────────────────────────────────────────────────────
def hsv_id_to_onehot(h_id: int, s_id: int, v_id: int) -> np.ndarray:
    """(h_id, s_id, v_id) → (46,) 拼接 one-hot,供下游颜色 MLP 用。"""
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


import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────
# 附加层:颜色感知 token 编码器(端到端可学习)
# ─────────────────────────────────────────────────────────────────────────
# 上面的 decompose_* 是确定性分解,这里追加一层"可学习编码",把
# (hsv_id, mask) → 32-d 向量 token,供下游分类/聚类/检索。
#
# 关键设计:多种 fuse_mode 可选,适应不同任务:
#   - "concat"    geom ⊕ color → Linear → out_dim
#                 最简单,最通用,默认
#   - "add"       geom + color_mlp
#                 假设 geom 和 color 在同一语义空间(参数少,快)
#   - "gate"      geom * sigmoid(color_gate)
#                 颜色"调制"几何 — "红 mask" 和 "蓝 mask" 表现不同
#                 颜色敏感任务推荐
#   - "bilinear"  geom ⊗ color 外积 → Linear → out_dim
#                 高阶交互,参数最多,样本少时易过拟合,但表达力最强
#
# 训练策略:跟下游任务 loss 一起端到端训练。encoder.parameters() 加进 optimizer 即可。


class _MaskEncoder(nn.Module):
    """单 mask → geom_dim 维几何向量(小 CNN)。"""

    def __init__(self, geom_dim: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.fc = nn.Linear(32, geom_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, H, W) ∈ [0, 1]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.mean(dim=(2, 3))
        x = self.fc(x)
        return x   # (B, geom_dim),不归一化 — 留给 fuse 决定


class _ColorMLP(nn.Module):
    """46 维 HSV one-hot → color_dim 维颜色向量。"""

    def __init__(self, in_dim: int = COLOR_ONEHOT_DIM, hidden: int = 64,
                 out_dim: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 46)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x   # (B, color_dim)


class _BilinearFusion(nn.Module):
    """外积融合:geom ⊗ color → Linear → out_dim"""

    def __init__(self, geom_dim: int, color_dim: int, out_dim: int):
        super().__init__()
        self.fc = nn.Linear(geom_dim * color_dim, out_dim)

    def forward(self, g: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # g: (B, G), c: (B, C) → outer → (B, G*C) → linear → (B, out_dim)
        outer = g.unsqueeze(-1) * c.unsqueeze(-2)   # (B, G, C)
        return self.fc(outer.flatten(1))              # (B, out_dim)


class _GateFusion(nn.Module):
    """门控融合:geom * sigmoid(color_gate_proj(color))"""

    def __init__(self, color_dim: int, geom_dim: int):
        super().__init__()
        self.gate = nn.Linear(color_dim, geom_dim)

    def forward(self, g: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # g: (B, G), c: (B, C) → (B, G) * (B, G) → (B, G)
        return g * torch.sigmoid(self.gate(c))


class ColorTokenEncoder(nn.Module):
    """
    把 (hsv_id, mask) 对编码成 out_dim 维 token 向量(端到端可学习)。

    多种 fuse_mode 适应不同任务:
      "concat" / "add" / "gate" / "bilinear"
    详细说明见模块顶部 docstring。

    用法:
        enc = ColorTokenEncoder(fuse_mode="gate")    # 颜色敏感任务
        enc.to(device).train()
        # 跟下游 head 一起放进 optimizer:
        #   opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()))
    """

    FUSE_MODES = ("concat", "add", "gate", "bilinear")

    def __init__(self,
                 geom_dim: int = 32,
                 color_in: int = COLOR_ONEHOT_DIM,
                 out_dim: int = 32,
                 fuse_mode: str = "concat"):
        super().__init__()
        assert fuse_mode in self.FUSE_MODES, \
            f"fuse_mode 必须是 {self.FUSE_MODES},得到 {fuse_mode!r}"
        self.fuse_mode = fuse_mode
        self.geom_dim = geom_dim
        self.out_dim = out_dim

        self.mask_enc = _MaskEncoder(geom_dim)
        self.color_enc = _ColorMLP(color_in, hidden=64, out_dim=geom_dim)

        if fuse_mode == "concat":
            self.fuse = nn.Linear(2 * geom_dim, out_dim)
        elif fuse_mode == "add":
            self.fuse = nn.Identity()
        elif fuse_mode == "gate":
            self.fuse = _GateFusion(color_dim=geom_dim, geom_dim=geom_dim)
        elif fuse_mode == "bilinear":
            self.fuse = _BilinearFusion(geom_dim=geom_dim, color_dim=geom_dim,
                                        out_dim=out_dim)

    def forward(self, hsv_ids: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """
        hsv_ids: (B, 3) int long   — 每行的 (h_id, s_id, v_id)
        masks:   (B, 1, H, W) float — 每行的 mask

        returns: (B, out_dim) — 完整 token 向量
        """
        # 颜色:HSV id → one-hot → MLP(全向量化,无 Python 循环)
        B = hsv_ids.size(0)
        device = masks.device
        # 构造 46 维拼接 one-hot:
        #   [0..N_H_BINS)               = H 段(N_H_BINS 维)
        #   [N_H_BINS..N_H_BINS+N_S)    = S 段(N_S_BINS 维)
        #   [N_H_BINS+N_S..46)          = V 段(N_V_BINS 维)
        colors = torch.zeros(B, COLOR_ONEHOT_DIM, device=device)
        h, s, v = hsv_ids[:, 0], hsv_ids[:, 1], hsv_ids[:, 2]
        # H 段
        colors[torch.arange(B, device=device), h] = 1.0
        # S 段
        colors[torch.arange(B, device=device), N_H_BINS + s] = 1.0
        # V 段
        colors[torch.arange(B, device=device), N_H_BINS + N_S_BINS + v] = 1.0
        c = self.color_enc(colors)                              # (B, geom_dim)
        # 几何:mask → CNN
        g = self.mask_enc(masks)                                # (B, geom_dim)
        # 融合
        if self.fuse_mode == "concat":
            return self.fuse(torch.cat([g, c], dim=-1))
        elif self.fuse_mode == "add":
            return g + c
        elif self.fuse_mode == "gate":
            return self.fuse(g, c)
        elif self.fuse_mode == "bilinear":
            return self.fuse(g, c)


def encode_color_tokens(rgb: np.ndarray,
                        encoder: Optional["ColorTokenEncoder"] = None,
                        char_hints: Optional[List[str]] = None,
                        device: Optional[torch.device] = None,
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    一站式:图像 → (K, out_dim) 颜色敏感 token 向量。

    rgb:        (H, W, 3) float32 [0, 1]
    encoder:    ColorTokenEncoder 实例;None 时返回原始 (hsv_id, mask) 对
    char_hints: 可选,前景按 x 排 = word 字符序
    device:     encoder 所在 device

    返回:
        vecs:   (K, out_dim) float32,每行一个 token 向量;encoder=None 时为 list
        bg_mask: (H, W) bool
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) 分解
    tokens, bg_mask = decompose_image_to_tokens(rgb, mode="spatial",
                                                char_hints=char_hints)

    if encoder is None:
        return tokens, bg_mask   # 原始 (color, mask) 对

    # 2) 编码
    encoder = encoder.to(device).eval()
    K = len(tokens)
    hsv_ids = torch.tensor([t[0] for t in tokens], dtype=torch.long)  # (K, 3)
    masks = np.stack([t[1] for t in tokens])[:, None, :, :]            # (K, 1, H, W)
    masks_t = torch.from_numpy(masks).to(device).float()

    with torch.no_grad():
        vecs = encoder(hsv_ids, masks_t)                                # (K, out_dim)

    return vecs.cpu().numpy(), bg_mask


# ─────────────────────────────────────────────────────────────────────────
# 烟测
# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 构造一张测试图:2 字母 "go",红字 + 黑背景
    h, w = 64, 64
    rgb = np.zeros((h, w, 3), np.float32)
    # 字母 'g':红色方块 (x=15-25, y=20-50)
    rgb[20:50, 15:25, 0] = 1.0
    # 字母 'o':绿色方块 (x=35-45, y=20-50)
    rgb[20:50, 35:45, 1] = 1.0

    tokens, bg_mask = decompose_image_to_tokens(rgb, mode="spatial",
                                                char_hints=["g", "o"])
    print(f"K = {len(tokens)}")
    for i, ((h, s, v), m) in enumerate(tokens):
        label = "BG" if i == 0 else f"FG{i}"
        print(f"  token {i} [{label}]: pixels={int(m.sum())}, hsv=({h},{s},{v})")

    print("\nstats:", token_stats(tokens))

    viz = visualize_tokens(rgb, tokens)
    cv2.imwrite("/tmp/decompose_viz.png",
                cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))
    print("  → saved /tmp/decompose_viz.png")

    # ── ColorTokenEncoder 4 种 fuse_mode 烟测 ──
    print("\n=== ColorTokenEncoder fuse modes ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    K = len(tokens)
    hsv_ids = torch.tensor([t[0] for t in tokens], dtype=torch.long)
    masks = np.stack([t[1] for t in tokens])[:, None, :, :]
    masks_t = torch.from_numpy(masks).to(device).float()

    for mode in ColorTokenEncoder.FUSE_MODES:
        enc = ColorTokenEncoder(fuse_mode=mode).to(device).eval()
        with torch.no_grad():
            vecs = enc(hsv_ids, masks_t)
        print(f"  {mode:>9s}: out shape = {tuple(vecs.shape)}  "
              f"(K={K}, out_dim={enc.out_dim})")
