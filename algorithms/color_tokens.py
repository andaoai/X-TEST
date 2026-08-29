"""
颜色分解 + 小 CNN token 编码(NLP 类比 word2vec)

架构(类比 NLP):
  inputs (N, H, W) 或 (N, H, W, 3)
    ↓ 提升到 RGB + HSV 桶量化(背景也算 token)
  K tokens per sample, each = (color, mask)            ← 类比"词"
    ↓ mask 过小 CNN → 32-d L2 归一化                   ← 类比 word embedding
  ⊕ 拼接 color (3-d) → 35-d 完整 token embedding
    ↓ 聚合 mean+max+min + token count → Linear → 128   ← 类比 sentence embedding
  (N, 128) L2-normalized embedding

注意:
  - K 天然可变(简单合成 K=2,真实图 K 可 20+);架构不依赖固定 K
  - mask 模式 (N,H,W) 走 _to_rgb 灰度复制 → HSV 分解(灰度 S=0 → H=0 → 全部归 Red 桶,这是预期退化)
  - 模型延迟构造,首次 encode() 时初始化,torch.manual_seed(42) 保证确定性
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

from algorithms.base import BaseAlgorithm, EMBEDDING_DIM

# ── HSV 桶量化配置 ──
N_HUE_BINS = 8
_BIN_CENTERS = (np.arange(N_HUE_BINS) + 0.5) * (180.0 / N_HUE_BINS)   # (N_HUE_BINS,)
_BG_V_THRESHOLD = 10                                                 # V < 10 → 背景(真黑)
_MIN_MASK_PIXELS = 4                                                 # 过滤 < 4 像素的小斑点

# ── Token 维度 ──
TOKEN_GEOM_DIM = 32     # CNN 输出的几何部分
TOKEN_COLOR_DIM = 3     # 颜色元数据(RGB)
TOKEN_DIM = TOKEN_GEOM_DIM + TOKEN_COLOR_DIM      # 35
AGG_IN_DIM = 3 * TOKEN_DIM + 1                    # mean+max+min+count = 106


def _hsv_decompose(rgb: np.ndarray):
    """
    rgb: (H, W, 3) float32 ∈ [0, 1]
    returns:
        tokens: list of (color, mask)  ← **背景也算 token**
            color: (3,) float32
            mask:  (H, W) float32 二值
        bucket_full: (H, W) int32(背景 = -1, 前景 = 桶 id)
        bg_mask: (H, W) bool
    """
    h, w, _ = rgb.shape
    rgb_u8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)             # (H, W, 3) uint8

    H_ch = hsv[:, :, 0].astype(np.float32)
    V_ch = hsv[:, :, 2]

    bg_mask = V_ch < _BG_V_THRESHOLD
    fg_pixels = ~bg_mask

    tokens = []

    # ── 背景 token(只要有黑像素就保留,颜色 = 平均 RGB)──
    if bg_mask.sum() > _MIN_MASK_PIXELS:
        bg_color = (rgb * bg_mask[..., None]).sum(axis=(0, 1)) / bg_mask.sum()
        tokens.append((bg_color.astype(np.float32), bg_mask.astype(np.float32)))

    # ── 前景 token:按 H 圆形距离分桶,每个非空桶一个 token ──
    if fg_pixels.sum() > _MIN_MASK_PIXELS:
        H_diff = np.abs(H_ch[:, :, None] - _BIN_CENTERS[None, None, :])
        H_diff_circ = np.minimum(H_diff, 180.0 - H_diff)
        bucket = np.argmin(H_diff_circ, axis=-1).astype(np.int32)

        fg_bins = np.unique(bucket[fg_pixels])
        for b in fg_bins:
            m = ((bucket == b) & fg_pixels).astype(np.float32)
            if m.sum() > _MIN_MASK_PIXELS:
                color = (rgb * m[..., None]).sum(axis=(0, 1)) / (m.sum() + 1e-8)
                tokens.append((color.astype(np.float32), m))

    # ── 兜底:完全没前景或没背景,给个全零 token 防 K=0 ──
    if not tokens:
        tokens.append((np.zeros(3, np.float32), np.zeros((h, w), np.float32)))

    bucket_full = bucket if fg_pixels.sum() > _MIN_MASK_PIXELS else np.full((h, w), -1, dtype=np.int32)
    bucket_full[bg_mask] = -1
    return tokens, bucket_full, bg_mask


class _MaskEncoder(nn.Module):
    """单 mask → 32-d L2 归一化几何 token(类比 word embedding)。"""

    def __init__(self, geom_dim: int = TOKEN_GEOM_DIM):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.fc = nn.Linear(32, geom_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, H, W) ∈ [0, 1] → (B, geom_dim) L2-normalized"""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.mean(dim=(2, 3))          # Global Avg Pool → (B, 32)
        x = self.fc(x)                  # (B, geom_dim)
        return F.normalize(x, dim=-1)   # L2 norm → 单位球面


class ColorTokensAlgo(BaseAlgorithm):
    name = "color_tokens"
    uses_rgb = True     # 意图;内部兼容 mask 输入(自动 gray2rgb)

    def __init__(self):
        # 延迟构造,首次 encode() 时初始化(避免 import 时 CUDA 初始化)
        self.encoder: _MaskEncoder | None = None
        self.agg: nn.Linear | None = None
        self.clf: nn.Linear | None = None     # 训练时用的分类头,推理不用
        self._device: torch.device | None = None

    def _build(self, device: torch.device):
        torch.manual_seed(42)                                  # 全局 seed,先于 encoder
        self.encoder = _MaskEncoder(TOKEN_GEOM_DIM).to(device)
        torch.manual_seed(42)                                  # 同样 seed 给聚合层
        self.agg = nn.Linear(AGG_IN_DIM, EMBEDDING_DIM).to(device)
        # 聚合层权重初始化(std=0.1 让初始 embedding 数量级合理,L2 norm 后稳定)
        with torch.no_grad():
            self.agg.weight.normal_(mean=0.0, std=0.1)
            self.agg.bias.zero_()
        self._device = device

    @staticmethod
    def _to_rgb(inputs: np.ndarray) -> np.ndarray:
        """(N, H, W) mask → (N, H, W, 3) 灰度复制,或直通 (N, H, W, 3) RGB。"""
        if inputs.ndim == 3:
            return np.repeat(inputs[..., None], 3, axis=-1)
        return inputs

    def _encode_one_train(self, rgb_i: np.ndarray) -> torch.Tensor:
        """单个样本前向(训练模式,带梯度)→ 返回 L2 归一化后 (EMBEDDING_DIM,) 向量。"""
        tokens, _, _ = _hsv_decompose(rgb_i)
        masks = np.stack([m for _, m in tokens])
        colors = np.stack([c for c, _ in tokens])

        masks_t = torch.from_numpy(masks).unsqueeze(1).to(self._device)
        colors_t = torch.from_numpy(colors).to(self._device)

        geom = self.encoder(masks_t)                                 # (K, 32) L2 normalized
        full = torch.cat([geom, colors_t], dim=-1)                   # (K, 35)
        mean = full.mean(dim=0)
        mx   = full.max(dim=0).values
        mn   = full.min(dim=0).values
        k_feat = torch.tensor([float(len(tokens))], device=self._device)
        agg_in = torch.cat([mean, mx, mn, k_feat])                   # (106,)
        emb = self.agg(agg_in)                                       # (128,)
        return F.normalize(emb, dim=-1)                              # L2 norm(可导)

    @torch.no_grad()
    def encode_tokens(self, inputs: np.ndarray) -> tuple:
        """每个 token 单独的几何向量(聚合前),用于 token-level 可视化。

        Returns:
            tokens_flat: (N*K, 32)  所有 token 拼起来的几何向量,L2 归一化
            sample_id:   (N*K,)     每个 token 属于哪个样本(子集索引)
            is_fg:       (N*K,)     是不是前景 token(0=背景,1=前景;按 HSV 分解顺序)
        """
        rgb_batch = self._to_rgb(inputs)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.encoder is None:
            self._build(device)
        self.encoder.eval(); self.agg.eval()

        all_vecs = []
        all_sample_id = []
        all_is_fg = []
        for sample_i in range(len(rgb_batch)):
            tokens, _, _ = _hsv_decompose(rgb_batch[sample_i])
            masks = np.stack([m for _, m in tokens])
            masks_t = torch.from_numpy(masks).unsqueeze(1).to(device)
            geom = self.encoder(masks_t)                              # (K, 32) L2 norm
            all_vecs.append(geom.cpu().numpy())
            # 第一个 token 是背景(如果有 bg_mask),后面是前景
            # 但 _hsv_decompose 的顺序:先 bg(若有),再 fg 按 bucket
            # 简化:第 0 个 token 当背景,其余前景
            n = len(tokens)
            is_fg = np.zeros(n, dtype=np.int32)
            if n > 1:
                is_fg[1:] = 1
            elif n == 1:
                # 单 token:看 mask 平均值,前景=mask 平均 > 0.1
                if tokens[0][1].mean() > 0.1:
                    is_fg[0] = 1
            all_is_fg.append(is_fg)
            all_sample_id.append(np.full(n, sample_i, dtype=np.int32))

        return (np.concatenate(all_vecs).astype(np.float32),
                np.concatenate(all_sample_id),
                np.concatenate(all_is_fg))

    @torch.no_grad()
    def encode(self, inputs: np.ndarray, verbose: bool = True) -> np.ndarray:
        """推理:输入 mask/RGB → 输出 (N, 128) L2 归一化 embedding。"""
        rgb_batch = self._to_rgb(inputs)                   # (N, H, W, 3)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.encoder is None:
            self._build(device)
        self.encoder.eval(); self.agg.eval()

        N = rgb_batch.shape[0]
        out = np.zeros((N, EMBEDDING_DIM), dtype=np.float32)

        for i in range(N):
            emb = self._encode_one_train(rgb_batch[i])     # (128,) 已 L2 归一化
            out[i] = emb.cpu().numpy()

        return out.astype(np.float32)

    def fit(self, inputs: np.ndarray, labels: dict,
            n_epochs: int = 20, lr: float = 1e-3,
            tasks: tuple = ("label",),
            task_weights: dict | None = None,
            verbose: bool = True) -> dict:
        """训练 CNN + agg + 多任务分类头。

        支持同时训练多个属性分类任务,loss 加权求和。例:
          tasks=("label",)                       → 单任务(字母)
          tasks=("label","position","rotation","size") → 多任务(全属性)

        Args:
            inputs: (N, H, W) mask 或 (N, H, W, 3) RGB
            labels: Dataset 风格 {field: {value: [indices]}}
                每个 task 必须是 labels 里存在的 field
            n_epochs: 训练轮数
            lr: Adam 学习率
            tasks: 要训练的字段名 tuple
            task_weights: dict {field: weight},默认每个 task 权重 1.0
            verbose: 是否打印训练日志

        Returns:
            {"final_loss": float, "final_acc": {task: float},
             "loss_curve": list[float], "tasks": tuple}
        """
        rgb_batch = self._to_rgb(inputs)                   # (N, H, W, 3)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.encoder is None:
            self._build(device)

        # ── 校验所有 task 都在 labels 里,并构造 per-sample 整数标签 ──
        task_weights = task_weights or {}
        self.clfs = nn.ModuleDict()                         # 多个分类头
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
            self.clfs[task] = nn.Linear(EMBEDDING_DIM, len(classes)).to(device)

        if verbose:
            print(f"  fit: 多任务 {tasks}, {len(inputs)} 张样本, n_epochs={n_epochs}")
            for task in tasks:
                print(f"    [{task}] {len(labels[task])} 类")

        # ── 优化器 ──
        params = (list(self.encoder.parameters()) +
                  list(self.agg.parameters()) +
                  list(self.clfs.parameters()))
        opt = torch.optim.Adam(params, lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        self.encoder.train(); self.agg.train(); self.clfs.train()

        N = len(inputs)
        loss_curve = []
        for epoch in range(n_epochs):
            perm = np.random.permutation(N)
            total_loss = 0.0
            correct = {t: 0 for t in tasks}
            for i in perm:
                emb = self._encode_one_train(rgb_batch[i])              # (128,)

                opt.zero_grad()
                loss = torch.tensor(0.0, device=device)
                for task in tasks:
                    logits = self.clfs[task](emb)                       # (n_classes,)
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

        self.encoder.eval(); self.agg.eval(); self.clfs.eval()
        return {"final_loss": loss_curve[-1], "final_acc": acc,
                "loss_curve": loss_curve, "tasks": tasks}