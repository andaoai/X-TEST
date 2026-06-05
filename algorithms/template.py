"""
============================================================
新算法模板 —— 复制 → 改名 → 写 encode()
============================================================
1. cp algorithms/template.py algorithms/myidea.py
2. 改 class 名和 name
3. 实现 encode(inputs) → (N, 128)  L2 归一化后的 embedding
4. 在 algorithms/__init__.py 里 import 并加入 ALGOS
5. uv run python run.py --algo myidea
============================================================
"""
import numpy as np
from algorithms.base import BaseAlgorithm, EMBEDDING_DIM


class MyAlgorithm(BaseAlgorithm):
    name = "my_algorithm"   # 命令行 --algo 用这个名字
    uses_rgb = False        # True=输入RGB, False=只输入二值mask

    def encode(self, inputs, verbose=True):
        """
        inputs: (N, H, W) mask 或 (N, H, W, 3) RGB
        returns: (N, EMBEDDING_DIM) L2 归一化后的向量
        """
        N, H, W = inputs.shape[:3]
        flat = inputs.reshape(N, -1)

        # ════════════════════════════════════════
        # TODO: 在这里写你的编码逻辑
        # ════════════════════════════════════════

        rng = np.random.RandomState(42)
        proj = rng.randn(flat.shape[1], EMBEDDING_DIM) / np.sqrt(EMBEDDING_DIM)
        emb = flat @ proj

        # ════════════════════════════════════════

        return emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
