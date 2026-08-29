"""
color_tokens 算法契约测试 —— 不调 ds.generate(),直接用 numpy 构造 mask。

验证:
  - 注册到 ALGOS 且 key == algo.name
  - 输出 shape = (N, EMBEDDING_DIM)
  - L2 归一化(每行 ||x||₂ ≈ 1)
  - 确定性(seed=42, 两次 encode 同一输入必须 bit-equal)
  - mask 模式 (N,H,W) 和 RGB 模式 (N,H,W,3) 都走通
  - 颜色拼接生效:RGB 输入应与 mask 输入产生不同 embedding
  - 端到端:相似度矩阵对角线 ≈ 1
"""
import numpy as np
import pytest

from algorithms import ALGOS
from algorithms.base import EMBEDDING_DIM


@pytest.fixture
def algo():
    return ALGOS["color_tokens"]


# ── 注册检查 ──

def test_registered_in_algos():
    assert "color_tokens" in ALGOS, "color_tokens 必须注册到 ALGOS"
    assert ALGOS["color_tokens"].name == "color_tokens", "key 必须等于 algo.name"


def test_uses_rgb_flag():
    assert ALGOS["color_tokens"].uses_rgb is True, "color_tokens 标记为 uses_rgb=True(内部兼容 mask)"


# ── 输出 shape + L2 norm ──

def test_mask_input_shape_and_norm(algo):
    inputs = np.zeros((4, 64, 64), dtype=np.float32)
    inputs[:, 8:56, 8:56] = 1.0       # 一个实心方块
    emb = algo.encode(inputs)
    assert emb.shape == (4, EMBEDDING_DIM), f"shape 应为 (4, {EMBEDDING_DIM}),实际 {emb.shape}"
    norms = np.linalg.norm(emb, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4), f"每行 L2 norm 应 ≈ 1,实际 {norms}"


def test_rgb_input_shape_and_norm(algo):
    inputs = np.random.rand(4, 64, 64, 3).astype(np.float32)
    inputs = (inputs > 0.5).astype(np.float32)    # 二值化(模拟颜色边界)
    emb = algo.encode(inputs)
    assert emb.shape == (4, EMBEDDING_DIM)
    assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-4)


# ── 确定性 ──

def test_determinism(algo):
    rng = np.random.RandomState(0)
    inputs = (rng.rand(8, 64, 64) > 0.7).astype(np.float32)
    e1 = algo.encode(inputs)
    e2 = algo.encode(inputs)
    assert np.allclose(e1, e2), "seed=42 必须可复现(两次 encode 应 bit-equal)"


# ── 颜色影响 embedding ──

def test_color_affects_embedding(algo):
    """同样形状的 mask + 不同颜色 → embedding 应该不同(颜色拼接到 token,理论上有影响)。

    注意:mask 灰度复制时 R=G=B,HSV 中 S=0 → H=0 → 全部归 Red 桶。
    测试用真实 RGB 输入绕过这个退化,直接验证颜色拼接生效。
    """
    H, W = 64, 64
    red = np.zeros((H, W, 3), dtype=np.float32)
    red[8:56, 8:56, 0] = 1.0          # 纯红块
    green = np.zeros((H, W, 3), dtype=np.float32)
    green[8:56, 8:56, 1] = 1.0        # 纯绿块
    rgb = np.stack([red, green], axis=0)   # (2, H, W, 3)

    emb = algo.encode(rgb)
    diff = np.linalg.norm(emb[0] - emb[1])
    assert diff > 1e-3, f"红绿 embedding 应不同,但 diff={diff}"


def test_mask_vs_rgb_produce_different_embedding(algo):
    """mask 模式 vs RGB 模式 → embedding 应不同(mask 颜色退化 ≠ RGB 真实颜色)。"""
    H, W = 64, 64
    mask = np.zeros((1, H, W), dtype=np.float32)
    mask[0, 8:56, 8:56] = 1.0
    rgb = np.zeros((1, H, W, 3), dtype=np.float32)
    rgb[0, 8:56, 8:56, 0] = 1.0   # 红色

    emb_mask = algo.encode(mask)
    emb_rgb = algo.encode(rgb)
    diff = np.linalg.norm(emb_mask[0] - emb_rgb[0])
    assert diff > 1e-3, f"mask 模式 vs RGB 模式 embedding 应不同,但 diff={diff}"


# ── 端到端 ──

def test_via_pipeline_similarity(algo):
    """端到端:encode → sim = emb @ emb.T → 对角线 ≈ 1。"""
    rng = np.random.RandomState(0)
    inputs = (rng.rand(20, 64, 64) > 0.8).astype(np.float32)
    emb = algo.encode(inputs)
    sim = emb @ emb.T
    assert sim.shape == (20, 20)
    assert np.allclose(np.diag(sim), 1.0, atol=1e-4), "L2 归一化后,自相似度应 ≈ 1"


def test_no_nan_no_inf(algo):
    """无 NaN / Inf。"""
    rng = np.random.RandomState(1)
    inputs = (rng.rand(16, 64, 64) > 0.8).astype(np.float32)
    emb = algo.encode(inputs)
    assert np.all(np.isfinite(emb)), "embedding 不应有 NaN/Inf"


def test_empty_input_handled(algo):
    """输入全 0(全背景)→ 应有兜底 token,embedding 不应为 NaN/Inf。"""
    inputs = np.zeros((4, 64, 64), dtype=np.float32)
    emb = algo.encode(inputs)
    assert emb.shape == (4, EMBEDDING_DIM)
    assert np.all(np.isfinite(emb))


def test_real_synthetic_data_smoke(algo):
    """小规模真实合成数据(40 张)→ encode() 不崩,embedding 形状 + norm 正确。

    不跑 ds.generate() 9600 张全集(慢),只跑小子集验证集成。
    """
    from experiments.source.synthetic.data import Dataset
    # Dataset.__init__ 要求 colors 是 dict{name: (R,G,B)} 格式
    ds = Dataset(letters=["A", "B"], colors={"Red": (255, 0, 0), "Blue": (0, 0, 255)}).generate(verbose=False)
    masks = ds.masks()                  # (1440, 64, 64) — 12 chars × 2 色 × 5 位 × 4 转 × 3 大小
    assert masks.shape[0] > 0 and masks.shape[1:] == (64, 64)
    emb = algo.encode(masks)
    assert emb.shape == (masks.shape[0], EMBEDDING_DIM)
    assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-4)