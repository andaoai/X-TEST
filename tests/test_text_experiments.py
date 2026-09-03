"""
数据流通性测试 —— 用随机 embedding 验证每个实验的 run()/viz() 能跑通。

不依赖真实算法和真实数据，只检查:
  1. load_data() 能正常返回 (inputs, labels)
  2. run(emb, labels, sim) 返回正确结构的 dict
  3. viz() 不抛异常
"""
import pytest
import numpy as np
from pathlib import Path

from algorithms.base import EMBEDDING_DIM
from experiments.text import TEXT_EXPERIMENTS
from experiments.vision import VISION_EXPERIMENTS

ALL_EXPERIMENTS = {**TEXT_EXPERIMENTS, **VISION_EXPERIMENTS}


# ── fixtures ──

@pytest.fixture
def make_random_emb():
    """工厂 fixture: make_random_emb(n) → (n, EMBEDDING_DIM) 随机 L2 归一化 embedding"""
    def _make(n, seed=42):
        rng = np.random.RandomState(seed)
        emb = rng.randn(n, EMBEDDING_DIM).astype(np.float32)
        return emb / np.linalg.norm(emb, axis=1, keepdims=True)
    return _make


# ── 测试 run() 数据流通性 ──

class TestRunDataFlow:
    """每个实验的 run() 用随机 embedding 跑一遍，检查返回结构"""

    @pytest.fixture(params=list(TEXT_EXPERIMENTS.keys()), ids=list(TEXT_EXPERIMENTS.keys()))
    def text_exp(self, request):
        return request.param, TEXT_EXPERIMENTS[request.param]

    @pytest.fixture(params=list(VISION_EXPERIMENTS.keys()), ids=list(VISION_EXPERIMENTS.keys()))
    def vision_exp(self, request):
        return request.param, VISION_EXPERIMENTS[request.param]

    def test_text_run_returns_dict(self, text_exp, make_random_emb):
        key, exp = text_exp
        inputs, labels = exp.load_data()
        n = inputs.shape[0]
        emb = make_random_emb(n)
        sim = emb @ emb.T

        result = exp.run(emb, labels, sim)

        assert isinstance(result, dict), f"{key}: run() 应返回 dict"
        assert "name" in result, f"{key}: 缺少 'name'"
        assert "is_correct" in result, f"{key}: 缺少 'is_correct'"
        assert "metric" in result, f"{key}: 缺少 'metric'"
        assert isinstance(result["is_correct"], (bool, np.bool_)), \
            f"{key}: is_correct 应为 bool, 实际 {type(result['is_correct'])}"

    def test_vision_run_returns_dict(self, vision_exp, make_random_emb):
        key, exp = vision_exp
        inputs, labels = exp.load_data()
        n = inputs.shape[0]
        emb = make_random_emb(n)
        sim = emb @ emb.T

        result = exp.run(emb, labels, sim)

        assert isinstance(result, dict), f"{key}: run() 应返回 dict"
        assert "name" in result, f"{key}: 缺少 'name'"
        assert "is_correct" in result, f"{key}: 缺少 'is_correct'"
        assert "metric" in result, f"{key}: 缺少 'metric'"
        assert isinstance(result["is_correct"], (bool, np.bool_)), \
            f"{key}: is_correct 应为 bool, 实际 {type(result['is_correct'])}"


# ── 测试 viz() 不崩溃 ──

class TestVizDataFlow:
    """每个实验的 viz() 用随机 embedding 跑一遍，确保不抛异常"""

    @pytest.fixture(params=list(TEXT_EXPERIMENTS.keys()), ids=list(TEXT_EXPERIMENTS.keys()))
    def text_exp(self, request):
        return request.param, TEXT_EXPERIMENTS[request.param]

    @pytest.fixture(params=list(VISION_EXPERIMENTS.keys()), ids=list(VISION_EXPERIMENTS.keys()))
    def vision_exp(self, request):
        return request.param, VISION_EXPERIMENTS[request.param]

    def test_text_viz_no_crash(self, text_exp, tmp_path, make_random_emb):
        key, exp = text_exp
        inputs, labels = exp.load_data()
        n = inputs.shape[0]
        emb = make_random_emb(n)
        sim = emb @ emb.T
        result = exp.run(emb, labels, sim)

        # 不应抛异常
        exp.viz(emb, labels, result, "test_algo", tmp_path)

    def test_vision_viz_no_crash(self, vision_exp, tmp_path, make_random_emb):
        key, exp = vision_exp
        inputs, labels = exp.load_data()
        n = inputs.shape[0]
        emb = make_random_emb(n)
        sim = emb @ emb.T
        result = exp.run(emb, labels, sim)

        exp.viz(emb, labels, result, "test_algo", tmp_path)


# ── 测试 load_data() 返回结构 ──

class TestLoadDataStructure:
    """验证每个实验的 load_data() 返回正确的数据结构"""

    @pytest.mark.parametrize("key", list(ALL_EXPERIMENTS.keys()))
    def test_load_data_returns_tuple(self, key):
        exp = ALL_EXPERIMENTS[key]
        inputs, labels = exp.load_data()

        assert isinstance(inputs, np.ndarray), f"{key}: inputs 应为 ndarray"
        assert isinstance(labels, dict), f"{key}: labels 应为 dict"
        assert inputs.ndim in (3, 4), f"{key}: inputs 应为 (N,H,W) 或 (N,H,W,3)"

    @pytest.mark.parametrize("key", list(TEXT_EXPERIMENTS.keys()))
    def test_text_labels_have_required_fields(self, key):
        exp = TEXT_EXPERIMENTS[key]
        _, labels = exp.load_data()

        for field in ["lang", "color", "position", "label"]:
            assert field in labels, f"{key}: labels 缺少 '{field}'"
            assert isinstance(labels[field], dict), f"{key}: labels['{field}'] 应为 dict"

    @pytest.mark.parametrize("key", list(VISION_EXPERIMENTS.keys()))
    def test_vision_labels_structure(self, key):
        exp = VISION_EXPERIMENTS[key]
        _, labels = exp.load_data()

        assert isinstance(labels, dict) and len(labels) >= 2, f"{key}: labels 至少 2 个字段"
        dict_fields = 0
        for field, values in labels.items():
            if not isinstance(values, dict):
                continue  # 部分数据集带扁平索引元数据(如 pixel 的 "single" 列表),跳过
            dict_fields += 1
            assert len(values) >= 2, f"{key}: 字段 '{field}' 至少 2 个取值"
        assert dict_fields >= 2, f"{key}: 至少 2 个 field:value dict 字段"
