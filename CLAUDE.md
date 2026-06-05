# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目本质

实验驱动的探索项目。核心问题：**一个 embedding 空间能不能同时表达 mask 的形状、大小、位置、颜色等多种属性？** 目前未知，需要大量实验来验证。

## 常用命令

```bash
uv sync                                        # 安装依赖
uv run python run.py --list                     # 列出所有实验
uv run python run.py                            # 跑全部文字实验 (默认 gabor_lift)
uv run python run.py --exp color                # 跑单个实验
uv run python run.py --category vision          # 跑视觉实验
uv run python run.py --algo random_proj         # 换算法
uv run python run.py --no-viz                   # 不生成可视化
uv run pytest tests/ -v                         # 运行测试
uv run pytest tests/test_text_experiments.py::TestRunDataFlow::test_text_run_returns_dict[lang] -v  # 跑单个测试
```

## 架构

两条扩展轴，一个 pipeline：

```
数据 (experiments/source/)  →  算法 (algorithms/)  →  实验 (experiments/)  →  可视化
     load_data()                encode()               run() + viz()
```

### 算法层 (algorithms/)

- 继承 `BaseAlgorithm`，实现 `encode(inputs) → (N, EMBEDDING_DIM)` L2 归一化向量
- `uses_rgb` 标记算法需要 RGB 还是只用二值 mask
- 注册到 `algorithms/__init__.py` 的 `ALGOS` 字典
- 当前算法：`gabor_lift`（Gabor 滤波器+PCA）、`random_proj`（随机投影）
- 新算法从 `algorithms/template.py` 复制开始

### 实验层 (experiments/)

- 继承 `BaseExperiment`，实现五要素：`load_data()`、`hypothesis`、`run()`、`passes_when`、`viz()`
- `run(emb, labels, sim) → dict` 是核心度量方法，`sim = emb @ emb.T`
- 返回 dict 必须含 `name`、`metric`、`separation`、`is_correct`、`details`
- 注册到 `experiments/text/__init__.py` 或 `experiments/vision/__init__.py`
- 新实验从复制现有实验开始（如 `text/color.py`）

### labels 结构

所有实验的 labels 统一格式：`{field_name: {value: [sample_indices]}}`
- 文字实验字段：`lang`、`color`、`position`、`label`
- 视觉实验字段：`x_pos`、`y_pos`、`count`、`single`

### 关键常量

- `EMBEDDING_DIM = 128`（`algorithms/base.py`）— 算法和实验都依赖
- `IMG_SIZE = 64`、`SEED = 42`（`experiments/source/synthetic/config.py`）
- 输出目录：`results/`（由 `OUTPUT_ROOT` 控制）

## 添加新算法

1. 复制 `algorithms/template.py` → `algorithms/your_algo.py`
2. 实现 `encode()`，输出 `(N, EMBEDDING_DIM)` L2 归一化向量
3. 在 `algorithms/__init__.py` 的 `ALGOS` 中注册

## 添加新实验

1. 在 `experiments/text/` 或 `experiments/vision/` 下新建 `.py`
2. 继承 `BaseExperiment`，实现 `load_data()`、`run()`、`viz()`
3. 在对应目录的 `__init__.py` 中注册到 `TEXT_EXPERIMENTS` 或 `VISION_EXPERIMENTS`

## 测试

- `tests/test_text_experiments.py` — 用随机 embedding 验证数据流通性
- 测试不依赖真实算法，用 `make_random_emb` fixture 生成随机 L2 归一化向量
- 三类测试：`TestRunDataFlow`（run 返回结构）、`TestVizDataFlow`（viz 不崩溃）、`TestLoadDataStructure`（数据结构正确）

## 注意事项

- 项目用 `uv` 管理依赖，不用 pip
- matplotlib 用 `Agg` backend（`experiments/viz.py`），无 GUI
- 中文字体依赖 Windows 的 simhei/msyh（`experiments/viz.py` 和 `source/synthetic/data.py`）
- 合成数据生成需要 Windows 字体文件（`C:/Windows/Fonts/`）
