# x-test

**Mask Embedding Lab** — 一个实验驱动的探索项目。

## 核心问题

一个 mask（二值图像，只有 0/1）包含很多属性：**形状、大小、位置、颜色……**

**问题是：一个 embedding 空间能不能同时表达这些属性？还是必须分开编码？**

答案是未知的。所以我们做实验——用不同的算法把 mask 编码成向量（128 或 512 维），然后从多个角度检验这个向量的质量：

- 字符形状能不能区分？（字符可分性）
- 颜色能不能区分？（颜色可分性）
- 位置信息有没有保留？（位置可编码 / 位置不变性）
- 中英文能不能分开？（语言可分性）
- ……

每个实验只检验一个属性。通过这些实验，我们可以判断一个算法是只能表达单一属性，还是能在一个 embedding 空间里同时表达多种属性。这是目前最大的未知数。

## 项目结构

```
x-test/
├── run.py                    # 主入口 — 数据 → 编码 → 实验 → 可视化
├── algorithms/               # 算法实现
│   ├── base.py               # 算法基类 (encode 接口)
│   ├── gabor.py              # Gabor 滤波器 + PCA
│   └── template.py           # 算法模板 (随机投影)
├── experiments/              # 实验定义
│   ├── base.py               # 实验基类 (五要素接口)
│   ├── viz.py                # 可视化工具
│   ├── text/                 # 文字实验 ×5
│   │   ├── lang.py           # 中英文可分性
│   │   ├── position.py       # 位置不变性
│   │   ├── color.py          # 颜色可分性
│   │   ├── pos_encode.py     # 位置可编码
│   │   └── char.py           # 字符可分性
│   ├── vision/               # 视觉实验 ×1
│   │   └── pixel.py          # 像素可分性
│   └── source/               # 数据源
│       └── synthetic/        # 合成数据生成
└── tests/                    # 测试
    └── test_text_experiments.py  # 随机 embedding 数据流通性测试
```

## 实验列表

每个实验都遵循统一的五要素：**数据从哪来 → 假设是什么 → 怎么度量 → 怎样算对 → 如何可视化**

### 文字实验

| 实验 | 文件 | 假设 | 通过条件 |
|------|------|------|----------|
| 中英文可分性 | `text/lang.py` | embedding 应能区分中英文 | separation > 0.05 |
| 位置不变性 | `text/position.py` | 同一字符不同位置 embedding 相似 | similarity > 0.90 |
| 颜色可分性 | `text/color.py` | 不同颜色 embedding 可区分 | separation > 0.05 |
| 位置可编码 | `text/pos_encode.py` | 同位置样本 embedding 有弱相似性 | separation > 0.02 |
| 字符可分性 | `text/char.py` | 不同字符 embedding 可区分 | separation > 0.05 |

### 视觉实验

| 实验 | 文件 | 假设 | 通过条件 |
|------|------|------|----------|
| 像素可分性 | `vision/pixel.py` | 单像素位置 embedding 可区分 | pos_sep > 0.05 或 cnt_sep > 0.03 |

## 快速开始

```bash
# 安装依赖
uv sync

# 列出所有实验
uv run python run.py --list

# 跑全部文字实验 (默认 gabor_lift 算法)
uv run python run.py

# 跑单个实验
uv run python run.py --exp color

# 跑视觉实验
uv run python run.py --category vision

# 用不同算法
uv run python run.py --algo random_proj

# 运行测试
uv run pytest tests/ -v
```

## 添加新算法

1. 复制 `algorithms/template.py` → `algorithms/your_algo.py`
2. 实现 `encode(inputs) → (N, EMBEDDING_DIM)` 方法
3. 在 `algorithms/__init__.py` 中注册

```python
class YourAlgorithm(BaseAlgorithm):
    name = "your_algo"
    uses_rgb = False  # True 如果需要 RGB 输入

    def encode(self, inputs, verbose=True):
        # inputs: (N, H, W) mask
        # return: (N, 128) L2 归一化 embedding
        ...
```

## 添加新实验

1. 在 `experiments/text/` 或 `experiments/vision/` 下新建 `.py` 文件
2. 继承 `BaseExperiment`，实现 `load_data()`、`run()`、`viz()`
3. 在对应目录的 `__init__.py` 中注册

## 当前状态

- 算法：gabor_lift（Gabor 滤波器）、random_proj（随机投影）
- 实验：6 个（5 文字 + 1 视觉），全部可运行
- 测试：24 个数据流通性测试，全部通过
