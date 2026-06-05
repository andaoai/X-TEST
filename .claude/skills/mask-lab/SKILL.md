---
name: mask-lab
description: 二值 mask -> 128维 embedding 实验平台。每天换算法，自动运行实验、输出对比报告。用于验证位置不变性、颜色可分性、旋转可分性、字母/中文分辨力等。
user-invocable: true
allowed-tools: Read Write Edit Bash Grep Glob
---

# Mask Embedding Lab

Mask → 128维 embedding 的实验平台。每天可以换算法，自动跑实验看结果。

> 数据集管理用 `/dataset-manager`
> 新建实验用 `/new-experiment`

## 核心概念: 每个实验包含完整五要素

一个实验 = **数据** + **假设** + **度量** + **判据** + **可视化**

每个实验文件 (experiments/text/xxx.py) 继承 BaseExperiment:
```python
class ExpXXX(BaseExperiment):
    name         = "..."         # 实验名称
    hypothesis   = "..."         # 验证什么假设
    passes_when  = "..."         # 通过条件
    uses_rgb     = True          # True=需要 RGB, False=只需 mask

    def load_data(self): ...     # 返回 (inputs, labels)
    def run(self, emb, labels, sim): ...   # 执行度量, 返回 dict
    def viz(self, emb, labels, result, algo_name, out_dir): ...   # 可视化
```

## 项目结构

```
algorithms/                           ← 算法: mask → 128维 embedding
├── base.py                           #   EMBEDDING_DIM = 128
├── gabor.py                          #   GaborLift (固定核 FFT)
├── random_proj.py                    #   随机投影
├── template.py                       #   新算法模板
└── __init__.py                       #   ALGOS 注册表

experiments/                          ← 实验
├── base.py                           #   BaseExperiment 基类
├── viz.py                            #   通用可视化 (tsne_plot, summary_bar)
├── source/                           #   数据源
│   ├── synthetic/config.py           #     合成数据定义 (字符/颜色/位置/旋转)
│   ├── synthetic/data.py             #     Dataset 生成器
│   └── real/                         #     真实数据集
├── text/                             #   文字实验 (每文件=五要素)
│   ├── __init__.py                   #     TEXT_EXPERIMENTS 注册表
│   ├── lang.py                       #     中英文可分
│   ├── position.py                   #     位置不变
│   ├── color.py                      #     颜色可分
│   ├── pos_encode.py                 #     位置编码
│   ├── char.py                       #     字符分离
│   ├── rotation_sep.py               #     旋转可分
│   └── rotation_inv.py               #     旋转不变
└── vision/                           #   视觉实验
    ├── __init__.py                   #     VISION_EXPERIMENTS 注册表
    └── pixel.py                      #     像素可分

results/                              ← 输出
run.py                                ← CLI 入口
```

## 实验列表

`uv run python run.py --list` 查看最新列表。

### 文字实验

| key | 文件 | 假设 | 判据 |
|------|------|------|------|
| lang | text/lang.py | 中英文两种书写系统，应能区分 | sep > 0.05 |
| position | text/position.py | 同字符在不同位置，应高度相似 | sim > 0.90 |
| color | text/color.py | 不同颜色在不同通道签名不同 | sep > 0.05 |
| pos_encode | text/pos_encode.py | 同位置有微量相似可编码 | sep > 0.02 |
| char | text/char.py | 不同字符形状不同，应能区分 | sep > 0.05 |
| rotation_sep | text/rotation_sep.py | 不同旋转角度应能区分 | sep > 0.05 |
| rotation_inv | text/rotation_inv.py | 同字符不同旋转应相似 | sim > 0.90 |

### 视觉实验

| key | 文件 | 假设 | 判据 |
|------|------|------|------|
| pixel | vision/pixel.py | 单/多像素位置和数量可编码 | pos>0.05 或 count>0.03 |

## 运行

```bash
uv run python run.py                          # 默认跑全部实验
uv run python run.py --exp rotation_sep       # 单实验
uv run python run.py --list                   # 列出所有实验
uv run python run.py --category vision        # 跑全部视觉实验
uv run python run.py --algo random_proj       # 换算法
uv run python run.py --no-viz                 # 跳过图片
uv run pytest tests/ -v                       # 跑测试
```

## 添加新算法

1. `cp algorithms/template.py algorithms/myidea.py`
2. 改 class 名、`name`、实现 `encode(inputs) → (N, 128)` L2 归一化
3. 在 `algorithms/__init__.py` import + ALGOS
4. `uv run python run.py --algo myidea`

## 添加新实验

> 详细流程用 `/new-experiment` 引导。

1. 在 `experiments/text/` 或 `experiments/vision/` 下新建 `.py`
2. 继承 `BaseExperiment`，实现 `load_data()`、`run()`、`viz()`
3. 在对应目录的 `__init__.py` 注册到 `TEXT_EXPERIMENTS` 或 `VISION_EXPERIMENTS`
4. `uv run pytest tests/ -v` 确保通过
5. `uv run python run.py --exp <key>` 跑实验
