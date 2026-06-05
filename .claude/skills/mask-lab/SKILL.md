---
name: mask-lab
description: 二值 mask -> 128维 embedding 实验平台。每天换算法，自动运行5个实验、输出对比报告。用于验证位置不变性、颜色可分性、字母/中文分辨力。
user-invocable: true
allowed-tools: Read Write Edit Bash Grep Glob
---

# Mask Embedding Lab

Mask → 128维 embedding 的实验平台。每天可以换算法，自动跑 5 个实验看结果。

> 数据集管理用 `/dataset-manager`

## 核心概念: 每个实验包含完整四要素

一个实验 = **数据从哪来** + **怎么度量** + **怎样算对** + **如何可视化**

每个实验文件 (experiments/text/xxx.py) 继承 BaseExperiment:
```python
class ExpXXX(BaseExperiment):
    name         = "..."         # 实验名称
    hypothesis   = "..."         # 验证什么假设
    data_source  = "..."         # 数据从哪来
    what_labels  = [...]        # 需要哪些标签
    passes_when  = "..."         # 通过条件

    def run(self, emb, labels, sim): ...   # 执行度量
    def viz(self, emb, labels, result, ...): ...   # 可视化
```

## 项目结构

```
algorithms/                           ← 算法: mask → 128维 embedding
├── base.py
├── gabor.py                          #   GaborLift (固定核 FFT)
├── template.py                       #   新算法模板
└── __init__.py                        #   ALGOS 注册表

experiments/                          ← 实验
├── base.py                           #   BaseExperiment 基类 (四要素)
├── viz.py                            #   通用可视化
├── source/                           #   数据源
│   ├── manage.py                     #     数据集管理 (/dataset-manager)
│   ├── synthetic/config.py           #     合成数据定义
│   ├── synthetic/data.py             #     Dataset 生成器
│   └── real/                         #     真实数据集
└── text/                             #   5 个实验 (每文件=四要素)
    ├── __init__.py
    ├── lang.py                       #     Exp1: 中英文可分
    ├── position.py                   #     Exp2: 位置不变
    ├── color.py                      #     Exp3: 颜色可分
    ├── pos_encode.py                 #     Exp4: 位置编码
    └── char.py                       #     Exp5: 字符分离

results/                              ← 输出
run.py                                ← CLI 入口
```

## 5 个实验

| 实验 | 文件 | 假设 |
|------|------|------|
| Exp1 中英文 | text/lang.py | 中英文两种书写系统，embedding 应能区分 |
| Exp2 位置不变 | text/position.py | 同字符在不同位置，embedding 应高度相似 |
| Exp3 颜色可分 | text/color.py | 不同颜色在物理通道上有不同签名，应能区分 |
| Exp4 位置编码 | text/pos_encode.py | 同位置样本间有微量相似性，可弱编码位置 |
| Exp5 字符可分 | text/char.py | 不同字符有不同形状结构，应能区分 |

## 运行

```bash
uv run python run.py                          # 全部实验
uv run python run.py --algo gabor_lift        # 选算法
uv run python run.py --exp exp3               # 单实验
uv run python run.py --letters A,B,C          # 自定义字母
uv run python run.py --colors Red,Blue        # 自定义颜色
uv run python run.py --no-viz                 # 跳过图片
```

## 添加新算法

1. `cp algorithms/template.py algorithms/myidea.py`
2. 改 class 名、`name`、实现 `encode(inputs) → (N, 128)`
3. 在 `algorithms/__init__.py` import + ALGOS
4. `uv run python run.py --algo myidea`

## 添加新实验

1. `cp experiments/text/lang.py experiments/text/my_exp.py`
2. 声明四要素: name, hypothesis, data_source, passes_when
3. 实现 `run()` + `viz()`
4. 在 `experiments/text/__init__.py` 注册
