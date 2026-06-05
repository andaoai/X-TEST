---
name: new-experiment
description: 引导用户一步步构建新实验。通过问答方式帮助声明四要素（数据来源、假设、度量、可视化），自动生成实验文件并注册运行。
user-invocable: true
allowed-tools: Read Write Edit Bash Grep Glob
---

# 构建新实验

一步步引导用户创建新的 Mask Embedding 实验。

## 流程概览

```
1. 问清楚 →  2. 选标签 →  3. 写代码 →  4. 注册 →  5. 跑实验
```

## Step 1: 问清楚假设

先问用户 3 个问题，逐题问，不要一次全抛：

> **Q1. 你想验证什么？**
> 一句话描述假设。例如「字母 A 与字母 H 由不同笔画构成，embedding 应能区分它们」。

> **Q2. 你关心的是对什么的 invariance / separation？**
> 例如：颜色不变？位置不变？字母之间分离？中英文分离？

> **Q3. 假设反过来的话，什么算「正确」？**
> 例如：如果同字母不同颜色 → embedding 相似度 > 0.9，那就算位置不变性成立。

等用户答完，总结给他确认：
- **假设**: xxx
- **判据**: 当 xxx 时算正确
- **实验名**: xxx（从假设中提炼，中文 4-8 字）

## Step 2: 选标签

根据用户的假设，帮他选出需要的数据标签。当前合成数据提供以下标签：

| 标签字段 | 可选值 | 例子 |
|---------|--------|------|
| `lang` | EN, ZH | 区分中英文用这个 |
| `label` | A-Z, 我-风 (20个) | 区分字符用这个 |
| `color` | Red,Green,Blue,Yellow,Cyan,Magenta,White,Orange (8个) | 区分颜色用这个 |
| `position` | TL,TR,CT,BL,BR (5个) | 区分位置用这个 |

向用户确认：**要用哪几个标签？**（通常 1-2 个）

## Step 3: 写实验文件

根据前面确认的信息，在 `experiments/text/` 下创建实验文件。

**模板** —— 填 `{{占位符}}`：

```python
"""
实N: {{实验名称}}

假设: {{一句话假设}}
"""
import numpy as np
from pathlib import Path
from experiments.base import BaseExperiment
from experiments.viz import tsne_plot
from experiments.source.synthetic.config import OUTPUT_ROOT, {{需要的色板}}


class Exp{{类名}}(BaseExperiment):
    name = "{{实验名称}}"
    hypothesis = "{{一句话假设}}"
    data_source = "experiments/source/synthetic"
    what_labels = [{{标签列表}}]
    passes_when = "{{判据描述}}"

    def run(self, emb, labels, sim):
        groups = labels["{{主标签字段}}"]
        keys = list(groups.keys())

        # ── 度量: {{描述怎么算}} ──
        {{度量代码}}

        return {
            "name": self.name,
            "hypothesis": self.hypothesis,
            "metric": f"{{指标名}}={{{变量名}}:.4f}",
            "separation": float({{变量名}}),
            "is_correct": {{判据表达式}},
            "details": {},
        }

    def viz(self, emb, labels, result, algo_name, out_dir):
        tsne_plot(
            emb,
            labels["{{用于视觉化的标签}}"],
            {{色板字典}},
            f"ExpN: {self.name} [{algo_name}]",
            out_dir / "expN_{{文件名}}.png",
            result.get("metric", ""),
        )
```

**写代码时的要求**:
- `run()` 永远用已有的 `sim` 矩阵（`emb @ emb.T`），不要重复算
- 度量方式用 `_within_between` 模式：计算同类内相似度均值、类间相似度均值、做差
- `separation` 值越大越好
- `is_correct` 用 > / < 阈值来判断
- `viz()` 调用 `tsne_plot()`，选对标签字段和色板

**色板参考**（从 `experiments/source/synthetic/config.py` 导入）:
- `COLOR_HEX` — 8 种颜色
- `POS_HEX` — 5 个位置
- 字符色板需自定义：`EN_LETTERS` → `plt.cm.tab10`, `ZH_CHARS` → `plt.cm.Set3`

## Step 4: 注册实验

在对应的 `experiments/{{类别}}/__init__.py` 中：

1. 顶部 import 新实验 class
2. 在注册字典里加一项，**key = 实验文件名**（不用数字）

```python
from experiments.{{类别}}.{{文件名}} import Exp{{类名}}

{{注册表名}} = {
    ...
    "{{文件名}}": Exp{{类名}}(),   # ← 新增
}
```

> **key 命名规则**: key = 文件名（不含 .py）。如 `stroke.py` → `"stroke"`。避免 exp1/exp2 数字混乱。

## Step 5: 跑实验

```bash
uv run python run.py --exp exp6
```

跑完后把结果展示给用户：指标、是否正确、t-SNE 图。

## 已有实验参考

| 实验 | 文件 | 假设 | 判据 |
|------|------|------|------|
| exp1 中英文 | text/lang.py | 中英文两种书写系统，应能区分 | sep > 0.05 |
| exp2 位置不变 | text/position.py | 同字符在不同位置，应高度相似 | sim > 0.90 |
| exp3 颜色可分 | text/color.py | 不同颜色在不同通道签名不同 | sep > 0.05 |
| exp4 位置编码 | text/pos_encode.py | 同位置有微量相似可编码 | sep > 0.02 |
| exp5 字符可分 | text/char.py | 不同字符形状不同，应能区分 | sep > 0.05 |

---

## 注意事项

- **不要**修改 `algorithms/` 或 `experiments/base.py`
- **不要**修改 `run.py`
- 只改两个地方：`experiments/text/{{新文件}}.py` 和 `experiments/text/__init__.py`
- 实验文件名用英文下划线，如 `stroke_type.py`
- 用户可能不懂代码。用自然语言确认逻辑后帮他写代码，写完后解释每一段做什么
