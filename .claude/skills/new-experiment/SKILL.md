---
name: new-experiment
description: 引导用户一步步构建新实验。通过问答方式帮助声明四要素（数据来源、假设、度量、可视化），自动生成实验文件并注册运行。
user-invocable: true
allowed-tools: Read Write Edit Bash Grep Glob
---

# 构建新实验

一步步引导用户创建新的 Mask Embedding 实验。

## 核心目标

验证一个 embedding 空间能否同时表达多种属性（字符、位置、旋转、大小），以及各属性之间是否独立。

## 实验分类

### 1. 独立性实验 (Independence Tests)

**问题**：属性 A 的变化是否影响 embedding？

**方法**：
- 固定其他属性
- 只变化目标属性
- 取每个属性值的代表性样本
- 计算这些样本之间的相似度

**通过标准**：sim > 0.90（不同属性值的样本高度相似）

**含义**：如果通过，说明该属性变化不影响 embedding，属性没有被编码。

### 2. 可分性实验 (Separability Tests)

**问题**：属性 A 是否被编码在 embedding 中？

**方法**：
- 固定其他属性
- 只变化目标属性
- 按属性值分组
- 计算组间分离度 = 组内相似度 - 组间相似度

**通过标准**：separation > 0.05（不同属性值的样本能区分）

**含义**：如果通过，说明该属性被编码了，不同属性值的 embedding 不同。

### 3. 容量实验 (Capacity Tests)

**问题**：embedding 能否同时编码多个属性？

**方法**：
- 按多个属性分组
- 计算组内相似度

**通过标准**：sim > 0.80（完全相同的样本高度相似）

**含义**：如果通过，说明 embedding 空间有足够容量表达多种属性。

## 实验之间的关系

### 独立性 vs 可分性：互斥

```
如果独立性通过（sim > 0.90）
  → 可分性应该失败（separation < 0.05）
  → 说明属性没有被编码

如果可分性通过（separation > 0.05）
  → 独立性应该失败（sim < 0.90）
  → 说明属性被编码了
```

**验证**：实验结果应该是自洽的。

## 流程概览

```
1. 确定实验类型 →  2. 问清楚假设 →  3. 选标签 →  4. 写代码 →  5. 注册 →  6. 测试 →  7. 跑实验
```

## Step 1: 确定实验类型

先问用户要创建哪种类型的实验：

> **Q1. 你想创建哪种类型的实验？**
> 1. **独立性实验**：测属性变化是否影响 embedding（sim > 0.90 通过）
> 2. **可分性实验**：测属性是否被编码（separation > 0.05 通过）
> 3. **容量实验**：测能否同时编码多个属性（sim > 0.80 通过）

然后问清楚假设：

> **Q2. 你想验证什么？**
> 一句话描述假设。例如「位置变化是否影响字符表示」。

> **Q3. 固定什么，变化什么？**
> 例如：固定字符、旋转、大小，只变位置。

等用户答完，总结给他确认：
- **实验类型**: 独立性/可分性/容量
- **假设**: xxx
- **固定**: xxx
- **变化**: xxx
- **判据**: 当 xxx 时算正确
- **实验名**: xxx（从假设中提炼，中文 4-8 字）

> **注意**：如果用户已经明确说了实验类型和假设，直接跳到 Step 2，不要重复问。

## Step 2: 选标签

根据用户的假设，帮他选出需要的数据标签。

### 文字实验标签

| 标签字段 | 可选值 | 例子 |
|---------|--------|------|
| `lang` | EN, ZH | 区分中英文用这个 |
| `label` | A-Z, 我-风 (20个) | 区分字符用这个 |
| `color` | Red,Green,Blue,Yellow,Cyan,Magenta,White,Orange (8个) | 区分颜色用这个 |
| `position` | TL,TR,CT,BL,BR (5个) | 区分位置用这个 |
| `rotation` | 0,90,180,270 (4个) | 区分旋转角度用这个 |
| `size` | small,medium,large (3个) | 区分字符大小用这个 |

### 视觉实验标签

**像素实验**：
| 标签字段 | 可选值 | 例子 |
|---------|--------|------|
| `count` | 1pt,2pt,3pt,5pt,10pt | 区分像素数量用这个 |
| `x_pos` | x=0,x=4,...,x=60 | 区分 x 坐标用这个 |
| `y_pos` | y=0,y=4,...,y=60 | 区分 y 坐标用这个 |

**线实验**：
| 标签字段 | 可选值 | 例子 |
|---------|--------|------|
| `position` | (x,y) 坐标 | 区分线起点位置用这个 |
| `rotation` | 0,45,90,135,180,225,270,315 | 区分线角度用这个 |
| `length` | 15,25,35 | 区分线长度用这个 |

向用户确认：**要用哪几个标签？**（通常 1-2 个）

> **注意**：如果假设涉及旋转或大小，需要确认 config.py 和 data.py 已支持。若未支持，先改数据层再写实验。

## Step 3: 写实验文件

根据前面确认的信息，在对应目录下创建实验文件：

- 独立性实验：`experiments/text/independence/`
- 可分性实验：`experiments/text/separability/`
- 容量实验：`experiments/text/capacity/`

### 独立性实验模板

```python
"""{{实验名称}} —— {{一句话描述}}"""
import numpy as np
from experiments.base import BaseExperiment
from experiments.viz import tsne_plot
from experiments.source.synthetic.config import {{需要的色板}}
from experiments.source.synthetic.data import Dataset


class Exp{{类名}}(BaseExperiment):
    name = "{{实验名称}}"
    hypothesis = "{{一句话假设}}"
    passes_when = "sim > 0.90"
    uses_rgb = False  # 使用 mask

    def load_data(self):
        ds = Dataset().generate(verbose=True)
        return ds.masks(), ds.labels

    def run(self, emb, labels, sim):
        if not self.check_labels(labels, [{{标签列表}}]): return {}

        all_sims = []

        # 遍历所有固定属性的组合
        for {{固定属性1}}, {{固定属性1}}_idx in labels["{{固定属性1字段}}"].items():
            for {{固定属性2}}, {{固定属性2}}_idx in labels["{{固定属性2字段}}"].items():
                # 取交集
                base = set({{固定属性1}}_idx) & set({{固定属性2}}_idx)
                if not base:
                    continue

                # 按变化属性分组
                groups = {}
                for val, val_idx in labels["{{变化属性字段}}"].items():
                    group = sorted(list(base & set(val_idx)))
                    if group:
                        groups[val] = group

                # 计算不同属性值样本之间的相似度
                if len(groups) >= 2:
                    representatives = [group[0] for group in groups.values()]
                    rep_arr = np.array(representatives)
                    s = sim[rep_arr][:, rep_arr]
                    mask = ~np.eye(len(representatives), dtype=bool)
                    if mask.sum() > 0:
                        all_sims.append(s[mask].mean())

        avg_sim = float(np.mean(all_sims)) if all_sims else 0.0
        return {
            "name": self.name,
            "metric": f"sim={avg_sim:.4f}",
            "separation": avg_sim,
            "is_correct": avg_sim > 0.90,
            "details": {"mean_sim": avg_sim, "n_conditions": len(all_sims)},
        }

    def viz(self, emb, labels, result, algo_name, out_dir):
        tsne_plot(emb, labels["{{变化属性字段}}"], {{色板}},
                  f"{{实验名称}} [{algo_name}]",
                  out_dir/"{{文件名}}.png", result.get("metric", ""))
```

### 可分性实验模板

```python
"""{{实验名称}} —— {{一句话描述}}"""
import numpy as np
from experiments.base import BaseExperiment
from experiments.viz import tsne_plot
from experiments.source.synthetic.config import {{需要的色板}}
from experiments.source.synthetic.data import Dataset
from experiments.metrics import calc_group_separation


class Exp{{类名}}(BaseExperiment):
    name = "{{实验名称}}"
    hypothesis = "{{一句话假设}}"
    passes_when = "separation > 0.05"
    uses_rgb = False  # 使用 mask

    def load_data(self):
        ds = Dataset().generate(verbose=True)
        return ds.masks(), ds.labels

    def run(self, emb, labels, sim):
        if not self.check_labels(labels, [{{标签列表}}]): return {}

        all_seps = []

        # 遍历所有固定属性的组合
        for {{固定属性1}}, {{固定属性1}}_idx in labels["{{固定属性1字段}}"].items():
            for {{固定属性2}}, {{固定属性2}}_idx in labels["{{固定属性2字段}}"].items():
                # 取交集
                base = set({{固定属性1}}_idx) & set({{固定属性2}}_idx)
                if not base:
                    continue

                # 按变化属性分组
                groups = {}
                for val, val_idx in labels["{{变化属性字段}}"].items():
                    group = sorted(list(base & set(val_idx)))
                    if group:
                        groups[val] = group

                # 计算分离度
                if len(groups) >= 2:
                    sep = calc_group_separation(sim, groups)
                    all_seps.append(sep)

        avg_sep = float(np.mean(all_seps)) if all_seps else 0.0
        return {
            "name": self.name,
            "metric": f"separation={avg_sep:+.4f}",
            "separation": avg_sep,
            "is_correct": avg_sep > 0.05,
            "details": {"mean_sep": avg_sep, "n_conditions": len(all_seps)},
        }

    def viz(self, emb, labels, result, algo_name, out_dir):
        tsne_plot(emb, labels["{{变化属性字段}}"], {{色板}},
                  f"{{实验名称}} [{algo_name}]",
                  out_dir/"{{文件名}}.png", result.get("metric", ""))
```

### 容量实验模板

```python
"""{{实验名称}} —— {{一句话描述}}"""
import numpy as np
from experiments.base import BaseExperiment
from experiments.source.synthetic.data import Dataset
from experiments.metrics import calc_group_similarity


class Exp{{类名}}(BaseExperiment):
    name = "{{实验名称}}"
    hypothesis = "{{一句话假设}}"
    passes_when = "sim > 0.80"
    uses_rgb = False  # 使用 mask

    def load_data(self):
        ds = Dataset().generate(verbose=True)
        return ds.masks(), ds.labels

    def run(self, emb, labels, sim):
        if not self.check_labels(labels, [{{标签列表}}]): return {}

        # 按多个属性分组
        groups = {}
        for {{属性1}}, {{属性1}}_idx in labels["{{属性1字段}}"].items():
            for {{属性2}}, {{属性2}}_idx in labels["{{属性2字段}}"].items():
                group = sorted(list(set({{属性1}}_idx) & set({{属性2}}_idx)))
                if len(group) >= 2:
                    groups[({{属性1}}, {{属性2}})] = group

        avg_sim = calc_group_similarity(sim, groups) if groups else 0.0
        return {
            "name": self.name,
            "metric": f"sim={avg_sim:.4f}",
            "separation": avg_sim,
            "is_correct": avg_sim > 0.80,
            "details": {"mean_sim": avg_sim, "n_groups": len(groups)},
        }

    def viz(self, emb, labels, result, algo_name, out_dir):
        pass  # 容量实验可视化较复杂，暂时跳过
```

### 通用度量函数

在 `experiments/metrics.py` 中：

- `calc_group_similarity(sim, groups)`：计算组内相似度
- `calc_group_separation(sim, groups)`：计算组间分离度

### 写代码时的要求

- `run()` 永远用已有的 `sim` 矩阵（`emb @ emb.T`），不要重复算
- 独立性实验：取每个属性值的代表性样本，计算它们之间的相似度
- 可分性实验：使用 `calc_group_separation()` 计算分离度
- 容量实验：使用 `calc_group_similarity()` 计算组内相似度
- `load_data()` 统一用 `Dataset().generate()` 返回 `ds.masks(), ds.labels`

### 色板参考（从 `experiments/source/synthetic/config.py` 导入）

- `COLOR_HEX` — 8 种颜色
- `POS_HEX` — 5 个位置
- `ROTATION_HEX` — 4 个旋转角度
- 字符色板需自定义：`EN_LETTERS` → `plt.cm.tab10`, `ZH_CHARS` → `plt.cm.Set3`

## Step 4: 注册实验

根据实验类型，在对应的 `__init__.py` 中注册：

- 独立性实验：`experiments/text/independence/__init__.py`
- 可分性实验：`experiments/text/separability/__init__.py`
- 容量实验：`experiments/text/capacity/__init__.py`

步骤：
1. 顶部 import 新实验 class
2. 在注册字典里加一项，**key = 实验文件名**（不用数字）

```python
from experiments.text.{{类型}}.{{文件名}} import Exp{{类名}}

{{类型大写}}_EXPERIMENTS = {
    ...
    "{{文件名}}": Exp{{类名}},   # ← 新增（注意：不加括号，由 run.py 实例化）
}
```

> **key 命名规则**: key = 文件名（不含 .py）。如 `position.py` → `"position"`。

## Step 5: 测试

写完代码后，先跑 pytest 确保数据流通性：

```bash
uv run pytest tests/test_text_experiments.py -v
```

测试会自动覆盖新注册的实验，检查：
1. `load_data()` 返回正确结构
2. `run()` 返回 dict 含 `name`/`is_correct`/`metric`
3. `viz()` 不崩溃

**必须测试通过后才能跑实验。**

## Step 6: 跑实验

```bash
# 跑单个实验
uv run python run.py --exp {{文件名}} --no-viz

# 跑某类实验
uv run python run.py --category independence --no-viz
uv run python run.py --category separability --no-viz
uv run python run.py --category capacity --no-viz
```

跑完后把结果展示给用户：指标、是否正确、t-SNE 图。

## 已有实验参考

### 文字实验

#### 独立性实验

| 实验 | 文件 | 假设 | 固定 | 变化 | 判据 |
|------|------|------|------|------|------|
| 位置独立性 | text/independence/position.py | 位置变化不影响字符表示 | 字符、旋转、大小 | 位置 | sim > 0.90 |
| 旋转独立性 | text/independence/rotation.py | 旋转变化不影响字符表示 | 字符、位置、大小 | 旋转 | sim > 0.90 |
| 大小独立性 | text/independence/scale.py | 大小变化不影响字符表示 | 字符、位置、旋转 | 大小 | sim > 0.90 |

#### 可分性实验

| 实验 | 文件 | 假设 | 固定 | 变化 | 判据 |
|------|------|------|------|------|------|
| 字符可分性 | text/separability/character.py | 字符形状被编码 | 位置、旋转、大小 | 字符 | sep > 0.05 |
| 位置可分性 | text/separability/position.py | 位置信息被编码 | 字符、旋转、大小 | 位置 | sep > 0.02 |
| 旋转可分性 | text/separability/rotation.py | 旋转角度被编码 | 字符、位置、大小 | 旋转 | sep > 0.05 |
| 大小可分性 | text/separability/scale.py | 字符大小被编码 | 字符、位置、旋转 | 大小 | sep > 0.05 |

#### 容量实验

| 实验 | 文件 | 假设 | 分组依据 | 判据 |
|------|------|------|----------|------|
| 字符+位置 | text/capacity/char_pos.py | 能同时编码字符和位置 | (字符, 位置) | sim > 0.85 |
| 字符+大小 | text/capacity/char_scale.py | 能同时编码字符和大小 | (字符, 大小) | sim > 0.85 |
| 全属性 | text/capacity/full.py | 能同时编码所有属性 | (字符, 位置, 旋转, 大小) | sim > 0.80 |

### 视觉实验

#### 独立性实验

| 实验 | 文件 | 假设 | 固定 | 变化 | 判据 |
|------|------|------|------|------|------|
| 像素位置独立性 | vision/independence/pixel_position.py | 位置变化不影响数量表示 | 数量 | 位置 | sim > 0.90 |
| 线位置独立性 | vision/independence/line_position.py | 位置变化不影响线的表示 | 旋转、长度 | 位置 | sim > 0.90 |
| 线旋转独立性 | vision/independence/line_rotation.py | 旋转变化不影响线的表示 | 位置、长度 | 旋转 | sim > 0.90 |
| 线长度独立性 | vision/independence/line_length.py | 长度变化不影响线的表示 | 位置、旋转 | 长度 | sim > 0.90 |

#### 可分性实验

| 实验 | 文件 | 假设 | 固定 | 变化 | 判据 |
|------|------|------|------|------|------|
| 像素位置可分性 | vision/separability/pixel_position.py | 位置信息被编码 | 数量 | 位置 | sep > 0.02 |
| 像素数量可分性 | vision/separability/pixel_count.py | 数量信息被编码 | - | 数量 | sep > 0.03 |
| 线位置可分性 | vision/separability/line_position.py | 位置信息被编码 | 旋转、长度 | 位置 | sep > 0.02 |
| 线旋转可分性 | vision/separability/line_rotation.py | 旋转角度被编码 | 位置、长度 | 旋转 | sep > 0.05 |
| 线长度可分性 | vision/separability/line_length.py | 长度信息被编码 | 位置、旋转 | 长度 | sep > 0.05 |

#### 容量实验

| 实验 | 文件 | 假设 | 分组依据 | 判据 |
|------|------|------|----------|------|
| 线位置+旋转 | vision/capacity/line_pos_rot.py | 能同时编码位置和旋转 | (位置, 旋转) | sim > 0.80 |
| 线全属性 | vision/capacity/line_full.py | 能同时编码所有属性 | (位置, 旋转, 长度) | sim > 0.80 |

---

## 注意事项

- **不要**修改 `algorithms/` 或 `experiments/base.py`
- **不要**修改 `run.py`
- 实验文件名用英文下划线，如 `stroke_type.py`
- 用户可能不懂代码。用自然语言确认逻辑后帮他写代码，写完后解释每一段做什么
- 如果需要新标签（如旋转或大小），先改 config.py 和 data.py，再写实验
- 独立性实验和可分性实验是互斥的：如果一个通过，另一个应该失败
- 容量实验是基本要求：相同输入必须得到相同输出
