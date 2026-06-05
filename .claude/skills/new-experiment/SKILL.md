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
1. 问清楚 →  2. 选标签 →  3. 写代码 →  4. 注册 →  5. 测试 →  6. 跑实验
```

## Step 1: 问清楚假设

先问用户 3 个问题，**逐题问，不要一次全抛**：

> **Q1. 你想验证什么？**
> 一句话描述假设。例如「字母 A 与字母 H 由不同笔画构成，embedding 应能区分它们」。

> **Q2. 你关心的是对什么的 invariance / separation？**
> 例如：颜色不变？位置不变？字母之间分离？中英文分离？旋转可分？

> **Q3. 假设反过来的话，什么算「正确」？**
> 例如：如果同字母不同颜色 → embedding 相似度 > 0.9，那就算位置不变性成立。

等用户答完，总结给他确认：
- **假设**: xxx
- **判据**: 当 xxx 时算正确
- **实验名**: xxx（从假设中提炼，中文 4-8 字）

> **注意**：如果用户已经明确说了假设和判据（比如「旋转可分性，sep > 0.05」），直接跳到 Step 2，不要重复问。

## Step 2: 选标签

根据用户的假设，帮他选出需要的数据标签。当前合成数据提供以下标签：

| 标签字段 | 可选值 | 例子 |
|---------|--------|------|
| `lang` | EN, ZH | 区分中英文用这个 |
| `label` | A-Z, 我-风 (20个) | 区分字符用这个 |
| `color` | Red,Green,Blue,Yellow,Cyan,Magenta,White,Orange (8个) | 区分颜色用这个 |
| `position` | TL,TR,CT,BL,BR (5个) | 区分位置用这个 |
| `rotation` | 0,90,180,270 (4个) | 区分旋转角度用这个 |

向用户确认：**要用哪几个标签？**（通常 1-2 个）

> **注意**：如果假设涉及旋转，需要确认 config.py 和 data.py 已支持旋转。若未支持，先改数据层再写实验。

## Step 3: 写实验文件

根据前面确认的信息，在 `experiments/text/` 或 `experiments/vision/` 下创建实验文件。

**模板** —— 填 `{{占位符}}`：

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
    passes_when = "{{判据描述}}"
    uses_rgb = True

    def load_data(self):
        ds = Dataset().generate(verbose=True)
        return ds.rgbs(), ds.labels

    def run(self, emb, labels, sim):
        if not self.check_labels(labels, [{{标签列表}}]): return {}
        groups = labels["{{主标签字段}}"]
        keys = list(groups.keys())

        # ── 度量: {{描述怎么算}} ──
        {{度量代码}}

        return {
            "name": self.name,
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
            f"{{实验名称}} [{algo_name}]",
            out_dir / "{{文件名}}.png",
            result.get("metric", ""),
        )
```

### 度量代码参考

**分离度模式** (separation，用于 lang/color/char/rotation_sep/pos_encode)：
```python
within, between = [], []
for g in keys:
    arr = np.array(groups[g])
    s = sim[arr][:, arr]
    within.append(s[~np.eye(len(arr), dtype=bool)].mean())
for i, gi in enumerate(keys):
    for gj in keys[i+1:]:
        cr = sim[np.array(groups[gi])][:, np.array(groups[gj])].mean()
        between.append(cr)
sep = float(np.mean(within) - np.mean(between))
```

**不变性模式** (invariance，用于 position/rotation_inv)：
```python
all_sims = []
for idxs in groups.values():
    arr = np.array(idxs)
    s = sim[arr][:, arr]
    all_sims.extend(s[~np.eye(len(arr), dtype=bool)].tolist())
avg = float(np.mean(all_sims))
```

### 写代码时的要求

- `run()` 永远用已有的 `sim` 矩阵（`emb @ emb.T`），不要重复算
- `separation` 值越大越好
- `is_correct` 用 > / < 阈值来判断
- `viz()` 调用 `tsne_plot()`，选对标签字段和色板
- `load_data()` 统一用 `Dataset().generate()` 返回 `ds.rgbs(), ds.labels`

### 色板参考（从 `experiments/source/synthetic/config.py` 导入）

- `COLOR_HEX` — 8 种颜色
- `POS_HEX` — 5 个位置
- `ROTATION_HEX` — 4 个旋转角度
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

> **key 命名规则**: key = 文件名（不含 .py）。如 `rotation_sep.py` → `"rotation_sep"`。避免 exp1/exp2 数字混乱。

## Step 5: 测试

写完代码后，先跑 pytest 确保数据流通性：

```bash
uv run pytest tests/test_text_experiments.py -v
```

测试会自动覆盖新注册的实验（因为 `TEXT_EXPERIMENTS` 是动态迭代的），检查：
1. `load_data()` 返回正确结构
2. `run()` 返回 dict 含 `name`/`is_correct`/`metric`
3. `viz()` 不崩溃

**必须测试通过后才能跑实验。**

## Step 6: 跑实验

```bash
uv run python run.py --exp {{文件名}}
```

跑完后把结果展示给用户：指标、是否正确、t-SNE 图。

## 已有实验参考

| 实验 | 文件 | 假设 | 判据 | 模式 |
|------|------|------|------|------|
| 中英文可分 | text/lang.py | 中英文两种书写系统，应能区分 | sep > 0.05 | separation |
| 位置不变 | text/position.py | 同字符在不同位置，应高度相似 | sim > 0.90 | invariance |
| 颜色可分 | text/color.py | 不同颜色在不同通道签名不同 | sep > 0.05 | separation |
| 位置编码 | text/pos_encode.py | 同位置有微量相似可编码 | sep > 0.02 | separation |
| 字符可分 | text/char.py | 不同字符形状不同，应能区分 | sep > 0.05 | separation |
| 旋转可分 | text/rotation_sep.py | 不同旋转角度应能区分 | sep > 0.05 | separation |
| 旋转不变 | text/rotation_inv.py | 同字符不同旋转应相似 | sim > 0.90 | invariance |
| 像素可分 | vision/pixel.py | 单/多像素位置和数量可编码 | pos>0.05 或 count>0.03 | separation |

---

## 注意事项

- **不要**修改 `algorithms/` 或 `experiments/base.py`
- **不要**修改 `run.py`
- 实验文件名用英文下划线，如 `stroke_type.py`
- 用户可能不懂代码。用自然语言确认逻辑后帮他写代码，写完后解释每一段做什么
- 如果需要新标签（如旋转），先改 config.py 和 data.py，再写实验
