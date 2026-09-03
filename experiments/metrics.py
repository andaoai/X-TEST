"""
通用度量函数 —— 用于控制变量实验

提供两个核心函数：
1. calc_group_similarity: 计算组内相似度
2. calc_group_separation: 计算组间分离度
"""
import numpy as np


def calc_group_similarity(sim: np.ndarray, groups: dict) -> float:
    """
    计算组内相似度均值

    Args:
        sim: (N, N) 相似度矩阵
        groups: {group_name: [sample_indices]}

    Returns:
        平均组内相似度
    """
    all_sims = []
    for idxs in groups.values():
        if len(idxs) < 2:
            continue
        arr = np.array(idxs)
        s = sim[arr][:, arr]
        # 排除对角线（自相似度=1）
        mask = ~np.eye(len(arr), dtype=bool)
        all_sims.extend(s[mask].tolist())
    return float(np.mean(all_sims)) if all_sims else 0.0


def calc_group_separation(sim: np.ndarray, groups: dict) -> float:
    """
    计算组间分离度 = mean(within) - mean(between)

    Args:
        sim: (N, N) 相似度矩阵
        groups: {group_name: [sample_indices]}

    Returns:
        分离度值
    """
    keys = list(groups.keys())
    if len(keys) < 2:
        return 0.0

    within, between = [], []

    # 组内相似度
    for k in keys:
        arr = np.array(groups[k])
        if len(arr) < 2:
            continue
        s = sim[arr][:, arr]
        within.append(s[~np.eye(len(arr), dtype=bool)].mean())

    # 组间相似度
    for i, ki in enumerate(keys):
        for kj in keys[i+1:]:
            arr_i = np.array(groups[ki])
            arr_j = np.array(groups[kj])
            if len(arr_i) == 0 or len(arr_j) == 0:
                continue
            cr = sim[arr_i][:, arr_j].mean()
            between.append(cr)

    if not within or not between:
        return 0.0

    return float(np.mean(within) - np.mean(between))


def _sample_value_array(labels: dict, field: str, n: int) -> np.ndarray:
    """labels 倒排索引 → 每样本的属性值数组(未覆盖位置为空字符串)。"""
    vals = np.full(n, "", dtype=object)
    for value, idxs in labels[field].items():
        vals[np.asarray(idxs)] = str(value)
    return vals


def calc_attr_effect(sim: np.ndarray, labels: dict, attr: str,
                     fixed: list[str]) -> float:
    """
    控制变量下"属性被编码的强度" = 1 − mean_sim(仅该属性不同的样本对)。

    每个(固定属性组合 × 变化属性值)只有 1 个样本时,没有组内重复,
    组内/组间分离度无法计算。改用严格配对:

      找出所有 fixed 属性完全相同、仅 attr 取值不同的样本对,
      它们的相似度下降量 1 - mean(sim) 就是该属性的编码强度。

    与独立性判据天然互斥: drop > 0.05 → 属性可分;
    1 - drop > 0.90(即 drop < 0.10)→ 属性独立。
    """
    n = sim.shape[0]
    triu = ~np.eye(n, dtype=bool)
    ai = _sample_value_array(labels, attr, n)
    covered = ai != ""

    diff = (ai[:, None] != ai[None, :]) & triu
    diff &= covered[:, None] & covered[None, :]
    for f in fixed:
        fa = _sample_value_array(labels, f, n)
        fc = fa != ""
        diff &= (fa[:, None] == fa[None, :]) & fc[:, None] & fc[None, :]

    if diff.sum() == 0:
        return 0.0
    return float(1.0 - sim[diff].mean())


def filter_labels(labels: dict, constraints: dict) -> list:
    """
    根据约束条件筛选样本索引

    Args:
        labels: {field: {value: [indices]}}
        constraints: {field: value} 或 {field: [values]}

    Returns:
        满足所有约束的样本索引列表
    """
    # 从所有样本开始
    all_indices = set()
    for idxs in labels[list(labels.keys())[0]].values():
        all_indices.update(idxs)

    result = all_indices

    for field, value in constraints.items():
        if field not in labels:
            continue

        field_indices = set()
        if isinstance(value, list):
            for v in value:
                if v in labels[field]:
                    field_indices.update(labels[field][v])
        else:
            if value in labels[field]:
                field_indices.update(labels[field][value])

        result = result & field_indices

    return sorted(list(result))
