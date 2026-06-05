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
