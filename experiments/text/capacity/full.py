"""Exp-C3: 全属性联合编码 —— 按 (字符, 位置, 旋转, 大小) 分组"""
import numpy as np
from experiments.base import BaseExperiment
from experiments.source.synthetic.data import Dataset
from experiments.metrics import calc_group_similarity


class ExpFullCapacity(BaseExperiment):
    name = "全属性联合编码"
    hypothesis = "embedding 能否同时编码字符、位置、旋转、大小？完全相同的样本应高度相似"
    passes_when = "mean_similarity > 0.80"
    uses_rgb = False  # 使用 mask

    def load_data(self):
        ds = Dataset().generate(verbose=True)
        return ds.masks(), ds.labels

    def run(self, emb, labels, sim):
        if not self.check_labels(labels, ["label", "position", "rotation", "size"]):
            return {}

        # 按 (字符, 位置, 旋转, 大小) 分组
        full_groups = {}

        for char, char_idx in labels["label"].items():
            for pos, pos_idx in labels["position"].items():
                for rot, rot_idx in labels["rotation"].items():
                    for size, size_idx in labels["size"].items():
                        # 取交集：同时满足所有属性的样本
                        group = sorted(list(
                            set(char_idx) & set(pos_idx) & set(rot_idx) & set(size_idx)
                        ))
                        if len(group) >= 2:
                            full_groups[(char, pos, rot, size)] = group

        # 计算组内相似度
        avg_sim = calc_group_similarity(sim, full_groups) if full_groups else 0.0

        return {
            "name": self.name,
            "metric": f"sim={avg_sim:.4f}",
            "separation": avg_sim,
            "is_correct": avg_sim > 0.80,
            "details": {
                "mean_sim": avg_sim,
                "n_groups": len(full_groups),
            },
        }

    def viz(self, emb, labels, result, algo_name, out_dir):
        # 全属性实验的可视化较复杂，暂时跳过
        pass
