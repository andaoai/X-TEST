"""Exp-C1: 字符+位置联合编码 —— 固定旋转、大小，按 (字符, 位置) 分组"""
import numpy as np
from experiments.base import BaseExperiment
from experiments.source.synthetic.data import Dataset
from experiments.metrics import calc_group_similarity


class ExpCharPosCapacity(BaseExperiment):
    name = "字符+位置联合编码"
    hypothesis = "embedding 能否同时编码字符和位置？固定旋转、大小，同字符同位置的样本应相似"
    passes_when = "mean_similarity > 0.85"
    uses_rgb = False  # 使用 mask

    def load_data(self):
        ds = Dataset().generate(verbose=True)
        return ds.masks(), ds.labels

    def run(self, emb, labels, sim):
        if not self.check_labels(labels, ["label", "position", "rotation", "size"]):
            return {}

        all_sims = []

        # 遍历所有 (旋转, 大小) 组合
        for rot, rot_idx in labels["rotation"].items():
            for size, size_idx in labels["size"].items():
                # 取交集：同时满足旋转、大小的样本
                base = set(rot_idx) & set(size_idx)
                if not base:
                    continue

                # 在这些样本中，按 (字符, 位置) 分组
                char_pos_groups = {}
                for char, char_idx in labels["label"].items():
                    for pos, pos_idx in labels["position"].items():
                        group = sorted(list(base & set(char_idx) & set(pos_idx)))
                        if len(group) >= 2:
                            char_pos_groups[(char, pos)] = group

                # 计算这个子集的组内相似度
                if len(char_pos_groups) >= 2:
                    sim_val = calc_group_similarity(sim, char_pos_groups)
                    all_sims.append(sim_val)

        avg_sim = float(np.mean(all_sims)) if all_sims else 0.0
        return {
            "name": self.name,
            "metric": f"sim={avg_sim:.4f}",
            "separation": avg_sim,
            "is_correct": avg_sim > 0.85,
            "details": {
                "mean_sim": avg_sim,
                "n_conditions": len(all_sims),
            },
        }

    def viz(self, emb, labels, result, algo_name, out_dir):
        # 容量实验的可视化较复杂，暂时跳过
        pass
