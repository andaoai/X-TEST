"""Exp-I1: 位置独立性 —— 固定字符、旋转、大小，只变位置，embedding 应该相似"""
import numpy as np
from experiments.base import BaseExperiment
from experiments.viz import tsne_plot
from experiments.source.synthetic.config import POS_HEX
from experiments.source.synthetic.data import Dataset
from experiments.metrics import calc_group_similarity


class ExpPositionIndependence(BaseExperiment):
    name = "位置独立性"
    hypothesis = "固定字符、旋转、大小，只变位置，embedding 应该相似（位置变化不影响字符表示）"
    passes_when = "mean_similarity > 0.90"
    uses_rgb = False  # 使用 mask

    def load_data(self):
        ds = Dataset().generate(verbose=True)
        return ds.masks(), ds.labels

    def run(self, emb, labels, sim):
        if not self.check_labels(labels, ["label", "rotation", "size", "position"]):
            return {}

        all_sims = []

        # 遍历所有 (字符, 旋转, 大小) 组合
        for char, char_idx in labels["label"].items():
            for rot, rot_idx in labels["rotation"].items():
                for size, size_idx in labels["size"].items():
                    # 取交集：同时满足字符、旋转、大小的样本
                    base = set(char_idx) & set(rot_idx) & set(size_idx)
                    if not base:
                        continue

                    # 在这些样本中，按位置分组
                    pos_groups = {}
                    for pos, pos_idx in labels["position"].items():
                        group = sorted(list(base & set(pos_idx)))
                        if group:
                            pos_groups[pos] = group

                    # 计算不同位置样本之间的相似度
                    if len(pos_groups) >= 2:
                        # 取每个位置的第一个样本
                        representatives = []
                        for pos, group in pos_groups.items():
                            representatives.append(group[0])

                        # 计算这些代表性样本之间的相似度
                        rep_arr = np.array(representatives)
                        s = sim[rep_arr][:, rep_arr]
                        # 排除对角线
                        mask = ~np.eye(len(representatives), dtype=bool)
                        if mask.sum() > 0:
                            all_sims.append(s[mask].mean())

        avg_sim = float(np.mean(all_sims)) if all_sims else 0.0
        return {
            "name": self.name,
            "metric": f"sim={avg_sim:.4f}",
            "separation": avg_sim,
            "is_correct": avg_sim > 0.90,
            "details": {
                "mean_sim": avg_sim,
                "n_conditions": len(all_sims),
            },
        }

    def viz(self, emb, labels, result, algo_name, out_dir):
        tsne_plot(emb, labels["position"], POS_HEX,
                  f"Exp-I1: {self.name} [{algo_name}]",
                  out_dir/"exp_i1_position.png", result.get("metric", ""))
