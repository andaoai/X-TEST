"""Exp-I2: 旋转独立性 —— 固定字符、位置、大小，只变旋转，embedding 应该相似"""
import numpy as np
from experiments.base import BaseExperiment
from experiments.viz import tsne_plot
from experiments.source.synthetic.config import ROTATION_HEX
from experiments.source.synthetic.data import Dataset
from experiments.metrics import calc_group_similarity


class ExpRotationIndependence(BaseExperiment):
    name = "旋转独立性"
    hypothesis = "固定字符、位置、大小，只变旋转，embedding 应该相似（旋转变化不影响字符表示）"
    passes_when = "mean_similarity > 0.90"
    uses_rgb = False  # 使用 mask

    def load_data(self):
        ds = Dataset().generate(verbose=True)
        return ds.masks(), ds.labels

    def run(self, emb, labels, sim):
        if not self.check_labels(labels, ["label", "position", "size", "rotation"]):
            return {}

        all_sims = []

        # 遍历所有 (字符, 位置, 大小) 组合
        for char, char_idx in labels["label"].items():
            for pos, pos_idx in labels["position"].items():
                for size, size_idx in labels["size"].items():
                    # 取交集：同时满足字符、位置、大小的样本
                    base = set(char_idx) & set(pos_idx) & set(size_idx)
                    if not base:
                        continue

                    # 在这些样本中，按旋转分组
                    rot_groups = {}
                    for rot, rot_idx in labels["rotation"].items():
                        group = sorted(list(base & set(rot_idx)))
                        if group:
                            rot_groups[rot] = group

                    # 计算不同旋转样本之间的相似度
                    if len(rot_groups) >= 2:
                        # 取每个旋转的第一个样本
                        representatives = []
                        for rot, group in rot_groups.items():
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
        tsne_plot(emb, labels["rotation"], ROTATION_HEX,
                  f"Exp-I2: {self.name} [{algo_name}]",
                  out_dir/"exp_i2_rotation.png", result.get("metric", ""))
