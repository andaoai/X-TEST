"""Exp-S4: 大小可分性 —— 固定字符、位置、旋转，只变大小，embedding 应该不同"""
import numpy as np
from experiments.base import BaseExperiment
from experiments.viz import tsne_plot
from experiments.source.synthetic.data import Dataset
from experiments.metrics import calc_group_separation


# 大小色板
SIZE_HEX = {"small": "#3498db", "medium": "#2ecc71", "large": "#e74c3c"}


class ExpScaleSeparation(BaseExperiment):
    name = "大小可分性"
    hypothesis = "固定字符、位置、旋转，只变大小，embedding 应该不同（字符大小被编码）"
    passes_when = "separation > 0.05"
    uses_rgb = False  # 使用 mask

    def load_data(self):
        ds = Dataset().generate(verbose=True)
        return ds.masks(), ds.labels

    def run(self, emb, labels, sim):
        if not self.check_labels(labels, ["label", "position", "rotation", "size"]):
            return {}

        all_seps = []

        # 遍历所有 (字符, 位置, 旋转) 组合
        for char, char_idx in labels["label"].items():
            for pos, pos_idx in labels["position"].items():
                for rot, rot_idx in labels["rotation"].items():
                    # 取交集：同时满足字符、位置、旋转的样本
                    base = set(char_idx) & set(pos_idx) & set(rot_idx)
                    if not base:
                        continue

                    # 在这些样本中，按大小分组
                    size_groups = {}
                    for size, size_idx in labels["size"].items():
                        group = sorted(list(base & set(size_idx)))
                        if group:
                            size_groups[size] = group

                    # 计算这个子集的分离度
                    if len(size_groups) >= 2:
                        sep = calc_group_separation(sim, size_groups)
                        all_seps.append(sep)

        avg_sep = float(np.mean(all_seps)) if all_seps else 0.0
        return {
            "name": self.name,
            "metric": f"separation={avg_sep:+.4f}",
            "separation": avg_sep,
            "is_correct": avg_sep > 0.05,
            "details": {
                "mean_sep": avg_sep,
                "n_conditions": len(all_seps),
            },
        }

    def viz(self, emb, labels, result, algo_name, out_dir):
        tsne_plot(emb, labels["size"], SIZE_HEX,
                  f"Exp-S4: {self.name} [{algo_name}]",
                  out_dir/"exp_s4_scale.png", result.get("metric", ""))
