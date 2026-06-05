"""Exp-S1: 字符可分性 —— 固定位置、旋转、大小，只变字符，embedding 应该不同"""
import numpy as np
import matplotlib.pyplot as plt
from experiments.base import BaseExperiment
from experiments.viz import tsne_plot
from experiments.source.synthetic.config import EN_LETTERS, ZH_CHARS
from experiments.source.synthetic.data import Dataset
from experiments.metrics import calc_group_separation


CHAR_PALETTE = {
    **{c: plt.cm.tab10(i / 10) for i, c in enumerate(EN_LETTERS)},
    **{c: plt.cm.Set3(i / 10) for i, c in enumerate(ZH_CHARS)},
}


class ExpCharSeparation(BaseExperiment):
    name = "字符可分性"
    hypothesis = "固定位置、旋转、大小，只变字符，embedding 应该不同（字符形状被编码）"
    passes_when = "separation > 0.05"
    uses_rgb = False  # 使用 mask

    def load_data(self):
        ds = Dataset().generate(verbose=True)
        return ds.masks(), ds.labels

    def run(self, emb, labels, sim):
        if not self.check_labels(labels, ["position", "rotation", "size", "label"]):
            return {}

        all_seps = []

        # 遍历所有 (位置, 旋转, 大小) 组合
        for pos, pos_idx in labels["position"].items():
            for rot, rot_idx in labels["rotation"].items():
                for size, size_idx in labels["size"].items():
                    # 取交集：同时满足位置、旋转、大小的样本
                    base = set(pos_idx) & set(rot_idx) & set(size_idx)
                    if not base:
                        continue

                    # 在这些样本中，按字符分组
                    char_groups = {}
                    for char, char_idx in labels["label"].items():
                        group = sorted(list(base & set(char_idx)))
                        if group:
                            char_groups[char] = group

                    # 计算这个子集的分离度
                    if len(char_groups) >= 2:
                        sep = calc_group_separation(sim, char_groups)
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
        tsne_plot(emb, labels["label"], CHAR_PALETTE,
                  f"Exp-S1: {self.name} [{algo_name}]",
                  out_dir/"exp_s1_char.png", result.get("metric", ""))
