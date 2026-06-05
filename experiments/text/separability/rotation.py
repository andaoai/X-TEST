"""Exp-S3: 旋转可分性 —— 固定字符、位置、大小，只变旋转，embedding 应该不同"""
import numpy as np
from experiments.base import BaseExperiment
from experiments.viz import tsne_plot
from experiments.source.synthetic.config import ROTATION_HEX
from experiments.source.synthetic.data import Dataset
from experiments.metrics import calc_group_separation


class ExpRotationSeparation(BaseExperiment):
    name = "旋转可分性"
    hypothesis = "固定字符、位置、大小，只变旋转，embedding 应该不同（旋转角度被编码）"
    passes_when = "separation > 0.05"
    uses_rgb = False  # 使用 mask

    def load_data(self):
        ds = Dataset().generate(verbose=True)
        return ds.masks(), ds.labels

    def run(self, emb, labels, sim):
        if not self.check_labels(labels, ["label", "position", "size", "rotation"]):
            return {}

        all_seps = []

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

                    # 计算这个子集的分离度
                    if len(rot_groups) >= 2:
                        sep = calc_group_separation(sim, rot_groups)
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
        tsne_plot(emb, labels["rotation"], ROTATION_HEX,
                  f"Exp-S3: {self.name} [{algo_name}]",
                  out_dir/"exp_s3_rotation.png", result.get("metric", ""))
