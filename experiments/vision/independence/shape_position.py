"""形状平移独立性 —— 同一形状平移到不同位置，embedding 应该相似（位置编码的"反面"对照）。

位置编码显式编码坐标，预期本实验失败（sim 远低于 0.90），
与"形状位置可分性通过"互斥自洽。
"""
import numpy as np
from experiments.base import BaseExperiment
from experiments.source.synthetic.shape import generate_shape_dataset
from experiments.source.synthetic.config import SEED


class ExpShapePositionIndependence(BaseExperiment):
    name = "形状平移独立性"
    hypothesis = "同一形状/大小/旋转平移到不同位置，embedding 应该相似（平移不影响形状表示）"
    passes_when = "sim > 0.90"
    uses_rgb = False

    def load_data(self):
        masks, labels = generate_shape_dataset(seed=SEED)
        return masks, labels

    def run(self, emb, labels, sim):
        if not self.check_labels(labels, ["shape", "size", "rotation", "position"]):
            return {}

        all_sims = []
        for shape, shape_idx in labels["shape"].items():
            for size, size_idx in labels["size"].items():
                for rot, rot_idx in labels["rotation"].items():
                    base = set(shape_idx) & set(size_idx) & set(rot_idx)
                    if not base:
                        continue
                    groups = {}
                    for pos, pos_idx in labels["position"].items():
                        group = sorted(base & set(pos_idx))
                        if group:
                            groups[pos] = group
                    if len(groups) >= 2:
                        reps = np.array([g[0] for g in groups.values()])
                        s = sim[reps][:, reps]
                        m = ~np.eye(len(reps), dtype=bool)
                        all_sims.append(s[m].mean())

        avg_sim = float(np.mean(all_sims)) if all_sims else 0.0
        return {
            "name": self.name,
            "metric": f"sim={avg_sim:.4f}",
            "separation": avg_sim,
            "is_correct": avg_sim > 0.90,
            "details": {"mean_sim": avg_sim, "n_conditions": len(all_sims)},
        }

    def viz(self, emb, labels, result, algo_name, out_dir):
        pass
