"""形状旋转独立性 —— 同一形状旋转后，embedding 应该相似（旋转不影响形状表示）。

只取旋转敏感形状(三角/椭圆/线/十字),圆旋转平凡不变没有信息量。
位置编码只看坐标,旋转会改变像素位置 → 预期失败,与"旋转可分性"互斥自洽。
"""
import numpy as np
from experiments.base import BaseExperiment
from experiments.source.synthetic.shape import generate_shape_dataset, ROTATION_SENSITIVE
from experiments.source.synthetic.config import SEED


class ExpShapeRotationIndependence(BaseExperiment):
    name = "形状旋转独立性"
    hypothesis = "同一形状/大小/位置旋转不同角度，embedding 应该相似（旋转不影响形状表示）"
    passes_when = "sim > 0.90"
    uses_rgb = False

    def load_data(self):
        masks, labels = generate_shape_dataset(seed=SEED)
        return masks, labels

    def run(self, emb, labels, sim):
        if not self.check_labels(labels, ["shape", "size", "rotation", "position"]):
            return {}

        all_sims = []
        for shape in ROTATION_SENSITIVE:
            if shape not in labels["shape"]:
                continue
            shape_idx = labels["shape"][shape]
            for pos, pos_idx in labels["position"].items():
                for size, size_idx in labels["size"].items():
                    base = set(shape_idx) & set(pos_idx) & set(size_idx)
                    if not base:
                        continue
                    groups = {}
                    for rot, rot_idx in labels["rotation"].items():
                        group = sorted(base & set(rot_idx))
                        if group:
                            groups[rot] = group
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
