"""形状旋转可分性 —— 固定形状、位置、大小，只变旋转角度，embedding 应该不同（旋转被编码）。

只取旋转下外观真正变化的形状(三角/椭圆/线/十字);圆旋转不变,正方形 90° 对称,
混入会稀释信号。
"""
import numpy as np
from experiments.base import BaseExperiment
from experiments.source.synthetic.shape import generate_shape_dataset, ROTATION_SENSITIVE
from experiments.source.synthetic.config import SEED
from experiments.metrics import calc_attr_effect


class ExpShapeRotationSeparation(BaseExperiment):
    name = "形状旋转可分性"
    hypothesis = "固定形状、位置、大小，只变旋转角度，embedding 应该不同（旋转被编码）"
    passes_when = "separation > 0.05"
    uses_rgb = False

    def load_data(self):
        masks, labels = generate_shape_dataset(seed=SEED)
        return masks, labels

    def run(self, emb, labels, sim):
        if not self.check_labels(labels, ["shape", "size", "rotation", "position"]):
            return {}

        # 只保留旋转敏感形状的样本索引(圆旋转不变,无信息量)
        keep = np.zeros(sim.shape[0], dtype=bool)
        for shape in ROTATION_SENSITIVE:
            if shape in labels["shape"]:
                keep[np.asarray(labels["shape"][shape])] = True
        sub = np.where(keep)[0]
        remap = -np.ones(sim.shape[0], dtype=int)
        remap[sub] = np.arange(len(sub))
        # labels 索引重映射到子矩阵编号
        sub_labels = {}
        for f, fv in labels.items():
            if not isinstance(fv, dict):
                continue
            sub_labels[f] = {}
            for v, idxs in fv.items():
                idxs_a = np.asarray(idxs)
                mapped = remap[idxs_a[keep[idxs_a]]]
                if len(mapped):
                    sub_labels[f][v] = list(mapped)
        sim_sub = sim[np.ix_(sub, sub)]

        sep = calc_attr_effect(sim_sub, sub_labels, attr="rotation",
                                   fixed=["shape", "position", "size"])
        return {
            "name": self.name,
            "metric": f"separation={sep:+.4f}",
            "separation": sep,
            "is_correct": sep > 0.05,
            "details": {"separation": sep},
        }

    def viz(self, emb, labels, result, algo_name, out_dir):
        pass
