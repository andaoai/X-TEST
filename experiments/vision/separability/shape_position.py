"""形状位置可分性 —— 固定形状、大小、旋转，只变位置，embedding 应该不同（位置信息被编码）"""
import numpy as np
from experiments.base import BaseExperiment
from experiments.source.synthetic.shape import generate_shape_dataset
from experiments.source.synthetic.config import SEED
from experiments.metrics import calc_attr_effect


class ExpShapePositionSeparation(BaseExperiment):
    name = "形状位置可分性"
    hypothesis = "固定形状、大小、旋转，只变位置，embedding 应该不同（位置编码应表达位置）"
    passes_when = "separation > 0.02"
    uses_rgb = False

    def load_data(self):
        masks, labels = generate_shape_dataset(seed=SEED)
        return masks, labels

    def run(self, emb, labels, sim):
        if not self.check_labels(labels, ["shape", "size", "rotation", "position"]):
            return {}

        sep = calc_attr_effect(sim, labels, attr="position",
                                   fixed=["shape", "size", "rotation"])
        return {
            "name": self.name,
            "metric": f"separation={sep:+.4f}",
            "separation": sep,
            "is_correct": sep > 0.02,
            "details": {"separation": sep},
        }

    def viz(self, emb, labels, result, algo_name, out_dir):
        pass
