"""像素数量可分性 —— 变数量，embedding 应该不同"""
import numpy as np
from experiments.base import BaseExperiment
from experiments.source.synthetic.pixel import generate_pixel_dataset
from experiments.source.synthetic.config import SEED
from experiments.metrics import calc_group_separation


class ExpPixelCountSeparation(BaseExperiment):
    name = "像素数量可分性"
    hypothesis = "不同数量的像素，embedding 应该不同（数量信息被编码）"
    passes_when = "separation > 0.03"
    uses_rgb = False

    def load_data(self):
        masks, labels, _ = generate_pixel_dataset(grid_step=4, n_multi=64, seed=SEED)
        return masks, labels

    def run(self, emb, labels, sim):
        if not self.check_labels(labels, ["count"]):
            return {}

        # 按数量分组
        groups = labels["count"]

        # 计算分离度
        sep = calc_group_separation(sim, groups) if len(groups) >= 2 else 0.0

        return {
            "name": self.name,
            "metric": f"separation={sep:+.4f}",
            "separation": sep,
            "is_correct": sep > 0.03,
            "details": {"separation": sep},
        }

    def viz(self, emb, labels, result, algo_name, out_dir):
        pass
