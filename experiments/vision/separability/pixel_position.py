"""像素位置可分性 —— 固定数量，只变位置，embedding 应该不同"""
import numpy as np
from experiments.base import BaseExperiment
from experiments.source.synthetic.pixel import generate_pixel_dataset
from experiments.source.synthetic.config import SEED
from experiments.metrics import calc_group_separation


class ExpPixelPositionSeparation(BaseExperiment):
    name = "像素位置可分性"
    hypothesis = "固定像素数量，只变位置，embedding 应该不同（位置信息被编码）"
    passes_when = "separation > 0.02"
    uses_rgb = False

    def load_data(self):
        masks, labels, _ = generate_pixel_dataset(grid_step=4, n_multi=64, seed=SEED)
        return masks, labels

    def run(self, emb, labels, sim):
        if not self.check_labels(labels, ["count", "x_pos"]):
            return {}

        all_seps = []

        # 对于单点情况，按 x 位置分组
        count_idx = labels["count"].get("1pt", [])
        if not count_idx:
            return {"name": self.name, "metric": "separation=+0.0000", "separation": 0.0,
                    "is_correct": False, "details": {}}

        # 按 x 位置分组
        x_groups = {}
        for x_key, x_idx in labels["x_pos"].items():
            group = sorted(list(set(count_idx) & set(x_idx)))
            if group:
                x_groups[x_key] = group

        # 计算分离度
        if len(x_groups) >= 2:
            sep = calc_group_separation(sim, x_groups)
            all_seps.append(sep)

        # 按 y 位置分组
        y_groups = {}
        for y_key, y_idx in labels.get("y_pos", {}).items():
            group = sorted(list(set(count_idx) & set(y_idx)))
            if group:
                y_groups[y_key] = group

        if len(y_groups) >= 2:
            sep = calc_group_separation(sim, y_groups)
            all_seps.append(sep)

        avg_sep = float(np.mean(all_seps)) if all_seps else 0.0
        return {
            "name": self.name,
            "metric": f"separation={avg_sep:+.4f}",
            "separation": avg_sep,
            "is_correct": avg_sep > 0.02,
            "details": {"mean_sep": avg_sep, "n_conditions": len(all_seps)},
        }

    def viz(self, emb, labels, result, algo_name, out_dir):
        pass
