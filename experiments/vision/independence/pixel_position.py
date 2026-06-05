"""像素位置独立性 —— 固定数量，只变位置，embedding 应该相似"""
import numpy as np
from experiments.base import BaseExperiment
from experiments.source.synthetic.pixel import generate_pixel_dataset
from experiments.source.synthetic.config import SEED


class ExpPixelPositionIndependence(BaseExperiment):
    name = "像素位置独立性"
    hypothesis = "固定像素数量，只变位置，embedding 应该相似（位置变化不影响数量表示）"
    passes_when = "sim > 0.90"
    uses_rgb = False

    def load_data(self):
        masks, labels, _ = generate_pixel_dataset(grid_step=4, n_multi=64, seed=SEED)
        return masks, labels

    def run(self, emb, labels, sim):
        if not self.check_labels(labels, ["count"]):
            return {}

        all_sims = []

        # 按数量分组，测试同一数量下不同位置的相似度
        for count_key, count_idx in labels["count"].items():
            if len(count_idx) < 2:
                continue

            # 对于单点情况，按位置分组
            if count_key == "1pt":
                # 取每个位置的代表性样本
                x_groups = labels.get("x_pos", {})
                y_groups = labels.get("y_pos", {})

                # 取不同 x 位置的样本
                representatives = []
                for x_key, x_idx in x_groups.items():
                    if x_idx:
                        representatives.append(x_idx[0])

                if len(representatives) >= 2:
                    rep_arr = np.array(representatives)
                    s = sim[rep_arr][:, rep_arr]
                    mask = ~np.eye(len(representatives), dtype=bool)
                    if mask.sum() > 0:
                        all_sims.append(s[mask].mean())

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
