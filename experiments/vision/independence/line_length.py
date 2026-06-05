"""线长度独立性 —— 固定位置、旋转，只变长度，embedding 应该相似"""
import numpy as np
from experiments.base import BaseExperiment
from experiments.source.synthetic.line import generate_line_dataset
from experiments.source.synthetic.config import SEED


class ExpLineLengthIndependence(BaseExperiment):
    name = "线长度独立性"
    hypothesis = "固定位置、旋转，只变长度，embedding 应该相似（长度变化不影响线的表示）"
    passes_when = "sim > 0.90"
    uses_rgb = False

    def load_data(self):
        masks, labels = generate_line_dataset(seed=SEED)
        return masks, labels

    def run(self, emb, labels, sim):
        if not self.check_labels(labels, ["position", "rotation", "length"]):
            return {}

        all_sims = []

        # 遍历所有 (位置, 旋转) 组合
        for pos, pos_idx in labels["position"].items():
            for rot, rot_idx in labels["rotation"].items():
                # 取交集
                base = set(pos_idx) & set(rot_idx)
                if not base:
                    continue

                # 按长度分组
                len_groups = {}
                for length, length_idx in labels["length"].items():
                    group = sorted(list(base & set(length_idx)))
                    if group:
                        len_groups[length] = group

                # 计算不同长度样本之间的相似度
                if len(len_groups) >= 2:
                    representatives = [group[0] for group in len_groups.values()]
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
