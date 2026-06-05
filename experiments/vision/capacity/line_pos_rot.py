"""线位置+旋转联合编码 —— 固定长度，按 (位置, 旋转) 分组"""
import numpy as np
from experiments.base import BaseExperiment
from experiments.source.synthetic.line import generate_line_dataset
from experiments.source.synthetic.config import SEED
from experiments.metrics import calc_group_similarity


class ExpLinePosRotCapacity(BaseExperiment):
    name = "线位置+旋转联合编码"
    hypothesis = "embedding 能否同时编码线的位置和旋转？固定长度，同位置同旋转的样本应相似"
    passes_when = "sim > 0.80"
    uses_rgb = False

    def load_data(self):
        masks, labels = generate_line_dataset(seed=SEED)
        return masks, labels

    def run(self, emb, labels, sim):
        if not self.check_labels(labels, ["position", "rotation", "length"]):
            return {}

        all_sims = []

        # 按长度分组，测试同一长度下 (位置, 旋转) 的相似度
        for length, length_idx in labels["length"].items():
            if not length_idx:
                continue

            # 按 (位置, 旋转) 分组
            groups = {}
            for pos, pos_idx in labels["position"].items():
                for rot, rot_idx in labels["rotation"].items():
                    group = sorted(list(set(length_idx) & set(pos_idx) & set(rot_idx)))
                    if len(group) >= 2:
                        groups[(pos, rot)] = group

            # 计算组内相似度
            if groups:
                sim_val = calc_group_similarity(sim, groups)
                all_sims.append(sim_val)

        avg_sim = float(np.mean(all_sims)) if all_sims else 0.0
        return {
            "name": self.name,
            "metric": f"sim={avg_sim:.4f}",
            "separation": avg_sim,
            "is_correct": avg_sim > 0.80,
            "details": {"mean_sim": avg_sim, "n_conditions": len(all_sims)},
        }

    def viz(self, emb, labels, result, algo_name, out_dir):
        pass
