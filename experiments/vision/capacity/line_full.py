"""线全属性联合编码 —— 按 (位置, 旋转, 长度) 分组"""
import numpy as np
from experiments.base import BaseExperiment
from experiments.source.synthetic.line import generate_line_dataset
from experiments.source.synthetic.config import SEED
from experiments.metrics import calc_group_similarity


class ExpLineFullCapacity(BaseExperiment):
    name = "线全属性联合编码"
    hypothesis = "embedding 能否同时编码线的位置、旋转和长度？完全相同的样本应高度相似"
    passes_when = "sim > 0.80"
    uses_rgb = False

    def load_data(self):
        masks, labels = generate_line_dataset(seed=SEED)
        return masks, labels

    def run(self, emb, labels, sim):
        if not self.check_labels(labels, ["position", "rotation", "length"]):
            return {}

        # 按 (位置, 旋转, 长度) 分组
        groups = {}
        for pos, pos_idx in labels["position"].items():
            for rot, rot_idx in labels["rotation"].items():
                for length, length_idx in labels["length"].items():
                    group = sorted(list(set(pos_idx) & set(rot_idx) & set(length_idx)))
                    if len(group) >= 2:
                        groups[(pos, rot, length)] = group

        # 计算组内相似度
        avg_sim = calc_group_similarity(sim, groups) if groups else 0.0

        return {
            "name": self.name,
            "metric": f"sim={avg_sim:.4f}",
            "separation": avg_sim,
            "is_correct": avg_sim > 0.80,
            "details": {"mean_sim": avg_sim, "n_groups": len(groups)},
        }

    def viz(self, emb, labels, result, algo_name, out_dir):
        pass
