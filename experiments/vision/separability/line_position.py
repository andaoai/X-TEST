"""线位置可分性 —— 固定旋转、长度，只变位置，embedding 应该不同"""
import numpy as np
from experiments.base import BaseExperiment
from experiments.source.synthetic.line import generate_line_dataset
from experiments.source.synthetic.config import SEED
from experiments.metrics import calc_group_separation


class ExpLinePositionSeparation(BaseExperiment):
    name = "线位置可分性"
    hypothesis = "固定旋转、长度，只变位置，embedding 应该不同（位置信息被编码）"
    passes_when = "separation > 0.02"
    uses_rgb = False

    def load_data(self):
        masks, labels = generate_line_dataset(seed=SEED)
        return masks, labels

    def run(self, emb, labels, sim):
        if not self.check_labels(labels, ["rotation", "length", "position"]):
            return {}

        all_seps = []

        # 遍历所有 (旋转, 长度) 组合
        for rot, rot_idx in labels["rotation"].items():
            for length, length_idx in labels["length"].items():
                # 取交集
                base = set(rot_idx) & set(length_idx)
                if not base:
                    continue

                # 按位置分组
                pos_groups = {}
                for pos, pos_idx in labels["position"].items():
                    group = sorted(list(base & set(pos_idx)))
                    if group:
                        pos_groups[pos] = group

                # 计算分离度
                if len(pos_groups) >= 2:
                    sep = calc_group_separation(sim, pos_groups)
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
