"""线长度可分性 —— 固定位置、旋转，只变长度，embedding 应该不同"""
import numpy as np
from experiments.base import BaseExperiment
from experiments.source.synthetic.line import generate_line_dataset
from experiments.source.synthetic.config import SEED
from experiments.metrics import calc_group_separation


class ExpLineLengthSeparation(BaseExperiment):
    name = "线长度可分性"
    hypothesis = "固定位置、旋转，只变长度，embedding 应该不同（长度信息被编码）"
    passes_when = "separation > 0.05"
    uses_rgb = False

    def load_data(self):
        masks, labels = generate_line_dataset(seed=SEED)
        return masks, labels

    def run(self, emb, labels, sim):
        if not self.check_labels(labels, ["position", "rotation", "length"]):
            return {}

        all_seps = []

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

                # 计算分离度
                if len(len_groups) >= 2:
                    sep = calc_group_separation(sim, len_groups)
                    all_seps.append(sep)

        avg_sep = float(np.mean(all_seps)) if all_seps else 0.0
        return {
            "name": self.name,
            "metric": f"separation={avg_sep:+.4f}",
            "separation": avg_sep,
            "is_correct": avg_sep > 0.05,
            "details": {"mean_sep": avg_sep, "n_conditions": len(all_seps)},
        }

    def viz(self, emb, labels, result, algo_name, out_dir):
        pass
