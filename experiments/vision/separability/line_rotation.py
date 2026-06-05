"""线旋转可分性 —— 固定位置、长度，只变旋转，embedding 应该不同"""
import numpy as np
from experiments.base import BaseExperiment
from experiments.source.synthetic.line import generate_line_dataset
from experiments.source.synthetic.config import SEED
from experiments.metrics import calc_group_separation


class ExpLineRotationSeparation(BaseExperiment):
    name = "线旋转可分性"
    hypothesis = "固定位置、长度，只变旋转，embedding 应该不同（旋转角度被编码）"
    passes_when = "separation > 0.05"
    uses_rgb = False

    def load_data(self):
        masks, labels = generate_line_dataset(seed=SEED)
        return masks, labels

    def run(self, emb, labels, sim):
        if not self.check_labels(labels, ["position", "length", "rotation"]):
            return {}

        all_seps = []

        # 遍历所有 (位置, 长度) 组合
        for pos, pos_idx in labels["position"].items():
            for length, length_idx in labels["length"].items():
                # 取交集
                base = set(pos_idx) & set(length_idx)
                if not base:
                    continue

                # 按旋转分组
                rot_groups = {}
                for rot, rot_idx in labels["rotation"].items():
                    group = sorted(list(base & set(rot_idx)))
                    if group:
                        rot_groups[rot] = group

                # 计算分离度
                if len(rot_groups) >= 2:
                    sep = calc_group_separation(sim, rot_groups)
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
