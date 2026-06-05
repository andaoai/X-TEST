"""旋转可分性 —— 不同旋转角度的 embedding 应能区分"""
import numpy as np
from experiments.base import BaseExperiment
from experiments.viz import tsne_plot
from experiments.source.synthetic.config import ROTATION_HEX
from experiments.source.synthetic.data import Dataset


class ExpRotationSeparation(BaseExperiment):
    name = "旋转可分性"
    hypothesis = "不同旋转角度在视觉上有不同特征, embedding 应能区分"
    passes_when = "separation > 0.05"
    uses_rgb = True

    def load_data(self):
        ds = Dataset().generate(verbose=True)
        return ds.rgbs(), ds.labels

    def run(self, emb, labels, sim):
        if not self.check_labels(labels, ["rotation"]): return {}
        groups = list(labels["rotation"].keys())
        within, between = [], []
        for g in groups:
            arr = np.array(labels["rotation"][g])
            s = sim[arr][:, arr]
            within.append(s[~np.eye(len(arr), dtype=bool)].mean())
        for i, gi in enumerate(groups):
            for gj in groups[i+1:]:
                cr = sim[np.array(labels["rotation"][gi])][:, np.array(labels["rotation"][gj])].mean()
                between.append(cr)
        sep = float(np.mean(within) - np.mean(between))
        return {
            "name": self.name, "metric": f"separation={sep:+.4f}", "separation": sep,
            "is_correct": sep > 0.05,
            "details": {"within": float(np.mean(within)), "between": float(np.mean(between))},
        }

    def viz(self, emb, labels, result, algo_name, out_dir):
        tsne_plot(emb, labels["rotation"], ROTATION_HEX,
                  f"旋转可分性 [{algo_name}]",
                  out_dir/"rotation_sep.png", result.get("metric",""))
