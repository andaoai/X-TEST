"""Ex3: 颜色可分性"""
import numpy as np
from experiments.base import BaseExperiment
from experiments.viz import tsne_plot
from experiments.source.synthetic.config import COLOR_HEX
from experiments.source.synthetic.data import Dataset


class ExpColorSeparation(BaseExperiment):
    name = "颜色可分性"
    hypothesis = "不同颜色在物理通道上有不同的能量签名, embedding 应能区分"
    passes_when = "separation > 0.05"
    uses_rgb = True

    def load_data(self):
        ds = Dataset().generate(verbose=True)
        return ds.rgbs(), ds.labels

    def run(self, emb, labels, sim):
        if not self.check_labels(labels, ["color"]): return {}
        colors = list(labels["color"].keys())
        within, between = [], []
        for c in colors:
            arr = np.array(labels["color"][c])
            s = sim[arr][:, arr]; within.append(s[~np.eye(len(arr), dtype=bool)].mean())
        for i, ci in enumerate(colors):
            for cj in colors[i+1:]:
                cr = sim[np.array(labels["color"][ci])][:, np.array(labels["color"][cj])].mean()
                between.append(cr)
        sep = float(np.mean(within) - np.mean(between))
        return {
            "name": self.name, "metric": f"separation={sep:+.4f}", "separation": sep,
            "is_correct": sep > 0.05,
            "details": {"within": float(np.mean(within)), "between": float(np.mean(between))},
        }

    def viz(self, emb, labels, result, algo_name, out_dir):
        tsne_plot(emb, labels["color"], COLOR_HEX,
                  f"Exp3: {self.name} [{algo_name}]",
                  out_dir/"exp3_color.png", result.get("metric",""))
