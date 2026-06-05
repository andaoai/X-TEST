"""Ex4: 位置编码"""
import numpy as np
from experiments.base import BaseExperiment
from experiments.viz import tsne_plot
from experiments.source.synthetic.config import POS_HEX
from experiments.source.synthetic.data import Dataset


class ExpPositionEncoding(BaseExperiment):
    name = "位置可编码"
    hypothesis = "同位置的样本之间 embedding 有微量相似性, 可弱编码位置差异"
    passes_when = "separation > 0.02"
    uses_rgb = True

    def load_data(self):
        ds = Dataset().generate(verbose=True)
        return ds.rgbs(), ds.labels

    def run(self, emb, labels, sim):
        if not self.check_labels(labels, ["position"]): return {}
        positions = list(labels["position"].keys())
        within, between = [], []
        for p in positions:
            arr = np.array(labels["position"][p])
            s = sim[arr][:, arr]; within.append(s[~np.eye(len(arr), dtype=bool)].mean())
        for i, pi in enumerate(positions):
            for pj in positions[i+1:]:
                cr = sim[np.array(labels["position"][pi])][:, np.array(labels["position"][pj])].mean()
                between.append(cr)
        sep = float(np.mean(within) - np.mean(between))
        return {
            "name": self.name, "metric": f"separation={sep:+.4f}", "separation": sep,
            "is_correct": sep > 0.02,
            "details": {"within": float(np.mean(within)), "between": float(np.mean(between))},
        }

    def viz(self, emb, labels, result, algo_name, out_dir):
        tsne_plot(emb, labels["position"], POS_HEX,
                  f"Exp4: {self.name} [{algo_name}]",
                  out_dir/"exp4_position.png", result.get("metric",""))
