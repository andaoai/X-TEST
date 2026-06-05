"""
实4: 位置编码

假设: 同类样本在同一位置时有微小相似性，embedding 能弱编码位置差异。
"""
import numpy as np
from pathlib import Path
from experiments.base import BaseExperiment
from experiments.viz import tsne_plot
from experiments.source.synthetic.config import OUTPUT_ROOT, POS_HEX


class ExpPositionEncoding(BaseExperiment):
    name = "位置可编码"
    hypothesis = "同位置的样本之间 embedding 有微量相似性，可弱编码位置差异"
    data_source = "experiments/source/synthetic"
    what_labels = ["position"]
    passes_when = "separation > 0.02"

    def run(self, emb, labels, sim):
        positions = list(labels["position"].keys())
        within, between = [], []
        for p in positions:
            arr = np.array(labels["position"][p])
            s = sim[arr][:, arr]
            within.append(s[~np.eye(len(arr), dtype=bool)].mean())
        for i, pi in enumerate(positions):
            for pj in positions[i+1:]:
                cr = sim[np.array(labels["position"][pi])][:, np.array(labels["position"][pj])].mean()
                between.append(cr)
        sep = float(np.mean(within) - np.mean(between))
        return {
            "name": self.name, "hypothesis": self.hypothesis,
            "metric": f"separation={sep:+.4f}", "separation": sep,
            "is_correct": sep > 0.02,
            "details": {"within": float(np.mean(within)), "between": float(np.mean(between))},
        }

    def viz(self, emb, labels, result, algo_name, out_dir):
        tsne_plot(emb, labels["position"], POS_HEX,
                  f"Exp4: {self.name} [{algo_name}]",
                  out_dir / "exp4_position.png", result.get("metric",""))
