"""
实2: 位置不变性

假设: 同一个字符无论放在哪个位置，embedding 应该高度相似。
"""
import numpy as np
from pathlib import Path
from experiments.base import BaseExperiment
from experiments.viz import tsne_plot
from experiments.source.synthetic.config import OUTPUT_ROOT, POS_HEX


class ExpPositionInvariance(BaseExperiment):
    name = "位置不变性"
    hypothesis = "同一个字符无论放在哪个位置，embedding 应该高度相似"
    data_source = "experiments/source/synthetic"
    what_labels = ["label", "color"]
    passes_when = "mean_similarity > 0.90"

    def run(self, emb, labels, sim):
        all_sims = []
        for char, idxs in labels["label"].items():
            arr = np.array(idxs)
            s = sim[arr][:, arr]
            mask = ~np.eye(len(arr), dtype=bool)
            all_sims.extend(s[mask].tolist())
        avg = float(np.mean(all_sims))
        return {
            "name": self.name, "hypothesis": self.hypothesis,
            "metric": f"sim={avg:.4f}", "separation": avg,
            "is_correct": avg > 0.90,
            "details": {"mean_sim": avg, "std_sim": float(np.std(all_sims)),
                        "n_pairs": len(all_sims)},
        }

    def viz(self, emb, labels, result, algo_name, out_dir):
        tsne_plot(emb, labels["position"], POS_HEX,
                  f"Exp2: {self.name} [{algo_name}]",
                  out_dir / "exp2_position.png", result.get("metric",""))
