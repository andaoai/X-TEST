"""旋转不变性 —— 同字符不同旋转角度下 embedding 仍应相似"""
import numpy as np
from experiments.base import BaseExperiment
from experiments.viz import tsne_plot
from experiments.source.synthetic.config import ROTATION_HEX
from experiments.source.synthetic.data import Dataset


class ExpRotationInvariance(BaseExperiment):
    name = "旋转不变性"
    hypothesis = "同一个字符无论旋转多少度, embedding 应该高度相似"
    passes_when = "mean_similarity > 0.90"
    uses_rgb = True

    def load_data(self):
        ds = Dataset().generate(verbose=True)
        return ds.rgbs(), ds.labels

    def run(self, emb, labels, sim):
        if not self.check_labels(labels, ["label"]): return {}
        all_sims = []
        for idxs in labels["label"].values():
            arr = np.array(idxs)
            s = sim[arr][:, arr]
            all_sims.extend(s[~np.eye(len(arr), dtype=bool)].tolist())
        avg = float(np.mean(all_sims))
        return {
            "name": self.name, "metric": f"sim={avg:.4f}", "separation": avg,
            "is_correct": avg > 0.90,
            "details": {"mean_sim": avg, "std_sim": float(np.std(all_sims)),
                        "n_pairs": len(all_sims)},
        }

    def viz(self, emb, labels, result, algo_name, out_dir):
        tsne_plot(emb, labels["rotation"], ROTATION_HEX,
                  f"旋转不变性 [{algo_name}]",
                  out_dir/"rotation_inv.png", result.get("metric",""))
