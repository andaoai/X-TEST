"""Ex5: 字符可分性"""
import numpy as np
import matplotlib.pyplot as plt
from experiments.base import BaseExperiment
from experiments.viz import tsne_plot
from experiments.source.synthetic.config import EN_LETTERS, ZH_CHARS
from experiments.source.synthetic.data import Dataset

CHAR_PALETTE = {
    **{c: plt.cm.tab10(i/10) for i,c in enumerate(EN_LETTERS)},
    **{c: plt.cm.Set3(i/10)  for i,c in enumerate(ZH_CHARS)},
}


class ExpCharSeparation(BaseExperiment):
    name = "字符可分性"
    hypothesis = "不同字符有不同的形状结构, embedding 应该能区分"
    passes_when = "separation > 0.05"
    uses_rgb = True

    def load_data(self):
        ds = Dataset().generate(verbose=True)
        return ds.rgbs(), ds.labels

    def run(self, emb, labels, sim):
        if not self.check_labels(labels, ["label"]): return {}
        chars = list(labels["label"].keys())
        within, between = [], []
        for c in chars:
            arr = np.array(labels["label"][c])
            s = sim[arr][:, arr]; within.append(s[~np.eye(len(arr), dtype=bool)].mean())
        for i, ci in enumerate(chars):
            for cj in chars[i+1:]:
                cr = sim[np.array(labels["label"][ci])][:, np.array(labels["label"][cj])].mean()
                between.append(cr)
        sep = float(np.mean(within) - np.mean(between))
        return {
            "name": self.name, "metric": f"separation={sep:+.4f}", "separation": sep,
            "is_correct": sep > 0.05,
            "details": {"within": float(np.mean(within)), "between": float(np.mean(between))},
        }

    def viz(self, emb, labels, result, algo_name, out_dir):
        tsne_plot(emb, labels["label"], CHAR_PALETTE,
                  f"Exp5: {self.name} [{algo_name}]",
                  out_dir/"exp5_char.png", result.get("metric",""))
