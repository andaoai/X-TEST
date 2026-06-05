"""
实5: 字符可分性

假设: 不同字符有不同的形状结构，embedding 应该能区分。
"""
import numpy as np
from pathlib import Path
from experiments.base import BaseExperiment
from experiments.viz import tsne_plot
from experiments.source.synthetic.config import (OUTPUT_ROOT, EN_LETTERS, ZH_CHARS)
import matplotlib.pyplot as plt


CHAR_PALETTE = {
    **{c: plt.cm.tab10(i/10)  for i, c in enumerate(EN_LETTERS)},
    **{c: plt.cm.Set3(i/10)   for i, c in enumerate(ZH_CHARS)},
}


class ExpCharSeparation(BaseExperiment):
    name = "字符可分性"
    hypothesis = "不同字符有不同的形状结构，embedding 应该能区分"
    data_source = "experiments/source/synthetic"
    what_labels = ["label"]
    passes_when = "separation > 0.05"

    def run(self, emb, labels, sim):
        chars = list(labels["label"].keys())
        within, between = [], []
        for c in chars:
            arr = np.array(labels["label"][c])
            s = sim[arr][:, arr]
            within.append(s[~np.eye(len(arr), dtype=bool)].mean())
        for i, ci in enumerate(chars):
            for cj in chars[i+1:]:
                cr = sim[np.array(labels["label"][ci])][:, np.array(labels["label"][cj])].mean()
                between.append(cr)
        sep = float(np.mean(within) - np.mean(between))
        return {
            "name": self.name, "hypothesis": self.hypothesis,
            "metric": f"separation={sep:+.4f}", "separation": sep,
            "is_correct": sep > 0.05,
            "details": {"within": float(np.mean(within)), "between": float(np.mean(between))},
        }

    def viz(self, emb, labels, result, algo_name, out_dir):
        tsne_plot(emb, labels["label"], CHAR_PALETTE,
                  f"Exp5: {self.name} [{algo_name}]",
                  out_dir / "exp5_char.png", result.get("metric",""))
