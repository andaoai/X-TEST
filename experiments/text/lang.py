"""
Ex1: 中英文可分性
"""
import numpy as np
from pathlib import Path
from experiments.base import BaseExperiment
from experiments.viz import tsne_plot
from experiments.source.synthetic.config import OUTPUT_ROOT
from experiments.source.synthetic.data import Dataset


class ExpLangSeparation(BaseExperiment):
    name = "中英文可分性"
    hypothesis = "中文和英文是两种不同的书写系统, embedding 应能区分语言来源"
    passes_when = "separation > 0.05"
    uses_rgb = True

    def load_data(self):
        ds = Dataset().generate(verbose=True)
        return ds.rgbs(), ds.labels

    def run(self, emb, labels, sim):
        if not self.check_labels(labels, ["lang"]): return {}
        en = np.array(labels["lang"]["EN"])
        zh = np.array(labels["lang"]["ZH"])
        we = sim[en][:, en][~np.eye(len(en), dtype=bool)].mean()
        wz = sim[zh][:, zh][~np.eye(len(zh), dtype=bool)].mean()
        cr = sim[en][:, zh].mean()
        sep = float((we + wz) / 2 - cr)
        return {
            "name": self.name, "metric": f"separation={sep:+.4f}",
            "separation": sep, "is_correct": sep > 0.05,
            "details": {"en_inner": float(we), "zh_inner": float(wz), "cross": float(cr)},
        }

    def viz(self, emb, labels, result, algo_name, out_dir):
        tsne_plot(emb, labels["lang"], {"EN":"#e74c3c","ZH":"#3498db"},
                  f"Exp1: {self.name} [{algo_name}]",
                  out_dir/"exp1_lang.png", result.get("metric",""))
