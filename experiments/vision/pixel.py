"""
视觉实验: 单像素 mask 的空间可分性

数据: 黑屏上的白点 (1px) vs 多个随机白点 (2/3/5/10px)
假设: 人类能瞬间定位白点 → 好的编码应保留位置信息
"""
import numpy as np
from pathlib import Path
from experiments.base import BaseExperiment
from experiments.source.synthetic.pixel import generate_pixel_dataset
from experiments.source.synthetic.config import OUTPUT_ROOT, SEED


class ExpPixelSeparation(BaseExperiment):
    name = "像素可分性"
    hypothesis = (
        "单像素在不同位置 -> embedding 应该不同(位置可编码); "
        "1个点 vs 多个点 -> embedding 应能区分数量。"
        "人类能瞬间定位黑屏上的白点, 好的编码也应保留空间信息。"
    )
    passes_when = "位置分离度 > 0.05 或 数量分离度 > 0.03"
    uses_rgb = True

    def load_data(self):
        masks, labels, _ = generate_pixel_dataset(grid_step=4, n_multi=64, seed=SEED)
        # GaborLift 需要 RGB → 复制为 3 通道
        rgb = np.stack([masks]*3, axis=-1).astype(np.float32)
        print(f"  像素数据: {len(masks)} 张 (单点+多点)")
        return rgb, labels

    def run(self, emb, labels, sim):
        # 位置: x 坐标分组
        x_sep = self._group_sep(labels.get("x_pos", {}), sim)
        y_sep = self._group_sep(labels.get("y_pos", {}), sim)
        pos_sep = (x_sep + y_sep) / 2

        # 数量: 1pt vs 2/3/5/10pt
        cnt_sep = self._group_sep(labels.get("count", {}), sim)

        overall = pos_sep > 0.05 or cnt_sep > 0.03
        return {
            "name": self.name,
            "metric": f"pos={pos_sep:+.4f} cnt={cnt_sep:+.4f}",
            "separation": max(pos_sep, cnt_sep),
            "is_correct": overall,
            "details": {"pos_sep_x": x_sep, "pos_sep_y": y_sep,
                        "pos_sep_avg": pos_sep, "cnt_sep": cnt_sep},
        }

    @staticmethod
    def _group_sep(groups, sim):
        if len(groups) < 2:
            return 0.0
        keys = sorted(groups.keys())
        within, between = [], []
        for k in keys:
            arr = np.array(groups[k])
            s = sim[arr][:, arr]
            within.append(s[~np.eye(len(arr), dtype=bool)].mean())
        for i, ki in enumerate(keys):
            for kj in keys[i+1:]:
                cr = sim[np.array(groups[ki])][:, np.array(groups[kj])].mean()
                between.append(cr)
        return float(np.mean(within) - np.mean(between))

    def viz(self, emb, labels, result, algo_name, out_dir):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE

        single_idx = labels.get("single", [])
        if not single_idx:
            return
        n_single = len(single_idx)
        xy = TSNE(n_components=2, random_state=SEED,
                  perplexity=min(30, n_single-1)).fit_transform(emb[single_idx])

        # 按 X 着色
        x_groups = labels.get("x_pos", {})
        x_vals = np.zeros(n_single)
        for i, idx in enumerate(single_idx):
            for k, v in x_groups.items():
                if idx in v: x_vals[i] = int(k.split("=")[1]); break

        fig, ax = plt.subplots(figsize=(8,7))
        sc = ax.scatter(xy[:,0], xy[:,1], c=x_vals, cmap="coolwarm",
                        alpha=0.7, s=20, edgecolors="none")
        ax.set_title(f"Exp6: {self.name} - X坐标 [{algo_name}]\n"
                     f"{result.get('metric','')}", fontsize=11, fontweight="bold")
        plt.colorbar(sc, ax=ax, label="X"); ax.set_xticks([]); ax.set_yticks([])
        plt.tight_layout(); fig.savefig(out_dir/"exp6_pixel_x.png", dpi=150); plt.close(fig)

        # 按 Y 着色
        y_groups = labels.get("y_pos", {})
        y_vals = np.zeros(n_single)
        for i, idx in enumerate(single_idx):
            for k, v in y_groups.items():
                if idx in v: y_vals[i] = int(k.split("=")[1]); break

        fig, ax = plt.subplots(figsize=(8,7))
        sc = ax.scatter(xy[:,0], xy[:,1], c=y_vals, cmap="viridis",
                        alpha=0.7, s=20, edgecolors="none")
        ax.set_title(f"Exp6: {self.name} - Y坐标 [{algo_name}]", fontsize=11, fontweight="bold")
        plt.colorbar(sc, ax=ax, label="Y"); ax.set_xticks([]); ax.set_yticks([])
        plt.tight_layout(); fig.savefig(out_dir/"exp6_pixel_y.png", dpi=150); plt.close(fig)

        # 数量对比 t-SNE
        cnt = labels.get("count", {})
        if cnt:
            all_idx, all_lbl = [], []
            for k, v in cnt.items():
                all_idx.extend(v); all_lbl.extend([k]*len(v))
            all_idx = np.array(all_idx)
            xy2 = TSNE(n_components=2, random_state=SEED,
                       perplexity=min(30, len(all_idx)-1)).fit_transform(emb[all_idx])
            pal = {"1pt":"#95a5a6","2pt":"#2ecc71","3pt":"#3498db",
                   "5pt":"#e74c3c","10pt":"#f39c12"}
            fig, ax = plt.subplots(figsize=(8,7))
            for lb in ["1pt","2pt","3pt","5pt","10pt"]:
                m = np.array(all_lbl)==lb
                if m.sum():
                    ax.scatter(xy2[m,0], xy2[m,1], c=pal[lb], label=lb,
                               alpha=0.6, s=20, edgecolors="black", linewidth=0.2)
            ax.set_title(f"Exp6: {self.name} - 数量 [{algo_name}]", fontsize=11, fontweight="bold")
            ax.legend(fontsize=9); ax.set_xticks([]); ax.set_yticks([])
            plt.tight_layout(); fig.savefig(out_dir/"exp6_pixel_count.png", dpi=150); plt.close(fig)
