"""
浏览已下载的实例分割数据集: 结构、标签格式、样本可视化。
"""
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

DATA_DIR = Path("data/instance_seg")


def _explore_one(dataset_path: Path):
    name = dataset_path.name
    imgs = sorted(list(dataset_path.rglob("*.jpg")) + list(dataset_path.rglob("*.png")))
    imgs = [f for f in imgs if "_mask" not in f.name]
    txts = sorted(dataset_path.rglob("*.txt"))
    yamls = sorted(dataset_path.rglob("*.yaml"))

    print(f"\n{'=' * 50}\nDataset: {name}\n{'=' * 50}")
    print(f"  Images: {len(imgs)}  Labels: {len(txts)}  YAML: {len(yamls)}")

    if yamls:
        content = yamls[0].read_text(encoding="utf-8")[:500]
        print(f"\n  YAML: {yamls[0].name}\n  {content}")

    if txts:
        lines = txts[0].read_text(encoding="utf-8").strip().split("\n")
        print(f"\n  Label: {txts[0].name} ({len(lines)} instances)")
        for i, line in enumerate(lines[:3]):
            parts = line.strip().split()
            coords = [float(x) for x in parts[1:]]
            n = len(coords) // 2 if len(coords) > 4 else None
            print(f"    [{i}] class={parts[0]}  "
                  + (f"polygon(n={n})" if n else f"bbox={coords}"))
    return imgs, txts


def _visualize(dataset_path, img_path, label_path, ax=None):
    img = Image.open(img_path)
    w, h = img.size
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img)
    ax.set_title(f"{dataset_path.name} - {img_path.name}")
    if label_path and label_path.exists():
        colors = plt.cm.tab20(np.linspace(0, 1, 20))
        for line in label_path.read_text(encoding="utf-8").strip().split("\n"):
            parts = line.strip().split()
            if len(parts) < 3: continue
            cls_id = int(parts[0])
            coords = np.array([float(x) for x in parts[1:]])
            if len(coords) == 4:
                x, y, bw, bh = coords
                rect = plt.Rectangle((x*w - bw*w/2, y*h - bh*h/2), bw*w, bh*h,
                                     fill=False, color=colors[cls_id%20], linewidth=2)
                ax.add_patch(rect)
            else:
                pts = coords.reshape(-1, 2) * [w, h]
                ax.add_patch(Polygon(pts, fill=True, alpha=0.3,
                               color=colors[cls_id%20], edgecolor=colors[cls_id%20], linewidth=1))
    ax.axis("off")


def _show_pennfudan(ped_dir):
    png_dir = ped_dir / "PNGImages"
    mask_dir = ped_dir / "PedMasks"
    if not png_dir.exists(): return
    imgs = sorted(png_dir.glob("*.png"))[:5]
    fig, axes = plt.subplots(2, len(imgs), figsize=(14, 6))
    for i, imp in enumerate(imgs):
        axes[0, i].imshow(Image.open(imp)); axes[0, i].set_title(imp.name[:15], fontsize=8); axes[0, i].axis("off")
        mp = mask_dir / imp.name.replace(".png", "_mask.png")
        if mp.exists(): axes[1, i].imshow(Image.open(mp), cmap="gray"); axes[1, i].axis("off")
    axes[0, 0].set_ylabel("Image", fontsize=10); axes[1, 0].set_ylabel("Mask", fontsize=10)
    plt.tight_layout(); plt.savefig(DATA_DIR / "pennfudanped_preview.png", dpi=100); plt.close()


def run():
    print("=" * 60 + "\nInstance Segmentation Datasets Explorer\n" + "=" * 60)
    for d in sorted(DATA_DIR.iterdir()):
        if not d.is_dir(): continue
        imgs, txts = _explore_one(d)
        if d.name == "PennFudanPed":
            _show_pennfudan(d)
        elif imgs and txts:
            img_to_label = {t.stem: t for t in txts}
            for imp in imgs[:1]:
                label = img_to_label.get(imp.stem)
                fig, ax = plt.subplots(figsize=(10, 8))
                _visualize(d, imp, label, ax)
                plt.savefig(DATA_DIR / f"{d.name}_preview.png", dpi=100, bbox_inches="tight")
                plt.close()
                print(f"  Preview saved: {DATA_DIR / f'{d.name}_preview.png'}")
    print("\nDone!")


if __name__ == "__main__":
    run()
