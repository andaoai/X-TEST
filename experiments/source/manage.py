"""
数据集管理 —— 查看、添加、快速可视化。

用法:
  uv run python experiments/source/manage.py list          # 列出所有可用数据集
  uv run python experiments/source/manage.py show <name>   # 查看某个数据集详情
  uv run python experiments/source/manage.py preview <name> # 快速可视化
  uv run python experiments/source/manage.py add <path>     # 添加新数据集
"""
import sys
from pathlib import Path

# 所有数据集的注册信息
DATA_REGISTRY = {
    "synthetic_text": {
        "name": "synthetic_text",
        "desc": "合成文字数据集: 10英+10中 x 8色 x 5位置 = 800张",
        "path": "experiments/source/synthetic/data.py",
        "type": "synthetic",
        "run": "from experiments.source.synthetic.data import Dataset; ds = Dataset().generate()",
    },
    "cifar10": {
        "name": "cifar10",
        "desc": "CIFAR-10: 60000 张 32x32 彩色图, 10 类",
        "path": "data/cifar10_images",
        "type": "real",
        "format": "图片按类别分子文件夹",
    },
    "coco128-seg": {
        "name": "coco128-seg",
        "desc": "COCO128 实例分割: 128 张, 80 类, YOLOv8-seg 格式",
        "path": "data/instance_seg/coco128-seg",
        "type": "real",
        "format": "YOLOv8-seg (images + labels + data.yaml)",
    },
    "crack-seg": {
        "name": "crack-seg",
        "desc": "裂缝分割: 4029 张, 道路裂缝检测",
        "path": "data/instance_seg/crack-seg",
        "type": "real",
        "format": "YOLOv8-seg",
    },
    "package-seg": {
        "name": "package-seg",
        "desc": "包装分割: 2197 张, 箱子/包裹检测",
        "path": "data/instance_seg/package-seg",
        "type": "real",
        "format": "YOLOv8-seg",
    },
}


def _count_files(d: Path):
    imgs = list(d.rglob("*.png")) + list(d.rglob("*.jpg")) + list(d.rglob("*.jpeg"))
    txts = list(d.rglob("*.txt"))
    yamls = list(d.rglob("*.yaml"))
    return imgs, txts, yamls


def cmd_list():
    """列出所有数据集"""
    print("=" * 60)
    print("  可用数据集")
    print("=" * 60)
    for key, info in DATA_REGISTRY.items():
        tag = "[合成]" if info["type"] == "synthetic" else "[真实]"
        p = Path(info["path"])
        exists = "存在" if p.exists() else "不存在"
        print(f"\n  {key}")
        print(f"    {tag}  {info['desc']}")
        print(f"    路径: {info['path']}  ({exists})")


def cmd_show(name: str):
    """查看数据集详情"""
    if name not in DATA_REGISTRY:
        print(f"未知数据集: {name}")
        print(f"可用: {list(DATA_REGISTRY.keys())}")
        return

    info = DATA_REGISTRY[name]
    print("=" * 60)
    print(f"  数据集: {name}")
    print("=" * 60)
    for k, v in info.items():
        print(f"  {k}: {v}")

    p = Path(info["path"])
    if p.exists() and info["type"] == "real":
        imgs, txts, yamls = _count_files(p)
        print(f"\n  实际文件:")
        print(f"    图片: {len(imgs)}")
        print(f"    标签: {len(txts)}")
        print(f"    YAML: {len(yamls)}")
        if imgs:
            print(f"    图片尺寸: {imgs[0]}")


def cmd_preview(name: str):
    """快速可视化数据样本"""
    if name not in DATA_REGISTRY:
        print(f"未知数据集: {name}")
        return

    info = DATA_REGISTRY[name]
    p = Path(info["path"])
    if not p.exists():
        print(f"数据不存在: {p}")
        return

    if info["type"] == "synthetic":
        # 合成数据: 生成几张看看
        import numpy as np
        import matplotlib.pyplot as plt
        from experiments.source.synthetic.data import Dataset

        ds = Dataset(letters=["A","B","C"], chinese=[], colors=["Red","Green"],
                     positions=["TL","CT","BR"]).generate()
        masks = ds.masks()
        rgbs = ds.rgbs()

        fig, axes = plt.subplots(2, 6, figsize=(14, 5))
        for i in range(6):
            axes[0,i].imshow(rgbs[i])
            axes[0,i].set_title(ds.samples[i]["label"], fontsize=8)
            axes[0,i].axis("off")
            axes[1,i].imshow(masks[i], cmap="gray")
            axes[1,i].axis("off")
        axes[0,0].set_ylabel("RGB", fontsize=10)
        axes[1,0].set_ylabel("Mask", fontsize=10)
        plt.tight_layout()
        out = Path("data") / "synthetic_preview.png"
        plt.savefig(out, dpi=100)
        plt.close()
        print(f"预览已保存: {out}")

    else:
        # 真实数据: 显示前几张
        import matplotlib.pyplot as plt
        from PIL import Image

        imgs, txts, _ = _count_files(p)
        if not imgs:
            print("无图片文件")
            return

        n = min(8, len(imgs))
        cols = 4
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(12, 3*rows))
        axes = axes.flatten() if n > 1 else [axes]

        for i in range(n):
            img = Image.open(imgs[i])
            axes[i].imshow(img)
            axes[i].set_title(imgs[i].name[:20], fontsize=7)
            axes[i].axis("off")
        for i in range(n, len(axes)):
            axes[i].axis("off")

        plt.tight_layout()
        out = Path("data") / f"{name}_preview.png"
        plt.savefig(out, dpi=100)
        plt.close()
        print(f"预览已保存: {out}  ({n}/{len(imgs)} 张)")


def cmd_add(path: str):
    """添加新数据集到注册表"""
    p = Path(path)
    if not p.exists():
        print(f"路径不存在: {p}")
        return

    imgs, txts, yamls = _count_files(p)
    print(f"扫描: {p}")
    print(f"  图片: {len(imgs)}  标签: {len(txts)}  YAML: {len(yamls)}")

    name = p.name
    print(f"\n建议在 DATA_REGISTRY 中添加:")
    print(f'  "{name}": {{')
    print(f'    "name": "{name}",')
    print(f'    "desc": "描述...",')
    print(f'    "path": "{path}",')
    print(f'    "type": "real",')
    print(f'    "format": "格式...",')
    print(f'  }}')


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: list | show <name> | preview <name> | add <path>")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "list":
        cmd_list()
    elif cmd == "show" and len(sys.argv) > 2:
        cmd_show(sys.argv[2])
    elif cmd == "preview" and len(sys.argv) > 2:
        cmd_preview(sys.argv[2])
    elif cmd == "add" and len(sys.argv) > 2:
        cmd_add(sys.argv[2])
    else:
        print(f"未知命令: {cmd}")
        print("用法: list | show <name> | preview <name> | add <path>")
