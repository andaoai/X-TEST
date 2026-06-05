"""
下载小型实例分割数据集 (直接 URL)。
"""
import urllib.request, zipfile, shutil
from pathlib import Path

DATA_DIR = Path("data/instance_seg")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _download(url: str, dest: Path, desc: str):
    if dest.exists():
        print(f"  [OK] Already exists: {dest.name}")
        return True
    print(f"  Downloading {desc} ...")
    dest.parent.mkdir(parents=True, exist_ok=True)

    def _progress(count, block, total):
        if total > 0:
            pct = min(int(count * block * 100 / total), 100)
            print(f"\r  {pct}%", end="", flush=True)

    try:
        urllib.request.urlretrieve(url, dest, _progress)
        print()
        return True
    except Exception as e:
        print(f"\n  [FAIL] {e}")
        if dest.exists():
            dest.unlink()
        return False


def _extract_zip(zip_path: Path, dest_dir: Path):
    print(f"  Extracting -> {dest_dir.name}/")
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    zip_path.unlink()


def _count_files(d: Path):
    imgs = len(list(d.rglob("*.png"))) + len(list(d.rglob("*.jpg")))
    txts = len(list(d.rglob("*.txt")))
    yamls = len(list(d.rglob("*.yaml")))
    return imgs, txts, yamls


DATASETS = [
    {"name": "coco128-seg",
     "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128-seg.zip",
     "desc": "COCO128-seg (128 images, 80 classes)", "size": "~7 MB"},
    {"name": "crack-seg",
     "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/crack-seg.zip",
     "desc": "Crack segmentation (4029 images, road cracks)", "size": "~92 MB"},
    {"name": "package-seg",
     "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/package-seg.zip",
     "desc": "Package segmentation (2197 images, boxes/packages)", "size": "~103 MB"},
]


def run():
    print("=" * 60)
    print("Downloading Small Instance Segmentation Datasets")
    print("=" * 60)
    for ds in DATASETS:
        print(f"\n--- {ds['name']} ({ds['size']}) ---")
        dest_dir = DATA_DIR / ds["name"]
        if dest_dir.exists() and (dest_dir / "data.yaml").exists():
            imgs, txts, _ = _count_files(dest_dir)
            print(f"  [OK] Already downloaded: {imgs} images, {txts} labels")
            continue
        zip_path = DATA_DIR / f"{ds['name']}.zip"
        if _download(ds["url"], zip_path, ds["desc"]):
            try:
                _extract_zip(zip_path, dest_dir)
                imgs, txts, _ = _count_files(dest_dir)
                print(f"  [OK] Done: {imgs} images, {txts} labels")
            except Exception as e:
                print(f"  [FAIL] Extract error: {e}")

    print("\n" + "=" * 60 + "\nSummary\n" + "=" * 60)
    total = 0
    for d in sorted(DATA_DIR.iterdir()):
        if d.is_dir():
            imgs, txts, yamls = _count_files(d)
            if imgs > 0:
                total += imgs
                print(f"  {d.name:30s}  images={imgs:5d}  labels={txts:5d}")
    print(f"\n  Total images: {total}")
    print(f"  Path: {DATA_DIR.resolve()}")


if __name__ == "__main__":
    run()
